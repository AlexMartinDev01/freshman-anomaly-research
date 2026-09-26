# -*- coding: utf-8 -*-
"""
Gate 9B-v0 -- Target-Conditioned Defect Feature Generator.

Gate 9A showed the discrete decision "(target, bank) -> utility" does not
generalise across targets: bank_only beat interaction, and every learned variant
landed at chance on AD2. The hypothesis here is an inductive-bias swap -- learn a
SHARED transformation instead of a per-pair scalar:

    z_hat = normalize( z_ext + G(z_ext, h_source_normal, h_target_normal, eps) )

One parameter set applies to every patch, every donor, every pseudo-target, so
the supervision per parameter is orders of magnitude denser than a selector's.
That is a hypothesis, and the spec says so: Gate 9B-v0 fails if it only learns
"perturb some source defects", which is what G1 (bank_only) can already do.

The three contexts are separate on purpose (Phase 8/9A evidence):
    z_ext     the anomaly content
    h_source  the donor object's coordinate frame
    h_target  where that content must land for THIS target
Feeding source context is not the old hand-built whitening: nothing is specified,
the split is handed to the network.

Losses:
    L_rank    put the generated support into the FROZEN dual-branch detector and
              require unseen defects to outrank unseen normals. This is the one
              that matters -- geometry similarity != intervention utility has been
              demonstrated repeatedly in this project.
    L_swd     sliced Wasserstein to real target defects, no patch correspondence
    L_resid   bound the transformation

Variants (pre-registered; G3 must beat G1 and G0 or the gate fails):
    G0 identity      z_ext unchanged                    raw external baseline
    G1 bank_only     z_ext + source context             donor-only capacity
    G2 target_only   noise + target normal              no external anomaly
    G3 interaction   z_ext + source + target            this method

Protocol: MVTec AD v1, leave-one-category-out, 1/2/4-shot target normals, real
target defects used ONLY for supervision on meta-train categories and as the
oracle on held-out ones. AD2 is not touched until this passes.

Usage: python experiments/model_v0/gate9b_generator.py --folds all --steps 400
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from gate7b3r import make_dist  # noqa: E402
from models.tail_adapter import l2norm  # noqa: E402
from tail_calib import RESULTS, materialize, mean_top1p  # noqa: E402

V1 = os.path.join(RESULTS, "cache_v1")
METRICS = os.path.join(RESULTS, "metrics")
CKPT = os.path.join(RESULTS, "checkpoints_g9b")
DEV = "cuda"
VARIANTS = ["identity", "bank_only", "target_only", "interaction"]
EPS_DIM = 64


# ----------------------------------------------------------------- architecture
class SetEncoder(nn.Module):
    """mean + std pooling + small MLP. Deliberately not attention."""

    def __init__(self, d=384, out=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * d, out), nn.ReLU(),
                                 nn.Linear(out, out))

    def forward(self, X):                       # (M, d) -> (out,)
        if X.dim() == 2:
            mu, sd = X.mean(0), X.std(0)
        else:
            mu, sd = X.mean(1), X.std(1)
        return self.net(torch.cat([mu, sd], dim=-1))


class Gen(nn.Module):
    def __init__(self, variant, d=384, hid=512, ctx=128):
        super().__init__()
        self.variant = variant
        if variant == "identity":
            self.net = None
            return
        d_in = {                       # + EPS_DIM for the noise vector
            "bank_only": d + ctx + EPS_DIM,
            "target_only": ctx + EPS_DIM,
            "interaction": d + 2 * ctx + EPS_DIM,
        }[variant]
        # 384 -> 512 -> 384 per the spec: one hidden layer, keep it small
        self.net = nn.Sequential(nn.Linear(d_in, hid), nn.ReLU(),
                                 nn.Linear(hid, d))

    def forward(self, z_ext, h_s, h_t, eps):
        if self.variant == "identity":
            return l2norm(z_ext)
        parts = {"bank_only": [z_ext, h_s, eps],
                 "target_only": [h_t, eps],
                 "interaction": [z_ext, h_s, h_t, eps]}[self.variant]
        dz = self.net(torch.cat(parts, dim=-1))
        if self.variant == "target_only":
            return l2norm(dz)              # nothing to be residual to
        return l2norm(z_ext + dz)


# ----------------------------------------------------------------------- losses
def swd2(X, Y, n_proj=64):
    """Sliced Wasserstein-2 estimate; no patch correspondence assumed."""
    n = min(len(X), len(Y))
    if n < 2:
        return torch.zeros((), device=X.device)
    rx = torch.randperm(len(X), device=X.device)[:n]
    ry = torch.randperm(len(Y), device=Y.device)[:n]
    dirs = torch.randn(n_proj, X.shape[1], device=X.device)
    dirs = dirs / dirs.norm(dim=1, keepdim=True)
    px = (X[rx] @ dirs.T).sort(dim=0).values
    py = (Y[ry] @ dirs.T).sort(dim=0).values
    return ((px - py) ** 2).mean()


def rank_loss(scores_n, scores_d, beta=0.05):
    """Every unseen defect query must outrank every unseen normal query.

    softplus(-d/beta)*beta, not relu(-d): the hinge is already exactly satisfied
    on easy classes (bottle scores ~98 at step 0), where its gradient is zero and
    the generator gets no learning signal at all. softplus keeps a non-zero push.
    beta is set on the scale of the fused score, whose differences are ~0.1.
    """
    if len(scores_n) == 0 or len(scores_d) == 0:
        return torch.zeros((), device=scores_d.device)
    d = scores_d[:, None] - scores_n[None, :]
    return F.softplus(-d / beta).mean() * beta


# ------------------------------------------------------------------------- data
class Bank:
    """Holds the v1 caches once; episodes slice out what they need."""

    def __init__(self, classes):
        self.classes = classes
        self.tr, self.te = {}, {}
        for c in classes:
            self.tr[c] = materialize(np.load(os.path.join(V1, f"{c}_train.npz"),
                                             allow_pickle=True))
            self.te[c] = materialize(np.load(os.path.join(V1, f"{c}_test.npz"),
                                             allow_pickle=True))
        self.defect = {}
        self.good_idx = {}
        self.bad_idx = {}
        for c in classes:
            te = self.te[c]
            types = list(te["types"])
            gt = te["gt_frac"].reshape(len(types), -1)
            parts = [te["feats"][i].astype(np.float32)[gt[i] > 0.10]
                     for i in range(len(types))
                     if types[i] == "bad" and (gt[i] > 0.10).sum()]
            self.defect[c] = (torch.from_numpy(np.concatenate(parts))
                              if parts else torch.zeros((0, 384)))
            self.good_idx[c] = [i for i, t in enumerate(types) if t == "good"]
            self.bad_idx[c] = [i for i, t in enumerate(types) if t == "bad"]
        print(f"  loaded {len(classes)} classes; "
              f"defect pools {sum(len(v) for v in self.defect.values())} patches",
              flush=True)

    def feats(self, c, idx):
        return torch.from_numpy(
            self.te[c]["feats"][np.asarray(idx)].astype(np.float32))

    def defect_of(self, c, idx):
        """gt-filtered defect patches from the given images.

        Must be gt-filtered: a defect image is mostly NORMAL region, so taking
        every patch of it builds a bank that is almost a normal bank with a few
        defect patches mixed in. That is what made the first oracle column sit
        BELOW the normal-only baseline on 13/15 classes.
        """
        te = self.te[c]
        gt = te["gt_frac"].reshape(len(te["types"]), -1)
        parts = [te["feats"][i].astype(np.float32)[gt[i] > 0.10]
                 for i in np.asarray(idx) if (gt[i] > 0.10).sum()]
        if not parts:
            return torch.zeros((0, 384))
        return torch.from_numpy(np.concatenate(parts))


def sample_shot(bank, c, shot, rng):
    tr = bank.tr[c]
    n = tr["feats"].shape[0]
    k = n if shot == -1 else min(shot, n)
    idx = rng.choice(n, size=k, replace=False)
    return torch.from_numpy(tr["feats"][idx].reshape(-1, 384).astype(np.float32))


def split_bad(bank, c, rng):
    b = bank.bad_idx[c]
    p = rng.permutation(len(b))
    h = len(b) // 2
    return [b[i] for i in p[:h]], [b[i] for i in p[h:]]


# ---------------------------------------------------------------------- episodes
def make_episode(bank, target, source, shot, rng, n_gen, n_query, device=DEV):
    """One episode: target C gets `shot` normals; source j supplies the anomaly."""
    normal_bank = sample_shot(bank, target, shot, rng).to(device)
    sup_bad, ev_bad = split_bad(bank, target, rng)
    gq = rng.choice(bank.good_idx[target],
                    size=min(n_query, len(bank.good_idx[target])), replace=False)
    bq = rng.choice(ev_bad, size=min(n_query, len(ev_bad)), replace=False)
    q_idx = np.concatenate([gq, bq])
    q = bank.feats(target, q_idx).to(device)
    is_def = np.array([False] * len(gq) + [True] * len(bq))

    dist_n = make_dist(normal_bank, device)
    with torch.no_grad():
        Sn = [dist_n(q[i]) for i in range(len(q))]

    # source contexts
    sp = bank.defect[source]
    sel = rng.choice(len(sp), size=min(n_gen, len(sp)), replace=False) if len(sp) else []
    z_ext = sp[sel].to(device) if len(sel) else torch.zeros((0, 384), device=device)
    src_norm = sample_shot(bank, source, 8, rng).to(device)
    tgt_norm = normal_bank
    return {"normal_bank": normal_bank, "q": q, "is_def": is_def, "Sn": Sn,
            "z_ext": z_ext, "src_norm": src_norm, "tgt_norm": tgt_norm,
            "sup_bad": sup_bad, "ev_bad": ev_bad}


def make_dist_t(bank, device=DEV, q_chunk=1024, b_chunk=8192):
    """make_dist, but returns TORCH tensors and stays differentiable in `bank`.

    gate7b3r.make_dist ends in .cpu().numpy(), which is right for the frozen
    evaluation path but silently detaches the graph. Here the whole point is to
    backprop through S_d into the generator, so the bank must keep its grad.
    """
    bn = l2norm(bank.to(device).float())

    def dist(z):
        q = l2norm(z.float())
        out = []
        for i in range(0, len(q), q_chunk):
            qc = q[i:i + q_chunk]
            best = torch.full((len(qc),), -1.0, device=qc.device)
            for j in range(0, len(bn), b_chunk):
                best = torch.maximum(best, (qc @ bn[j:j + b_chunk].T).max(dim=1).values)
            out.append(1.0 - best)
        return torch.cat(out)
    return dist


def gen_support(gen, ep, enc_s, enc_t, n_gen, device=DEV):
    """Apply the generator. G2 has no external content, so it draws pure noise."""
    if gen.variant == "target_only":
        z_ext = torch.zeros((n_gen, 384), device=device)
    else:
        z_ext = ep["z_ext"]
    if len(z_ext) == 0:
        return z_ext
    h_s = enc_s(ep["src_norm"]).unsqueeze(0).expand(len(z_ext), -1)
    h_t = enc_t(ep["tgt_norm"]).unsqueeze(0).expand(len(z_ext), -1)
    eps = torch.randn(len(z_ext), EPS_DIM, device=device)
    return gen(z_ext, h_s, h_t, eps)


def scores_of(ep, support, device=DEV):
    """mean_top1p(S_n - S_d) per query -- differentiable through `support`."""
    Sn = torch.stack([torch.as_tensor(np.asarray(ep["Sn"][i]), dtype=torch.float32,
                                      device=device) for i in range(len(ep["q"]))])
    dist_d = make_dist_t(support, device)
    Sd = torch.stack([dist_d(ep["q"][i]) for i in range(len(ep["q"]))])
    k = max(1, int(Sn.shape[1] * 0.01))
    return (Sn - Sd).topk(k, dim=1).values.mean(dim=1)


def train_fold(bank, heldout, variant, steps, lr, lam_swd, gam_resid,
               b_episodes, n_gen, n_query, seed=0, log_every=100):
    torch.manual_seed(seed)
    classes = [c for c in bank.classes if c != heldout]
    enc_s, enc_t = SetEncoder().to(DEV), SetEncoder().to(DEV)
    gen = Gen(variant).to(DEV)
    params = [p for p in list(gen.parameters()) + list(enc_s.parameters())
              + list(enc_t.parameters()) if p.requires_grad]
    if not params:
        return gen, enc_s, enc_t, []
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    hist = []
    for step in range(steps):
        opt.zero_grad()
        tot = tot_r = tot_s = 0.0
        for b in range(b_episodes):
            rng = np.random.default_rng(seed * 100003 + step * 977 + b)
            tgt = classes[rng.integers(len(classes))]
            src = classes[rng.integers(len(classes))]
            while src == tgt:
                src = classes[rng.integers(len(classes))]
            shot = int(rng.choice([1, 2, 4]))
            ep = make_episode(bank, tgt, src, shot, rng, n_gen, n_query)
            sup = gen_support(gen, ep, enc_s, enc_t, n_gen)
            if len(sup) == 0:
                continue
            sc = scores_of(ep, sup)
            m = torch.as_tensor(ep["is_def"], device=DEV)
            lr_ = rank_loss(sc[~m], sc[m])
            # SWD to the REAL target defects (available: tgt is meta-train)
            rp = bank.defect[tgt]
            k = min(n_gen, len(rp))
            real = rp[torch.randperm(len(rp))[:k]].to(DEV)
            ls = swd2(sup, real)
            # bound the transformation; G2 has no residual reference
            if gen.variant in ("bank_only", "interaction") and len(ep["z_ext"]):
                lres = ((sup - l2norm(ep["z_ext"])) ** 2).sum(-1).mean()
            else:
                lres = torch.zeros((), device=DEV)
            tot = tot + lr_ + lam_swd * ls + gam_resid * lres
            tot_r += float(lr_); tot_s += float(ls)
        if isinstance(tot, float):
            continue
        (tot / b_episodes).backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        hist.append({"step": step, "rank": tot_r / b_episodes,
                     "swd": tot_s / b_episodes})
        if log_every and step % log_every == 0:
            print(f"    step {step:>4}  rank {tot_r / b_episodes:.4f}  "
                  f"swd {tot_s / b_episodes:.4f}", flush=True)
    return gen, enc_s, enc_t, hist


# ------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(bank, heldout, gen, enc_s, enc_t, shots=(1, 2, 4),
             n_gen=256, n_query=40, seed=12345):
    from sklearn.metrics import roc_auc_score
    rows = []
    for shot in shots:
        for rep in range(3):
            rng = np.random.default_rng(seed + 100 * rep + shot)
            sup_bad, _ = split_bad(bank, heldout, rng)
            # eval set: all good + the held-out half of bad
            ev = bank.good_idx[heldout] + [b for b in bank.bad_idx[heldout]
                                           if b not in set(sup_bad)]
            q = bank.feats(heldout, ev).to(DEV)
            y = np.array([0] * len(bank.good_idx[heldout])
                         + [1] * (len(ev) - len(bank.good_idx[heldout])))
            nb = sample_shot(bank, heldout, shot, rng).to(DEV)
            dist_n = make_dist(nb, DEV)
            Sn = [np.asarray(dist_n(q[i])) for i in range(len(q))]

            def auroc_of(Sd):
                sc = np.array([mean_top1p(Sn[i] - Sd[i]) for i in range(len(q))])
                return roc_auc_score(y, sc) * 100

            rows.append({"heldout": heldout, "shot": shot, "rep": rep,
                         "baseline": auroc_of([np.zeros_like(Sn[0])] * len(q))})
            # oracle: real target defects from the support half
            real_sup = bank.defect_of(heldout, sup_bad)
            dist_o = make_dist(real_sup.to(DEV), DEV)
            rows[-1]["oracle"] = auroc_of(
                [np.asarray(dist_o(q[i])) for i in range(len(q))])

            # external sources: same (heldout, source) pairing for G0/G1/G2/G3
            srcs = [c for c in bank.classes if c != heldout]
            for src in srcs:
                sp = bank.defect[src]
                if len(sp) == 0:
                    continue
                sel = rng.choice(len(sp), size=min(n_gen, len(sp)), replace=False)
                z_ext = sp[sel].to(DEV)
                src_norm = sample_shot(bank, src, 8, rng).to(DEV)
                eps = torch.randn(len(z_ext), EPS_DIM, device=DEV)
                h_s = enc_s(src_norm).unsqueeze(0).expand(len(z_ext), -1)
                h_t = enc_t(nb).unsqueeze(0).expand(len(z_ext), -1)
                if gen.variant == "target_only":
                    dz = gen.net(torch.cat([h_t, eps], -1))
                    sup = l2norm(dz)
                else:
                    sup = gen(z_ext, h_s, h_t, eps)
                raw = l2norm(z_ext)
                d_raw = make_dist(raw, DEV)
                d_gen = make_dist(sup, DEV)
                rows[-1][f"raw|{gen.variant}"] = auroc_of(
                    [np.asarray(d_raw(q[i])) for i in range(len(q))])
                rows[-1][f"gen|{gen.variant}"] = auroc_of(
                    [np.asarray(d_gen(q[i])) for i in range(len(q))])
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="all")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam-swd", type=float, default=0.1)
    ap.add_argument("--gam-resid", type=float, default=1e-3)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--n-gen", type=int, default=256)
    ap.add_argument("--n-query", type=int, default=16)
    ap.add_argument("--variants", default="all")
    ap.add_argument("--eval-only", action="store_true")
    a = ap.parse_args()

    os.makedirs(CKPT, exist_ok=True)
    classes = sorted(f[:-len("_train.npz")] for f in os.listdir(V1)
                     if f.endswith("_train.npz"))
    bank = Bank(classes)
    folds = classes if a.folds == "all" else a.folds.split(",")
    variants = VARIANTS if a.variants == "all" else a.variants.split(",")

    out_rows = []
    cpath = os.path.join(METRICS, "gate9b_eval.csv")
    for heldout in folds:
        for variant in variants:
            ck = os.path.join(CKPT, f"{heldout}_{variant}.pt")
            if a.eval_only and os.path.exists(ck):
                sd = torch.load(ck, map_location=DEV)
                gen, enc_s, enc_t = Gen(variant).to(DEV), SetEncoder().to(DEV), \
                    SetEncoder().to(DEV)
                gen.load_state_dict(sd["gen"])
                enc_s.load_state_dict(sd["enc_s"])
                enc_t.load_state_dict(sd["enc_t"])
                gen.eval(); enc_s.eval(); enc_t.eval()
            else:
                t0 = time.time()
                print(f"=== fold {heldout} / {variant} ===", flush=True)
                if variant == "identity":
                    gen, enc_s, enc_t = Gen(variant).to(DEV), SetEncoder().to(DEV), \
                        SetEncoder().to(DEV)
                else:
                    gen, enc_s, enc_t, _ = train_fold(
                        bank, heldout, variant, a.steps, a.lr, a.lam_swd,
                        a.gam_resid, a.episodes, a.n_gen, a.n_query)
                    torch.save({"gen": gen.state_dict(),
                                "enc_s": enc_s.state_dict(),
                                "enc_t": enc_t.state_dict()}, ck)
                print(f"    trained in {time.time() - t0:.0f}s", flush=True)
                gen.eval(); enc_s.eval(); enc_t.eval()
            df = evaluate(bank, heldout, gen, enc_s, enc_t, n_gen=a.n_gen)
            df["variant"] = variant
            out_rows.append(df)
            df.to_csv(os.path.join(METRICS, f"gate9b_{heldout}_{variant}.csv"),
                      index=False)
    out = pd.concat(out_rows, ignore_index=True)
    out.to_csv(cpath, index=False)
    print(f"\nwrote {cpath}  rows={len(out)}")


if __name__ == "__main__":
    main()
