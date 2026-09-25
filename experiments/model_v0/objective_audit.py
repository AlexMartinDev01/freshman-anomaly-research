# -*- coding: utf-8 -*-
"""
Gate 6 step 1 -- Objective Audit.

Before changing the objective, establish what the current one actually does.
Trains the K4 (bootstrap) configuration exactly as Gate 5D did, but instruments
every logging step with:

  grad_cos      cosine between grad(L_tail) and grad(L_defect) on the adapter
                parameters. Persistently negative => the two terms really fight
                for the same parameters (PCGrad would then have a basis).
  grad_ratio    ||grad(L_defect)|| / ||grad(L_tail)||
  norm_n/norm_d mean feature norm of adapter(normal) / adapter(defect)

  p99_n         cosine p99 of the normal query distances to the bank
  mean_d        cosine mean of the defect distances to the bank

NORM AUDIT (why this matters): both the training loss (models.nn_distances) and
the evaluation (tail_calib.patch_dists_all) L2-normalise before measuring
distance. So the adapter cannot exploit radial norm -- every number above is
already a cosine quantity on the unit sphere. If the norms grow while the
cosine p99 also grows, that is a genuine DIRECTIONAL change, not radial
inflation, and the hyperspherical-separation remedy has no basis.

Usage: python experiments/model_v0/objective_audit.py [--steps N]
Writes results/model_v0/metrics/objective_audit.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, r"E:\work\freshman")
sys.path.insert(0, r"E:\work\freshman\experiments\model_v0")
from models import ResidualAdapter, tail_loss, preserve_loss, soft_separation_loss
from models.tail_adapter import l2norm, nn_distances
from tail_calib import RESULTS, load_cache, split_train, mean_top1p

OBJECTS = ["wallplugs", "sheet_metal", "vial"]
KS = 8192          # bank patches used for the loss
QS = 6144          # normal query patches per step
DS = 2048          # defect patches per step
METRICS = os.path.join(RESULTS, "metrics")


def grad_vec(loss, params):
    g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return [x.detach().clone() if x is not None else torch.zeros_like(p)
            for x, p in zip(g, params)]


def flat(v):
    return torch.cat([x.reshape(-1) for x in v])


def run(obj, steps, device="cuda"):
    tr, te = load_cache(obj)
    bank_idx, tailq_idx = split_train(tr["feats"].shape[0], 0.8, 0)
    bank = torch.from_numpy(tr["feats"][bank_idx].astype(np.float32)).to(device)
    tailq = torch.from_numpy(tr["feats"][tailq_idx].astype(np.float32)).to(device)
    defect = torch.from_numpy(np.load(os.path.join(
        RESULTS, "proxy", f"{obj}_K4boot.npy")).astype(np.float32)).to(device)

    gen = torch.Generator(device=device).manual_seed(0)
    bank_flat = bank.reshape(-1, 384)
    bank_flat = bank_flat[torch.randperm(len(bank_flat), generator=gen,
                                         device=device)[:KS]]
    tq_flat = tailq.reshape(-1, 384)

    adapter = ResidualAdapter(384, 128).to(device)
    params = list(adapter.parameters())
    opt = torch.optim.Adam(params, lr=2e-3)

    rows = []
    for step in range(steps):
        z = tq_flat[torch.randint(0, len(tq_flat), (QS,), generator=gen,
                                  device=device)]
        d = defect[torch.randint(0, len(defect), (DS,), generator=gen,
                                 device=device)]
        bank_new = adapter(bank_flat)
        L_tail = tail_loss(nn_distances(adapter(z), bank_new), 0.01)
        L_def = soft_separation_loss(
            nn_distances(adapter(z), bank_new),
            nn_distances(adapter(d), bank_new), 0.05, 0.1)
        L_pres = preserve_loss(adapter(z), z)
        loss = L_tail + L_def + L_pres

        if step % 25 == 0 or step == steps - 1:
            gt, gd = grad_vec(L_tail, params), grad_vec(L_def, params)
            ft, fd = flat(gt), flat(gd)
            cos = float(torch.dot(ft, fd) /
                        (ft.norm() * fd.norm() + 1e-12))
            with torch.no_grad():
                zn, zd = adapter(z), adapter(d)
                dmap = 1.0 - l2norm(zn) @ l2norm(bank_new).T
                # per-query 1-NN in chunks
                best = []
                for i in range(0, len(zn), 1024):
                    best.append((1.0 - l2norm(zn[i:i+1024]) @
                                 l2norm(bank_new).T).max(dim=1).values)
                dn = torch.cat(best)
                bestd = []
                for i in range(0, len(zd), 1024):
                    bestd.append((1.0 - l2norm(zd[i:i+1024]) @
                                  l2norm(bank_new).T).max(dim=1).values)
                dd = torch.cat(bestd)
            rows.append({
                "object": obj, "step": step,
                "L_tail": float(L_tail.detach()), "L_defect": float(L_def.detach()),
                "L_preserve": float(L_pres.detach()),
                "grad_cos": cos,
                "grad_ratio": float(fd.norm() / (ft.norm() + 1e-12)),
                "norm_normal": float(zn.norm(dim=1).mean()),
                "norm_defect": float(zd.norm(dim=1).mean()),
                "p99_normal_cos": float(torch.quantile(dn, 0.99)),
                "mean_defect_cos": float(dd.mean()),
                "mean_normal_cos": float(dn.mean()),
            })
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    args = ap.parse_args()
    os.makedirs(METRICS, exist_ok=True)
    all_rows = []
    for o in OBJECTS:
        print(f"=== {o} ===", flush=True)
        all_rows.extend(run(o, args.steps))
    df = pd.DataFrame(all_rows)
    out = os.path.join(METRICS, "objective_audit.csv")
    df.to_csv(out, index=False)

    print("\n" + "=" * 96)
    print("OBJECTIVE AUDIT -- is the soft loss inflating radially, or rotating "
          "directionally?")
    print("=" * 96)
    for o in OBJECTS:
        d = df[df.object == o]
        print(f"\n--- {o} ---")
        print(d[["step", "L_tail", "L_defect", "grad_cos", "grad_ratio",
                 "norm_normal", "norm_defect", "p99_normal_cos",
                 "mean_defect_cos"]].round(3).to_string(index=False))
        ng = d.norm_normal.iloc[-1] / max(d.norm_normal.iloc[0], 1e-9)
        pg = d.p99_normal_cos.iloc[-1] / max(d.p99_normal_cos.iloc[0], 1e-9)
        print(f"  norm growth x{ng:.2f}   cosine-p99 growth x{pg:.2f}   "
              f"neg grad_cos {(d.grad_cos < 0).mean()*100:.0f}% of steps")
    print("\n  Distance is a COSINE quantity in both training and evaluation "
          "(both L2-normalise),")
    print("  so a rising cosine p99 is a DIRECTIONAL change even if norms also "
          "grow.")
    print("  A persistently negative grad_cos is the basis for PCGrad; "
          "without it, adding")
    print("  gradient surgery would be complexity with no evidence behind it.")


if __name__ == "__main__":
    main()
