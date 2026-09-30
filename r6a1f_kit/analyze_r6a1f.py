# -*- coding: utf-8 -*-
"""R6-A1F reports.

The verdict rules implemented here are frozen in
`R6A1F_VERDICT_RULES_FROZEN.md`, written BEFORE the trace was run. No layer,
radius or alpha is selected anywhere; `early` is a fixed label (block 2, 25%
depth by continuation of the project's 5/8/11 pin drop), while `onset_layer` is
a RESULT read off the dense curve.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "r6a1f_kit"))
sys.path.insert(0, str(ROOT / "r6a1e_preencoding"))

from r6a1f_utils import (fit_linear, resid_from_fit, permutation_null,   # noqa: E402
                         cluster_bootstrap, spearman, LAYER_NAME)
from scipy.stats import wilcoxon                                         # noqa: E402

OUT = ROOT / "results" / "model_v0" / "metrics" / "r6a1f_layerwise"
A1E = ROOT / "results" / "model_v0" / "metrics" / "r6a1e_preencoding"
N_PERM = 2000
N_BOOT = 2000

S = pd.read_csv(OUT / "per_series_layer.csv")
CELL = pd.read_csv(OUT / "matched_cells.csv")
RING = pd.read_csv(OUT / "radius_profile_raw.csv")
OBJECTS = list(dict.fromkeys(S.object))
LAYERS = sorted(S.layer_index.unique())

# ------------------------------------------------------------------ per stratum
rows = []
for (obj, r, don, l), g in S.groupby(["object", "radius", "donor_rank", "layer_index"]):
    is_bad = (g.series == "bad").to_numpy()
    cluster = g.bad_index.to_numpy()
    drift = g.drift_mean.to_numpy(float)
    topo = g.drift_top01.to_numpy(float)
    dpix = g.delta_pixel.to_numpy(float)
    dtok = g.delta_token.to_numpy(float)

    fit = fit_linear(dpix[~is_bad], drift[~is_bad])
    flag = "OK"
    if fit is None:
        flag = "LOW_SPREAD"
    d_good = dpix[~is_bad]
    lo, hi = np.quantile(d_good, [.05, .95]) if len(d_good) else (np.nan, np.nan)
    overlap = float(np.mean((dpix[is_bad] >= d_good.min()) & (dpix[is_bad] <= d_good.max()))) \
        if len(d_good) else np.nan
    extrap = float(np.mean(dpix[is_bad] > d_good.max())) if len(d_good) else np.nan
    if np.isfinite(extrap) and extrap > 0.2 and flag == "OK":
        flag = "EXTRAPOLATED"

    rec = dict(object=obj, radius=r, donor_rank=don, layer_index=l,
               layer_name=LAYER_NAME.get(l, ""), n_bad=int(is_bad.sum()),
               n_good=int((~is_bad).sum()),
               drift_bad_mean=float(drift[is_bad].mean()),
               drift_good_mean=float(drift[~is_bad].mean()),
               drift_diff=float(drift[is_bad].mean() - drift[~is_bad].mean()),
               drift_top01_bad=float(topo[is_bad].mean()),
               drift_top01_good=float(topo[~is_bad].mean()),
               drift_top01_diff=float(topo[is_bad].mean() - topo[~is_bad].mean()),
               frac_bad_gt_good=float(np.mean(
                   [np.mean(drift[is_bad][cluster[is_bad] == c] >
                            np.median(drift[~is_bad][cluster[~is_bad] == c]))
                    for c in np.unique(cluster[is_bad])])),
               delta_pixel_bad_mean=float(dpix[is_bad].mean()),
               delta_pixel_good_mean=float(dpix[~is_bad].mean()),
               delta_token_bad_mean=float(dtok[is_bad].mean()),
               delta_token_good_mean=float(dtok[~is_bad].mean()),
               overlap_frac=overlap, extrap_frac=extrap, flag=flag,
               sd_delta_good=float(np.std(d_good, ddof=1)) if len(d_good) > 1 else np.nan)
    if fit is None:
        rec.update(dict(slope_good=np.nan, intercept_good=np.nan, r2_good=np.nan,
                        resid_mean=np.nan, resid_lo=np.nan, resid_hi=np.nan,
                        perm_p=np.nan, perm_p_lower=np.nan,
                        resid_mean_tokencov=np.nan, slope_diff_tokencov=np.nan,
                        matched_estimate=np.nan, matched_n=0, matched_sign_agrees=False))
    else:
        res = resid_from_fit(fit, dpix[is_bad], drift[is_bad])
        perm = permutation_null(dpix, drift, cluster, is_bad, N_PERM, seed=20260930 + l * 17 + r)
        boot = cluster_bootstrap(dpix, drift, cluster, is_bad, N_BOOT, seed=20260930 + l * 31 + r)
        fit_tok = fit_linear(dtok[~is_bad], drift[~is_bad])
        # matched-subset: keep only bads whose delta lies inside the good central 90%
        keep = (dpix[is_bad] >= lo) & (dpix[is_bad] <= hi)
        if keep.sum() >= 3:
            matched = float(drift[is_bad][keep].mean()
                            - drift[~is_bad][(dpix[~is_bad] >= lo) & (dpix[~is_bad] <= hi)].mean())
            mn = int(keep.sum())
        else:
            matched, mn = np.nan, 0
        rec.update(dict(
            slope_good=fit["b"], intercept_good=fit["a"], r2_good=fit["r2"],
            resid_mean=float(res.mean()), resid_lo=boot["lo"], resid_hi=boot["hi"],
            perm_p=perm["p"], perm_p_lower=perm["p_lower"], perm_null_sd=perm["null_sd"],
            resid_mean_tokencov=float(resid_from_fit(fit_tok, dtok[is_bad], drift[is_bad]).mean())
            if fit_tok is not None else np.nan,
            slope_diff_tokencov=(fit_tok["b"] - fit_linear(dtok[~is_bad], drift[~is_bad])["b"])
            if fit_tok is not None else np.nan,
            matched_estimate=matched, matched_n=mn,
            matched_sign_agrees=bool(np.isfinite(matched) and np.sign(matched) == np.sign(res.mean())),
            matched_lo=float(lo), matched_hi=float(hi)))
    rows.append(rec)

T = pd.DataFrame(rows)
T.to_csv(OUT / "confound_control.csv", index=False)

# donor-collapsed
dm = T.groupby(["object", "radius", "layer_index"]).agg(
    n_bad=("n_bad", "max"), n_good=("n_good", "max"),
    delta_pixel_bad=("delta_pixel_bad_mean", "mean"),
    delta_pixel_good=("delta_pixel_good_mean", "mean"),
    delta_token_bad=("delta_token_bad_mean", "mean"),
    delta_token_good=("delta_token_good_mean", "mean"),
    drift_bad=("drift_bad_mean", "mean"), drift_good=("drift_good_mean", "mean"),
    drift_diff=("drift_diff", "mean"), drift_top01_diff=("drift_top01_diff", "mean"),
    resid=("resid_mean", "mean"), perm_p=("perm_p", "max"),
    perm_p_lower=("perm_p_lower", "max"),
    resid_tokencov=("resid_mean_tokencov", "mean"),
    matched=("matched_estimate", "mean"), matched_n=("matched_n", "min"),
    sign_agree=("matched_sign_agrees", lambda x: bool(np.all(x))),
    overlap=("overlap_frac", "mean"), extrap=("extrap_frac", "mean"),
    slope_good=("slope_good", "mean"), r2_good=("r2_good", "mean"),
    flag=("flag", lambda x: "LOW_SPREAD" if (x == "LOW_SPREAD").any() else x.iloc[0])
).reset_index().rename(columns={"resid": "resid_mean"})

dm["layer_name"] = dm.layer_index.map(lambda l: LAYER_NAME.get(l, ""))
dm["depth_frac"] = (dm.layer_index + 1) / 12.0
# donor sign agreement must be read off the DONOR-LEVEL table, before collapsing
dsa = (T.groupby(["object", "radius", "layer_index"]).resid_mean
       .apply(lambda s: bool(len(s) == 2 and ((s < 0).all() or (s > 0).all())))
       .rename("donor_sign_agree").reset_index())
dm = dm.merge(dsa, on=["object", "radius", "layer_index"], how="left")

# per-bad wilcoxon on drift_diff at the required layers
bad_rows = []
for (obj, r, don, l), g in S.groupby(["object", "radius", "donor_rank", "layer_index"]):
    piv = g.pivot_table(index="bad_index", columns="series", values="drift_mean")
    if "bad" not in piv:
        continue
    goods = [c for c in piv.columns if c.startswith("good")]
    d = piv["bad"] - piv[goods].mean(axis=1)
    try:
        wp = float(wilcoxon(d.to_numpy(), alternative="two-sided",
                            zero_method="wilcox").pvalue) if np.any(d.to_numpy() != 0) else 1.0
    except Exception:
        wp = np.nan
    bad_rows.append(dict(object=obj, radius=r, donor_rank=don, layer_index=l,
                         n_bad=len(d), drift_diff_mean=float(d.mean()),
                         frac_pos=float((d > 0).mean()),
                         q05=float(d.quantile(.05)), q50=float(d.median()),
                         q95=float(d.quantile(.95)), wilcoxon_p=wp))
B = pd.DataFrame(bad_rows)
B.to_csv(OUT / "bad_good_difference.csv", index=False)

dm = dm.merge(
    B.groupby(["object", "radius", "layer_index"]).wilcoxon_p.max().reset_index(),
    on=["object", "radius", "layer_index"], how="left")

# ------------------------------------------------- required summary files
CFG_LAYERS = {2: "early", 5: "mid", 8: "midlate", 11: "final"}
dm[dm.layer_index.isin(CFG_LAYERS)].to_csv(OUT / "layer_drift_summary.csv", index=False)
dm.to_csv(OUT / "dense_layer_curve.csv", index=False)

rp = RING[RING.ring != "ALL"].groupby(
    ["object", "radius", "layer_index", "ring", "series"]).drift_mean.mean().reset_index()
rpv = rp.pivot_table(index=["object", "radius", "layer_index", "ring"],
                     columns="series", values="drift_mean").reset_index()
goods = [c for c in rpv.columns if str(c).startswith("good")]
rpv["drift_good_mean"] = rpv[goods].mean(axis=1)
rpv["drift_diff"] = rpv["bad"] - rpv["drift_good_mean"]
rpv.rename(columns={"bad": "drift_bad_mean"}, inplace=True)
for c in ["object", "radius", "layer_index", "ring", "drift_bad_mean",
          "drift_good_mean", "drift_diff"]:
    pass
rpv["layer_name"] = rpv.layer_index.map(lambda l: LAYER_NAME.get(l, ""))
rpv[["object", "radius", "layer_index", "layer_name", "ring",
     "drift_bad_mean", "drift_good_mean", "drift_diff"]].to_csv(
    OUT / "radius_profile.csv", index=False)

# ------------------------------------------------- score link (Q3)
pp = pd.read_csv(A1E / "per_pair.csv")
pp = pp[(pp.alpha == 0.01) & (pp.object.isin(OBJECTS))]
pb = pp.groupby(["object", "bad_index", "radius", "donor_rank"]).apply(
    lambda g: pd.Series({"pwr_change": g.groupby("good_index").apply(
        lambda h: h.cf_win.mean() - h.raw_win.mean()).mean()}),
    include_groups=False).reset_index()
link = S[S.layer_index == 11].pivot_table(
    index=["object", "bad_index", "radius", "donor_rank"], columns="series",
    values="drift_mean").reset_index()
gcols = [c for c in link.columns if str(c).startswith("good")]
link["drift_diff"] = link["bad"] - link[gcols].mean(axis=1)
link = link.merge(pb, on=["object", "bad_index", "radius", "donor_rank"], how="inner")
sl = []
for (obj, r), g in link.groupby(["object", "radius"]):
    rho, pv = spearman(g.drift_diff, g.pwr_change)
    sl.append(dict(object=obj, radius=r, n=int(len(g)), spearman_rho=rho, p= pv))
    # NOTE: PWR is a win-rate and the replacement is non-neutral for it (A1E);
    # this link is reported for completeness, the drift residual is the primary.
sl.append(dict(object="ALL_POOLED", radius=-1, n=int(len(link)),
               spearman_rho=spearman(link.drift_diff, link.pwr_change)[0],
               p=spearman(link.drift_diff, link.pwr_change)[1]))
pd.DataFrame(sl).to_csv(OUT / "score_link.csv", index=False)

# ------------------------------------------------- verdicts (frozen rules)
verd = []
coll = pd.read_csv(A1E / "raw_full_good_baseline.csv")
coll = coll[coll.alpha == 0.01]
for obj in OBJECTS:
    z = dm[dm.object == obj]
    if not len(z):
        continue
    # onset: first layer with resid>0 at r=16, perm p<.05, sustained to block 11
    c16 = z[z.radius == 16].sort_values("layer_index")
    onset, rule = "NONE_DETECTED", "resid>0 & perm_p<.05 & sustained to block 11"
    for i, row in c16.iterrows():
        pos = (c16.loc[i:, "resid_mean"] > 0).all()
        if row.resid_mean > 0 and np.isfinite(row.perm_p) and row.perm_p < 0.05 and pos:
            onset = int(row.layer_index)
            break
    rho, pv = spearman(c16.layer_index.to_numpy(), c16.resid_mean.to_numpy())
    f = c16[c16.layer_index == 11].resid_mean.mean()
    e = c16[c16.layer_index == 2].resid_mean.mean()
    # NOTE (diagnostic, does not alter any frozen verdict): the INPUT_STAGE
    # comparison resid(2) >= 0.8*resid(11) was written for a POSITIVE effect.
    # On a near-zero or negative residual it is a comparison outside its domain
    # and its label is not interpretable. The flag below records that, and the
    # verdict itself is left exactly as the frozen rule computes it.
    trend_applicable = bool(np.isfinite(f) and f > 0)
    if np.isfinite(rho) and rho >= 0.6 and pv < 0.05 and f > e:
        trend = "HIERARCHICAL"
    elif np.isfinite(e) and np.isfinite(f) and f != 0 and e >= 0.8 * f:
        trend = "INPUT_STAGE"
    else:
        trend = "MIXED"

    r8 = z[(z.radius == 8) & (z.layer_index == 11)]
    r16 = z[(z.radius == 16) & (z.layer_index == 11)]
    v8, v16 = float(r8.resid_mean.iloc[0]) if len(r8) else np.nan, \
        float(r16.resid_mean.iloc[0]) if len(r16) else np.nan
    p8, p16 = float(r8.perm_p.iloc[0]) if len(r8) else np.nan, \
        float(r16.perm_p.iloc[0]) if len(r16) else np.nan
    sa8 = bool(r8.donor_sign_agree.iloc[0]) if len(r8) else False
    sa16 = bool(r16.donor_sign_agree.iloc[0]) if len(r16) else False
    naive_max = float(z[z.radius.isin([8, 16])].drift_diff.abs().max())
    if v8 > 0 and v16 > 0 and p8 < 0.05 and p16 < 0.05 and sa8 and sa16:
        verdict = "PROPAGATION_SUPPORTED"
    elif v8 < 0 and v16 < 0 and p8 < 0.05 and p16 < 0.05:
        verdict = "UNDER_PROPAGATION"
    elif naive_max > 0.01:
        verdict = "MAGNITUDE_EXPLAINED"
    else:
        verdict = "MIXED_INDETERMINATE"

    cb = coll[coll.object == obj].set_index("radius")
    score_r16 = float(cb.loc[16, "raw_full_good_pairwise_auc"]) if 16 in cb.index else np.nan
    score_r0 = float(cb.loc[0, "raw_full_good_pairwise_auc"]) if 0 in cb.index else np.nan
    verd.append(dict(object=obj, n_bad=int(c16.n_bad.max() if len(c16) else 0),
                     onset_layer=onset, onset_rule=rule,
                     depth_trend_spearman_rho=rho, depth_trend_p=pv, depth_trend=trend,
                     depth_trend_applicable=trend_applicable,
                     resid_r8=v8, resid_r16=v16, perm_p_r8=p8, perm_p_r16=p16,
                     donor_sign_agree_r8=sa8, donor_sign_agree_r16=sa16,
                     naive_drift_diff_abs_max=naive_max,
                     propagation_score=float(np.nanmean([v8, v16])),
                     score_baseline_r0=score_r0, score_at_r16=score_r16,
                     score_collapse=score_r0 - score_r16 if np.isfinite(score_r0) else np.nan,
                     verdict=verdict))
V = pd.DataFrame(verd)
V.to_csv(OUT / "object_verdict.csv", index=False)

# Q3 across objects (frozen rule): propagation strength vs anomaly collapse.
# n=5, so |rho|=1 gives p~0.017 -- directional evidence only.
rho_q3, p_q3 = spearman(V.propagation_score.to_numpy(), V.score_collapse.to_numpy())
pd.DataFrame([dict(n_objects=len(V), spearman_rho=rho_q3, p=p_q3,
                   power_note="n=5; |rho|=1 gives p~0.017; directional evidence only")]
             ).to_csv(OUT / "propagation_vs_collapse.csv", index=False)
print("=== Q3: propagation_score vs anomaly collapse (across %d objects) ===" % len(V))
print(pd.DataFrame([dict(object=r.object, propagation_score=round(r.propagation_score,5),
                         score_collapse=round(r.score_collapse,2)) for r in V.itertuples()]
                   ).to_string(index=False))
print("Spearman rho = %.3f  p = %.3f   (n=5, directional only)" % (rho_q3, p_q3))
print()
print("=== R6-A1F OBJECT VERDICTS ===")
print(V[["object", "onset_layer", "depth_trend", "depth_trend_spearman_rho",
         "resid_r8", "resid_r16", "perm_p_r16", "verdict"]].round(4).to_string(index=False))
print()
print("=== final-layer (block 11) drift_diff vs magnitude-controlled residual ===")
z = dm[(dm.layer_index == 11) & (dm.radius.isin([8, 16]))]
print(z[["object", "radius", "drift_diff", "resid_mean", "perm_p",
         "delta_pixel_bad", "delta_pixel_good", "overlap", "extrap",
         "matched", "matched_n"]].round(4).to_string(index=False))
print()
costat = pd.read_csv(OUT / "confound_control.csv")
print("NaN audit: resid_mean", int(costat.resid_mean.isna().sum()), "/", len(costat),
      "| perm_p", int(costat.perm_p.isna().sum()))
print("\nDONE", OUT)
