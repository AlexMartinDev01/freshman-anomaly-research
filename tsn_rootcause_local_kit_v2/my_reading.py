# -*- coding: utf-8 -*-
"""Independent reading of the causal audit.  Diagnosis only."""
import numpy as np
import pandas as pd

D = r"results\model_v0\metrics\tsn_failure_audit_v2"
pd.set_option("display.width", 240)
d = pd.read_csv(D + r"\causal_audit_cells.csv")
f = pd.read_csv(r"results\model_v0\metrics\tsn_formal_v2.csv")
base = f.set_index(["object", "shot", "split"]).baseline_agg_img_AUROC
d["baseline"] = [base.loc[(o, s, sp)] for o, s, sp in
                 zip(d.object, d.shot, d["split"])]
d["sign_gap"] = d.auc_q - d.auc_minus_q          # >0 : q right-signed
d["q_inverted"] = d.sign_gap < 0

print("=== replay ===")
print(f"  max |naive_replay - formal_tsn| = {(d.auc_naive_replay-d.formal_tsn).abs().max():.3e}")
print(f"  max q_ref abs diff              = {d.q_ref_abs_diff.max():.3e}")

print("\n=== is the q direction object- or baseline-dependent? ===")
print(f"  cells where q is right-signed : {int((~d.q_inverted).sum())}/{len(d)}")
print(f"  cells where q is INVERTED     : {int(d.q_inverted.sum())}/{len(d)}")
by_obj = d.groupby("object").agg(n=("sign_gap", "size"),
                                 mean_sign_gap=("sign_gap", "mean"),
                                 inverted=("q_inverted", "sum"),
                                 baseline=("baseline", "mean"),
                                 formal_tsn=("formal_tsn", "mean")).sort_values("baseline")
print(by_obj.round(1).to_string())
r = np.corrcoef(d.baseline, d.sign_gap)[0, 1]
print(f"\n  corr(baseline_agg, auc_q - auc_minus_q) = {r:+.3f}  "
      f"(n={len(d)}, cell level)")
ro = np.corrcoef(by_obj.baseline, by_obj.mean_sign_gap)[0, 1]
print(f"  corr at OBJECT level (n=14)             = {ro:+.3f}")

print("\n=== per-object: which direction is the anomaly score? ===")
print(by_obj.assign(verdict=np.where(by_obj.mean_sign_gap > 0,
                                     "q right-signed", "q INVERTED")).round(1).to_string())

print("\n=== does the |q - ref| construction fix an inverted object? ===")
for tag, sub in (("q right-signed", d[~d.q_inverted]), ("q inverted", d[d.q_inverted])):
    print(f"  {tag:15s} n={len(sub):2d}  naive {sub.formal_tsn.mean():6.2f}  "
          f"oracle {sub.auc_abs_oracle_full.mean():6.2f}  "
          f"matched {sub.auc_matched_support.mean():6.2f}  "
          f"matched_oracle {sub.auc_matched_oracle.mean():6.2f}  "
          f"|q| {sub.auc_q.mean():6.2f}")

print("\n=== the four pre-registered readings, on raw numbers ===")
def g(a, b):
    return float((d[a] - d[b]).mean())
print(f"  (1) Oracle Full  - Naive          = {g('auc_abs_oracle_full','auc_naive_replay'):+7.2f}"
      f"   [>0 would mean center-calibration is a cause]")
print(f"  (2) Matched      - Naive          = {g('auc_matched_support','auc_naive_replay'):+7.2f}"
      f"   [>0 means bank matching helps]")
print(f"      Matched      - Shuffled       = {g('auc_matched_support','auc_shuffle_mean'):+7.2f}"
      f"   [must exceed (2)'s effect to be causal, not smoothing]")
print(f"  (3) MatchedOracle- MatchedSupport = {g('auc_matched_oracle','auc_matched_support'):+7.2f}"
      f"   [support cannot identify the test-normal reference]")
print(f"  (4) max(q,-q)    - OracleAbs      = {float((d[['auc_q','auc_minus_q']].max(axis=1)-d.auc_abs_oracle_full).mean()):+7.2f}"
      f"   [absolute value destroys information]")
print(f"\n  ceiling check: the BEST single oracle arm is "
      f"{max(d.auc_abs_oracle_full.mean(), d.auc_matched_oracle.mean()):.2f} AUROC, "
      f"vs the baseline agg_all3 {d.baseline.mean():.2f}")

d.to_csv(D + r"\causal_audit_cells_enriched.csv", index=False)
print("\nwrote causal_audit_cells_enriched.csv")
