# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import pandas as pd, numpy as np

ap=argparse.ArgumentParser()
ap.add_argument("--input",default=r"results\model_v0\metrics\tsn_failure_audit\causal_audit_cells.csv")
args=ap.parse_args()
d=pd.read_csv(args.input)
pd.set_option("display.width",240)

methods=[
"auc_naive_replay","auc_q","auc_minus_q","auc_abs_oracle_full",
"auc_matched_support","auc_size_only_common_ref","auc_wrong_cyclic",
"auc_shuffle_mean","auc_matched_oracle"
]
print("=== replay sanity ===")
print("max q_ref abs diff:",d.q_ref_abs_diff.max())
print("max naive AUC vs formal:",(d.auc_naive_replay-d.formal_tsn).abs().max())
print("max final raw replay diff:",(d.auc_final_raw_replay-d.formal_baseline_final).abs().max())

print("\n=== method means ===")
for c in methods:
    print(f"{c:30s} {d[c].mean():8.3f}")

print("\n=== causal gains ===")
for c in ["gain_oracle_full_vs_naive","gain_matched_vs_naive",
          "gain_matched_vs_shuffle","gain_matched_oracle_vs_matched"]:
    print(f"{c:34s} mean={d[c].mean():+.3f} median={d[c].median():+.3f} "
          f"positive={(d[c]>0).sum()}/{len(d)}")

print("\n=== per cell ===")
cols=["object","shot","split","reason","formal_tsn","auc_naive_replay","auc_q","auc_minus_q",
      "auc_abs_oracle_full","auc_matched_support","auc_shuffle_mean","auc_matched_oracle",
      "calibration_gap_full","matched_gap_abs_mean","bank_center_shift_abs_mean"]
print(d[cols].round(3).sort_values("formal_tsn").to_string(index=False))

# Conservative interpretation flags; these are prompts for scientific review, not automatic truth.
oracle_gain=d.gain_oracle_full_vs_naive.mean()
matched_gain=d.gain_matched_vs_naive.mean()
matched_shuffle=d.gain_matched_vs_shuffle.mean()
trainshift=d.gain_matched_oracle_vs_matched.mean()
direction_gain=(d[["auc_q","auc_minus_q"]].max(axis=1)-d.auc_abs_oracle_full).mean()

print("\n=== provisional causal reading ===")
if oracle_gain >= 10:
    print("* Strong evidence that center calibration is a major failure source (oracle center restores >=10 AUROC on average).")
else:
    print("* Oracle-center rescue is <10 AUROC on average: center error alone is unlikely to explain the collapse.")

if matched_gain >= 5 and matched_shuffle > 2:
    print("* Matched bank identity has evidence of causal value: matched > naive and > shuffled pairing.")
elif matched_gain >= 5:
    print("* Matched-domain helps, but shuffled control does not clearly isolate bank identity; could be ensemble/bank-size effect.")
else:
    print("* Matched support-only calibration does not strongly rescue the selected cells.")

if trainshift >= 10:
    print("* Matched oracle >> matched support-only: train-to-test normal calibration/identifiability is a major remaining cause.")
else:
    print("* Matched oracle adds <10 AUROC on average over matched support-only.")

if direction_gain >= 10:
    print("* One-sided q/-q beats oracle-centered absolute deviation by >=10 on average: directionality deserves primary attention.")
else:
    print("* No strong average evidence that absolute value alone is the dominant cause.")

print("\nDo NOT promote any new mechanism from this 23-cell diagnosis directly to a paper claim.")
