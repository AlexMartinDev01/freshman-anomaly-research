# R5 MAIN AUTOMATIC REPORT

## R5-A Strength-only
Objects: 27
Objects with mean rho_q < 0: 11/27; exact sign p=0.876106
Objects with mean rho_top1 >= 0: 27/27; exact sign p=7.45058e-09
Paired object-level (rho_top1-rho_q) one-sided Wilcoxon p=7.45058e-09

Dataset summary:
dataset  n  mean_rho_top1  mean_rho_q  frac_q_negative
  mvtec 15       0.998001   -0.367099         0.733333
   visa 12       0.988228    0.294459         0.000000

Frozen A verdict:
A1 positive-control top1 >=0 in >=70% objects: True
A2 q negative in >=70% objects and p<.05: False
A3 top1-q paired Wilcoxon p<.05: True
A4 both datasets mean rho_q<0: False

## R5-B Extent-only on real normal backgrounds
Objects with mean rho_q <0: 27/27; sign p=7.45058e-09
All good-image fraction rho_q<0: 1.0000
All good-image fraction rho_top1>=0: 1.0000

Frozen B verdict:
B1 top1 nonnegative in 100% images: True
B2 >=80% images q-negative + every object median/mean q-negative + object sign p<.05:
True

## R5-C Spatial signal
No post-hoc best-alpha selection is used.
 alpha  n_objects  lcc_objects_over_q95  adj_objects_over_q95  mean_lcc_effect  mean_adj_effect
 0.005         27                    27                    27        24.835601        25.110469
 0.010         27                    27                    27        30.948775        32.743864
 0.020         27                    26                    27        33.086218        36.387761
 0.050         27                    24                    27        31.083162        34.427121

Interpretation is NOT automated beyond the frozen gates.
Object is the independent block.
Do not run LOCKED replication until this MAIN report is reviewed and frozen.
