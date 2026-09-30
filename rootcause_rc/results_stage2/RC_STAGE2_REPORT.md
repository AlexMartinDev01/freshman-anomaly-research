# RC Stage2 Results

Frozen verdict:
 loc2_A  loc2_B  loc2_C  loc2_D  loc2_E  macaroni2_median_patch_auc  macaroni2_frac_patch_auc_ge90  macaroni2_image_auc_1nn  macaroni2_median_extent_ratio  macaroni2_frac_defect_lt_tail  extent_oracle_gain  defect_excision_centered_auc  translated_null_mean  excision_specificity_gap                        loc2_verdict  other_hard_patch_stage_condition                    architecture_verdict
   True    True    True    True    True                    0.999184                       0.966667                   0.6535                          0.125                            1.0            6.433333                          57.9             71.327556                 13.427556 AGGREGATION_DILUTION_ROOT_SUPPORTED                              True DOWNSTREAM_READOUT_BOTTLENECK_SUPPORTED

Object diagnostics:
    object  n_bad  median_within_image_patch_auc_1nn  frac_patch_auc_ge90  median_extent_ratio  frac_defect_lt_tail_k  image_auc_1nn  patch_ap_1nn  patch_ap_probe  patch_ap_gap  patch_auc_probe
    bottle     63                           0.987728             1.000000             7.272727               0.000000       1.000000      0.867373        0.913362      0.045988         0.978401
     cable     92                           0.967635             0.815217             5.090909               0.010870       0.937594      0.675947        0.786626      0.110679         0.978172
chewinggum     98                           0.998856             0.887755             0.961538               0.500000       0.970600      0.824756        0.929279      0.104523         0.997217
 macaroni2     90                           0.999184             0.966667             0.125000               1.000000       0.653500      0.310111        0.824877      0.514766         0.992842
      pcb2    100                           0.930651             0.670000             0.678571               0.750000       0.806500      0.258149        0.635243      0.377093         0.980547
     screw    119                           0.910597             0.521008             0.545455               0.848739       0.774954      0.159046        0.816029      0.656983         0.995171

Interpretation boundary:
LOC2 is object-specific. ARCH2 establishes a downstream readout bottleneck at two different stages, but does not yet identify the exact high-dimensional variable lost by the patch-level scalar readout.