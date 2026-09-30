# RC Root-Cause Stage 1 Results\n\n## Frozen verdict\n\n   geo1_verdict  geo1_A  geo1_B  geo1_C  geo1_D  suff1_hard_pass_count suff1_hard_pass_objects  suff1_root_supported  cov1_median_object_gain  cov1_positive_objects  cov1_dominant_undercoverage_pattern            stage1_verdict
NOT_ESTABLISHED   False   False   False    True                      1                   screw                 False                 1.401108                     12                                False ROOT_CAUSE_NOT_YET_PINNED\n\n## RC-GEO1 kNN paired D vs N_H\n\n    object axis metric  n_images  median_delta  mean_delta  sign_fraction  p_greater   ci95_lo   ci95_hi
    bottle  knn     d1         2      0.042245    0.042245       1.000000   0.250000  0.013136  0.071355
    bottle  knn     d2         2      0.039755    0.039755       1.000000   0.250000  0.013967  0.065543
    bottle  knn     d5         2      0.040272    0.040272       1.000000   0.250000  0.015565  0.064979
    bottle  knn    d10         2      0.040932    0.040932       1.000000   0.250000  0.009194  0.072669
     cable  knn     d1        25      0.087586    0.084823       0.920000   0.000002  0.073752  0.117954
     cable  knn     d2        25      0.090293    0.086381       0.880000   0.000001  0.073424  0.119274
     cable  knn     d5        25      0.096278    0.091497       0.880000   0.000005  0.066607  0.136582
     cable  knn    d10        25      0.107021    0.096290       0.880000   0.000004  0.068916  0.133996
chewinggum  knn     d1        31     -0.028697   -0.040347       0.354839   0.986282 -0.073460  0.008075
chewinggum  knn     d2        31     -0.026987   -0.040945       0.322581   0.990654 -0.076016 -0.002915
chewinggum  knn     d5        31     -0.023629   -0.035519       0.387097   0.977185 -0.065348  0.007732
chewinggum  knn    d10        31     -0.028020   -0.029444       0.322581   0.973652 -0.072368 -0.003117
 macaroni2  knn     d1        90     -0.049058   -0.050611       0.355556   0.999999 -0.081748 -0.008701
 macaroni2  knn     d2        90     -0.062226   -0.057796       0.311111   1.000000 -0.089092 -0.020564
 macaroni2  knn     d5        90     -0.066518   -0.063754       0.300000   1.000000 -0.094934 -0.032249
 macaroni2  knn    d10        90     -0.066242   -0.070790       0.255556   1.000000 -0.106607 -0.030189
      pcb2  knn     d1        99      0.036527    0.034904       0.767677   0.000001  0.027087  0.046880
      pcb2  knn     d2        99      0.037168    0.039935       0.767677   0.000000  0.031151  0.048170
      pcb2  knn     d5        99      0.050428    0.050781       0.787879   0.000000  0.036123  0.064502
      pcb2  knn    d10        99      0.069057    0.061923       0.858586   0.000000  0.046871  0.076303
     screw  knn     d1       115      0.052867    0.044726       0.765217   0.000000  0.038794  0.068022
     screw  knn     d2       115      0.053812    0.043921       0.747826   0.000000  0.033638  0.064905
     screw  knn     d5       115      0.041926    0.030970       0.686957   0.000026  0.026731  0.055185
     screw  knn    d10       115      0.030773    0.024879       0.660870   0.000359  0.013123  0.056456\n\n## RC-GEO1 layer replication\n\n    object  axis  metric  n_images  median_delta  mean_delta  sign_fraction  p_greater   ci95_lo   ci95_hi
    bottle layer   final         2      0.042245    0.042245       1.000000   0.250000  0.013136  0.071355
    bottle layer     mid         2      0.005233    0.005233       0.500000   0.500000 -0.037853  0.048319
    bottle layer midlate         2      0.005115    0.005115       0.500000   0.500000 -0.023965  0.034195
     cable layer   final        25      0.087586    0.084823       0.920000   0.000002  0.073752  0.117954
     cable layer     mid        25      0.012062    0.011028       0.600000   0.245393 -0.023512  0.040854
     cable layer midlate        25      0.065798    0.060803       0.760000   0.000094  0.034148  0.090017
chewinggum layer   final        31     -0.028697   -0.040347       0.354839   0.986282 -0.073460  0.008075
chewinggum layer     mid        31     -0.042382   -0.053150       0.225806   0.999972 -0.074137 -0.023738
chewinggum layer midlate        31     -0.050429   -0.045467       0.290323   0.999920 -0.066604 -0.024518
 macaroni2 layer   final        90     -0.049058   -0.050611       0.355556   0.999999 -0.081749 -0.008701
 macaroni2 layer     mid        90     -0.086845   -0.085434       0.055556   1.000000 -0.099557 -0.070300
 macaroni2 layer midlate        90     -0.056035   -0.064087       0.144444   1.000000 -0.074463 -0.036583
      pcb2 layer   final        99      0.036527    0.034904       0.767677   0.000001  0.027087  0.046880
      pcb2 layer     mid        99     -0.009488    0.000477       0.444444   0.549996 -0.019473  0.011890
      pcb2 layer midlate        99      0.023576    0.021700       0.656566   0.000802  0.009706  0.038424
     screw layer   final       115      0.052867    0.044726       0.765217   0.000000  0.038794  0.068022
     screw layer     mid       115     -0.056905   -0.069193       0.200000   1.000000 -0.075010 -0.048264
     screw layer midlate       115     -0.050162   -0.057904       0.313043   1.000000 -0.071465 -0.027349\n\n## RC-GEO1 neighbour provenance control\n\n    object  n_images  median_distinct_support_images  median_frac_from_one_image
    bottle         2                        3.166667                    0.466667
     cable        25                        3.500000                    0.514286
chewinggum        31                        3.555556                    0.500000
 macaroni2        90                        3.200000                    0.527500
      pcb2        99                        3.384615                    0.483333
     screw       115                        2.750000                    0.600000\n\n## RC-SUFF1 scalar-1NN matched pairs\n\n    object  probe  n_pairs  caliper  one_nn_smd  median_abs_1nn_mismatch  median_delta_probe  mean_delta_probe  p_wilcoxon  p_pairswap   ci95_lo  ci95_hi  pass_gate
    bottle linsvm        0 0.032851         NaN                      NaN                 NaN               NaN         NaN         NaN       NaN      NaN      False
    bottle logreg        0 0.032851         NaN                      NaN                 NaN               NaN         NaN         NaN       NaN      NaN      False
     cable linsvm       11 0.029166    0.317662                 0.008955            1.514186          1.613932    0.000488    0.014749  0.957825 2.182558      False
     cable logreg       11 0.029166    0.317662                 0.008955            0.634226          0.536748    0.000488    0.015499  0.272981 0.740902      False
chewinggum linsvm        8 0.042150    0.231327                 0.010850            2.233511          1.938001    0.019531    0.028899 -0.694667 2.406581      False
chewinggum logreg        8 0.042150    0.231327                 0.010850            0.405159          0.419734    0.007812    0.030598  0.095123 0.683860      False
 macaroni2 linsvm       49 0.006062    0.111573                 0.002124            0.713650          0.755581    0.000000    0.000050  0.467096 0.965708      False
 macaroni2 logreg       49 0.006062    0.111573                 0.002124            0.186110          0.226133    0.000000    0.000050  0.139184 0.255226      False
      pcb2 linsvm       32 0.011508    0.179276                 0.004426            1.478117          1.521319    0.000001    0.000100  1.148411 2.096839      False
      pcb2 logreg       32 0.011508    0.179276                 0.004426            0.223809          0.287614    0.000007    0.000350  0.166904 0.430325      False
     screw linsvm       29 0.017050    0.032783                 0.004630            3.462680          3.721584    0.000000    0.000150  2.208455 4.360772       True
     screw logreg       29 0.017050    0.032783                 0.004630            0.679822          0.602946    0.000000    0.000050  0.416884 0.725428       True\n\nHard-object pass map: {"screw": true, "macaroni2": false, "pcb2": false}\n\n## RC-COV1 supporting 1-shot vs 4-shot, non-nested\n\n    object  n_splits  mean_delta_img_auc  median_delta_img_auc  positive_splits
    bottle         3            1.031746              0.396825                3
     cable         3            3.485757              3.504498                3
   capsule         3            5.956655              2.592740                3
    carpet         3            0.000000              0.000000                0
      grid         3            0.417711              0.250627                2
  hazelnut         3            2.404762              0.464286                2
   leather         3            0.000000              0.000000                0
 metal_nut         3            1.401108              0.439883                3
      pill         3            2.009456              1.936716                3
     screw         3           11.293298             11.190818                3
      tile         3            0.000000              0.000000                0
toothbrush         3            0.648148              0.277778                2
transistor         3            4.527778              4.750000                3
      wood         3            2.046784              0.087719                2
    zipper         3            1.400560             -0.026261                1\n\nMedian object mean gain: 1.4011; positive objects: 12; dominant pattern: False\n\n## Interpretation guardrail\n\nStage-1 support does not establish residual direction specifically. If Stage-1 is strongly supported, RC-DIR remains required.