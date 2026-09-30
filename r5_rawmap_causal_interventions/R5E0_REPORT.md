# R5-E0: Same-scene constant-extent readout monotonicity

Natural-control design:
- same physical defect scene
- same defect-patch count
- different lighting/acquisition views
- x-axis = true GT-region mean score minus same-image clean-region mean score
- compare global mean, top1%, max, and q=log(max/top1%)

Number of eligible constant-extent scene groups: 95

    readout  n_scenes  median_rho  mean_rho  positive_scenes  negative_scenes expected_direction  expected_direction_count  exact_binom_p
global_mean        95   -0.285714 -0.088571               42               52           positive                        42   8.909732e-01
       top1        95    0.714286  0.545414               81               13           positive                        81   6.201254e-13
 global_max        95    0.392857  0.255489               62               31           positive                        62   1.916436e-03
          q        61   -0.657143 -0.499757               11               50           negative                        50   2.294416e-07

Interpretation:
- If ordinary magnitude readouts track GT evidence positively while q tracks it negatively, the map contains a usable strength signal and q specifically reverses/compresses it.
- This is not a synthetic intervention; it is a natural repeated-scene control. It does not replace the raw-map R5-A/B/C interventions.
