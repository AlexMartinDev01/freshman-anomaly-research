python experiments/model_v0/cache_m1_repr.py --repr wrn50 > results/model_v0/cache_m1_wrn50.log 2>&1
echo "WRN50_EXIT=$?"
python experiments/model_v0/cache_m1_repr.py --repr dino672 > results/model_v0/cache_m1_dino672.log 2>&1
echo "DINO672_EXIT=$?"
echo "M1_CACHE_DONE"
