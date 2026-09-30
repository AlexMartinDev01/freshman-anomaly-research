set -x
python experiments/model_v0/subspacead_run.py --shots 8 --seeds 2 --augs 0 >> results/subspacead_b/run_aug0.log 2>&1
echo "AUG0_REMAINDER_EXIT=$?"
python experiments/model_v0/subspacead_run.py --shots 2 --seeds 1,2 --augs 30 >> results/subspacead_b/run_aug30.log 2>&1
echo "AUG30_K2_EXIT=$?"
python experiments/model_v0/subspacead_run.py --shots 4,8 --seeds 0,1,2 --augs 30 >> results/subspacead_b/run_aug30.log 2>&1
echo "AUG30_K48_EXIT=$?"
echo "ALL_DONE"
