# RC Stage1 run amendment

The first workflow execution (GitHub Actions run 36721800471) completed the scientific analysis and printed all frozen results successfully, but the job failed afterwards while persisting outputs because the repository-wide .gitignore ignores *.log and the workflow attempted to git add RC_STAGE1_RUN.log.

This is an output-persistence bug only:
- the preregistration is unchanged;
- run_rc_stage1.py is unchanged;
- no scientific statistic, threshold, object selection, or verdict rule is changed;
- the rerun must execute the identical analysis code.

Fix: save/add the run transcript with force, then commit the exact result bundle.
