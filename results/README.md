# results/ — figure/table → script mapping (traceability, v2)

- `master_regression.csv` ← `run_subset` regression eval (5-fold grouped CV + capped test metrics for every model; `scripts/fill_cv_gaps.py` backfilled hgb_tuned/mean CV deterministically)
- `master_classification.csv` ← tidy long rows (fd,task,approach,model,class,precision,recall,f1,support,accuracy,balanced_acc,mcc,kappa,roc_auc,pr_auc); regress-threshold computed once per subset
- `{FD}_summary.json` ← sizes, CV, best HGB params, sensors, threshold{rule,target_met,test recall/precision+CIs}, lead_time{medians/IQRs/premature/FAR/anytime}, costs, deep info, clf_cv headlines
- `{FD}_regression_bins.csv` ← per-RUL-bin errors (capped, last cycle); `{FD}_regression_cis.csv` ← engine-bootstrap CIs for RMSE_cap/PHM_cap
- `{FD}_classification_cis.csv` ← engine-bootstrap CIs (critical recall, macro-F1, positive recall/F1)
- `{FD}_cost_sensitivity.csv` ← FN:FP = 5/10/20/50 OOF-optimal thresholds + frozen test cost/recall/precision
- `{FD}_leadtime_{model}.csv` ← `evaluate.lead_time_analysis` on deployed model
- `{FD}_importance_sensor_{model}.csv` ← grouped neg-RMSE permutation on grouped holdout; `{FD}_permutation_{model}.csv` ← per-feature top-15
- `{FD}_conformal.txt` (mode "w") + `{FD}_conformal_bins.csv` ← Mondrian split-conformal on deployed model, OOF truncation residuals
- `ablations.csv` ← (subset,ablation,setting,metric_name,cv_mean,cv_std,test_value); `deep_window_ablation.csv` ← LSTM/CNN × {20,30,50}
- `before_after.csv` ← v1→v2 comparison (regression capped RMSE/PHM_cap + classification macro-F1/critical recall)
- `test_class_counts.csv` ← all-rows test class counts; `wallclock.json` ← per-subset runtimes
- `results_v1/` + `figures_v1/` ← frozen v1 outputs for the before/after comparison
- figures ← `run_pipeline` (per-subset) + `scripts/build_extra_charts.py` (cross-cutting: pipeline diagram, compare bars, cost, ablations, conformal calibration, deep window)
