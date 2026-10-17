# results/ — figure/table → script mapping (traceability)

- `master_regression.csv`, `master_classification.csv` ← `run_pipeline.main` (concats per-subset)
- `{FD}_regression.csv`, `{FD}_classification.csv` ← `run_pipeline.run_subset`
- `{FD}_summary.json` (sizes, CV, best params, sensors, thresholds, FAR, cost) ← `run_subset`
- `{FD}_leadtime_{model}.csv` ← `src.evaluate.lead_time_analysis`
- `{FD}_permutation_{model}.csv` ← `src.explain.permutation_top`
- `{FD}_conformal.txt` ← `run_pipeline.conformal_interval`
- `FD001_ablations.csv` ← `run_pipeline.ablations_fd001`
- `figures/{FD}_lifetime_hist|trajectories|corr_rul|monotonicity|classdist|smoothing.png` ← `run_pipeline.eda_subset`
- `figures/{FD}_kmeans_select|condnorm_effect.png` ← `run_pipeline.condition_cluster_plots`
- `figures/{FD}_pred_vs_true|residuals|trajectories_pred_{model}.png` ← `run_subset` (regression plots)
- `figures/{FD}_cm_3class|binary_hgb(_norm).png`, `{FD}_roc|pr_hgb.png`, `{FD}_thresh_tuning.png`, `{FD}_leadtime_*.png`, `{FD}_permutation_*.png`, `FD001_ablation_cap.png`, `master_rmse_heatmap.png` ← `run_subset`/`main`

All metrics computed from code outputs; report/slides read these files (see `scripts/build_report.py`, `scripts/build_slides.py`).
