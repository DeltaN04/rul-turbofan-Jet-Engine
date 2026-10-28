# CHANGELOG — v1 → v2 correction pass

## v2 (complete, 15 Oct)
All blocks done; full `--fresh` run (≈2.2 h CPU: 17.5/37.5/24/51.5 min per subset) + CV backfill.
pytest 11/11, 5/5 notebooks execute, app HTTP 200, self-audit 13/13 PASS.
Headline: FD002 HGB capped RMSE 18.27→12.87, FD004 23.87→13.17 (A1 fix); HGB CV-selected on
all subsets (11.61/13.19/9.85/13.06); tuned-t recall 1.00 everywhere; warn leads ≈50.
Scorecard: C1 PASS; C2 3/4 (FD001 hgb 0.769 narrow FAIL, CI covers target; logreg 1.00);
C3 PASS via policy (direct 3/4); C4 PASS. Honest negatives kept: class-weights hurt,
ops add nothing, deep < boosting, FD003 conformal uselessly wide, FD001/FD004 slight
conformal under-cover. 12/12 references verified (2 dropped); report 54 figs/14 tabs;
slides 13 + notes; viva 50 Q&A with v2 numbers.
- A1 DONE + verified: `build_feature_store` order fixed to normalise → variance-drop
  → smooth → rolling; KMeans inputs standardised; within-condition float-noise floor
  (1e-9 relative) so s1/s5/s19 normalise to exactly 0 and drop. FD002 smoke (default
  HGB, capped last-cycle): RMSE 12.75 / PHM_cap 750 (sanity range ≈12.5 / ≈690 ✓);
  dropped=[s18,s5,s19,s1]; top importance = cycle/s11/s4 (no condition proxies).
- A2/A3: 5-fold grouped CV for every model incl. MLP; selection by CV capped RMSE;
  full configs by default, `--smoke` for fast runs.
- A4: LSTM/CNN on all 4 subsets, by-engine holdout, ≤60 epochs/patience 8,
  seeds {42,43,44}, stratified-by-engine subsample ≥60% when >30k windows.
- A5-A9, Block B/C: in progress (see run_pipeline.py).
- A1: `build_feature_store` order fixed to normalise → variance-drop → smooth → rolling;
  KMeans inputs standardised; FD002/FD004 now drop condition-constant sensors.
- A2: model selection by 5-fold grouped-CV RMSE (capped) for every model incl. MLP/deep; test never used for selection.
- A3: final results use full configs (`fast=False`); `--smoke` flag for quick runs.
- A4: LSTM/CNN on all 4 subsets, by-engine validation, ≤60 epochs, seeds {42,43,44}, capped metrics + holdout score; deep window ablation.
- A5: tidy long-format classification CSV (precision/recall/F1/support per class, no duplicates); labelled confusion matrices; CV-best headline classifier; bootstrap CIs.
- A6: OOF threshold tuning with deployed model, truncation-point operating points, cost-optimal fallback + sensitivity table.
- A7: permutation importance with neg-RMSE scoring on validation, per-sensor groups, SHAP; cycle/ops ablations.
- A8: capped last-cycle metrics primary; per-RUL-bin errors; heatmap/bars on capped metric.
- A9: Mondrian (RUL-bin) conformal on deployed model, OOF residuals, overwrite mode, coverage ± SE + width per bin.
- A10: condnorm plots on informative sensors; test class counts; `results/ablations.csv`; multi-seed; loud reuse log; new tests.

## v1 (baseline — output folders removed as duplicates; v1→v2 numbers kept in `results/before_after.csv`)
- FD001 HGB test RMSE 12.79 (true) / 11.63 (capped); FD002 MLP 27.14; FD003 HGB 13.87; FD004 MLP 29.53.
- Known issues: smooth-before-normalise (A1); test-set model selection (A2); fast configs (A3);
  no deep models on FD002/FD004 (A4); duplicated/wide classification rows (A5); threshold/model mismatch (A6);
  R²-scored importance on 100 rows (A7); uncapped headline metrics (A8); under-covering conformal, append mode (A9).
