# PROGRESS — Turbofan RUL project (TE7483)

## Done
- [x] Env inspected (Python 3.12, sklearn 1.9.1, torch 2.14.1 CPU, matplotlib 3.11.2). xgboost/lightgbm importable wheels need libomp (no brew) → documented fallback to sklearn HistGradientBoosting. (`requirements.txt` pins what is actually imported; report states fallback.)
- [x] Data unzipped to `data/raw/`, validated 26 cols, train/test/RUL counts asserted in `src/data.py`.
- [x] Project structure created per spec; `config.yaml`, `src/{data,preprocess,models,train,evaluate,explain,alerts,plots}.py`, `run_pipeline.py`, `tests/test_pipeline.py` (5 tests PASS).
- [x] `app.py` (Streamlit, causal features, green/amber/red policy), 5 notebooks (wrappers over real outputs).
- [x] Full pipeline launched in background: EDA → grouped-CV regression → tuned HGB → RF/HGB/Ridge/MLP + LSTM/CNN (FD001/FD003) → direct vs regress-threshold classification → threshold tuning (val only) → lead-time/cost → ablations → conformal → master tables.

## Key metrics so far
- Pending — pipeline running; figures appearing in `figures/` (FD001 EDA done). Will fill table here once `results/master_*.csv` exist.

## Open issues
- xgb/lgbm unusable without system libomp → using HGB/RF as gradient-boosting representative (honest fallback, stated in report).
- **Leakage caught & fixed (15 Oct):** `cycle_norm=cycle/max_cycle` let train RUL be reconstructed exactly (CV RMSE ~2, test RMSE ~85). Removed; fresh rerun in progress. Added `test_no_future_leak_features`. Deep-window models (no such feature) were unaffected and set the bar (LSTM RMSE ~15.4 on FD001).
- Report/slides/viva builders to run after pipeline finishes (scripts below).
- Notebooks need `jupyter nbconvert --execute` verification pass + `make all` reproduce check.

## Completion (15 Oct)
- Fresh full run post-fix: 4 subsets + ablations + conformal + master tables. pytest 6/6. All 5 notebooks execute via nbconvert. Demo HTTP 200.
- Deliverables: report.docx/pdf, presentation.pptx (13 slides + notes), viva_prep.md (numbers filled), app.py, 90 figures, 28 results files, 4 model pickles.
- Self-audit: 12/12 PASS (boundary FAIL was an audit-snippet typo; code verified PASS directly).
