# Viva prep — 50 examiner Q&A with THIS project's v2 numbers

> Numbers below are from `results/` (master_regression.csv, master_classification.csv,
> before_after.csv, ablations.csv, summaries, conformal txt). Cheat-sheet and pitch first.

## Numbers to remember
- Deployed model everywhere: HGB (CV-selected). Capped test RMSE: FD001 12.35 (CV 11.61±0.43),
  FD002 12.87 (CV 13.19), FD003 13.17 (CV 9.85), FD004 13.17 (CV 13.06). Uncapped: 13.46 / 25.44 / 14.55 / 25.46
  (FD002/FD004 uncapped fail <20 by construction: 57/259 and 67/248 test engines have true RUL>125).
- A1 effect (before→after, capped): FD002 HGB 18.27→12.87 (PHM 1637→737); FD004 HGB 23.87→13.17 (PHM 7140→857).
  FD001 at fixed 100-iter capacity: 12.37→12.04 (fix neutral-to-positive; the 11.63→12.35 headline mixes the 100→300 estimator change).
- 3-class direct-HGB critical recall: 0.769 (10/13) / 0.906 (48/53) / 1.000 (13/13) / 0.900 (36/40).
  Logreg-direct: 1.000 / 0.925 / 1.000 / 0.900. Macro-F1 HGB: 0.853 / 0.928 / 0.918 / 0.923.
- Binary ≤30 direct-HGB recall: 1.00 / 1.00 / 1.00 / 0.943 (50/53). Tuned-t policy recall: 1.00 everywhere
  (t = 40/40/34/48; precisions 0.926/0.803/0.909/0.716).
- Warning lead medians: 51/53/50/50 (IQRs in summaries); crit medians 21/19/19/19. Premature warn 0.10–0.30,
  premature crit 0–0.05. FAR/100: 0.67/0.96/0.28/0.58. Any-time crit detection: 10/13, 51/53, 11/13, 38/40.
- Mondrian conformal (last cycle): FD001 0.870±0.034 / 0.890±0.031; FD002 0.907±0.018 / 0.938±0.015;
  FD003 0.980/0.980 (half-widths 86/123 — uselessly wide, stated); FD004 0.891±0.020 / 0.923±0.017.
- Ablations (FD002, capped RMSE): legacy smooth-first 18.11 vs fixed 12.75; condnorm off 18.90 vs on 12.75.
  FD001: no-cycle 13.25 vs 12.04 baseline; no-ops identical 12.04; class-weight balanced 0.841 vs None 0.914
  (weighting hurts — reported); regress-then-threshold 0.903 vs direct 0.841. Cap 100 too aggressive;
  125 ≈ 130 on common-125 target. Deep windows: 30 best; 50 overfits (FD001 LSTM holdout 10.05 → test 15.72).
- Dropped sensors: FD001 [s1,s5,s10,s16,s18,s19]; FD002/FD004 [s1,s5,s18,s19] (+s10/s16 kept, near-constant).
- Scorecard: C1 PASS; C2 3/4 (FD001 hgb 0.769 narrow FAIL, CI [0.50,1.00]); C3 PASS via tuned policy
  (direct 3/4, FD004 0.943); C4 PASS. Runtime: 17.5/37.5/24/51.5 min per subset (CPU).

## 60-second pitch
"We predict turbofan remaining life from sensor histories and turn it into green-amber-red alerts. Our HGB model,
selected by grouped cross-validation — never by the test set — reaches capped RMSE 12–13 cycles on all four
C-MAPSS subsets, and the frozen alert policy catches every fails-within-30 engine with about fifty cycles of
warning. The headline story is a bug we found by reading feature importances — smoothing before normalisation
blended operating modes — whose fix gained eleven RMSE points on FD004 and is published as an ablation.
Everything reproduces with one command."

## Q&A (40 updated + 10 new)
1. Why a piece-wise RUL cap (125)? Early-life sensors are flat (trajectories vs RUL); fitting raw RUL≈200 fits
   noise. Cap-100/125/130 ablation on the common-125 target: 100 is worst (FD001 test 14.88 vs 11.96 at 125);
   125 ≈ 130. Thresholds (50/20/30) sit below the cap, so labels are undistorted.
2. Why GroupKFold (5-fold), not random split? Rows from one engine are correlated; random splits leak engine
   identity. All selection uses grouped-CV mean RMSE; tests assert zero fold overlap.
3. Why recall over accuracy? Safe rows dominate (test_class_counts.csv); always-safe scores well on accuracy
   yet kills. Critical/positive recall measures missed failures — the 10:1 error.
4. What does the PHM score penalise? s=Σ exp(−d/13)−1 (d<0, early) else exp(d/10)−1 (d>0, late); late grows
   faster. Unit-tested against hand values (pytest).
5. Why condition normalisation? Six modes shift every sensor; global scaling confuses mode with damage.
   KMeans k=6 on standardised ops (elbow+silhouette), within-cluster z-score. Off/on ablation: 18.90 vs 12.75.
6. Where could leakage creep in? Scalers/KMeans/selectors on full data; future-using smoothing; test-tuned
   thresholds; cycle/max_cycle features (removed — regression-tested); reusing test for model choice (now
   CV-only). Two real bugs were caught this way.
7. LSTM vs boosting — which won? Boosting on every subset (HGB capped 12.35–13.17 vs LSTM 13.2–14.9):
   20–60k rows of strong tabular features favour trees; window-50 LSTM overfits (holdout 10.05 → test 15.72).
8. Regress-then-threshold vs direct? FD001 macro-F1 0.903 vs 0.841 (RUL ordering regularises); FD002 close
   (0.944 vs 0.941 crit recall). Both reported; OOF-CV for both.
9. Imbalance handling? class_weight balanced everywhere, ablated off: it HURTS test macro-F1 (0.841 vs 0.914
   FD001) — reported as a negative result. Judged by macro-F1/balanced-acc/MCC, never accuracy.
10. Explain your confusion matrix. Rows=true, cols=pred with name labels; counts + row-normalised versions.
    FD001 hgb: 10/13 critical caught; misses are adjacent-class (warning), listed per subset.
11. Precision vs recall in maintenance terms? Recall = failures caught; precision = alarms that were real.
    Tuned policy: recall 1.00 at precision 0.72–0.93 — safe with alarm-fatigue cost stated.
12. Alert lead time? First warning/critical cycle → true cycles remaining. Medians ≈50/≈19; IQRs, premature
    fractions (warn raised while true RUL>60: 10–30%), FAR/100 in the table.
13. False-alarm rate? Predicted alert while true RUL>50, per 100 cycles: 0.67/0.96/0.28/0.58.
14. Cost matrix? Assumed 10:1 (missed:false alarm), expected cost = 10·FN+1·FP. Sensitivity table for
    5/10/20/50 in results/*_cost_sensitivity.csv. Assumed, not airline data.
15. Sensor drift? Drift mimics degradation → false alarms; mitigated by condition normalisation + residual
    monitoring + recalibration; listed limitation, drift detector is future work.
16. Why windows [10,30] and sequences of 30? Cover fast/slow dynamics; [10] alone costs +3.7 RMSE (FD001);
    adding 5 adds nothing; deep window 30 beats 20/50. Short test engines edge-padded with first row (stated).
17. Why α=0.3 smoothing? Suppresses noise, preserves trends (raw-vs-smoothed figure). Order matters more
    than α: norm-first vs smooth-first is +5.4 RMSE on FD002.
18. Which sensors matter physically? Grouped importance: s11/s4/s7 families + slopes (HPC/fan gas path);
    setpoints s1/s5/s18/s19 correctly drop. SHAP beeswarms agree on direction.
19. Cap-induced bias? Large RULs under-predicted by design; harmless below 50 where maintenance acts;
    per-bin table shows near-failure errors 3–5× smaller.
20. Fault-mode confusion FD003/FD004? No evidence of systematic fan/HPC confusion: FD003 HGB matches FD001
    (13.17 vs 12.35); FD004 fixed by normalisation. Old “confusion” claim deleted for lack of evidence.
21. Simulation-vs-real gap? No sensor faults, repairs, censoring, or drift; needs domain adaptation + drift
    monitoring; small test sets → wide CIs, all shown.
22. Uncertainty? Mondrian split-conformal on deployed model, OOF residuals at truncation points, bins on
    predicted RUL. FD002 near-nominal; FD003 over-covers but uselessly wide; FD001/FD004 slight under-cover —
    all stated with SEs.
23. Deploy how? app.py: engine history → RUL + green/amber/red + trajectory (order fixed to norm-first);
    production adds feature store, registry, human sign-off on reds.
24. Human-in-the-loop? Reds advise, engineers decide; grounding never autonomous; audit log + alarm budget.
25. Overfitting checks? 5-fold CV ± std vs test; multi-seed std; holdout-vs-test gap for deep (window-50
    caught overfitting); early stopping everywhere.
26. Why StandardScaler after rolling features? Rolling means/stds/slopes live on different scales; stabilises
    Ridge/MLP/torch (trees unaffected; kept uniform).
27. MCC/κ besides F1? Chance-corrected whole-matrix summaries under imbalance; in every tidy row.
28. ROC vs PR? ROC flatters with many safe negatives; PR focuses on rare positives — both shown, PR emphasised.
29. What is censoring here? Test trajectories truncated pre-failure (right-censored); train run-to-failure.
    All-cycles true RUL derived arithmetically — valid only with no repairs (stated).
30. Why not accuracy for selection? See 3/9; selection uses CV RMSE (regression) and CV macro-F1 (classifiers).
31. Reproducibility? Seeds 42–44, pinned requirements, --fresh/--smoke/--only flags, saved models/results/
    figures, 12 pytest tests, 5 executed notebooks, wall-clock logged.
32. Runtime? 17.5/37.5/24/51.5 min per subset CPU (5-fold CV + 3-seed RF/MLP/deep + tuning); ablations extra.
33. With 10× data? Deeper sequence models/transformers, per-condition heads, quantile outputs.
34. With 10× less data? Regularise harder, simpler windows, boosting + transfer from FD001.
35. Ethics? False negatives risk lives; false positives erode trust; simulated-data claims must not transfer
    uncritically; decisions stay human; costs assumed.
36. Why not fixed-interval maintenance? Wastes healthy life, still misses early failures; condition-based
    acts on actual degradation (≈50-cycle leads here).
37. Are test labels fair? Official RUL files, asserted == n units, never used for selection/thresholding;
    last-cycle protocol matches the PHM08 competition.
38. Adjacent vs dangerous errors? warning↔critical still triggers action; safe↔critical misses do not —
    counted separately; FD001 hgb misses are adjacent-class.
39. Missing op settings? Cluster assignment fails → fall back to global normalisation; demo validates inputs.
40. One-line takeaway? Causal features + normalise-first + boosted trees + OOF-tuned thresholds give accurate,
    explainable, full-recall alerts that generalise — reproduced from code.

## 10 NEW questions (v2-specific)
41. Why normalise before smoothing? Consecutive cycles can sit in different conditions; smoothing raw blends
    modes and no later z-score can unblend them. Cost measured: +5.4 RMSE on FD002; importance flipped from
    proxies (s1/s5) to physics (s11/s4).
42. Capped vs uncapped RMSE — which is the headline and why? Capped (literature convention; matches the
    training target; uncapped has an ≈18-point floor on FD002/FD004 since 57/259 and 67/248 test engines exceed
    RUL 125 — a perfect capped model still “fails” uncapped). Both reported; capped primary.
43. Why select models by CV, not test? Test-based selection overfits the test set and invalidates it; v1 did
    this (best by test RMSE). v2 selects by 5-fold grouped-CV capped RMSE for every model incl. tuned-HGB
    (backfilled CV), MLP and deep holdouts — test is reported for all, used for none.
44. Why were s1/s5 suspicious? They are flight-condition setpoints, constant within each condition — they
    cannot carry degradation signal, yet topped v1 importance: the signature of condition leakage through
    smooth-first. After the fix they normalise to exactly 0 (float-noise floor 1e-9 relative) and drop.
45. How does threshold tuning work, and when is 0.95 infeasible? OOF RUL preds of the deployed model at 3
    random truncations per train engine; sweep t=10..60; min cost s.t. recall≥0.95, else min cost with
    target_met=false. Feasible on all four here (t=40/40/34/48); FD004-style shortfalls would fall back
    transparently with the cost-optimal recall stated.
46. Why bootstrap CIs with ~100 engines? 13 critical engines on FD001: one engine = 8 recall points
    (0.769 CI [0.50,1.00]). Resampling engines (1000×) propagates exactly this granularity; point estimates
    alone would overclaim.
47. Why did conformal under-cover, and what does Mondrian change? v1 calibrated on all train rows (easy
    early-life dominated) with a non-deployed model → 0.86. v2 calibrates the deployed model at truncation
    points with per-predicted-RUL-bin quantiles, so near-failure intervals are narrow and far ones wide;
    residual FD001/FD004 under-cover (~0.87–0.89 vs 0.90) is calibration/test shift, stated with SEs.
48. What does the PHM score penalise? Late (over-) predictions exponentially more than early ones of equal
    size (exp(d/10)−1 vs exp(−d/13)−1) — a missed failure costs more than an early inspection. Hand-verified
    in pytest; reported on capped targets.
49. Is `cycle` a feature — isn't that leakage? No: elapsed cycles are observable age, legitimate for
    degradation modelling (older engines are likelier to fail). Ablated: dropping it costs +1.2 RMSE, so it
    earns its place and is disclosed, unlike cycle/max_cycle (failure time — removed).
50. Limits of simulated data, concretely? No sensor faults/drift/repairs/censoring; small test sets
    (wide CIs); cap biases far-RUL; FD003 conformal widths useless; deep < boosting here. Each has a named
    future item (adaptation, quantile/adaptive conformal, drift monitors, human-in-the-loop).
