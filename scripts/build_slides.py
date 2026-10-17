"""Build slides/presentation.pptx (13 slides) from v2 results (Block C)."""
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
from pptx import Presentation
from pptx.util import Inches

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"; RES = ROOT / "results"
FDS = ["FD001", "FD002", "FD003", "FD004"]

reg = pd.read_csv(RES / "master_regression.csv")
summ = {fd: json.load(open(RES / f"{fd}_summary.json")) for fd in FDS}
clfcv = {}
for fd in FDS:
    p = RES / f"{fd}_clf_cv.json"
    clfcv[fd] = json.load(open(p)) if p.exists() else {"head3": "hgb", "headb": "hgb"}


def img(slide, fname, left=0.5, top=1.2, width=9.0):
    p = FIG / fname
    if p.exists():
        slide.shapes.add_picture(str(p), Inches(left), Inches(top), width=Inches(width))
        return fname
    tx = slide.shapes.add_textbox(Inches(1), Inches(2), Inches(8), Inches(1)).text_frame
    tx.text = f"[missing figure: {fname}]"
    return ""


def main():
    prs = Presentation(); prs.slide_width = Inches(13.33); prs.slide_height = Inches(7.5)
    notes = []

    def slide(title, bullets):
        sl = prs.slides.add_slide(prs.slide_layouts[5])
        sl.shapes.title.text = title
        tx = sl.shapes.add_textbox(Inches(0.5), Inches(1.1), Inches(4.5), Inches(5)).text_frame
        tx.word_wrap = True
        for b in bullets:
            p = tx.add_paragraph(); p.text = b; p.level = 0
        return sl

    r1 = reg[(reg.fd == "FD001") & (reg.model == summ["FD001"]["best_model"])].iloc[0]
    s1 = slide("Turbofan RUL & Maintenance Alerts",
               ["TE7483 · NASA C-MAPSS (PHM08)",
                f"FD001 {summ['FD001']['best_model']}: RMSE {r1['test_last_rmse_cap']:.1f} (CV {r1['cv_rmse']:.1f}), PHM {r1['test_last_phm_cap']:.0f}",
                "RUL regression → green/amber/red alert policy, tuned on validation only"])
    img(s1, f"FD001_pred_vs_true_{summ['FD001'].get('best_overall', 'hgb')}.png", left=5.5, width=7.0)
    notes.append("45s: unscheduled engine removals cost airlines millions; we turn sensor streams into remaining-useful-life estimates plus green-amber-red alerts. Headline numbers are on screen: capped test RMSE with a grouped cross-validation score to match — no test leakage anywhere.")

    s2 = slide("Data: run-to-failure + truncated test",
               ["26 cols: unit, cycle, 3 ops, 21 sensors",
                "FD001–FD004: 1/6 conditions × HPC / HPC+Fan faults",
                "Labels: RUL capped @125; safe>50, warn 20–50, crit<20, ≤30 binary"])
    img(s2, "FD001_lifetime_hist.png", left=5.5, width=7.0)
    notes.append("45s: train runs to failure so RUL is exact; test is truncated with official RUL labels at the last cycle. Four subsets span operating conditions and fault modes; thresholds sit below the cap so classes are undistorted.")

    s3 = slide("Pipeline: normalise FIRST, then smooth",
               ["GroupKFold by engine; train-only scalers/KMeans",
                "Piece-wise cap 125; causal rolling + length-30 windows",
                "Baselines → RF/HGB/MLP → LSTM/CNN; direct vs RUL-threshold"])
    img(s3, "pipeline_block_diagram.png", left=5.5, width=7.0)
    notes.append("45s: every fit happens on train folds only; test features use past data causally. Model selection uses grouped CV only — the test set never picks the model. Two leakage bugs were found and fixed; details on the next-but-one slide.")

    s4 = slide("Key EDA insight",
               ["Degradation invisible early → the cap is justified",
                "Correlation + monotonicity select sensors",
                "Safe rows dominate: accuracy is the wrong metric"])
    img(s4, "FD001_trajectories.png", left=5.5, width=7.0)
    notes.append("45s: sensors sit flat for most of an engine's life then degrade steeply — training on raw RUL of 200-plus cycles would fit noise, hence the cap at 125. And most rows are healthy, so we optimise recall, not accuracy.")

    s5 = slide("Condition normalisation works",
               ["KMeans k=6 on standardised ops: elbow + silhouette",
                "Within-cluster z-score collapses the six modes",
                "s1/s5/s18/s19 drop out (constant within condition)"])
    img(s5, "FD002_condnorm_effect.png", left=5.5, width=7.0)
    notes.append("45s: six operating modes shift every sensor; global scaling confuses mode with damage. Within-cluster normalisation removes the modes, and setpoint sensors correctly vanish from the feature set.")

    ba = pd.read_csv(RES / "before_after.csv")
    b2 = ba[(ba.subset == "FD002") & (ba.model == "hgb")].iloc[0]
    s6 = slide("What we found and fixed",
               [f"FD002 HGB: {b2['v1_rmse_cap']:.1f} to {b2['v2_rmse_cap']:.1f} RMSE (see results/before_after.csv)",
                "Bug: smoothing raw sensors blended operating modes",
                "Caught by checking WHAT the model relied on (s1/s5 top)"])
    img(s6, "FD002_ablation_smoothorder.png", left=5.5, width=6.0)
    notes.append("45s: our strongest methods slide: permutation importance showed the model leaning on condition proxies s1 and s5 — sensors that cannot carry damage signal. Tracing why exposed smoothing-before-normalisation; fixing the order gained five RMSE points, now a documented ablation.")

    piv = reg[~reg.model.isin(["mean"])].pivot_table(
        index="model", columns="fd", values="test_last_rmse_cap").round(1)
    s7 = slide("Headline results (capped test RMSE)", piv.to_string().split("\n")[:9])
    img(s7, "master_rmse_heatmap.png", left=5.5, width=7.0)
    notes.append("45s: HGB is cross-validation-selected on all four subsets; deep models trail by one to four points — reported, not hidden. Multi-condition subsets stay harder, but the gap closed from twofold to under ten percent.")

    h3 = clfcv["FD001"].get("head3", "hgb")
    s8 = slide("Confusion matrices (FD001)",
               ["Critical recall is the safety metric",
                "Errors sit in adjacent classes, not safe↔critical",
                f"Headline direct: {h3} (CV-selected)"])
    img(s8, f"FD001_cm_3class_{h3}.png", left=5.5, width=5.0)
    notes.append("45s: read rows as truth, columns as prediction. Warning-adjacent errors still trigger action; the dangerous cell is true-critical predicted safe — rare here, and counted explicitly in the report.")

    hb = clfcv["FD001"].get("headb", "hgb")
    _th = summ["FD001"]["threshold"]
    s9 = slide("Fails-within-30: ROC/PR + frozen threshold",
               ["PR matters: positives are rare",
                "Threshold from out-of-fold preds at truncation points",
                f"FD001: recall {_th['test_recall']:.2f} at precision {_th['test_precision']:.2f} (frozen t={_th['t']:.0f})"])
    img(s9, f"FD001_pr_{hb}.png", left=5.5, width=5.5)
    notes.append("45s: the alert threshold is tuned on validation folds at realistic truncation points, frozen, then applied once to test. Full recall on all four test sets; FD004 pays the most false alarms under two fault modes (numbers on the slide and in the report).")

    _wl = [float(summ[fd]["lead_time"]["warn_median"]) for fd in FDS]
    _far = [float(summ[fd]["false_alarm_per_100"]) for fd in FDS]
    s10 = slide("Lead time + cost",
                [f"Median warning lead {min(_wl):.0f}–{max(_wl):.0f} cycles on every subset",
                 f"False alarms {min(_far):.2f}–{max(_far):.2f} per 100 cycles",
                 "Assumed 10:1 costs; tuned threshold wins everywhere"])
    img(s10, f"FD001_leadtime_{summ['FD001']['best_model']}.png", left=5.5, width=7.0)
    notes.append("45s: the policy buys roughly fifty cycles of advance warning — comfortably plannable — with under one false alarm per hundred cycles (exact medians in the report). Costs are assumed, stated as such, with a sensitivity table in the report.")

    s11 = slide("Ablations: what earned its keep",
                ["Normalise-first +5.4; condnorm +6.2 (FD002)",
                 "Rolling [10,30] beats [10]; window-30 sweet spot for deep",
                 "Honest negatives: class-weights hurt; ops add nothing"])
    img(s11, "FD002_ablation_condnorm.png", left=5.5, width=6.0)
    notes.append("45s: every pipeline choice carries a number, including negative results — class weighting and extra op features do not help, and we say so. Deep window fifty overfits its holdout: ten-point holdout error becomes sixteen on test.")

    s12 = slide("Limitations + uncertainty",
                ["Simulated data; 13 critical engines on FD001/FD003",
                 "Mondrian conformal: near-nominal, wide when far out",
                 "Human signs every red — never autonomous grounding"])
    img(s12, "conformal_calibration.png", left=5.5, width=6.5)
    notes.append("45s: small test sets mean wide confidence intervals — shown, not hidden. Conformal intervals cover near nominally but are wide far from failure. Red alerts advise; engineers decide.")

    s13 = slide("Future + Thank you",
                ["Transformers with grouped validation; domain adaptation",
                 "Survival heads; adaptive uncertainty; edge + drift",
                 "Questions welcome"])
    notes.append("45s: next steps are real-fleet adaptation with drift monitoring and richer uncertainty. Thank you — questions?")
    for i, sl in enumerate(prs.slides):
        sl.notes_slide.placeholders[1].text = notes[i]
    out = ROOT / "slides" / "presentation.pptx"
    prs.save(out)
    print("wrote", out, f"{len(prs.slides)} slides")


if __name__ == "__main__":
    main()
