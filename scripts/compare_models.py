"""
Model comparison figures and the selection summary (report section 3).

Every figure is generated from results/model_comparison.csv and
results/cv_folds.csv - real runs, no hand-entered numbers. Models that do not
produce epoch-wise training histories get no learning curve, as the brief
requires.

Run:  python scripts/compare_models.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.viz.style import (                       # noqa: E402
    apply_style, strip_spines, MODEL_COLOURS, MODEL_LABELS, TEXT_SECONDARY)

FIG = ROOT / "results" / "figures"


def bar_with_labels(ax, names, values, errs=None, fmt="{:.3f}"):
    """Horizontal bars, direct-labelled. Rounded data-ends, 2px gaps."""
    y = np.arange(len(names))
    colours = [MODEL_COLOURS[n] for n in names]
    ax.barh(y, values, height=0.62, color=colours,
            xerr=errs, error_kw=dict(ecolor=TEXT_SECONDARY, lw=1.2, capsize=3))
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABELS[n] for n in names])
    ax.invert_yaxis()
    span = max(values) - min(0, min(values))
    for yi, v in zip(y, values):
        ax.text(v + span * 0.015, yi, fmt.format(v), va="center",
                fontsize=9, color=TEXT_SECONDARY)
    strip_spines(ax, keep=("bottom",))
    ax.grid(axis="y", visible=False)


def main():
    apply_style()
    FIG.mkdir(parents=True, exist_ok=True)
    comp = pd.read_csv(ROOT / "results" / "model_comparison.csv")
    cv = pd.read_csv(ROOT / "results" / "cv_folds.csv")

    # --- 1. cross-validated Macro F1, with fold spread ---------------------
    d = comp.sort_values("cv_macro_f1_mean", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    bar_with_labels(ax, d.model.tolist(), d.cv_macro_f1_mean.tolist(),
                    errs=d.cv_macro_f1_std.tolist())
    ax.set_xlabel("Cross-validated Macro F1 (5 trip-grouped folds, ± SD)")
    ax.set_title("Macro F1 across cross-validation folds")
    fig.savefig(FIG / "cv_macro_f1.png"); plt.close(fig)

    # --- 2. student-test Top-1 and Macro F1 --------------------------------
    d = comp.sort_values("test_top1", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    bar_with_labels(axes[0], d.model.tolist(), d.test_top1.tolist())
    axes[0].set_xlabel("Top-1 accuracy"); axes[0].set_title("Student-test Top-1 accuracy")
    axes[0].set_xlim(0, 1)
    d2 = comp.sort_values("test_macro_f1", ascending=False)
    bar_with_labels(axes[1], d2.model.tolist(), d2.test_macro_f1.tolist())
    axes[1].set_xlabel("Macro F1"); axes[1].set_title("Student-test Macro F1")
    axes[1].set_xlim(0, 1)
    fig.savefig(FIG / "student_test_scores.png"); plt.close(fig)

    # --- 3. log loss: lower is better, and this is our selection metric ----
    d = comp.sort_values("test_log_loss")
    fig, ax = plt.subplots(figsize=(7, 3.4))
    bar_with_labels(ax, d.model.tolist(), d.test_log_loss.tolist(), fmt="{:.4f}")
    ax.set_xlabel("Log loss on student-test trips (lower is better)")
    ax.set_title("Probability calibration — the HMM consumes these directly")
    fig.savefig(FIG / "log_loss.png"); plt.close(fig)

    # --- 4. the key diagnostic: folds vary more than models ----------------
    piv = cv.pivot_table(index="fold", columns="model", values="top1_accuracy")
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for m in piv.columns:
        ax.plot(piv.index, piv[m], marker="o", markersize=7, linewidth=2,
                color=MODEL_COLOURS[m], label=MODEL_LABELS[m])
    ax.set_xticks(piv.index)
    ax.set_xlabel("Cross-validation fold (each is a disjoint set of 16 trips)")
    ax.set_ylabel("Top-1 accuracy")
    ax.set_title("All models rise and fall together — the fold drives the score, not the model")
    # Legend below the axes: the series lines sweep through the lower-right
    # of the plot area, so an inset legend collides with them.
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    strip_spines(ax)
    fold_spread = piv.mean(axis=1).max() - piv.mean(axis=1).min()
    model_spread = piv.mean(axis=0).max() - piv.mean(axis=0).min()
    ax.annotate(f"spread across folds: {fold_spread:.3f}\n"
                f"spread across models: {model_spread:.3f}\n"
                f"ratio: {fold_spread/model_spread:.1f}x",
                xy=(0.02, 0.97), xycoords="axes fraction", va="top",
                fontsize=9, color=TEXT_SECONDARY)
    fig.savefig(FIG / "fold_variance.png"); plt.close(fig)

    # --- 5. accuracy vs calibration: why the winner wins -------------------
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for _, r in comp.iterrows():
        ax.scatter(r.test_log_loss, r.test_top1, s=170,
                   color=MODEL_COLOURS[r.model], zorder=3,
                   edgecolor="#fcfcfb", linewidth=2)
        ax.annotate(MODEL_LABELS[r.model], (r.test_log_loss, r.test_top1),
                    textcoords="offset points", xytext=(0, 14), ha="center",
                    fontsize=9, color=TEXT_SECONDARY)
    ax.set_xlabel("Log loss (lower is better) →  better calibrated")
    ax.set_ylabel("Top-1 accuracy")
    ax.set_title("Top-1 barely separates these models; calibration does")
    ax.invert_xaxis()
    strip_spines(ax)
    fig.savefig(FIG / "accuracy_vs_calibration.png"); plt.close(fig)

    # --- the comparison table the report requires --------------------------
    table = comp[[
        "model", "family", "config", "cv_macro_f1_mean", "cv_top1_mean",
        "cv_route_continuity_mean", "test_top1", "test_macro_f1",
        "test_log_loss", "test_route_continuity", "inference_us_per_row",
    ]].sort_values("test_log_loss")
    table.to_csv(ROOT / "results" / "comparison_table.csv", index=False)

    print(table.round(4).to_string(index=False))
    print(f"\nfold spread {fold_spread:.3f} vs model spread {model_spread:.3f} "
          f"= {fold_spread/model_spread:.1f}x")
    print(f"figures written to {FIG}")


if __name__ == "__main__":
    main()
