"""
Shared chart styling, so every figure in the report reads as one system.

The categorical hues are a fixed, validated order - assigned by slot, never
cycled and never reassigned when the set of models on a chart changes. Colour
always follows the entity: LightGBM is blue on every figure it appears in.
"""
import matplotlib
matplotlib.use("Agg")           # no display on this machine
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2de"

# Fixed categorical order. Index = slot, not rank.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
          "#008300", "#6250d6", "#e34948"]

# Colour bound to the model, so a model keeps its hue across every figure.
MODEL_COLOURS = {
    "lightgbm":            SERIES[0],
    "random_forest":       SERIES[1],
    "mlp":                 SERIES[2],
    "knn":                 SERIES[3],
    "logistic_regression": SERIES[4],
}

MODEL_LABELS = {
    "lightgbm": "LightGBM",
    "random_forest": "Random Forest",
    "mlp": "MLP",
    "knn": "KNN",
    "logistic_regression": "Logistic Reg.",
}


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.titlesize": 12,
        "axes.titleweight": "600",
        "axes.labelsize": 10,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.size": 10,
        "savefig.facecolor": SURFACE,
        "savefig.bbox": "tight",
        "savefig.dpi": 150,
    })


def strip_spines(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        if side not in keep:
            ax.spines[side].set_visible(False)
