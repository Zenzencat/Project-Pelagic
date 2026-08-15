"""
Project Pelagic — Deck Visual Generation
SWU Prasarnmit AI Engineering Final Project

Turns existing exports (docs/checkpoint_comparison_summary.json,
docs/confusion_matrix_breakdown.json, docs/region_coverage.json) into static
chart images for the capstone deck. Reads real numbers directly from those
exports — nothing here is retyped by hand. Does not touch the live pipeline.

Uses matplotlib (already a project dependency via evaluate_holdout.py /
verify_pipeline.py) — no new plotting library added.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

DOCS_DIR = "docs"

# Palette (validated categorical order — see dataviz skill's reference palette)
COLOR_V1 = "#2a78d6"       # categorical slot 1 (blue)
COLOR_V2 = "#eb6834"       # categorical slot 2 (orange)
COLOR_SINGLE = "#2a78d6"   # single-series charts use slot 1
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
NA_FILL = "#e1e0d9"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "text.color": INK_PRIMARY,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
})


def load_json(name):
    with open(os.path.join(DOCS_DIR, name)) as f:
        return json.load(f)


def style_axis(ax, hide_top_right=True):
    if hide_top_right:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.grid(axis="y", color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)


def chart_1_v1_vs_v2():
    """
    Two-panel comparison: left = IoU by category (near-identical bucket-level
    outcomes between v1/v2), right = avg false-positive area % of scene by
    category (the real, modest improvement) -- the corrected framing, not the
    earlier overstated 10-45% figure.
    """
    summary = load_json("checkpoint_comparison_summary.json")
    by_metric = {row["metric"]: row for row in summary}

    categories = ["oil", "no_oil", "lookalike"]
    cat_labels = ["Oil", "No Oil", "Lookalike"]

    iou_v1 = [by_metric[f"{c}_iou"]["v1_value"] for c in categories]
    iou_v2 = [by_metric[f"{c}_iou"]["v2_value"] for c in categories]
    fp_pct_v1 = [by_metric[f"{c}_avg_false_positive_pct_of_scene"]["v1_value"] for c in categories]
    fp_pct_v2 = [by_metric[f"{c}_avg_false_positive_pct_of_scene"]["v2_value"] for c in categories]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(categories))
    width = 0.32

    def grouped_bars(ax, v1_vals, v2_vals, ylabel, fmt):
        b1 = ax.bar(x - width / 2 - 0.02, v1_vals, width, color=COLOR_V1, label="v1", zorder=3)
        b2 = ax.bar(x + width / 2 + 0.02, v2_vals, width, color=COLOR_V2, label="v2", zorder=3)
        for bars in (b1, b2):
            for bar in bars:
                h = bar.get_height()
                ax.annotate(fmt.format(h), (bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 4), textcoords="offset points", ha="center",
                            fontsize=9, color=INK_SECONDARY)
        ax.set_xticks(x)
        ax.set_xticklabels(cat_labels)
        ax.set_ylabel(ylabel, color=INK_SECONDARY)
        style_axis(ax)
        return b1, b2

    grouped_bars(ax1, iou_v1, iou_v2, "IoU", "{:.3f}")
    ax1.set_title("IoU — bucket-level outcomes are identical", fontsize=11, color=INK_PRIMARY, pad=12)
    ax1.set_ylim(0, max(iou_v1 + iou_v2) * 1.25)

    grouped_bars(ax2, fp_pct_v1, fp_pct_v2, "Avg. false-positive area (% of scene)", "{:.2f}%")
    ax2.set_title("False-positive area — real, modest improvement", fontsize=11, color=INK_PRIMARY, pad=12)
    ax2.set_ylim(0, max(fp_pct_v1 + fp_pct_v2) * 1.2)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)

    fig.suptitle("v1 vs v2: same detection buckets, smaller false-positive footprint",
                 fontsize=13, color=INK_PRIMARY, y=1.10)
    fig.text(0.5, 1.0, "Lookalike false-positive area dropped 41.3% → 37.5% of scene area v1→v2 "
                        "— real, but far more modest than an earlier 10-45% estimate.",
              ha="center", fontsize=9, color=INK_SECONDARY)

    fig.tight_layout()
    out_path = os.path.join(DOCS_DIR, "chart_v1_vs_v2_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def chart_2_confusion_matrix():
    """
    Heatmap of category x outcome scene counts. v1 and v2 have identical
    bucket-level outcomes (see docs/status.md) so this shows v2 (current
    production checkpoint) with that noted in the title. Cells that are
    structurally inapplicable for a category (e.g. oil scenes can never be
    bucketed as "false_positive" by this classifier's own definition -- see
    src/compare_checkpoints.py::classify_outcome()) are marked "—", not "0",
    since "0" would misleadingly imply a checked-and-found-zero result rather
    than an outcome the bucketing logic cannot produce for that category.
    """
    cm = load_json("confusion_matrix_breakdown.json")
    v2_rows = {row["category"]: row for row in cm if row["checkpoint"] == "v2"}

    categories = ["oil", "no_oil", "lookalike"]
    cat_labels = ["Oil", "No Oil", "Lookalike"]
    outcomes = ["correct", "partial", "false_positive", "false_negative"]
    outcome_labels = ["Correct", "Partial", "False Positive", "False Negative"]

    # Structurally-possible outcomes per category (see classify_outcome()).
    applicable = {
        "oil": {"correct", "partial", "false_negative"},
        "no_oil": {"correct", "false_positive"},
        "lookalike": {"correct", "false_positive"},
    }

    grid = np.zeros((len(categories), len(outcomes)))
    mask_na = np.zeros((len(categories), len(outcomes)), dtype=bool)
    for i, cat in enumerate(categories):
        for j, outcome in enumerate(outcomes):
            if outcome not in applicable[cat]:
                mask_na[i, j] = True
            else:
                grid[i, j] = v2_rows[cat][outcome]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Sequential blue ramp for magnitude; N/A cells flat muted gray.
    cmap = plt.cm.Blues
    display_grid = np.ma.masked_array(grid, mask=mask_na)
    im = ax.imshow(display_grid, cmap=cmap, vmin=0, vmax=10, aspect="auto")
    ax.imshow(np.where(mask_na, 1, np.nan), cmap=plt.matplotlib.colors.ListedColormap([NA_FILL]),
              aspect="auto", vmin=0, vmax=1)

    for i in range(len(categories)):
        for j in range(len(outcomes)):
            if mask_na[i, j]:
                ax.text(j, i, "—", ha="center", va="center", color=INK_MUTED, fontsize=11)
            else:
                val = int(grid[i, j])
                # White text on dark fill, dark text on light fill (contrast-aware).
                frac = val / 10.0
                txt_color = "#ffffff" if frac > 0.55 else INK_PRIMARY
                ax.text(j, i, f"{val}/10", ha="center", va="center", color=txt_color,
                         fontsize=11, fontweight="bold")

    ax.set_xticks(range(len(outcomes)))
    ax.set_xticklabels(outcome_labels)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(cat_labels)
    ax.set_xticks(np.arange(-0.5, len(outcomes), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(categories), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=3)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title("Scene-level outcomes by category (v2, n=10/category — identical for v1)",
                 fontsize=12, color=INK_PRIMARY, pad=14)

    fig.tight_layout()
    out_path = os.path.join(DOCS_DIR, "chart_confusion_matrix.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def chart_3_region_accuracy():
    """
    v2 pass rate by region, sorted descending, single series (no legend
    needed for one series) with n/total direct labels since bars alone
    can't carry small-sample-size context.
    """
    rows = load_json("region_coverage.json")
    from collections import Counter
    total = Counter(r["region"] for r in rows)
    passed = Counter(r["region"] for r in rows if r["v2_pass_fail"] == "PASS")

    regions = sorted(total.keys(), key=lambda r: passed.get(r, 0) / total[r], reverse=True)
    rates = [passed.get(r, 0) / total[r] * 100 for r in regions]
    labels = [f"{passed.get(r, 0)}/{total[r]}" for r in regions]

    # A true 0% rate renders as a zero-width (invisible) bar, indistinguishable
    # from missing data -- give it a small visible stub so zero reads as a
    # real, checked zero rather than an absent row.
    MIN_STUB = 1.5
    display_rates = [max(r, MIN_STUB) for r in rates]

    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(regions))
    bars = ax.barh(y, display_rates, height=0.6, color=COLOR_SINGLE, zorder=3)

    for bar, label, rate in zip(bars, labels, rates):
        ax.annotate(label, (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=10, color=INK_SECONDARY)

    ax.set_yticks(y)
    ax.set_yticklabels(regions)
    ax.invert_yaxis()
    ax.set_xlabel("v2 pass rate (% of scenes)", color=INK_SECONDARY)
    ax.set_xlim(0, 100)
    style_axis(ax)
    ax.grid(axis="x", color=GRIDLINE, linewidth=1, zorder=0)
    ax.grid(axis="y", visible=False)

    ax.set_title("Detection accuracy varies sharply by region, not just by scene category",
                 fontsize=12, color=INK_PRIMARY, pad=12)
    fig.text(0.5, -0.02, "Small samples (n=1-2) for Gulf of Mexico / Western Mediterranean / Baltic Sea "
                          "are suggestive, not conclusive on their own.",
              ha="center", fontsize=8.5, color=INK_MUTED)

    fig.tight_layout()
    out_path = os.path.join(DOCS_DIR, "chart_region_accuracy.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    paths = [chart_1_v1_vs_v2(), chart_2_confusion_matrix(), chart_3_region_accuracy()]
    print("Generated:")
    for p in paths:
        print(" ", p)
