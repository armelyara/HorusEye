#!/usr/bin/env python3
"""
HorusEye — Q1 (scene understanding, level 1): calibration analysis.

Reproduces, from the raw per-prediction records, the two calibration figures
and the headline metrics used in the Q1 findings:

  Figure 1  q1_reliability_by_condition_EN.png  — reliability diagram per condition
  Figure 2  q1_signed_gap_EN.png                — signed calibration gap
                                                  (all plausible classes vs 'normal'-only)

Metrics printed: per-condition accuracy, F1 macro, ECE, AUROC, and the signed
calibration gap (global and matched on the 'normal' class).

No sklearn — ECE / AUROC / F1 are implemented from scratch for full reproducibility.

Input  : q1_all_records.json  (list of {condition, true_label, pred_label, confidence, correct})
Usage  : python q1_calibration.py [path/to/q1_all_records.json]
"""
import json
import sys
from collections import defaultdict

CONDS = ["clean", "fog", "smoke"]
COLORS = {"clean": "#2563eb", "fog": "#0891b2", "smoke": "#dc2626"}


# metrics
def accuracy(rows):
    return sum(r["correct"] for r in rows) / len(rows)


def mean_conf(rows):
    return sum(r["confidence"] for r in rows) / len(rows)


def signed_gap(rows):
    """Signed calibration gap = mean confidence - accuracy.
    >0 = over-confident, <0 = under-confident."""
    return mean_conf(rows) - accuracy(rows)


def f1_macro(rows):
    """Unweighted mean of per-class F1 (macro) over the classes with support
    (i.e. present as ground truth in this condition), matching q1_summary.json."""
    labels = sorted({r["true_label"] for r in rows})
    f1s = []
    for c in labels:
        tp = sum(r["true_label"] == c and r["pred_label"] == c for r in rows)
        fp = sum(r["true_label"] != c and r["pred_label"] == c for r in rows)
        fn = sum(r["true_label"] == c and r["pred_label"] != c for r in rows)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def ece(rows):
    """Expected Calibration Error: weighted mean over confidence bins of
    |confidence - accuracy|. Confidence here is discrete (self-reported)."""
    n = len(rows)
    bins = defaultdict(list)
    for r in rows:
        bins[round(r["confidence"], 3)].append(r["correct"])
    return sum(len(v) / n * abs(c - sum(v) / len(v)) for c, v in bins.items())


def auroc(rows):
    """AUROC using confidence as the score to separate correct (positive) from
    incorrect (negative) predictions. Probability that a random correct
    prediction is scored higher than a random incorrect one (ties = 0.5)."""
    pos = [r["confidence"] for r in rows if r["correct"] == 1]
    neg = [r["confidence"] for r in rows if r["correct"] == 0]
    if not pos or not neg:
        return float("nan")
    wins = ties = 0
    for p in pos:
        for q in neg:
            if p > q:
                wins += 1
            elif p == q:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def reliability_points(rows, min_n=3):
    """(stated confidence, empirical accuracy) per confidence bin with >= min_n samples."""
    bins = defaultdict(list)
    for r in rows:
        bins[round(r["confidence"], 2)].append(r["correct"])
    xs = sorted(c for c in bins if len(bins[c]) >= min_n)
    return xs, [sum(bins[c]) / len(bins[c]) for c in xs]


# figures
def make_figures(records):
    import matplotlib.pyplot as plt
    import numpy as np

    def rows(cond, cls=None):
        return [r for r in records if r["condition"] == cond
                and (cls is None or r["true_label"] == cls)]

    # Figure 1 — reliability by condition
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.plot([0, 1], [0, 1], "--", color="#9aa5b1", lw=1.6, label="Perfect calibration")
    for c in CONDS:
        xs, ys = reliability_points(rows(c))
        ax.plot(xs, ys, "o-", color=COLORS[c], lw=2.4, ms=8,
                label=f"{c}  (acc={accuracy(rows(c)):.2f})")
    ax.set_xlim(0.65, 1.02); ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Stated confidence"); ax.set_ylabel("Empirical accuracy")
    ax.set_title("Q1 — Reliability by condition (zoom 0.65-1.0)", fontweight="bold")
    ax.legend(loc="upper left"); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig("q1_reliability_by_condition_EN.png", dpi=200,
                                    bbox_inches="tight"); plt.close(fig)

    # Figure 2 — signed gap: all plausible classes vs 'normal'-only
    g_all = [signed_gap(rows(c)) for c in CONDS]
    g_norm = [signed_gap(rows(c, "normal")) for c in CONDS]
    x = np.arange(3); w = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    b1 = ax.bar(x - w / 2, g_all, w, color="#9fb3c8", label="All plausible classes", zorder=3)
    b2 = ax.bar(x + w / 2, g_norm, w, color="#e4572e", label="'normal' class only (matched)", zorder=3)
    ax.axhline(0, color="#333", lw=1.0, zorder=4)
    ax.set_xticks(x); ax.set_xticklabels(CONDS); ax.set_ylim(-0.065, 0.205)
    ax.set_ylabel("Signed gap  (confidence - accuracy)")
    ax.set_title("Q1 — Signed calibration gap: veil flips the sign", fontweight="bold")
    ax.grid(axis="y", alpha=0.25, zorder=0); ax.legend(loc="upper left")
    for bars, vals in ((b1, g_all), (b2, g_norm)):
        for b, v in zip(bars, vals):
            va, off = ("bottom", 0.005) if v >= 0 else ("top", -0.005)
            ax.text(b.get_x() + b.get_width() / 2, v + off, f"{v:+.3f}",
                    ha="center", va=va, fontsize=10, fontweight="bold", color="#333")
    fig.tight_layout(); fig.savefig("q1_signed_gap_EN.png", dpi=200,
                                    bbox_inches="tight"); plt.close(fig)


# main
def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "q1_all_records.json"
    records = json.load(open(path))

    print(f"{'cond':6} {'n':>4} {'acc':>6} {'F1':>6} {'ECE':>6} {'AUROC':>7} {'gap(all)':>9} {'gap(normal)':>12}")
    for cond in CONDS:
        rows = [r for r in records if r["condition"] == cond]
        norm = [r for r in rows if r["true_label"] == "normal"]
        print(f"{cond:6} {len(rows):>4} {accuracy(rows):>6.3f} {f1_macro(rows):>6.3f} "
              f"{ece(rows):>6.3f} {auroc(rows):>7.3f} {signed_gap(rows):>+9.3f} "
              f"{signed_gap(norm):>+12.3f}")
    g = records
    print(f"\nGLOBAL n={len(g)}  ECE={ece(g):.3f}  AUROC={auroc(g):.3f}")

    try:
        make_figures(records)
        print("Figures written: q1_reliability_by_condition_EN.png, q1_signed_gap_EN.png")
    except ImportError:
        print("(matplotlib/numpy not installed — metrics only; skipping figures)")


if __name__ == "__main__":
    main()