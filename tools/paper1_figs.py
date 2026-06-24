"""Render the script-generated Paper 1 figures for the ACPC paper.

Run:
    python -m tools.paper1_figs --out-dir assets/paper1_figs

Script-generated PNG filenames match the figure numbers used in the released
manuscript where applicable. The default render set contains only the
script-generated figures used for the Paper 1 release:

    fig2_sweep.png      — unperturbed / observation-noise 0.08 vs sigma_max

    fig5_scatter.png    — PushT n=9 LeWM scatter: fragility ratio
                          vs unperturbed / corruption-drop

    (fig3_pareto, fig4_radar, and fig6_mechanism were pruned from the paper
    after the figure-density audit. Their generator functions are kept below
    for reference but are no longer in the default render set.)

Figures are produced without an in-PNG ``Fig. N. ...'' suptitle so that the
LaTeX caption is the single source of truth and figure numbers cannot drift.

Data sources (no new computation needed):

- Eval tables (§4.2, §4.3): assets/paper1_data/canonical_evals_20260517.json
- Diagnostic tables / scatter predictor metrics:
  assets/paper1_data/canonical_diagnostics_20260517.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from matplotlib.transforms import Affine2D
import numpy as np


# ============================================================================
# Canonical diagnostic data mirrored from the paper release artifacts.
# ============================================================================

SWEEP_STDS = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
ROBUST_EVAL_METRIC = "pixels_std0.08"
ROBUST_EVAL_LABEL = r"Eval: observation noise $\sigma=0.08$ (unperturbed goal)"

# §4.4 Table 3 — 6 diagnostic metrics × {base, representative} × 4 tasks
# Metric order chosen so "compression" metrics group on one side of the radar.
DIAG_METRICS = [
    "clean_effective_rank",
    "clean_nn_cos_dist_median",
    "transition_resolution_ratio_l2",
    "transition_resolution_ratio_cos",
    "id_probe_r2",
    "action_mean_pred_shift_norm",
]
def _setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


_CANONICAL_EVALS_CACHE: Dict[str, Dict] = {}
_CANONICAL_TABLES_CACHE: Dict[str, Dict] = {}
_CANONICAL_DIAGNOSTICS_CACHE: Dict[str, Dict] = {}
_CANONICAL_DIAG_TABLES_CACHE: Dict[str, Dict] = {}


def _load_canonical_evals() -> Dict:
    """Load assets/paper1_data/canonical_evals_20260517.json (cached)."""
    if _CANONICAL_EVALS_CACHE:
        return _CANONICAL_EVALS_CACHE
    fp = (Path(__file__).resolve().parent.parent
          / "assets" / "paper1_data" / "canonical_evals_20260517.json")
    if not fp.exists():
        raise FileNotFoundError(
            f"Missing canonical eval aggregate: {fp}. "
            "This script requires canonical_evals_20260517.json."
        )
    with open(fp) as f:
        _CANONICAL_EVALS_CACHE.update(json.load(f))
    return _CANONICAL_EVALS_CACHE


def _canonical_eval_tables() -> Dict[str, Dict]:
    """Return sweep/base/representative-high tables derived directly from canonical eval JSON."""
    if _CANONICAL_TABLES_CACHE:
        return _CANONICAL_TABLES_CACHE

    canon = _load_canonical_evals()
    tasks = ["TwoRoom", "PushT", "Reacher", "Cube"]
    sweep: Dict[str, Dict[str, List[float]]] = {}
    base: Dict[str, Dict[str, float]] = {}
    clean_point_best: Dict[str, Dict[str, float]] = {}
    corrupted_point_best: Dict[str, Dict[str, float]] = {}

    for task in tasks:
        rows = {float(k): v for k, v in canon[task].items()}
        clean_vals, clean_stds = [], []
        px08_vals, px08_stds = [], []
        for std in SWEEP_STDS:
            entry = rows[std]
            metrics = entry["metrics"]
            clean_vals.append(float(metrics["clean"]["mean"]))
            clean_stds.append(float(metrics["clean"]["std"]))
            px08_vals.append(float(metrics[ROBUST_EVAL_METRIC]["mean"]))
            px08_stds.append(float(metrics[ROBUST_EVAL_METRIC]["std"]))

        sweep[task] = {
            "clean": clean_vals,
            "clean_std": clean_stds,
            "px08": px08_vals,
            "px08_std": px08_stds,
        }
        base[task] = {"clean": clean_vals[0], "px08": px08_vals[0]}

        clean_idx = int(np.argmax(clean_vals))
        px08_idx = int(np.argmax(px08_vals))
        clean_point_best[task] = {
            "std": SWEEP_STDS[clean_idx],
            "clean": clean_vals[clean_idx],
            "px08": px08_vals[clean_idx],
        }
        corrupted_point_best[task] = {
            "std": SWEEP_STDS[px08_idx],
            "clean": clean_vals[px08_idx],
            "px08": px08_vals[px08_idx],
        }

    _CANONICAL_TABLES_CACHE.update({
        "tasks": tasks,
        "sweep": sweep,
        "base": base,
        "clean_point_best": clean_point_best,
        "corrupted_point_best": corrupted_point_best,
    })
    return _CANONICAL_TABLES_CACHE


def _load_canonical_diagnostics() -> Dict:
    """Load assets/paper1_data/canonical_diagnostics_20260517.json (cached)."""
    if _CANONICAL_DIAGNOSTICS_CACHE:
        return _CANONICAL_DIAGNOSTICS_CACHE
    fp = (Path(__file__).resolve().parent.parent
          / "assets" / "paper1_data" / "canonical_diagnostics_20260517.json")
    if not fp.exists():
        raise FileNotFoundError(
            f"Missing canonical diagnostics aggregate: {fp}. "
            "This script requires canonical_diagnostics_20260517.json."
        )
    with open(fp) as f:
        _CANONICAL_DIAGNOSTICS_CACHE.update(json.load(f))
    return _CANONICAL_DIAGNOSTICS_CACHE


def _canonical_diag_tables() -> Dict[str, Dict]:
    """Return Table 3 representative diagnostics derived from canonical JSON."""
    if _CANONICAL_DIAG_TABLES_CACHE:
        return _CANONICAL_DIAG_TABLES_CACHE

    diag = _load_canonical_diagnostics()["table3_representative_diagnostics"]
    metric_order = diag["metric_order"]
    tasks = ["TwoRoom", "PushT", "Reacher", "Cube"]
    values = {}
    rep = {}
    for task in tasks:
        task_vals = diag["values"][task]
        values[task] = {
            "base": [float(task_vals["base"][m]) for m in metric_order],
            "representative": [float(task_vals["representative"][m]) for m in metric_order],
        }
        rep[task] = {"std": float(diag["representative_std_by_task"][task])}

    _CANONICAL_DIAG_TABLES_CACHE.update({
        "tasks": tasks,
        "values": values,
        "representative": rep,
    })
    return _CANONICAL_DIAG_TABLES_CACHE


# ============================================================================
# Figure 1 — Conceptual schematic: action-conditioned predictive consistency
# ============================================================================

def _box(ax, xy, w, h, fc, ec="#333333", lw=1.0, radius=0.018):
    patch = FancyBboxPatch(
        xy, w, h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=lw, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(patch)
    return patch


def _arrow(ax, start, end, color="#555555", lw=1.6, style="-|>"):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=13,
        linewidth=lw, color=color, shrinkA=0, shrinkB=0,
    ))


def _draw_tworoom_card(ax, x, y, w, h, mode, label):
    _box(ax, (x, y), w, h, "#FFFFFF", "#9AA5B1", 0.9, radius=0.012)
    if mode == "noise":
        rng = np.random.default_rng(7)
        pts_x = x + rng.uniform(0.02 * w, 0.98 * w, 420)
        pts_y = y + rng.uniform(0.10 * h, 0.90 * h, 420)
        cols = rng.choice(["#6AAED6", "#E56B6F", "#7FB069", "#AAAAAA"], 420)
        ax.scatter(pts_x, pts_y, s=1.0, c=cols, alpha=0.35, linewidths=0)
    elif mode == "lighting":
        ax.add_patch(Rectangle((x, y), w, h, facecolor="#E9F2FF", edgecolor="none", alpha=0.8))
        ax.add_patch(Rectangle((x, y), w, 0.45 * h, facecolor="#D7E7F7", edgecolor="none", alpha=0.8))
    else:
        ax.add_patch(Rectangle((x, y), w, h, facecolor="#F7F7F4", edgecolor="none", alpha=0.9))
    # Room walls and doorway.
    ax.plot([x + 0.08*w, x + 0.92*w], [y + 0.82*h, y + 0.82*h], color="#111111", lw=1.2)
    ax.plot([x + 0.08*w, x + 0.92*w], [y + 0.18*h, y + 0.18*h], color="#111111", lw=1.2)
    ax.plot([x + 0.08*w, x + 0.08*w], [y + 0.18*h, y + 0.82*h], color="#111111", lw=1.2)
    ax.plot([x + 0.92*w, x + 0.92*w], [y + 0.18*h, y + 0.82*h], color="#111111", lw=1.2)
    ax.plot([x + 0.50*w, x + 0.50*w], [y + 0.18*h, y + 0.43*h], color="#111111", lw=1.5)
    ax.plot([x + 0.50*w, x + 0.50*w], [y + 0.57*h, y + 0.82*h], color="#111111", lw=1.5)
    ax.add_patch(Circle((x + 0.35*w, y + 0.50*h), 0.055*h, fc="#D62828", ec="#8B0000", lw=0.8))
    ax.add_patch(Circle((x + 0.78*w, y + 0.50*h), 0.040*h, fc="#2A9D8F", ec="#106B5F", lw=0.8))
    ax.text(x + 0.5*w, y + h + 0.012, label, ha="center", va="bottom", fontsize=8.2, color="#204B84")


def _draw_pusht_card(ax, x, y, w, h, angle, contact_dx, label, color):
    _box(ax, (x, y), w, h, "#FFFFFF", "#9AA5B1", 0.9, radius=0.012)
    ax.add_patch(Rectangle((x + 0.08*w, y + 0.10*h), 0.84*w, 0.80*h,
                           facecolor="#F7FAFC", edgecolor="#DEE2E6", lw=0.5))
    # Goal slot.
    goal = Rectangle((x + 0.55*w, y + 0.53*h), 0.26*w, 0.12*h,
                     facecolor="#B7E4A8", edgecolor="#5E9D55", lw=0.6)
    goal.set_transform(Affine2D().rotate_deg_around(x + 0.68*w, y + 0.59*h, angle) + ax.transData)
    ax.add_patch(goal)
    # T-block: stem + bar.
    cx, cy = x + 0.48*w, y + 0.45*h
    stem = Rectangle((cx - 0.035*w, cy - 0.16*h), 0.07*w, 0.26*h,
                     facecolor="#9FB3C8", edgecolor="#4A5568", lw=0.7)
    bar = Rectangle((cx - 0.15*w, cy + 0.08*h), 0.30*w, 0.075*h,
                    facecolor="#CBD5E0", edgecolor="#4A5568", lw=0.7)
    tr = Affine2D().rotate_deg_around(cx, cy, angle) + ax.transData
    stem.set_transform(tr)
    bar.set_transform(tr)
    ax.add_patch(stem)
    ax.add_patch(bar)
    ax.add_patch(Circle((x + (0.50 + contact_dx)*w, y + 0.30*h), 0.055*h,
                        fc=color, ec="#2B2B2B", lw=0.6))
    ax.text(x + 0.5*w, y + h - 0.006, label, ha="center", va="top", fontsize=7.8, color=color)


def _draw_latent_axes(ax, origin, size, label_color="#4A5568"):
    ox, oy = origin
    sx, sy = size
    ax.plot([ox, ox + sx], [oy, oy], color="#111111", lw=1.1)
    ax.plot([ox, ox], [oy, oy + sy], color="#111111", lw=1.1)
    _arrow(ax, (ox + sx, oy), (ox + sx + 0.02, oy), "#111111", 1.0)
    _arrow(ax, (ox, oy + sy), (ox, oy + sy + 0.02), "#111111", 1.0)
    ax.text(ox + sx + 0.030, oy - 0.006, r"$\hat z_1$", fontsize=10, color=label_color)
    ax.text(ox - 0.010, oy + sy + 0.030, r"$\hat z_2$", fontsize=10, color=label_color)


def fig1_concept(out_path: Path):
    """Conceptual schematic for action-conditioned predictive consistency."""
    fig, ax = plt.subplots(figsize=(13.2, 7.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.965, "Action-Conditioned Predictive Consistency",
            ha="center", va="top", fontsize=19, fontweight="bold", color="#1A202C")
    ax.text(
        0.5, 0.925,
        "Same-state visual perturbations should agree after prediction; action-relevant differences must remain separable.",
        ha="center", va="top", fontsize=11.2, color="#4A5568"
    )

    # Panel A.
    _box(ax, (0.035, 0.365), 0.45, 0.50, "#F8FBFF", "#2B6CB0", 1.2, radius=0.025)
    ax.text(0.060, 0.835, "A. Same-state predictive consistency",
            fontsize=12.5, fontweight="bold", color="#1E4E8C")
    ax.text(0.060, 0.784, "Clean/noisy views may encode differently;\n"
            "the same action should predict the same future",
            fontsize=7.7, color="#1E4E8C", linespacing=1.05)
    card_x, card_w, card_h = 0.070, 0.135, 0.090
    _draw_tworoom_card(ax, card_x, 0.625, card_w, card_h, "original", "original")
    _draw_tworoom_card(ax, card_x, 0.500, card_w, card_h, "noise", "Gaussian pixel noise")
    _draw_tworoom_card(ax, card_x, 0.375, card_w, card_h, "lighting", "lighting / texture shift")
    _draw_latent_axes(ax, (0.305, 0.470), (0.115, 0.170), "#1E4E8C")
    cluster = [(0.352, 0.565), (0.365, 0.595), (0.377, 0.550), (0.390, 0.580)]
    for p in cluster:
        ax.add_patch(Circle(p, 0.008, fc="#5B8DEF", ec="#1D4ED8", lw=0.8))
    ax.add_patch(Circle((0.371, 0.573), 0.060, fill=False, ec="#1E4E8C", ls=(0, (4, 4)), lw=1.1))
    for y in [0.670, 0.545, 0.420]:
        _arrow(ax, (0.215, y), (0.292, 0.572), "#2B6CB0", 1.5)
    ax.text(0.305, 0.415, "same state + same action\n-> consistent predictions",
            ha="center", va="top", fontsize=10, color="#1E4E8C", fontweight="bold")

    # Panel B.
    _box(ax, (0.515, 0.365), 0.45, 0.50, "#FAFFF8", "#2F855A", 1.2, radius=0.025)
    ax.text(0.540, 0.835, "B. Action-relevant discriminability",
            fontsize=12.5, fontweight="bold", color="#276749")
    ax.text(0.540, 0.784, "State differences that change transition, cost, or action\n"
            "stay separable after prediction",
            fontsize=7.7, color="#276749", linespacing=1.05)
    px, py = 0.545, 0.675
    state_specs = [
        ("state A: contact left", -18, -0.11, "#2F855A", 0.625),
        ("state B: centered", -5, 0.00, "#C53030", 0.500),
        ("state C: contact right", 10, 0.10, "#805AD5", 0.375),
    ]
    for label, angle, dx, color, y in state_specs:
        _draw_pusht_card(ax, px, y, card_w, card_h, angle, dx, label, color)
    _draw_latent_axes(ax, (0.785, 0.470), (0.115, 0.170), "#276749")
    clusters = [
        ((0.838, 0.620), "#2F855A"),
        ((0.835, 0.555), "#C53030"),
        ((0.865, 0.490), "#805AD5"),
    ]
    for center, color in clusters:
        cx, cy = center
        pts = [(cx - 0.016, cy + 0.005), (cx + 0.006, cy + 0.020), (cx + 0.018, cy - 0.012)]
        for p in pts:
            ax.add_patch(Circle(p, 0.008, fc=color, ec="#2B2B2B", lw=0.5, alpha=0.75))
        ax.add_patch(Circle(center, 0.045, fill=False, ec=color, ls=(0, (4, 4)), lw=1.0))
    for _, _, _, color, y in state_specs:
        _arrow(ax, (0.690, y + 0.045), (0.770, y + 0.005), color, 1.5)
    ax.text(0.920, 0.610, "different transition /\ncost / action\n-> separated predictions",
            ha="center", va="center", fontsize=8.8, color="#276749", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#FAFFF8", edgecolor="none", alpha=0.92))

    # Panel C.
    _box(ax, (0.035, 0.070), 0.93, 0.225, "#FBFBFA", "#A0AEC0", 1.0, radius=0.025)
    ax.text(0.500, 0.262, "C. Task-dependent selective-consistency demand",
            fontsize=13.5, fontweight="bold", ha="center", color="#1A202C")
    x0, x1, y = 0.175, 0.825, 0.185
    ax.plot([x0, x1], [y, y], color="#4A5568", lw=1.4)
    _arrow(ax, (0.500, y), (x0, y), "#B91C1C", 2.0)
    _arrow(ax, (0.500, y), (x1, y), "#1B7F3A", 2.0)
    ax.text(x0 - 0.014, y + 0.040, "Needs stronger\ndiscriminability\nguard",
            ha="right", va="center", fontsize=9.5, color="#B91C1C", fontweight="bold")
    ax.text(x1 + 0.014, y + 0.040, "Tolerates stronger\nsame-state\nconsistency",
            ha="left", va="center", fontsize=9.5, color="#1B7F3A", fontweight="bold")
    task_pos = [
        ("PushT", 0.245, "#B91C1C", "contact precision"),
        ("Reacher", 0.430, "#2B6CB0", "moderate resolution"),
        ("Cube", 0.575, "#6B46C1", "structured manipulation"),
        ("TwoRoom", 0.745, "#1B7F3A", "coarse topological\nresolution sufficient"),
    ]
    for name, x, color, sub in task_pos:
        ax.add_patch(Circle((x, y), 0.0095, fc=color, ec="white", lw=0.7, zorder=3))
        ax.text(x, y - 0.030, name, ha="center", va="top",
                fontsize=10.8, color=color, fontweight="bold")
        ax.text(x, y - 0.060, sub, ha="center", va="top", fontsize=8.3, color="#4A5568")

    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ============================================================================
# Figure 2 — Sweep curves: unperturbed and corrupted eval vs sigma_max, per task
# ============================================================================

def fig2_sweep(out_path: Path):
    tables = _canonical_eval_tables()
    sweep = tables["sweep"]
    tasks = tables["tasks"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.7), sharey=True)
    for ax, t in zip(axes, tasks):
        ax.errorbar(SWEEP_STDS, sweep[t]["clean"], yerr=sweep[t]["clean_std"],
                    fmt="o-", color="#4477AA",
                    label="Eval: unperturbed images",
                    linewidth=1.7, markersize=4.5, capsize=2.2, elinewidth=0.85)
        ax.errorbar(SWEEP_STDS, sweep[t]["px08"], yerr=sweep[t]["px08_std"],
                    fmt="s-", color="#EE6677",
                    label=ROBUST_EVAL_LABEL,
                    linewidth=1.7, markersize=4.5, capsize=2.2, elinewidth=0.85)
        ax.set_title(t, fontsize=11)
        ax.set_xlabel(r"Train-time noise level $\sigma_{\max}$", fontsize=10)
        ax.set_xticks([0, 0.02, 0.04, 0.06, 0.08])
        ax.set_xticklabels(["0", "0.02", "0.04", "0.06", "0.08"])
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3, linewidth=0.5)
        ax.tick_params(labelsize=9.5)
    axes[0].set_ylabel("Success rate (%)", fontsize=10.5)
    # Shared legend above the panels so the two evaluation curves are unambiguous.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.04), ncol=2,
               frameon=False, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ============================================================================
# Figure 3 — Scatter: fragility ratio vs eval drop
# Uses the n=9 LeWM PushT sweep only.
# ============================================================================

def _canonical_fig3_rows() -> List[Tuple[float, float, float, float]]:
    """Return [(std_max, ratio, clean, px08), ...] from canonical release JSON."""
    evals = _load_canonical_evals()["PushT"]
    diag = _load_canonical_diagnostics()["predictor_metrics_by_task"]["PushT"]
    rows = []
    for std_key in sorted(diag, key=float):
        eval_entry = evals[std_key]["metrics"]
        diag_entry = diag[std_key]
        rows.append((
            float(std_key),
            float(diag_entry["predictor_target_to_nn_cos_ratio_at_max_std"]),
            float(eval_entry["clean"]["mean"]),
            float(eval_entry[ROBUST_EVAL_METRIC]["mean"]),
        ))
    return rows


def _spearman(xs: np.ndarray, ys: np.ndarray) -> float:
    """Spearman ρ via rank-Pearson, with mid-rank tie averaging (no scipy)."""
    if len(xs) < 3:
        return float("nan")
    def _rank(a):
        a = np.asarray(a, dtype=float)
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty(len(a), dtype=float)
        i = 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks
    rx, ry = _rank(xs), _rank(ys)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = max(float((rx ** 2).sum() ** 0.5 * (ry ** 2).sum() ** 0.5), 1e-9)
    return float((rx * ry).sum() / denom)


def fig3_scatter(out_path: Path, data_root: Path):
    del data_root  # legacy arg; Figure 3 is now driven entirely by canonical JSON.
    rows = _canonical_fig3_rows()
    if len(rows) < 5:
        print(f"  WARN: only {len(rows)} canonical PushT LeWM ckpts; skipping fig3")
        return

    # Two side-by-side panels: vs unperturbed | vs corruption drop.
    fig, (ax_clean, ax_drop) = plt.subplots(1, 2, figsize=(10.5, 4.3))

    xs = np.array([r[1] for r in rows])
    cleans = np.array([r[2] for r in rows])
    px08s = np.array([r[3] for r in rows])
    drops = cleans - px08s
    stds_ax = np.array([r[0] for r in rows])

    def _panel(ax_, ys, ylabel, title, anchor_y_top):
        sc = ax_.scatter(xs, ys, marker="o", s=90,
                         c=stds_ax, cmap="Blues", edgecolor="black", linewidth=0.5,
                         vmin=0, vmax=0.08)
        # log-linear fit
        logxs = np.log10(xs)
        p = np.polyfit(logxs, ys, 1)
        xf = np.linspace(logxs.min() - 0.1, logxs.max() + 0.1, 50)
        ax_.plot(10 ** xf, p[0] * xf + p[1], "k--", linewidth=1.0, alpha=0.6)
        rho = _spearman(xs, ys)
        ax_.text(0.04, anchor_y_top,
                 f"Spearman ρ (LeWM, n={len(rows)}): {rho:+.2f}\n"
                 f"linear-fit slope (on log x): {p[0]:+.1f}",
                 transform=ax_.transAxes,
                 va="top" if anchor_y_top > 0.5 else "bottom", ha="left",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                           edgecolor="gray", alpha=0.92), fontsize=8.5,
                 family="DejaVu Sans Mono")
        ax_.set_xscale("log")
        ax_.set_xlabel("Fragility ratio (target/NN at max training noise)")
        ax_.set_ylabel(ylabel)
        ax_.set_title(title, fontsize=10, loc="left", pad=8)
        ax_.grid(alpha=0.25, linewidth=0.4)
        return rho

    rho_clean = _panel(ax_clean, cleans, "PushT unperturbed success rate (%)",
                       "(a)  vs unperturbed success — residual ckpt-quality signal",
                       anchor_y_top=0.18)
    rho_drop = _panel(ax_drop, drops, "PushT eval drop (unperturbed - obs-noise 0.08, pts)",
                      "(b)  vs corruption drop - mediated by training noise",
                      anchor_y_top=0.96)

    # colourbar for the training noise level
    cbar = fig.colorbar(ax_clean.collections[0], ax=[ax_clean, ax_drop],
                        fraction=0.025, pad=0.04)
    cbar.set_label(r"training noise level $\sigma_{\max}$", fontsize=8.5)

    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path} (n={len(rows)} LeWM ckpts;  "
          f"ρ_unperturbed={rho_clean:+.2f}, ρ_drop={rho_drop:+.2f})")


# ============================================================================
# Figure 4 — Radar: 4 tasks × 6 diagnostic metrics
# base vs representative diagnostic checkpoint.
# Metrics are normalized per-task to [0,1] using the wider of {base, representative}
# extents so radial axes are visually comparable.
# ============================================================================

def fig4_radar(out_path: Path):
    metrics = DIAG_METRICS
    diag_tables = _canonical_diag_tables()
    short_names = [
        "eff. rank",
        "NN cos dist",
        "trans. res. L2",
        "trans. res. cos",
        "id-probe R²",
        "action shift",
    ]
    tasks = diag_tables["tasks"]
    diag_data = diag_tables["values"]
    diag_representative = diag_tables["representative"]

    # Normalize each metric across all 8 (4 tasks × 2 ckpts) values to [0,1].
    metric_min = [float("inf")] * len(metrics)
    metric_max = [-float("inf")] * len(metrics)
    for t in tasks:
        for which in ("base", "representative"):
            for i, v in enumerate(diag_data[t][which]):
                metric_min[i] = min(metric_min[i], v)
                metric_max[i] = max(metric_max[i], v)
    def _norm(vals):
        out = []
        for i, v in enumerate(vals):
            lo, hi = metric_min[i], metric_max[i]
            out.append((v - lo) / max(hi - lo, 1e-9))
        return out

    angles = np.linspace(0, 2 * math.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    # 2x2 grid is more readable than 1x4 for radar charts with 6 long labels.
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 9.0), subplot_kw=dict(polar=True))
    axes = axes.flatten()
    for ax, t in zip(axes, tasks):
        base = _norm(diag_data[t]["base"])
        representative = _norm(diag_data[t]["representative"])
        base += base[:1]
        representative += representative[:1]
        ax.plot(angles, base, "o-", linewidth=1.6, color="#4477AA",
                label="base", markersize=4.5)
        ax.fill(angles, base, alpha=0.18, color="#4477AA")
        diag_std = diag_representative[t]["std"]
        ax.plot(angles, representative, "s-", linewidth=1.6, color="#228833",
                label=f"representative diag (σ={diag_std:.3f})", markersize=4.5)
        ax.fill(angles, representative, alpha=0.18, color="#228833")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(short_names, fontsize=9)
        ax.tick_params(axis="x", pad=14)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_yticklabels(["0", ".5", "1"], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(t, fontsize=12, pad=28)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.16),
                  frameon=False, fontsize=8.5, ncol=2)

    fig.subplots_adjust(left=0.07, right=0.94, top=0.94, bottom=0.05,
                        wspace=0.6, hspace=0.65)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ============================================================================
# Figure 5 — Mechanism schematic only
# ============================================================================

def fig5_mechanism(out_path: Path):
    # Wider canvas + clearly separated bands eliminates the box / callout
    # collisions of the previous layout. Vertical bands (top -> bottom):
    #   0.82  upper callout (auxiliary sanity-check)
    #   0.50  pipeline boxes + arrows
    #   0.36  layer labels under arrows
    #   0.20  quantitative attribution callout
    #   0.06  one-line interpretation
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    ax.axis("off")

    box_hw, box_hh = 0.085, 0.07
    box_y = 0.50
    box_centers = [0.10, 0.36, 0.62, 0.88]
    box_labels = ["pixels\n(noise +)", "encoder $f$", "predictor $g$", "CEM /\nplanner"]
    box_colors = ["#FDE7E9", "#E7F0FA", "#FCEFD8", "#E8F4EA"]

    for x, lab, color in zip(box_centers, box_labels, box_colors):
        ax.add_patch(plt.Rectangle(
            (x - box_hw, box_y - box_hh), 2 * box_hw, 2 * box_hh,
            edgecolor="black", facecolor=color, linewidth=1.0
        ))
        ax.text(x, box_y, lab, ha="center", va="center", fontsize=10)

    for i in range(len(box_centers) - 1):
        x0 = box_centers[i] + box_hw + 0.015
        x1 = box_centers[i + 1] - box_hw - 0.015
        ax.annotate("", xy=(x1, box_y), xytext=(x0, box_y),
                    arrowprops=dict(arrowstyle="->", lw=1.5))

    layer_labels = [
        "encoder shift",
        "predictor drift",
        "planning-time action selection",
    ]
    layer_x = [
        (box_centers[0] + box_centers[1]) / 2,
        (box_centers[1] + box_centers[2]) / 2,
        (box_centers[2] + box_centers[3]) / 2,
    ]
    for x, label in zip(layer_x, layer_labels):
        ax.text(x, 0.36, label, ha="center", va="center",
                fontsize=9, style="italic", color="#555555")

    ax.text(
        0.50, 0.86,
        "Auxiliary cost-swap sanity check (not part of the canonical 36-ckpt table):\n"
        "TwoRoom one-off ablation, px+goal 0.03  36.0 → 42.0 after cosine→mse swap; unperturbed ref = 69.7",
        ha="center", va="center", fontsize=8.5, color="#7A4B00",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF6E5",
                  edgecolor="#B57F2A", linewidth=0.7)
    )

    ax.text(
        0.50, 0.20,
        "Quantitative attribution in §4.6.2 uses the two full-coverage LeWM n=9 predictor metrics.\n"
        "After conditioning on std_max, Reacher's multi-step drift is the only non-trivial residual signal.",
        ha="center", va="center", fontsize=8.8,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F5",
                  edgecolor="#AAAAAA", linewidth=0.7)
    )

    ax.text(
        0.50, 0.06,
        "Interpretation: the common qualitative path is encoder shift transduced by the predictor;\n"
        "the cost function alone is unlikely to explain the collapse.",
        ha="center", va="center", fontsize=9.2
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ============================================================================
# Figure 6 — Pareto frontier: (unperturbed, observation-noise 0.08) per
# task and training noise level. Kept for archival reproduction.
# ============================================================================

def fig6_pareto(out_path: Path):
    tables = _canonical_eval_tables()
    sweep = tables["sweep"]
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    colors = {"TwoRoom": "#4477AA", "PushT": "#EE6677",
              "Reacher": "#228833", "Cube": "#CCBB44"}
    markers = {"TwoRoom": "o", "PushT": "s", "Reacher": "^", "Cube": "D"}

    for task in ["TwoRoom", "PushT", "Reacher", "Cube"]:
        xs = sweep[task]["clean"]
        ys = sweep[task]["px08"]
        ax.plot(xs, ys, "-", color=colors[task], linewidth=1.0, alpha=0.6)
        # mark base (std=0) separately
        ax.scatter(xs[0], ys[0], s=90, marker=markers[task],
                   facecolor="white", edgecolor=colors[task],
                   linewidth=1.5, zorder=4,
                   label=f"{task} (base, σ=0)")
        # mark sweep points
        for x, y, s in zip(xs[1:], ys[1:], SWEEP_STDS[1:]):
            ax.scatter(x, y, s=55, marker=markers[task],
                       color=colors[task], alpha=0.55 + 0.05 * SWEEP_STDS[1:].index(s),
                       edgecolor="black", linewidth=0.3, zorder=3)
        # Highlight the representative high-corruption observation-noise row.
        best_std = tables["corrupted_point_best"][task]["std"]
        if best_std in SWEEP_STDS:
            i = SWEEP_STDS.index(best_std)
            ax.scatter(xs[i], ys[i], s=180, marker=markers[task],
                       facecolor="none", edgecolor=colors[task],
                       linewidth=2.0, zorder=5)

    # Diagonal y=x — ckpts on or above this diagonal are "corrupted ≥ unperturbed"
    lo, hi = 0, 100
    ax.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=0.8, alpha=0.5)
    ax.text(95, 97, "y = x", color="gray", fontsize=8, alpha=0.7,
            ha="right", va="bottom")

    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Unperturbed success rate (%)")
    ax.set_ylabel("Corrupted success rate (%) - obs-noise 0.08")
    ax.grid(alpha=0.25, linewidth=0.4)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ============================================================================
# Entry point
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="assets/paper1_figs",
                    help="output directory (relative to repo root)")
    ap.add_argument("--data-root",
                    default="dataset/ag_data/data/world_model/quentinll",
                    help="legacy relative path under the local dataset prefix; no longer required once assets/paper1_data/canonical_diagnostics_20260517.json is present")
    ap.add_argument("--only", nargs="+", choices=["1", "2", "3", "4", "5", "6"],
                    help="render only these figures (default: all)")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)

    _setup_style()
    print(f"Output dir: {out_dir}")
    print(f"Data root:  {data_root} (legacy, relative to the configured dataset prefix)")
    # Default renders only the script-generated figures used in the paper.
    # Pruned slots remain callable via --only for archival reproduction.
    selected = set(args.only or ["2", "5"])

    if "1" in selected:
        print("  skip fig1_concept.png (not generated by this script)")
    if "2" in selected:
        fig2_sweep(out_dir / "fig2_sweep.png")
    if "3" in selected:
        fig6_pareto(out_dir / "fig3_pareto.png")
    if "4" in selected:
        fig4_radar(out_dir / "fig4_radar.png")
    if "5" in selected:
        fig3_scatter(out_dir / "fig5_scatter.png", data_root)
    if "6" in selected:
        fig5_mechanism(out_dir / "fig6_mechanism.png")

    print("done.")


if __name__ == "__main__":
    main()
