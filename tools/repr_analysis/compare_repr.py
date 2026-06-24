"""
compare_repr.py - Side-by-side comparison for two representation analysis runs.

Reads two analysis directories produced by analyze_repr.py and renders a
side-by-side scatter plot for quick comparison, e.g.:
  - SIGReg vs BN+uniformity
  - baseline vs new loss

Typical usage:

  python compare_repr.py \
      --left-dir /tmp/repr_sigreg \
      --right-dir /tmp/repr_bn_uniform \
      --projection tsne \
      --output /tmp/repr_compare_tsne.png \
      --color-dims 0 1 \
      --left-label SIGReg \
      --right-label BN+uniformity
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_rows(analysis_dir: Path, projection: str) -> List[Dict[str, float]]:
    file_name = f"{projection}_projection.json"
    path = analysis_dir / file_name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    with open(path, "r") as f:
        rows = json.load(f)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def load_summary(analysis_dir: Path) -> Dict:
    path = analysis_dir / "summary.json"
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def resolve_color_key(rows: List[Dict[str, float]], dim: int) -> str:
    preferred = f"state_{dim}"
    if preferred in rows[0]:
        return preferred
    fallback = "time_id" if dim == 0 else "seq_id" if dim == 1 else None
    if fallback and fallback in rows[0]:
        return fallback
    raise KeyError(
        f"{preferred} not found in projection file, and no fallback color key is available."
    )


def make_caption(summary: dict) -> str:
    if not summary:
        return ""
    parts = []
    meta = summary.get("meta", {})
    emb = summary.get("embedding", {})
    pred = summary.get("prediction", {})
    rollout = summary.get("rollout", {})
    planning = summary.get("planning", {})
    ref = summary.get("reference_probe", {})
    if meta.get("dataset"):
        parts.append(f"dataset={meta['dataset']}")
    if "effective_rank" in emb:
        parts.append(f"rank={emb['effective_rank']:.1f}")
    if "pred_target_cosine_distance_mean" in pred:
        parts.append(f"pred_cos_dist={pred['pred_target_cosine_distance_mean']:.3f}")
    if "rollout_cosine_distance_last_mean" in rollout:
        parts.append(f"rollout_last={rollout['rollout_cosine_distance_last_mean']:.3f}")
    if "cost_margin_mean" in planning:
        parts.append(f"cost_margin={planning['cost_margin_mean']:.3f}")
    if "expert_beats_random_rate" in planning:
        parts.append(f"expert>{planning['expert_beats_random_rate']:.2f}")
    if "distance_rank_corr_cross_seq" in ref:
        parts.append(f"ref_rank={ref['distance_rank_corr_cross_seq']:.3f}")
    return " | ".join(parts)


def save_comparison_plot(
    *,
    left_rows: List[Dict[str, float]],
    right_rows: List[Dict[str, float]],
    left_summary: Dict,
    right_summary: Dict,
    left_label: str,
    right_label: str,
    projection: str,
    color_dims: List[int],
    output: Path,
    title: str | None,
    alpha: float,
    size: float,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "compare_repr.py requires matplotlib. Install it with `pip install matplotlib`."
        ) from exc

    ncols = len(color_dims)
    fig, axes = plt.subplots(2, ncols, figsize=(6 * ncols, 10), squeeze=False)

    for col, dim in enumerate(color_dims):
        key_left = resolve_color_key(left_rows, dim)
        key_right = resolve_color_key(right_rows, dim)

        left_x = [row["x"] for row in left_rows]
        left_y = [row["y"] for row in left_rows]
        left_c = [row[key_left] for row in left_rows]

        right_x = [row["x"] for row in right_rows]
        right_y = [row["y"] for row in right_rows]
        right_c = [row[key_right] for row in right_rows]

        vmin = min(min(left_c), min(right_c))
        vmax = max(max(left_c), max(right_c))

        ax_l = axes[0][col]
        ax_r = axes[1][col]

        sc_l = ax_l.scatter(
            left_x,
            left_y,
            c=left_c,
            s=size,
            alpha=alpha,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        sc_r = ax_r.scatter(
            right_x,
            right_y,
            c=right_c,
            s=size,
            alpha=alpha,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )

        ax_l.set_title(f"{left_label} | {projection.upper()} | {key_left}")
        ax_r.set_title(f"{right_label} | {projection.upper()} | {key_right}")
        ax_l.set_xlabel("x")
        ax_l.set_ylabel("y")
        ax_r.set_xlabel("x")
        ax_r.set_ylabel("y")

        cap_l = make_caption(left_summary)
        cap_r = make_caption(right_summary)
        if cap_l:
            ax_l.text(0.01, -0.12, cap_l, transform=ax_l.transAxes, fontsize=9, va="top")
        if cap_r:
            ax_r.text(0.01, -0.12, cap_r, transform=ax_r.transAxes, fontsize=9, va="top")

        fig.colorbar(sc_l, ax=[ax_l, ax_r], fraction=0.03, pad=0.02)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-dir", type=str, required=True)
    parser.add_argument("--right-dir", type=str, required=True)
    parser.add_argument("--left-label", type=str, default="Left")
    parser.add_argument("--right-label", type=str, default="Right")
    parser.add_argument("--projection", type=str, default="tsne", choices=["pca", "tsne"])
    parser.add_argument("--color-dims", type=int, nargs="+", default=[0], help="State dimensions to color by")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--size", type=float, default=6.0)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    left_dir = Path(args.left_dir)
    right_dir = Path(args.right_dir)
    left_rows = load_rows(left_dir, args.projection)
    right_rows = load_rows(right_dir, args.projection)
    left_summary = load_summary(left_dir)
    right_summary = load_summary(right_dir)

    output = Path(args.output)
    save_comparison_plot(
        left_rows=left_rows,
        right_rows=right_rows,
        left_summary=left_summary,
        right_summary=right_summary,
        left_label=args.left_label,
        right_label=args.right_label,
        projection=args.projection,
        color_dims=args.color_dims,
        output=output,
        title=args.title,
        alpha=args.alpha,
        size=args.size,
    )
    print(f"[compare_repr] saved to {output}")


if __name__ == "__main__":
    main()
