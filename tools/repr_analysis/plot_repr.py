"""
plot_repr.py - Plot PCA / t-SNE exports from analyze_repr.py.

This script turns:
  - pca_projection.json
  - tsne_projection.json

into simple scatter plots that are easy to compare across checkpoints, e.g.
SIGReg vs BN+uniformity.

Typical usage:

  python plot_repr.py \
      --input /tmp/repr_analysis_pusht/tsne_projection.json \
      --output /tmp/repr_analysis_pusht/tsne_state0_state1.png \
      --color-dims 0 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_rows(path: Path) -> List[Dict[str, float]]:
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


def save_projection_plot(
    rows: List[Dict[str, float]],
    *,
    output: Path,
    color_dims: List[int],
    title: str | None,
    alpha: float,
    size: float,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plot_repr.py requires matplotlib. Install it with `pip install matplotlib`."
        ) from exc

    if not rows:
        raise ValueError("No rows found in projection input.")

    x = [row["x"] for row in rows]
    y = [row["y"] for row in rows]

    n_panels = len(color_dims)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5), squeeze=False)
    axes = axes[0]

    for ax, dim in zip(axes, color_dims):
        key = resolve_color_key(rows, dim)
        c = [row[key] for row in rows]
        sc = ax.scatter(x, y, c=c, s=size, alpha=alpha, cmap="viridis")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{output.stem} colored by {key}")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Projection JSON from analyze_repr.py")
    parser.add_argument("--output", type=str, required=True, help="Output image path, e.g. out.png")
    parser.add_argument("--color-dims", type=int, nargs="+", default=[0], help="State dimensions to color by")
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--size", type=float, default=6.0)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    rows = load_rows(Path(args.input))
    output = Path(args.output)
    save_projection_plot(
        rows,
        output=output,
        color_dims=args.color_dims,
        title=args.title,
        alpha=args.alpha,
        size=args.size,
    )
    print(f"[plot_repr] saved to {output}")


if __name__ == "__main__":
    main()
