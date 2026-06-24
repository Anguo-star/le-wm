"""
diagnostic_correlation.py — P0.7 automation

Compute Spearman + Pearson correlations between diagnostic metrics and eval
scores, with bootstrap confidence intervals. Produces a CSV table and a
heatmap PNG.

This is the automation backbone of the diagnostic ↔ eval correlation
workflow described in research_notebook_swm.md §6 P0.4 / P0.7. The intent is to take a
fixed set of label-free diagnostic indicators (from `run_full_diagnostics`)
and rank them by predictive value against measured eval scores, so that
strong indicators can be promoted to "main metric" status (P0.5) and weak
ones can be discarded.

References:
    - Spearman rank correlation as a label-free performance predictor for
      pretrained models: Garg et al., "Leveraging Unlabeled Data to Predict
      Out-of-Distribution Performance" (ATC), ICLR 2022; Deng & Zheng,
      "Are Labels Always Necessary for Classifier Accuracy Evaluation?",
      CVPR 2021.
    - Bootstrap percentile confidence interval for correlation:
      Efron & Tibshirani, "An Introduction to the Bootstrap", 1993,
      Ch. 13 (BCa / percentile method).
    - Active validation on holdout checkpoints (P0.6 protocol): Kossen
      et al., "Active Testing", ICML 2021.

Spearman ρ implementation note:
    `_average_rank` matches scipy.stats.rankdata(..., method="average") without
    requiring scipy. This matters because eval scores often have exact ties in
    small checkpoint grids; ordinal ranks can move ρ noticeably.

Usage::

    python -m tools.repr_analysis.diagnostic_correlation \
        --diagnostics /path/to/diagnostics_summary.json \
        --eval-scores /path/to/eval_scores.json \
        --out-dir /path/to/output

`eval_scores.json` format::

    {
        "model_label_1": 87.3,
        "model_label_2": 94.0,
        ...
    }

If `--eval-scores` is omitted, the script looks for `eval_scores.json` in the
same directory as `diagnostics_summary.json`.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np


def _load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def _save_json(path: Path, data: Any) -> None:
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def _spearman_bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 1000,
    rng: np.random.Generator | None = None,
) -> Tuple[float, float, float]:
    """Return (rho, ci_low, ci_high) using bootstrap percentile method.

    Bootstrap CI: percentile method (Efron & Tibshirani 1993). For each
    bootstrap sample we resample (x, y) pairs with replacement and recompute
    ρ; the 2.5 / 97.5 percentiles of the resulting distribution are the
    CI bounds. With n=8–11 (our typical N) the CI is wide and should be
    interpreted as "is the sign reliable" rather than a precise estimate.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(x)
    if n < 4:
        return float("nan"), float("nan"), float("nan")

    # scipy may not be available; implement average ranks for ties.
    def _average_rank(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a, kind="mergesort")
        sorted_vals = a[order]
        ranks = np.empty(len(a), dtype=float)
        start = 0
        while start < len(a):
            end = start + 1
            while end < len(a) and sorted_vals[end] == sorted_vals[start]:
                end += 1
            # Zero-based average rank; Pearson correlation is invariant to the
            # shared affine shift relative to one-based ranks.
            avg_rank = 0.5 * (start + end - 1)
            ranks[order[start:end]] = avg_rank
            start = end
        return ranks

    def _rho(a: np.ndarray, b: np.ndarray) -> float:
        ra = _average_rank(a)
        rb = _average_rank(b)
        return _pearson(ra, rb)

    rho_obs = _rho(x, y)
    boot_rhos = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        rho_i = _rho(x[idx], y[idx])
        if not math.isnan(rho_i):
            boot_rhos.append(rho_i)
    boot_rhos = np.array(boot_rhos)
    if boot_rhos.size == 0:
        return float(rho_obs), float("nan"), float("nan")
    ci_low = float(np.percentile(boot_rhos, 2.5))
    ci_high = float(np.percentile(boot_rhos, 97.5))
    return float(rho_obs), ci_low, ci_high


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    xm = x - x.mean()
    ym = y - y.mean()
    denom = np.sqrt(np.sum(xm * xm) * np.sum(ym * ym))
    if denom < 1e-12:
        return float("nan")
    return float(np.sum(xm * ym) / denom)


def _is_numeric_scalar(v: Any) -> bool:
    if isinstance(v, (int, float, np.floating, np.integer)):
        # Correlation code assumes finite inputs. Treat NaN/Inf as missing so
        # censored diagnostic fields do not leak invalid values into Pearson or
        # rank statistics.
        return math.isfinite(float(v))
    return False


def _is_constant(vals: Sequence[float]) -> bool:
    return len(set(round(v, 8) for v in vals)) <= 1


def compute_correlations(
    diagnostics: Sequence[Mapping[str, Any]],
    eval_scores: Mapping[str, float],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Compute correlations for every numeric diagnostic field vs eval_score."""
    # Build model -> eval lookup
    model_to_eval = {k: float(v) for k, v in eval_scores.items()}

    # Collect numeric fields from diagnostics
    numeric_fields: set[str] = set()
    for row in diagnostics:
        for k, v in row.items():
            if k in ("model", "ckpt", "geometry_flag", "recommendation", "latent_noise_geometry"):
                continue
            if _is_numeric_scalar(v):
                numeric_fields.add(k)

    rng = np.random.default_rng(seed)
    results: List[Dict[str, Any]] = []

    for field in sorted(numeric_fields):
        x_vals = []
        y_vals = []
        models = []
        for row in diagnostics:
            model = row.get("model", "")
            if model not in model_to_eval:
                continue
            v = row.get(field)
            if not _is_numeric_scalar(v):
                continue
            x_vals.append(float(v))
            y_vals.append(model_to_eval[model])
            models.append(model)

        n = len(x_vals)
        if n < 3:
            continue
        if _is_constant(x_vals) or _is_constant(y_vals):
            continue

        x_arr = np.array(x_vals)
        y_arr = np.array(y_vals)

        r_pearson = _pearson(x_arr, y_arr)
        rho, ci_low, ci_high = _spearman_bootstrap(x_arr, y_arr, n_bootstrap, rng)

        results.append({
            "diagnostic_field": field,
            "n": n,
            "pearson_r": round(r_pearson, 4),
            "spearman_rho": round(rho, 4),
            "spearman_ci_low": round(ci_low, 4),
            "spearman_ci_high": round(ci_high, 4),
            "models": ",".join(models),
        })

    # Sort by absolute Spearman rho descending
    results.sort(key=lambda row: abs(row["spearman_rho"]), reverse=True)
    return results


def plot_correlation_heatmap(
    results: Sequence[Mapping[str, Any]],
    out_path: Path,
    top_k: int = 20,
) -> None:
    """Plot a horizontal bar chart of |Spearman rho| with CI."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in results if not math.isnan(r["spearman_rho"])]
    rows = rows[:top_k]
    if not rows:
        return

    labels = [r["diagnostic_field"] for r in rows]
    rhos = [r["spearman_rho"] for r in rows]
    ci_lows = [r["spearman_ci_low"] for r in rows]
    ci_highs = [r["spearman_ci_high"] for r in rows]

    y_pos = np.arange(len(labels))
    colors = ["#2ca02c" if r >= 0 else "#d62728" for r in rhos]

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.4)))
    ax.barh(y_pos, rhos, color=colors, alpha=0.7)

    # Error bars (asymmetric)
    err_low = [max(0, r - cl) for r, cl in zip(rhos, ci_lows)]
    err_high = [max(0, ch - r) for r, ch in zip(rhos, ci_highs)]
    ax.errorbar(rhos, y_pos, xerr=[err_low, err_high], fmt="none", color="black", capsize=3, linewidth=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Spearman ρ (with 95% bootstrap CI)", fontsize=10)
    ax.set_title(f"Diagnostic ↔ Eval Correlation (top {len(rows)})", fontsize=11)
    ax.set_xlim([-1.05, 1.05])
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="P0.7: diagnostic ↔ eval correlation")
    parser.add_argument("--diagnostics", required=True, type=Path, help="Path to diagnostics_summary.json")
    parser.add_argument("--eval-scores", type=Path, default=None, help="Path to eval_scores.json")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: same as diagnostics)")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Bootstrap samples for CI")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--top-k", type=int, default=20, help="Number of top fields to plot")
    args = parser.parse_args()

    diagnostics = _load_json(args.diagnostics)
    if not isinstance(diagnostics, list):
        raise ValueError("diagnostics_summary.json should contain a list of rows")

    eval_path = args.eval_scores
    if eval_path is None:
        eval_path = args.diagnostics.parent / "eval_scores.json"
    if not eval_path.exists():
        raise FileNotFoundError(
            f"eval_scores.json not found at {eval_path}. "
            "Provide --eval-scores or create the file."
        )
    eval_scores = _load_json(eval_path)
    if not isinstance(eval_scores, dict):
        raise ValueError("eval_scores.json should contain a {model: score} mapping")

    out_dir = args.out_dir or args.diagnostics.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    results = compute_correlations(
        diagnostics, eval_scores, n_bootstrap=args.n_bootstrap, seed=args.seed
    )

    csv_path = out_dir / "diagnostic_correlation.csv"
    with csv_path.open("w", newline="") as f:
        import csv
        if results:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        else:
            f.write("diagnostic_field,n,pearson_r,spearman_rho,spearman_ci_low,spearman_ci_high,models\n")

    png_path = out_dir / "diagnostic_correlation.png"
    plot_correlation_heatmap(results, png_path, top_k=args.top_k)

    summary = {
        "n_models": len(eval_scores),
        "n_fields_tested": len(results),
        "top_3_by_spearman": results[:3],
    }
    _save_json(out_dir / "diagnostic_correlation_summary.json", summary)

    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")
    print(f"Top 3 by |Spearman|:")
    for r in results[:3]:
        print(f"  {r['diagnostic_field']:45s} rho={r['spearman_rho']:+.3f} [{r['spearman_ci_low']:+.3f}, {r['spearman_ci_high']:+.3f}] n={r['n']}")


if __name__ == "__main__":
    main()
