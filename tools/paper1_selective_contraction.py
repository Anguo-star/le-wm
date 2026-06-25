"""Selective-contraction branch diagnostics for Paper 1.

This script is intentionally separate from the paper-facing figure generator.
It answers a narrower mechanism question on existing full-sequence
perturbed-target checkpoints:

    do same-state perturbation clusters shrink, and what happens to simple
    state/action discriminability proxies?

The default path only reads released JSON artifacts and writes a compact branch
table.  The optional plotting paths load checkpoints and dataset windows to
render real encoder/predictor feature clouds, so they are eval-only but not
artifact-only.  Cluster envelopes are visualization summaries in a 2-D
projection; the printed panel statistics remain the high-D evidence.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("TwoRoom", "PushT", "Reacher", "Cube")
DEFAULT_DATA_DIR = ROOT / "assets" / "paper1_data"
DEFAULT_FIG_DIR = ROOT / "assets" / "phase1_figs" / "selective_contraction_3d"
DEFAULT_FIG2D_DIR = ROOT / "assets" / "phase1_figs" / "selective_contraction_2d"
DEFAULT_CLUSTER_DIR = ROOT / "assets" / "phase1_figs" / "selective_contraction_clusters"
DEFAULT_ATLAS_DIR = ROOT / "assets" / "phase1_figs" / "selective_contraction_atlas"
DEFAULT_OUT_JSON = DEFAULT_DATA_DIR / "selective_contraction_fullseq_branch.json"
DEFAULT_OUT_MD = DEFAULT_DATA_DIR / "selective_contraction_fullseq_branch.md"
TASK_DATASETS = {
    "TwoRoom": "tworoom",
    "PushT": "pusht_expert_train",
    "Reacher": "reacher",
    "Cube": "ogbench/cube_single_expert",
}


@dataclass(frozen=True)
class CkptSpec:
    label: str
    task: str
    std_key: str
    subdir: str
    model_file: Path


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_summary_paths(method: str, out_json: Path, out_md: Path) -> tuple[Path, Path]:
    if method == "LeWM":
        return out_json, out_md

    branch = "noise"
    slug = method.lower()
    if out_json == DEFAULT_OUT_JSON:
        out_json = DEFAULT_DATA_DIR / f"selective_contraction_{slug}_{branch}_branch.json"
    if out_md == DEFAULT_OUT_MD:
        out_md = DEFAULT_DATA_DIR / f"selective_contraction_{slug}_{branch}_branch.md"
    return out_json, out_md


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def _artifact_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(path)


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _ratio(num: Any, den: Any) -> float:
    if not _finite(num) or not _finite(den) or abs(float(den)) <= 1e-12:
        return float("nan")
    return float(num) / float(den)


def _drop_frac(base: Any, best: Any) -> float:
    if not _finite(base) or abs(float(base)) <= 1e-12 or not _finite(best):
        return float("nan")
    return 1.0 - float(best) / float(base)


def _fmt(x: Any, digits: int = 3) -> str:
    if not _finite(x):
        return "n/a"
    return f"{float(x):.{digits}g}"


def _fmt_arrow(a: Any, b: Any, digits: int = 3) -> str:
    return f"{_fmt(a, digits)} -> {_fmt(b, digits)}"


def _rows_by_task_std(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    out: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        if row.get("status", "ok") != "ok":
            continue
        key = (str(row["task"]), str(row["std_key"]))
        out[key] = row
    return out


def _phase_rows_by_task_std(
    rows: Iterable[Mapping[str, Any]], *, method: str
) -> dict[tuple[str, str], Mapping[str, Any]]:
    out: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        if row.get("status") != "ok" or row.get("method") != method:
            continue
        out[(str(row["task"]), str(row["std_key"]))] = row
    return out


def _best_row(rows: Sequence[Mapping[str, Any]], robust_metric: str) -> Mapping[str, Any]:
    return max(
        rows,
        key=lambda r: (
            float(r.get(robust_metric, float("nan"))),
            float(r.get("clean_success", float("nan"))),
        ),
    )


def build_summary(
    *,
    acpc_basin_path: Path,
    acpc_phase0_path: Path,
    robust_metric: str,
    method: str,
    method_label: str | None,
    robust_label: str | None,
) -> dict[str, Any]:
    basin_payload = _load_json(acpc_basin_path)
    phase_payload = _load_json(acpc_phase0_path)
    method_name = str(method)
    display_method = method_label or method_name
    basin_rows = [
        r
        for r in basin_payload["rows"]
        if r.get("status") == "ok" and str(r.get("method", method_name)) == method_name
    ]
    phase_by_key = _phase_rows_by_task_std(phase_payload["rows"], method=method_name)

    out_rows: list[dict[str, Any]] = []
    for task in TASKS:
        task_rows = [r for r in basin_rows if r["task"] == task]
        base = next(r for r in task_rows if r["std_key"] == "0.0")
        best = _best_row(task_rows, robust_metric)
        phase_base = phase_by_key.get((task, "0.0"), {})
        phase_best = phase_by_key.get((task, str(best["std_key"])), {})

        re_base = base["encoder_view_pair_l2_norm_by_nn"]
        re_best = best["encoder_view_pair_l2_norm_by_nn"]
        rf_base = base["pred_view_pair_l2_norm_by_transition"]
        rf_best = best["pred_view_pair_l2_norm_by_transition"]
        clean_nn_base = base["clean_nn_l2_median"]
        clean_nn_best = best["clean_nn_l2_median"]
        trans_base = base["clean_transition_l2_median"]
        trans_best = best["clean_transition_l2_median"]

        adm_base = phase_base.get("adm_l2_median", float("nan"))
        adm_best = phase_best.get("adm_l2_median", float("nan"))
        sprr_base = phase_base.get("sprr", float("nan"))
        sprr_best = phase_best.get("sprr", float("nan"))

        out_rows.append(
            {
                "task": task,
                "target_view_branch": (
                    "full_sequence_perturbed_target" if method_name == "LeWM" else "noise_sweep"
                ),
                "best_std_key": str(best["std_key"]),
                "best_subdir": best.get("subdir"),
                "clean_success_base": base["clean_success"],
                f"{robust_metric}_base": base[robust_metric],
                "clean_success_best": best["clean_success"],
                f"{robust_metric}_best": best[robust_metric],
                "encoder_radius_RE_base": re_base,
                "encoder_radius_RE_best": re_best,
                "encoder_radius_RE_drop_frac": _drop_frac(re_base, re_best),
                "prediction_radius_RF_base": rf_base,
                "prediction_radius_RF_best": rf_best,
                "prediction_radius_RF_drop_frac": _drop_frac(rf_base, rf_best),
                "prediction_selective_ratio_base": _ratio(1.0, rf_base),
                "prediction_selective_ratio_best": _ratio(1.0, rf_best),
                "clean_nn_l2_base": clean_nn_base,
                "clean_nn_l2_best": clean_nn_best,
                "clean_nn_l2_ratio_best_over_base": _ratio(clean_nn_best, clean_nn_base),
                "clean_transition_l2_base": trans_base,
                "clean_transition_l2_best": trans_best,
                "clean_transition_l2_ratio_best_over_base": _ratio(trans_best, trans_base),
                "phase0_aux_pxgoal_adm_l2_base": adm_base,
                "phase0_aux_pxgoal_adm_l2_best": adm_best,
                "phase0_aux_pxgoal_adm_ratio_best_over_base": _ratio(adm_best, adm_base),
                "phase0_aux_pxgoal_sprr_base": sprr_base,
                "phase0_aux_pxgoal_sprr_best": sprr_best,
                "readable_conclusion": _readable_conclusion(
                    re_base=re_base,
                    re_best=re_best,
                    rf_base=rf_base,
                    rf_best=rf_best,
                    trans_base=trans_base,
                    trans_best=trans_best,
                    adm_base=adm_base,
                    adm_best=adm_best,
                ),
            }
        )

    return {
        "metadata": {
            "schema_version": "paper1-selective-contraction-branch-0.2",
            "source_acpc_basin": _artifact_path(acpc_basin_path),
            "source_acpc_phase0": _artifact_path(acpc_phase0_path),
            "robust_metric": robust_metric,
            "method": method_name,
            "method_label": display_method,
            "robust_label": robust_label,
            "branch": (
                "existing full-sequence perturbed-target LeWM sweep"
                if method_name == "LeWM"
                else f"existing {display_method} noise sweep"
            ),
            "interpretation": (
                "RE/RF are same-state perturbation radii from the primary "
                "observation-only ACPC basin diagnostic. ADM/SPRR are auxiliary "
                "observation+goal Phase-0 proxies and should be read only as branch "
                "sanity checks, not paper-facing proof. Optional cluster plots "
                "use repeated perturbation samples and 2-D visualization "
                "envelopes; their envelopes are not high-D basin boundaries."
            ),
        },
        "rows": out_rows,
    }


def _readable_conclusion(
    *,
    re_base: float,
    re_best: float,
    rf_base: float,
    rf_best: float,
    trans_base: float,
    trans_best: float,
    adm_base: float,
    adm_best: float,
) -> str:
    same_state = (
        "same-state encoder/predictor radii contract"
        if re_best < re_base and rf_best < rf_base
        else "same-state radius contraction is not monotone"
    )
    trans = _ratio(trans_best, trans_base)
    adm = _ratio(adm_best, adm_base)
    if _finite(adm):
        if adm >= 0.95:
            disc = "auxiliary ADM is preserved"
        elif adm >= 0.8:
            disc = "auxiliary ADM mildly decreases"
        else:
            disc = "auxiliary ADM decreases"
    elif _finite(trans):
        if trans >= 0.95:
            disc = "transition scale is preserved"
        elif trans >= 0.8:
            disc = "transition scale mildly decreases"
        else:
            disc = "transition scale decreases"
    else:
        disc = "discriminability proxy unavailable"
    return f"{same_state}; {disc}."


def write_markdown(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    method_label = payload["metadata"].get("method_label", payload["metadata"].get("method", "LeWM"))
    lines = [
        "# Selective-Contraction Branch Table",
        "",
        f"Scope: existing {method_label} sweep. This is a branch diagnostic, not a new main claim.",
        "",
        "| Task | best std | obs-noise 0.08 success | encoder radius R_E | prediction radius R_F | original NN L2 | transition L2 | aux ADM | aux SPRR | read |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    metric = payload["metadata"]["robust_metric"]
    for row in payload["rows"]:
        lines.append(
            "| {task} | {std} | {succ} | {re} | {rf} | {nn} | {tr} | {adm} | {sprr} | {read} |".format(
                task=row["task"],
                std=row["best_std_key"],
                succ=_fmt_arrow(row[f"{metric}_base"], row[f"{metric}_best"], 3),
                re=_fmt_arrow(row["encoder_radius_RE_base"], row["encoder_radius_RE_best"], 3),
                rf=_fmt_arrow(row["prediction_radius_RF_base"], row["prediction_radius_RF_best"], 3),
                nn=_fmt_arrow(row["clean_nn_l2_base"], row["clean_nn_l2_best"], 3),
                tr=_fmt_arrow(row["clean_transition_l2_base"], row["clean_transition_l2_best"], 3),
                adm=_fmt_arrow(
                    row["phase0_aux_pxgoal_adm_l2_base"],
                    row["phase0_aux_pxgoal_adm_l2_best"],
                    3,
                ),
                sprr=_fmt_arrow(
                    row["phase0_aux_pxgoal_sprr_base"],
                    row["phase0_aux_pxgoal_sprr_best"],
                    3,
                ),
                read=row["readable_conclusion"],
            )
        )
    lines.extend(
        [
            "",
            "Reading: lower R_E/R_F means smaller same-state perturbation spread "
            "in the reported feature space. "
            "Higher SPRR means the auxiliary action-distance margin is larger relative "
            "to paired rollout disagreement. ADM/SPRR come from the exploratory observation+goal "
            "Phase-0 diagnostic, so they are supportive visualization/branch evidence only.",
            "",
            "Visualization note: selective-contraction cluster plots should be read through "
            "the high-D panel statistics. The 2-D t-SNE envelopes are qualitative summaries "
            "of repeated same-state perturbation samples, not estimates of the true high-D "
            "basin boundary.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _ensure_plot_deps():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    return plt


def _ensure_runtime_deps():
    from tools import paper1_phase0_acpc as phase0

    phase0._ensure_runtime_deps()
    return phase0


def _clone_batch(phase0, batch: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v.clone() if phase0.torch.is_tensor(v) else copy.deepcopy(v) for k, v in batch.items()}


def _checkpoint_specs(
    *,
    task: str,
    summary: Mapping[str, Any],
    acpc_basin_path: Path,
) -> list[CkptSpec]:
    rows = _load_json(acpc_basin_path)["rows"]
    task_rows = [r for r in rows if r.get("status") == "ok" and r["task"] == task]
    base = next(r for r in task_rows if r["std_key"] == "0.0")
    branch_row = next(r for r in summary["rows"] if r["task"] == task)
    best = next(r for r in task_rows if str(r["std_key"]) == str(branch_row["best_std_key"]))
    specs = []
    for label, row in (("base", base), ("fullseq_robust", best)):
        model_file = Path(str(row["model_file"]))
        if not model_file.exists():
            raise FileNotFoundError(f"Missing model file for {task}/{label}: {model_file}")
        specs.append(
            CkptSpec(
                label=label,
                task=task,
                std_key=str(row["std_key"]),
                subdir=str(row["subdir"]),
                model_file=model_file,
            )
        )
    return specs


def _method_slug(summary: Mapping[str, Any]) -> str:
    method = str(summary.get("metadata", {}).get("method", "LeWM")).lower()
    return "" if method == "lewm" else f"{method}_"


def _branch_slug(summary: Mapping[str, Any]) -> str:
    method = str(summary.get("metadata", {}).get("method", "LeWM"))
    return "fullseq" if method == "LeWM" else "noise"


def _display_labels(summary: Mapping[str, Any], robust_std_key: str) -> dict[str, str]:
    meta = summary.get("metadata", {})
    method = str(meta.get("method", "LeWM"))
    method_label = str(meta.get("method_label") or method)
    if method == "LeWM":
        robust = str(meta.get("robust_label") or f"noise-trained {method_label} {robust_std_key}")
        return {
            "base": f"no-noise {method_label}",
            "fullseq_robust": robust,
        }
    robust = str(meta.get("robust_label") or f"noise-trained {method_label} {robust_std_key}")
    return {
        "base": f"no-noise {method_label}",
        "fullseq_robust": robust,
    }


def _pca_fit_transform(arrays: Sequence[np.ndarray]) -> list[np.ndarray]:
    flat = np.concatenate([a.reshape(-1, a.shape[-1]) for a in arrays], axis=0)
    mean = flat.mean(axis=0, keepdims=True)
    centered = flat - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:3].T
    out = []
    for a in arrays:
        z = (a.reshape(-1, a.shape[-1]) - mean) @ components
        out.append(z.reshape(a.shape[:-1] + (3,)))
    return out


def _pca_fit_transform_2d(arrays: Sequence[np.ndarray]) -> list[np.ndarray]:
    return [a[..., :2] for a in _pca_fit_transform(arrays)]


def _nearest_original_indices(features: np.ndarray, anchors: Sequence[int]) -> dict[int, int]:
    """Nearest other original-state index for each anchor in original feature space."""
    origin = features[0].reshape(features.shape[1], features.shape[2])
    dists = np.linalg.norm(origin[:, None, :] - origin[None, :, :], axis=-1)
    np.fill_diagonal(dists, np.inf)
    return {int(i): int(np.argmin(dists[int(i)])) for i in anchors}


def _axis_limits_2d(arrays: Sequence[np.ndarray]) -> tuple[tuple[float, float], tuple[float, float]]:
    flat = np.concatenate([a.reshape(-1, 2) for a in arrays], axis=0)
    limits = []
    for dim in range(2):
        lo = float(np.nanmin(flat[:, dim]))
        hi = float(np.nanmax(flat[:, dim]))
        pad = 0.08 * max(hi - lo, 1e-6)
        limits.append((lo - pad, hi + pad))
    return limits[0], limits[1]


def _axis_limits_2d_single(array: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    return _axis_limits_2d([array])


def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    pts = sorted({(float(x), float(y)) for x, y in np.asarray(points, dtype=np.float64)})
    if len(pts) <= 1:
        return np.asarray(pts, dtype=np.float64)

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def _draw_circle_envelope(plt, ax, points: np.ndarray, color: Any, min_radius: float) -> None:
    center = points.mean(axis=0)
    radius_2d = max(float(np.linalg.norm(points - center[None, :], axis=-1).max()), 1e-6)
    radius_2d = max(radius_2d * 1.22, min_radius)
    ax.add_patch(
        plt.Circle(
            center,
            radius_2d,
            facecolor=color,
            edgecolor="none",
            alpha=0.055,
            zorder=1,
        )
    )
    ax.add_patch(
        plt.Circle(
            center,
            radius_2d,
            fill=False,
            edgecolor=color,
            alpha=0.38,
            linewidth=0.75,
            zorder=2,
        )
    )


def _draw_cluster_envelope(
    plt,
    ax,
    points: np.ndarray,
    *,
    color: Any,
    mode: str,
    coverage: float,
    min_radius: float,
) -> None:
    points = np.asarray(points, dtype=np.float64)
    if mode == "none" or points.size == 0:
        return
    if mode == "circle":
        _draw_circle_envelope(plt, ax, points, color, min_radius)
        return
    if mode == "hull":
        from matplotlib.patches import Polygon

        hull = _convex_hull_2d(points)
        if hull.shape[0] >= 3:
            ax.add_patch(
                Polygon(
                    hull,
                    closed=True,
                    facecolor=color,
                    edgecolor="none",
                    alpha=0.055,
                    zorder=1,
                )
            )
            ax.add_patch(
                Polygon(
                    hull,
                    closed=True,
                    fill=False,
                    edgecolor=color,
                    alpha=0.38,
                    linewidth=0.75,
                    zorder=2,
                )
            )
            return
        _draw_circle_envelope(plt, ax, points, color, min_radius)
        return

    from matplotlib.patches import Ellipse

    if points.shape[0] < 3:
        _draw_circle_envelope(plt, ax, points, color, min_radius)
        return
    center = points.mean(axis=0)
    cov = np.cov(points.T)
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        _draw_circle_envelope(plt, ax, points, color, min_radius)
        return
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[order], 0.0, None)
    eigvecs = eigvecs[:, order]
    if float(eigvals[0]) <= 1e-12:
        _draw_circle_envelope(plt, ax, points, color, min_radius)
        return

    q = min(max(float(coverage), 0.01), 0.999)
    scale = math.sqrt(-2.0 * math.log1p(-q))
    width = max(2.0 * scale * math.sqrt(float(eigvals[0])), 2.0 * min_radius)
    height = max(2.0 * scale * math.sqrt(float(eigvals[1])), 2.0 * min_radius)
    angle = math.degrees(math.atan2(float(eigvecs[1, 0]), float(eigvecs[0, 0])))
    ax.add_patch(
        Ellipse(
            xy=center,
            width=width,
            height=height,
            angle=angle,
            facecolor=color,
            edgecolor="none",
            alpha=0.055,
            zorder=1,
        )
    )
    ax.add_patch(
        Ellipse(
            xy=center,
            width=width,
            height=height,
            angle=angle,
            fill=False,
            edgecolor=color,
            alpha=0.38,
            linewidth=0.75,
            zorder=2,
        )
    )


def _draw_cluster_links(ax, points: np.ndarray, view_stds: Sequence[float], color: Any) -> None:
    stds = np.asarray([float(s) for s in view_stds], dtype=np.float64)
    for std in sorted({float(s) for s in stds[1:] if float(s) > 0.0}):
        group = np.flatnonzero(np.isclose(stds, std))
        group = group[group > 0]
        if group.size == 0:
            continue
        p = points[group].mean(axis=0)
        ax.plot(
            [points[0, 0], p[0]],
            [points[0, 1], p[1]],
            color=color,
            alpha=0.34,
            linewidth=0.68,
            zorder=2,
        )


def _axis_limits(arrays: Sequence[np.ndarray]) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    flat = np.concatenate([a.reshape(-1, 3) for a in arrays], axis=0)
    limits = []
    for dim in range(3):
        lo = float(np.nanmin(flat[:, dim]))
        hi = float(np.nanmax(flat[:, dim]))
        pad = 0.08 * max(hi - lo, 1e-6)
        limits.append((lo - pad, hi + pad))
    return limits[0], limits[1], limits[2]


def _pca_reduce_for_embedding(points: np.ndarray, max_dim: int = 50) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    if centered.shape[1] <= max_dim or centered.shape[0] <= max_dim:
        return centered
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:max_dim].T


def _tsne_fit_transform_2d(
    array: np.ndarray,
    *,
    seed: int,
    perplexity: float,
    max_iter: int,
) -> np.ndarray:
    from sklearn.manifold import TSNE

    flat = array.reshape(-1, array.shape[-1])
    reduced = _pca_reduce_for_embedding(flat)
    max_perplexity = max(5.0, (reduced.shape[0] - 1) / 3.0)
    px = min(float(perplexity), max_perplexity)
    px = max(5.0, min(px, reduced.shape[0] - 1.0))
    tsne = TSNE(
        n_components=2,
        perplexity=px,
        init="pca",
        learning_rate="auto",
        max_iter=max_iter,
        metric="euclidean",
        random_state=seed,
    )
    out = tsne.fit_transform(reduced)
    return out.reshape(array.shape[:-1] + (2,))


def _select_spread_anchors(origin_features: np.ndarray, count: int) -> np.ndarray:
    count = min(int(count), int(origin_features.shape[0]))
    if count <= 0:
        return np.zeros((0,), dtype=int)
    dists = np.linalg.norm(
        origin_features[:, None, :] - origin_features[None, :, :],
        axis=-1,
    )
    center = origin_features.mean(axis=0, keepdims=True)
    selected = [int(np.argmin(np.linalg.norm(origin_features - center, axis=-1)))]
    min_dist = dists[selected[0]].copy()
    while len(selected) < count:
        min_dist[selected] = -np.inf
        idx = int(np.argmax(min_dist))
        if not np.isfinite(min_dist[idx]):
            break
        selected.append(idx)
        min_dist = np.minimum(min_dist, dists[idx])
    return np.asarray(selected, dtype=int)


def _cluster_isolation_stats(array: np.ndarray) -> dict[str, float]:
    origin = array[0]
    same_view_dist = np.linalg.norm(array[1:] - origin[None, :, :], axis=-1)
    radius = np.nanmax(same_view_dist, axis=0)
    origin_dists = np.linalg.norm(origin[:, None, :] - origin[None, :, :], axis=-1)
    np.fill_diagonal(origin_dists, np.inf)
    nearest_origin = np.nanmin(origin_dists, axis=1)
    ratio = radius / np.maximum(nearest_origin, 1e-12)
    pair_ratio = (radius[:, None] + radius[None, :]) / np.maximum(origin_dists, 1e-12)
    np.fill_diagonal(pair_ratio, -np.inf)
    max_pair_ratio = np.nanmax(pair_ratio, axis=1)
    return {
        "median_radius_over_nn": float(np.nanmedian(ratio)),
        "frac_radius_lt_nn": float(np.nanmean(ratio < 1.0)),
        "frac_disjoint_balls": float(np.nanmean(max_pair_ratio < 1.0)),
        "median_radius": float(np.nanmedian(radius)),
        "median_nearest_origin": float(np.nanmedian(nearest_origin)),
    }


def _cluster_stats_title(stats: Mapping[str, float]) -> str:
    return (
        f"r/NN {stats['median_radius_over_nn']:.2f}; "
        f"r<NN {100.0 * stats['frac_radius_lt_nn']:.0f}%; "
        f"disjoint {100.0 * stats['frac_disjoint_balls']:.0f}%"
    )


def _cluster_point_counts(array: np.ndarray, anchor_count: int) -> dict[str, int]:
    view_count = int(array.shape[0])
    state_count = int(array.shape[1])
    anchors = int(anchor_count)
    return {
        "view_count_per_state": view_count,
        "sampled_state_count": state_count,
        "background_origin_points": state_count,
        "background_perturbed_points": max(0, view_count - 1) * state_count,
        "background_total_points": view_count * state_count,
        "colored_anchor_count": anchors,
        "colored_anchor_origin_points": anchors,
        "colored_anchor_perturbed_points": max(0, view_count - 1) * anchors,
        "colored_anchor_total_points": view_count * anchors,
    }


def _expanded_view_stds(view_stds: Sequence[float], perturb_repeats: int) -> list[float]:
    out = []
    repeats = max(1, int(perturb_repeats))
    saw_origin = False
    for std in view_stds:
        value = float(std)
        if value == 0.0:
            if not saw_origin:
                out.append(0.0)
                saw_origin = True
            continue
        out.extend([value] * repeats)
    if not saw_origin:
        out.insert(0, 0.0)
    return out


def _local_cluster_projection(
    array: np.ndarray,
    *,
    state_idx: int,
    neighbor_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    origin = array[0]
    center = origin[state_idx]
    dists = np.linalg.norm(origin - center[None, :], axis=-1)
    neighbor_idx = np.argsort(dists)[1 : neighbor_count + 1]
    local_points = np.concatenate(
        [array[:, state_idx, :], origin[neighbor_idx]],
        axis=0,
    )
    centered = local_points - center[None, :]
    if centered.shape[0] <= 2:
        coords = np.zeros((centered.shape[0], 2), dtype=np.float64)
    else:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        components = vt[:2]
        if components.shape[0] < 2:
            components = np.pad(components, ((0, 2 - components.shape[0]), (0, 0)))
        coords = centered @ components.T
    nearest_origin = max(float(dists[neighbor_idx[0]]) if len(neighbor_idx) else 0.0, 1e-12)
    coords = coords / nearest_origin
    return coords[: array.shape[0]], coords[array.shape[0] :]


def _draw_local_atlas_panel(
    *,
    plt,
    ax,
    array: np.ndarray,
    anchors: np.ndarray,
    colors: np.ndarray,
    title: str,
    neighbor_count: int,
) -> dict[str, float]:
    stats = _cluster_isolation_stats(array)
    cols = min(6, max(1, int(math.ceil(math.sqrt(max(1, len(anchors)) * 1.4)))))
    rows = int(math.ceil(max(1, len(anchors)) / cols))
    span = 5.0
    cell_radius = 2.15
    for slot, state_idx in enumerate(anchors):
        row = slot // cols
        col = slot % cols
        offset = np.asarray([col * span, -row * span], dtype=np.float64)
        color = colors[slot % len(colors)]
        view_coords, neighbor_coords = _local_cluster_projection(
            array,
            state_idx=int(state_idx),
            neighbor_count=neighbor_count,
        )
        ax.add_patch(
            plt.Circle(
                offset,
                1.0,
                fill=False,
                edgecolor="#666666",
                linewidth=0.55,
                alpha=0.38,
            )
        )
        ax.add_patch(
            plt.Rectangle(
                offset - cell_radius,
                2 * cell_radius,
                2 * cell_radius,
                fill=False,
                edgecolor="#E2E2E2",
                linewidth=0.45,
                alpha=0.9,
            )
        )
        if len(neighbor_coords):
            pts = neighbor_coords + offset[None, :]
            ax.scatter(pts[:, 0], pts[:, 1], s=8, c="#8A8A8A", marker="x", alpha=0.42, linewidths=0.55)
        pts = view_coords + offset[None, :]
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=0.9, alpha=0.8)
        ax.scatter(pts[0, 0], pts[0, 1], s=30, color=color, marker="o", edgecolor="#222222", linewidth=0.35)
        ax.scatter(pts[1:, 0], pts[1:, 1], s=18, color=color, marker="^", alpha=0.82, linewidth=0)

    ax.set_title(f"{title}\nhigh-D: {_cluster_stats_title(stats)}", fontsize=7.8, pad=4)
    ax.set_xlim(-cell_radius - 0.2, (cols - 1) * span + cell_radius + 0.2)
    ax.set_ylim(-(rows - 1) * span - cell_radius - 0.2, cell_radius + 0.2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    return stats


def _extract_view_features(
    *,
    phase0,
    model,
    batch: Mapping[str, Any],
    history_size: int,
    rollout_horizon: int,
    view_stds: Sequence[float],
    seed: int,
    embedding_space: str,
) -> tuple[np.ndarray, np.ndarray]:
    encoder_views = []
    predictor_views = []
    clean_outputs_for_actions = phase0.encode_sequences(model, _clone_batch(phase0, batch))
    act_emb = clean_outputs_for_actions["act_emb"].detach()
    for idx, std in enumerate(view_stds):
        if float(std) == 0.0:
            view_batch = _clone_batch(phase0, batch)
        else:
            view_batch = phase0.make_paired_noisy_batch(
                batch,
                history_size=history_size,
                noise_std=float(std),
                seed=seed + 131 * (idx + 1),
                corruption_type="gaussian_noise",
                corrupt_goal=False,
            )
        outputs = phase0.encode_sequences(model, _clone_batch(phase0, view_batch))
        emb = phase0.get_embedding_space(outputs, embedding_space).detach()
        encoder_views.append(emb[:, history_size - 1].detach().float().cpu().numpy())
        chain = phase0._autoregressive_rollout(
            model,
            emb[:, :history_size],
            act_emb,
            history_size,
            rollout_horizon,
        )
        final = chain[:, history_size + rollout_horizon - 1]
        predictor_views.append(final.detach().float().cpu().numpy())
    return np.stack(encoder_views, axis=0), np.stack(predictor_views, axis=0)


def _load_task_features(
    *,
    task: str,
    summary: Mapping[str, Any],
    acpc_basin_path: Path,
    n_sequences: int,
    view_stds: Sequence[float],
    rollout_horizon: int,
    seed: int,
    device: str | None,
    img_size: int,
    frameskip: int,
) -> tuple[dict[str, dict[str, np.ndarray]], list[CkptSpec]]:
    phase0 = _ensure_runtime_deps()
    device_value = device or "cpu"
    specs = _checkpoint_specs(task=task, summary=summary, acpc_basin_path=acpc_basin_path)

    encoded: dict[str, dict[str, np.ndarray]] = {}
    batch_cache: dict[tuple[int, int], Mapping[str, Any]] = {}
    for spec in specs:
        with phase0.torch.no_grad():
            model = phase0.load_model(str(spec.model_file), device_value)
            history_size = phase0.infer_history_size(model)
            future_steps = max(rollout_horizon + 1, 9)
            batch_key = (history_size, future_steps)
            if batch_key not in batch_cache:
                batch_cache[batch_key] = phase0.load_dataset_samples(
                    dataset_name=TASK_DATASETS[task],
                    state_key=None,
                    n_sequences=n_sequences,
                    history_size=history_size,
                    future_steps=future_steps,
                    frameskip=frameskip,
                    img_size=img_size,
                    seed=seed,
                    device=device_value,
                )
            batch = batch_cache[batch_key]
            spaces = phase0.get_model_spaces(model)
            embedding_space = spaces["inference_cost_space"]
            enc, pred = _extract_view_features(
                phase0=phase0,
                model=model,
                batch=batch,
                history_size=history_size,
                rollout_horizon=rollout_horizon,
                view_stds=view_stds,
                seed=seed,
                embedding_space=embedding_space,
            )
            encoded[spec.label] = {"encoder": enc, "predictor": pred}
    return encoded, specs


def render_2d_task(
    *,
    task: str,
    summary: Mapping[str, Any],
    acpc_basin_path: Path,
    out_dir: Path,
    n_sequences: int,
    view_stds: Sequence[float],
    rollout_horizon: int,
    seed: int,
    device: str | None,
    img_size: int,
    frameskip: int,
    anchor_count: int,
) -> Path:
    plt = _ensure_plot_deps()
    encoded, specs = _load_task_features(
        task=task,
        summary=summary,
        acpc_basin_path=acpc_basin_path,
        n_sequences=n_sequences,
        view_stds=view_stds,
        rollout_horizon=rollout_horizon,
        seed=seed,
        device=device,
        img_size=img_size,
        frameskip=frameskip,
    )

    enc_pca = _pca_fit_transform_2d([encoded["base"]["encoder"], encoded["fullseq_robust"]["encoder"]])
    pred_pca = _pca_fit_transform_2d([encoded["base"]["predictor"], encoded["fullseq_robust"]["predictor"]])
    encoded["base"]["encoder_2d"], encoded["fullseq_robust"]["encoder_2d"] = enc_pca
    encoded["base"]["predictor_2d"], encoded["fullseq_robust"]["predictor_2d"] = pred_pca
    axis_limits = {
        "encoder_2d": _axis_limits_2d([encoded["base"]["encoder_2d"], encoded["fullseq_robust"]["encoder_2d"]]),
        "predictor_2d": _axis_limits_2d([encoded["base"]["predictor_2d"], encoded["fullseq_robust"]["predictor_2d"]]),
    }

    anchor_count = min(anchor_count, n_sequences)
    anchors = np.linspace(0, n_sequences - 1, anchor_count, dtype=int)
    if anchor_count > 0:
        anchors = np.unique(anchors)
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(anchors))))
    label_by_spec = _display_labels(summary, specs[1].std_key)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5), sharex="col", sharey="col")
    panels = [
        ("base", "encoder_2d", "Encoder features"),
        ("base", "predictor_2d", "Predictor H8 features"),
        ("fullseq_robust", "encoder_2d", "Encoder features"),
        ("fullseq_robust", "predictor_2d", "Predictor H8 features"),
    ]
    for ax, (label, feature, title) in zip(axes.reshape(-1), panels):
        arr = encoded[label][feature]
        origin = arr[0]
        xlim, ylim = axis_limits[feature]
        ax.scatter(origin[:, 0], origin[:, 1], s=9, c="#C7C7C7", alpha=0.38, linewidths=0)
        for ci, state_idx in enumerate(anchors):
            color = colors[ci % len(colors)]
            pts = arr[:, state_idx, :]
            ax.plot(pts[:, 0], pts[:, 1], color=color, alpha=0.75, linewidth=1.0)
            ax.scatter(pts[0, 0], pts[0, 1], s=46, color=color, marker="o", edgecolor="#222222", linewidth=0.45)
            ax.scatter(pts[1:, 0], pts[1:, 1], s=24, color=color, marker="^", alpha=0.78, linewidth=0)
        ax.set_title(f"{label_by_spec[label]}: {title}")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, color="#E7E7E7", linewidth=0.6)
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle(
        f"{task}: original states and same-state perturbation clusters "
        f"(n={n_sequences}, anchors={len(anchors)}, view stds={','.join(f'{s:g}' for s in view_stds)})",
        y=0.98,
    )
    fig.text(
        0.5,
        0.025,
        "Gray points: many original states. Colored circle: selected original state. "
        "Same-color triangles/lines: its perturbed views. Shorter colored clusters mean stronger same-state contraction.",
        ha="center",
        va="bottom",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.96))
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{task.lower()}_{_method_slug(summary)}{_branch_slug(summary)}_selective_contraction_2d.png"
    fig.savefig(out, dpi=190)
    plt.close(fig)
    return out


def render_atlas_task(
    *,
    task: str,
    summary: Mapping[str, Any],
    acpc_basin_path: Path,
    out_dir: Path,
    n_sequences: int,
    view_stds: Sequence[float],
    rollout_horizon: int,
    seed: int,
    device: str | None,
    img_size: int,
    frameskip: int,
    anchor_count: int,
    neighbor_count: int,
) -> Path:
    plt = _ensure_plot_deps()
    encoded, specs = _load_task_features(
        task=task,
        summary=summary,
        acpc_basin_path=acpc_basin_path,
        n_sequences=n_sequences,
        view_stds=view_stds,
        rollout_horizon=rollout_horizon,
        seed=seed,
        device=device,
        img_size=img_size,
        frameskip=frameskip,
    )

    anchors = _select_spread_anchors(encoded["fullseq_robust"]["predictor"][0], anchor_count)
    colors = plt.cm.turbo(np.linspace(0.05, 0.95, max(1, len(anchors))))
    label_by_spec = _display_labels(summary, specs[1].std_key)
    feature_by_name = {
        "encoder": "Encoder",
        "predictor": "Predictor H8",
    }

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 7.2))
    panels = [
        ("base", "encoder"),
        ("base", "predictor"),
        ("fullseq_robust", "encoder"),
        ("fullseq_robust", "predictor"),
    ]
    for ax, (label, feature) in zip(axes.reshape(-1), panels):
        _draw_local_atlas_panel(
            plt=plt,
            ax=ax,
            array=encoded[label][feature],
            anchors=anchors,
            colors=colors,
            title=f"{label_by_spec[label]}: {feature_by_name[feature]} local clusters",
            neighbor_count=neighbor_count,
        )

    fig.suptitle(
        f"{task}: local cluster atlas normalized by nearest original-state distance "
        f"(n={n_sequences}, anchors={len(anchors)}, neighbors={neighbor_count})",
        y=0.975,
        fontsize=9.2,
    )
    fig.text(
        0.5,
        0.012,
        "Each box is one original state; circle radius is its nearest different-state distance in high-D space.",
        ha="center",
        va="bottom",
        fontsize=6.8,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.945), h_pad=1.6, w_pad=0.8)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{task.lower()}_{_method_slug(summary)}{_branch_slug(summary)}_selective_contraction_atlas.png"
    fig.savefig(out, dpi=320)
    plt.close(fig)
    return out


def render_cluster_task(
    *,
    task: str,
    summary: Mapping[str, Any],
    acpc_basin_path: Path,
    out_dir: Path,
    n_sequences: int,
    view_stds: Sequence[float],
    rollout_horizon: int,
    seed: int,
    device: str | None,
    img_size: int,
    frameskip: int,
    anchor_count: int,
    perplexity: float,
    tsne_max_iter: int,
    perturb_repeats: int,
    envelope: str,
    envelope_coverage: float,
    anchor_selection: str,
) -> Path:
    plt = _ensure_plot_deps()
    cluster_view_stds = _expanded_view_stds(view_stds, perturb_repeats)
    encoded, specs = _load_task_features(
        task=task,
        summary=summary,
        acpc_basin_path=acpc_basin_path,
        n_sequences=n_sequences,
        view_stds=cluster_view_stds,
        rollout_horizon=rollout_horizon,
        seed=seed,
        device=device,
        img_size=img_size,
        frameskip=frameskip,
    )

    label_by_spec = _display_labels(summary, specs[1].std_key)
    feature_by_name = {
        "encoder": "Encoder features",
        "predictor": "Predictor H8 features",
    }
    panels = [
        ("base", "encoder"),
        ("base", "predictor"),
        ("fullseq_robust", "encoder"),
        ("fullseq_robust", "predictor"),
    ]
    if anchor_selection == "random":
        n_states = int(encoded["fullseq_robust"]["predictor"].shape[1])
        count = min(int(anchor_count), n_states)
        rng_seed = int(seed) + 1009
        rng = np.random.default_rng(rng_seed)
        anchors = np.sort(rng.choice(n_states, size=count, replace=False)).astype(int)
        anchor_selection_meta = {
            "strategy": "fixed_seed_random",
            "seed": rng_seed,
            "selected": [int(x) for x in anchors.tolist()],
            "note": "Paper-facing anchors are a fixed-seed random subset; t-SNE and high-D statistics are not used for selection.",
        }
    elif anchor_selection == "spread":
        anchors = _select_spread_anchors(encoded["fullseq_robust"]["predictor"][0], anchor_count)
        anchor_selection_meta = {
            "strategy": "spread",
            "selected": [int(x) for x in anchors.tolist()],
            "note": "Legacy farthest-point anchor selection in robust predictor high-D space.",
        }
    else:
        raise ValueError(f"Unknown cluster anchor selection: {anchor_selection}")
    colors = plt.cm.turbo(np.linspace(0.05, 0.95, max(1, len(anchors))))

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 7.2))
    sample_shapes = {
        f"{label}:{feature}": encoded[label][feature].shape[:2]
        for label, feature in panels
    }
    if len(set(sample_shapes.values())) != 1:
        raise ValueError(f"cluster panels use inconsistent sample counts: {sample_shapes}")
    panel_point_counts = []
    for panel_idx, (ax, (label, feature)) in enumerate(zip(axes.reshape(-1), panels)):
        arr = encoded[label][feature]
        panel_point_counts.append(
            {
                "panel": f"{label}:{feature}",
                "row_label": label,
                "feature": feature,
                **_cluster_point_counts(arr, len(anchors)),
            }
        )
        projected = _tsne_fit_transform_2d(
            arr,
            seed=seed + 17 * (panel_idx + 1),
            perplexity=perplexity,
            max_iter=tsne_max_iter,
        )
        stats = _cluster_isolation_stats(arr)
        origin = projected[0]
        perturbed = projected[1:]
        xlim, ylim = _axis_limits_2d_single(projected)
        min_envelope_radius = 0.018 * max(xlim[1] - xlim[0], ylim[1] - ylim[0])

        ax.scatter(origin[:, 0], origin[:, 1], s=8, c="#7F7F7F", alpha=0.24, linewidths=0)
        ax.scatter(
            perturbed.reshape(-1, 2)[:, 0],
            perturbed.reshape(-1, 2)[:, 1],
            s=5,
            c="#BFBFBF",
            alpha=0.11,
            linewidths=0,
        )

        for ci, state_idx in enumerate(anchors):
            color = colors[ci % len(colors)]
            pts = projected[:, state_idx, :]
            _draw_cluster_envelope(
                plt,
                ax,
                pts,
                color=color,
                mode=envelope,
                coverage=envelope_coverage,
                min_radius=min_envelope_radius,
            )
            _draw_cluster_links(ax, pts, cluster_view_stds, color)
            ax.scatter(
                pts[1:, 0],
                pts[1:, 1],
                s=18,
                color=color,
                marker="o",
                alpha=0.78,
                edgecolor="white",
                linewidth=0.18,
            )
            ax.scatter(
                pts[0, 0],
                pts[0, 1],
                s=54,
                color=color,
                marker="o",
                edgecolor="#111111",
                linewidth=0.52,
                zorder=4,
            )

        ax.set_title(
            f"{label_by_spec[label]}: {feature_by_name[feature]}\n"
            f"high-D: {_cluster_stats_title(stats)}",
            fontsize=7.8,
            pad=4,
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("t-SNE 1", fontsize=7.4)
        ax.set_ylabel("t-SNE 2", fontsize=7.4)
        ax.tick_params(labelsize=6.8, pad=1)
        ax.grid(True, color="#EEEEEE", linewidth=0.45)
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle(
        f"{task}: same-state perturbation clusters in encoder and H8 predictor spaces",
        y=0.975,
        fontsize=9.2,
    )
    envelope_note = {
        "ellipse": (
            f"colored ellipses are {100.0 * envelope_coverage:.0f}% covariance envelopes "
            "in the t-SNE plane"
        ),
        "hull": "colored hulls are sample convex hulls in the t-SNE plane",
        "circle": "colored circles use the legacy max-distance envelope in the t-SNE plane",
        "none": "no colored envelope is drawn",
    }[envelope]
    fig.text(
        0.5,
        0.012,
        "t-SNE is visualization only; panel annotations are computed in high-D space.\n"
        f"Gray dots show sampled views; {envelope_note}.",
        ha="center",
        va="bottom",
        fontsize=6.6,
        linespacing=1.18,
    )
    fig.tight_layout(rect=(0, 0.075, 1, 0.945), h_pad=2.0, w_pad=0.9)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{task.lower()}_{_method_slug(summary)}{_branch_slug(summary)}_selective_contraction_clusters.png"
    fig.savefig(out, dpi=320)
    point_counts_out = out.with_name(f"{out.stem}_point_counts.json")
    _write_json(
        point_counts_out,
        {
            "schema_version": "paper1-selective-contraction-cluster-point-counts-0.1",
            "figure": out.name,
            "task": task,
            "method": str(summary.get("metadata", {}).get("method", "")),
            "n_sequences": int(n_sequences),
            "view_stds": [float(x) for x in view_stds],
            "expanded_view_stds": [float(x) for x in cluster_view_stds],
            "perturb_repeats": int(perturb_repeats),
            "rollout_horizon": int(rollout_horizon),
            "seed": int(seed),
            "anchor_indices": [int(x) for x in anchors.tolist()],
            "anchor_selection": anchor_selection_meta,
            "panels": panel_point_counts,
            "note": (
                "All four cluster panels must have identical view_count_per_state "
                "and sampled_state_count. Colored anchor points are overlaid on the "
                "same background sample, so contraction can make the lower-row points "
                "visually overlap even when the counts match."
            ),
        },
    )
    plt.close(fig)
    return out


def render_3d_task(
    *,
    task: str,
    summary: Mapping[str, Any],
    acpc_basin_path: Path,
    out_dir: Path,
    n_sequences: int,
    view_stds: Sequence[float],
    rollout_horizon: int,
    seed: int,
    device: str | None,
    img_size: int,
    frameskip: int,
    anchor_count: int,
) -> Path:
    plt = _ensure_plot_deps()
    encoded, specs = _load_task_features(
        task=task,
        summary=summary,
        acpc_basin_path=acpc_basin_path,
        n_sequences=n_sequences,
        view_stds=view_stds,
        rollout_horizon=rollout_horizon,
        seed=seed,
        device=device,
        img_size=img_size,
        frameskip=frameskip,
    )

    enc_pca = _pca_fit_transform([encoded["base"]["encoder"], encoded["fullseq_robust"]["encoder"]])
    pred_pca = _pca_fit_transform([encoded["base"]["predictor"], encoded["fullseq_robust"]["predictor"]])
    encoded["base"]["encoder_3d"], encoded["fullseq_robust"]["encoder_3d"] = enc_pca
    encoded["base"]["predictor_3d"], encoded["fullseq_robust"]["predictor_3d"] = pred_pca
    nearest_indices = {
        "base": {
            "encoder_3d": _nearest_original_indices(encoded["base"]["encoder"], []),
            "predictor_3d": _nearest_original_indices(encoded["base"]["predictor"], []),
        },
        "fullseq_robust": {
            "encoder_3d": _nearest_original_indices(encoded["fullseq_robust"]["encoder"], []),
            "predictor_3d": _nearest_original_indices(encoded["fullseq_robust"]["predictor"], []),
        },
    }
    axis_limits = {
        "encoder_3d": _axis_limits([encoded["base"]["encoder_3d"], encoded["fullseq_robust"]["encoder_3d"]]),
        "predictor_3d": _axis_limits([encoded["base"]["predictor_3d"], encoded["fullseq_robust"]["predictor_3d"]]),
    }

    rng = np.random.default_rng(seed)
    anchor_count = min(anchor_count, n_sequences)
    anchors = np.linspace(0, n_sequences - 1, anchor_count, dtype=int)
    if anchor_count > 0:
        anchors = np.unique(anchors)
    for label in ("base", "fullseq_robust"):
        nearest_indices[label]["encoder_3d"] = _nearest_original_indices(
            encoded[label]["encoder"], anchors
        )
        nearest_indices[label]["predictor_3d"] = _nearest_original_indices(
            encoded[label]["predictor"], anchors
        )
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(anchors))))
    label_by_spec = _display_labels(summary, specs[1].std_key)

    fig = plt.figure(figsize=(12, 9))
    panels = [
        ("base", "encoder_3d", "Encoder"),
        ("base", "predictor_3d", "Predictor H8"),
        ("fullseq_robust", "encoder_3d", "Encoder"),
        ("fullseq_robust", "predictor_3d", "Predictor H8"),
    ]
    for i, (label, feature, title) in enumerate(panels, start=1):
        ax = fig.add_subplot(2, 2, i, projection="3d")
        arr = encoded[label][feature]
        origin = arr[0]
        ax.scatter(origin[:, 0], origin[:, 1], origin[:, 2], s=8, c="#999999", alpha=0.22, depthshade=False)
        for ci, state_idx in enumerate(anchors):
            color = colors[ci % len(colors)]
            pts = arr[:, state_idx, :]
            nn_idx = nearest_indices[label][feature][int(state_idx)]
            nn_pt = origin[nn_idx]
            origin_pt = pts[0]
            ax.plot(
                [origin_pt[0], nn_pt[0]],
                [origin_pt[1], nn_pt[1]],
                [origin_pt[2], nn_pt[2]],
                color="#222222",
                alpha=0.35,
                linewidth=0.9,
                linestyle="--",
            )
            ax.scatter(
                nn_pt[0:1],
                nn_pt[1:2],
                nn_pt[2:3],
                s=34,
                color="#222222",
                marker="x",
                alpha=0.72,
                depthshade=False,
            )
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, alpha=0.7, linewidth=1.0)
            ax.scatter(pts[0:1, 0], pts[0:1, 1], pts[0:1, 2], s=42, color=[color], marker="o", depthshade=False)
            ax.scatter(pts[1:, 0], pts[1:, 1], pts[1:, 2], s=24, color=[color], marker="^", alpha=0.78, depthshade=False)
        ax.set_title(f"{label_by_spec[label]}: {title}", pad=10)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        xlim, ylim, zlim = axis_limits[feature]
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        ax.view_init(elev=22, azim=38)
    fig.suptitle(
        f"{task}: original-state points and same-state perturbation clusters "
        f"(view stds={','.join(f'{s:g}' for s in view_stds)})",
        y=0.98,
    )
    fig.text(
        0.5,
        0.02,
        "Gray dots: original-view states. Colored circles: selected originals. "
        "Colored triangles/lines: same-state perturbed views. Black x/dashed line: nearest other original state.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{task.lower()}_{_method_slug(summary)}{_branch_slug(summary)}_selective_contraction_3d.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build Paper 1 selective-contraction branch diagnostics.")
    p.add_argument("--acpc-basin", type=Path, default=DEFAULT_DATA_DIR / "acpc_basin_diagnostics.json")
    p.add_argument("--acpc-phase0", type=Path, default=DEFAULT_DATA_DIR / "acpc_phase0_diagnostics.json")
    p.add_argument("--robust-metric", default="pixels_std0.08_success")
    p.add_argument("--method", choices=["LeWM", "PLDM"], default="LeWM")
    p.add_argument("--method-label", default=None)
    p.add_argument("--robust-label", default=None)
    p.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    p.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    p.add_argument("--plot-2d", action="store_true")
    p.add_argument("--plot-3d", action="store_true")
    p.add_argument("--plot-clusters", action="store_true")
    p.add_argument("--plot-atlas", action="store_true")
    p.add_argument("--plot-tasks", nargs="+", choices=TASKS, default=["PushT"])
    p.add_argument("--plot-out-dir", type=Path, default=DEFAULT_FIG_DIR)
    p.add_argument("--plot2d-out-dir", type=Path, default=DEFAULT_FIG2D_DIR)
    p.add_argument("--cluster-out-dir", type=Path, default=DEFAULT_CLUSTER_DIR)
    p.add_argument("--atlas-out-dir", type=Path, default=DEFAULT_ATLAS_DIR)
    p.add_argument("--n-sequences", type=int, default=160)
    p.add_argument("--view-stds", nargs="+", type=float, default=[0.0, 0.02, 0.04, 0.08])
    p.add_argument("--rollout-horizon", type=int, default=8)
    p.add_argument("--seed", type=int, default=3072)
    p.add_argument("--device", default=None, help="Default: cpu, to avoid interfering with active training.")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--frameskip", type=int, default=5)
    p.add_argument("--anchor-count", type=int, default=8)
    p.add_argument("--cluster-anchor-count", type=int, default=24)
    p.add_argument("--cluster-perplexity", type=float, default=35.0)
    p.add_argument("--cluster-tsne-max-iter", type=int, default=650)
    p.add_argument("--cluster-perturb-repeats", type=int, default=6)
    p.add_argument("--cluster-envelope", choices=["ellipse", "hull", "circle", "none"], default="ellipse")
    p.add_argument("--cluster-envelope-coverage", type=float, default=0.90)
    p.add_argument("--cluster-anchor-selection", choices=["random", "spread"], default="random")
    p.add_argument("--atlas-anchor-count", type=int, default=24)
    p.add_argument("--atlas-neighbor-count", type=int, default=8)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.out_json, args.out_md = _resolve_summary_paths(args.method, args.out_json, args.out_md)
    summary = build_summary(
        acpc_basin_path=args.acpc_basin,
        acpc_phase0_path=args.acpc_phase0,
        robust_metric=args.robust_metric,
        method=args.method,
        method_label=args.method_label,
        robust_label=args.robust_label,
    )
    _write_json(args.out_json, summary)
    write_markdown(args.out_md, summary)
    print(f"[selective-contraction] wrote {args.out_json}")
    print(f"[selective-contraction] wrote {args.out_md}")

    if args.plot_2d:
        for task in args.plot_tasks:
            out = render_2d_task(
                task=task,
                summary=summary,
                acpc_basin_path=args.acpc_basin,
                out_dir=args.plot2d_out_dir,
                n_sequences=args.n_sequences,
                view_stds=args.view_stds,
                rollout_horizon=args.rollout_horizon,
                seed=args.seed,
                device=args.device,
                img_size=args.img_size,
                frameskip=args.frameskip,
                anchor_count=args.anchor_count,
            )
            print(f"[selective-contraction] wrote {out}")

    if args.plot_atlas:
        for task in args.plot_tasks:
            out = render_atlas_task(
                task=task,
                summary=summary,
                acpc_basin_path=args.acpc_basin,
                out_dir=args.atlas_out_dir,
                n_sequences=args.n_sequences,
                view_stds=args.view_stds,
                rollout_horizon=args.rollout_horizon,
                seed=args.seed,
                device=args.device,
                img_size=args.img_size,
                frameskip=args.frameskip,
                anchor_count=args.atlas_anchor_count,
                neighbor_count=args.atlas_neighbor_count,
            )
            print(f"[selective-contraction] wrote {out}")

    if args.plot_clusters:
        for task in args.plot_tasks:
            out = render_cluster_task(
                task=task,
                summary=summary,
                acpc_basin_path=args.acpc_basin,
                out_dir=args.cluster_out_dir,
                n_sequences=args.n_sequences,
                view_stds=args.view_stds,
                rollout_horizon=args.rollout_horizon,
                seed=args.seed,
                device=args.device,
                img_size=args.img_size,
                frameskip=args.frameskip,
                anchor_count=args.cluster_anchor_count,
                perplexity=args.cluster_perplexity,
                tsne_max_iter=args.cluster_tsne_max_iter,
                perturb_repeats=args.cluster_perturb_repeats,
                envelope=args.cluster_envelope,
                envelope_coverage=args.cluster_envelope_coverage,
                anchor_selection=args.cluster_anchor_selection,
            )
            print(f"[selective-contraction] wrote {out}")

    if args.plot_3d:
        for task in args.plot_tasks:
            out = render_3d_task(
                task=task,
                summary=summary,
                acpc_basin_path=args.acpc_basin,
                out_dir=args.plot_out_dir,
                n_sequences=args.n_sequences,
                view_stds=args.view_stds,
                rollout_horizon=args.rollout_horizon,
                seed=args.seed,
                device=args.device,
                img_size=args.img_size,
                frameskip=args.frameskip,
                anchor_count=args.anchor_count,
            )
            print(f"[selective-contraction] wrote {out}")


if __name__ == "__main__":
    main()
