"""
noise_sensitivity.py - Noise robustness diagnostics for LeWM / SWM embeddings.

References for the indicators emitted here:
    - `clean_effective_rank`: matrix entropy of singular values.
        Garrido et al., "RankMe: Assessing the downstream performance of
        pretrained self-supervised representations by their rank", ICML 2023.
    - `clean_pair_cos_dist_*`: aggregate pairwise distance is the same primitive
        as Wang & Isola's `L_uniform` (sphere uniformity loss).
        Wang & Isola, "Understanding Contrastive Representation Learning
        through Alignment and Uniformity on the Hypersphere", ICML 2020.
    - `clean_nn_cos_dist_*` / `clean_nn_l2_*`: latent nearest-neighbor distance
        is the same primitive used in KNN-based OOD scoring.
        Sun et al., "Out-of-Distribution Detection with Deep Nearest
        Neighbors", NeurIPS 2022; Liu et al., "SNGP", NeurIPS 2020.
    - `noise_l2_*` / `noise_cos_*` / `noise_angle_deg_*`: empirical input-noise
        Jacobian / Lipschitz probing.
        Virmaux & Scaman, "Lipschitz regularity of deep neural networks",
        NeurIPS 2018; Hoffman et al., "Robust Learning with Jacobian
        Regularization", arXiv:1908.02729 (2019); Cohen et al.,
        "Certified Adversarial Robustness via Randomized Smoothing", ICML 2019.
    - `cka_linear_clean_vs_noisy`: Centered Kernel Alignment between clean
        and noisy embedding subspaces.
        Kornblith et al., "Similarity of Neural Network Representations
        Revisited", ICML 2019.
    - `noise_to_nn_cos_ratio_*`, `robust_radius_std`,
        `noise_angle_slope_deg_per_std`: composite ratios introduced by this
        toolkit; see inline metric definitions for novelty discussion.

Notebook use:

    from tools.repr_analysis.noise_sensitivity import run_noise_sensitivity, format_noise_table

    rows = run_noise_sensitivity(
        models={"swm": "/path/to/swm/model_object.ckpt", "lewm": "/path/to/lewm/model_object.ckpt"},
        dataset="tworoom",
        stds=[0.0, 0.01, 0.03, 0.05],
    )
    format_noise_table(rows)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch
import torch.nn.functional as F

from tools.repr_analysis.analyze_repr import (
    effective_rank,
    encode_sequences,
    get_embedding_space,
    get_model_spaces,
    infer_history_size,
    load_dataset_samples,
    load_model,
    resolve_space_name,
    to_serializable,
)
from utils import make_eval_corruption


def _clone_batch(batch: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}


def _add_eval_corruption(x: torch.Tensor, magnitude: float, seed: int,
                         corruption_type: str = "gaussian_noise") -> torch.Tensor:
    """Apply the configured corruption (gaussian_noise / gaussian_blur
    / resize) at a single magnitude, deterministically seeded so a given
    probe point is reproducible across runs."""
    transform = make_eval_corruption(magnitude, corruption_type)
    if transform is None:
        return x.clone()
    with torch.random.fork_rng(devices=[x.device] if x.device.type == "cuda" else []):
        torch.manual_seed(seed)
        return transform(x)


def _select_frames(z: torch.Tensor, frame_scope: str) -> torch.Tensor:
    if frame_scope == "goal":
        return z[:, -1]
    if frame_scope == "history":
        # all frames except the last (predictor input scope; aligns with pixels-only failure)
        return z[:, :-1].reshape(-1, z.size(-1))
    if frame_scope == "all":
        return z.reshape(-1, z.size(-1))
    raise ValueError(f"Unsupported frame_scope: {frame_scope}")


def _safe_quantile(x: torch.Tensor, q: float) -> float:
    if x.numel() == 0:
        return float("nan")
    return float(torch.quantile(x.float().cpu(), q))


def _pairwise_reference(z: torch.Tensor) -> Dict[str, float]:
    # NN distance: KNN-OOD primitive (Sun et al., NeurIPS 2022; SNGP, Liu 2020).
    # Pair distance median: discrete analogue of Wang & Isola L_uniform (ICML 2020).
    # Effective rank (matrix entropy of SVs): RankMe (Garrido et al., ICML 2023).
    if z.size(0) < 2:
        return {
            "clean_nn_cos_dist_median": float("nan"),
            "clean_pair_cos_dist_median": float("nan"),
            "clean_nn_l2_median": float("nan"),
            "clean_pair_l2_median": float("nan"),
            "clean_effective_rank": float("nan"),
        }

    z_norm = F.normalize(z, dim=-1, eps=1e-8)
    cos_dist = 1.0 - z_norm @ z_norm.T
    l2_dist = torch.cdist(z, z, p=2)
    eye = torch.eye(z.size(0), dtype=torch.bool, device=z.device)

    cos_offdiag = cos_dist[~eye].clamp_min(0.0)
    l2_offdiag = l2_dist[~eye]
    cos_nn = cos_dist.masked_fill(eye, float("inf")).min(dim=1).values.clamp_min(0.0)
    l2_nn = l2_dist.masked_fill(eye, float("inf")).min(dim=1).values

    return {
        "clean_nn_cos_dist_median": _safe_quantile(cos_nn, 0.5),
        "clean_pair_cos_dist_median": _safe_quantile(cos_offdiag, 0.5),
        "clean_nn_l2_median": _safe_quantile(l2_nn, 0.5),
        "clean_pair_l2_median": _safe_quantile(l2_offdiag, 0.5),
        # matrix entropy from singular values; disambiguates clustered vs collapsed
        "clean_effective_rank": effective_rank(z),
    }


def _linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA between two (N, D) feature matrices.

    Centered Kernel Alignment, Kornblith et al. (ICML 2019). Captures global
    subspace alignment between clean and noisy embeddings — complements
    per-point shift metrics, which can miss the case where the cloud is
    shifted/rotated as a whole but its internal geometry is preserved.

    Returns 1.0 for identical, 0.0 for orthogonal subspaces.
    """
    if X.numel() == 0 or Y.numel() == 0 or X.size(0) < 2:
        return float("nan")
    Xc = X - X.mean(0, keepdim=True)
    Yc = Y - Y.mean(0, keepdim=True)
    XtY = Xc.T @ Yc
    XtX = Xc.T @ Xc
    YtY = Yc.T @ Yc
    num = float((XtY * XtY).sum())
    denom = float(torch.sqrt(((XtX * XtX).sum() * (YtY * YtY).sum()).clamp_min(1e-24)))
    if denom < 1e-12:
        return float("nan")
    return num / denom


def _shift_metrics(clean: torch.Tensor, noisy: torch.Tensor) -> Dict[str, float]:
    # angle / l2 / cos shift: Monte-Carlo finite-difference Lipschitz probe at
    # the data manifold; conceptually equivalent to Hoffman et al. (2019)
    # Jacobian regularization probes and Virmaux & Scaman (NeurIPS 2018) local
    # Lipschitz, restricted to Gaussian pixel noise (Cohen et al., ICML 2019).
    clean_norm = F.normalize(clean, dim=-1, eps=1e-8)
    noisy_norm = F.normalize(noisy, dim=-1, eps=1e-8)
    cos = (clean_norm * noisy_norm).sum(dim=-1).clamp(-1.0, 1.0)
    cos_dist = (1.0 - cos).clamp_min(0.0)
    angle_deg = torch.rad2deg(torch.acos(cos))
    l2_shift = torch.linalg.vector_norm(noisy - clean, dim=-1)

    return {
        "noise_cos_sim_mean": float(cos.mean()),
        "noise_cos_dist_median": _safe_quantile(cos_dist, 0.5),
        "noise_cos_dist_p90": _safe_quantile(cos_dist, 0.9),
        "noise_angle_deg_median": _safe_quantile(angle_deg, 0.5),
        "noise_angle_deg_p90": _safe_quantile(angle_deg, 0.9),
        "noise_l2_median": _safe_quantile(l2_shift, 0.5),
        "noise_l2_p90": _safe_quantile(l2_shift, 0.9),
        "clean_norm_mean": float(torch.linalg.vector_norm(clean, dim=-1).mean()),
        "noisy_norm_mean": float(torch.linalg.vector_norm(noisy, dim=-1).mean()),
        "cka_linear_clean_vs_noisy": _linear_cka(clean, noisy),
    }


def _risk_label(ratio_median: float, ratio_p90: float) -> str:
    if ratio_median >= 1.0 or ratio_p90 >= 2.0:
        return "high"
    if ratio_median >= 0.5 or ratio_p90 >= 1.0:
        return "medium"
    return "low"


def analyze_model_noise(
    *,
    label: str,
    ckpt: str,
    batch: Mapping[str, torch.Tensor],
    stds: Sequence[float],
    embedding_space: str | None = None,
    seed: int = 3072,
    device: str = "cuda",
    corruption_type: str = "gaussian_noise",
) -> list[Dict[str, Any]]:
    """``stds`` carries the corruption magnitudes regardless of family
    (``std`` for ``gaussian_noise``, ``kernel_size`` for
    ``gaussian_blur``, ``factor`` for ``resize``). The variable name is
    kept for back-compat with the original noise-only API."""
    model = load_model(ckpt, device)
    spaces = get_model_spaces(model)
    space = resolve_space_name(embedding_space or spaces["inference_cost_space"])

    clean_outputs = encode_sequences(model, _clone_batch(batch))
    clean_z = get_embedding_space(clean_outputs, space).detach()

    rows: list[Dict[str, Any]] = []
    for std_idx, std in enumerate(stds):
        noisy_batch = _clone_batch(batch)
        noisy_batch["pixels"] = _add_eval_corruption(
            noisy_batch["pixels"], float(std),
            seed + 1009 * std_idx, corruption_type=corruption_type,
        )
        noisy_outputs = encode_sequences(model, noisy_batch)
        noisy_z = get_embedding_space(noisy_outputs, space).detach()

        for frame_scope in ("goal", "history", "all"):
            clean_frame = _select_frames(clean_z, frame_scope)
            noisy_frame = _select_frames(noisy_z, frame_scope)
            shift = _shift_metrics(clean_frame, noisy_frame)
            ref = _pairwise_reference(clean_frame)

            cos_ratio = (
                shift["noise_cos_dist_median"] / ref["clean_nn_cos_dist_median"]
                if ref["clean_nn_cos_dist_median"] > 0
                else float("nan")
            )
            l2_ratio = (
                shift["noise_l2_median"] / ref["clean_nn_l2_median"]
                if ref["clean_nn_l2_median"] > 0
                else float("nan")
            )
            cos_ratio_p90 = (
                shift["noise_cos_dist_p90"] / ref["clean_nn_cos_dist_median"]
                if ref["clean_nn_cos_dist_median"] > 0
                else float("nan")
            )

            rows.append(
                {
                    "model": label,
                    "ckpt": ckpt,
                    "std": float(std),
                    "frame_scope": frame_scope,
                    "embedding_space": space,
                    "n_points": int(clean_frame.size(0)),
                    **shift,
                    **ref,
                    "noise_to_nn_cos_ratio_median": float(cos_ratio),
                    "noise_to_nn_cos_ratio_p90": float(cos_ratio_p90),
                    "noise_to_nn_l2_ratio_median": float(l2_ratio),
                    "risk": _risk_label(float(cos_ratio), float(cos_ratio_p90)),
                }
            )

    return rows


def run_noise_sensitivity(
    *,
    models: Mapping[str, str],
    dataset: str = "tworoom",
    stds: Sequence[float] = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05),
    state_key: str | None = None,
    n_sequences: int = 256,
    future_steps: int = 8,
    frameskip: int = 1,
    img_size: int = 224,
    embedding_space: str | None = None,
    seed: int = 3072,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    corruption_type: str = "gaussian_noise",
) -> list[Dict[str, Any]]:
    if not models:
        raise ValueError("models must contain at least one label -> checkpoint path.")

    first_ckpt = next(iter(models.values()))
    first_model = load_model(first_ckpt, device)
    history_size = infer_history_size(first_model)
    del first_model

    batch = load_dataset_samples(
        dataset_name=dataset,
        state_key=state_key,
        n_sequences=n_sequences,
        history_size=history_size,
        future_steps=future_steps,
        frameskip=frameskip,
        img_size=img_size,
        seed=seed,
        device=device,
    )

    rows: list[Dict[str, Any]] = []
    for label, ckpt in models.items():
        rows.extend(
            analyze_model_noise(
                label=label,
                ckpt=ckpt,
                batch=batch,
                stds=stds,
                embedding_space=embedding_space,
                seed=seed,
                device=device,
                corruption_type=corruption_type,
            )
        )
    return rows


def format_noise_table(rows: Sequence[Mapping[str, Any]], frame_scope: str = "goal"):
    """Return a compact pandas table for notebook display."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("format_noise_table requires pandas.") from exc

    columns = [
        "model",
        "std",
        "frame_scope",
        "embedding_space",
        "noise_cos_sim_mean",
        "noise_angle_deg_median",
        "noise_angle_deg_p90",
        "clean_nn_cos_dist_median",
        "clean_effective_rank",
        "noise_to_nn_cos_ratio_median",
        "noise_to_nn_cos_ratio_p90",
        "noise_l2_median",
        "clean_nn_l2_median",
        "noise_to_nn_l2_ratio_median",
        "cka_linear_clean_vs_noisy",
        "risk",
    ]
    df = pd.DataFrame(rows)
    df = df[df["frame_scope"] == frame_scope].copy()
    cols_present = [c for c in columns if c in df.columns]
    df = df[cols_present].sort_values(["model", "std"]).reset_index(drop=True)
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].round(4)
    return df


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("This helper requires pandas.") from exc
    return pd


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("This helper requires matplotlib.") from exc
    return plt


# `_interpolate_threshold` and `_near_zero_slope` underpin the empirical
# `robust_radius_std` and `noise_angle_slope_deg_per_std` summary indicators.
# These are composite ratios introduced by this toolkit (inline metric definitions);
# the underlying idea echoes the certified radius framing of Cohen et al.
# (Randomized Smoothing, ICML 2019), but here applied to a *planning latent*
# rather than a classifier output, and produced empirically without
# certification guarantees.
def _interpolate_threshold(xs, ys, threshold: float) -> float:
    points = sorted(
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if not math.isnan(float(x)) and not math.isnan(float(y))
    )
    if not points:
        return float("nan")

    prev_x, prev_y = points[0]
    if prev_y >= threshold:
        return prev_x

    for x, y in points[1:]:
        if y >= threshold:
            if y == prev_y:
                return x
            alpha = (threshold - prev_y) / (y - prev_y)
            return prev_x + alpha * (x - prev_x)
        prev_x, prev_y = x, y
    return float("nan")


def _near_zero_slope(xs, ys, max_std: float) -> float:
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if 0.0 < float(x) <= max_std
        and not math.isnan(float(x))
        and not math.isnan(float(y))
    ]
    if not pairs:
        return float("nan")
    denom = sum(x * x for x, _ in pairs)
    if denom <= 0:
        return float("nan")
    return sum(x * y for x, y in pairs) / denom


def _first_crossing_std(xs, ys, threshold: float) -> float:
    pairs = sorted(
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if not math.isnan(float(x)) and not math.isnan(float(y))
    )
    for x, y in pairs:
        if y >= threshold:
            return x
    return float("nan")


def _geometry_flags(
    *,
    robust_radius: float,
    angle_slope: float,
    clean_nn_cos: float,
    clean_nn_l2: float,
    effective_rank_value: float = float("nan"),
) -> str:
    flags = []
    if not math.isnan(robust_radius) and robust_radius < 0.01:
        flags.append("fragile")
    elif not math.isnan(robust_radius) and robust_radius >= 0.02:
        flags.append("robust")

    if not math.isnan(angle_slope) and angle_slope >= 1500:
        flags.append("high_angle_gain")

    # Use cosine NN for the compactness flag. Absolute L2 scales differ across
    # LeWM/SWM spaces and tasks, so an L2 threshold can mislabel otherwise
    # healthy low-scale embeddings as clustered. Keep L2 ratios in the tables,
    # but avoid using them for the rule-of-thumb label.
    nn_compact = not math.isnan(clean_nn_cos) and clean_nn_cos < 0.02
    if nn_compact:
        # Disambiguate: tight NN with low rank => collapse; tight NN with high rank => clustering
        if not math.isnan(effective_rank_value) and effective_rank_value < 4.0:
            flags.append("collapsed")
        else:
            flags.append("clustered")

    return ",".join(flags) if flags else "balanced"


def _recommendation(flags: str) -> str:
    flag_set = set(flags.split(",")) if flags else set()
    if "collapsed" in flag_set:
        return "representation collapsed; strengthen anti-collapse regularizer before any robustness work"
    if "clustered" in flag_set and "fragile" in flag_set:
        return "avoid stronger invariance; add transition/action guardrail before more noise"
    if "clustered" in flag_set:
        return "watch precision-sensitive tasks; reduce noise or add resolution guardrail"
    if "fragile" in flag_set or "high_angle_gain" in flag_set:
        return "try weak noise consistency or encoder-sensitivity ablation"
    if "robust" in flag_set:
        return "geometry is noise-robust; prioritize clean planning/action metrics"
    return "collect eval correlation before changing training"


def summarize_noise_geometry(
    rows: Sequence[Mapping[str, Any]],
    *,
    frame_scope: str = "goal",
    threshold: float = 1.0,
    slope_max_std: float = 0.01,
):
    """Summarize noise rows into geometry design indicators.

    Returns one row per `(model, frame_scope, embedding_space)` with:
    - empirical robust radius: interpolated std where shift / clean NN reaches 1
    - near-zero angular slope: degrees per pixel-space std near zero
    - first high-risk std
    - a compact rule-based geometry flag and recommendation
    """
    pd = _require_pandas()
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()

    df = df[df["frame_scope"] == frame_scope].copy()
    summaries = []
    group_cols = ["model", "frame_scope", "embedding_space"]
    for keys, group in df.groupby(group_cols, sort=False):
        group = group.sort_values("std")
        robust_radius = _interpolate_threshold(
            group["std"],
            group["noise_to_nn_cos_ratio_median"],
            threshold,
        )
        angle_slope = _near_zero_slope(
            group["std"],
            group["noise_angle_deg_median"],
            slope_max_std,
        )
        ratio_slope = _near_zero_slope(
            group["std"],
            group["noise_to_nn_cos_ratio_median"],
            slope_max_std,
        )
        first_high = _first_crossing_std(
            group["std"],
            group["noise_to_nn_cos_ratio_median"],
            threshold,
        )

        clean_row = group.iloc[0]
        clean_nn_cos = float(clean_row.get("clean_nn_cos_dist_median", float("nan")))
        clean_nn_l2 = float(clean_row.get("clean_nn_l2_median", float("nan")))
        clean_eff_rank = float(clean_row.get("clean_effective_rank", float("nan")))
        flags = _geometry_flags(
            robust_radius=float(robust_radius),
            angle_slope=float(angle_slope),
            clean_nn_cos=clean_nn_cos,
            clean_nn_l2=clean_nn_l2,
            effective_rank_value=clean_eff_rank,
        )

        summaries.append(
            {
                "model": keys[0],
                "frame_scope": keys[1],
                "embedding_space": keys[2],
                "robust_radius_std": float(robust_radius),
                "first_high_risk_std": float(first_high),
                "noise_angle_slope_deg_per_std": float(angle_slope),
                "noise_ratio_slope_per_std": float(ratio_slope),
                "clean_nn_cos_dist_median": clean_nn_cos,
                "clean_nn_l2_median": clean_nn_l2,
                "clean_effective_rank": clean_eff_rank,
                "clean_norm_mean": float(clean_row.get("clean_norm_mean", float("nan"))),
                "geometry_flag": flags,
                "recommendation": _recommendation(flags),
            }
        )

    out = pd.DataFrame(summaries)
    numeric_cols = out.select_dtypes(include="number").columns
    out[numeric_cols] = out[numeric_cols].round(5)
    return out


def plot_noise_curves(
    rows: Sequence[Mapping[str, Any]],
    *,
    frame_scope: str = "goal",
    metric: str = "noise_to_nn_cos_ratio_median",
    threshold: float = 1.0,
    ax=None,
):
    """Plot noise curves for notebook/report use."""
    pd = _require_pandas()
    plt = _require_matplotlib()
    df = pd.DataFrame(rows)
    df = df[df["frame_scope"] == frame_scope].copy()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))

    for label, group in df.groupby("model", sort=False):
        group = group.sort_values("std")
        ax.plot(group["std"], group[metric], marker="o", label=str(label))

    if metric == "noise_to_nn_cos_ratio_median":
        ax.axhline(threshold, color="black", linestyle="--", linewidth=1, alpha=0.7)
        ax.text(
            0.99,
            threshold,
            "NN crossing",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=9,
        )
    ax.set_xlabel("pixel noise std")
    ax.set_ylabel(metric)
    ax.set_title(f"Noise sensitivity ({frame_scope})")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return ax.figure


def plot_geometry_tradeoff(summary_rows, *, ax=None):
    """Plot robustness vs clean resolution from `summarize_noise_geometry()`."""
    plt = _require_matplotlib()
    df = summary_rows.copy()
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4.8))

    ax.scatter(
        df["robust_radius_std"],
        df["clean_nn_cos_dist_median"],
        s=80,
        alpha=0.85,
    )
    for _, row in df.iterrows():
        ax.annotate(
            str(row["model"]),
            (row["robust_radius_std"], row["clean_nn_cos_dist_median"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
        )
    ax.set_xlabel("empirical robust radius (std at ratio=1)")
    ax.set_ylabel("clean nearest-neighbor cosine distance")
    ax.set_title("Robustness / resolution geometry map")
    ax.grid(True, alpha=0.25)
    return ax.figure


def _parse_model_specs(specs: Sequence[str]) -> Dict[str, str]:
    models: Dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Model spec must be label=/path/to/ckpt, got: {spec}")
        label, ckpt = spec.split("=", 1)
        models[label.strip()] = ckpt.strip()
    return models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose pixel-noise sensitivity in latent space.")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model spec as label=/path/to/model_object.ckpt. Repeat for comparisons.",
    )
    parser.add_argument("--dataset", default="tworoom")
    parser.add_argument("--stds", type=float, nargs="+", default=[0.0, 0.005, 0.01, 0.02, 0.03, 0.05])
    parser.add_argument("--state-key", default=None)
    parser.add_argument("--n-sequences", type=int, default=256)
    parser.add_argument("--future-steps", type=int, default=8)
    parser.add_argument("--frameskip", type=int, default=1)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--embedding-space", default=None, choices=[None, "raw", "normalized"])
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--plot", action="store_true", help="Save diagnostic PNG plots when --save-dir is set.")
    return parser


def main():
    args = build_parser().parse_args()
    rows = run_noise_sensitivity(
        models=_parse_model_specs(args.model),
        dataset=args.dataset,
        stds=args.stds,
        state_key=args.state_key,
        n_sequences=args.n_sequences,
        future_steps=args.future_steps,
        frameskip=args.frameskip,
        img_size=args.img_size,
        embedding_space=args.embedding_space,
        seed=args.seed,
        device=args.device,
    )

    print(format_noise_table(rows, frame_scope="goal").to_string(index=False))
    summary = summarize_noise_geometry(rows, frame_scope="goal")
    print("\n=== GEOMETRY SUMMARY ===")
    print(summary.to_string(index=False))

    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        with (save_dir / "noise_sensitivity.json").open("w") as f:
            json.dump(to_serializable(rows), f, indent=2)
        with (save_dir / "noise_sensitivity.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        summary.to_csv(save_dir / "geometry_summary.csv", index=False)
        with (save_dir / "geometry_summary.json").open("w") as f:
            json.dump(to_serializable(summary.to_dict(orient="records")), f, indent=2)
        if args.plot:
            fig = plot_noise_curves(rows, frame_scope="goal")
            fig.savefig(save_dir / "noise_ratio_curve_goal.png", dpi=200, bbox_inches="tight")
            fig = plot_noise_curves(
                rows,
                frame_scope="goal",
                metric="noise_angle_deg_median",
            )
            fig.savefig(save_dir / "noise_angle_curve_goal.png", dpi=200, bbox_inches="tight")
            fig = plot_geometry_tradeoff(summary)
            fig.savefig(save_dir / "geometry_tradeoff_goal.png", dpi=200, bbox_inches="tight")
        print(f"\n[noise_sensitivity] saved outputs to: {save_dir}")


if __name__ == "__main__":
    main()
