"""latent_noise_sensitivity.py — Encoder-decoupled latent-noise diagnostic.

Companion to `noise_sensitivity.py` and `predictor_sensitivity.py`. Those tools
inject Gaussian noise in *pixel* space and let it propagate through the encoder;
this tool injects Gaussian noise directly into the encoded latent `z`, skipping
the encoder. That isolates two downstream amplifiers:

1. The predictor's local Lipschitz / smoothness with respect to `z`
   (`predictor_target_shift_z`, `predictor_rollout_drift_z(T)`).
2. The planning cost surface's slope with respect to a perturbed goal latent
   (`cost_surface_slope_z`).

Combined with the input-space tools, this gives a three-layer attribution
(encoder / predictor / cost) for noise-induced planning failure.

References:
    - Cohen et al., "Certified Adversarial Robustness via Randomized Smoothing",
      ICML 2019. Framework borrowed: Gaussian sampling around a query point
      gives an empirical robustness radius. Here applied to a *latent* query
      point for a *world model*, not an input for a classifier.
    - Fazlyab et al., "Efficient and Accurate Estimation of Lipschitz Constants
      for Deep Neural Networks" (LipSDP), NeurIPS 2019. Conceptual reference
      for the Jacobian / slope estimates emitted here.
    - Hoffman et al., "Robust Learning with Jacobian Regularization",
      arXiv:1908.02729 (2019). Empirical Jacobian probing.
    - Li et al., "RobustZero: Enhancing MuZero Reinforcement Learning
      Robustness to State Perturbations", ICML 2025. Closest related work:
      they use latent perturbation as a *training* objective for MuZero;
      we use it as a *post-hoc diagnostic* for JEPA-style world models.

Notebook use:

    from tools.repr_analysis.latent_noise_sensitivity import run_latent_noise_sensitivity
    rows = run_latent_noise_sensitivity(
        models={"swm": "/path/...", "lewm": "/path/..."},
        dataset="tworoom",
        stds=[0.0, 0.005, 0.01, 0.02, 0.05],
        rollout_steps=[1, 2, 4, 8],
        noise_geometry="auto",      # SWM normalized -> tangent; LeWM/raw -> ambient
    )
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
    encode_sequences,
    get_embedding_space,
    get_model_spaces,
    infer_history_size,
    load_dataset_samples,
    load_model,
    resolve_space_name,
    to_serializable,
)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _clone_batch(batch: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}


def _safe_quantile(x: torch.Tensor, q: float) -> float:
    if x.numel() == 0:
        return float("nan")
    return float(torch.quantile(x.float().cpu(), q))


def _clean_nn_dist(z: torch.Tensor) -> Dict[str, float]:
    z = z.reshape(-1, z.size(-1))
    if z.size(0) < 2:
        return {"cos": float("nan"), "l2": float("nan")}
    z_norm = F.normalize(z, dim=-1, eps=1e-8)
    cos_dist = 1.0 - z_norm @ z_norm.T
    l2_dist = torch.cdist(z, z, p=2)
    eye = torch.eye(z.size(0), dtype=torch.bool, device=z.device)
    cos_nn = cos_dist.masked_fill(eye, float("inf")).min(dim=1).values.clamp_min(0.0)
    l2_nn = l2_dist.masked_fill(eye, float("inf")).min(dim=1).values
    return {
        "cos": _safe_quantile(cos_nn, 0.5),
        "l2": _safe_quantile(l2_nn, 0.5),
    }


def _shift_stats(clean: torch.Tensor, noisy: torch.Tensor) -> Dict[str, float]:
    """Per-token cosine / angle / L2 shift between clean and noisy embeddings."""
    clean = clean.reshape(-1, clean.size(-1))
    noisy = noisy.reshape(-1, noisy.size(-1))
    cn = F.normalize(clean, dim=-1, eps=1e-8)
    nn = F.normalize(noisy, dim=-1, eps=1e-8)
    cos = (cn * nn).sum(dim=-1).clamp(-1.0, 1.0)
    cos_dist = (1.0 - cos).clamp_min(0.0)
    angle_deg = torch.rad2deg(torch.acos(cos))
    l2 = torch.linalg.vector_norm(noisy - clean, dim=-1)
    return {
        "cos_dist_median": _safe_quantile(cos_dist, 0.5),
        "cos_dist_p90": _safe_quantile(cos_dist, 0.9),
        "angle_deg_median": _safe_quantile(angle_deg, 0.5),
        "angle_deg_p90": _safe_quantile(angle_deg, 0.9),
        "l2_median": _safe_quantile(l2, 0.5),
        "l2_p90": _safe_quantile(l2, 0.9),
    }


# ---------------------------------------------------------------------------
# Latent-space noise injection
# ---------------------------------------------------------------------------

def _frame_slice(T: int, scope: str, history_size: int) -> slice:
    if scope == "all":
        return slice(0, T)
    if scope == "history":
        return slice(0, min(history_size, T))
    if scope == "goal":
        return slice(T - 1, T)
    raise ValueError(f"Unsupported frame_scope: {scope}")


def _inject_latent_noise(
    z: torch.Tensor,
    *,
    std: float,
    seed: int,
    scope: str,
    history_size: int,
    geometry: str,
    base_norm: float | None,
) -> torch.Tensor:
    """Inject Gaussian noise into selected frames of the latent tensor.

    Args:
        z: (B, T, D) clean latent.
        std: noise std. If `base_norm` is None, std is interpreted as a fraction
            of each token's clean norm (relative scale). If `base_norm` is set,
            std is interpreted in absolute units multiplied by `base_norm`.
        scope: "history" | "goal" | "all".
        geometry: "ambient" (z+ε) or "tangent" (project ε onto T_z S^{d-1},
            then re-project to original radius). "tangent" is the natural
            choice for spherical (SWM) representations.
    """
    if std <= 0:
        return z.clone()

    out = z.clone()
    idx = _frame_slice(z.size(1), scope, history_size)
    sub = out[:, idx]  # (B, T_sub, D)

    with torch.random.fork_rng(devices=[z.device] if z.device.type == "cuda" else []):
        torch.manual_seed(seed)
        eps = torch.randn_like(sub)

    if base_norm is None:
        scale = torch.linalg.vector_norm(sub, dim=-1, keepdim=True).clamp_min(1e-8)
    else:
        scale = torch.full_like(sub[..., :1], float(base_norm))
    eps = eps * std * scale

    if geometry == "tangent":
        # Project onto tangent of unit sphere at the (normalized) direction of
        # `sub`, then re-project the perturbed point back to the same radius.
        sub_dir = F.normalize(sub, dim=-1, eps=1e-8)
        radial = (sub_dir * eps).sum(dim=-1, keepdim=True) * sub_dir
        eps_tan = eps - radial
        radius = torch.linalg.vector_norm(sub, dim=-1, keepdim=True).clamp_min(1e-8)
        perturbed = F.normalize(sub + eps_tan, dim=-1, eps=1e-8) * radius
    elif geometry == "ambient":
        perturbed = sub + eps
    else:
        raise ValueError(f"Unsupported geometry: {geometry}")

    out[:, idx] = perturbed
    return out


# ---------------------------------------------------------------------------
# Predictor target shift / rollout drift / cost slope (latent-only)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _open_loop_target_shift(
    model,
    clean_emb: torch.Tensor,
    noisy_emb: torch.Tensor,
    act_emb: torch.Tensor,
    history_size: int,
) -> Dict[str, torch.Tensor]:
    B, T, _ = clean_emb.shape
    H = history_size
    if T <= H:
        return {"clean_pred": clean_emb[:, :0], "noisy_pred": noisy_emb[:, :0]}

    clean_preds, noisy_preds = [], []
    # Iterate over all H-token windows, including the final window ending at
    # T-1. Unlike predictor_sensitivity.py this latent-only probe compares
    # clean-vs-noisy predictor outputs directly, so it does not need a dataset
    # target after the window. Including the final window lets goal-scope latent
    # perturbations test whether the predictor would consume a perturbed goal.
    for s in range(T - H + 1):
        c_win = clean_emb[:, s : s + H]
        n_win = noisy_emb[:, s : s + H]
        a_win = act_emb[:, s : s + H]
        clean_preds.append(model.predict(c_win, a_win)[:, -1])
        noisy_preds.append(model.predict(n_win, a_win)[:, -1])
    return {
        "clean_pred": torch.stack(clean_preds, dim=1),  # (B, T-H+1, D)
        "noisy_pred": torch.stack(noisy_preds, dim=1),
    }


@torch.no_grad()
def _autoregressive_rollout(
    model,
    init_emb: torch.Tensor,
    act_emb: torch.Tensor,
    history_size: int,
    n_steps: int,
) -> torch.Tensor:
    H = history_size
    chain = init_emb.clone()
    for t in range(n_steps):
        a_win = act_emb[:, t : t + H]
        if a_win.size(1) < H:
            break
        pred = model.predict(chain[:, -H:], a_win)[:, -1:]
        chain = torch.cat([chain, pred], dim=1)
    return chain  # (B, H + steps, D)


def _planning_cost(
    pred: torch.Tensor,
    goal: torch.Tensor,
    cost_type: str,
) -> torch.Tensor:
    """Match `JEPA.criterion` / `SphericalJEPA.criterion` on the relevant tokens.

    pred, goal: (..., D). Returns per-sample scalar cost.
    """
    if cost_type == "cosine":
        return 1.0 - F.cosine_similarity(pred, goal, dim=-1)
    if cost_type == "mse":
        return F.mse_loss(pred, goal, reduction="none").sum(dim=-1)
    raise ValueError(f"Unsupported cost_type: {cost_type}")


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------

def analyze_model_latent_noise(
    *,
    label: str,
    ckpt: str,
    batch: Mapping[str, torch.Tensor],
    stds: Sequence[float],
    rollout_steps: Sequence[int],
    embedding_space: str | None,
    frame_scopes: Sequence[str],
    noise_geometry: str,
    std_mode: str,
    n_noise_samples: int,
    seed: int,
    device: str,
) -> list[Dict[str, Any]]:
    model = load_model(ckpt, device)
    spaces = get_model_spaces(model)
    space = resolve_space_name(embedding_space or spaces["inference_cost_space"])
    cost_type = spaces["inference_cost_type"]
    history_size = infer_history_size(model)
    if noise_geometry == "auto":
        # SphericalJEPA exposes normalize_embeddings and commonly analyzes the
        # normalized cost space; tangent noise respects that sphere. LeWM and
        # raw-space ablations remain Euclidean, so ambient noise is correct.
        noise_geometry = (
            "tangent"
            if space == "normalized" and hasattr(model, "normalize_embeddings")
            else "ambient"
        )

    # Encode clean batch once. We work in the inference cost space throughout.
    clean_outputs = encode_sequences(model, _clone_batch(batch))
    clean_emb = get_embedding_space(clean_outputs, space).detach()
    act_emb = clean_outputs["act_emb"].detach()

    nn_ref = _clean_nn_dist(clean_emb)
    clean_norm_mean = float(torch.linalg.vector_norm(clean_emb, dim=-1).mean())
    base_norm = None if std_mode == "relative" else clean_norm_mean

    # Reference clean predictor outputs (for cost slope and as a baseline).
    @torch.no_grad()
    def _clean_last_pred() -> torch.Tensor:
        # Last-token prediction over the final H-window of the clean sequence,
        # mirroring how planning cost reads `predicted_emb[..., -1, :]`.
        H = history_size
        T = clean_emb.size(1)
        if T < H:
            return clean_emb[:, -1:, :]
        c_win = clean_emb[:, T - H : T]
        a_win = act_emb[:, T - H : T]
        return model.predict(c_win, a_win)[:, -1]  # (B, D)

    clean_pred_last = _clean_last_pred()

    rows: list[Dict[str, Any]] = []
    max_steps = max(rollout_steps) if rollout_steps else 0

    for std_idx, std in enumerate(stds):
        std_seed_base = seed + 1009 * std_idx

        for scope in frame_scopes:
            # Average over n_noise_samples to reduce stochastic variance.
            target_stats_acc: Dict[str, list[float]] = {}
            rollout_acc: Dict[str, list[float]] = {}
            cost_slope_acc: list[float] = []
            shift_acc: Dict[str, list[float]] = {}

            for k in range(max(1, n_noise_samples)):
                noisy_emb = _inject_latent_noise(
                    clean_emb,
                    std=float(std),
                    seed=std_seed_base + 31 * k,
                    scope=scope,
                    history_size=history_size,
                    geometry=noise_geometry,
                    base_norm=base_norm,
                )

                # Realized latent shift on the perturbed slice only. Measuring
                # the whole sequence would dilute goal-only noise with many
                # unchanged history tokens and under-report the injected shift.
                idx = _frame_slice(clean_emb.size(1), scope, history_size)
                shift = _shift_stats(clean_emb[:, idx], noisy_emb[:, idx])
                for kk, v in shift.items():
                    shift_acc.setdefault(kk, []).append(v)

                # Predictor open-loop target shift over all H-windows.
                ol = _open_loop_target_shift(
                    model, clean_emb, noisy_emb, act_emb, history_size
                )
                if ol["clean_pred"].numel() > 0:
                    ts = _shift_stats(ol["clean_pred"], ol["noisy_pred"])
                else:
                    ts = {kk: float("nan") for kk in (
                        "cos_dist_median", "cos_dist_p90",
                        "angle_deg_median", "angle_deg_p90",
                        "l2_median", "l2_p90",
                    )}
                for kk, v in ts.items():
                    target_stats_acc.setdefault(kk, []).append(v)

                # Autoregressive rollout drift, init from noisy z[:, :H].
                if max_steps > 0 and clean_emb.size(1) >= history_size:
                    init_clean = clean_emb[:, :history_size]
                    init_noisy = noisy_emb[:, :history_size]
                    chain_clean = _autoregressive_rollout(
                        model, init_clean, act_emb, history_size, max_steps
                    )
                    chain_noisy = _autoregressive_rollout(
                        model, init_noisy, act_emb, history_size, max_steps
                    )
                    achieved = chain_clean.size(1) - history_size
                    for T in rollout_steps:
                        if T > achieved:
                            rollout_acc.setdefault(f"rollout_T{T}_l2_median", []).append(float("nan"))
                            rollout_acc.setdefault(f"rollout_T{T}_cos_dist_median", []).append(float("nan"))
                            rollout_acc.setdefault(f"rollout_T{T}_angle_deg_median", []).append(float("nan"))
                            continue
                        idx = history_size + T - 1
                        stat = _shift_stats(
                            chain_clean[:, idx : idx + 1],
                            chain_noisy[:, idx : idx + 1],
                        )
                        rollout_acc.setdefault(f"rollout_T{T}_l2_median", []).append(stat["l2_median"])
                        rollout_acc.setdefault(f"rollout_T{T}_cos_dist_median", []).append(stat["cos_dist_median"])
                        rollout_acc.setdefault(f"rollout_T{T}_angle_deg_median", []).append(stat["angle_deg_median"])

                # Planning cost slope (only when we perturb the goal token).
                if scope == "goal":
                    clean_goal = clean_emb[:, -1, :]
                    noisy_goal = noisy_emb[:, -1, :]
                    cost_clean = _planning_cost(clean_pred_last, clean_goal, cost_type)
                    cost_noisy = _planning_cost(clean_pred_last, noisy_goal, cost_type)
                    cost_slope_acc.append(float((cost_noisy - cost_clean).abs().mean()))

            def _avg(d: Dict[str, list[float]], k: str) -> float:
                vals = [v for v in d.get(k, []) if not math.isnan(v)]
                return float(sum(vals) / len(vals)) if vals else float("nan")

            target_stats = {k: _avg(target_stats_acc, k) for k in target_stats_acc}
            rollout_stats = {k: _avg(rollout_acc, k) for k in rollout_acc}
            shift_stats_avg = {k: _avg(shift_acc, k) for k in shift_acc}

            # Normalize predictor target shift by clean NN distance (cross-model
            # comparable). This is the latent-space analogue of
            # `noise_to_nn_cos_ratio` from `noise_sensitivity.py`.
            cos_norm = nn_ref["cos"] if nn_ref["cos"] and nn_ref["cos"] > 0 else float("nan")
            l2_norm = nn_ref["l2"] if nn_ref["l2"] and nn_ref["l2"] > 0 else float("nan")
            target_to_nn_cos = (
                target_stats.get("cos_dist_median", float("nan")) / cos_norm
                if not math.isnan(cos_norm) else float("nan")
            )
            target_to_nn_l2 = (
                target_stats.get("l2_median", float("nan")) / l2_norm
                if not math.isnan(l2_norm) else float("nan")
            )

            cost_delta_mean = (
                float(sum(cost_slope_acc) / len(cost_slope_acc))
                if cost_slope_acc else float("nan")
            )
            cost_slope = (
                cost_delta_mean / float(std)
                if std > 0 and not math.isnan(cost_delta_mean)
                else float("nan")
            )

            rows.append(
                {
                    "model": label,
                    "ckpt": ckpt,
                    "std": float(std),
                    "frame_scope": scope,
                    "noise_geometry": noise_geometry,
                    "std_mode": std_mode,
                    "embedding_space": space,
                    "cost_type": cost_type,
                    "history_size": int(history_size),
                    "clean_norm_mean": clean_norm_mean,
                    # Encoder-side realized shift on the perturbed slice
                    **{f"latent_{k}": v for k, v in shift_stats_avg.items()},
                    # Predictor open-loop target shift in latent
                    **{f"target_{k}": v for k, v in target_stats.items()},
                    "target_to_nn_cos_ratio": float(target_to_nn_cos),
                    "target_to_nn_l2_ratio": float(target_to_nn_l2),
                    # Multi-step rollout drift between noisy/clean conditioning
                    **rollout_stats,
                    # Cost surface slope (goal scope only)
                    "cost_delta_mean": cost_delta_mean,
                    "cost_surface_slope_z": cost_slope,
                    "clean_nn_cos_dist_median": float(nn_ref["cos"]),
                    "clean_nn_l2_median": float(nn_ref["l2"]),
                }
            )

    return rows


def run_latent_noise_sensitivity(
    *,
    models: Mapping[str, str],
    dataset: str = "tworoom",
    stds: Sequence[float] = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05),
    rollout_steps: Sequence[int] = (1, 2, 4, 8),
    state_key: str | None = None,
    n_sequences: int = 256,
    future_steps: int = 8,
    frameskip: int = 1,
    img_size: int = 224,
    embedding_space: str | None = None,
    frame_scopes: Sequence[str] = ("history", "goal", "all"),
    noise_geometry: str = "auto",
    std_mode: str = "relative",
    n_noise_samples: int = 1,
    seed: int = 3072,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> list[Dict[str, Any]]:
    if not models:
        raise ValueError("models must contain at least one label -> checkpoint path.")

    first_ckpt = next(iter(models.values()))
    first_model = load_model(first_ckpt, device)
    history_size = infer_history_size(first_model)
    del first_model

    needed_future = max(max(rollout_steps) if rollout_steps else 0, future_steps)
    batch = load_dataset_samples(
        dataset_name=dataset,
        state_key=state_key,
        n_sequences=n_sequences,
        history_size=history_size,
        future_steps=needed_future,
        frameskip=frameskip,
        img_size=img_size,
        seed=seed,
        device=device,
    )

    rows: list[Dict[str, Any]] = []
    for label, ckpt in models.items():
        rows.extend(
            analyze_model_latent_noise(
                label=label,
                ckpt=ckpt,
                batch=batch,
                stds=stds,
                rollout_steps=rollout_steps,
                embedding_space=embedding_space,
                frame_scopes=frame_scopes,
                noise_geometry=noise_geometry,
                std_mode=std_mode,
                n_noise_samples=n_noise_samples,
                seed=seed,
                device=device,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Summarization (mirrors `summarize_noise_geometry` from noise_sensitivity)
# ---------------------------------------------------------------------------

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


def summarize_latent_noise_geometry(
    rows: Sequence[Mapping[str, Any]],
    *,
    frame_scope: str = "goal",
    threshold: float = 1.0,
    slope_max_std: float = 0.02,
):
    """One-row-per-model summary with `robust_radius_z` and slope estimates."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("summarize_latent_noise_geometry requires pandas.") from exc

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()

    df = df[df["frame_scope"] == frame_scope].copy()
    summaries = []
    group_cols = ["model", "frame_scope", "embedding_space", "noise_geometry"]
    for keys, group in df.groupby(group_cols, sort=False):
        group = group.sort_values("std")
        # Primary: target shift relative to clean NN (single-step predictor sensitivity)
        robust_radius = _interpolate_threshold(
            group["std"], group["target_to_nn_cos_ratio"], threshold
        )
        # Fallback: rollout drift relative to clean NN (multi-step amplification)
        if math.isnan(robust_radius) and "rollout_T8_l2_median" in group.columns and "clean_nn_l2_median" in group.columns:
            drift_ratio = group["rollout_T8_l2_median"] / group["clean_nn_l2_median"].clip(lower=1e-8)
            robust_radius = _interpolate_threshold(group["std"], drift_ratio, threshold)
        # Single-step predictor target shift slope (may be flat in latent-noise)
        target_angle_slope = _near_zero_slope(
            group["std"], group["target_angle_deg_median"], slope_max_std
        )
        target_l2_slope = _near_zero_slope(
            group["std"], group["target_l2_median"], slope_max_std
        )
        # Multi-step rollout drift slope (more informative in latent-noise)
        rollout_angle_slope = _near_zero_slope(
            group["std"], group.get("rollout_T8_angle_deg_median"), slope_max_std
        ) if "rollout_T8_angle_deg_median" in group.columns else float("nan")
        rollout_l2_slope = _near_zero_slope(
            group["std"], group.get("rollout_T8_l2_median"), slope_max_std
        ) if "rollout_T8_l2_median" in group.columns else float("nan")
        cost_slope = _near_zero_slope(
            group["std"], group["cost_delta_mean"], slope_max_std
        ) if "cost_delta_mean" in group.columns else float("nan")
        clean_nn_cos = float(group.iloc[0].get("clean_nn_cos_dist_median", float("nan")))
        summaries.append(
            {
                "model": keys[0],
                "frame_scope": keys[1],
                "embedding_space": keys[2],
                "noise_geometry": keys[3],
                "robust_radius_z": float(robust_radius),
                "predictor_angle_slope_deg_per_std_z": float(target_angle_slope),
                "predictor_l2_slope_per_std_z": float(target_l2_slope),
                "rollout_angle_slope_deg_per_std_z": float(rollout_angle_slope),
                "rollout_l2_slope_per_std_z": float(rollout_l2_slope),
                "cost_surface_slope_z": float(cost_slope),
                "clean_nn_cos_dist_median": clean_nn_cos,
            }
        )

    out = pd.DataFrame(summaries)
    numeric_cols = out.select_dtypes(include="number").columns
    out[numeric_cols] = out[numeric_cols].round(5)
    return out


def format_latent_noise_table(rows: Sequence[Mapping[str, Any]], frame_scope: str = "goal"):
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("format_latent_noise_table requires pandas.") from exc
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[df["frame_scope"] == frame_scope].copy()

    columns = [c for c in [
        "model", "std", "frame_scope", "noise_geometry", "embedding_space",
        "target_angle_deg_median", "target_cos_dist_median", "target_l2_median",
        "target_to_nn_cos_ratio", "target_to_nn_l2_ratio",
        "rollout_T1_l2_median", "rollout_T2_l2_median",
        "rollout_T4_l2_median", "rollout_T8_l2_median",
        "cost_delta_mean", "cost_surface_slope_z",
        "clean_nn_cos_dist_median", "clean_norm_mean",
    ] if c in df.columns]
    df = df[columns].sort_values(["model", "std"]).reset_index(drop=True)
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].round(4)
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_latent_noise_curves(
    rows: Sequence[Mapping[str, Any]],
    *,
    frame_scope: str = "goal",
    metric: str = "target_to_nn_cos_ratio",
    threshold: float = 1.0,
    ax=None,
):
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("plot_latent_noise_curves requires pandas + matplotlib.") from exc
    df = pd.DataFrame(rows)
    df = df[df["frame_scope"] == frame_scope].copy()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    for label, group in df.groupby("model", sort=False):
        group = group.sort_values("std")
        ax.plot(group["std"], group[metric], marker="o", label=str(label))
    if metric == "target_to_nn_cos_ratio":
        ax.axhline(threshold, color="black", linestyle="--", linewidth=1, alpha=0.7)
        ax.text(0.99, threshold, "NN crossing",
                transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=9)
    ax.set_xlabel("latent noise std (relative to clean norm)")
    ax.set_ylabel(metric)
    ax.set_title(f"Latent noise sensitivity ({frame_scope})")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return ax.figure


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_model_specs(specs: Sequence[str]) -> Dict[str, str]:
    models: Dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Model spec must be label=/path/to/ckpt, got: {spec}")
        label, ckpt = spec.split("=", 1)
        models[label.strip()] = ckpt.strip()
    return models


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Diagnose predictor / cost sensitivity to latent-space Gaussian noise."
    )
    p.add_argument("--model", action="append", required=True,
                   help="Model spec as label=/path/to/model_object.ckpt. Repeat for comparisons.")
    p.add_argument("--dataset", default="tworoom")
    p.add_argument("--stds", type=float, nargs="+",
                   default=[0.0, 0.005, 0.01, 0.02, 0.03, 0.05])
    p.add_argument("--rollout-steps", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--state-key", default=None)
    p.add_argument("--n-sequences", type=int, default=256)
    p.add_argument("--future-steps", type=int, default=8)
    p.add_argument("--frameskip", type=int, default=1)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--embedding-space", default=None, choices=[None, "raw", "normalized"])
    p.add_argument("--frame-scopes", nargs="+", default=["history", "goal", "all"],
                   choices=["history", "goal", "all"])
    p.add_argument("--noise-geometry", default="auto", choices=["auto", "ambient", "tangent"],
                   help="`auto` uses tangent noise for SWM normalized space and ambient otherwise.")
    p.add_argument("--std-mode", default="relative", choices=["relative", "absolute"],
                   help="`relative` scales std by per-token clean norm "
                        "(comparable across LeWM/SWM). `absolute` uses raw σ × global clean norm.")
    p.add_argument("--n-noise-samples", type=int, default=1,
                   help="Number of independent noise samples averaged per (std, scope).")
    p.add_argument("--seed", type=int, default=3072)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-dir", default=None)
    p.add_argument("--plot", action="store_true",
                   help="Save diagnostic PNG plots when --save-dir is set.")
    return p


def main():
    args = build_parser().parse_args()
    rows = run_latent_noise_sensitivity(
        models=_parse_model_specs(args.model),
        dataset=args.dataset,
        stds=args.stds,
        rollout_steps=args.rollout_steps,
        state_key=args.state_key,
        n_sequences=args.n_sequences,
        future_steps=args.future_steps,
        frameskip=args.frameskip,
        img_size=args.img_size,
        embedding_space=args.embedding_space,
        frame_scopes=args.frame_scopes,
        noise_geometry=args.noise_geometry,
        std_mode=args.std_mode,
        n_noise_samples=args.n_noise_samples,
        seed=args.seed,
        device=args.device,
    )

    print(format_latent_noise_table(rows, frame_scope="goal").to_string(index=False))
    summary = summarize_latent_noise_geometry(rows, frame_scope="goal")
    print("\n=== LATENT GEOMETRY SUMMARY (goal scope) ===")
    print(summary.to_string(index=False))

    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        with (save_dir / "latent_noise_sensitivity.json").open("w") as f:
            json.dump(to_serializable(rows), f, indent=2)
        with (save_dir / "latent_noise_sensitivity.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        summary.to_csv(save_dir / "latent_geometry_summary.csv", index=False)
        with (save_dir / "latent_geometry_summary.json").open("w") as f:
            json.dump(to_serializable(summary.to_dict(orient="records")), f, indent=2)
        if args.plot:
            try:
                fig = plot_latent_noise_curves(rows, frame_scope="goal")
                fig.savefig(save_dir / "latent_noise_ratio_curve_goal.png", dpi=200, bbox_inches="tight")
                fig = plot_latent_noise_curves(
                    rows, frame_scope="goal", metric="target_angle_deg_median",
                )
                fig.savefig(save_dir / "latent_noise_angle_curve_goal.png", dpi=200, bbox_inches="tight")
            except Exception as e:
                print(f"[latent_noise_sensitivity] plotting skipped: {e}")
        print(f"\n[latent_noise_sensitivity] saved outputs to: {save_dir}")


if __name__ == "__main__":
    main()
