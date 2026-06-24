#!/usr/bin/env python3
"""
Offline w_t / critical_t visualization for adaptive consistency checkpoints.

Two analysis modes are supported:

1. Aggregate token statistics from random sequence windows.
2. Per-episode trajectory analysis with sliding-window gate recomputation.

The trajectory mode is the paper-facing diagnostic:
- reconstruct full σ + A_t gate signals from an existing checkpoint
- aggregate overlapping context-window weights back to episode time
- render time series + salient keyframes without retraining
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import stable_pretraining as spt
import stable_worldmodel as swm

from tools.repr_analysis.analyze_repr import (
    encode_sequences,
    infer_history_size,
    load_dataset_samples,
    load_model,
)
from utils import get_column_normalizer, get_img_preprocessor, resolve_h5_dataset_path


def compute_action_gate_metrics_offline(
    model,
    ctx_emb: torch.Tensor,
    ctx_action_raw: torch.Tensor,
    pred_emb_clean: torch.Tensor,
    *,
    s_t: torch.Tensor | None,
    delta_scale: float = 0.25,
    num_delta_samples: int = 4,
    delta_norm_floor: float = 1e-6,
    log_a_floor: float = 1e-8,
    w_min: float = 0.2,
    w_max: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Reproduce train.py::compute_action_gate_metrics in eval mode."""
    with torch.no_grad():
        ctx_emb_d = ctx_emb.detach()
        pred_clean_d = pred_emb_clean.detach()
        bsz, t_ctx = ctx_emb_d.shape[:2]

        action_std = ctx_action_raw.float().std(dim=(0, 1), unbiased=False).clamp(min=1e-6)

        bn_states = []
        for mod in model.modules():
            if isinstance(mod, nn.modules.batchnorm._BatchNorm) and mod.training:
                bn_states.append(mod)
                mod.eval()
        try:
            a_samples = []
            for _ in range(num_delta_samples):
                delta = torch.randn_like(ctx_action_raw) * (delta_scale * action_std)
                act_pert = ctx_action_raw + delta
                act_emb_pert = model.action_encoder(act_pert)
                pred_pert = model.predict(ctx_emb_d, act_emb_pert)
                diff = (pred_pert - pred_clean_d).pow(2).sum(dim=-1).clamp(min=0).sqrt()
                delta_norm = delta.pow(2).sum(dim=-1).clamp(min=0).sqrt().clamp(min=delta_norm_floor)
                a_samples.append(diff / delta_norm)
        finally:
            for mod in bn_states:
                mod.train()

        a_stack = torch.stack(a_samples, dim=0)
        a_mean = a_stack.mean(dim=0)
        log_a = torch.log(a_mean.clamp(min=log_a_floor))

        def _zscore(x: torch.Tensor, name: str) -> torch.Tensor:
            inited = getattr(model, f"gate_{name}_inited").item() > 0.5
            if inited:
                mean = getattr(model, f"gate_{name}_mean")
                var = getattr(model, f"gate_{name}_var")
            else:
                mean = x.mean()
                var = x.var(unbiased=False)
            return (x - mean) / var.clamp(min=1e-6).sqrt()

        g_a = torch.sigmoid(_zscore(log_a, "log_A"))
        if s_t is not None:
            g_s = torch.sigmoid(_zscore(s_t.detach(), "s"))
            critical = g_a * (0.5 + 0.5 * g_s)
        else:
            g_s = None
            critical = g_a * 0.5
        w_t = w_max - (w_max - w_min) * critical

    out = {
        "w_t": w_t,
        "critical": critical,
        "gA": g_a,
        "A_mean": a_mean,
        "log_A": log_a,
    }
    if g_s is not None:
        out["gS"] = g_s
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline w_t visualization")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model object checkpoint")
    parser.add_argument("--dataset", type=str, default="pusht_expert_train", help="Dataset name")
    parser.add_argument("--mode", type=str, default="both", choices=["aggregate", "trajectory", "both"])
    parser.add_argument("--n-sequences", type=int, default=256, help="Aggregate-mode random sequences")
    parser.add_argument("--future-steps", type=int, default=8, help="Future steps for random sequences")
    parser.add_argument("--frameskip", type=int, default=5, help="Frameskip used by the dataset")
    parser.add_argument("--img-size", type=int, default=224, help="Image size")
    parser.add_argument("--seed", type=int, default=3072, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--save-dir", type=str, default="assets/diagnostics", help="Output directory")
    parser.add_argument("--episode-ids", type=int, nargs="*", default=None, help="Explicit episode ids")
    parser.add_argument("--n-episodes", type=int, default=4, help="Trajectory-mode random episodes")
    parser.add_argument("--episode-seed", type=int, default=4096, help="Seed for random episode sampling")
    parser.add_argument("--n-keyframes", type=int, default=4, help="Keyframes per trajectory panel")
    parser.add_argument("--min-keyframe-gap", type=int, default=6, help="Minimum gap between keyframes")
    parser.add_argument("--delta-scale", type=float, default=0.25, help="Action perturbation scale")
    parser.add_argument("--num-delta-samples", type=int, default=4, help="Number of delta samples")
    parser.add_argument("--delta-norm-floor", type=float, default=1e-6)
    parser.add_argument("--log-a-floor", type=float, default=1e-8)
    parser.add_argument("--w-min", type=float, default=0.2)
    parser.add_argument("--w-max", type=float, default=1.0)
    parser.add_argument("--s-min", type=float, default=-4.0)
    parser.add_argument("--s-max", type=float, default=4.0)
    return parser


def build_episode_dataset(dataset_name: str, img_size: int, frameskip: int = 5):
    # Resolve the H5 path explicitly so we tolerate both swm layouts
    # (0.0.6 flat vs post-PR-#221 `datasets/` subdir). Nested names like
    # "ogbench/cube_single_expert" are preserved verbatim.
    h5_path = resolve_h5_dataset_path(dataset_name)
    dataset = swm.data.HDF5Dataset(
        path=str(h5_path),
        num_steps=1,
        frameskip=frameskip,
        transform=None,
    )
    transform = spt.data.transforms.Compose(
        get_img_preprocessor("pixels", "pixels", img_size),
        get_column_normalizer(dataset, "action", "action"),
    )
    dataset.transform = transform
    return dataset


def resolve_episode_column(dataset) -> str:
    return "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"


def load_episode(
    dataset,
    raw_dataset,
    episode_id: int,
    *,
    device: str,
) -> Dict[str, torch.Tensor | np.ndarray | int]:
    ep_col = resolve_episode_column(raw_dataset)
    episode_idx = raw_dataset.get_col_data(ep_col)
    step_idx = raw_dataset.get_col_data("step_idx")
    row_indices = np.flatnonzero(episode_idx == episode_id)
    if row_indices.size == 0:
        raise ValueError(f"Episode {episode_id} not found in dataset")
    row_indices = row_indices[np.argsort(step_idx[row_indices])]

    proc_samples = [dataset[int(idx)] for idx in row_indices.tolist()]
    raw_samples = [raw_dataset[int(idx)] for idx in row_indices.tolist()]

    # Single-frame samples may have a leading batch dim (1, C, H, W) from the
    # image preprocessor.  Use cat(dim=0) to collapse it robustly.
    pixels = torch.cat([sample["pixels"] for sample in proc_samples], dim=0).to(device)
    action = torch.nan_to_num(
        torch.cat([sample["action"] for sample in proc_samples], dim=0),
        0.0,
    ).to(device)
    raw_pixels = np.stack([np.asarray(sample["pixels"]) for sample in raw_samples], axis=0)
    steps = np.asarray([int(step_idx[idx]) for idx in row_indices], dtype=np.int64)

    return {
        "episode_id": int(episode_id),
        "pixels": pixels,
        "action": action,
        "raw_pixels": raw_pixels,
        "step_idx": steps,
    }


def sample_episode_ids(raw_dataset, *, n_episodes: int, seed: int, history_size: int) -> List[int]:
    ep_col = resolve_episode_column(raw_dataset)
    episode_idx = raw_dataset.get_col_data(ep_col)
    step_idx = raw_dataset.get_col_data("step_idx")
    episode_ids = np.unique(episode_idx)
    valid_ids = []
    for ep_id in episode_ids.tolist():
        length = int(step_idx[episode_idx == ep_id].max()) + 1
        if length > history_size:
            valid_ids.append(int(ep_id))
    rng = np.random.default_rng(seed)
    if n_episodes >= len(valid_ids):
        return valid_ids
    choice = rng.choice(valid_ids, size=n_episodes, replace=False)
    return [int(v) for v in sorted(choice.tolist())]


def to_display_image(raw_pixels: np.ndarray) -> np.ndarray:
    arr = np.asarray(raw_pixels)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr.squeeze(0)
    if arr.ndim != 3:
        raise ValueError(f"Expected image with ndim=3, got shape={arr.shape}")
    if arr.shape[0] in {1, 3} and arr.shape[-1] not in {1, 3}:
        arr = np.transpose(arr, (1, 2, 0))
    arr = arr.astype(np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def select_salient_steps(signal: np.ndarray, *, k: int, min_gap: int) -> List[int]:
    order = np.argsort(signal)[::-1]
    selected: List[int] = []
    for idx in order.tolist():
        if all(abs(idx - prev) >= min_gap for prev in selected):
            selected.append(int(idx))
        if len(selected) >= k:
            break
    if not selected:
        selected = [int(np.argmax(signal))]
    return sorted(selected)


def aggregate_per_timestep(values_by_timestep: List[List[float]]) -> np.ndarray:
    out = np.full(len(values_by_timestep), np.nan, dtype=np.float32)
    for idx, vals in enumerate(values_by_timestep):
        if vals:
            out[idx] = float(np.mean(vals))
    return out


def analyze_episode_trajectory(
    model,
    episode: Dict[str, torch.Tensor | np.ndarray | int],
    *,
    history_size: int,
    delta_scale: float,
    num_delta_samples: int,
    delta_norm_floor: float,
    log_a_floor: float,
    w_min: float,
    w_max: float,
    s_min: float,
    s_max: float,
) -> Dict[str, np.ndarray | int | float | Sequence[int]]:
    pixels = episode["pixels"].unsqueeze(0)
    action = episode["action"].unsqueeze(0)

    outputs = encode_sequences(model, {"pixels": pixels, "action": action})
    emb = outputs["emb"]
    act_emb = outputs["act_emb"]
    total_steps = emb.size(1)

    w_vals = [[] for _ in range(total_steps)]
    critical_vals = [[] for _ in range(total_steps)]
    g_a_vals = [[] for _ in range(total_steps)]
    g_s_vals = [[] for _ in range(total_steps)]

    for end_idx in range(history_size, total_steps):
        start_idx = end_idx - history_size
        ctx_emb = emb[:, start_idx:end_idx, :]
        ctx_action = action[:, start_idx:end_idx, :]
        ctx_act_emb = act_emb[:, start_idx:end_idx, :]

        if hasattr(model, "predict_with_logvar") and getattr(model, "pred_logvar_proj", None) is not None:
            pred_emb, logvar_hat = model.predict_with_logvar(ctx_emb, ctx_act_emb, detach_logvar_input=False)
            s_t = logvar_hat.squeeze(-1).clamp(min=s_min, max=s_max) if logvar_hat is not None else None
        else:
            pred_emb = model.predict(ctx_emb, ctx_act_emb)
            s_t = None

        gate = compute_action_gate_metrics_offline(
            model,
            ctx_emb,
            ctx_action,
            pred_emb,
            s_t=s_t,
            delta_scale=delta_scale,
            num_delta_samples=num_delta_samples,
            delta_norm_floor=delta_norm_floor,
            log_a_floor=log_a_floor,
            w_min=w_min,
            w_max=w_max,
        )

        w_local = gate["w_t"][0].detach().cpu().numpy()
        critical_local = gate["critical"][0].detach().cpu().numpy()
        g_a_local = gate["gA"][0].detach().cpu().numpy()
        g_s_local = gate.get("gS")
        g_s_local_np = g_s_local[0].detach().cpu().numpy() if g_s_local is not None else None

        for offset, global_idx in enumerate(range(start_idx, end_idx)):
            w_vals[global_idx].append(float(w_local[offset]))
            critical_vals[global_idx].append(float(critical_local[offset]))
            g_a_vals[global_idx].append(float(g_a_local[offset]))
            if g_s_local_np is not None:
                g_s_vals[global_idx].append(float(g_s_local_np[offset]))

    w_series = aggregate_per_timestep(w_vals)
    critical_series = aggregate_per_timestep(critical_vals)
    g_a_series = aggregate_per_timestep(g_a_vals)
    g_s_series = aggregate_per_timestep(g_s_vals) if any(g_s_vals) else None

    action_np = episode["action"].detach().cpu().numpy()
    action_norm = np.linalg.norm(action_np, axis=-1)
    action_delta = np.zeros_like(action_norm)
    if len(action_norm) > 1:
        action_delta[1:] = np.linalg.norm(action_np[1:] - action_np[:-1], axis=-1)

    emb_np = emb[0].detach().cpu().numpy()
    latent_disp = np.zeros(total_steps, dtype=np.float32)
    if total_steps > 1:
        latent_disp[1:] = np.linalg.norm(emb_np[1:] - emb_np[:-1], axis=-1)

    valid_mask = np.isfinite(w_series)
    stats = {
        "episode_id": int(episode["episode_id"]),
        "num_steps": int(total_steps),
        "valid_steps": int(valid_mask.sum()),
        "w_mean": float(np.nanmean(w_series)),
        "w_std": float(np.nanstd(w_series)),
        "critical_mean": float(np.nanmean(critical_series)),
        "critical_std": float(np.nanstd(critical_series)),
        "corr_w_action_norm": float(np.corrcoef(w_series[valid_mask], action_norm[valid_mask])[0, 1]),
        "corr_w_latent_disp": float(np.corrcoef(w_series[valid_mask], latent_disp[valid_mask])[0, 1]),
    }
    if g_s_series is not None:
        stats["gS_mean"] = float(np.nanmean(g_s_series))
        stats["gS_std"] = float(np.nanstd(g_s_series))

    return {
        **stats,
        "step_idx": episode["step_idx"],
        "raw_pixels": episode["raw_pixels"],
        "w_t": w_series,
        "critical_t": critical_series,
        "gA_t": g_a_series,
        "gS_t": g_s_series,
        "action_norm": action_norm,
        "action_delta_norm": action_delta,
        "latent_disp": latent_disp,
    }


def draw_episode_panel(
    analysis: Dict[str, np.ndarray | int | float | Sequence[int]],
    save_path: Path,
    *,
    n_keyframes: int,
    min_keyframe_gap: int,
):
    step_idx = np.asarray(analysis["step_idx"])
    w_series = np.asarray(analysis["w_t"])
    critical = np.asarray(analysis["critical_t"])
    action_norm = np.asarray(analysis["action_norm"])
    action_delta = np.asarray(analysis["action_delta_norm"])
    latent_disp = np.asarray(analysis["latent_disp"])
    raw_pixels = np.asarray(analysis["raw_pixels"])

    valid = np.isfinite(critical)
    critical_for_pick = np.where(valid, critical, -np.inf)
    salient_steps = select_salient_steps(
        critical_for_pick,
        k=n_keyframes,
        min_gap=min_keyframe_gap,
    )
    jump_steps = select_salient_steps(
        action_delta,
        k=min(3, max(1, n_keyframes - 1)),
        min_gap=min_keyframe_gap,
    )

    fig = plt.figure(figsize=(15, 6))
    grid = gridspec.GridSpec(2, max(n_keyframes, 4), height_ratios=[3.3, 1.2], hspace=0.35)
    ax = fig.add_subplot(grid[0, :])

    ax.plot(step_idx, critical, color="#b22222", linewidth=2.0, label=r"$critical_t$")
    ax.plot(step_idx, w_series, color="#1f77b4", linewidth=1.8, label=r"$w_t$")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Gate value")

    proxy_ax = ax.twinx()
    proxy_ax.plot(step_idx, action_norm, color="#ff7f0e", alpha=0.65, linewidth=1.4, label=r"$||a_t||$")
    proxy_ax.plot(step_idx, latent_disp, color="#2ca02c", alpha=0.65, linewidth=1.4, label=r"$||z_t-z_{t-1}||$")
    proxy_ax.set_ylabel("Proxy magnitude")

    for s in salient_steps:
        ax.axvline(step_idx[s], color="#b22222", linestyle="--", alpha=0.25, linewidth=1.0)
    for s in jump_steps:
        ax.axvline(step_idx[s], color="#666666", linestyle=":", alpha=0.35, linewidth=1.0)

    title = (
        f"Episode {analysis['episode_id']}: "
        f"w_mean={analysis['w_mean']:.3f}, "
        f"corr(w,a)={analysis['corr_w_action_norm']:.3f}, "
        f"corr(w,disp)={analysis['corr_w_latent_disp']:.3f}"
    )
    ax.set_title(title)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = proxy_ax.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    for col, step in enumerate(salient_steps[:n_keyframes]):
        key_ax = fig.add_subplot(grid[1, col])
        key_ax.imshow(to_display_image(raw_pixels[step]))
        key_ax.set_title(
            f"t={int(step_idx[step])}\ncrit={critical[step]:.2f}\nw={w_series[step]:.2f}",
            fontsize=9,
        )
        key_ax.axis("off")

    for col in range(len(salient_steps[:n_keyframes]), max(n_keyframes, 4)):
        empty_ax = fig.add_subplot(grid[1, col])
        empty_ax.axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)

    summary = {
        "episode_id": int(analysis["episode_id"]),
        "num_steps": int(analysis["num_steps"]),
        "salient_steps_by_critical": [int(step_idx[idx]) for idx in salient_steps],
        "action_jump_steps": [int(step_idx[idx]) for idx in jump_steps],
        "w_mean": float(analysis["w_mean"]),
        "w_std": float(analysis["w_std"]),
        "critical_mean": float(analysis["critical_mean"]),
        "critical_std": float(analysis["critical_std"]),
        "corr_w_action_norm": float(analysis["corr_w_action_norm"]),
        "corr_w_latent_disp": float(analysis["corr_w_latent_disp"]),
    }
    return summary


def run_aggregate_analysis(args, model, history_size: int, save_dir: Path):
    batch = load_dataset_samples(
        dataset_name=args.dataset,
        state_key=None,
        n_sequences=args.n_sequences,
        history_size=history_size,
        future_steps=args.future_steps,
        frameskip=args.frameskip,
        img_size=args.img_size,
        seed=args.seed,
        device=args.device,
    )
    outputs = encode_sequences(model, batch)

    emb = outputs["emb"]
    action = outputs["action"]
    ctx_emb = emb[:, :history_size, :]
    ctx_action = action[:, :history_size, :]
    ctx_act_emb = outputs["act_emb"][:, :history_size, :]

    if hasattr(model, "predict_with_logvar") and getattr(model, "pred_logvar_proj", None) is not None:
        pred_emb, logvar_hat = model.predict_with_logvar(ctx_emb, ctx_act_emb, detach_logvar_input=False)
        s_t = logvar_hat.squeeze(-1).clamp(min=args.s_min, max=args.s_max) if logvar_hat is not None else None
        title_suffix = r"$\sigma + A_t$"
    else:
        pred_emb = model.predict(ctx_emb, ctx_act_emb)
        s_t = None
        title_suffix = r"$A_t$ only"

    gate = compute_action_gate_metrics_offline(
        model,
        ctx_emb,
        ctx_action,
        pred_emb,
        s_t=s_t,
        delta_scale=args.delta_scale,
        num_delta_samples=args.num_delta_samples,
        delta_norm_floor=args.delta_norm_floor,
        log_a_floor=args.log_a_floor,
        w_min=args.w_min,
        w_max=args.w_max,
    )

    w_t = gate["w_t"]
    critical = gate["critical"]
    action_norm = action[:, :history_size, :].pow(2).sum(dim=-1).sqrt()
    latent_disp = (emb[:, 1 : history_size + 1, :] - emb[:, :history_size, :]).pow(2).sum(dim=-1).sqrt()

    w_flat = w_t.cpu().numpy().reshape(-1)
    critical_flat = critical.cpu().numpy().reshape(-1)
    action_flat = action_norm.cpu().numpy().reshape(-1)
    disp_flat = latent_disp.cpu().numpy().reshape(-1)
    mask = np.isfinite(w_flat) & np.isfinite(action_flat) & np.isfinite(disp_flat)
    w_flat = w_flat[mask]
    critical_flat = critical_flat[mask]
    action_flat = action_flat[mask]
    disp_flat = disp_flat[mask]

    print(f"[visualize_wt] aggregate tokens: {len(w_flat)}")
    print(f"  corr(w_t, action_norm) = {np.corrcoef(w_flat, action_flat)[0, 1]:.3f}")
    print(f"  corr(w_t, latent_disp) = {np.corrcoef(w_flat, disp_flat)[0, 1]:.3f}")

    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    hb = ax.hexbin(action_flat, w_flat, gridsize=50, cmap="YlOrRd", mincnt=1, alpha=0.88)
    ax.set_xlabel(r"Action norm $||a_t||$")
    ax.set_ylabel(r"Adaptive weight $w_t$")
    ax.set_ylim(0.15, 1.05)
    ax.set_title(f"{args.dataset}: {title_suffix} $w_t$ vs action norm")
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label("Count")
    plt.tight_layout()
    fig.savefig(save_dir / "wt_vs_action_norm.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    hb = ax.hexbin(disp_flat, w_flat, gridsize=50, cmap="YlOrRd", mincnt=1, alpha=0.88)
    ax.set_xlabel(r"Latent displacement $||z_t-z_{t-1}||$")
    ax.set_ylabel(r"Adaptive weight $w_t$")
    ax.set_ylim(0.15, 1.05)
    ax.set_title(f"{args.dataset}: {title_suffix} $w_t$ vs latent displacement")
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label("Count")
    plt.tight_layout()
    fig.savefig(save_dir / "wt_vs_latent_disp.png", dpi=220)
    plt.close(fig)

    q25, q50, q75 = np.percentile(action_flat, [25, 50, 75])
    quartiles = {
        "Q1 (low)": action_flat <= q25,
        "Q2": (action_flat > q25) & (action_flat <= q50),
        "Q3": (action_flat > q50) & (action_flat <= q75),
        "Q4 (high)": action_flat > q75,
    }
    fig, ax = plt.subplots(figsize=(6.0, 4.1))
    colors = ["#2ca02c", "#98df8a", "#ff7f0e", "#d62728"]
    for (label, qmask), color in zip(quartiles.items(), colors):
        ax.hist(w_flat[qmask], bins=30, alpha=0.6, label=label, color=color, density=True)
    ax.set_xlabel(r"Adaptive weight $w_t$")
    ax.set_ylabel("Density")
    ax.set_xlim(0.15, 1.05)
    ax.set_title(f"{args.dataset}: {title_suffix} $w_t$ by action quartile")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(save_dir / "wt_histogram_by_action_norm.png", dpi=220)
    plt.close(fig)

    summary = {
        "dataset": args.dataset,
        "mode": "aggregate",
        "gate_type": "sigma_plus_action" if s_t is not None else "action_only",
        "num_tokens": int(len(w_flat)),
        "w_mean": float(np.mean(w_flat)),
        "w_std": float(np.std(w_flat)),
        "critical_mean": float(np.mean(critical_flat)),
        "critical_std": float(np.std(critical_flat)),
        "corr_w_action_norm": float(np.corrcoef(w_flat, action_flat)[0, 1]),
        "corr_w_latent_disp": float(np.corrcoef(w_flat, disp_flat)[0, 1]),
        "quartile_w_mean": {
            label: float(np.mean(w_flat[qmask])) for label, qmask in quartiles.items()
        },
    }
    with open(save_dir / "wt_stats.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def run_trajectory_analysis(args, model, history_size: int, save_dir: Path):
    dataset = build_episode_dataset(args.dataset, args.img_size, frameskip=args.frameskip)
    raw_h5_path = resolve_h5_dataset_path(args.dataset)
    raw_dataset = swm.data.HDF5Dataset(
        path=str(raw_h5_path),
        num_steps=1,
        frameskip=args.frameskip,
        transform=None,
    )

    if args.episode_ids:
        episode_ids = [int(ep_id) for ep_id in args.episode_ids]
    else:
        episode_ids = sample_episode_ids(
            raw_dataset,
            n_episodes=args.n_episodes,
            seed=args.episode_seed,
            history_size=history_size,
        )

    print(f"[visualize_wt] trajectory episodes: {episode_ids}")
    episode_summaries = []
    for episode_id in episode_ids:
        episode = load_episode(dataset, raw_dataset, episode_id, device=args.device)
        analysis = analyze_episode_trajectory(
            model,
            episode,
            history_size=history_size,
            delta_scale=args.delta_scale,
            num_delta_samples=args.num_delta_samples,
            delta_norm_floor=args.delta_norm_floor,
            log_a_floor=args.log_a_floor,
            w_min=args.w_min,
            w_max=args.w_max,
            s_min=args.s_min,
            s_max=args.s_max,
        )
        fig_path = save_dir / f"wt_episode_{episode_id}.png"
        summary = draw_episode_panel(
            analysis,
            fig_path,
            n_keyframes=args.n_keyframes,
            min_keyframe_gap=args.min_keyframe_gap,
        )
        episode_summaries.append(summary)
        with open(save_dir / f"wt_episode_{episode_id}.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)

    with open(save_dir / "wt_episode_summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset": args.dataset,
                "history_size": history_size,
                "episodes": episode_summaries,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )


def main():
    args = build_parser().parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.ckpt, args.device)
    model.eval().requires_grad_(False)
    history_size = infer_history_size(model)
    print(f"[visualize_wt] model loaded, history_size={history_size}")

    if args.mode in {"aggregate", "both"}:
        run_aggregate_analysis(args, model, history_size, save_dir)
    if args.mode in {"trajectory", "both"}:
        run_trajectory_analysis(args, model, history_size, save_dir)

    print("[visualize_wt] done")


if __name__ == "__main__":
    main()
