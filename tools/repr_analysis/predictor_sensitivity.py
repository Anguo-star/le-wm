"""
predictor_sensitivity.py — Diagnose how the predictor amplifies pixel noise.

Companion to `noise_sensitivity.py` (which only measures encoder shift).
This tool measures the latent drift of the predictor's own outputs when its
history input is corrupted — the layer between encoder shift and planner
failure under pixel-only / pix+goal noise.

References:
    - `predictor_target_shift` (single-step open-loop prediction error):
        analogous to single-step rollout MSE reported in Dreamer (Hafner
        et al., ICLR 2020/21) and TD-MPC2 (Hansen et al., ICLR 2024).
        Recently used as a runtime noise indicator by
        "World Model Robustness via Surprise Recognition", arXiv:2512.01119
        (2025); we use it pre-hoc, on calibration data, as a failure
        predictor (inline metric definitions).
    - `predictor_rollout_drift(T)` (autoregressive multi-step latent drift
        between noisy- and clean-history conditioning): named here; not
        identified in prior work.
    - Normalization by `clean_nn_cos_dist` reuses the KNN distance primitive
        from KNN-OOD (Sun et al., NeurIPS 2022).

Two complementary measurements per std:

1. `predictor_target_shift` (open-loop, single-step):
       For each H-window, run predict() on clean and noisy history with the
       same actions. Compare the last predicted token. Isolates the
       predictor's local sensitivity without rollout compounding.

2. `predictor_rollout_drift(T)` (autoregressive, T steps):
       Initialize chain_clean = clean_emb[:H], chain_noisy = noisy_emb[:H].
       Roll out predict() T steps with the dataset's own action sequence.
       Drift at step t = ||chain_noisy[H+t] - chain_clean[H+t]||.

Both are reported in the model's inference cost space (cosine for SWM,
raw for LeWM) and are normalized by the clean nearest-neighbor distance
for cross-model comparability.

Notebook use:

    from tools.repr_analysis.predictor_sensitivity import run_predictor_sensitivity
    rows = run_predictor_sensitivity(
        models={"swm": "/path/...", "lewm": "/path/..."},
        dataset="tworoom",
        stds=[0.0, 0.005, 0.01, 0.02, 0.05],
        rollout_steps=[1, 2, 4, 8],
        history_noise_only=True,
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
from utils import make_eval_corruption


def _clone_batch(batch: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}


def _add_noise(x: torch.Tensor, magnitude: float, seed: int,
               corruption_type: str = "gaussian_noise") -> torch.Tensor:
    """Apply the configured corruption family at a single magnitude.
    Named ``_add_noise`` for symmetry with the rest of the diagnostic
    suite even though it now also handles blur / resize."""
    transform = make_eval_corruption(magnitude, corruption_type)
    if transform is None:
        return x.clone()
    with torch.random.fork_rng(devices=[x.device] if x.device.type == "cuda" else []):
        torch.manual_seed(seed)
        return transform(x)


def _safe_quantile(x: torch.Tensor, q: float) -> float:
    if x.numel() == 0:
        return float("nan")
    return float(torch.quantile(x.float().cpu(), q))


def _shift_stats(clean: torch.Tensor, noisy: torch.Tensor) -> Dict[str, float]:
    """Per-token L2 / cosine shift between clean and noisy embedding tensors."""
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


@torch.no_grad()
def _open_loop_target_shift(
    model,
    clean_emb: torch.Tensor,
    noisy_emb: torch.Tensor,
    act_emb: torch.Tensor,
    history_size: int,
) -> Dict[str, torch.Tensor]:
    """Single-step shift over all valid H-windows in the sequence."""
    B, T, _ = clean_emb.shape
    H = history_size
    if T <= H:
        return {"clean_pred": clean_emb[:, :0], "noisy_pred": noisy_emb[:, :0]}

    clean_preds, noisy_preds = [], []
    for s in range(T - H):
        c_win = clean_emb[:, s : s + H]
        n_win = noisy_emb[:, s : s + H]
        a_win = act_emb[:, s : s + H]
        c_pred = model.predict(c_win, a_win)[:, -1]  # (B, D)
        n_pred = model.predict(n_win, a_win)[:, -1]
        clean_preds.append(c_pred)
        noisy_preds.append(n_pred)
    return {
        "clean_pred": torch.stack(clean_preds, dim=1),  # (B, T-H, D)
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
    """Predict n_steps autoregressively starting from init_emb (last H frames)."""
    H = history_size
    chain = init_emb.clone()  # (B, H, D)
    for t in range(n_steps):
        a_win = act_emb[:, t : t + H]
        if a_win.size(1) < H:
            break
        pred = model.predict(chain[:, -H:], a_win)[:, -1:]
        chain = torch.cat([chain, pred], dim=1)
    return chain  # (B, H + steps, D)


def _make_history_noise_batch(
    batch: Mapping[str, torch.Tensor],
    history_size: int,
    std: float,
    seed: int,
    history_noise_only: bool,
    corruption_type: str = "gaussian_noise",
) -> Dict[str, torch.Tensor]:
    out = _clone_batch(batch)
    # The "is corruption a no-op" check has to know the family because
    # the zero-magnitude convention differs:
    #   gaussian_noise: std <= 0
    #   gaussian_blur:  kernel_size <= 1
    #   resize:         factor >= 1.0
    # _add_noise handles all three internally, but a cheap short-circuit
    # here avoids an unnecessary clone for the common clean-frame case.
    if corruption_type == "gaussian_noise" and std <= 0:
        return out
    if corruption_type == "gaussian_blur" and std <= 1:
        return out
    if corruption_type == "resize" and std >= 1.0:
        return out
    pixels = out["pixels"]
    if history_noise_only:
        H = history_size
        h_noisy = _add_noise(pixels[:, :H], std, seed, corruption_type=corruption_type)
        out["pixels"] = torch.cat([h_noisy, pixels[:, H:]], dim=1)
    else:
        out["pixels"] = _add_noise(pixels, std, seed, corruption_type=corruption_type)
    return out


def analyze_model_predictor_noise(
    *,
    label: str,
    ckpt: str,
    batch: Mapping[str, torch.Tensor],
    stds: Sequence[float],
    rollout_steps: Sequence[int],
    embedding_space: str | None = None,
    history_noise_only: bool = True,
    seed: int = 3072,
    device: str = "cuda",
    corruption_type: str = "gaussian_noise",
) -> list[Dict[str, Any]]:
    model = load_model(ckpt, device)
    spaces = get_model_spaces(model)
    space = resolve_space_name(embedding_space or spaces["inference_cost_space"])
    history_size = infer_history_size(model)

    # encode clean once
    clean_outputs = encode_sequences(model, _clone_batch(batch))
    clean_emb = get_embedding_space(clean_outputs, space).detach()
    act_emb = clean_outputs["act_emb"].detach()

    # clean NN distance reference (used as denominator for normalized drift)
    nn_ref = _clean_nn_dist(clean_emb)

    rows: list[Dict[str, Any]] = []
    max_steps = max(rollout_steps) if rollout_steps else 0

    for std_idx, std in enumerate(stds):
        noisy_batch = _make_history_noise_batch(
            batch, history_size, float(std), seed + 1009 * std_idx, history_noise_only,
            corruption_type=corruption_type,
        )
        noisy_outputs = encode_sequences(model, noisy_batch)
        noisy_emb = get_embedding_space(noisy_outputs, space).detach()

        # 1. Open-loop single-step target shift
        ol = _open_loop_target_shift(model, clean_emb, noisy_emb, act_emb, history_size)
        if ol["clean_pred"].numel() > 0:
            target_stats = _shift_stats(ol["clean_pred"], ol["noisy_pred"])
        else:
            target_stats = {k: float("nan") for k in (
                "cos_dist_median", "cos_dist_p90", "angle_deg_median",
                "angle_deg_p90", "l2_median", "l2_p90",
            )}

        # 2. Autoregressive rollout drift at each requested step
        rollout_drifts: Dict[str, float] = {}
        if max_steps > 0:
            init_clean = clean_emb[:, :history_size]
            init_noisy = noisy_emb[:, :history_size]
            chain_clean = _autoregressive_rollout(
                model, init_clean, act_emb, history_size, max_steps
            )
            chain_noisy = _autoregressive_rollout(
                model, init_noisy, act_emb, history_size, max_steps
            )
            achieved = chain_clean.size(1) - history_size  # steps actually rolled
            for T in rollout_steps:
                if T > achieved:
                    rollout_drifts[f"rollout_T{T}_l2_median"] = float("nan")
                    rollout_drifts[f"rollout_T{T}_cos_dist_median"] = float("nan")
                    rollout_drifts[f"rollout_T{T}_angle_deg_median"] = float("nan")
                    continue
                idx = history_size + T - 1  # 1-based T -> index T-1 past history
                stat = _shift_stats(
                    chain_clean[:, idx : idx + 1], chain_noisy[:, idx : idx + 1]
                )
                rollout_drifts[f"rollout_T{T}_l2_median"] = stat["l2_median"]
                rollout_drifts[f"rollout_T{T}_cos_dist_median"] = stat["cos_dist_median"]
                rollout_drifts[f"rollout_T{T}_angle_deg_median"] = stat["angle_deg_median"]

        # Normalize drifts by clean NN distance for cross-model comparison
        cos_norm = nn_ref["cos"] if nn_ref["cos"] and nn_ref["cos"] > 0 else float("nan")
        l2_norm = nn_ref["l2"] if nn_ref["l2"] and nn_ref["l2"] > 0 else float("nan")
        target_to_nn_cos = (
            target_stats["cos_dist_median"] / cos_norm
            if not math.isnan(cos_norm) else float("nan")
        )
        target_to_nn_l2 = (
            target_stats["l2_median"] / l2_norm
            if not math.isnan(l2_norm) else float("nan")
        )

        rows.append(
            {
                "model": label,
                "ckpt": ckpt,
                "std": float(std),
                "history_noise_only": bool(history_noise_only),
                "embedding_space": space,
                "history_size": int(history_size),
                **{f"target_{k}": v for k, v in target_stats.items()},
                "target_to_nn_cos_ratio": float(target_to_nn_cos),
                "target_to_nn_l2_ratio": float(target_to_nn_l2),
                **rollout_drifts,
                "clean_nn_cos_dist_median": float(nn_ref["cos"]),
                "clean_nn_l2_median": float(nn_ref["l2"]),
            }
        )

    return rows


def run_predictor_sensitivity(
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
    history_noise_only: bool = True,
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
            analyze_model_predictor_noise(
                label=label,
                ckpt=ckpt,
                batch=batch,
                stds=stds,
                rollout_steps=rollout_steps,
                embedding_space=embedding_space,
                history_noise_only=history_noise_only,
                seed=seed,
                device=device,
                corruption_type=corruption_type,
            )
        )
    return rows


def format_predictor_table(rows: Sequence[Mapping[str, Any]]):
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("format_predictor_table requires pandas.") from exc

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    columns = [c for c in [
        "model", "std", "history_noise_only", "embedding_space",
        "target_angle_deg_median", "target_cos_dist_median", "target_l2_median",
        "target_to_nn_cos_ratio", "target_to_nn_l2_ratio",
        "rollout_T1_angle_deg_median", "rollout_T2_angle_deg_median",
        "rollout_T4_angle_deg_median", "rollout_T8_angle_deg_median",
        "clean_nn_cos_dist_median", "clean_nn_l2_median",
    ] if c in df.columns]
    df = df[columns].sort_values(["model", "std"]).reset_index(drop=True)
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].round(4)
    return df


def _parse_model_specs(specs: Sequence[str]) -> Dict[str, str]:
    models: Dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Model spec must be label=/path/to/ckpt, got: {spec}")
        label, ckpt = spec.split("=", 1)
        models[label.strip()] = ckpt.strip()
    return models


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Diagnose predictor sensitivity to history-pixel noise.")
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
    p.add_argument("--history-noise-only", action="store_true", default=True,
                   help="Only add noise to history frames (default; matches pixels-only eval).")
    p.add_argument("--full-noise", dest="history_noise_only", action="store_false",
                   help="Add noise to all frames including goal (matches pix+goal eval).")
    p.add_argument("--seed", type=int, default=3072)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-dir", default=None)
    return p


def main():
    args = build_parser().parse_args()
    rows = run_predictor_sensitivity(
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
        history_noise_only=args.history_noise_only,
        seed=args.seed,
        device=args.device,
    )

    print(format_predictor_table(rows).to_string(index=False))

    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        with (save_dir / "predictor_sensitivity.json").open("w") as f:
            json.dump(to_serializable(rows), f, indent=2)
        with (save_dir / "predictor_sensitivity.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[predictor_sensitivity] saved outputs to: {save_dir}")


if __name__ == "__main__":
    main()
