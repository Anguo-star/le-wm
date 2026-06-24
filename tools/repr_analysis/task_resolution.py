"""
task_resolution.py — Diagnose how much state resolution a latent retains.

Companion to noise_sensitivity (encoder shift) and predictor_sensitivity
(predictor drift). This tool quantifies whether the encoder preserves the
fine-grained state distinctions a controller needs.

References for the indicators emitted here:
    - `clean_effective_rank`: matrix entropy of singular values.
        Garrido et al., RankMe, ICML 2023.
    - `lidar_rank`: positive-pair-aware rank using Linear Discriminant
        Analysis structure.
        Thilak et al., "LiDAR: Sensing Linear Probing Performance in Joint
        Embedding SSL Architectures", ICLR 2024.
    - `id_probe_r2*`: inverse-dynamics linear probe.
        Brandfonbrener et al., "Inverse Dynamics Pretraining Learns Good
        Representations", NeurIPS 2023; Pathak et al., "Curiosity-Driven
        Exploration / ICM", ICML 2017.
        Linear-probing methodology: Alain & Bengio, "Linear classifier
        probes", ICLR-W 2017.
    - `consecutive_*` / `far_*` distances: latent neighbor distance primitive
        (Sun et al., KNN-OOD, NeurIPS 2022); the pair distance median is
        a discrete analogue of Wang & Isola L_uniform (ICML 2020).
    - `transition_resolution_ratio_*`: temporal-neighbor variant of the
        retrieval intra/inter-class gap; named here. See research_notebook_swm.md §7.2.

Two indicators per checkpoint × dataset:

1. `transition_resolution_ratio` (label-free):
       median d(z_t, z_{t+1}) over consecutive pairs
       median d(z_t, z_{t'}) over random cross-sequence pairs
       ratio = consecutive / far. Smaller ratio means transitions are clearly
       separated from random pairs. Conceptually a temporal-neighbor variant
       of retrieval intra/inter gap; complements `effective_rank` (separates
       collapse from clustering).

2. Inverse-dynamics linear probe accuracy (action readout):
       Closed-form ridge regression W from concat(z_t, z_{t+1}) to action_t.
       Reports train R^2 (overall and per-dim). Acts as a label-free proxy
       for "how much controllable state survives in the latent". The probe is
       readout-only; the world model is not updated. Inspired by Brandfonbrener
       et al. (NeurIPS 2023): inverse-dynamics is a strong probing signal that
       recovers the controllable state when the encoder is in the function
       class.

Outputs are reported in both the model's inference cost space (cosine for
SWM, raw for LeWM) and the L2 metric.

Notebook use:

    from tools.repr_analysis.task_resolution import run_task_resolution
    rows = run_task_resolution(
        models={"swm": "/path/...", "lewm": "/path/..."},
        dataset="tworoom",
    )
"""

from __future__ import annotations

import argparse
import csv
import json
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


def _safe_quantile(x: torch.Tensor, q: float) -> float:
    if x.numel() == 0:
        return float("nan")
    return float(torch.quantile(x.float().cpu(), q))


def _consecutive_pairs(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (z_t, z_{t+1}) flattened across (B, T-1) consecutive pairs."""
    a = z[:, :-1].reshape(-1, z.size(-1))
    b = z[:, 1:].reshape(-1, z.size(-1))
    return a, b


def _far_pair_distances(z: torch.Tensor, n_pairs: int, seed: int) -> Dict[str, torch.Tensor]:
    """Sample random *cross-sequence* pairs (different episodes), compute distances."""
    B, T, D = z.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    seq_a = torch.randint(0, B, (n_pairs,), generator=g)
    seq_b = torch.randint(0, B, (n_pairs,), generator=g)
    same = seq_a == seq_b
    seq_b[same] = (seq_b[same] + 1) % B  # ensure different sequence
    t_a = torch.randint(0, T, (n_pairs,), generator=g)
    t_b = torch.randint(0, T, (n_pairs,), generator=g)
    a = z[seq_a, t_a]
    b = z[seq_b, t_b]
    cos = (F.normalize(a, dim=-1, eps=1e-8) * F.normalize(b, dim=-1, eps=1e-8)).sum(-1).clamp(-1.0, 1.0)
    cos_dist = (1.0 - cos).clamp_min(0.0)
    l2 = torch.linalg.vector_norm(a - b, dim=-1)
    return {"cos_dist": cos_dist, "l2": l2}


def _transition_metrics(z: torch.Tensor, *, n_far_pairs: int, seed: int) -> Dict[str, float]:
    a, b = _consecutive_pairs(z)
    if a.numel() == 0:
        return {
            "consecutive_cos_dist_median": float("nan"),
            "consecutive_l2_median": float("nan"),
            "far_cos_dist_median": float("nan"),
            "far_l2_median": float("nan"),
            "transition_resolution_ratio_cos": float("nan"),
            "transition_resolution_ratio_l2": float("nan"),
        }
    cos = (F.normalize(a, dim=-1, eps=1e-8) * F.normalize(b, dim=-1, eps=1e-8)).sum(-1).clamp(-1.0, 1.0)
    cons_cos = (1.0 - cos).clamp_min(0.0)
    cons_l2 = torch.linalg.vector_norm(b - a, dim=-1)

    far = _far_pair_distances(z, n_far_pairs, seed)
    cons_cos_med = _safe_quantile(cons_cos, 0.5)
    cons_l2_med = _safe_quantile(cons_l2, 0.5)
    far_cos_med = _safe_quantile(far["cos_dist"], 0.5)
    far_l2_med = _safe_quantile(far["l2"], 0.5)

    def _ratio(num: float, den: float) -> float:
        if den is None or den == 0 or den != den:  # NaN-safe
            return float("nan")
        return num / den

    return {
        "consecutive_cos_dist_median": cons_cos_med,
        "consecutive_l2_median": cons_l2_med,
        "far_cos_dist_median": far_cos_med,
        "far_l2_median": far_l2_med,
        "transition_resolution_ratio_cos": _ratio(cons_cos_med, far_cos_med),
        "transition_resolution_ratio_l2": _ratio(cons_l2_med, far_l2_med),
    }


def _lidar_rank(z: torch.Tensor, eps: float = 1e-3) -> float:
    """LiDAR rank (Thilak et al., ICLR 2024) with temporal positive pairs.

    Treats each (z_t, z_{t+1}) as a 2-sample positive class (these *should*
    map close together because they share state context). Computes:

        S_w = average per-pair within-class scatter
        S_b = scatter of pair midpoints (between-class)
        L   = S_w^{-1/2} S_b S_w^{-1/2}
        LiDAR rank = entropy effective rank of singular values of L

    A high LiDAR rank means many directions both (a) discriminate between
    different *transitions* and (b) are stable within a transition. RankMe /
    effective_rank is `unsupervised`; LiDAR adds a positive-pair structure
    that better predicts downstream linear-probe quality.

    Caveat: with only 2 samples per class, S_w is rank-deficient — the eps
    regularization is essential. Result is comparable across models trained
    on the same data, not as an absolute number.

    z: (B, T, D)
    """
    if z.size(1) < 2:
        return float("nan")
    a = z[:, :-1].reshape(-1, z.size(-1))
    b = z[:, 1:].reshape(-1, z.size(-1))
    half_diff = (b - a) / 2.0  # (M, D); each class contributes ±half_diff around midpoint
    midpoint = (a + b) / 2.0    # (M, D)
    M, D = half_diff.shape
    if M < 2:
        return float("nan")
    Sw = (half_diff.T @ half_diff) / float(M)  # average within-class scatter
    centered = midpoint - midpoint.mean(0, keepdim=True)
    Sb = (centered.T @ centered) / float(M)

    Sw_reg = Sw + eps * torch.eye(D, device=z.device, dtype=Sw.dtype)
    eigvals, eigvecs = torch.linalg.eigh(Sw_reg)
    eigvals = eigvals.clamp_min(1e-8)
    Sw_inv_sqrt = (eigvecs * eigvals.rsqrt().unsqueeze(0)) @ eigvecs.T

    Lidar = Sw_inv_sqrt @ Sb @ Sw_inv_sqrt
    sv = torch.linalg.svdvals(Lidar)
    # Match `effective_rank`: entropy over singular values, not squared energy,
    # so the reported rank follows the RankMe / effective-rank convention.
    total = sv.sum()
    if float(total) < 1e-12:
        return 0.0
    p = sv / total
    entropy = -(p * torch.log(p.clamp_min(1e-12))).sum()
    return float(torch.exp(entropy))


def _ridge_probe(
    X: torch.Tensor, Y: torch.Tensor, *, ridge: float = 1e-3
) -> Dict[str, float]:
    """Closed-form ridge regression. Reports overall and per-dim R^2.

    Methodology: Alain & Bengio (ICLR-W 2017) linear probing applied to
    inverse dynamics (Pathak et al., ICML 2017). Justified as a resolution
    proxy by Brandfonbrener et al. (NeurIPS 2023): under realizability the
    ID linear probe recovers the controllable state from the encoder.
    """
    X = X.float()
    Y = Y.float()
    if X.numel() == 0 or Y.numel() == 0:
        return {"id_probe_r2": float("nan"), "id_probe_r2_min": float("nan")}
    # Augment with bias column
    X_aug = torch.cat([X, torch.ones(X.size(0), 1, device=X.device)], dim=1)
    A = X_aug.T @ X_aug
    A = A + ridge * torch.eye(A.size(0), device=A.device)
    B = X_aug.T @ Y
    W = torch.linalg.solve(A, B)
    Y_hat = X_aug @ W
    Y_centered = Y - Y.mean(0, keepdim=True)
    ss_tot = (Y_centered ** 2).sum(0).clamp_min(1e-12)
    ss_res = ((Y - Y_hat) ** 2).sum(0)
    r2_per_dim = 1.0 - ss_res / ss_tot
    r2_overall = float(r2_per_dim.mean())
    r2_min = float(r2_per_dim.min())
    return {
        "id_probe_r2": r2_overall,
        "id_probe_r2_min": r2_min,
    }


def _build_id_probe_data(
    z: torch.Tensor, action: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """X = concat(z_t, z_{t+1}), Y = action_t. Drops NaN actions."""
    a, b = _consecutive_pairs(z)
    X = torch.cat([a, b], dim=-1)
    # action is (B, T, A_dim); use action[:, :-1] aligned with consecutive pair index
    Y = action[:, :-1].reshape(-1, action.size(-1))
    valid = ~torch.isnan(Y).any(dim=-1)
    return X[valid], Y[valid]


@torch.no_grad()
def analyze_model_resolution(
    *,
    label: str,
    ckpt: str,
    batch: Mapping[str, torch.Tensor],
    embedding_space: str | None = None,
    n_far_pairs: int = 4096,
    ridge: float = 1e-3,
    seed: int = 3072,
    device: str = "cuda",
) -> Dict[str, Any]:
    model = load_model(ckpt, device)
    spaces = get_model_spaces(model)
    space = resolve_space_name(embedding_space or spaces["inference_cost_space"])

    outputs = encode_sequences(model, {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()})
    z = get_embedding_space(outputs, space).detach()
    action = batch["action"]

    transition = _transition_metrics(z, n_far_pairs=n_far_pairs, seed=seed)

    X, Y = _build_id_probe_data(z, action.to(device))
    probe = _ridge_probe(X, Y, ridge=ridge)

    z_flat = z.reshape(-1, z.size(-1))
    eff_rank = effective_rank(z_flat)
    lidar = _lidar_rank(z)

    return {
        "model": label,
        "ckpt": ckpt,
        "embedding_space": space,
        "n_sequences": int(z.size(0)),
        "history_size": int(infer_history_size(model)),
        **transition,
        **probe,
        "clean_effective_rank": eff_rank,
        "lidar_rank": lidar,
    }


def run_task_resolution(
    *,
    models: Mapping[str, str],
    dataset: str = "tworoom",
    state_key: str | None = None,
    n_sequences: int = 256,
    history_size: int | None = None,
    future_steps: int = 8,
    frameskip: int = 1,
    img_size: int = 224,
    n_far_pairs: int = 4096,
    ridge: float = 1e-3,
    embedding_space: str | None = None,
    seed: int = 3072,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> list[Dict[str, Any]]:
    if not models:
        raise ValueError("models must contain at least one label -> checkpoint path.")

    first_ckpt = next(iter(models.values()))
    first_model = load_model(first_ckpt, device)
    H = history_size or infer_history_size(first_model)
    del first_model

    batch = load_dataset_samples(
        dataset_name=dataset,
        state_key=state_key,
        n_sequences=n_sequences,
        history_size=H,
        future_steps=future_steps,
        frameskip=frameskip,
        img_size=img_size,
        seed=seed,
        device=device,
    )

    rows = []
    for label, ckpt in models.items():
        rows.append(
            analyze_model_resolution(
                label=label,
                ckpt=ckpt,
                batch=batch,
                embedding_space=embedding_space,
                n_far_pairs=n_far_pairs,
                ridge=ridge,
                seed=seed,
                device=device,
            )
        )
    return rows


def format_resolution_table(rows: Sequence[Mapping[str, Any]]):
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("format_resolution_table requires pandas.") from exc

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    columns = [c for c in [
        "model", "embedding_space",
        "consecutive_cos_dist_median", "far_cos_dist_median",
        "transition_resolution_ratio_cos",
        "consecutive_l2_median", "far_l2_median",
        "transition_resolution_ratio_l2",
        "id_probe_r2", "id_probe_r2_min",
        "clean_effective_rank", "lidar_rank",
    ] if c in df.columns]
    df = df[columns].sort_values("model").reset_index(drop=True)
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
    p = argparse.ArgumentParser(description="Diagnose latent state resolution: transition gap + ID linear probe.")
    p.add_argument("--model", action="append", required=True,
                   help="Model spec as label=/path/to/model_object.ckpt. Repeat for comparisons.")
    p.add_argument("--dataset", default="tworoom")
    p.add_argument("--state-key", default=None)
    p.add_argument("--n-sequences", type=int, default=256)
    p.add_argument("--future-steps", type=int, default=8)
    p.add_argument("--frameskip", type=int, default=1)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--n-far-pairs", type=int, default=4096)
    p.add_argument("--ridge", type=float, default=1e-3)
    p.add_argument("--embedding-space", default=None, choices=[None, "raw", "normalized"])
    p.add_argument("--seed", type=int, default=3072)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-dir", default=None)
    return p


def main():
    args = build_parser().parse_args()
    rows = run_task_resolution(
        models=_parse_model_specs(args.model),
        dataset=args.dataset,
        state_key=args.state_key,
        n_sequences=args.n_sequences,
        future_steps=args.future_steps,
        frameskip=args.frameskip,
        img_size=args.img_size,
        n_far_pairs=args.n_far_pairs,
        ridge=args.ridge,
        embedding_space=args.embedding_space,
        seed=args.seed,
        device=args.device,
    )
    print(format_resolution_table(rows).to_string(index=False))

    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        with (save_dir / "task_resolution.json").open("w") as f:
            json.dump(to_serializable(rows), f, indent=2)
        with (save_dir / "task_resolution.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[task_resolution] saved outputs to: {save_dir}")


if __name__ == "__main__":
    main()
