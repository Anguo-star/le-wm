"""Offline DGC (Diagnostic-Gated Consistency) signal probe.

Loads a trained checkpoint, runs one batch through the model, and computes the
candidate DGC gate signal:

    fragile_t = || predict(noisy_z, a) - predict(clean_z, a) || / batch_knn_dist(z_clean)

Optionally, when the checkpoint has a sigma-probe head and an action embedder,
also reproduces a *batch-local* version of the existing AAAC criticality
(critical = gA * (0.5 + 0.5 * gS) with K=1 action sensitivity and batch-zscore,
i.e. no EMA), then reports Pearson/Spearman correlations between the two
signals and dumps a scatter plot.

This is a fast (< 1 min on one GPU) sanity check used before kicking off any
DGC training run.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F

from tools.repr_analysis.analyze_repr import (
    infer_history_size,
    load_dataset_samples,
    load_model,
)


# Mirror train.batch_knn_distance so this script has no train.py import-time
# side-effect (train.py constructs Hydra defaults on import).
def batch_knn_distance(z: torch.Tensor, k: int = 5, eps: float = 1e-8) -> torch.Tensor:
    B, T, D = z.shape
    flat = z.reshape(B * T, D)
    dist = torch.cdist(flat, flat)
    big = dist.new_full((B * T,), float("inf"))
    dist = dist + torch.diag(big)
    knn = dist.topk(k, largest=False).values
    return knn.mean(-1).reshape(B, T).clamp(min=eps)


@torch.no_grad()
def _encode(model, batch):
    out = model.encode({"pixels": batch["pixels"], "action": batch["action"]})
    return out


@torch.no_grad()
def _predict_with_optional_logvar(model, ctx_emb, act_emb):
    """Return (pred_emb, logvar_hat or None)."""
    if hasattr(model, "predict_with_logvar"):
        try:
            return model.predict_with_logvar(ctx_emb, act_emb, detach_logvar_input=True)
        except Exception:
            pass
    return model.predict(ctx_emb, act_emb), None


@torch.no_grad()
def _add_pixel_noise(pixels: torch.Tensor, std_max: float) -> torch.Tensor:
    if std_max <= 0:
        return pixels
    std = pixels.new_empty(pixels.shape[:2] + (1, 1, 1)).uniform_(0.0, float(std_max))
    return pixels + torch.randn_like(pixels) * std


def _zscore(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean()) / x.std(unbiased=False).clamp(min=1e-6)


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp(min=1e-8)
    return float((a * b).sum() / denom)


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ar = a.flatten().float().argsort().argsort().float()
    br = b.flatten().float().argsort().argsort().float()
    return _pearson(ar, br)


def run(
    ckpt: str,
    dataset: str,
    *,
    n_sequences: int = 256,
    noise_std_max: float = 0.04,
    delta_scale: float = 0.25,
    knn_k: int = 5,
    seed: int = 3072,
    device: str = "cuda",
    save_dir: str | None = None,
    data_name: str | None = None,
    frameskip: int = 1,
) -> Dict[str, float]:
    model = load_model(ckpt, device)
    history_size = infer_history_size(model)
    batch = load_dataset_samples(
        dataset_name=data_name if data_name is not None else dataset,
        state_key=None,
        n_sequences=n_sequences,
        history_size=history_size,
        future_steps=history_size,  # plenty for ctx
        frameskip=frameskip,
        img_size=224,
        seed=seed,
        device=device,
    )

    # Clean forward
    clean_out = _encode(model, batch)
    emb_clean = clean_out["emb"]                       # (B, T, D)
    act_emb = clean_out["act_emb"]
    ctx_emb_clean = emb_clean[:, :history_size]
    ctx_act = act_emb[:, :history_size]

    pred_clean, logvar_hat = _predict_with_optional_logvar(model, ctx_emb_clean, ctx_act)

    # Noisy forward (for fragility)
    noisy_batch = dict(batch)
    noisy_batch["pixels"] = _add_pixel_noise(batch["pixels"], noise_std_max)
    noisy_out = _encode(model, noisy_batch)
    ctx_emb_noisy = noisy_out["emb"][:, :history_size]
    pred_noisy = model.predict(ctx_emb_noisy, ctx_act)

    # DGC fragility signal
    target_shift = (pred_noisy - pred_clean).pow(2).sum(-1).clamp(min=0).sqrt()  # (B, T_ctx)
    nn_dist = batch_knn_distance(ctx_emb_clean, k=knn_k)
    fragile_t = target_shift / nn_dist.clamp(min=1e-6)
    log_fragile = torch.log(fragile_t.clamp(min=1e-8))

    # K=1 action sensitivity (cheap reproduction of A channel, no EMA)
    raw_action = torch.nan_to_num(batch["action"], 0.0)[:, :history_size]
    action_std = raw_action.float().std(dim=(0, 1), unbiased=False).clamp(min=1e-6)
    delta = torch.randn_like(raw_action) * (delta_scale * action_std)
    # Encode perturbed action through model.action_encoder if it exists, else
    # fall back to act_emb + delta projection. Most LeWM models expose
    # action_encoder.
    if hasattr(model, "action_encoder"):
        act_emb_pert = model.action_encoder(raw_action + delta)
    else:
        # No action encoder: skip A_t channel.
        act_emb_pert = None
    if act_emb_pert is not None:
        pred_pert = model.predict(ctx_emb_clean, act_emb_pert)
        diff = (pred_pert - pred_clean).pow(2).sum(-1).clamp(min=0).sqrt()
        delta_norm = delta.pow(2).sum(-1).clamp(min=0).sqrt().clamp(min=1e-6)
        A_t_k1 = diff / delta_norm
        log_A = torch.log(A_t_k1.clamp(min=1e-8))
        gA = torch.sigmoid(_zscore(log_A))
    else:
        gA = None
        log_A = None

    # Optional sigma channel
    if logvar_hat is not None:
        s_t = logvar_hat.squeeze(-1).clamp(min=-4.0, max=4.0)
        gS = torch.sigmoid(_zscore(s_t))
    else:
        s_t = None
        gS = None

    # Reproduce critical_old (batch-local, no EMA — purely diagnostic)
    if gA is not None and gS is not None:
        critical_old = gA * (0.5 + 0.5 * gS)
    elif gA is not None:
        critical_old = gA * 0.5
    else:
        critical_old = None

    # DGC critical: normalize log_fragile the same way (batch-zscore + sigmoid)
    gA_dgc = torch.sigmoid(_zscore(log_fragile))
    if gS is not None:
        critical_dgc = gA_dgc * (0.5 + 0.5 * gS)
    else:
        critical_dgc = gA_dgc * 0.5

    # Distribution / correlation stats
    stats: Dict[str, float] = {
        "ckpt": ckpt,
        "dataset": dataset,
        "n_sequences": n_sequences,
        "noise_std_max": noise_std_max,
        "history_size": history_size,
        "n_tokens": int(fragile_t.numel()),
        "fragile_median": float(fragile_t.median()),
        "fragile_q10": float(torch.quantile(fragile_t.flatten().float(), 0.10)),
        "fragile_q90": float(torch.quantile(fragile_t.flatten().float(), 0.90)),
        "fragile_q90_over_q10": float(
            torch.quantile(fragile_t.flatten().float(), 0.90)
            / torch.quantile(fragile_t.flatten().float(), 0.10).clamp(min=1e-8)
        ),
        "log_fragile_std": float(log_fragile.std(unbiased=False)),
    }
    if critical_old is not None:
        stats["critical_old_median"] = float(critical_old.median())
        stats["pearson_critical_old_vs_dgc"] = _pearson(critical_old, critical_dgc)
        stats["spearman_critical_old_vs_dgc"] = _spearman(critical_old, critical_dgc)
        stats["pearson_fragile_vs_A_t_k1"] = _pearson(fragile_t, A_t_k1) if log_A is not None else float("nan")
        stats["spearman_fragile_vs_A_t_k1"] = _spearman(fragile_t, A_t_k1) if log_A is not None else float("nan")
    if gS is not None:
        stats["pearson_fragile_vs_sigma"] = _pearson(fragile_t, s_t)
        stats["spearman_fragile_vs_sigma"] = _spearman(fragile_t, s_t)

    # Optional scatter / histogram
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].hist(fragile_t.flatten().cpu().numpy(), bins=60, color="C0", alpha=0.8)
            axes[0].set_yscale("log")
            axes[0].set_xlabel("fragile_t  =  ||pred_noisy - pred_clean|| / nn_dist")
            axes[0].set_ylabel("count (log)")
            axes[0].set_title("fragility distribution")
            if critical_old is not None:
                axes[1].scatter(
                    critical_old.flatten().cpu().numpy(),
                    critical_dgc.flatten().cpu().numpy(),
                    s=4, alpha=0.4,
                )
                axes[1].set_xlabel("critical (current AAAC) = gA·(0.5+0.5·gS)")
                axes[1].set_ylabel("critical (DGC) = gA_fragile·(0.5+0.5·gS)")
                axes[1].set_title(
                    f"Pearson={stats['pearson_critical_old_vs_dgc']:+.2f}, "
                    f"Spearman={stats['spearman_critical_old_vs_dgc']:+.2f}"
                )
            fig.tight_layout()
            fig.savefig(save_dir / "dgc_offline_scatter.png", dpi=130)
            plt.close(fig)
        except ImportError:
            pass

        with open(save_dir / "dgc_offline_stats.json", "w") as fp:
            json.dump(stats, fp, indent=2)

    return stats


def build_parser():
    p = argparse.ArgumentParser(description="Offline DGC fragility probe.")
    p.add_argument("--ckpt", required=True, help="Path to model_object.ckpt (or .pt).")
    p.add_argument("--dataset", required=True, choices=["tworoom", "pusht", "reacher", "cube"])
    p.add_argument("--data-name", default=None, help="Override HDF5Dataset name (e.g. pusht_expert_train).")
    p.add_argument("--frameskip", type=int, default=1, help="Frameskip for action dim expansion.")
    p.add_argument("--n-sequences", type=int, default=256)
    p.add_argument("--noise-std-max", type=float, default=0.04)
    p.add_argument("--delta-scale", type=float, default=0.25)
    p.add_argument("--knn-k", type=int, default=5)
    p.add_argument("--seed", type=int, default=3072)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-dir", default=None, help="Where to write stats JSON + scatter PNG.")
    return p


def main():
    args = build_parser().parse_args()
    stats = run(
        ckpt=args.ckpt,
        dataset=args.dataset,
        n_sequences=args.n_sequences,
        noise_std_max=args.noise_std_max,
        delta_scale=args.delta_scale,
        knn_k=args.knn_k,
        seed=args.seed,
        device=args.device,
        save_dir=args.save_dir,
        data_name=args.data_name,
        frameskip=args.frameskip,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
