from functools import partial
from pathlib import Path
from contextlib import contextmanager

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger
from torch import nn

try:
    from swanlab.integration.pytorch_lightning import SwanLabLogger
except ImportError:
    SwanLabLogger = None
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
from module import (
    ARPredictor,
    Embedder,
    MLP,
    SIGReg,
    WassersteinSIGReg,
    inverse_dynamics_loss,
    temporal_straightness,
    transition_distance_prediction_loss,
)
from utils import (
    AddNormalizedGaussianNoise,
    get_column_normalizer,
    get_img_noise_transform,
    get_img_preprocessor,
    ModelObjectCallBack,
    resolve_h5_dataset_path,
    TransformDataset,
)


def get_pred_loss_tensor(tensor: torch.Tensor, *, space: str) -> torch.Tensor:
    space = space.lower()
    if space == "raw":
        return tensor
    if space in {"normalized", "l2_norm", "sphere"}:
        return F.normalize(tensor, dim=-1, eps=1e-8)
    raise ValueError(f"Unsupported loss.pred.space: {space}")


def mse_token(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).pow(2).mean(dim=-1)


def self_bounded_aux_loss(
    base_loss: torch.Tensor,
    aux_raw: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = (base_loss.detach() / aux_raw.detach().clamp_min(eps)).clamp(max=1.0)
    return aux_raw * scale, scale


def resolve_norm_fn(norm_name: str):
    """Resolve a config string to an nn norm class (or None for identity).
    Mirrors train_swm.resolve_norm_fn so both training paths accept the same
    encoder.projection_head.norm_fn vocabulary."""
    norm_name = norm_name.lower()
    if norm_name in {"none", "identity"}:
        return None
    if norm_name in {"ln", "layernorm"}:
        return nn.LayerNorm
    if norm_name in {"bn", "batchnorm", "batchnorm1d"}:
        return nn.BatchNorm1d
    raise ValueError(f"Unsupported encoder.projection_head.norm_fn: {norm_name}")


def compute_hetero_pred_loss(
    pred_loss_emb: torch.Tensor,
    tgt_loss_emb: torch.Tensor,
    logvar_hat: torch.Tensor,
    *,
    s_min: float = -4.0,
    s_max: float = 4.0,
    tau_floor: float = 1e-6,
):
    """Scale-preserving heteroscedastic prediction loss (sigma-conditioned JEPA).

    err_token  = mean((mu_hat - mu_target)^2, dim=-1)              # (B, T)
    s          = clamp(logvar_hat.squeeze(-1), s_min, s_max)        # (B, T)
    tau        = stop_grad(mean(err_token))                         # scalar (per-batch)
    hetero     = mean( exp(-s) * err_token + tau * s )

    At s ≡ 0 this equals plain MSE — so SIGReg's relative weight need not be
    retuned. After training, exp(-s) downweights high-error samples (the known
    hard-transition risk; monitor `hetero/weight_q10_q90_ratio` for this).

    Returns (hetero_loss, monitors_dict). The monitors dict carries logging
    fields only — none of them are part of the optimization graph.
    """
    err = (pred_loss_emb - tgt_loss_emb).pow(2).mean(dim=-1)        # (B, T)
    s = logvar_hat.squeeze(-1).clamp(min=s_min, max=s_max)          # (B, T)
    tau = err.detach().mean().clamp(min=tau_floor)
    weight = torch.exp(-s)                                          # exp(-s); large = upweight
    hetero_loss = (weight * err + tau * s).mean()

    with torch.no_grad():
        s_flat = s.reshape(-1)
        w_flat = weight.reshape(-1)
        e_flat = err.reshape(-1)
        # Spearman-ish: Pearson on rank — cheaper to just use Pearson on values
        # since exp/log are monotone, sign tells us the calibration direction.
        s_centered = s_flat - s_flat.mean()
        loge_centered = torch.log(e_flat.clamp(min=tau_floor)) - torch.log(
            e_flat.clamp(min=tau_floor)
        ).mean()
        denom = s_centered.norm() * loge_centered.norm()
        s_loge_corr = (s_centered * loge_centered).sum() / denom.clamp(min=1e-8)
        # Per-sample weight quantiles → measure of hard-transition downweighting.
        q10 = torch.quantile(w_flat, 0.1)
        q90 = torch.quantile(w_flat, 0.9)
        monitors = {
            "hetero_s_mean": s_flat.mean(),
            "hetero_s_std": s_flat.std(unbiased=False),
            "hetero_s_abs_max": s_flat.abs().max(),
            "hetero_weight_mean": w_flat.mean(),
            "hetero_weight_q10": q10,
            "hetero_weight_q90": q90,
            "hetero_weight_q10_q90_ratio": q10 / q90.clamp(min=1e-8),
            "hetero_tau": tau.detach(),
            "hetero_err_mean": e_flat.mean(),
            "hetero_s_logerr_corr": s_loge_corr,
        }
    return hetero_loss, monitors


def compute_sigma_probe_loss(
    pred_loss_emb: torch.Tensor,
    tgt_loss_emb: torch.Tensor,
    logvar_hat: torch.Tensor,
    *,
    s_min: float = -4.0,
    s_max: float = 4.0,
    tau_floor: float = 1e-6,
):
    """Detached sigma calibration loss.

    The prediction error target is detached, and callers should also detach the
    logvar head input so this loss only trains pred_logvar_proj. The mean path
    remains the exact LeWM MSE + SIGReg objective.
    """
    err = (pred_loss_emb.detach() - tgt_loss_emb.detach()).pow(2).mean(dim=-1)
    target_logerr = torch.log(err.clamp(min=tau_floor)).clamp(min=s_min, max=s_max)
    s = logvar_hat.squeeze(-1)
    sigma_probe_loss = F.smooth_l1_loss(s, target_logerr)

    with torch.no_grad():
        s_flat = s.reshape(-1)
        e_flat = err.reshape(-1)
        target_flat = target_logerr.reshape(-1)
        weight = torch.exp(-s.clamp(min=s_min, max=s_max))
        w_flat = weight.reshape(-1)
        s_centered = s_flat - s_flat.mean()
        loge_centered = target_flat - target_flat.mean()
        denom = s_centered.norm() * loge_centered.norm()
        s_loge_corr = (s_centered * loge_centered).sum() / denom.clamp(min=1e-8)
        q10 = torch.quantile(w_flat, 0.1)
        q90 = torch.quantile(w_flat, 0.9)
        monitors = {
            "hetero_s_mean": s_flat.mean(),
            "hetero_s_std": s_flat.std(unbiased=False),
            "hetero_s_abs_max": s_flat.abs().max(),
            "hetero_weight_mean": w_flat.mean(),
            "hetero_weight_q10": q10,
            "hetero_weight_q90": q90,
            "hetero_weight_q10_q90_ratio": q10 / q90.clamp(min=1e-8),
            "hetero_tau": e_flat.mean(),
            "hetero_err_mean": e_flat.mean(),
            "hetero_s_logerr_corr": s_loge_corr,
            "sigma_probe_target_logerr_mean": target_flat.mean(),
            "sigma_probe_target_logerr_std": target_flat.std(unbiased=False),
        }
    return sigma_probe_loss, monitors


def batch_knn_distance(z: torch.Tensor, k: int = 5, eps: float = 1e-8) -> torch.Tensor:
    """Per-token mean Euclidean distance to k nearest non-self tokens in the
    batch. z: (B, T, D) → (B, T). Used by DGC mode to normalize predictor
    target shift against local latent neighborhood scale. See
    plan_adaptive_resolution.md §3.2.5 / §3.8.2 (DGC)."""
    B, T, D = z.shape
    flat = z.reshape(B * T, D)
    dist = torch.cdist(flat, flat)
    big = dist.new_full((B * T,), float("inf"))
    dist = dist + torch.diag(big)
    knn = dist.topk(k, largest=False).values
    return knn.mean(-1).reshape(B, T).clamp(min=eps)


def compute_action_gate_metrics(
    model,
    ctx_emb,
    ctx_action_raw,
    pred_emb_origin,
    s_t,
    *,
    K: int,
    delta_scale: float,
    delta_norm_floor: float,
    log_a_floor: float,
    in_warmup: bool,
    ema_momentum: float,
    w_min: float = 0.2,
    w_max: float = 1.0,
    mode: str = "full",
    intervention: str = "none",
    fragile_t: torch.Tensor = None,
):
    """Logging-only action-aware adaptive resolution gate.

    Computes per-token action sensitivity A_t, combines with sigma probe s_t,
    and emits diagnostic gate statistics. Does NOT modify the training loss.

    All compute is under no_grad and EMA buffers are mutated in-place. Caller
    must provide the model with `gate_log_A_mean`, `gate_log_A_var`,
    `gate_s_mean`, `gate_s_var` buffers (plus `_inited` flags). See
    plan_adaptive_resolution.md §8.3.2 for the design rationale.

    Inputs:
      ctx_emb         (B, T_ctx, D)  — encoder output, used as predictor input
      ctx_action_raw  (B, T_ctx, action_dim) — raw actions (post nan_to_num)
      pred_emb_origin (B, T_ctx, D)  — predictor output on original actions
      s_t             (B, T_ctx) or None — clamped sigma from hetero probe
      mode            "full" (default, gA + gS) | "sigma_only" (gS-only ablation;
                      skips K perturbation forwards, sets gA≡0.5, critical = gS*0.5).
                      A_t-only is achieved implicitly by mode=full + s_t=None.
      intervention    Causal-necessity controls (plan_adaptive_resolution.md §6 P0-2):
                        "none"           — real σ+A_t (default).
                        "shuffle_sigma"  — permute s_t over the flattened (B,T) dim
                                           before z-scoring, breaking σ↔state mapping
                                           while preserving the σ marginal.
                        "shuffle_action" — permute log_A over (B,T) before z-scoring.
                        "random_gate"    — replace `critical` with Uniform[0,1] of
                                           the same shape (kills σ and A signals).
                        "constant_w"     — replace per-token w_t with a scalar equal
                                           to its current batch mean (preserves overall
                                           consistency pressure, kills per-token spread).
      fragile_t       Required when mode="dgc". (B, T_ctx) tensor of detached
                      per-token predictor fragility = ||predict(noisy_z, a) -
                      predict(origin_z, a)|| / nn_dist (single-sample). Replaces
                      the K-perturb action sensitivity A_t. See
                      plan_adaptive_resolution.md §3.8.2.
    """
    if mode not in {"full", "sigma_only", "dgc"}:
        raise ValueError(f"Unsupported action_gate.mode: {mode}")
    if intervention not in {"none", "shuffle_sigma", "shuffle_action", "random_gate", "constant_w"}:
        raise ValueError(f"Unsupported action_gate.intervention: {intervention}")
    sigma_only = mode == "sigma_only"
    dgc_mode = mode == "dgc"
    if sigma_only and s_t is None:
        raise RuntimeError(
            "action_gate.mode=sigma_only requires loss.hetero.enabled=true with mode=probe"
        )
    if dgc_mode and fragile_t is None:
        raise RuntimeError(
            "action_gate.mode=dgc requires precomputed fragile_t (see lejepa_forward)"
        )
    metrics = {}
    with torch.no_grad():
        ctx_emb_d = ctx_emb.detach()
        pred_origin_d = pred_emb_origin.detach()
        B, T_ctx = ctx_emb_d.shape[:2]

        if sigma_only:
            # σ-only ablation: skip K perturbation forwards entirely. A-channel
            # metrics are emitted as zeros so logging schema stays stable.
            zero_bt = ctx_emb_d.new_zeros(B, T_ctx)
            A_mean = zero_bt
            A_cv = zero_bt
            log_A = zero_bt
        elif dgc_mode:
            # DGC mode: replace K-perturb action sensitivity with single-shot
            # fragility = ||predict(noisy_z, a) - predict(origin_z, a)|| / nn_dist
            # precomputed in lejepa_forward. CV is undefined for K=1, set to 0.
            frag = fragile_t.detach().to(ctx_emb_d.dtype)
            A_mean = frag
            A_cv = torch.zeros_like(frag)
            log_A = torch.log(frag.clamp(min=log_a_floor))
        else:
            # Per-action-dim std over (B, T_ctx); guards against degenerate dims.
            action_std = ctx_action_raw.float().std(dim=(0, 1), unbiased=False).clamp(min=1e-6)
            # Freeze BN stats during the K perturbation forwards. Otherwise the
            # OOD-ish perturbed activations update BatchNorm running mean/var on
            # every train step, drifting them away from the original-data distribution.
            # See plan_adaptive_resolution.md §8.3.6.4.
            bn_states = []
            for m in model.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.training:
                    bn_states.append(m)
                    m.eval()
            try:
                A_samples = []
                for _ in range(K):
                    delta = torch.randn_like(ctx_action_raw) * (delta_scale * action_std)
                    act_pert = ctx_action_raw + delta
                    act_emb_pert = model.action_encoder(act_pert)
                    pred_pert = model.predict(ctx_emb_d, act_emb_pert)
                    diff = (pred_pert - pred_origin_d).pow(2).sum(dim=-1).clamp(min=0).sqrt()
                    delta_norm = delta.pow(2).sum(dim=-1).clamp(min=0).sqrt().clamp(min=delta_norm_floor)
                    A_samples.append(diff / delta_norm)
            finally:
                for m in bn_states:
                    m.train()

            A_stack = torch.stack(A_samples, dim=0)            # (K, B, T_ctx)
            A_mean = A_stack.mean(dim=0)                       # (B, T_ctx)
            A_cv = A_stack.std(dim=0, unbiased=False) / A_mean.clamp(min=log_a_floor)
            log_A = torch.log(A_mean.clamp(min=log_a_floor))

        # Intervention: shuffle σ↔state or A_t↔state correspondence at the
        # input of the gate. EMA and zscore stats then track the shuffled
        # marginal; corr_sigma_action diagnostic should drop near zero.
        if intervention == "shuffle_sigma" and s_t is not None:
            s_flat_shuf = s_t.detach().reshape(-1)
            perm = torch.randperm(s_flat_shuf.numel(), device=s_flat_shuf.device)
            s_t = s_flat_shuf[perm].reshape(s_t.shape)
        elif intervention == "shuffle_action" and not sigma_only:
            la_flat = log_A.reshape(-1)
            perm = torch.randperm(la_flat.numel(), device=la_flat.device)
            log_A = la_flat[perm].reshape(log_A.shape)

        # EMA update (outside warmup only).
        def _ema_update(name: str, x: torch.Tensor):
            mean_buf = getattr(model, f"gate_{name}_mean")
            var_buf = getattr(model, f"gate_{name}_var")
            inited = getattr(model, f"gate_{name}_inited")
            m_new = x.mean()
            v_new = x.var(unbiased=False)
            if inited.item() < 0.5:
                mean_buf.copy_(m_new)
                var_buf.copy_(v_new)
                inited.fill_(1.0)
            else:
                mu = ema_momentum
                mean_buf.mul_(mu).add_(m_new, alpha=1.0 - mu)
                var_buf.mul_(mu).add_(v_new, alpha=1.0 - mu)

        if not in_warmup:
            if not sigma_only:
                _ema_update("log_A", log_A)
            if s_t is not None:
                _ema_update("s", s_t)

        def _zscore(x: torch.Tensor, name: str) -> torch.Tensor:
            inited = getattr(model, f"gate_{name}_inited").item() > 0.5
            if inited:
                m = getattr(model, f"gate_{name}_mean")
                v = getattr(model, f"gate_{name}_var")
            else:
                m = x.mean()
                v = x.var(unbiased=False)
            return (x - m) / v.clamp(min=1e-6).sqrt()

        if sigma_only:
            gA = ctx_emb_d.new_full((B, T_ctx), 0.5)
        else:
            gA = torch.sigmoid(_zscore(log_A, "log_A"))
        if s_t is not None:
            gS = torch.sigmoid(_zscore(s_t.detach(), "s"))
            if sigma_only:
                critical = gS * 0.5
            else:
                critical = gA * (0.5 + 0.5 * gS)
        else:
            gS = None
            critical = gA * 0.5
        w_t = w_max - (w_max - w_min) * critical

        # Intervention: replace per-token w_t to test controller-side necessity.
        # random_gate: critical ~ U(0,1) (kills σ/A signal entirely).
        # constant_w: scalar batch-mean w_t (kills per-token spread, preserves mean pressure).
        if intervention == "random_gate":
            critical = torch.rand_like(critical)
            w_t = w_max - (w_max - w_min) * critical
        elif intervention == "constant_w":
            w_t = w_t.mean().expand_as(w_t).contiguous()

        # Cast to fp32 for torch.quantile (which rejects half/bf16) and to keep
        # all downstream metric reductions dtype-uniform under AMP.
        cv_flat = A_cv.reshape(-1).float()
        A_mean_flat = A_mean.reshape(-1).float()
        thresh = torch.quantile(A_mean_flat, 0.75)
        high_A_mask = A_mean_flat >= thresh
        high_cv = cv_flat[high_A_mask].mean() if high_A_mask.any() else cv_flat.mean()

        if s_t is not None:
            s_flat = s_t.detach().reshape(-1)
            la_flat = log_A.reshape(-1)
            sc = s_flat - s_flat.mean()
            lac = la_flat - la_flat.mean()
            denom = sc.norm() * lac.norm()
            corr_sigma_action = (sc * lac).sum() / denom.clamp(min=1e-8)
        else:
            corr_sigma_action = log_A.new_tensor(0.0)

        w_flat = w_t.reshape(-1).float()
        crit_flat = critical.reshape(-1).float()
        metrics = {
            "adaptive_action_sensitivity_mean": A_mean.mean(),
            "adaptive_action_sensitivity_std": A_mean.std(unbiased=False),
            "adaptive_action_sensitivity_log_mean": log_A.mean(),
            "adaptive_action_sensitivity_cv_mean": cv_flat.mean(),
            "adaptive_action_sensitivity_cv_high_A": high_cv,
            "adaptive_gA_mean": gA.mean(),
            "adaptive_critical_mean": crit_flat.mean(),
            "adaptive_critical_std": crit_flat.std(unbiased=False),
            "adaptive_weight_mean": w_flat.mean(),
            "adaptive_weight_q10": torch.quantile(w_flat, 0.1),
            "adaptive_weight_q90": torch.quantile(w_flat, 0.9),
            "adaptive_corr_sigma_action": corr_sigma_action,
            "adaptive_in_warmup": log_A.new_tensor(1.0 if in_warmup else 0.0),
        }
        if gS is not None:
            metrics["adaptive_gS_mean"] = gS.mean()
        metrics["_adaptive_weight_tokens"] = w_t.detach()
        metrics["_adaptive_critical_tokens"] = critical.detach()
    return metrics


def apply_pixel_gaussian_noise(x, *, std_min: float, std_max: float, noise_prob: float):
    """Apply per-frame Gaussian pixel noise to a normalized image tensor."""
    if std_max <= 0.0 or noise_prob <= 0.0:
        return x
    std_min = min(std_min, std_max)
    noise = torch.randn_like(x)
    if std_min == std_max:
        std = x.new_full(x.shape[:2] + (1, 1, 1), float(std_max))
    else:
        std = torch.empty(x.shape[:2] + (1, 1, 1), device=x.device, dtype=x.dtype)
        std.uniform_(float(std_min), float(std_max))
    if noise_prob < 1.0:
        keep = torch.rand(x.shape[:2] + (1, 1, 1), device=x.device) < float(noise_prob)
        std = std * keep.to(dtype=x.dtype)
    return x + noise * std


def resolve_pred_target_view(cfg) -> str:
    """Resolve the prediction target view used by the LeWM loss."""
    pred_cfg = cfg.loss.get("pred", {})
    target_view = pred_cfg.get("target_view", "perturbed")
    target_view = str(target_view).lower()
    aliases = {
        "perturb": "perturbed",
        "perturbed": "perturbed",
        "corrupt": "perturbed",
        "corrupted": "perturbed",
        "aug": "perturbed",
        "augmented": "perturbed",
        "full": "perturbed",
        "full_sequence": "perturbed",
        "origin": "origin",
        "orig": "origin",
        "original": "origin",
        "unperturbed": "origin",
        "origin_target": "origin",
        "origin_future": "origin",
    }
    if target_view not in aliases:
        valid = ", ".join(sorted(aliases))
        raise ValueError(
            f"Unsupported loss.pred.target_view={target_view!r}; valid: {valid}"
        )
    return aliases[target_view]


def generic_latent_consistency_enabled(cfg) -> bool:
    return bool(
        cfg.loss.get("generic_latent_consistency", {}).get("enabled", False)
    )


def snap_acpc_enabled(cfg) -> bool:
    return bool(cfg.loss.get("snap_acpc", {}).get("enabled", False))


def paired_view_control_enabled(cfg) -> bool:
    return bool(cfg.loss.get("paired_view_control", {}).get("enabled", False))


def in_forward_noise_control_enabled(cfg) -> bool:
    return bool(cfg.loss.get("in_forward_noise_control", {}).get("enabled", False))


def paired_view_method_enabled(cfg) -> bool:
    enabled_methods = [
        generic_latent_consistency_enabled(cfg),
        snap_acpc_enabled(cfg),
        paired_view_control_enabled(cfg),
    ]
    if sum(enabled_methods) > 1:
        raise ValueError(
            "Enable at most one paired-view method: "
            "loss.generic_latent_consistency, loss.snap_acpc, "
            "or loss.paired_view_control."
        )
    return any(enabled_methods)


@contextmanager
def preserve_batchnorm_eval(module: nn.Module):
    """Temporarily freeze BatchNorm stats without changing other module modes."""
    bn_modules = [
        m
        for m in module.modules()
        if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.training
    ]
    try:
        for m in bn_modules:
            m.eval()
        yield
    finally:
        for m in bn_modules:
            m.train()


def image_perturbation_enabled_for_stage(cfg, stage: str) -> bool:
    """Match train-set perturbation semantics for in-forward origin-target ablations."""
    image_cfg = cfg.get("image_noise", {})
    if image_cfg is None:
        return False
    if float(image_cfg.get("std_max", 0.0)) <= 0.0:
        return False
    if float(image_cfg.get("noise_prob", 1.0)) <= 0.0:
        return False
    if stage in {"train", "training"}:
        return True
    return bool(image_cfg.get("apply_to_val", False))


def apply_configured_pixel_perturbation(batch, cfg, stage: str):
    """Apply the configured image perturbation to batch pixels for this stage."""
    if not image_perturbation_enabled_for_stage(cfg, stage):
        return batch["pixels"]
    image_cfg = cfg.get("image_noise", {})
    noise = AddNormalizedGaussianNoise(
        image_cfg.get("std_min", 0.0),
        image_cfg.get("std_max", 0.0),
        noise_prob=image_cfg.get("noise_prob", 1.0),
    )
    return noise(batch["pixels"])


def adaptive_consistency_loss(origin_emb, noisy_emb, weights, *, distance: str, detach_origin: bool):
    """Weighted original/noisy encoder consistency for Stage C."""
    if detach_origin:
        origin_emb = origin_emb.detach()
    distance = distance.lower()
    if distance == "l2":
        dist = torch.linalg.vector_norm(noisy_emb - origin_emb, dim=-1)
    elif distance == "cosine":
        origin_n = torch.nn.functional.normalize(origin_emb, dim=-1)
        noisy_n = torch.nn.functional.normalize(noisy_emb, dim=-1)
        dist = (1.0 - (origin_n * noisy_n).sum(dim=-1)).clamp_min(0.0)
    else:
        raise ValueError(f"Unsupported adaptive consistency distance: {distance}")
    return (weights.detach() * dist).mean(), dist.detach()


def resolve_adaptive_detach_origin(cons_cfg) -> bool:
    """Prefer detach_origin; accept detach_clean only as a legacy override."""
    detach_origin = cons_cfg.get("detach_origin", None)
    if detach_origin is None:
        detach_origin = cons_cfg.get("detach_clean", True)
    if isinstance(detach_origin, str):
        return detach_origin.strip().lower() not in {"0", "false", "no", "off"}
    return bool(detach_origin)


def compute_temporal_hinge(output, *, model, cfg):
    """Upper hinge loss on consecutive latent pairs (LeWM variant).

    Mirrors the SWM compute_temporal_hinge exactly, except distance is
    computed with L2 (Euclidean) instead of cosine because LeWM does not
    L2-normalise its embeddings.
    """
    emb = output["emb"]
    if emb.size(1) <= 1:
        output["temporal_hinge_active_ratio"] = emb.new_tensor(0.0)
        return emb.new_tensor(0.0)

    hinge_cfg = cfg.loss.temporal_hinge
    dynamic_cfg = hinge_cfg.get("dynamic", {})
    z_t = emb[:, :-1]
    z_tp1 = emb[:, 1:]

    if not dynamic_cfg.get("enabled", False):
        dist = torch.linalg.vector_norm(z_tp1 - z_t, dim=-1)
        margin = hinge_cfg.margin
        hinge = torch.clamp_min(dist - margin, 0.0)
        output["temporal_hinge_active_ratio"] = (hinge > 0).float().mean()
        if hinge_cfg.squared:
            hinge = hinge.square()
        return hinge.mean()

    if not hasattr(model, "dynamic_margin_head"):
        raise AttributeError(
            "loss.temporal_hinge.dynamic.enabled=True requires "
            "model.dynamic_margin_head to be initialized"
        )

    act_emb = output["act_emb"][:, :-1]
    margin_input = torch.cat([z_t.detach(), act_emb.detach()], dim=-1)
    raw_score = model.dynamic_margin_head(margin_input).squeeze(-1)
    score = torch.sigmoid(raw_score)
    score = score / score.detach().mean().clamp_min(1e-6)

    margin = dynamic_cfg.get("base_margin", hinge_cfg.margin) * score
    margin = margin.clamp(
        min=dynamic_cfg.get("min_margin", 0.05),
        max=dynamic_cfg.get("max_margin", 1.0),
    )

    dist = torch.linalg.vector_norm(z_tp1 - z_t, dim=-1)
    hinge = torch.clamp_min(dist - margin, 0.0)
    output["temporal_hinge_active_ratio"] = (hinge > 0).float().mean()
    output["temporal_margin_mean"] = margin.mean()
    output["temporal_margin_std"] = margin.std(unbiased=False)
    margin_flat = margin.detach().float().flatten()
    output["temporal_margin_p10"] = torch.quantile(margin_flat, 0.10)
    output["temporal_margin_p50"] = torch.quantile(margin_flat, 0.50)
    output["temporal_margin_p90"] = torch.quantile(margin_flat, 0.90)

    if hinge_cfg.squared:
        hinge = hinge.square()
    return hinge.mean()


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    sigreg_lambd = cfg.loss.sigreg.weight
    hinge_cfg = cfg.loss.temporal_hinge

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    pred_cfg = cfg.loss.get("pred", {})
    pred_space = pred_cfg.get("space", "raw")
    target_view = resolve_pred_target_view(cfg)
    glc_enabled = generic_latent_consistency_enabled(cfg)
    snap_enabled = snap_acpc_enabled(cfg)
    paired_control_enabled = paired_view_control_enabled(cfg)
    paired_view = paired_view_method_enabled(cfg)
    in_forward_noise_enabled = in_forward_noise_control_enabled(cfg)
    if paired_view and in_forward_noise_enabled:
        raise ValueError(
            "loss.in_forward_noise_control is mutually exclusive with paired-view "
            "training methods."
        )
    if paired_view and target_view != "perturbed":
        raise ValueError(
            "paired-view training methods require loss.pred.target_view=perturbed"
        )
    if in_forward_noise_enabled and target_view != "perturbed":
        raise ValueError(
            "loss.in_forward_noise_control requires loss.pred.target_view=perturbed"
        )

    origin_output = None
    origin_emb = None
    if paired_view:
        with torch.no_grad(), preserve_batchnorm_eval(self.model):
            origin_output = self.model.encode(dict(batch))
        perturbed_pixels = apply_configured_pixel_perturbation(batch, cfg, stage)
        if perturbed_pixels is batch["pixels"]:
            output = self.model.encode(batch)
        else:
            perturbed_batch = dict(batch)
            perturbed_batch["pixels"] = perturbed_pixels
            output = self.model.encode(perturbed_batch)
        origin_emb = origin_output["emb"]
    elif in_forward_noise_enabled:
        perturbed_pixels = apply_configured_pixel_perturbation(batch, cfg, stage)
        if perturbed_pixels is batch["pixels"]:
            output = self.model.encode(batch)
        else:
            perturbed_batch = dict(batch)
            perturbed_batch["pixels"] = perturbed_pixels
            output = self.model.encode(perturbed_batch)
    else:
        output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]
    sigreg_emb = emb
    if in_forward_noise_enabled:
        output["in_forward_noise_active"] = emb.new_tensor(1.0)

    if paired_view:
        ctx_emb = emb[:, :ctx_len]
        with torch.no_grad():
            output["target_view_paired_context_noise_l2"] = torch.linalg.vector_norm(
                ctx_emb.detach() - origin_emb[:, :ctx_len].detach(), dim=-1
            ).mean()
            if paired_control_enabled:
                output["paired_noaux_active"] = emb.new_tensor(1.0)
                output["paired_noaux_context_noise_l2"] = output[
                    "target_view_paired_context_noise_l2"
                ]
    elif target_view == "perturbed":
        ctx_emb = emb[:, :ctx_len]
    elif target_view == "origin":
        perturbed_pixels = apply_configured_pixel_perturbation(batch, cfg, stage)
        if perturbed_pixels is batch["pixels"]:
            perturbed_output = output
        else:
            perturbed_batch = dict(batch)
            perturbed_batch["pixels"] = perturbed_pixels
            perturbed_output = self.model.encode(perturbed_batch)
        perturbed_emb = perturbed_output["emb"]
        ctx_emb = perturbed_emb[:, :ctx_len]
        sigreg_emb = perturbed_emb
        with torch.no_grad():
            origin_ctx = emb[:, :ctx_len]
            output["target_view_origin_future"] = emb.new_tensor(1.0)
            output["target_view_context_noise_l2"] = torch.linalg.vector_norm(
                ctx_emb.detach() - origin_ctx.detach(), dim=-1
            ).mean()
    else:
        raise RuntimeError(f"Unhandled target_view={target_view}")

    ctx_act = act_emb[:, :ctx_len]

    tgt_emb = emb[:, n_preds:]  # label
    # Optional SimSiam-style stop-grad on target. Together with the existing
    # predictor head this turns the JEPA into an asymmetric architecture and
    # is the standard recipe for BN-free non-contrastive training (see Chen
    # & He, "Exploring Simple Siamese Representation Learning", CVPR 2021).
    # Without this asymmetry, dropping BN tends to collapse because pred and
    # target share gradient through the same encoder.
    if cfg.loss.get("target_stop_grad", False):
        tgt_emb = tgt_emb.detach()
    hetero_cfg = cfg.loss.get("hetero", {})
    hetero_enabled = bool(hetero_cfg.get("enabled", False))
    hetero_mode = hetero_cfg.get("mode", "loss").lower()
    if hetero_enabled and hetero_mode not in {"loss", "probe"}:
        raise ValueError(f"Unsupported loss.hetero.mode: {hetero_mode}")
    if (glc_enabled or snap_enabled) and hetero_enabled and hetero_mode == "loss":
        raise ValueError(
            "paired-view auxiliary losses are defined for the MSE pred_loss "
            "baseline; disable loss.hetero or use loss.hetero.mode=probe."
        )
    if hetero_enabled:
        pred_emb, logvar_hat = self.model.predict_with_logvar(
            ctx_emb,
            ctx_act,
            detach_logvar_input=(hetero_mode == "probe"),
        )
        if logvar_hat is None:
            raise RuntimeError(
                "loss.hetero.enabled=True requires model.pred_logvar_proj to be built"
            )
    else:
        pred_emb = self.model.predict(ctx_emb, ctx_act)  # pred
        logvar_hat = None
    pred_loss_emb = get_pred_loss_tensor(pred_emb, space=pred_space)
    tgt_loss_emb = get_pred_loss_tensor(tgt_emb, space=pred_space)
    pred_mse_tokens = mse_token(pred_loss_emb, tgt_loss_emb)
    pred_mse_loss = pred_mse_tokens.mean()

    # LeWM loss, optional hetero replacement, or detached sigma probe.
    if hetero_enabled and hetero_mode == "loss":
        hetero_loss, hetero_monitors = compute_hetero_pred_loss(
            pred_loss_emb,
            tgt_loss_emb,
            logvar_hat,
            s_min=hetero_cfg.get("s_min", -4.0),
            s_max=hetero_cfg.get("s_max", 4.0),
            tau_floor=hetero_cfg.get("tau_floor", 1e-6),
        )
        output["pred_loss"] = hetero_loss
        # Also report the underlying MSE for direct comparability with the
        # LeWM baseline (loss curves stay readable when toggling hetero).
        output["pred_loss_mse_equiv"] = pred_mse_loss.detach()
        for k, v in hetero_monitors.items():
            output[k] = v
    else:
        output["pred_loss"] = pred_mse_loss
        if hetero_enabled and hetero_mode == "probe":
            sigma_probe_loss, probe_monitors = compute_sigma_probe_loss(
                pred_loss_emb,
                tgt_loss_emb,
                logvar_hat,
                s_min=hetero_cfg.get("s_min", -4.0),
                s_max=hetero_cfg.get("s_max", 4.0),
                tau_floor=hetero_cfg.get("tau_floor", 1e-6),
            )
            output["sigma_probe_loss"] = sigma_probe_loss
            output["pred_loss_mse_equiv"] = output["pred_loss"].detach()
            for k, v in probe_monitors.items():
                output[k] = v
    if glc_enabled:
        latent_raw = mse_token(
            emb[:, :ctx_len],
            origin_emb[:, :ctx_len].detach(),
        ).mean()
        latent_loss, latent_scale = self_bounded_aux_loss(
            pred_mse_loss,
            latent_raw,
        )
        output["glc_latent_raw"] = latent_raw
        output["glc_latent_scale"] = latent_scale
        output["glc_latent_loss"] = latent_loss
        output["glc_pair_to_base"] = (
            latent_raw.detach() / pred_mse_loss.detach().clamp_min(1e-8)
        )
        output["pred_loss"] = output["pred_loss"] + latent_loss
    if snap_enabled:
        with torch.no_grad(), preserve_batchnorm_eval(self.model):
            clean_pred_emb = self.model.predict(
                origin_emb[:, :ctx_len],
                origin_output["act_emb"][:, :ctx_len],
            )
        clean_pred_loss_emb = get_pred_loss_tensor(clean_pred_emb, space=pred_space)
        acpc_raw = mse_token(
            pred_loss_emb,
            clean_pred_loss_emb.detach(),
        ).mean()
        acpc_loss, acpc_scale = self_bounded_aux_loss(
            pred_mse_loss,
            acpc_raw,
        )
        output["snap_acpc_pred_raw"] = acpc_raw
        output["snap_acpc_pred_scale"] = acpc_scale
        output["snap_acpc_pred_loss"] = acpc_loss
        output["snap_acpc_pair_to_base"] = (
            acpc_raw.detach() / pred_mse_loss.detach().clamp_min(1e-8)
        )
        output["pred_loss"] = output["pred_loss"] + acpc_loss
    if paired_control_enabled:
        with torch.no_grad(), preserve_batchnorm_eval(self.model):
            clean_pred_emb = self.model.predict(
                origin_emb[:, :ctx_len],
                origin_output["act_emb"][:, :ctx_len],
            )
            clean_pred_loss_emb = get_pred_loss_tensor(clean_pred_emb, space=pred_space)
            noaux_pred_raw = mse_token(
                pred_loss_emb.detach(),
                clean_pred_loss_emb.detach(),
            ).mean()
        output["paired_noaux_pred_raw"] = noaux_pred_raw
        output["paired_noaux_pair_to_base"] = (
            noaux_pred_raw.detach() / pred_mse_loss.detach().clamp_min(1e-8)
        )
    gate_cfg = cfg.loss.get("action_gate", {})
    if gate_cfg.get("enabled", False):
        warmup_epochs = int(gate_cfg.get("warmup_epochs", 3))
        current_epoch = int(getattr(self, "current_epoch", 0))
        in_warmup = current_epoch < warmup_epochs
        if hetero_enabled and logvar_hat is not None:
            s_t = logvar_hat.squeeze(-1).clamp(
                min=hetero_cfg.get("s_min", -4.0),
                max=hetero_cfg.get("s_max", 4.0),
            )
        else:
            s_t = None
        gate_mode_str = str(gate_cfg.get("mode", "full"))
        fragile_t_arg = None
        if gate_mode_str == "dgc":
            cons_cfg_for_dgc = cfg.loss.get("adaptive_consistency", {})
            dgc_std_raw = gate_cfg.get("dgc_noise_std_max", None)
            if dgc_std_raw is None:
                dgc_std_raw = cons_cfg_for_dgc.get("noise_std_max", 0.04)
            dgc_std_max = float(dgc_std_raw)
            dgc_knn_k = int(gate_cfg.get("dgc_knn_k", 5))
            with torch.no_grad():
                noisy_batch_dgc = dict(batch)
                noisy_batch_dgc["pixels"] = apply_pixel_gaussian_noise(
                    batch["pixels"],
                    std_min=0.0,
                    std_max=dgc_std_max,
                    noise_prob=1.0,
                )
                noisy_emb_full = self.model.encode(noisy_batch_dgc)["emb"]
                noisy_ctx = noisy_emb_full[:, :ctx_len]
                pred_emb_noisy = self.model.predict(noisy_ctx, ctx_act)
                target_shift = (
                    (pred_emb_noisy - pred_emb).pow(2).sum(-1).clamp(min=0).sqrt()
                )
                nn_dist_t = batch_knn_distance(ctx_emb.detach(), k=dgc_knn_k)
                fragile_t_arg = target_shift / nn_dist_t.clamp(min=1e-6)
            # Cache for adaptive_consistency reuse (avoid double noisy forward).
            output["_dgc_noisy_emb"] = noisy_emb_full.detach()
            output["_dgc_noise_std_max"] = dgc_std_max
        gate_metrics = compute_action_gate_metrics(
            self.model,
            ctx_emb=ctx_emb,
            ctx_action_raw=output["action"][:, :ctx_len],
            pred_emb_origin=pred_emb,
            s_t=s_t,
            K=int(gate_cfg.get("num_delta_samples", 4)),
            delta_scale=float(gate_cfg.get("delta_scale", 0.25)),
            delta_norm_floor=float(gate_cfg.get("delta_norm_floor", 1e-6)),
            log_a_floor=float(gate_cfg.get("log_a_floor", 1e-8)),
            in_warmup=in_warmup,
            ema_momentum=float(gate_cfg.get("ema_momentum", 0.99)),
            w_min=float(gate_cfg.get("w_min", 0.2)),
            w_max=float(gate_cfg.get("w_max", 1.0)),
            mode=gate_mode_str,
            intervention=str(gate_cfg.get("intervention", "none")),
            fragile_t=fragile_t_arg,
        )
        for k, v in gate_metrics.items():
            output[k] = v

    cons_cfg = cfg.loss.get("adaptive_consistency", {})
    cons_weight = float(cons_cfg.get("weight", 0.0))
    if cons_cfg.get("enabled", False) and cons_weight > 0.0:
        if cons_cfg.get("require_action_gate", True) and "_adaptive_weight_tokens" not in output:
            raise RuntimeError(
                "loss.adaptive_consistency requires loss.action_gate.enabled=true "
                "unless require_action_gate=false"
            )
        cons_std_max = float(cons_cfg.get("noise_std_max", 0.04))
        cons_std_min = float(cons_cfg.get("noise_std_min", 0.0))
        cons_noise_prob = float(cons_cfg.get("noise_prob", 1.0))
        # Reuse DGC-computed noisy embedding when configs match (no extra forward).
        # The match is exact only when cons noise is the deterministic uniform
        # [0, std_max] with prob=1, matching the DGC sampler in the gate block.
        reuse_dgc_noisy = (
            "_dgc_noisy_emb" in output
            and cons_std_min == 0.0
            and cons_noise_prob == 1.0
            and float(output.get("_dgc_noise_std_max", -1.0)) == cons_std_max
        )
        if reuse_dgc_noisy:
            noisy_emb = output["_dgc_noisy_emb"][:, :ctx_len]
        else:
            origin_pixels = batch["pixels"]
            noisy_batch = dict(batch)
            noisy_batch["pixels"] = apply_pixel_gaussian_noise(
                origin_pixels,
                std_min=cons_std_min,
                std_max=cons_std_max,
                noise_prob=cons_noise_prob,
            )
            noisy_emb = self.model.encode(noisy_batch)["emb"][:, :ctx_len]
        if "_adaptive_weight_tokens" in output:
            cons_weights = output["_adaptive_weight_tokens"]
        else:
            cons_weights = emb.new_ones(emb.shape[:2])[:, :ctx_len]
        (
            output["adaptive_consistency_loss"],
            adaptive_consistency_dist,
        ) = adaptive_consistency_loss(
            emb[:, :ctx_len],
            noisy_emb,
            cons_weights,
            distance=cons_cfg.get("distance", "l2"),
            detach_origin=resolve_adaptive_detach_origin(cons_cfg),
        )
        output["adaptive_consistency_dist_mean"] = adaptive_consistency_dist.mean()
        output["adaptive_consistency_weight_mean"] = cons_weights.detach().mean()

    # Anti-collapse regularizer: SIGReg (Epps-Pulley) and optional Wasserstein
    # companion. Two schedule modes (warmup.mode):
    #   - replace (legacy): during the warmup window Wass replaces SIGReg; after,
    #     only SIGReg. Phase switch can perturb pred_loss via the shared encoder.
    #   - add_decay: SIGReg is always on; Wass runs at full weight for
    #     `warmup.epochs` epochs, then linearly decays to 0 over `decay_epochs`,
    #     producing a continuous regularizer landscape with no abrupt switch.
    # `sigreg_loss` always carries the Epps-Pulley value; `wass_loss` is the
    # Wasserstein value (when the module is built). Their loss contributions
    # are scaled by `sigreg_scale` / `wass_scale` (also logged for diagnostics).
    warmup_cfg = cfg.loss.sigreg.get("warmup", {})
    warmup_mode = str(warmup_cfg.get("mode", "replace")).lower()
    if warmup_mode not in {"replace", "add_decay"}:
        raise ValueError(f"Unsupported loss.sigreg.warmup.mode: {warmup_mode}")

    warmup_epochs = int(warmup_cfg.get("epochs", 0))
    decay_epochs = int(warmup_cfg.get("decay_epochs", 0))
    current_epoch_int = int(getattr(self, "current_epoch", 0))
    has_warmup = getattr(self, "sigreg_warmup", None) is not None
    in_warmup = current_epoch_int < warmup_epochs

    if warmup_mode == "replace":
        wass_scale = 1.0 if (has_warmup and in_warmup) else 0.0
        sigreg_scale = 0.0 if (has_warmup and in_warmup) else 1.0
    else:  # add_decay
        sigreg_scale = 1.0
        if not has_warmup:
            wass_scale = 0.0
        elif in_warmup:
            wass_scale = 1.0
        elif decay_epochs > 0 and (current_epoch_int - warmup_epochs) < decay_epochs:
            wass_scale = 1.0 - float(current_epoch_int - warmup_epochs) / float(decay_epochs)
        else:
            wass_scale = 0.0

    # Shape anti-collapse on the predictor input view. For origin-target
    # training this is the perturbed/noisy input branch, while the target stays
    # the original future embedding for pred_loss.
    emb_tbd = sigreg_emb.transpose(0, 1)

    # SIGReg: detached forward when not active so the curve stays visible.
    if sigreg_scale > 0:
        output["sigreg_loss"] = self.sigreg(emb_tbd)
    else:
        with torch.no_grad():
            output["sigreg_loss"] = self.sigreg(emb_tbd)

    # Wasserstein: only computed when the module is built. Detached forward
    # post-decay so the curve continues to track drift on the trained encoder.
    if has_warmup:
        if wass_scale > 0:
            output["wass_loss"] = self.sigreg_warmup(emb_tbd)
        else:
            with torch.no_grad():
                output["wass_loss"] = self.sigreg_warmup(emb_tbd)

    warmup_weight_cfg = warmup_cfg.get("weight", None)
    wass_lambd = sigreg_lambd if warmup_weight_cfg is None else float(warmup_weight_cfg)

    output["sigreg_scale"] = emb.new_tensor(float(sigreg_scale))
    output["wass_scale"] = emb.new_tensor(float(wass_scale))
    output["sigreg_warmup_active"] = emb.new_tensor(1.0 if (has_warmup and in_warmup) else 0.0)
    output["temporal_hinge_loss"] = compute_temporal_hinge(
        output, model=self.model, cfg=cfg
    )
    inverse_cfg = cfg.loss.get("inverse_dynamics", {})
    inverse_weight = inverse_cfg.get("weight", 0.0)
    if inverse_weight > 0.0:
        if not hasattr(self.model, "inverse_dynamics_head"):
            raise AttributeError(
                "loss.inverse_dynamics requires model.inverse_dynamics_head"
            )
        output["inverse_dynamics_loss"] = inverse_dynamics_loss(
            emb[:, :-1],
            emb[:, 1:],
            output["action"][:, :-1],
            self.model.inverse_dynamics_head,
            detach_input=inverse_cfg.get("detach_input", False),
        )

    dist_cfg = cfg.loss.get("transition_distance", {})
    dist_weight = dist_cfg.get("weight", 0.0)
    if dist_weight > 0.0:
        if not hasattr(self.model, "transition_distance_head"):
            raise AttributeError(
                "loss.transition_distance requires model.transition_distance_head"
            )
        (
            output["transition_distance_loss"],
            pred_dist,
            target_dist,
        ) = transition_distance_prediction_loss(
            emb[:, :-1],
            emb[:, 1:],
            self.model.transition_distance_head,
            metric=dist_cfg.get("metric", "l2"),
            detach_input=dist_cfg.get("detach_input", True),
        )
        output["transition_distance_pred_mean"] = pred_dist.mean()
        output["transition_distance_target_mean"] = target_dist.mean()
        output["transition_distance_target_std"] = target_dist.std(unbiased=False)

    output["loss"] = (
        output["pred_loss"]
        + sigreg_lambd * sigreg_scale * output["sigreg_loss"]
        + hinge_cfg.weight * output["temporal_hinge_loss"]
    )
    if "wass_loss" in output:
        output["loss"] = output["loss"] + wass_lambd * wass_scale * output["wass_loss"]
    if "inverse_dynamics_loss" in output:
        output["loss"] = output["loss"] + inverse_weight * output["inverse_dynamics_loss"]
    if "transition_distance_loss" in output:
        output["loss"] = (
            output["loss"]
            + dist_weight * output["transition_distance_loss"]
        )
    if "sigma_probe_loss" in output:
        output["loss"] = (
            output["loss"]
            + hetero_cfg.get("probe_weight", 1.0) * output["sigma_probe_loss"]
        )
    if "adaptive_consistency_loss" in output:
        output["loss"] = (
            output["loss"]
            + cons_weight * output["adaptive_consistency_loss"]
        )
    output["temporal_straightness"] = temporal_straightness(emb)

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    metrics_dict = {
        f"{stage}/{k}": v.detach()
        for k, v in output.items()
        if (
            torch.is_tensor(v)
            and (
                k == "temporal_straightness"
                or k == "temporal_hinge_active_ratio"
                or k.startswith("temporal_margin_")
                or (
                    k.startswith("transition_distance_")
                    and not k.endswith("_loss")
                )
                or k.startswith("hetero_")
                or k.startswith("sigma_probe_")
                or k.startswith("adaptive_")
                or k.startswith("glc_")
                or k.startswith("snap_acpc_")
                or k.startswith("paired_noaux_")
                or k.startswith("in_forward_noise_")
                or k.startswith("target_view_")
                or k == "pred_loss_mse_equiv"
                or k == "sigreg_warmup_active"
                or k == "sigreg_scale"
                or k == "wass_scale"
            )
        )
    }
    self.log_dict({**losses_dict, **metrics_dict}, on_step=True, sync_dist=True)
    return output


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    # Resolve H5 path explicitly so we tolerate both layouts:
    #   0.0.6 wheel:           <STABLEWM_HOME>/<name>.h5   (flat)
    #   post-PR-#221 source:   <STABLEWM_HOME>/datasets/<name>.h5
    # Passing `path=` bypasses the hard-coded sub_folder in source-version
    # HDF5Dataset.__init__. With `path=` set, `name` is ignored, but we keep
    # it in cfg for downstream introspection.
    _data_cfg_for_h5 = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    _h5_name = _data_cfg_for_h5.pop("name")
    _h5_path = resolve_h5_dataset_path(_h5_name)
    dataset = swm.data.HDF5Dataset(path=str(_h5_path), transform=None, **_data_cfg_for_h5)
    transforms = [
        get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)
    ]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )
    target_view = resolve_pred_target_view(cfg)
    paired_view = paired_view_method_enabled(cfg)
    in_forward_noise = in_forward_noise_control_enabled(cfg)
    if paired_view and in_forward_noise:
        raise ValueError(
            "loss.in_forward_noise_control is mutually exclusive with paired-view "
            "training methods."
        )
    if paired_view and target_view != "perturbed":
        raise ValueError(
            "paired-view training methods require loss.pred.target_view=perturbed"
        )
    if in_forward_noise and target_view != "perturbed":
        raise ValueError(
            "loss.in_forward_noise_control requires loss.pred.target_view=perturbed"
        )
    img_noise = get_img_noise_transform(cfg.get("image_noise"))
    if (
        img_noise is not None
        and target_view == "perturbed"
        and not paired_view
        and not in_forward_noise
    ):
        train_set = TransformDataset(train_set, img_noise)
        if cfg.image_noise.get("apply_to_val", False):
            val_set = TransformDataset(val_set, img_noise)
    elif img_noise is not None and paired_view:
        print(
            "[image_noise] paired-view method enabled: "
            "using in-forward clean/noisy branches"
        )
    elif img_noise is not None and in_forward_noise:
        print(
            "[image_noise] in-forward noise control enabled: "
            "using noisy-only in-forward perturbation"
        )
    elif img_noise is not None:
        print(
            f"[image_noise] target_view={target_view}: "
            "using in-forward perturbed history with origin future targets"
        )

    train = torch.utils.data.DataLoader(
        train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen
    )
    val = torch.utils.data.DataLoader(
        val_set, **cfg.loader, shuffle=False, drop_last=False
    )

    ##############################
    ##       model / optim      ##
    ##############################

    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)

    head_cfg = cfg.get("encoder", {}).get("projection_head", {})
    proj_norm_name = head_cfg.get("norm_fn", "batchnorm1d")
    proj_hidden_dim = head_cfg.get("hidden_dim", 2048)
    proj_norm_fn = resolve_norm_fn(proj_norm_name)

    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=proj_hidden_dim,
        norm_fn=proj_norm_fn,
    )

    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=proj_hidden_dim,
        norm_fn=proj_norm_fn,
    )

    hetero_cfg = cfg.loss.get("hetero", {})
    if hetero_cfg.get("enabled", False):
        # scalar log-variance head sharing the predictor backbone hidden state.
        # Adds ~0.5M params for hidden_dim=192 + hidden=2048; negligible.
        # Initialised to output 0 so the loss starts at MSE-equivalent.
        pred_logvar_head = MLP(
            input_dim=hidden_dim,
            output_dim=1,
            hidden_dim=hetero_cfg.get("logvar_hidden_dim", 256),
            norm_fn=proj_norm_fn,
        )
        # Zero the final linear so logvar starts at 0 (i.e. weight = exp(-0) = 1
        # everywhere; loss reduces to plain MSE on the first step).
        with torch.no_grad():
            final = pred_logvar_head.net[-1]
            final.weight.zero_()
            if final.bias is not None:
                final.bias.zero_()
    else:
        pred_logvar_head = None

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
        pred_logvar_proj=pred_logvar_head,
    )
    gate_cfg_init = cfg.loss.get("action_gate", {})
    if gate_cfg_init.get("enabled", False):
        # Scalar EMA buffers for zscore normalisation of log A_t and s_t.
        # Non-persistent: they re-converge quickly during warmup_epochs, and
        # making them persistent breaks resume from checkpoints that predate
        # this feature.
        for name in ("log_A", "s"):
            world_model.register_buffer(
                f"gate_{name}_mean", torch.zeros(()), persistent=False
            )
            world_model.register_buffer(
                f"gate_{name}_var", torch.ones(()), persistent=False
            )
            world_model.register_buffer(
                f"gate_{name}_inited", torch.zeros(()), persistent=False
            )

    if cfg.loss.temporal_hinge.get("dynamic", {}).get("enabled", False):
        world_model.dynamic_margin_head = nn.Linear(2 * embed_dim, 1)
        nn.init.zeros_(world_model.dynamic_margin_head.weight)
        nn.init.zeros_(world_model.dynamic_margin_head.bias)
    inverse_cfg = cfg.loss.get("inverse_dynamics", {})
    if inverse_cfg.get("weight", 0.0) > 0.0:
        world_model.inverse_dynamics_head = MLP(
            input_dim=2 * embed_dim,
            hidden_dim=inverse_cfg.get("hidden_dim", embed_dim),
            output_dim=effective_act_dim,
            norm_fn=None,
        )
    dist_cfg = cfg.loss.get("transition_distance", {})
    if dist_cfg.get("weight", 0.0) > 0.0:
        world_model.transition_distance_head = MLP(
            input_dim=2 * embed_dim,
            hidden_dim=dist_cfg.get("hidden_dim", embed_dim),
            output_dim=1,
            norm_fn=None,
        )

    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    # Optional scale-aware Wasserstein warmup for the anti-collapse regularizer.
    warmup_cfg = cfg.loss.sigreg.get("warmup", {})
    warmup_type = str(warmup_cfg.get("type", "none")).lower()
    if warmup_type not in {"none", "wasserstein"}:
        raise ValueError(f"Unsupported loss.sigreg.warmup.type: {warmup_type}")
    sigreg_warmup = None
    warmup_epochs = int(warmup_cfg.get("epochs", 0))
    warmup_decay_epochs = int(warmup_cfg.get("decay_epochs", 0))
    warmup_mode_cfg = str(warmup_cfg.get("mode", "replace")).lower()
    if warmup_mode_cfg not in {"replace", "add_decay"}:
        raise ValueError(f"Unsupported loss.sigreg.warmup.mode: {warmup_mode_cfg}")
    # Build the Wasserstein module whenever it could see any active step:
    # warmup_epochs > 0 covers the legacy replace mode and the full-weight head
    # of add_decay; decay_epochs > 0 covers a "decay only" add_decay with no
    # full-weight head (rare but valid).
    if warmup_type == "wasserstein" and (warmup_epochs > 0 or warmup_decay_epochs > 0):
        sigreg_warmup = WassersteinSIGReg(num_proj=int(warmup_cfg.get("num_proj", 1024)))
        _warmup_weight = warmup_cfg.get("weight", None)
        _weight_desc = (
            f"(reuse loss.sigreg.weight={cfg.loss.sigreg.weight})"
            if _warmup_weight is None
            else str(_warmup_weight)
        )
        print(
            f"[sigreg] Wasserstein warmup ENABLED: mode={warmup_mode_cfg}, "
            f"epochs={warmup_epochs}, decay_epochs={warmup_decay_epochs}, "
            f"num_proj={int(warmup_cfg.get('num_proj', 1024))}, weight={_weight_desc}"
        )
    else:
        print(
            f"[sigreg] Wasserstein warmup DISABLED "
            f"(type={warmup_type}, mode={warmup_mode_cfg}, "
            f"epochs={warmup_epochs}, decay_epochs={warmup_decay_epochs}); "
            "using Epps-Pulley SIGReg for all epochs"
        )

    data_module = spt.data.DataModule(train=train, val=val)
    module_kwargs = dict(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )
    if sigreg_warmup is not None:
        module_kwargs["sigreg_warmup"] = sigreg_warmup
    world_model = spt.Module(**module_kwargs)

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    logger = None
    backend = cfg.get("logger_backend", "swanlab")
    if backend == "swanlab" and cfg.swanlab.enabled:
        if SwanLabLogger is None:
            raise ImportError("swanlab is not installed. Run: pip install swanlab")
        logger = SwanLabLogger(**cfg.swanlab.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))
    elif backend == "wandb" and cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir,
        filename=cfg.output_model_name,
        epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )

    manager()
    return


if __name__ == "__main__":
    run()
