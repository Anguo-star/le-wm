import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger

try:
    from swanlab.integration.pytorch_lightning import SwanLabLogger
except ImportError:
    SwanLabLogger = None
from omegaconf import OmegaConf, open_dict
from torch import nn

from jepa import SphericalJEPA
from module import (
    ARPredictor,
    Embedder,
    MLP,
    infonce_loss,
    inverse_dynamics_loss,
    spread_loss,
    temporal_straightness,
    transition_distance_prediction_loss,
    uniformity_loss,
)
from utils import (
    get_column_normalizer,
    get_img_noise_transform,
    get_img_preprocessor,
    ModelObjectCallBack,
    resolve_h5_dataset_path,
    TransformDataset,
)


def resolve_norm_fn(norm_name: str):
    norm_name = norm_name.lower()
    if norm_name in {"none", "identity"}:
        return None
    if norm_name in {"ln", "layernorm"}:
        return nn.LayerNorm
    if norm_name in {"bn", "batchnorm", "batchnorm1d"}:
        return nn.BatchNorm1d
    raise ValueError(f"Unsupported projection_head.norm_fn: {norm_name}")


def build_projection_head(input_dim: int, output_dim: int, cfg) -> nn.Module:
    head_type = cfg.type.lower()
    norm_name = cfg.get("norm_fn", "none")

    # Backward-compatible aliases for older configs.
    if head_type == "bn":
        head_type = "linear"
        norm_name = "batchnorm1d"
    elif head_type == "ln":
        head_type = "linear"
        norm_name = "layernorm"

    norm_fn = resolve_norm_fn(norm_name)

    if head_type == "linear":
        layers = [nn.Linear(input_dim, output_dim)]
        if norm_fn is not None:
            layers.append(norm_fn(output_dim))
        return nn.Sequential(*layers) if len(layers) > 1 else layers[0]

    if head_type == "mlp":
        return MLP(
            input_dim=input_dim,
            hidden_dim=cfg.get("hidden_dim", 2048),
            output_dim=output_dim,
            norm_fn=norm_fn,
        )

    raise ValueError(f"Unsupported projection_head.type: {head_type}")


def get_loss_space_tensors(output, *, pred_raw, pred_norm, n_preds: int, space: str):
    pred_len = pred_raw.size(1)
    space = space.lower()
    if space == "raw":
        return pred_raw, output["emb_raw"][:, n_preds : n_preds + pred_len]
    if space in {"normalized", "sphere"}:
        return pred_norm, output["emb"][:, n_preds : n_preds + pred_len]
    raise ValueError(f"Unsupported loss space: {space}")


def get_regularizer_tensor(output, *, space: str):
    space = space.lower()
    if space == "raw":
        return output["emb_raw"]
    if space in {"normalized", "sphere"}:
        return output["emb"]
    raise ValueError(f"Unsupported loss.regularizer.space: {space}")


def get_context_tensor(output, *, space: str):
    space = space.lower()
    if space == "raw":
        return output["emb_raw"]
    if space in {"normalized", "sphere"}:
        return output["emb"]
    raise ValueError(f"Unsupported loss.pred.context_space: {space}")


def get_embedding_tensor(output, *, space: str):
    space = space.lower()
    if space == "raw":
        return output["emb_raw"]
    if space in {"normalized", "sphere"}:
        return output["emb"]
    raise ValueError(f"Unsupported embedding space: {space}")


def compute_embedding_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_type: str,
) -> torch.Tensor:
    loss_type = loss_type.lower()
    if loss_type == "cosine":
        return (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()
    if loss_type == "mse":
        return F.mse_loss(pred, target)
    raise ValueError(f"Unsupported prediction loss type: {loss_type}")


def compute_pred_loss(pred: torch.Tensor, target: torch.Tensor, cfg) -> torch.Tensor:
    pred_type = cfg.loss.pred.get("type", "cosine")
    return compute_embedding_loss(pred, target, loss_type=pred_type)


def compute_multistep_rollout_loss(output, *, model, cfg):
    rollout_cfg = cfg.loss.get("rollout", {})
    rollout_steps = int(rollout_cfg.get("steps", 1))
    if rollout_cfg.get("weight", 0.0) <= 0.0 or rollout_steps <= 1:
        return None

    ctx_len = cfg.wm.history_size
    available_future = output["emb"].size(1) - ctx_len
    rollout_steps = min(rollout_steps, available_future)
    if rollout_steps <= 1:
        return None

    context_space = rollout_cfg.get("context_space", cfg.loss.pred.get("context_space", cfg.loss.pred.get("space", "normalized")))
    target_space = rollout_cfg.get("space", cfg.loss.pred.get("space", "normalized"))
    loss_type = rollout_cfg.get("type", cfg.loss.pred.get("type", "cosine"))

    full_act_emb = output["act_emb"][:, : ctx_len + rollout_steps]
    rollout_raw = output["emb_raw"][:, :ctx_len].clone()
    rollout_norm = output["emb"][:, :ctx_len].clone()
    target = get_embedding_tensor(output, space=target_space)[:, ctx_len : ctx_len + rollout_steps]

    pred_steps = []
    for step in range(rollout_steps):
        action_end = ctx_len + step
        window = min(ctx_len, action_end)
        rollout_ctx = get_embedding_tensor(
            {"emb_raw": rollout_raw, "emb": rollout_norm},
            space=context_space,
        )[:, -window:]
        act_trunc = full_act_emb[:, action_end - window : action_end]

        pred_raw = model.predict_raw(rollout_ctx, act_trunc)[:, -1:]
        pred_norm = model.normalize_embeddings(pred_raw)
        pred_steps.append(
            get_embedding_tensor({"emb_raw": pred_raw, "emb": pred_norm}, space=target_space)
        )

        rollout_raw = torch.cat([rollout_raw, pred_raw], dim=1)
        rollout_norm = torch.cat([rollout_norm, pred_norm], dim=1)

    if len(pred_steps) <= 1:
        return None

    # Keep the first rollout step only as autoregressive context so this
    # auxiliary term supervises strictly beyond the one-step pred_loss.
    pred = torch.cat(pred_steps[1:], dim=1)
    target = target[:, 1:]
    return compute_embedding_loss(pred, target, loss_type=loss_type)


def compute_temporal_hinge(output, *, model, cfg):
    emb = output["emb"]
    if emb.size(1) <= 1:
        output["temporal_hinge_active_ratio"] = emb.new_tensor(0.0)
        return emb.new_tensor(0.0)

    hinge_cfg = cfg.loss.temporal_hinge
    dynamic_cfg = hinge_cfg.get("dynamic", {})
    z_t = emb[:, :-1]
    z_tp1 = emb[:, 1:]

    if not dynamic_cfg.get("enabled", False):
        dist = 1.0 - (z_t * z_tp1).sum(dim=-1)
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

    dist = 1.0 - (z_t * z_tp1).sum(dim=-1)
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


def swm_forward(self, batch, stage, cfg):
    """Encode observations, predict next states, compute spherical losses.

    Losses:
      pred_loss   — configurable prediction loss in raw or normalized space
      reg_loss    — configurable anti-collapse loss in raw or normalized space
      temporal_hinge_loss — optional hinge loss on consecutive latent pairs
      loss        — pred_loss + λ_reg * reg_loss + λ_hinge * temporal_hinge_loss

    Note:
      temporal_straightness is logged as a monitoring metric using Euclidean
      displacement vectors between consecutive unit-norm latents. On the sphere
      this is only a chord-space approximation, so it may not reflect true
      geodesic straightening accurately.
    """
    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    reg_type = cfg.loss.regularizer.type
    reg_lambd = cfg.loss.regularizer.weight
    hinge_cfg = cfg.loss.temporal_hinge
    reg_space = cfg.loss.regularizer.get("space", "normalized")
    pred_space = cfg.loss.pred.get("space", "normalized")
    context_space = cfg.loss.pred.get("context_space", pred_space)

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D), unit vectors
    act_emb = output["act_emb"]

    # Make training-time autoregressive context explicit so raw-consistent
    # experiments can feed predictor history from the same space used for the
    # prediction target and planning rollout.
    ctx_emb = get_context_tensor(output, space=context_space)[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]

    pred_raw = self.model.predict_raw(ctx_emb, ctx_act)
    pred_emb = self.model.normalize_embeddings(pred_raw)

    pred_source, tgt_source = get_loss_space_tensors(
        output,
        pred_raw=pred_raw,
        pred_norm=pred_emb,
        n_preds=n_preds,
        space=pred_space,
    )
    output["pred_loss"] = compute_pred_loss(pred_source, tgt_source, cfg)
    output["rollout_loss"] = compute_multistep_rollout_loss(
        output,
        model=self.model,
        cfg=cfg,
    )

    reg_emb = get_regularizer_tensor(output, space=reg_space)
    reg_scope = cfg.loss.regularizer.get("scope", "full").lower()
    if reg_scope == "pred_window":
        reg_emb = reg_emb[:, : ctx_len + n_preds]
    elif reg_scope != "full":
        raise ValueError(f"Unsupported loss.regularizer.scope: {reg_scope}")
    if reg_type == "spread":
        output["spread_loss"] = spread_loss(reg_emb, cfg.loss.spread.margin)
        output["reg_loss"] = output["spread_loss"]
    elif reg_type == "uniformity":
        output["uniformity_loss"] = uniformity_loss(
            reg_emb,
            cfg.loss.uniformity.t,
            mode=cfg.loss.uniformity.get("mode", "all_pairs"),
            temporal_exclusion=cfg.loss.uniformity.get("temporal_exclusion", 0),
        )
        output["reg_loss"] = output["uniformity_loss"]
    elif reg_type == "infonce":
        tgt_emb = output["emb"][:, n_preds:]
        output["infonce_loss"] = infonce_loss(
            pred_emb, tgt_emb, cfg.loss.infonce.temperature
        )
        output["reg_loss"] = output["infonce_loss"]
    else:
        raise ValueError(f"Unsupported loss.regularizer.type: {reg_type}")

    output["temporal_hinge_loss"] = compute_temporal_hinge(
        output,
        model=self.model,
        cfg=cfg,
    )
    inverse_cfg = cfg.loss.get("inverse_dynamics", {})
    inverse_weight = inverse_cfg.get("weight", 0.0)
    if inverse_weight > 0.0:
        if not hasattr(self.model, "inverse_dynamics_head"):
            raise AttributeError(
                "loss.inverse_dynamics requires model.inverse_dynamics_head"
            )
        inverse_emb = get_embedding_tensor(
            output,
            space=inverse_cfg.get("space", "normalized"),
        )
        output["inverse_dynamics_loss"] = inverse_dynamics_loss(
            inverse_emb[:, :-1],
            inverse_emb[:, 1:],
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
        dist_space = dist_cfg.get("space", "normalized")
        dist_emb = get_embedding_tensor(output, space=dist_space)
        default_metric = "l2" if dist_space.lower() == "raw" else "cosine"
        (
            output["transition_distance_loss"],
            pred_dist,
            target_dist,
        ) = transition_distance_prediction_loss(
            dist_emb[:, :-1],
            dist_emb[:, 1:],
            self.model.transition_distance_head,
            metric=dist_cfg.get("metric", default_metric),
            detach_input=dist_cfg.get("detach_input", True),
        )
        output["transition_distance_pred_mean"] = pred_dist.mean()
        output["transition_distance_target_mean"] = target_dist.mean()
        output["transition_distance_target_std"] = target_dist.std(unbiased=False)

    output["loss"] = (
        output["pred_loss"]
        + reg_lambd * output["reg_loss"]
        + hinge_cfg.weight * output["temporal_hinge_loss"]
    )
    if "inverse_dynamics_loss" in output:
        output["loss"] = output["loss"] + inverse_weight * output["inverse_dynamics_loss"]
    if "transition_distance_loss" in output:
        output["loss"] = (
            output["loss"]
            + dist_weight * output["transition_distance_loss"]
        )
    if output["rollout_loss"] is not None:
        rollout_weight = cfg.loss.rollout.get("weight", 0.0)
        output["loss"] = output["loss"] + rollout_weight * output["rollout_loss"]
    # Approximate monitor only: on spherical latents this uses chord-space
    # displacements, not a geodesic straightness measure on the manifold.
    output["temporal_straightness"] = temporal_straightness(emb)

    losses_dict = {
        f"{stage}/{k}": v.detach()
        for k, v in output.items()
        if "loss" in k and torch.is_tensor(v)
    }
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
            )
        )
    }
    self.log_dict({**losses_dict, **metrics_dict}, on_step=True, sync_dist=True)
    return output


@hydra.main(version_base=None, config_path="./config/train", config_name="swm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    # Resolve H5 path explicitly to tolerate both swm layouts:
    #   0.0.6 wheel:          <STABLEWM_HOME>/<name>.h5
    #   post-PR-#221 source:  <STABLEWM_HOME>/datasets/<name>.h5
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
    img_noise = get_img_noise_transform(cfg.get("image_noise"))
    if img_noise is not None:
        train_set = TransformDataset(train_set, img_noise)
        if cfg.image_noise.get("apply_to_val", False):
            val_set = TransformDataset(val_set, img_noise)

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

    head_cfg = cfg.encoder.projection_head
    projector = build_projection_head(hidden_dim, embed_dim, head_cfg)
    pred_proj = build_projection_head(hidden_dim, embed_dim, head_cfg)
    inference_cfg = cfg.wm.get("inference", {})
    pred_cfg = cfg.loss.get("pred", {})
    training_context_space = pred_cfg.get(
        "context_space", pred_cfg.get("space", "normalized")
    )

    world_model = SphericalJEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj,
        inference_rollout_state_space=inference_cfg.get(
            "rollout_state_space", "normalized"
        ),
        inference_cost_space=inference_cfg.get("cost_space", "normalized"),
        inference_cost_type=inference_cfg.get("cost_type", "cosine"),
        analysis_prediction_space=cfg.loss.pred.get("space", "normalized"),
        training_context_space=training_context_space,
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

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        forward=partial(swm_forward, cfg=cfg),
        optim=optimizers,
    )

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
