"""Train PLDM with optional input-side noise augmentation (Gaussian, per-frame).

This is a thin wrapper around the upstream PLDM training script
(``stable-worldmodel/scripts/train/pldm.py``). The wrapper:

  * uses Hydra config files in ``config/train/`` (this repo), so noise sweeps
    can be expressed in the same style as ``config/train/lewm.yaml``;
  * inserts our local ``utils.AddNormalizedGaussianNoise`` transform on the
    training split (matching ``train.py:815-819``), so the noise pipeline is
    bit-identical to the LeWM baseline; and
  * defers model/loss/forward to the upstream package so we never duplicate
    model code.

Use:

    python train_pldm.py data=pusht exp_name=pusht_pldm
    python train_pldm.py data=pusht exp_name=pusht_pldm_noise_0to006_p1 \\
        image_noise.std_max=0.06 image_noise.noise_prob=1.0
"""

from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
from stable_pretraining import data as dt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from loguru import logger as logging
from omegaconf import OmegaConf, open_dict
from torch import nn
from torch.utils.data import DataLoader

try:
    from swanlab.integration.pytorch_lightning import SwanLabLogger
except ImportError:
    SwanLabLogger = None

# --- upstream PLDM components (no duplication) -----------------------------
from stable_worldmodel.data import column_normalizer as get_column_normalizer
from stable_worldmodel.wm.pldm.module import MLP, Embedder, Predictor
from stable_worldmodel.wm.pldm import PLDM
from stable_worldmodel.wm.loss import PLDMLoss, TemporalStraighteningLoss

# --- our additions ---------------------------------------------------------
from utils import (
    ModelObjectCallBack,
    TransformDataset,
    get_img_noise_transform,
    get_img_preprocessor,
    resolve_h5_dataset_path,
)


# Mirror upstream pldm_forward verbatim so behaviour matches their published
# checkpoints. (Copied locally to avoid an upstream-import that depends on
# the upstream script-file path being on PYTHONPATH.)
def pldm_forward(self, batch, stage, cfg):
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode(batch)
    emb = output["emb"]
    act_emb = output["act_emb"]
    inpt_emb = emb[:, : cfg.wm.history_size]
    inpt_act = act_emb[:, : cfg.wm.history_size]
    tgt_emb = emb[:, cfg.wm.num_preds:]
    pred_emb = self.model.predict(inpt_emb, inpt_act)
    output["idm_emb"] = torch.cat([emb[:, 1:], emb[:, :-1]], dim=-1)
    output["act_label"] = batch["action"][:, :-1].detach()
    output["act_pred"] = self.idm(output["idm_emb"])
    output["pred_loss"] = (pred_emb - tgt_emb).square().mean()
    output["temp_straight_loss"] = self.path_straight(emb)
    output.update(self.pldm(emb, output["act_pred"], output["act_label"]))
    output["loss"] = output["pred_loss"]
    for k, v in cfg.loss.items():
        loss_key = f"{k}_loss"
        if not v.enabled or (loss_key not in output):
            continue
        output["loss"] = output["loss"] + v.weight * output[loss_key]
    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


@hydra.main(version_base=None, config_path="./config/train", config_name="pldm")
def run(cfg):
    # ----- dataset --------------------------------------------------------
    # Resolve the H5 path explicitly so we work with both the 0.0.6-wheel
    # flat layout (<STABLEWM_HOME>/<name>.h5) and the post-PR-#221 layout
    # (<STABLEWM_HOME>/datasets/<name>.h5). Passing `path=` bypasses the
    # hard-coded `sub_folder='datasets'` in the source version of
    # HDF5Dataset.__init__.
    data_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = data_cfg.pop("name")
    h5_path = resolve_h5_dataset_path(dataset_name)
    dataset = swm.data.HDF5Dataset(path=str(h5_path), transform=None, **data_cfg)

    img_processor = get_img_preprocessor("pixels", "pixels", cfg.img_size)
    extra_transforms = []
    for col in cfg.data.dataset.keys_to_load:
        if col == "pixels":
            continue
        extra_transforms.append(get_column_normalizer(dataset, col, col))
    if hasattr(cfg.data.dataset, "keys_to_merge"):
        for col in cfg.data.dataset.keys_to_merge:
            extra_transforms.append(get_column_normalizer(dataset, col, col))

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col == "pixels":
                continue
            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    dataset.transform = spt.data.transforms.Compose(img_processor, *extra_transforms)

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen,
    )

    # ----- input-side noise (the only addition vs upstream pldm.py) -------
    img_noise = get_img_noise_transform(cfg.get("image_noise"))
    if img_noise is not None:
        train_set = TransformDataset(train_set, img_noise)
        if cfg.image_noise.get("apply_to_val", False):
            val_set = TransformDataset(val_set, img_noise)

    train = DataLoader(train_set, **cfg.loader, generator=rnd_gen)
    val_cfg = {**cfg.loader, "shuffle": False, "drop_last": False}
    val = DataLoader(val_set, **val_cfg)

    # ----- model / optim --------------------------------------------------
    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )
    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)

    predictor = Predictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim
    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    projector = MLP(input_dim=hidden_dim, output_dim=embed_dim,
                    hidden_dim=2048, norm_fn=nn.BatchNorm1d)
    predictor_proj = MLP(input_dim=hidden_dim, output_dim=embed_dim,
                         hidden_dim=2048, norm_fn=nn.BatchNorm1d)
    idm = MLP(input_dim=2 * embed_dim, hidden_dim=512, output_dim=effective_act_dim)

    world_model = PLDM(encoder=encoder, predictor=predictor,
                       action_encoder=action_encoder,
                       projector=projector, pred_proj=predictor_proj)

    models = {"model": world_model, "idm": idm}
    losses = {"pldm": PLDMLoss(), "path_straight": TemporalStraighteningLoss()}

    total_steps = cfg.trainer.max_epochs * len(train)
    optimizers = {
        f"{name}_opt": {
            "modules": str(name),
            "optimizer": dict(cfg.optimizer),
            "scheduler": {
                "type": "LinearWarmupCosineAnnealingLR",
                "warmup_steps": max(1, int(0.01 * total_steps)),
                "max_steps": total_steps,
            },
            "interval": "epoch",
        }
        for name in models.keys()
    }

    data_module = spt.data.DataModule(train=train, val=val)
    module = spt.Module(
        **models, **losses, forward=partial(pldm_forward, cfg=cfg), optim=optimizers,
    )

    # ----- run dir / logger ----------------------------------------------
    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)
    logging.info(f"🫆 PLDM run dir: {run_dir}")

    logger_obj = None
    backend = str(cfg.get("logger_backend", "wandb"))
    if backend == "swanlab" and SwanLabLogger is not None and cfg.get("swanlab", {}).get("enabled", False):
        logger_obj = SwanLabLogger(**cfg.swanlab.config)
        logger_obj.log_hyperparams(OmegaConf.to_container(cfg))
    elif backend == "wandb" and cfg.get("wandb", {}).get("enabled", False):
        logger_obj = WandbLogger(**cfg.wandb.config)
        logger_obj.log_hyperparams(OmegaConf.to_container(cfg))

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir,
        filename=cfg.output_model_name,
        epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger_obj,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )
    manager()


if __name__ == "__main__":
    run()
