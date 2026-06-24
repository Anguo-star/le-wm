"""Train PreJEPA (DINO-WM family) with optional input-side noise augmentation.

Thin wrapper around the upstream PreJEPA training script
(``stable-worldmodel/scripts/train/prejepa.py``). The wrapper:

  * uses Hydra configs in ``config/train/`` so noise sweeps share style with
    ``config/train/lewm.yaml`` and ``config/train/pldm.yaml``;
  * inserts our local ``utils.AddNormalizedGaussianNoise`` transform on the
    training split (matching ``train.py:815-819``);
  * defers model / loss / forward to the upstream ``stable_worldmodel`` package.

Two recommended configurations:

  * **no-prop (default, main result for this paper)** — set
    ``wm.encoding={action: 10}`` (only the action stream remains as an
    auxiliary encoder). The model is then vision-only, comparable to LeWM
    and PLDM under our visual-OOD framing.
  * **with-prop (appendix sensitivity check)** — set
    ``wm.encoding={proprio: 10, action: 10}``. This matches DINO-WM's
    original benchmark configuration and helps argue that visual noise
    still hurts even with a clean proprio side-channel.

Use:

    python train_prejepa.py exp_name=pusht_dinowm_noprop
    python train_prejepa.py exp_name=pusht_dinowm_noprop_noise_0to006_p1 \\
        image_noise.std_max=0.06 image_noise.noise_prob=1.0
    # appendix variant with proprio:
    python train_prejepa.py exp_name=pusht_dinowm_prop \\
        wm.encoding.proprio=10
"""

from collections import OrderedDict
from functools import partial
import os
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from loguru import logger as logging
from omegaconf import OmegaConf, open_dict
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoVideoProcessor

try:
    from swanlab.integration.pytorch_lightning import SwanLabLogger
except ImportError:
    SwanLabLogger = None

# --- upstream PreJEPA components (no duplication) ---------------------------
from stable_worldmodel.data import column_normalizer as get_column_normalizer

# Re-use the upstream training script's helper functions verbatim so behaviour
# stays in lock-step with their published checkpoints.
import importlib.util as _ilu
import sys as _sys

_UPSTREAM_SCRIPT = (
    Path(os.environ.get("STABLE_WORLDMODEL_REPO", "external/stable-worldmodel"))
    / "scripts"
    / "train"
    / "prejepa.py"
)
if _UPSTREAM_SCRIPT.exists():
    _spec = _ilu.spec_from_file_location("_upstream_prejepa", _UPSTREAM_SCRIPT)
    _mod = _ilu.module_from_spec(_spec)
    _sys.modules["_upstream_prejepa"] = _mod
    _spec.loader.exec_module(_mod)
    get_encoder = _mod.get_encoder
    VideoPipeline = _mod.VideoPipeline
    dinowm_forward = _mod.dinowm_forward
    get_img_preprocessor = _mod.get_img_preprocessor
else:  # fallback: assume upstream is on PYTHONPATH as a package, not script
    raise RuntimeError(
        "Could not locate upstream prejepa.py. Set STABLE_WORLDMODEL_REPO to "
        "a checkout of stable-worldmodel, or place it under "
        "external/stable-worldmodel before running."
    )

# --- our additions ----------------------------------------------------------
from utils import ModelObjectCallBack, TransformDataset, get_img_noise_transform


@hydra.main(version_base=None, config_path="./config/train", config_name="prejepa")
def run(cfg):
    # --- Dataset ---
    encoding_keys = list(cfg.wm.get("encoding", {}).keys())
    keys_to_load = ["pixels"] + encoding_keys
    dataset = swm.data.load_dataset(
        cfg.dataset_name,
        num_steps=cfg.n_steps,
        frameskip=cfg.frameskip,
        transform=None,
        cache_dir=cfg.get("cache_dir", None),
        keys_to_load=keys_to_load,
        keys_to_cache=encoding_keys,
    )

    normalizers = [
        get_column_normalizer(dataset, col, col)
        for col in cfg.wm.get("encoding", {})
    ]

    if cfg.backbone.get("is_video_encoder", False):
        processor = AutoVideoProcessor.from_pretrained(cfg.backbone.name)
        transform = spt.data.transforms.Compose(
            VideoPipeline(processor, source="pixels", target="pixels"),
            spt.data.transforms.Resize(cfg.image_size, source="pixels", target="pixels"),
            *normalizers,
        )
    else:
        transform = spt.data.transforms.Compose(
            get_img_preprocessor("pixels", "pixels", cfg.image_size),
            *normalizers,
        )
    dataset.transform = transform

    with open_dict(cfg) as cfg:
        cfg.extra_dims = {}
        for key in cfg.wm.get("encoding", {}):
            if key not in dataset.column_names:
                raise ValueError(f"Encoding key '{key}' not found in dataset columns.")
            dim = dataset.get_dim(key)
            cfg.extra_dims[key] = dim if key != "action" else dim * cfg.frameskip

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, [cfg.train_split, 1 - cfg.train_split], generator=rnd_gen,
    )

    # --- input-side noise (the only addition vs upstream prejepa.py) -------
    img_noise = get_img_noise_transform(cfg.get("image_noise"))
    if img_noise is not None:
        train_set = TransformDataset(train_set, img_noise)
        if cfg.image_noise.get("apply_to_val", False):
            val_set = TransformDataset(val_set, img_noise)

    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
        drop_last=True, persistent_workers=True, pin_memory=True, shuffle=True,
        generator=rnd_gen,
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True,
    )

    # --- Model (verbatim from upstream) ------------------------------------
    encoder, embed_dim, num_patches, interp_pos_enc = get_encoder(cfg)
    embed_dim += sum(cfg.wm.get("encoding", {}).values())
    if cfg.backbone.get("is_video_encoder", False):
        num_patches += num_patches * (cfg.n_steps // 4)

    predictor_kwargs = {k: v for k, v in cfg.predictor.items() if k != "size"}
    predictor = swm.wm.prejepa.CausalPredictor(
        num_patches=num_patches,
        num_frames=cfg.wm.history_size,
        dim=embed_dim,
        **predictor_kwargs,
    )

    extra_encoders = nn.ModuleDict(
        OrderedDict(
            (key, swm.wm.prejepa.Embedder(in_chans=cfg.extra_dims[key], emb_dim=emb_dim))
            for key, emb_dim in cfg.wm.get("encoding", {}).items()
        )
    )

    world_model = swm.wm.PreJEPA(
        encoder=spt.backbone.EvalOnly(encoder),
        predictor=predictor,
        extra_encoders=extra_encoders,
        history_size=cfg.wm.history_size,
        num_pred=cfg.wm.num_preds,
        interpolate_pos_encoding=interp_pos_enc,
    )

    module = spt.Module(
        model=world_model,
        forward=partial(dinowm_forward, cfg=cfg),
        optim={"model_opt": {"modules": "model", "optimizer": dict(cfg.optimizer)}},
    )

    # --- Run dir / logger -------------------------------------------------
    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"PreJEPA run dir: {run_dir}")
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    logger_obj = None
    backend = str(cfg.get("logger_backend", "wandb"))
    if backend == "swanlab" and SwanLabLogger is not None and cfg.get("swanlab", {}).get("enabled", False):
        logger_obj = SwanLabLogger(**cfg.swanlab.config)
        logger_obj.log_hyperparams(OmegaConf.to_container(cfg))
    elif backend == "wandb" and cfg.get("wandb", {}).get("enabled", False):
        logger_obj = WandbLogger(**cfg.wandb.config)
        logger_obj.log_hyperparams(OmegaConf.to_container(cfg))

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[
            spt.callbacks.CPUOffloadCallback(),
            ModelObjectCallBack(
                dirpath=run_dir,
                filename=cfg.output_model_name,
                epoch_interval=1,
            ),
            pl.pytorch.callbacks.LearningRateMonitor(logging_interval="step"),
        ],
        num_sanity_val_steps=1,
        logger=logger_obj,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=module,
        data=spt.data.DataModule(train=train_loader, val=val_loader),
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )
    manager()


if __name__ == "__main__":
    run()
