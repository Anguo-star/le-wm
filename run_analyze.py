from pathlib import Path

import hydra
import torch
import stable_pretraining as spt
import stable_worldmodel as swm
from omegaconf import DictConfig, OmegaConf

from analyze.core import (
    collect_embeddings,
    load_encoder_model,
    run_mlp_probes,
    run_linear_probes,
    run_spherical_tsne,
    run_tsne,
    run_umap,
    save_results,
    write_embedding_dump,
)
from utils import get_img_preprocessor, resolve_h5_dataset_path


def build_dataset(cfg: DictConfig):
    print(f"[analyze] Loading dataset: {cfg.data.dataset.name}")
    # Resolve the H5 path explicitly so we tolerate both swm layouts
    # (0.0.6 flat vs post-PR-#221 `datasets/` subdir).
    data_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    h5_name = data_cfg.pop("name")
    h5_path = resolve_h5_dataset_path(h5_name, cache_dir=cfg.cache_dir)
    dataset = swm.data.HDF5Dataset(
        path=str(h5_path),
        transform=None,
        **data_cfg,
    )
    print(f"[analyze] Dataset path: {dataset.h5_path}")
    dataset.transform = spt.data.transforms.Compose(
        get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)
    )
    return dataset


@hydra.main(version_base=None, config_path="./config/analyze", config_name="base")
def run(cfg: DictConfig):
    model, checkpoint_path = load_encoder_model(cfg.checkpoint, cfg.cache_dir)
    dataset = build_dataset(cfg)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )

    output_dir = Path(cfg.output.dir)
    embeddings, targets = collect_embeddings(
        model=model,
        loader=loader,
        target_keys=list(cfg.analysis.target_keys),
        frame_index=cfg.analysis.frame_index,
        max_samples=cfg.analysis.max_samples,
        device=cfg.device,
    )

    if cfg.output.save_intermediate:
        write_embedding_dump(output_dir / "embeddings.npz", embeddings, targets)

    tsne_results = None
    if cfg.analysis.tsne.enabled:
        tsne_results = run_tsne(
            embeddings=embeddings,
            targets=targets,
            cfg=cfg.analysis.tsne,
            seed=cfg.seed,
            output_dir=output_dir,
            save_intermediate=cfg.output.save_intermediate,
        )

    umap_results = None
    if cfg.analysis.umap.enabled:
        umap_results = run_umap(
            embeddings=embeddings,
            targets=targets,
            cfg=cfg.analysis.umap,
            seed=cfg.seed,
            output_dir=output_dir,
            save_intermediate=cfg.output.save_intermediate,
        )

    spherical_tsne_results = None
    if cfg.analysis.spherical_tsne.enabled:
        spherical_tsne_results = run_spherical_tsne(
            embeddings=embeddings,
            targets=targets,
            cfg=cfg.analysis.spherical_tsne,
            seed=cfg.seed,
            output_dir=output_dir,
            save_intermediate=cfg.output.save_intermediate,
        )

    linear_probe_results = None
    if cfg.analysis.linear_probe.enabled:
        linear_probe_results = run_linear_probes(
            embeddings=embeddings,
            targets=targets,
            cfg=cfg.analysis.linear_probe,
            seed=cfg.seed,
        )

    mlp_probe_results = None
    if cfg.analysis.mlp_probe.enabled:
        mlp_probe_results = run_mlp_probes(
            embeddings=embeddings,
            targets=targets,
            cfg=cfg.analysis.mlp_probe,
            seed=cfg.seed,
        )

    save_results(
        output_dir,
        cfg,
        checkpoint_path,
        embeddings,
        targets,
        linear_probe_results,
        mlp_probe_results,
        tsne_results,
        umap_results=umap_results,
        spherical_tsne_results=spherical_tsne_results,
    )


if __name__ == "__main__":
    run()
