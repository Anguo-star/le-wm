import csv
import json
from pathlib import Path

import numpy as np
import torch
import stable_worldmodel as swm
from omegaconf import OmegaConf


def resolve_checkpoint_path(checkpoint: str, cache_dir: str | None = None) -> Path:
    path = Path(checkpoint)
    if not path.exists():
        root = (
            Path(cache_dir) if cache_dir is not None else swm.data.utils.get_cache_dir()
        )
        path = Path(root, checkpoint)

    if path.is_dir():
        candidates = sorted(
            path.glob("*_object.ckpt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(f"No *_object.ckpt found in {path}")
        return candidates[0]

    if path.suffix == ".ckpt":
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    path = Path(f"{path}_object.ckpt")
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def load_encoder_model(
    checkpoint: str, cache_dir: str | None = None
) -> tuple[torch.nn.Module, Path]:
    ckpt_path = resolve_checkpoint_path(checkpoint, cache_dir)
    print(f"[analyze] Loading checkpoint: {ckpt_path}")
    payload = torch.load(ckpt_path, weights_only=False, map_location="cpu")

    def scan(module):
        if isinstance(module, torch.nn.Module) and hasattr(module, "encode"):
            return module.eval()
        if isinstance(module, torch.nn.Module):
            for child in module.children():
                result = scan(child)
                if result is not None:
                    return result
        return None

    model = scan(payload)
    if model is None:
        raise RuntimeError(f"No module with encode() found in checkpoint {ckpt_path}")
    return model, ckpt_path


def to_numpy(value):
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def summarize_array(value: np.ndarray) -> dict:
    value = np.asarray(value)
    flat = value.reshape(value.shape[0], -1) if value.ndim > 1 else value.reshape(-1, 1)
    return {
        "shape": list(value.shape),
        "mean": flat.mean(axis=0).tolist(),
        "std": flat.std(axis=0).tolist(),
        "min": flat.min(axis=0).tolist(),
        "max": flat.max(axis=0).tolist(),
    }


def slice_frame(value, frame_index: int, sequence_len: int):
    value_np = to_numpy(value)
    if value_np.ndim >= 2 and value_np.shape[1] == sequence_len:
        return value_np[:, frame_index]
    return value_np


def collect_embeddings(model, loader, target_keys, frame_index, max_samples, device):
    print(
        f"[analyze] Extracting embeddings: frame_index={frame_index}, "
        f"max_samples={max_samples}, device={device}"
    )
    embeddings = []
    targets = {key: [] for key in target_keys}
    seen = 0
    available_keys = None

    model = model.to(device).eval()

    with torch.inference_mode():
        for batch in loader:
            if available_keys is None:
                available_keys = set(batch.keys())
                missing = [key for key in target_keys if key not in available_keys]
                if missing:
                    raise KeyError(
                        f"Requested target keys missing from dataset batch: {missing}"
                    )

            batch_size = batch["pixels"].shape[0]
            remaining = None if max_samples is None else max_samples - seen
            if remaining is not None and remaining <= 0:
                break
            take = batch_size if remaining is None else min(batch_size, remaining)

            pixels = batch["pixels"][:take].to(device)
            output = model.encode({"pixels": pixels})
            emb = output["emb"][:, frame_index]
            embeddings.append(emb.detach().cpu().numpy())

            sequence_len = batch["pixels"].shape[1]
            for key in target_keys:
                targets[key].append(
                    slice_frame(batch[key][:take], frame_index, sequence_len)
                )

            seen += take

    stacked_embeddings = np.concatenate(embeddings, axis=0)
    stacked_targets = {
        key: np.concatenate([to_numpy(v) for v in values], axis=0)
        for key, values in targets.items()
    }
    print(
        f"[analyze] Extracted embeddings: n_samples={stacked_embeddings.shape[0]}, "
        f"dim={stacked_embeddings.shape[1]}"
    )
    return stacked_embeddings, stacked_targets


def maybe_import_sklearn():
    try:
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.manifold import TSNE
        from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for analyze.py. Install it with `pip install scikit-learn`."
        ) from exc

    return {
        "LogisticRegression": LogisticRegression,
        "MLPClassifier": MLPClassifier,
        "MLPRegressor": MLPRegressor,
        "Ridge": Ridge,
        "TSNE": TSNE,
        "accuracy_score": accuracy_score,
        "mean_squared_error": mean_squared_error,
        "r2_score": r2_score,
        "train_test_split": train_test_split,
        "make_pipeline": make_pipeline,
        "StandardScaler": StandardScaler,
    }


def maybe_import_umap():
    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "umap-learn is required for UMAP analysis. Install it with `pip install umap-learn`."
        ) from exc

    return umap


def write_embedding_dump(
    path: Path, embeddings: np.ndarray, targets: dict[str, np.ndarray]
):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, embeddings=embeddings, **targets)


def _flatten_columns(name: str, value: np.ndarray):
    value = np.asarray(value)
    if value.ndim == 1:
        return {name: value}
    flat = value.reshape(value.shape[0], -1)
    return {f"{name}_{i}": flat[:, i] for i in range(flat.shape[1])}


def _save_projection(
    prefix: str,
    coords: np.ndarray,
    targets: dict[str, np.ndarray],
    output_dir: Path,
    save_intermediate: bool,
):
    if not save_intermediate:
        return None

    np.savez(output_dir / f"{prefix}.npz", coords=coords, **targets)

    columns = {f"{prefix}_{i}": coords[:, i] for i in range(coords.shape[1])}
    for key, value in targets.items():
        columns.update(_flatten_columns(key, value))

    csv_path = output_dir / f"{prefix}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        headers = list(columns.keys())
        writer.writerow(headers)
        for row in zip(*(columns[h] for h in headers)):
            writer.writerow(row)

    return csv_path


def _maybe_plot_projection(
    prefix: str,
    coords: np.ndarray,
    targets: dict[str, np.ndarray],
    color_by: str | None,
    output_dir: Path,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    color = None
    title = prefix.replace("_", " ")
    if color_by and color_by in targets:
        color = np.asarray(targets[color_by])
        if color.ndim > 1:
            color = color.reshape(color.shape[0], -1)[:, 0]
        title = f"{title} | color={color_by}"

    plot_path = output_dir / f"{prefix}.png"
    if coords.shape[1] >= 3:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")
        if color is None:
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
                s=6,
                alpha=0.8,
                color="tab:blue",
            )
        else:
            sc = ax.scatter(
                coords[:, 0], coords[:, 1], coords[:, 2], c=color, s=6, cmap="viridis"
            )
            fig.colorbar(sc, shrink=0.75)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close(fig)
        return str(plot_path)

    if coords.shape[1] >= 2:
        plt.figure(figsize=(6, 6))
        if color is None:
            plt.scatter(coords[:, 0], coords[:, 1], s=6, alpha=0.8, color="tab:blue")
        else:
            plt.scatter(coords[:, 0], coords[:, 1], c=color, s=6, cmap="viridis")
            plt.colorbar()
        plt.xlabel("dim 1")
        plt.ylabel("dim 2")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        return str(plot_path)

    return None


def run_tsne(
    embeddings: np.ndarray,
    targets: dict[str, np.ndarray],
    cfg,
    seed: int,
    output_dir: Path,
    save_intermediate: bool,
):
    sk = maybe_import_sklearn()
    n_samples = embeddings.shape[0]
    if n_samples < 2:
        raise ValueError("t-SNE requires at least 2 samples")

    perplexity = min(float(cfg.perplexity), max(1.0, n_samples - 1.0))
    print(f"[analyze] Running t-SNE: n_samples={n_samples}, perplexity={perplexity}")
    tsne = sk["TSNE"](
        n_components=cfg.n_components,
        perplexity=perplexity,
        init=cfg.init,
        learning_rate=cfg.learning_rate,
        random_state=seed,
    )
    coords = tsne.fit_transform(embeddings)
    csv_path = _save_projection("tsne", coords, targets, output_dir, save_intermediate)
    plot_path = _maybe_plot_projection(
        "tsne", coords, targets, cfg.get("color_by"), output_dir
    )
    return {
        "perplexity": perplexity,
        "csv": str(csv_path) if csv_path is not None else None,
        "plot": plot_path,
    }


def run_umap(
    embeddings: np.ndarray,
    targets: dict[str, np.ndarray],
    cfg,
    seed: int,
    output_dir: Path,
    save_intermediate: bool,
):
    if embeddings.shape[0] < 2:
        raise ValueError("UMAP requires at least 2 samples")

    umap = maybe_import_umap()
    n_neighbors = min(int(cfg.n_neighbors), max(2, embeddings.shape[0] - 1))
    print(
        f"[analyze] Running UMAP: n_samples={embeddings.shape[0]}, "
        f"n_neighbors={n_neighbors}, n_components={cfg.n_components}"
    )
    reducer = umap.UMAP(
        n_components=cfg.n_components,
        n_neighbors=n_neighbors,
        min_dist=cfg.min_dist,
        metric=cfg.metric,
        random_state=seed,
    )
    coords = reducer.fit_transform(embeddings)
    csv_path = _save_projection("umap", coords, targets, output_dir, save_intermediate)
    plot_path = _maybe_plot_projection(
        "umap", coords, targets, cfg.get("color_by"), output_dir
    )
    return {
        "n_neighbors": n_neighbors,
        "csv": str(csv_path) if csv_path is not None else None,
        "plot": plot_path,
    }


def run_spherical_tsne(
    embeddings: np.ndarray,
    targets: dict[str, np.ndarray],
    cfg,
    seed: int,
    output_dir: Path,
    save_intermediate: bool,
):
    sk = maybe_import_sklearn()
    n_samples = embeddings.shape[0]
    if n_samples < 2:
        raise ValueError("Spherical t-SNE requires at least 2 samples")

    perplexity = min(float(cfg.perplexity), max(1.0, n_samples - 1.0))
    print(
        f"[analyze] Running spherical t-SNE: n_samples={n_samples}, "
        f"perplexity={perplexity}"
    )
    tsne = sk["TSNE"](
        n_components=3,
        perplexity=perplexity,
        init=cfg.init,
        learning_rate=cfg.learning_rate,
        random_state=seed,
    )
    coords = tsne.fit_transform(embeddings)
    norms = np.linalg.norm(coords, axis=1, keepdims=True)
    coords = coords / np.clip(norms, a_min=1e-12, a_max=None)

    csv_path = _save_projection(
        "spherical_tsne", coords, targets, output_dir, save_intermediate
    )
    plot_path = _maybe_plot_projection(
        "spherical_tsne", coords, targets, cfg.get("color_by"), output_dir
    )
    return {
        "perplexity": perplexity,
        "csv": str(csv_path) if csv_path is not None else None,
        "plot": plot_path,
    }


def infer_probe_task(y: np.ndarray, max_classes: int) -> str:
    y = np.asarray(y)
    if y.ndim == 1:
        rounded = np.round(y)
        unique = np.unique(rounded)
        if np.allclose(y, rounded) and unique.size <= max_classes:
            return "classification"
    return "regression"


def run_linear_probes(
    embeddings: np.ndarray, targets: dict[str, np.ndarray], cfg, seed: int
):
    sk = maybe_import_sklearn()
    print(f"[analyze] Running linear probes: targets={list(cfg.target_keys)}")
    results = {}

    for key in cfg.target_keys:
        y = np.asarray(targets[key])
        task = infer_probe_task(y, cfg.classification_max_classes)

        split = sk["train_test_split"](
            embeddings, y, test_size=cfg.test_size, random_state=seed
        )
        x_train, x_test, y_train, y_test = split

        if task == "classification":
            model = sk["make_pipeline"](
                sk["StandardScaler"](),
                sk["LogisticRegression"](max_iter=cfg.max_iter),
            )
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            results[key] = {
                "task": task,
                "accuracy": float(sk["accuracy_score"](y_test, pred)),
                "n_classes": int(np.unique(y_train).size),
            }
        else:
            model = sk["make_pipeline"](
                sk["StandardScaler"](),
                sk["Ridge"](alpha=cfg.ridge_alpha),
            )
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            results[key] = {
                "task": task,
                "r2": float(sk["r2_score"](y_test, pred)),
                "mse": float(sk["mean_squared_error"](y_test, pred)),
                "target_dim": int(y_train.reshape(y_train.shape[0], -1).shape[1]),
            }

    return results


def run_mlp_probes(
    embeddings: np.ndarray, targets: dict[str, np.ndarray], cfg, seed: int
):
    sk = maybe_import_sklearn()
    print(
        f"[analyze] Running MLP probes: targets={list(cfg.target_keys)}, "
        f"hidden_layers={list(cfg.hidden_layers)}"
    )
    results = {}

    hidden_layers = tuple(int(x) for x in cfg.hidden_layers)

    for key in cfg.target_keys:
        y = np.asarray(targets[key])
        task = infer_probe_task(y, cfg.classification_max_classes)

        split = sk["train_test_split"](
            embeddings, y, test_size=cfg.test_size, random_state=seed
        )
        x_train, x_test, y_train, y_test = split

        if task == "classification":
            model = sk["make_pipeline"](
                sk["StandardScaler"](),
                sk["MLPClassifier"](
                    hidden_layer_sizes=hidden_layers,
                    max_iter=cfg.max_iter,
                    learning_rate_init=cfg.learning_rate_init,
                    random_state=seed,
                ),
            )
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            results[key] = {
                "task": task,
                "accuracy": float(sk["accuracy_score"](y_test, pred)),
                "n_classes": int(np.unique(y_train).size),
                "hidden_layers": list(hidden_layers),
            }
        else:
            model = sk["make_pipeline"](
                sk["StandardScaler"](),
                sk["MLPRegressor"](
                    hidden_layer_sizes=hidden_layers,
                    max_iter=cfg.max_iter,
                    learning_rate_init=cfg.learning_rate_init,
                    random_state=seed,
                ),
            )
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            results[key] = {
                "task": task,
                "r2": float(sk["r2_score"](y_test, pred)),
                "mse": float(sk["mean_squared_error"](y_test, pred)),
                "target_dim": int(y_train.reshape(y_train.shape[0], -1).shape[1]),
                "hidden_layers": list(hidden_layers),
            }

    return results


def save_results(
    output_dir: Path,
    cfg,
    checkpoint_path: Path,
    embeddings: np.ndarray,
    targets: dict[str, np.ndarray],
    linear_probe_results,
    mlp_probe_results,
    tsne_results,
    umap_results=None,
    spherical_tsne_results=None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[analyze] Saving analysis outputs to: {output_dir}")
    print(
        f"[analyze] Summary stats: n_samples={embeddings.shape[0]}, "
        f"embedding_dim={embeddings.shape[1]}, targets={list(targets.keys())}"
    )
    with (output_dir / "summary.json").open("w") as f:
        json.dump(
            {
                "checkpoint": str(checkpoint_path),
                "dataset_stats": {
                    "num_samples": int(embeddings.shape[0]),
                    "embedding_dim": int(embeddings.shape[1]),
                    "targets": {
                        key: summarize_array(value) for key, value in targets.items()
                    },
                },
                "linear_probe_results": linear_probe_results,
                "mlp_probe_results": mlp_probe_results,
                "tsne": tsne_results,
                "umap": umap_results,
                "spherical_tsne": spherical_tsne_results,
                "config": OmegaConf.to_container(cfg, resolve=True),
            },
            f,
            indent=2,
        )
