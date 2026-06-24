"""
analyze_repr.py - Representation analysis for LeWM / SWM checkpoints.

The analysis now treats representation quality in three layers:

1. embedding
   anti-collapse / capacity usage
2. prediction and rollout
   can the model predict the next latent, and does autoregressive rollout drift
   when driven by real action sequences from the dataset
3. planning
   does the model's own cost function assign lower cost to expert futures than
   to random action futures for the same start and goal

Optional:
4. reference_probe
   if `state_key` is provided, compare latent geometry to that external proxy.
   This is a probe only, not the main judge of representation quality.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F

import stable_pretraining as spt
import stable_worldmodel as swm

from utils import get_column_normalizer, get_img_preprocessor, resolve_h5_dataset_path

SECTION_ORDER = [
    "meta",
    "embedding",
    "prediction",
    "rollout",
    "planning",
    "action_effect",
    "reference_probe",
    "visualization",
]
SECTION_TITLES = {
    "meta": "Meta",
    "embedding": "Embedding",
    "prediction": "Prediction",
    "rollout": "Rollout",
    "planning": "Planning",
    "action_effect": "Action Effect",
    "reference_probe": "Reference Probe",
    "visualization": "Visualization",
}

METRIC_SPECS: Dict[str, Dict[str, Dict[str, str]]] = {
    "embedding": {
        "n_embeddings": {
            "better": "context",
            "summary": "Number of flattened latent points included in the analysis.",
            "use": "Sanity check that different runs were analyzed on the same point budget.",
        },
        "dim": {
            "better": "context",
            "summary": "Latent embedding dimensionality.",
            "use": "Context only; this is not a quality metric by itself.",
        },
        "norm_mean": {
            "better": "context",
            "summary": "Average L2 norm of latent embeddings.",
            "use": "Useful before comparing any scale-dependent latent error or shift metrics.",
        },
        "norm_std": {
            "better": "context",
            "summary": "Standard deviation of latent embedding norms.",
            "use": "Near-zero values often indicate a thin-shell or unit-sphere representation.",
        },
        "dim_std_mean": {
            "better": "higher",
            "summary": "Average standard deviation across latent dimensions.",
            "use": "Quick anti-collapse check. Low values mean many coordinates barely move.",
        },
        "dim_std_min": {
            "better": "higher",
            "summary": "Smallest per-dimension standard deviation.",
            "use": "Detects dead latent directions.",
        },
        "dim_std_max": {
            "better": "context",
            "summary": "Largest per-dimension standard deviation.",
            "use": "Helps spot a representation dominated by only a few coordinates.",
        },
        "inter_sample_cosine_mean": {
            "better": "lower_abs",
            "summary": "Average pairwise cosine similarity between different samples.",
            "use": "Collapse check. Values drifting toward 1 mean different samples point in nearly the same direction.",
        },
        "inter_sample_cosine_std": {
            "better": "context",
            "summary": "Spread of pairwise cosine similarities.",
            "use": "Read together with the mean to tell whether the cloud is uniform or anisotropic.",
        },
        "effective_rank": {
            "better": "higher",
            "summary": "Entropy-based effective rank of the centered embedding matrix.",
            "use": "Approximate number of directions that actually carry variance, not just the nominal embedding size.",
        },
    },
    "prediction": {
        "n_prediction_windows": {
            "better": "context",
            "summary": "Number of teacher-forced prediction windows evaluated.",
            "use": "Sanity check that two runs used the same evaluation budget.",
        },
        "prediction_horizon": {
            "better": "context",
            "summary": "Number of one-step prediction windows per sampled sequence.",
            "use": "Context for interpreting the aggregate prediction metrics.",
        },
        "pred_error_mean": {
            "better": "lower",
            "summary": "Average Euclidean one-step prediction error in the model's analysis space.",
            "use": "Within-family predictor metric. Read together with norm statistics before comparing across very different latent scales.",
        },
        "pred_error_std": {
            "better": "lower",
            "summary": "Standard deviation of one-step Euclidean prediction error.",
            "use": "Tells whether the predictor is consistently wrong or fails only on certain transitions.",
        },
        "pred_target_cosine_mean": {
            "better": "higher",
            "summary": "Average cosine similarity between one-step prediction and target embedding.",
            "use": "Scale-free one-step prediction metric.",
        },
        "pred_target_cosine_std": {
            "better": "lower",
            "summary": "Standard deviation of prediction-target cosine similarity.",
            "use": "Lower values mean prediction quality is more consistent across windows.",
        },
        "pred_target_cosine_distance_mean": {
            "better": "lower",
            "summary": "Average cosine distance between one-step prediction and target embedding.",
            "use": "Lower-is-better version of one-step cosine prediction quality.",
        },
    },
    "rollout": {
        "rollout_horizon": {
            "better": "context",
            "summary": "Number of future steps rolled out autoregressively with real dataset actions.",
            "use": "Context for interpreting rollout drift metrics.",
        },
        "rollout_error_mean": {
            "better": "lower",
            "summary": "Average autoregressive rollout error against encoded future embeddings.",
            "use": "Primary self-consistency metric for multi-step latent prediction.",
        },
        "rollout_error_std": {
            "better": "lower",
            "summary": "Standard deviation of autoregressive rollout error.",
            "use": "Large spread means rollout quality depends strongly on the sampled future window.",
        },
        "rollout_error_last_mean": {
            "better": "lower",
            "summary": "Average rollout error at the last predicted horizon step.",
            "use": "Direct measure of compounding error at the longest horizon tested.",
        },
        "rollout_cosine_distance_mean": {
            "better": "lower",
            "summary": "Average cosine distance between rollout predictions and encoded future targets.",
            "use": "Scale-free autoregressive drift metric.",
        },
        "rollout_cosine_distance_last_mean": {
            "better": "lower",
            "summary": "Cosine distance at the last rollout step.",
            "use": "Useful when deciding whether a model stays coherent over the whole planning horizon.",
        },
        "rollout_error_growth": {
            "better": "lower",
            "summary": "Ratio of last-step rollout error to first-step rollout error.",
            "use": "Values far above 1 mean compounding error grows quickly over time.",
        },
    },
    "planning": {
        "planning_horizon": {
            "better": "context",
            "summary": "Number of future actions optimized or compared when probing planning cost.",
            "use": "Context for the planning-signal metrics below.",
        },
        "random_action_trials": {
            "better": "context",
            "summary": "Number of random action futures sampled per sequence for the cost probe.",
            "use": "Sanity check for the stability of expert-vs-random comparisons.",
        },
        "expert_cost_mean": {
            "better": "lower",
            "summary": "Average model cost assigned to the dataset future action sequence.",
            "use": "Reference cost for the expert/action-from-data future.",
        },
        "expert_cost_std": {
            "better": "lower",
            "summary": "Standard deviation of expert cost across sampled sequences.",
            "use": "High variance means the cost signal is unstable across contexts.",
        },
        "random_cost_mean": {
            "better": "context",
            "summary": "Average model cost assigned to random future action sequences.",
            "use": "This should be higher than expert cost if the planner signal is useful.",
        },
        "random_cost_std": {
            "better": "context",
            "summary": "Standard deviation of random-sequence costs.",
            "use": "Helps judge whether the random baseline is narrow or very noisy.",
        },
        "best_random_cost_mean": {
            "better": "context",
            "summary": "Average best (lowest) random cost per sequence.",
            "use": "Harder comparison than the random mean. Expert should usually beat this too.",
        },
        "cost_margin_mean": {
            "better": "higher",
            "summary": "Average random-minus-expert cost gap.",
            "use": "Primary planning-signal metric. Higher means the model separates expert futures from random futures.",
        },
        "cost_margin_std": {
            "better": "lower",
            "summary": "Standard deviation of the random-minus-expert cost gap.",
            "use": "High spread means the planning signal is unreliable across start-goal pairs.",
        },
        "expert_beats_random_rate": {
            "better": "higher",
            "summary": "Fraction of random candidates whose cost is higher than the expert candidate.",
            "use": "Easy-to-read planning signal. Near 1 means expert usually ranks better than random.",
        },
        "expert_beats_best_random_rate": {
            "better": "higher",
            "summary": "Fraction of sequences where expert beats even the best random candidate.",
            "use": "Stricter planning metric. Useful when random mean is easy to beat but best-random is not.",
        },
    },
    "action_effect": {
        "n_trials": {
            "better": "context",
            "summary": "Number of perturbation trials used in action-effect analysis.",
            "use": "Context field for the robustness of the perturbation statistics.",
        },
        "n_action_pairs": {
            "better": "context",
            "summary": "Number of action-perturbation / prediction-shift pairs evaluated.",
            "use": "Sanity check that the action-effect estimate is based on enough samples.",
        },
        "n_interp_anchors": {
            "better": "context",
            "summary": "Number of anchors used for action interpolation tests.",
            "use": "Context field for how many contexts contribute to interpolation smoothness.",
        },
        "perturb_scale": {
            "better": "context",
            "summary": "Scale multiplier applied to action standard deviation when perturbing actions.",
            "use": "Keep this fixed across runs if you want fair action-effect comparisons.",
        },
        "mean_action_perturb_norm": {
            "better": "context",
            "summary": "Average norm of the sampled action perturbations.",
            "use": "Reference scale for interpreting the resulting latent shift.",
        },
        "mean_pred_shift_norm": {
            "better": "context",
            "summary": "Average norm of the prediction change caused by action perturbations.",
            "use": "Within-method measure of how much action can move the prediction.",
        },
        "action_perturb_pred_shift_corr": {
            "better": "higher",
            "summary": "Correlation between perturbation magnitude and prediction-shift magnitude.",
            "use": "Checks whether larger action changes reliably create larger latent changes.",
        },
        "interpolation_endpoint_shift": {
            "better": "context",
            "summary": "Average latent distance from the interpolation start to the interpolation end.",
            "use": "Context for how much the chosen interpolation actually moves the prediction.",
        },
        "interpolation_endpoint_shift_std": {
            "better": "lower",
            "summary": "Standard deviation of endpoint shift across interpolation anchors.",
            "use": "High spread means action sensitivity is unstable across contexts.",
        },
        "interpolation_monotonicity": {
            "better": "higher",
            "summary": "Fraction of interpolation steps whose latent distance grows monotonically from the start.",
            "use": "Smoothness check. High values mean action interpolation produces orderly latent motion instead of jagged jumps.",
        },
        "interpolation_monotonicity_std": {
            "better": "lower",
            "summary": "Standard deviation of interpolation monotonicity across anchors.",
            "use": "Low spread means the smoothness pattern is stable across contexts.",
        },
    },
    "reference_probe": {
        "reference_state_key": {
            "better": "context",
            "summary": "Dataset key used as an external geometry proxy.",
            "use": "Reference only. This probe does not define the ground truth planning geometry.",
        },
        "n_points": {
            "better": "context",
            "summary": "Number of flattened points used for the reference probe.",
            "use": "Sanity check that two runs used the same comparison budget.",
        },
        "k": {
            "better": "context",
            "summary": "Neighborhood size used in the reference kNN overlap metric.",
            "use": "Context field for interpreting `knn_overlap` values.",
        },
        "distance_rank_corr_cross_seq": {
            "better": "higher",
            "summary": "Rank correlation between latent and reference-proxy distances on cross-sequence pairs.",
            "use": "Optional probe only. Useful if the chosen reference state is known to be task-meaningful.",
        },
        "knn_overlap_cross_seq": {
            "better": "higher",
            "summary": "kNN overlap between latent and reference-proxy neighborhoods after excluding same-sequence points.",
            "use": "Optional local-geometry probe against the external state proxy.",
        },
        "latent_reference_step_corr": {
            "better": "higher",
            "summary": "Correlation between latent step size and reference-proxy step size.",
            "use": "Optional motion-alignment probe if the reference state is trustworthy.",
        },
    },
    "visualization": {
        "tsne_exported": {
            "better": "context",
            "summary": "Whether t-SNE export succeeded.",
            "use": "Context only. t-SNE is for visual inspection, not for quantitative model selection.",
        },
        "tsne_perplexity": {
            "better": "context",
            "summary": "Perplexity actually used for t-SNE export.",
            "use": "Useful when comparing plots produced from different sample counts.",
        },
        "tsne_error": {
            "better": "context",
            "summary": "Error message captured when t-SNE export failed.",
            "use": "Debug field for the visualization step.",
        },
    },
}

NON_TABLE_SECTIONS = {"meta", "interpretation", "visualization"}

DEFAULT_KEY_METRICS: List[Dict[str, str]] = [
    {
        "group": "Embedding",
        "metric_name": "embedding.effective_rank",
        "label": "effective_rank",
        "short_label": "rank",
        "better": "higher",
        "brief": "used latent directions",
        "summary": "How many latent directions carry meaningful variance.",
        "diagnosis": "Low means the code underuses capacity or is partially collapsed.",
    },
    {
        "group": "Embedding",
        "metric_name": "embedding.inter_sample_cosine_mean",
        "label": "inter_sample_cosine_mean",
        "short_label": "inter_cos",
        "better": "lower_abs",
        "brief": "sample spread",
        "summary": "Average alignment between different samples.",
        "diagnosis": "Large magnitude means samples crowd into similar directions instead of spreading cleanly.",
    },
    {
        "group": "Prediction",
        "metric_name": "prediction.pred_target_cosine_distance_mean",
        "label": "pred_target_cosine_distance_mean",
        "short_label": "pred_cos_dist",
        "better": "lower",
        "brief": "one-step predictor accuracy",
        "summary": "Scale-free one-step prediction error.",
        "diagnosis": "High means the predictor is missing the next latent target.",
    },
    {
        "group": "Rollout",
        "metric_name": "rollout.rollout_cosine_distance_last_mean",
        "label": "rollout_cosine_distance_last_mean",
        "short_label": "rollout_last",
        "better": "lower",
        "brief": "multi-step drift at horizon end",
        "summary": "Autoregressive rollout error at the last tested horizon step.",
        "diagnosis": "High means rollout drifts even if one-step prediction looks good.",
    },
    {
        "group": "Planning",
        "metric_name": "planning.cost_margin_mean",
        "label": "cost_margin_mean",
        "short_label": "cost_margin",
        "better": "higher",
        "brief": "expert vs random cost gap",
        "summary": "Random-minus-expert cost gap under the model's own planner objective.",
        "diagnosis": "Low or negative means the planner cost cannot separate good futures from random ones.",
    },
    {
        "group": "Planning",
        "metric_name": "planning.expert_beats_random_rate",
        "label": "expert_beats_random_rate",
        "short_label": "expert>random",
        "better": "higher",
        "brief": "planner preference rate",
        "summary": "Fraction of random candidates whose cost is worse than the expert future.",
        "diagnosis": "Low means the planner cannot reliably rank expert futures above random ones.",
    },
    {
        "group": "Action Effect",
        "metric_name": "action_effect.mean_pred_shift_norm",
        "label": "mean_pred_shift_norm",
        "short_label": "pred_shift",
        "better": "higher",
        "brief": "action-induced shift size",
        "summary": "How much action perturbations move the predicted latent state.",
        "diagnosis": "Too low means actions barely change the rollout, even if correlation looks decent.",
    },
    {
        "group": "Action Effect",
        "metric_name": "action_effect.action_perturb_pred_shift_corr",
        "label": "action_perturb_pred_shift_corr",
        "short_label": "act_shift_corr",
        "better": "higher",
        "brief": "action magnitude consistency",
        "summary": "Whether larger action perturbations cause larger latent shifts.",
        "diagnosis": "Low means action magnitude is not translated into latent dynamics reliably.",
    },
]


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "This helper requires pandas. Install it with `pip install pandas`."
        ) from exc
    return pd


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "This helper requires matplotlib. Install it with `pip install matplotlib`."
        ) from exc
    return plt


def load_model(ckpt_path: str, device: str = "cpu"):
    model = torch.load(ckpt_path, map_location=device, weights_only=False)
    if hasattr(model, "model"):
        model = model.model
    return model.to(device).eval().requires_grad_(False)


def infer_history_size(model) -> int:
    predictor = getattr(model, "predictor", None)
    if predictor is None or not hasattr(predictor, "pos_embedding"):
        raise ValueError("Unable to infer history_size from model.predictor.pos_embedding")
    return int(predictor.pos_embedding.shape[1])


def resolve_space_name(space: str | None, default: str = "normalized") -> str:
    space = (space or default).lower()
    return "normalized" if space == "sphere" else space


def get_embedding_space(outputs: Mapping[str, torch.Tensor], space: str) -> torch.Tensor:
    space = resolve_space_name(space)
    if space == "raw":
        return outputs["emb_raw"]
    if space == "normalized":
        return outputs["emb"]
    raise ValueError(f"Unsupported embedding space: {space}")


def get_model_spaces(model) -> Dict[str, str]:
    analysis_space = resolve_space_name(getattr(model, "analysis_prediction_space", "normalized"))
    context_space = resolve_space_name(getattr(model, "training_context_space", analysis_space))
    rollout_space = resolve_space_name(getattr(model, "inference_rollout_state_space", "normalized"))
    cost_space = resolve_space_name(getattr(model, "inference_cost_space", "normalized"))
    # Plain JEPA does not carry the newer inference_* attributes and its
    # planner criterion is MSE. SphericalJEPA sets inference_cost_type itself,
    # usually to cosine, so this default only affects the LeWM baseline.
    cost_type = getattr(model, "inference_cost_type", "mse").lower()
    return {
        "analysis_prediction_space": analysis_space,
        "training_context_space": context_space,
        "inference_rollout_state_space": rollout_space,
        "inference_cost_space": cost_space,
        "inference_cost_type": cost_type,
    }


def load_dataset_samples(
    *,
    dataset_name: str,
    state_key: str | None,
    n_sequences: int,
    history_size: int,
    future_steps: int,
    frameskip: int,
    img_size: int,
    seed: int,
    device: str,
):
    num_steps = history_size + future_steps
    keys_to_load = ["pixels", "action"]
    if state_key:
        keys_to_load.append(state_key)

    # Resolve the H5 path explicitly so the diagnostic suite works against
    # both swm layouts: 0.0.6 wheel (flat <base>/<name>.h5) and post-PR-#221
    # source (<base>/datasets/<name>.h5). Passing `path=` bypasses the
    # hard-coded sub_folder='datasets' in the source version. Nested names
    # like "ogbench/cube_single_expert" are preserved verbatim.
    h5_path = resolve_h5_dataset_path(dataset_name)
    ds = swm.data.HDF5Dataset(
        path=str(h5_path),
        num_steps=num_steps,
        frameskip=frameskip,
        keys_to_load=keys_to_load,
        transform=None,
    )

    transforms = [
        get_img_preprocessor("pixels", "pixels", img_size),
        get_column_normalizer(ds, "action", "action"),
    ]
    if state_key:
        transforms.append(get_column_normalizer(ds, state_key, state_key))
    ds.transform = spt.data.transforms.Compose(*transforms)

    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(ds), generator=g)[:n_sequences].tolist()
    samples = [ds[i] for i in indices]

    batch: Dict[str, torch.Tensor] = {
        "pixels": torch.stack([s["pixels"] for s in samples]).to(device),
        "action": torch.nan_to_num(torch.stack([s["action"] for s in samples]), 0.0).to(device),
    }
    if state_key:
        batch["state"] = torch.stack([s[state_key] for s in samples]).to(device)
    return batch


@torch.no_grad()
def encode_sequences(model, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    info = model.encode({"pixels": batch["pixels"], "action": batch["action"]})
    emb = info["emb"]
    emb_raw = info.get("emb_raw", emb)
    act_emb = info["act_emb"]
    outputs = {
        "emb": emb,
        "emb_raw": emb_raw,
        "act_emb": act_emb,
        "pixels": batch["pixels"],
        "action": batch["action"],
    }
    if "state" in batch:
        outputs["state"] = batch["state"]
    return outputs


def exclude_diagonal(mat: torch.Tensor) -> torch.Tensor:
    n = mat.size(0)
    mask = ~torch.eye(n, dtype=torch.bool, device=mat.device)
    return mat[mask]


def pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float().reshape(-1)
    y = y.float().reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if float(denom) < 1e-12:
        return 0.0
    return float((x @ y) / denom)


def spearman_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    def rankdata(v: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(v, stable=True)
        sorted_v = v[order]
        ranks = torch.empty(order.numel(), dtype=torch.float32, device=v.device)
        start = 0
        n = order.numel()
        while start < n:
            end = start + 1
            while end < n and sorted_v[end] == sorted_v[start]:
                end += 1
            avg_rank = 0.5 * float(start + end - 1)
            ranks[order[start:end]] = avg_rank
            start = end
        return ranks

    x = x.float().reshape(-1)
    y = y.float().reshape(-1)
    return pearson_corr(rankdata(x), rankdata(y))


def effective_rank(z: torch.Tensor) -> float:
    """Entropy effective rank of centered features.

    We use the RankMe / Roy-Vetterli convention: normalize singular values
    directly, then exponentiate their entropy. This estimates how many
    independent latent directions carry variance.
    """
    z = z - z.mean(0, keepdim=True)
    singular_values = torch.linalg.svdvals(z)
    total = singular_values.sum()
    if float(total) < 1e-12:
        return 0.0
    p = singular_values / total
    entropy = -(p * torch.log(p.clamp_min(1e-12))).sum()
    return float(torch.exp(entropy))


def flatten_sequence_tensor(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1, x.size(-1)).cpu()


def flattened_sequence_ids(x: torch.Tensor) -> torch.Tensor:
    b, t = x.shape[:2]
    return torch.arange(b).repeat_interleave(t)


def sample_flat_points(
    z: torch.Tensor,
    ref: torch.Tensor,
    *,
    max_points: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat_z = flatten_sequence_tensor(z)
    flat_ref = flatten_sequence_tensor(ref)
    seq_ids = flattened_sequence_ids(ref)

    n_total = flat_z.shape[0]
    n_keep = min(max_points, n_total)
    if n_keep == n_total:
        return flat_z, flat_ref, seq_ids

    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_total, generator=g)[:n_keep]
    return flat_z[indices], flat_ref[indices], seq_ids[indices]


def pca_projection(z: torch.Tensor, out_dim: int = 2) -> torch.Tensor:
    z = z - z.mean(0, keepdim=True)
    q = min(max(out_dim, 2), z.shape[1], z.shape[0] - 1)
    u, s, _ = torch.pca_lowrank(z, q=q)
    return u[:, :out_dim] * s[:out_dim]


def tsne_projection(
    z: torch.Tensor,
    *,
    out_dim: int = 2,
    perplexity: float = 30.0,
    random_state: int = 3072,
    pca_dim: int = 32,
) -> torch.Tensor:
    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise ImportError(
            "t-SNE requires scikit-learn. Install it with `pip install scikit-learn`."
        ) from exc

    z = z.cpu()
    n = z.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 points for t-SNE projection.")

    max_perplexity = max(1.0, float((n - 1) // 3))
    perplexity = min(perplexity, max_perplexity)
    if pca_dim and z.shape[1] > pca_dim:
        z = pca_projection(z, out_dim=min(pca_dim, z.shape[1]))

    tsne = TSNE(
        n_components=out_dim,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )
    proj = tsne.fit_transform(z.numpy())
    return torch.from_numpy(proj)


def analyze_embedding(z: torch.Tensor) -> Dict[str, float]:
    flat = z.reshape(-1, z.size(-1)).cpu()
    norms = flat.norm(dim=-1)
    cos = F.normalize(flat, dim=-1) @ F.normalize(flat, dim=-1).T
    dim_std = flat.std(dim=0)
    return {
        "n_embeddings": int(flat.shape[0]),
        "dim": int(flat.shape[1]),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "dim_std_mean": float(dim_std.mean()),
        "dim_std_min": float(dim_std.min()),
        "dim_std_max": float(dim_std.max()),
        "inter_sample_cosine_mean": float(exclude_diagonal(cos).mean()),
        "inter_sample_cosine_std": float(exclude_diagonal(cos).std()),
        "effective_rank": effective_rank(flat),
    }


def _predict_last_step(
    model,
    *,
    ctx_emb: torch.Tensor,
    ctx_act_emb: torch.Tensor,
    analysis_space: str,
) -> torch.Tensor:
    if hasattr(model, "predict_raw"):
        pred_raw = model.predict_raw(ctx_emb, ctx_act_emb)[:, -1]
    else:
        pred_raw = model.predict(ctx_emb, ctx_act_emb)[:, -1]
    if analysis_space == "raw":
        return pred_raw
    if hasattr(model, "normalize_embeddings"):
        return model.normalize_embeddings(pred_raw)
    return F.normalize(pred_raw, dim=-1, eps=1e-8)


@torch.no_grad()
def collect_prediction_windows(
    model,
    outputs: Dict[str, torch.Tensor],
    *,
    history_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    spaces = get_model_spaces(model)
    context = get_embedding_space(outputs, spaces["training_context_space"])
    target = get_embedding_space(outputs, spaces["analysis_prediction_space"])
    act_emb = outputs["act_emb"]
    total_steps = context.size(1)

    preds: List[torch.Tensor] = []
    tgts: List[torch.Tensor] = []
    for end_idx in range(history_size, total_steps):
        pred = _predict_last_step(
            model,
            ctx_emb=context[:, end_idx - history_size : end_idx],
            ctx_act_emb=act_emb[:, end_idx - history_size : end_idx],
            analysis_space=spaces["analysis_prediction_space"],
        )
        preds.append(pred)
        tgts.append(target[:, end_idx])

    return torch.stack(preds, dim=1), torch.stack(tgts, dim=1)


def analyze_prediction(pred: torch.Tensor, tgt: torch.Tensor) -> Dict[str, float]:
    pred_err = (pred - tgt).norm(dim=-1).reshape(-1).cpu()
    pred_flat = pred.reshape(-1, pred.size(-1))
    tgt_flat = tgt.reshape(-1, tgt.size(-1))
    pred_cos = F.cosine_similarity(pred_flat, tgt_flat, dim=-1).cpu()
    pred_cos_dist = (1.0 - pred_cos).cpu()

    return {
        "n_prediction_windows": int(pred.size(0) * pred.size(1)),
        "prediction_horizon": int(pred.size(1)),
        "pred_error_mean": float(pred_err.mean()),
        "pred_error_std": float(pred_err.std()),
        "pred_target_cosine_mean": float(pred_cos.mean()),
        "pred_target_cosine_std": float(pred_cos.std()),
        "pred_target_cosine_distance_mean": float(pred_cos_dist.mean()),
    }


@torch.no_grad()
def collect_autoregressive_rollout(
    model,
    outputs: Dict[str, torch.Tensor],
    *,
    history_size: int,
    future_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    spaces = get_model_spaces(model)
    pixels = outputs["pixels"][:, :history_size].unsqueeze(1)
    action_sequence = outputs["action"][:, : history_size + future_steps - 1].unsqueeze(1)
    rollout_info = model.rollout({"pixels": pixels}, action_sequence, history_size=history_size)

    if spaces["analysis_prediction_space"] == "raw" and "predicted_emb_raw" in rollout_info:
        predicted = rollout_info["predicted_emb_raw"][:, 0, history_size : history_size + future_steps]
    else:
        predicted = rollout_info["predicted_emb"][:, 0, history_size : history_size + future_steps]

    target = get_embedding_space(outputs, spaces["analysis_prediction_space"])[:, history_size : history_size + future_steps]
    return predicted, target


def analyze_rollout(pred: torch.Tensor, tgt: torch.Tensor) -> Dict[str, float]:
    step_err = (pred - tgt).norm(dim=-1).cpu()
    step_cos = 1.0 - F.cosine_similarity(pred, tgt, dim=-1).cpu()

    first_error = step_err[:, 0].mean().clamp_min(1e-8)
    last_error = step_err[:, -1].mean()

    return {
        "rollout_horizon": int(pred.size(1)),
        "rollout_error_mean": float(step_err.mean()),
        "rollout_error_std": float(step_err.std()),
        "rollout_error_last_mean": float(last_error),
        "rollout_cosine_distance_mean": float(step_cos.mean()),
        "rollout_cosine_distance_last_mean": float(step_cos[:, -1].mean()),
        "rollout_error_growth": float(last_error / first_error),
    }


def sample_random_future_actions(
    future_action: torch.Tensor,
    *,
    n_trials: int,
    seed: int,
) -> torch.Tensor:
    b, horizon, act_dim = future_action.shape
    pool = future_action.reshape(-1, act_dim).cpu()
    if pool.size(0) == 0:
        raise ValueError("Need at least one future action to sample random candidates.")
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, pool.size(0), (b, n_trials, horizon), generator=g)
    sampled = pool[idx]
    return sampled.to(future_action.device)


@torch.no_grad()
def analyze_planning_signal(
    model,
    outputs: Dict[str, torch.Tensor],
    *,
    history_size: int,
    future_steps: int,
    random_action_trials: int,
    seed: int,
) -> Dict[str, float]:
    if future_steps < 2:
        raise ValueError("planning signal probe requires future_steps >= 2.")

    pixels = outputs["pixels"]
    action = outputs["action"]
    b = pixels.size(0)

    context_pixels = pixels[:, :history_size].unsqueeze(1)
    context_action = action[:, :history_size].unsqueeze(1)
    goal = pixels[:, -1:].unsqueeze(1)

    future_action_steps = future_steps - 1
    expert_future = action[:, history_size : history_size + future_action_steps]
    expert_candidate = action[:, : history_size + future_action_steps].unsqueeze(1)

    random_future = sample_random_future_actions(
        expert_future,
        n_trials=random_action_trials,
        seed=seed,
    )
    history = action[:, :history_size].unsqueeze(1).expand(b, random_action_trials, history_size, -1)
    random_candidates = torch.cat([history, random_future], dim=2)
    action_candidates = torch.cat([expert_candidate, random_candidates], dim=1)

    info_dict = {
        "pixels": context_pixels,
        "action": context_action,
        "goal": goal,
    }
    costs = model.get_cost(info_dict, action_candidates).detach().cpu()
    expert_cost = costs[:, 0]
    random_cost = costs[:, 1:]
    best_random = random_cost.min(dim=1).values
    margin = random_cost.mean(dim=1) - expert_cost

    return {
        "planning_horizon": int(future_steps),
        "random_action_trials": int(random_action_trials),
        "expert_cost_mean": float(expert_cost.mean()),
        "expert_cost_std": float(expert_cost.std()),
        "random_cost_mean": float(random_cost.mean()),
        "random_cost_std": float(random_cost.std()),
        "best_random_cost_mean": float(best_random.mean()),
        "cost_margin_mean": float(margin.mean()),
        "cost_margin_std": float(margin.std()),
        "expert_beats_random_rate": float((expert_cost[:, None] < random_cost).float().mean()),
        "expert_beats_best_random_rate": float((expert_cost < best_random).float().mean()),
    }


@torch.no_grad()
def analyze_action_effect(
    model,
    outputs: Dict[str, torch.Tensor],
    *,
    n_trials: int,
    interp_steps: int,
    perturb_scale: float,
) -> Dict[str, float]:
    ctx_len = infer_history_size(model)
    spaces = get_model_spaces(model)
    rollout_space = spaces["inference_rollout_state_space"]
    ctx_emb = get_embedding_space(outputs, rollout_space)[:, :ctx_len]
    action = outputs["action"]
    base_action = action[:, :ctx_len].clone()
    base_pred = _predict_last_step(
        model,
        ctx_emb=ctx_emb,
        ctx_act_emb=model.action_encoder(base_action),
        analysis_space=rollout_space,
    )

    action_std = action.reshape(-1, action.size(-1)).std(dim=0).clamp_min(1e-6)

    perturb_norms = []
    pred_shift_norms = []
    for _ in range(n_trials):
        delta = torch.randn_like(base_action[:, -1]) * action_std * perturb_scale
        perturbed = base_action.clone()
        perturbed[:, -1] = perturbed[:, -1] + delta
        pred = _predict_last_step(
            model,
            ctx_emb=ctx_emb,
            ctx_act_emb=model.action_encoder(perturbed),
            analysis_space=rollout_space,
        )

        perturb_norms.append(delta.norm(dim=-1))
        pred_shift_norms.append((pred - base_pred).norm(dim=-1))

    perturb_norms = torch.cat(perturb_norms).cpu()
    pred_shift_norms = torch.cat(pred_shift_norms).cpu()

    n_interp_anchors = min(8, base_action.size(0))
    anchor_idx = torch.linspace(0, base_action.size(0) - 1, steps=n_interp_anchors).long().unique()
    alphas = torch.linspace(0, 1, interp_steps, device=base_action.device)

    endpoint_shifts = []
    monotonicities = []
    for idx in anchor_idx.tolist():
        action_a = base_action[idx : idx + 1].clone()
        action_b = base_action[idx : idx + 1].clone()
        action_b[:, -1] = action_b[:, -1] + action_std.unsqueeze(0) * perturb_scale

        interp_preds: List[torch.Tensor] = []
        single_ctx = ctx_emb[idx : idx + 1]
        for alpha in alphas:
            act = (1 - alpha) * action_a + alpha * action_b
            pred = _predict_last_step(
                model,
                ctx_emb=single_ctx,
                ctx_act_emb=model.action_encoder(act),
                analysis_space=rollout_space,
            )
            interp_preds.append(pred.squeeze(0))

        interp_preds = torch.stack(interp_preds)
        interp_dist = (interp_preds - interp_preds[0]).norm(dim=-1).cpu()
        endpoint_shifts.append(interp_dist[-1])
        monotonicities.append(((interp_dist[1:] - interp_dist[:-1]) >= 0).float().mean())

    endpoint_shifts = torch.stack(endpoint_shifts)
    monotonicities = torch.stack(monotonicities)

    return {
        "n_trials": int(n_trials),
        "n_action_pairs": int(perturb_norms.numel()),
        "n_interp_anchors": int(anchor_idx.numel()),
        "perturb_scale": float(perturb_scale),
        "mean_action_perturb_norm": float(perturb_norms.mean()),
        "mean_pred_shift_norm": float(pred_shift_norms.mean()),
        "action_perturb_pred_shift_corr": pearson_corr(perturb_norms, pred_shift_norms),
        "interpolation_endpoint_shift": float(endpoint_shifts.mean()),
        "interpolation_endpoint_shift_std": float(endpoint_shifts.std()),
        "interpolation_monotonicity": float(monotonicities.mean()),
        "interpolation_monotonicity_std": float(monotonicities.std()),
    }


def knn_overlap(
    a_dist: torch.Tensor,
    b_dist: torch.Tensor,
    k: int,
    *,
    invalid_mask: torch.Tensor | None = None,
) -> float:
    n = a_dist.size(0)
    a = a_dist.clone()
    b = b_dist.clone()
    eye = torch.eye(n, dtype=torch.bool, device=a.device)
    a[eye] = float("inf")
    b[eye] = float("inf")
    if invalid_mask is not None:
        a[invalid_mask] = float("inf")
        b[invalid_mask] = float("inf")
    a_idx = torch.topk(a, k=k, largest=False).indices
    b_idx = torch.topk(b, k=k, largest=False).indices
    overlaps = []
    for i in range(n):
        overlaps.append(len(set(a_idx[i].tolist()) & set(b_idx[i].tolist())) / float(k))
    return float(sum(overlaps) / len(overlaps))


def analyze_reference_probe(
    z: torch.Tensor,
    reference_state: torch.Tensor,
    *,
    reference_state_key: str,
    k: int,
    max_points: int,
    seed: int,
) -> Dict[str, float]:
    flat_z, flat_ref, seq_ids = sample_flat_points(z, reference_state, max_points=max_points, seed=seed)
    n = flat_z.shape[0]
    ref_norm = (flat_ref - flat_ref.mean(0, keepdim=True)) / flat_ref.std(0, keepdim=True).clamp_min(1e-6)
    z_dist = torch.cdist(flat_z, flat_z)
    ref_dist = torch.cdist(ref_norm, ref_norm)

    offdiag = ~torch.eye(n, dtype=torch.bool)
    cross_seq = offdiag & (seq_ids[:, None] != seq_ids[None, :])
    invalid_same_seq = ~cross_seq & ~torch.eye(n, dtype=torch.bool)

    latent_step = (z[:, 1:] - z[:, :-1]).norm(dim=-1).reshape(-1).cpu()
    ref_step = (reference_state[:, 1:] - reference_state[:, :-1]).norm(dim=-1).reshape(-1).cpu()

    result = {
        "reference_state_key": reference_state_key,
        "n_points": int(n),
        "k": int(k),
        "latent_reference_step_corr": pearson_corr(latent_step, ref_step),
    }
    if bool(cross_seq.any()):
        result["distance_rank_corr_cross_seq"] = spearman_corr(z_dist[cross_seq], ref_dist[cross_seq])
        result["knn_overlap_cross_seq"] = knn_overlap(
            z_dist,
            ref_dist,
            k=min(k, n - 1),
            invalid_mask=invalid_same_seq,
        )
    return result


def build_local_neighbor_report(
    z: torch.Tensor,
    reference_state: torch.Tensor,
    *,
    k: int,
    n_anchors: int,
) -> List[Dict[str, Any]]:
    flat_z = z.reshape(-1, z.size(-1)).cpu()
    flat_ref = reference_state.reshape(-1, reference_state.size(-1)).cpu()
    n = flat_z.shape[0]
    seq_len = reference_state.size(1)

    z_dist = torch.cdist(flat_z, flat_z)
    ref_norm = (flat_ref - flat_ref.mean(0, keepdim=True)) / flat_ref.std(0, keepdim=True).clamp_min(1e-6)
    ref_dist = torch.cdist(ref_norm, ref_norm)

    eye = torch.eye(n, dtype=torch.bool)
    z_dist[eye] = float("inf")
    ref_dist[eye] = float("inf")

    anchors = torch.linspace(0, n - 1, steps=min(n_anchors, n)).long().unique()
    report: List[Dict[str, Any]] = []
    for anchor in anchors.tolist():
        z_nn = torch.topk(z_dist[anchor], k=min(k, n - 1), largest=False).indices.tolist()
        ref_nn = torch.topk(ref_dist[anchor], k=min(k, n - 1), largest=False).indices.tolist()
        item = {
            "anchor_index": anchor,
            "anchor_seq_id": anchor // seq_len,
            "anchor_time_id": anchor % seq_len,
            "anchor_reference_state": flat_ref[anchor].tolist(),
            "overlap": len(set(z_nn) & set(ref_nn)) / float(min(k, n - 1)),
            "latent_nn": [],
            "reference_nn": [],
        }
        for idx in z_nn:
            item["latent_nn"].append({
                "index": idx,
                "seq_id": idx // seq_len,
                "time_id": idx % seq_len,
                "latent_distance": float(z_dist[anchor, idx]),
                "reference_distance": float(ref_dist[anchor, idx]),
                "reference_state": flat_ref[idx].tolist(),
            })
        for idx in ref_nn:
            item["reference_nn"].append({
                "index": idx,
                "seq_id": idx // seq_len,
                "time_id": idx % seq_len,
                "latent_distance": float(z_dist[anchor, idx]),
                "reference_distance": float(ref_dist[anchor, idx]),
                "reference_state": flat_ref[idx].tolist(),
            })
        report.append(item)
    return report


def make_interpretation(dataset: str, result: Dict[str, Dict[str, float]]) -> List[str]:
    dataset_l = dataset.lower()
    emb = result["embedding"]
    pred = result["prediction"]
    rollout = result["rollout"]
    planning = result.get("planning")
    act = result["action_effect"]

    hints: List[str] = []

    if emb["effective_rank"] < 4:
        hints.append("Embedding rank is very low; the representation may still be partially collapsed.")
    elif emb["effective_rank"] < 12:
        hints.append("Embedding rank is moderate; anti-collapse works, but capacity usage may still be limited.")
    else:
        hints.append("Embedding rank looks healthy; remaining issues likely come from rollout or planning signal rather than collapse.")

    if pred["pred_target_cosine_distance_mean"] > 0.05:
        hints.append("One-step prediction is still weak; improving rollout or planning before fixing this is premature.")
    else:
        hints.append("One-step prediction is reasonably accurate; if eval is weak, multi-step drift or cost quality is the next place to look.")

    if rollout["rollout_error_growth"] > 2.0:
        hints.append("Autoregressive rollout error grows quickly with horizon; compounding error is a likely bottleneck.")
    elif rollout["rollout_cosine_distance_last_mean"] < 0.05:
        hints.append("Autoregressive rollout stays fairly consistent over the tested horizon.")

    if not isinstance(planning, Mapping) or not planning:
        hints.append("Planning probe is unavailable for this run; interpret rollout quality separately from planner quality.")
    elif "error" in planning:
        hints.append("Planning probe failed at runtime; inspect the stored planning.error field before drawing planner conclusions.")
    elif planning["cost_margin_mean"] <= 0.0:
        hints.append("The model's cost does not prefer expert futures over random futures; planner guidance is likely too weak.")
    elif planning["expert_beats_best_random_rate"] < 0.4:
        hints.append("Expert futures beat the random mean but not the best random candidates often enough; planning signal is fragile.")
    else:
        hints.append("Expert futures usually rank better than random ones under the model cost; planning signal looks usable.")

    if act["action_perturb_pred_shift_corr"] < 0.2:
        hints.append("Action perturbations do not produce structured latent branching; action usage may be weak.")
    elif act["mean_pred_shift_norm"] < 0.2:
        hints.append("Action perturbations change predictions only weakly; latent dynamics may be too insensitive even if correlated.")

    ref = result.get("reference_probe")
    if isinstance(ref, Mapping) and ref:
        rank_corr = ref.get("distance_rank_corr_cross_seq")
        if rank_corr is not None and rank_corr < 0.1:
            hints.append("The optional external state proxy disagrees strongly with latent geometry. Treat this as a probe, not a verdict, unless that proxy is known to be task-correct.")

    if "pusht" in dataset_l:
        hints.append("For PushT, prioritize cost separation and rollout drift over raw state-geometry matching unless the reference state is explicitly task-curated.")
    elif "tworoom" in dataset_l:
        hints.append("For TwoRoom, watch whether long-horizon rollout stays coherent across door transitions instead of drifting across disconnected regions.")
    elif "cube" in dataset_l or "ogb" in dataset_l:
        hints.append("For Cube, check whether the planner signal separates contact-consistent futures from random action futures.")
    elif "reacher" in dataset_l or "dmc" in dataset_l:
        hints.append("For Reacher, focus on whether rollout remains smooth and goal-directed over horizon rather than matching a proxy state exactly.")

    return hints


def to_serializable(obj):
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    return obj


def metric_spec(section: str, key: str) -> Dict[str, str]:
    return METRIC_SPECS.get(section, {}).get(key, {})


def metric_entries(result: Dict[str, Dict[str, Any]]) -> List[Tuple[str, str, Any]]:
    entries: List[Tuple[str, str, Any]] = []
    for section in SECTION_ORDER:
        metrics = result.get(section)
        if not isinstance(metrics, dict):
            continue
        preferred_keys = list(METRIC_SPECS.get(section, {}).keys())
        extra_keys = [key for key in metrics if key not in preferred_keys]
        for key in preferred_keys + extra_keys:
            if key in metrics:
                entries.append((section, key, metrics[key]))
    return entries


def format_metric_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def format_section(title: str, metrics: Dict[str, Any]) -> str:
    lines = [f"\n{'=' * 72}", title, f"{'=' * 72}"]
    for key, value in metrics.items():
        lines.append(f"{key}: {format_metric_value(value)}")
    return "\n".join(lines)


def format_hints(hints: List[str]) -> str:
    lines = [f"\n{'=' * 72}", "Interpretation", f"{'=' * 72}"]
    for hint in hints:
        lines.append(f"- {hint}")
    return "\n".join(lines)


def format_analysis_report(result: Dict[str, Dict[str, Any]]) -> str:
    chunks: List[str] = []
    for section in SECTION_ORDER:
        metrics = result.get(section)
        if isinstance(metrics, dict) and metrics:
            chunks.append(format_section(SECTION_TITLES.get(section, section.title()), metrics))
    hints = result.get("interpretation")
    if isinstance(hints, list):
        chunks.append(format_hints(hints))
    return "\n".join(chunks).lstrip()


def slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip()).strip("._-")
    return slug or "model"


def make_unique_slug(label: str, used: set[str]) -> str:
    base = slugify(label)
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def parse_model_spec(spec: str | Sequence[str]) -> Tuple[str, str]:
    if isinstance(spec, str):
        if "=" in spec:
            label, ckpt = spec.split("=", 1)
            label = label.strip()
            ckpt = ckpt.strip()
            if not label or not ckpt:
                raise ValueError(f"Invalid model spec: {spec}")
            return label, ckpt
        ckpt = spec.strip()
        if not ckpt:
            raise ValueError("Empty model spec.")
        return Path(ckpt).stem, ckpt

    if len(spec) != 2:
        raise ValueError(f"Expected (label, ckpt) pair, got: {spec}")
    label, ckpt = str(spec[0]).strip(), str(spec[1]).strip()
    if not label or not ckpt:
        raise ValueError(f"Invalid model spec: {spec}")
    return label, ckpt


def parse_compare_pair(spec: str) -> Tuple[str, str]:
    if "=" in spec:
        left, right = spec.split("=", 1)
    elif "," in spec:
        left, right = spec.split(",", 1)
    else:
        raise ValueError(f"Invalid compare pair spec: {spec}")

    left = left.strip()
    right = right.strip()
    if not left or not right:
        raise ValueError(f"Invalid compare pair spec: {spec}")
    return left, right


def metric_table_entries(result: Dict[str, Dict[str, Any]]) -> List[Tuple[str, str, Any]]:
    return [
        (section, key, value)
        for section, key, value in metric_entries(result)
        if section not in NON_TABLE_SECTIONS
    ]


def records_to_metrics_wide(records: Sequence[Mapping[str, Any]]):
    pd = _require_pandas()
    rows: List[Dict[str, Any]] = []
    for record in records:
        row: Dict[str, Any] = {
            "model_label": record["label"],
            "model_slug": record.get("slug", slugify(record["label"])),
            "analysis_dir": record.get("analysis_dir", ""),
            "ckpt": record["result"]["meta"]["ckpt"],
        }
        for section, key, value in metric_table_entries(record["result"]):
            row[f"{section}.{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _apply_model_order(df, model_order: Sequence[str] | None):
    if not model_order:
        return df
    pd = _require_pandas()
    order_map = {label: idx for idx, label in enumerate(model_order)}
    tmp = df.copy()
    tmp["_model_order"] = tmp["model_label"].map(order_map).fillna(len(order_map))
    tmp = tmp.sort_values(["_model_order", "model_label"]).drop(columns="_model_order")
    return pd.DataFrame(tmp)


def _key_metric_specs(metrics: Sequence[Mapping[str, str]] | None) -> List[Mapping[str, str]]:
    return list(metrics or DEFAULT_KEY_METRICS)


def _wide_table_from_df(
    wide,
    *,
    metrics: Sequence[Mapping[str, str]] | None = None,
    eval_scores: Mapping[str, float] | None = None,
    model_order: Sequence[str] | None = None,
    include_analysis_dir: bool = False,
):
    pd = _require_pandas()
    specs = _key_metric_specs(metrics)
    table = wide.copy()

    if eval_scores is not None:
        table["eval_score"] = table["model_label"].map(eval_scores)

    columns = ["model_label"]
    rename_map: Dict[str, str] = {}
    if include_analysis_dir and "analysis_dir" in table.columns:
        columns.append("analysis_dir")
    if "eval_score" in table.columns:
        columns.append("eval_score")

    for spec in specs:
        metric_name = spec["metric_name"]
        if metric_name not in table.columns:
            table[metric_name] = float("nan")
        columns.append(metric_name)
        rename_map[metric_name] = spec["label"]

    out = table[columns].rename(columns=rename_map)
    out = _apply_model_order(out, model_order)
    return pd.DataFrame(out)


def _long_table_from_df(
    wide,
    *,
    metrics: Sequence[Mapping[str, str]] | None = None,
    eval_scores: Mapping[str, float] | None = None,
    model_order: Sequence[str] | None = None,
):
    pd = _require_pandas()
    specs = _key_metric_specs(metrics)
    table = _wide_table_from_df(
        wide,
        metrics=specs,
        eval_scores=eval_scores,
        model_order=model_order,
        include_analysis_dir=True,
    )

    rows: List[Dict[str, Any]] = []
    for spec in specs:
        label = spec["label"]
        for _, row in table.iterrows():
            rows.append(
                {
                    "model_label": row["model_label"],
                    "analysis_dir": row.get("analysis_dir", ""),
                    "group": spec["group"],
                    "metric": label,
                    "metric_name": spec["metric_name"],
                    "better": spec["better"],
                    "value": row.get(label),
                    "eval_score": row.get("eval_score"),
                }
            )
    return pd.DataFrame(rows)


def build_key_metric_table_from_records(
    records: Sequence[Mapping[str, Any]],
    *,
    metrics: Sequence[Mapping[str, str]] | None = None,
    eval_scores: Mapping[str, float] | None = None,
    model_order: Sequence[str] | None = None,
    include_analysis_dir: bool = False,
):
    wide = records_to_metrics_wide(records)
    return _wide_table_from_df(
        wide,
        metrics=metrics,
        eval_scores=eval_scores,
        model_order=model_order,
        include_analysis_dir=include_analysis_dir,
    )


def build_key_metric_long_table_from_records(
    records: Sequence[Mapping[str, Any]],
    *,
    metrics: Sequence[Mapping[str, str]] | None = None,
    eval_scores: Mapping[str, float] | None = None,
    model_order: Sequence[str] | None = None,
):
    wide = records_to_metrics_wide(records)
    return _long_table_from_df(
        wide,
        metrics=metrics,
        eval_scores=eval_scores,
        model_order=model_order,
    )


def build_metric_reference_table(
    metrics: Sequence[Mapping[str, str]] | None = None,
):
    pd = _require_pandas()
    rows = []
    for spec in _key_metric_specs(metrics):
        rows.append(
            {
                "group": spec["group"],
                "metric": spec["label"],
                "short_label": spec.get("short_label", spec["label"]),
                "better": _metric_goal_text(spec.get("better", "higher")),
                "summary": spec.get("summary", ""),
                "diagnosis": spec.get("diagnosis", ""),
            }
        )
    return pd.DataFrame(rows)


def save_key_metric_tables(
    wide_table,
    long_table,
    *,
    output_dir: str | Path,
    stem: str = "key_metrics",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wide_csv = output_dir / f"{stem}_wide.csv"
    long_csv = output_dir / f"{stem}_long.csv"
    html_path = output_dir / f"{stem}_wide.html"

    wide_table.to_csv(wide_csv, index=False)
    long_table.to_csv(long_csv, index=False)
    html_path.write_text(wide_table.to_html(index=False))
    return {
        "wide_csv": wide_csv,
        "long_csv": long_csv,
        "wide_html": html_path,
    }


def save_metric_reference_table(
    table,
    *,
    output_dir: str | Path,
    stem: str = "key_metric_reference",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{stem}.csv"
    html_path = output_dir / f"{stem}.html"
    table.to_csv(csv_path, index=False)
    html_path.write_text(table.to_html(index=False))
    return {
        "csv": csv_path,
        "html": html_path,
    }


def _normalize_metric(values, better: str):
    clean = [float(v) for v in values if v == v]
    if not clean:
        return [float("nan")] * len(values)

    if better == "lower_abs":
        transformed = [abs(float(v)) if v == v else float("nan") for v in values]
        clean = [v for v in transformed if v == v]
        lo, hi = min(clean), max(clean)
        if math.isclose(lo, hi):
            return [0.5 if v == v else float("nan") for v in transformed]
        return [1.0 - ((v - lo) / (hi - lo)) if v == v else float("nan") for v in transformed]

    lo, hi = min(clean), max(clean)
    if math.isclose(lo, hi):
        return [0.5 if v == v else float("nan") for v in values]

    if better == "lower":
        return [1.0 - ((float(v) - lo) / (hi - lo)) if v == v else float("nan") for v in values]

    return [((float(v) - lo) / (hi - lo)) if v == v else float("nan") for v in values]


def _metric_arrow(better: str) -> str:
    return "↓" if better in {"lower", "lower_abs"} else "↑"


def _metric_goal_text(better: str) -> str:
    if better == "lower":
        return "lower is better"
    if better == "lower_abs":
        return "closer to 0 is better"
    return "higher is better"


def _metric_title(spec: Mapping[str, str]) -> str:
    short_label = spec.get("short_label", spec["label"])
    brief = spec.get("brief", spec.get("summary", ""))
    return (
        f"{spec['group']} | {short_label} {_metric_arrow(spec.get('better', 'higher'))}\n"
        f"{brief}"
    )


def _metric_note_lines(specs: Sequence[Mapping[str, str]], *, notes_per_line: int = 2) -> List[str]:
    entries = [
        (
            f"{spec.get('short_label', spec['label'])} {_metric_arrow(spec.get('better', 'higher'))}: "
            f"{spec.get('summary', '')} "
            f"{spec.get('diagnosis', '')}"
        ).strip()
        for spec in specs
    ]
    lines: List[str] = []
    for idx in range(0, len(entries), notes_per_line):
        lines.append("  |  ".join(entries[idx : idx + notes_per_line]))
    return lines


def _apply_metric_notes(fig, specs: Sequence[Mapping[str, str]]):
    lines = _metric_note_lines(specs)
    if not lines:
        return
    bottom_margin = min(0.38, 0.08 + 0.045 * len(lines))
    fig.tight_layout(rect=[0, bottom_margin, 1, 1])
    fig.text(
        0.01,
        0.01,
        "\n".join(lines),
        ha="left",
        va="bottom",
        fontsize=8.5,
        family="monospace",
    )


def plot_metric_bars(
    table,
    *,
    metrics: Sequence[Mapping[str, str]] | None = None,
    output: str | Path | None = None,
    ncols: int = 3,
    annotate: bool = True,
    figsize_scale: float = 4.0,
):
    plt = _require_matplotlib()
    specs = [spec for spec in _key_metric_specs(metrics) if spec["label"] in table.columns]
    if not specs:
        raise ValueError("No requested metrics are present in the table.")
    nplots = len(specs)
    ncols = max(1, min(ncols, nplots))
    nrows = (nplots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_scale * ncols, 3.8 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    x_labels = table["model_label"].tolist()
    for ax, spec in zip(axes_flat, specs):
        col = spec["label"]
        values = table[col].tolist()
        bars = ax.bar(x_labels, values, color="#4C78A8")
        ax.set_title(_metric_title(spec))
        ax.tick_params(axis="x", rotation=30)
        if annotate:
            for bar, value in zip(bars, values):
                if value == value:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        bar.get_height(),
                        f"{value:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )

    for ax in axes_flat[nplots:]:
        ax.axis("off")

    _apply_metric_notes(fig, specs)
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200, bbox_inches="tight")
    return fig


def plot_metric_heatmap(
    table,
    *,
    metrics: Sequence[Mapping[str, str]] | None = None,
    output: str | Path | None = None,
    annotate: bool = True,
):
    plt = _require_matplotlib()
    specs = [spec for spec in _key_metric_specs(metrics) if spec["label"] in table.columns]
    if not specs:
        raise ValueError("No requested metrics are present in the table.")
    metric_labels = [
        f"{spec.get('short_label', spec['label'])}\n{_metric_arrow(spec.get('better', 'higher'))}"
        for spec in specs
    ]
    normalized = [
        _normalize_metric(table[spec["label"]].tolist(), spec.get("better", "higher"))
        for spec in specs
    ]

    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(metric_labels)), max(3, 0.7 * len(table))))
    im = ax.imshow(list(zip(*normalized)), aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(metric_labels)))
    ax.set_xticklabels(metric_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(table)))
    ax.set_yticklabels(table["model_label"].tolist())
    ax.set_title("Direction-aware metric heatmap")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    if annotate:
        for row_idx in range(len(table)):
            for col_idx, spec in enumerate(specs):
                value = table.iloc[row_idx][spec["label"]]
                if value == value:
                    ax.text(col_idx, row_idx, f"{value:.3f}", ha="center", va="center", color="white", fontsize=8)

    _apply_metric_notes(fig, specs)
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200, bbox_inches="tight")
    return fig


def _projection_rows_from_tensor(proj: torch.Tensor, reference_state: torch.Tensor | None, *, seq_len: int) -> List[Dict[str, float]]:
    flat_ref = reference_state.reshape(-1, reference_state.size(-1)).cpu() if reference_state is not None else None
    rows: List[Dict[str, float]] = []
    for idx in range(proj.size(0)):
        row: Dict[str, float] = {
            "seq_id": float(idx // seq_len),
            "time_id": float(idx % seq_len),
            "x": float(proj[idx, 0]),
            "y": float(proj[idx, 1]),
        }
        if flat_ref is not None:
            for dim in range(flat_ref.size(1)):
                row[f"state_{dim}"] = float(flat_ref[idx, dim])
        rows.append(row)
    return rows


def _projection_rows_from_outputs(
    outputs: Mapping[str, Any],
    *,
    projection: str,
    tsne_perplexity: float = 30.0,
    seed: int = 3072,
) -> List[Dict[str, float]]:
    flat_z = outputs["emb"].reshape(-1, outputs["emb"].size(-1)).cpu()
    if projection == "pca":
        proj = pca_projection(flat_z, out_dim=2)
    elif projection == "tsne":
        proj = tsne_projection(
            flat_z,
            out_dim=2,
            perplexity=tsne_perplexity,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown projection type: {projection}")
    return _projection_rows_from_tensor(
        proj,
        outputs.get("state"),
        seq_len=outputs["emb"].size(1),
    )


def _resolve_projection_color_key(rows: List[Mapping[str, float]], dim: int) -> str:
    preferred = f"state_{dim}"
    if preferred in rows[0]:
        return preferred
    fallback = "time_id" if dim == 0 else "seq_id" if dim == 1 else None
    if fallback and fallback in rows[0]:
        return fallback
    raise KeyError(
        f"{preferred} not found in projection rows, and no fallback color key is available."
    )


def plot_projection_grid_from_records(
    records: Sequence[Mapping[str, Any]],
    *,
    model_labels: Sequence[str] | None = None,
    projection: str = "pca",
    color_dims: Sequence[int] = (0, 1),
    output: str | Path | None = None,
    alpha: float = 0.7,
    size: float = 6.0,
    tsne_perplexity: float = 30.0,
    seed: int = 3072,
):
    plt = _require_matplotlib()
    model_labels = list(model_labels or [record["label"] for record in records])
    if not model_labels:
        raise ValueError("No model labels available for projection plotting.")
    record_by_label = {record["label"]: record for record in records}
    rows_by_label = {
        label: _projection_rows_from_outputs(
            record_by_label[label]["outputs"],
            projection=projection,
            tsne_perplexity=tsne_perplexity,
            seed=seed,
        )
        for label in model_labels
    }

    fig, axes = plt.subplots(
        len(model_labels),
        len(color_dims),
        figsize=(6 * len(color_dims), 4 * len(model_labels)),
        squeeze=False,
    )

    for col_idx, dim in enumerate(color_dims):
        key_by_label = {
            label: _resolve_projection_color_key(rows_by_label[label], dim) for label in model_labels
        }
        values = [
            row[key_by_label[label]]
            for label in model_labels
            for row in rows_by_label[label]
        ]
        vmin, vmax = min(values), max(values)

        for row_idx, label in enumerate(model_labels):
            rows = rows_by_label[label]
            key = key_by_label[label]
            ax = axes[row_idx][col_idx]
            sc = ax.scatter(
                [row["x"] for row in rows],
                [row["y"] for row in rows],
                c=[row[key] for row in rows],
                s=size,
                alpha=alpha,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(f"{label} | {projection.upper()} | {key}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200, bbox_inches="tight")
    return fig


def run_analysis(
    *,
    ckpt: str,
    dataset: str,
    state_key: str | None,
    frameskip: int,
    img_size: int,
    n_sequences: int,
    future_steps: int,
    max_points: int,
    knn_k: int,
    action_trials: int,
    planning_random_trials: int,
    include_planning: bool,
    interp_steps: int,
    perturb_scale: float,
    seed: int,
    device: str,
    log: Callable[[str], None] | None = print,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, torch.Tensor]]:
    if future_steps < 2:
        raise ValueError("future_steps must be >= 2 so rollout and planning probes have a meaningful future horizon.")

    reference_state_key = (state_key or "").strip() or None
    if log is not None:
        log(f"[analyze_repr] loading model: {ckpt}")
    model = load_model(ckpt, device)
    history_size = infer_history_size(model)
    if log is not None:
        log(
            f"[analyze_repr] dataset={dataset} reference_state_key={reference_state_key} "
            f"history_size={history_size} future_steps={future_steps}"
        )

    batch = load_dataset_samples(
        dataset_name=dataset,
        state_key=reference_state_key,
        n_sequences=n_sequences,
        history_size=history_size,
        future_steps=future_steps,
        frameskip=frameskip,
        img_size=img_size,
        seed=seed,
        device=device,
    )
    outputs = encode_sequences(model, batch)
    spaces = get_model_spaces(model)

    pred, tgt = collect_prediction_windows(model, outputs, history_size=history_size)
    rollout_pred, rollout_tgt = collect_autoregressive_rollout(
        model,
        outputs,
        history_size=history_size,
        future_steps=future_steps,
    )

    result: Dict[str, Dict[str, Any]] = {
        "meta": {
            "ckpt": ckpt,
            "dataset": dataset,
            "reference_state_key": reference_state_key,
            "n_sequences": n_sequences,
            "history_size": history_size,
            "future_steps": future_steps,
            "device": device,
            **spaces,
        },
        "embedding": analyze_embedding(outputs["emb"]),
        "prediction": analyze_prediction(pred, tgt),
        "rollout": analyze_rollout(rollout_pred, rollout_tgt),
        "action_effect": analyze_action_effect(
            model,
            outputs,
            n_trials=action_trials,
            interp_steps=interp_steps,
            perturb_scale=perturb_scale,
        ),
    }

    if include_planning:
        try:
            result["planning"] = analyze_planning_signal(
                model,
                outputs,
                history_size=history_size,
                future_steps=future_steps,
                random_action_trials=planning_random_trials,
                seed=seed,
            )
        except Exception as exc:
            if log is not None:
                log(f"[analyze_repr] warning: planning probe failed: {exc}")
            result["planning"] = {
                "error": f"{type(exc).__name__}: {exc}",
                "planning_horizon": int(future_steps),
                "random_action_trials": int(planning_random_trials),
            }
    else:
        result["planning"] = {
            "skipped": True,
            "planning_horizon": int(future_steps),
            "random_action_trials": int(planning_random_trials),
        }

    if reference_state_key and "state" in outputs:
        result["reference_probe"] = analyze_reference_probe(
            outputs["emb"],
            outputs["state"],
            reference_state_key=reference_state_key,
            k=knn_k,
            max_points=max_points,
            seed=seed,
        )

    result["interpretation"] = make_interpretation(dataset, result)
    outputs["prediction_pred"] = pred
    outputs["prediction_tgt"] = tgt
    outputs["rollout_pred"] = rollout_pred
    outputs["rollout_tgt"] = rollout_tgt
    return result, outputs


def save_projection_rows(
    proj: torch.Tensor,
    reference_state: torch.Tensor | None,
    *,
    seq_len: int,
    save_path: Path,
):
    rows = []
    flat_ref = reference_state.reshape(-1, reference_state.size(-1)).cpu() if reference_state is not None else None

    for idx in range(proj.size(0)):
        row = {
            "seq_id": idx // seq_len,
            "time_id": idx % seq_len,
            "x": float(proj[idx, 0]),
            "y": float(proj[idx, 1]),
        }
        if reference_state is not None and flat_ref is not None:
            for dim in range(flat_ref.size(1)):
                row[f"state_{dim}"] = float(flat_ref[idx, dim])
        rows.append(row)

    with open(save_path, "w") as f:
        json.dump(rows, f, indent=2)


def save_metric_guide(save_path: Path):
    with open(save_path, "w") as f:
        json.dump(to_serializable(METRIC_SPECS), f, indent=2)


def save_outputs(
    save_dir: Path,
    result: Dict[str, Dict[str, Any]],
    z: torch.Tensor,
    reference_state: torch.Tensor | None,
    *,
    export_tsne: bool,
    tsne_perplexity: float,
    seed: int,
):
    save_dir.mkdir(parents=True, exist_ok=True)

    flat_z = z.reshape(-1, z.size(-1)).cpu()
    pca_proj = pca_projection(flat_z, out_dim=2)
    save_projection_rows(
        pca_proj,
        reference_state,
        seq_len=z.size(1),
        save_path=save_dir / "pca_projection.json",
    )

    if export_tsne:
        try:
            tsne_proj = tsne_projection(
                flat_z,
                out_dim=2,
                perplexity=tsne_perplexity,
                random_state=seed,
            )
            save_projection_rows(
                tsne_proj,
                reference_state,
                seq_len=z.size(1),
                save_path=save_dir / "tsne_projection.json",
            )
            result.setdefault("visualization", {})["tsne_exported"] = True
            result["visualization"]["tsne_perplexity"] = float(
                min(tsne_perplexity, max(1.0, float((flat_z.size(0) - 1) // 3)))
            )
        except Exception as exc:
            result.setdefault("visualization", {})["tsne_exported"] = False
            result["visualization"]["tsne_error"] = str(exc)

    if reference_state is not None:
        with open(save_dir / "local_neighbors.json", "w") as f:
            json.dump(build_local_neighbor_report(z, reference_state, k=8, n_anchors=12), f, indent=2)

    with open(save_dir / "summary.json", "w") as f:
        json.dump(to_serializable(result), f, indent=2)
    save_metric_guide(save_dir / "metric_guide.json")


class TeeLogger:
    """Mirror progress to stdout and to an in-memory log file."""

    def __init__(self):
        self.lines: List[str] = []

    def log(self, message: str):
        print(message)
        self.lines.append(message)

    def block(self, message: str):
        print(f"\n{message}")
        self.lines.append(message)

    def save(self, path: Path):
        path.write_text("\n".join(self.lines).rstrip() + "\n")


def projection_plot_name(projection: str, color_dims: Sequence[int]) -> str:
    dims = "_".join(f"state{dim}" for dim in color_dims)
    return f"{projection}_{dims}.png"


def compare_plot_name(left_slug: str, right_slug: str, projection: str, color_dims: Sequence[int]) -> str:
    dims = "_".join(f"state{dim}" for dim in color_dims)
    return f"{left_slug}__{right_slug}_{projection}_{dims}.png"


def write_wide_csv(path: Path, rows: List[Dict[str, Any]]):
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_long_csv(path: Path, rows: List[Dict[str, Any]]):
    if not rows:
        raise ValueError("No rows to save in long CSV.")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_cell(value: Any) -> str:
    return format_metric_value(value).replace("|", "\\|").replace("\n", " ")


def write_markdown_report(path: Path, model_records: List[Dict[str, Any]]):
    labels = [record["label"] for record in model_records]
    lines = [
        "# Representation Batch Analysis",
        "",
        "This file compares the scalar metrics from multiple representation-analysis runs.",
        "",
        "## Models",
        "",
        "| label | slug | analysis_dir | ckpt |",
        "| --- | --- | --- | --- |",
    ]
    for record in model_records:
        lines.append(
            f"| {markdown_cell(record['label'])} | {markdown_cell(record['slug'])} | "
            f"{markdown_cell(record['analysis_dir'])} | {markdown_cell(record['result']['meta']['ckpt'])} |"
        )

    for section in SECTION_ORDER:
        if section in {"meta", "visualization"}:
            continue
        if not any(section in record["result"] for record in model_records):
            continue

        lines.extend([
            "",
            f"## {SECTION_TITLES.get(section, section.title())}",
            "",
            "| metric | better | summary | " + " | ".join(markdown_cell(label) for label in labels) + " |",
            "| --- | --- | --- | " + " | ".join("---" for _ in labels) + " |",
        ])

        metric_keys: List[str] = []
        for record in model_records:
            metrics = record["result"].get(section, {})
            if not isinstance(metrics, dict):
                continue
            preferred_keys = list(METRIC_SPECS.get(section, {}).keys())
            extra_keys = [key for key in metrics.keys() if key not in preferred_keys]
            for key in preferred_keys + extra_keys:
                if key in metrics and key not in metric_keys:
                    metric_keys.append(key)

        for key in metric_keys:
            spec = metric_spec(section, key)
            row = [
                markdown_cell(key),
                markdown_cell(spec.get("better", "context")),
                markdown_cell(spec.get("summary", "")),
            ]
            for record in model_records:
                value = record["result"].get(section, {}).get(key, "")
                row.append(markdown_cell(value))
            lines.append("| " + " | ".join(row) + " |")

    path.write_text("\n".join(lines).rstrip() + "\n")


def resolve_compare_pairs(
    *,
    compare_all_pairs: bool,
    compare_pair_specs: Sequence[str],
    compare_projections: Sequence[str],
    model_records: List[Dict[str, Any]],
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    by_label = {record["label"]: record for record in model_records}
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    seen: set[Tuple[str, str]] = set()

    if compare_all_pairs:
        requested = [(left["label"], right["label"]) for left, right in itertools.combinations(model_records, 2)]
    elif compare_pair_specs:
        requested = [parse_compare_pair(spec) for spec in compare_pair_specs]
    elif len(model_records) == 2 and compare_projections:
        requested = [(model_records[0]["label"], model_records[1]["label"])]
    else:
        requested = []

    for left_label, right_label in requested:
        if left_label not in by_label:
            raise KeyError(f"Unknown compare pair label: {left_label}")
        if right_label not in by_label:
            raise KeyError(f"Unknown compare pair label: {right_label}")
        if left_label == right_label:
            raise ValueError(f"Compare pair must use two different labels: {left_label}")
        left = by_label[left_label]
        right = by_label[right_label]
        pair_key = tuple(sorted((left["slug"], right["slug"])))
        if pair_key in seen:
            continue
        seen.add(pair_key)
        pairs.append((left, right))
    return pairs


def render_pairwise_comparisons(
    *,
    root_dir: Path,
    model_records: List[Dict[str, Any]],
    compare_projections: Sequence[str],
    compare_pair_specs: Sequence[str],
    compare_all_pairs: bool,
    color_dims: Sequence[int],
    alpha: float,
    size: float,
    title: str,
    logger: TeeLogger,
) -> List[Dict[str, str]]:
    compare_pairs = resolve_compare_pairs(
        compare_all_pairs=compare_all_pairs,
        compare_pair_specs=compare_pair_specs,
        compare_projections=compare_projections,
        model_records=model_records,
    )
    if not compare_pairs or not compare_projections:
        return []

    from tools.repr_analysis.compare_repr import load_rows as load_compare_rows
    from tools.repr_analysis.compare_repr import load_summary, save_comparison_plot

    compare_dir = root_dir / "pairwise_compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    manifests: List[Dict[str, str]] = []

    for left, right in compare_pairs:
        for projection in compare_projections:
            left_rows = load_compare_rows(Path(left["analysis_dir"]), projection)
            right_rows = load_compare_rows(Path(right["analysis_dir"]), projection)
            left_summary = load_summary(Path(left["analysis_dir"]))
            right_summary = load_summary(Path(right["analysis_dir"]))
            output = compare_dir / compare_plot_name(left["slug"], right["slug"], projection, color_dims)
            save_comparison_plot(
                left_rows=left_rows,
                right_rows=right_rows,
                left_summary=left_summary,
                right_summary=right_summary,
                left_label=left["label"],
                right_label=right["label"],
                projection=projection,
                color_dims=list(color_dims),
                output=output,
                title=title,
                alpha=alpha,
                size=size,
            )
            logger.log(
                f"[analyze_repr] saved compare plot: {output} "
                f"({left['label']} vs {right['label']}, {projection})"
            )
            manifests.append({
                "left_label": left["label"],
                "right_label": right["label"],
                "left_slug": left["slug"],
                "right_slug": right["slug"],
                "projection": projection,
                "output": str(output),
            })

    with open(compare_dir / "compare_manifest.json", "w") as f:
        json.dump(to_serializable(manifests), f, indent=2)
    return manifests


def run_batch_analysis(
    *,
    model_specs: Sequence[str | Sequence[str]],
    dataset: str,
    save_dir: str | Path,
    state_key: str | None,
    frameskip: int,
    img_size: int,
    n_sequences: int,
    future_steps: int,
    max_points: int,
    knn_k: int,
    action_trials: int,
    planning_random_trials: int,
    include_planning: bool,
    interp_steps: int,
    perturb_scale: float,
    seed: int,
    device: str,
    export_tsne: bool,
    tsne_perplexity: float,
    plot_projections: Sequence[str],
    compare_projections: Sequence[str],
    compare_pair_specs: Sequence[str],
    compare_all_pairs: bool,
    color_dims: Sequence[int],
    alpha: float,
    size: float,
) -> List[Dict[str, Any]]:
    if "tsne" in plot_projections and not export_tsne:
        raise ValueError("`plot_projections=['tsne']` requires `export_tsne=True`.")
    if "tsne" in compare_projections and not export_tsne:
        raise ValueError("`compare_projections=['tsne']` requires `export_tsne=True`.")
    if compare_pair_specs and not compare_projections:
        raise ValueError("`compare_pair_specs` requires `compare_projections`.")
    if compare_all_pairs and not compare_projections:
        raise ValueError("`compare_all_pairs=True` requires `compare_projections`.")

    root_dir = Path(save_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    save_metric_guide(root_dir / "metric_guide.json")

    from tools.repr_analysis.plot_repr import load_rows, save_projection_plot

    logger = TeeLogger()
    parsed_specs = [parse_model_spec(spec) for spec in model_specs]
    used_slugs: set[str] = set()
    model_records: List[Dict[str, Any]] = []

    logger.log(f"[analyze_repr] models={len(parsed_specs)} save_dir={root_dir}")
    for index, (label, ckpt) in enumerate(parsed_specs, start=1):
        slug = make_unique_slug(label, used_slugs)
        model_dir = root_dir / slug

        logger.log(f"\n[analyze_repr] ({index}/{len(parsed_specs)}) {label}")
        result, outputs = run_analysis(
            ckpt=ckpt,
            dataset=dataset,
            state_key=state_key,
            frameskip=frameskip,
            img_size=img_size,
            n_sequences=n_sequences,
            future_steps=future_steps,
            max_points=max_points,
            knn_k=knn_k,
            action_trials=action_trials,
            planning_random_trials=planning_random_trials,
            include_planning=include_planning,
            interp_steps=interp_steps,
            perturb_scale=perturb_scale,
            seed=seed,
            device=device,
            log=logger.log,
        )

        save_outputs(
            model_dir,
            result,
            outputs["emb"],
            outputs.get("state"),
            export_tsne=export_tsne,
            tsne_perplexity=tsne_perplexity,
            seed=seed,
        )
        logger.log(f"[analyze_repr] saved analysis dir: {model_dir}")

        for projection in plot_projections:
            projection_json = model_dir / f"{projection}_projection.json"
            if not projection_json.exists():
                logger.log(f"[analyze_repr] skip {projection} plot for {label}: {projection_json.name} not found")
                continue
            plot_path = model_dir / projection_plot_name(projection, color_dims)
            save_projection_plot(
                load_rows(projection_json),
                output=plot_path,
                color_dims=list(color_dims),
                title=f"{label} {projection.upper()}",
                alpha=alpha,
                size=size,
            )
            logger.log(f"[analyze_repr] saved {projection} plot: {plot_path}")

        logger.block(format_analysis_report(result))
        model_records.append({
            "label": label,
            "slug": slug,
            "analysis_dir": str(model_dir),
            "result": result,
        })

    wide_rows: List[Dict[str, Any]] = []
    long_rows: List[Dict[str, Any]] = []
    for record in model_records:
        wide_row: Dict[str, Any] = {
            "model_label": record["label"],
            "model_slug": record["slug"],
            "analysis_dir": record["analysis_dir"],
        }
        for section, key, value in metric_table_entries(record["result"]):
            metric_name = f"{section}.{key}"
            wide_row[metric_name] = value
            spec = metric_spec(section, key)
            long_rows.append({
                "model_label": record["label"],
                "model_slug": record["slug"],
                "analysis_dir": record["analysis_dir"],
                "section": section,
                "metric": key,
                "metric_name": metric_name,
                "value": value,
                "better": spec.get("better", "context"),
                "summary": spec.get("summary", ""),
                "use": spec.get("use", ""),
            })
        wide_rows.append(wide_row)

    compare_manifest = render_pairwise_comparisons(
        root_dir=root_dir,
        model_records=model_records,
        compare_projections=compare_projections,
        compare_pair_specs=compare_pair_specs,
        compare_all_pairs=compare_all_pairs,
        color_dims=color_dims,
        alpha=alpha,
        size=size,
        title=f"{dataset} Representation Comparison",
        logger=logger,
    )

    wide_csv = root_dir / "metrics_wide.csv"
    long_csv = root_dir / "metrics_long.csv"
    markdown_report = root_dir / "metrics_table.md"
    batch_summary_json = root_dir / "batch_summary.json"
    batch_log = root_dir / "batch_report.txt"

    write_wide_csv(wide_csv, wide_rows)
    write_long_csv(long_csv, long_rows)
    write_markdown_report(markdown_report, model_records)
    with open(batch_summary_json, "w") as f:
        json.dump(to_serializable({"models": model_records, "compare_manifest": compare_manifest}), f, indent=2)

    logger.log(f"\n[analyze_repr] saved wide table: {wide_csv}")
    logger.log(f"[analyze_repr] saved long table: {long_csv}")
    logger.log(f"[analyze_repr] saved markdown table: {markdown_report}")
    if compare_manifest:
        logger.log(f"[analyze_repr] saved pairwise compares: {root_dir / 'pairwise_compare'}")
    logger.log(f"[analyze_repr] saved batch summary: {batch_summary_json}")
    logger.save(batch_log)
    print(f"[analyze_repr] saved batch log: {batch_log}")
    return model_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None, help="Single-model object checkpoint path")
    parser.add_argument("--model", action="append", default=[], help="Repeatable `label=ckpt_path` entry for batch mode.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name/path")
    parser.add_argument(
        "--state-key",
        type=str,
        default=None,
        help="Optional external state key used only for the reference probe, e.g. proprio",
    )
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--n-sequences", type=int, default=128)
    parser.add_argument("--future-steps", type=int, default=8, help="Future steps used for rollout and planning-signal probes")
    parser.add_argument("--max-points", type=int, default=512)
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--action-trials", type=int, default=8)
    parser.add_argument("--planning-random-trials", type=int, default=16)
    parser.add_argument(
        "--skip-planning",
        action="store_true",
        help="Skip the planning-signal probe. Useful when comparing checkpoints whose planner code paths are unavailable or incompatible.",
    )
    parser.add_argument("--interp-steps", type=int, default=9)
    parser.add_argument("--perturb-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--export-tsne", action="store_true", help="Export t-SNE projection if sklearn is installed")
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument(
        "--plot-projections",
        type=str,
        nargs="*",
        default=[],
        choices=["pca", "tsne"],
        help="Projection types to render as per-model PNG files in batch mode.",
    )
    parser.add_argument(
        "--compare-projections",
        type=str,
        nargs="*",
        default=[],
        choices=["pca", "tsne"],
        help="Projection types to render as pairwise comparison PNG files in batch mode.",
    )
    parser.add_argument(
        "--compare-pair",
        action="append",
        default=[],
        help="Optional pair spec using model labels, e.g. `SIGReg=BN+uniformity`. Repeatable.",
    )
    parser.add_argument(
        "--compare-all-pairs",
        action="store_true",
        help="If set, render comparison plots for every model pair in batch mode.",
    )
    parser.add_argument("--color-dims", type=int, nargs="+", default=[0], help="State dimensions used for plot colors.")
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--size", type=float, default=6.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if bool(args.ckpt) == bool(args.model):
        parser.error("Provide exactly one of `--ckpt` or `--model`.")

    if args.export_tsne:
        print(
            "\n[analyze_repr] t-SNE requested: this is for local-neighborhood visualisation only; "
            "do not use it as a quantitative judge of planning signal or global latent quality."
        )

    common_kwargs = dict(
        state_key=args.state_key,
        frameskip=args.frameskip,
        img_size=args.img_size,
        n_sequences=args.n_sequences,
        future_steps=args.future_steps,
        max_points=args.max_points,
        knn_k=args.knn_k,
        action_trials=args.action_trials,
        planning_random_trials=args.planning_random_trials,
        include_planning=not args.skip_planning,
        interp_steps=args.interp_steps,
        perturb_scale=args.perturb_scale,
        seed=args.seed,
        device=args.device,
    )

    if args.model:
        if not args.save_dir:
            parser.error("Batch mode requires `--save-dir`.")
        run_batch_analysis(
            model_specs=args.model,
            dataset=args.dataset,
            save_dir=args.save_dir,
            export_tsne=args.export_tsne,
            tsne_perplexity=args.tsne_perplexity,
            plot_projections=args.plot_projections,
            compare_projections=args.compare_projections,
            compare_pair_specs=args.compare_pair,
            compare_all_pairs=args.compare_all_pairs,
            color_dims=args.color_dims,
            alpha=args.alpha,
            size=args.size,
            **common_kwargs,
        )
        return

    result, outputs = run_analysis(
        ckpt=args.ckpt,
        dataset=args.dataset,
        log=print,
        **common_kwargs,
    )

    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_outputs(
            save_dir,
            result,
            outputs["emb"],
            outputs.get("state"),
            export_tsne=args.export_tsne,
            tsne_perplexity=args.tsne_perplexity,
            seed=args.seed,
        )
        print(f"\n[analyze_repr] saved outputs to: {save_dir}")

    print(f"\n{format_analysis_report(result)}")


if __name__ == "__main__":
    main()
