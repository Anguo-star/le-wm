"""Phase-0 ACPC diagnostics for Paper 1.

This runner computes paired clean/corrupted predictive-dynamics diagnostics on
existing LeWM/PLDM checkpoints when loadable model objects are available. It is
intended to produce a diagnostic artifact, not paper-facing numbers by itself.

The candidate action set is fixed across clean and corrupted branches. ADM is a
latent action-distance proxy unless a stronger task oracle is added later.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

torch = None
F = None
encode_sequences = None
get_embedding_space = None
get_model_spaces = None
infer_history_size = None
load_dataset_samples = None
load_model = None
sample_random_future_actions = None
spearman_corr = None
_autoregressive_rollout = None
_clean_nn_dist = None
_open_loop_target_shift = None
_shift_stats = None
make_eval_corruption = None


def _ensure_runtime_deps() -> None:
    """Import torch/model utilities only for real computation.

    This keeps ``--dry-run`` usable on machines that only need to inspect the
    canonical manifests and do not have the training stack installed.
    """
    global torch, F
    global encode_sequences, get_embedding_space, get_model_spaces
    global infer_history_size, load_dataset_samples, load_model
    global sample_random_future_actions, spearman_corr
    global _autoregressive_rollout, _clean_nn_dist, _open_loop_target_shift, _shift_stats
    global make_eval_corruption

    if torch is not None:
        return

    import torch as torch_mod
    import torch.nn.functional as functional_mod

    from tools.repr_analysis.analyze_repr import (
        encode_sequences as encode_sequences_fn,
        get_embedding_space as get_embedding_space_fn,
        get_model_spaces as get_model_spaces_fn,
        infer_history_size as infer_history_size_fn,
        load_dataset_samples as load_dataset_samples_fn,
        load_model as load_model_fn,
        sample_random_future_actions as sample_random_future_actions_fn,
        spearman_corr as spearman_corr_fn,
    )
    from tools.repr_analysis.predictor_sensitivity import (
        _autoregressive_rollout as autoregressive_rollout_fn,
        _clean_nn_dist as clean_nn_dist_fn,
        _open_loop_target_shift as open_loop_target_shift_fn,
        _shift_stats as shift_stats_fn,
    )
    from utils import make_eval_corruption as make_eval_corruption_fn

    torch = torch_mod
    F = functional_mod
    encode_sequences = encode_sequences_fn
    get_embedding_space = get_embedding_space_fn
    get_model_spaces = get_model_spaces_fn
    infer_history_size = infer_history_size_fn
    load_dataset_samples = load_dataset_samples_fn
    load_model = load_model_fn
    sample_random_future_actions = sample_random_future_actions_fn
    spearman_corr = spearman_corr_fn
    _autoregressive_rollout = autoregressive_rollout_fn
    _clean_nn_dist = clean_nn_dist_fn
    _open_loop_target_shift = open_loop_target_shift_fn
    _shift_stats = shift_stats_fn
    make_eval_corruption = make_eval_corruption_fn


TASK_DATASETS = {
    "TwoRoom": "tworoom",
    "PushT": "pusht_expert_train",
    "Reacher": "reacher",
    "Cube": "ogbench/cube_single_expert",
}
TASKS = tuple(TASK_DATASETS)
STD_KEYS = ("0.0", "0.01", "0.02", "0.03", "0.04", "0.05", "0.06", "0.07", "0.08")
METHOD_EVALS = {
    "LeWM": "assets/paper1_data/canonical_evals_20260517.json",
    "PLDM": "assets/paper1_data/canonical_evals_pldm_20260522.json",
}


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _jsonable(obj: Any) -> Any:
    if torch is not None and torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _mean_metric(entry: Mapping[str, Any], metric: str) -> float:
    value = entry.get("metrics", {}).get(metric, {}).get("mean", float("nan"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _safe_quantile(x: torch.Tensor, q: float) -> float:
    if x.numel() == 0:
        return float("nan")
    return float(torch.quantile(x.detach().float().cpu(), q))


def _safe_mean(x: torch.Tensor) -> float:
    if x.numel() == 0:
        return float("nan")
    return float(x.detach().float().mean().cpu())


def _candidate_model_files(directory: Path) -> list[Path]:
    if not directory.exists() or not directory.is_dir():
        return []
    patterns = (
        "*_object.ckpt",
        "model_object*.ckpt",
        "*.ckpt",
        "*.pt",
        "*.pth",
        "*epoch_*",
    )
    files: list[Path] = []
    for pattern in patterns:
        files.extend(p for p in directory.glob(pattern) if p.is_file())
    files = [p for p in files if "eval_results" not in p.parts]

    def score(path: Path) -> tuple[int, int, float, str]:
        name = path.name
        is_object = int("object" in name and path.suffix == ".ckpt")
        match = re.search(r"epoch[_-](\d+)", name)
        epoch = int(match.group(1)) if match else -1
        size = path.stat().st_size if path.exists() else 0
        return (is_object, epoch, size, name)

    return sorted(set(files), key=score, reverse=True)


def _alternate_dirs(run_path: Path, subdir: str, model_roots: Sequence[Path]) -> list[Path]:
    dirs = [run_path]
    for root in model_roots:
        dirs.extend([root / subdir, root / "ckpt" / subdir, root / "checkpoints" / subdir])
    # Canonical eval paths point at <task-root>/ckpt/<subdir>. Some PLDM
    # training runs keep weights under the sibling <task-root>/checkpoints/.
    if run_path.parent.name == "ckpt":
        task_root = run_path.parent.parent
        dirs.append(task_root / "checkpoints" / subdir)
    return list(dict.fromkeys(dirs))


def resolve_model_file(
    run_path: str,
    subdir: str,
    model_roots: Sequence[Path],
) -> tuple[Path | None, list[str]]:
    path = Path(run_path).expanduser()
    tried: list[str] = []
    if path.is_file():
        return path, [str(path)]
    for directory in _alternate_dirs(path, subdir, model_roots):
        tried.append(str(directory))
        candidates = _candidate_model_files(directory)
        if candidates:
            return candidates[0], tried
    return None, tried


def _clone_batch(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.clone() if torch.is_tensor(v) else copy.deepcopy(v) for k, v in batch.items()}


def _corrupt_pixels(x: torch.Tensor, magnitude: float, seed: int, corruption_type: str) -> torch.Tensor:
    transform = make_eval_corruption(magnitude, corruption_type)
    if transform is None:
        return x.clone()
    with torch.random.fork_rng(devices=[x.device] if x.device.type == "cuda" else []):
        torch.manual_seed(seed)
        return transform(x)


def make_paired_noisy_batch(
    batch: Mapping[str, torch.Tensor],
    *,
    history_size: int,
    noise_std: float,
    seed: int,
    corruption_type: str,
    corrupt_goal: bool,
) -> dict[str, torch.Tensor]:
    noisy = _clone_batch(batch)
    pixels = noisy["pixels"]
    noisy_history = _corrupt_pixels(pixels[:, :history_size], noise_std, seed, corruption_type)
    pixels = torch.cat([noisy_history, pixels[:, history_size:]], dim=1)
    if corrupt_goal and pixels.size(1) > history_size:
        goal = _corrupt_pixels(pixels[:, -1:], noise_std, seed + 17, corruption_type)
        pixels = torch.cat([pixels[:, :-1], goal], dim=1)
    noisy["pixels"] = pixels
    return noisy


def _transition_l2_reference(clean_emb: torch.Tensor, history_size: int, horizon: int) -> float:
    start = history_size
    stop = min(clean_emb.size(1), history_size + horizon)
    if stop <= start:
        return float("nan")
    current = clean_emb[:, start:stop]
    previous = clean_emb[:, start - 1 : stop - 1]
    return _safe_quantile(torch.linalg.vector_norm(current - previous, dim=-1).reshape(-1), 0.5)


def compute_acpc_prediction_metrics(
    model,
    clean_outputs: Mapping[str, torch.Tensor],
    noisy_outputs: Mapping[str, torch.Tensor],
    *,
    history_size: int,
    rollout_horizon: int,
    embedding_space: str,
) -> dict[str, float]:
    clean_emb = get_embedding_space(clean_outputs, embedding_space).detach()
    noisy_emb = get_embedding_space(noisy_outputs, embedding_space).detach()
    act_emb = clean_outputs["act_emb"].detach()

    epd = _shift_stats(clean_emb[:, :history_size], noisy_emb[:, :history_size])
    nn_ref = _clean_nn_dist(clean_emb[:, :history_size])

    one_step = _open_loop_target_shift(model, clean_emb, noisy_emb, act_emb, history_size)
    one_step_stats = _shift_stats(one_step["clean_pred"], one_step["noisy_pred"])

    max_steps = min(rollout_horizon, max(0, act_emb.size(1) - history_size + 1))
    chain_clean = _autoregressive_rollout(
        model, clean_emb[:, :history_size], act_emb, history_size, max_steps
    )
    chain_noisy = _autoregressive_rollout(
        model, noisy_emb[:, :history_size], act_emb, history_size, max_steps
    )
    pred_clean = chain_clean[:, history_size : history_size + max_steps]
    pred_noisy = chain_noisy[:, history_size : history_size + max_steps]
    rollout_stats = _shift_stats(pred_clean, pred_noisy)

    final_stats = _shift_stats(pred_clean[:, -1:], pred_noisy[:, -1:]) if max_steps else {}
    transition_ref = _transition_l2_reference(clean_emb, history_size, max_steps)
    one_step_transition_ref = _transition_l2_reference(clean_emb, history_size, 1)

    acpc_1_l2 = one_step_stats["l2_median"]
    acpc_h_l2 = rollout_stats["l2_median"]
    return {
        "embedding_space": embedding_space,
        "history_size": float(history_size),
        "rollout_horizon_requested": float(rollout_horizon),
        "rollout_horizon_actual": float(max_steps),
        "encoder_shift_l2_median": epd["l2_median"],
        "encoder_shift_cos_median": epd["cos_dist_median"],
        "encoder_shift_to_nn_l2": epd["l2_median"] / nn_ref["l2"] if nn_ref["l2"] > 0 else float("nan"),
        "acpc_1_l2_median": acpc_1_l2,
        "acpc_1_cos_median": one_step_stats["cos_dist_median"],
        "acpc_1_norm_by_transition": (
            acpc_1_l2 / one_step_transition_ref if one_step_transition_ref > 0 else float("nan")
        ),
        "acpc_h_l2_median": acpc_h_l2,
        "acpc_h_l2_p90": rollout_stats["l2_p90"],
        "acpc_h_cos_median": rollout_stats["cos_dist_median"],
        "acpc_h_final_l2_median": final_stats.get("l2_median", float("nan")),
        "acpc_h_final_cos_median": final_stats.get("cos_dist_median", float("nan")),
        "acpc_h_norm_by_transition": (
            acpc_h_l2 / transition_ref if transition_ref > 0 else float("nan")
        ),
        "clean_transition_l2_median": transition_ref,
        "clean_nn_l2_median": nn_ref["l2"],
        "clean_nn_cos_median": nn_ref["cos"],
    }


def build_action_candidates(
    action: torch.Tensor,
    *,
    history_size: int,
    future_steps: int,
    random_action_trials: int,
    seed: int,
) -> torch.Tensor:
    if future_steps < 2:
        raise ValueError("future_steps must be >= 2 for candidate-cost diagnostics.")
    b = action.size(0)
    future_action_steps = future_steps - 1
    expert_future = action[:, history_size : history_size + future_action_steps]
    expert_candidate = action[:, : history_size + future_action_steps].unsqueeze(1)
    random_future = sample_random_future_actions(
        expert_future, n_trials=random_action_trials, seed=seed
    )
    history = action[:, :history_size].unsqueeze(1).expand(
        b, random_action_trials, history_size, -1
    )
    random_candidates = torch.cat([history, random_future], dim=2)
    return torch.cat([expert_candidate, random_candidates], dim=1)


def _cost_info(batch: Mapping[str, torch.Tensor], history_size: int) -> dict[str, torch.Tensor]:
    return {
        "pixels": batch["pixels"][:, :history_size].unsqueeze(1),
        "action": batch["action"][:, :history_size].unsqueeze(1),
        "goal": batch["pixels"][:, -1:].unsqueeze(1),
    }


def _manual_candidate_costs(
    model,
    batch: Mapping[str, torch.Tensor],
    candidates: torch.Tensor,
    *,
    history_size: int,
) -> torch.Tensor:
    """Compute final-goal candidate costs without using model.get_cost().

    PLDM checkpoints expose the same encode/action_encoder/predict primitives as
    LeWM, but the upstream PLDM get_cost path broadcasts a (B, 1, D) goal
    embedding directly against a (B, S, T, D) rollout tensor. That aligns the
    batch dimension with the candidate dimension when S != B. This helper keeps
    the dimensions explicit and returns the same planner-facing (B, S) cost
    surface needed by PCC/CRA/MAF.
    """
    b, n_candidates, action_steps = candidates.shape[:3]
    context = {
        "pixels": batch["pixels"][:, :history_size],
        "action": batch["action"][:, :history_size],
    }
    goal = {"pixels": batch["pixels"][:, -1:]}

    context_outputs = model.encode(context)
    goal_outputs = model.encode(goal)
    context_emb = context_outputs["emb"].detach()
    goal_emb = goal_outputs["emb"][:, -1].detach()

    init = context_emb.unsqueeze(1).expand(
        b, n_candidates, history_size, context_emb.size(-1)
    )
    init = init.reshape(b * n_candidates, history_size, context_emb.size(-1)).clone()

    act_flat = candidates.reshape(b * n_candidates, action_steps, candidates.size(-1))
    act_emb = model.action_encoder(act_flat)
    rollout_steps = max(0, action_steps - history_size + 1)
    chain = _autoregressive_rollout(model, init, act_emb, history_size, rollout_steps)
    pred_final = chain[:, -1].reshape(b, n_candidates, -1)

    goal_final = goal_emb.unsqueeze(1).expand_as(pred_final)
    spaces = get_model_spaces(model)
    if spaces["inference_cost_type"] == "cosine":
        return 1.0 - F.cosine_similarity(pred_final, goal_final, dim=-1)
    return F.mse_loss(pred_final, goal_final, reduction="none").sum(dim=-1)


def _ranking_metrics(
    clean_costs: torch.Tensor,
    noisy_costs: torch.Tensor,
    *,
    topk: int,
    margin_delta: float,
) -> dict[str, float]:
    clean_cpu = clean_costs.detach().float().cpu()
    noisy_cpu = noisy_costs.detach().float().cpu()
    abs_diff = (clean_cpu - noisy_cpu).abs()

    spearmans = []
    for clean_row, noisy_row in zip(clean_cpu, noisy_cpu):
        if clean_row.numel() > 1:
            spearmans.append(spearman_corr(clean_row, noisy_row))
    spearman_t = torch.tensor(spearmans, dtype=torch.float32)

    k = max(1, min(topk, clean_cpu.size(1)))
    clean_top = torch.topk(clean_cpu, k=k, largest=False).indices
    noisy_top = torch.topk(noisy_cpu, k=k, largest=False).indices
    overlaps = []
    for a, b in zip(clean_top, noisy_top):
        overlaps.append(len(set(a.tolist()) & set(b.tolist())) / float(k))
    overlap_t = torch.tensor(overlaps, dtype=torch.float32)

    clean_sorted = torch.sort(clean_cpu, dim=1).values
    margins = clean_sorted[:, 1] - clean_sorted[:, 0] if clean_cpu.size(1) > 1 else torch.empty(0)
    clean_best = torch.argmin(clean_cpu, dim=1)
    noisy_best = torch.argmin(noisy_cpu, dim=1)
    eligible = margins > float(margin_delta)
    if eligible.numel() and bool(eligible.any()):
        flip_rate = (clean_best[eligible] != noisy_best[eligible]).float().mean()
    else:
        flip_rate = torch.tensor(float("nan"))

    return {
        "pcc_abs_median": _safe_quantile(abs_diff.reshape(-1), 0.5),
        "pcc_abs_p90": _safe_quantile(abs_diff.reshape(-1), 0.9),
        "cra_spearman_mean": _safe_mean(spearman_t),
        "cra_spearman_median": _safe_quantile(spearman_t, 0.5),
        "elite_overlap_topk": float(k),
        "elite_overlap_mean": _safe_mean(overlap_t),
        "margin_delta": float(margin_delta),
        "margin_clean_q50": _safe_quantile(margins, 0.5),
        "margin_clean_q90": _safe_quantile(margins, 0.9),
        "maf_eligible_fraction": _safe_mean(eligible.float()) if eligible.numel() else float("nan"),
        "maf_flip_rate": float(flip_rate),
    }


def compute_cost_metrics(
    model,
    clean_batch: Mapping[str, torch.Tensor],
    noisy_batch: Mapping[str, torch.Tensor],
    *,
    method: str,
    history_size: int,
    future_steps: int,
    random_action_trials: int,
    topk: int,
    margin_delta: float,
    seed: int,
) -> dict[str, float]:
    candidates = build_action_candidates(
        clean_batch["action"],
        history_size=history_size,
        future_steps=future_steps,
        random_action_trials=random_action_trials,
        seed=seed,
    )
    if method == "PLDM":
        clean_costs = _manual_candidate_costs(
            model, clean_batch, candidates, history_size=history_size
        )
        noisy_costs = _manual_candidate_costs(
            model, noisy_batch, candidates, history_size=history_size
        )
    else:
        clean_costs = model.get_cost(_cost_info(clean_batch, history_size), candidates)
        noisy_costs = model.get_cost(_cost_info(noisy_batch, history_size), candidates)
    return {
        "candidate_count": float(candidates.size(1)),
        "future_steps": float(future_steps),
        **_ranking_metrics(
            clean_costs,
            noisy_costs,
            topk=topk,
            margin_delta=margin_delta,
        ),
    }


def compute_adm_proxy(
    clean_outputs: Mapping[str, torch.Tensor],
    clean_rollout_final: torch.Tensor,
    *,
    history_size: int,
    rollout_horizon: int,
    action_quantile: float,
    acpc_h_l2: float,
    eps: float,
) -> dict[str, float]:
    action = clean_outputs["action"]
    stop = min(action.size(1), history_size + rollout_horizon)
    future = action[:, history_size:stop]
    if future.size(1) == 0 or future.size(0) < 2:
        return {
            "adm_pair_rule": "action_distance_proxy",
            "adm_action_quantile": float(action_quantile),
            "adm_pair_count": 0.0,
            "adm_l2_median": float("nan"),
            "sprr": float("nan"),
        }

    action_flat = future.reshape(future.size(0), -1).float()
    psi = clean_rollout_final.float()
    action_dist = torch.cdist(action_flat, action_flat, p=2)
    psi_dist = torch.cdist(psi, psi, p=2)
    mask = torch.triu(torch.ones_like(action_dist, dtype=torch.bool), diagonal=1)
    pair_action = action_dist[mask]
    pair_psi = psi_dist[mask]
    if pair_action.numel() == 0:
        adm = float("nan")
        count = 0
    else:
        threshold = torch.quantile(pair_action, float(action_quantile))
        keep = pair_action >= threshold
        count = int(keep.sum())
        adm = _safe_quantile(pair_psi[keep], 0.5) if count else float("nan")
    sprr = adm / (float(acpc_h_l2) + eps) if math.isfinite(adm) and math.isfinite(acpc_h_l2) else float("nan")
    return {
        "adm_pair_rule": "action_distance_proxy",
        "adm_action_quantile": float(action_quantile),
        "adm_pair_count": float(count),
        "adm_l2_median": adm,
        "sprr": sprr,
    }


def run_checkpoint(
    *,
    method: str,
    task: str,
    std_key: str,
    entry: Mapping[str, Any],
    model_file: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    _ensure_runtime_deps()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        model = load_model(str(model_file), device)
        history_size = infer_history_size(model)
        future_steps = max(args.future_steps, args.rollout_horizon + 1)
        batch = load_dataset_samples(
            dataset_name=TASK_DATASETS[task],
            state_key=args.state_key,
            n_sequences=args.n_sequences,
            history_size=history_size,
            future_steps=future_steps,
            frameskip=args.frameskip,
            img_size=args.img_size,
            seed=args.seed,
            device=device,
        )
        noisy_batch = make_paired_noisy_batch(
            batch,
            history_size=history_size,
            noise_std=args.noise_std,
            seed=args.seed + 1009,
            corruption_type=args.corruption_type,
            corrupt_goal=args.corrupt_goal,
        )

        spaces = get_model_spaces(model)
        embedding_space = args.embedding_space or spaces["inference_cost_space"]
        clean_outputs = encode_sequences(model, _clone_batch(batch))
        noisy_outputs = encode_sequences(model, _clone_batch(noisy_batch))
        pred_metrics = compute_acpc_prediction_metrics(
            model,
            clean_outputs,
            noisy_outputs,
            history_size=history_size,
            rollout_horizon=args.rollout_horizon,
            embedding_space=embedding_space,
        )

        cost_metrics = compute_cost_metrics(
            model,
            batch,
            noisy_batch,
            method=method,
            history_size=history_size,
            future_steps=args.future_steps,
            random_action_trials=args.random_action_trials,
            topk=args.elite_topk,
            margin_delta=args.margin_delta,
            seed=args.seed + 2027,
        )

        clean_emb = get_embedding_space(clean_outputs, embedding_space).detach()
        clean_chain = _autoregressive_rollout(
            model,
            clean_emb[:, :history_size],
            clean_outputs["act_emb"].detach(),
            history_size,
            int(pred_metrics["rollout_horizon_actual"]),
        )
        final_idx = history_size + int(pred_metrics["rollout_horizon_actual"]) - 1
        clean_final = clean_chain[:, final_idx] if final_idx >= history_size else clean_emb[:, history_size - 1]
        adm_metrics = compute_adm_proxy(
            clean_outputs,
            clean_final,
            history_size=history_size,
            rollout_horizon=int(pred_metrics["rollout_horizon_actual"]),
            action_quantile=args.adm_action_quantile,
            acpc_h_l2=pred_metrics["acpc_h_l2_median"],
            eps=args.eps,
        )

        clean_success = _mean_metric(entry, "clean")
        corrupted_success = _mean_metric(entry, "pixels_goal_std0.08")
        return {
            "status": "ok",
            "method": method,
            "task": task,
            "std_key": std_key,
            "subdir": entry.get("subdir"),
            "run_path": entry.get("path"),
            "model_file": str(model_file),
            "clean_success": clean_success,
            "pixels_goal_std0.08_success": corrupted_success,
            "corruption_drop": clean_success - corrupted_success,
            "noise_std": float(args.noise_std),
            "corruption_type": args.corruption_type,
            "corrupt_goal": bool(args.corrupt_goal),
            "n_sequences": int(args.n_sequences),
            **pred_metrics,
            **cost_metrics,
            **adm_metrics,
        }


def iter_manifest_rows(
    *,
    methods: Sequence[str],
    tasks: Sequence[str],
    std_keys: Sequence[str],
    eval_files: Mapping[str, Path],
) -> Iterable[tuple[str, str, str, Mapping[str, Any]]]:
    for method in methods:
        data = _load_json(eval_files[method])
        for task in tasks:
            task_block = data.get(task, {})
            for std_key in std_keys:
                if std_key in task_block:
                    yield method, task, std_key, task_block[std_key]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run Paper 1 Phase-0 ACPC diagnostics.")
    p.add_argument("--methods", nargs="+", default=["LeWM"], choices=sorted(METHOD_EVALS))
    p.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    p.add_argument("--std-keys", nargs="+", default=list(STD_KEYS))
    p.add_argument("--evals-lewm", default=METHOD_EVALS["LeWM"])
    p.add_argument("--evals-pldm", default=METHOD_EVALS["PLDM"])
    p.add_argument("--model-root", action="append", default=[], help="Additional root to search for model files.")
    p.add_argument("--out", default="assets/paper1_data/acpc_phase0_diagnostics.json")
    p.add_argument("--dry-run", action="store_true", help="Resolve manifests and model files without loading models.")
    p.add_argument("--limit", type=int, default=None, help="Maximum number of manifest rows to process.")

    p.add_argument("--n-sequences", type=int, default=100)
    p.add_argument("--future-steps", type=int, default=9)
    p.add_argument("--rollout-horizon", type=int, default=8)
    p.add_argument("--random-action-trials", type=int, default=64)
    p.add_argument("--elite-topk", type=int, default=8)
    p.add_argument("--margin-delta", type=float, default=0.0)
    p.add_argument("--adm-action-quantile", type=float, default=0.75)
    p.add_argument("--noise-std", type=float, default=0.08)
    p.add_argument("--corruption-type", default="gaussian_noise")
    p.add_argument("--clean-goal", dest="corrupt_goal", action="store_false", help="Do not corrupt the goal image in PCC/CRA probes.")
    p.set_defaults(corrupt_goal=True)

    p.add_argument("--state-key", default=None)
    p.add_argument(
        "--frameskip",
        type=int,
        default=5,
        help="Dataset action frameskip/action-block size. Canonical Paper 1 runs use 5.",
    )
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--embedding-space", choices=["raw", "normalized"], default=None)
    p.add_argument("--seed", type=int, default=3072)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--device", default=None, help="Default: cuda if torch sees CUDA else cpu.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    eval_files = {
        "LeWM": Path(args.evals_lewm),
        "PLDM": Path(args.evals_pldm),
    }
    model_roots = [Path(p).expanduser() for p in args.model_root]
    rows: list[dict[str, Any]] = []

    manifest = list(
        iter_manifest_rows(
            methods=args.methods,
            tasks=args.tasks,
            std_keys=args.std_keys,
            eval_files=eval_files,
        )
    )
    if args.limit is not None:
        manifest = manifest[: args.limit]

    for method, task, std_key, entry in manifest:
        model_file, tried = resolve_model_file(
            str(entry.get("path", "")),
            str(entry.get("subdir", "")),
            model_roots,
        )
        if args.dry_run or model_file is None:
            status = "dry_run" if model_file is not None else "skipped_missing_model"
            rows.append(
                {
                    "status": status,
                    "method": method,
                    "task": task,
                    "std_key": std_key,
                    "subdir": entry.get("subdir"),
                    "run_path": entry.get("path"),
                    "model_file": str(model_file) if model_file else None,
                    "model_search_dirs": tried,
                    "clean_success": _mean_metric(entry, "clean"),
                    "pixels_goal_std0.08_success": _mean_metric(entry, "pixels_goal_std0.08"),
                }
            )
            continue
        try:
            rows.append(
                run_checkpoint(
                    method=method,
                    task=task,
                    std_key=std_key,
                    entry=entry,
                    model_file=model_file,
                    args=args,
                )
            )
        except Exception as exc:  # noqa: BLE001 - artifact should record per-row failure.
            rows.append(
                {
                    "status": "error",
                    "method": method,
                    "task": task,
                    "std_key": std_key,
                    "subdir": entry.get("subdir"),
                    "run_path": entry.get("path"),
                    "model_file": str(model_file),
                    "error": repr(exc),
                }
            )

    payload = {
        "metadata": {
            "schema_version": "paper1-acpc-phase0-0.1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "methods": list(args.methods),
            "tasks": list(args.tasks),
            "std_keys": list(args.std_keys),
            "dry_run": bool(args.dry_run),
            "note": (
                "ADM uses an action-distance latent proxy; it is a Phase-0 "
                "diagnostic and should not be treated as a task-oracle result."
            ),
        },
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, indent=2)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(f"[paper1_phase0_acpc] wrote {out}")
    print("[paper1_phase0_acpc] status counts:", counts)


if __name__ == "__main__":
    main()
