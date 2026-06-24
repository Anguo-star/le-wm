"""Gaussian-noise ACPC basin diagnostics for Paper 1.

This runner complements ``paper1_phase0_acpc.py``.  Phase-0 measures one
clean/corrupted pair at a time; this script measures whether several Gaussian
noise magnitudes of the same underlying state occupy a larger encoder region
than the action-conditioned predicted futures.

Default scope is the LeWM canonical sweep:

    4 tasks x 9 train-time noise levels, using epoch_10 object checkpoints.

The script is eval-only: it loads existing checkpoints, samples fixed dataset
windows, applies the same corruption family used by the training sweep
(Gaussian pixel noise) to the history frames and optionally the goal frames,
then rolls all views forward under the same action sequence.  It intentionally
rejects blur/resize specs so the paper-facing diagnostic does not mix train/eval
corruption families.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tools import paper1_phase0_acpc as phase0


TASK_CKPT_ROOTS = {
    "TwoRoom": "lewm-tworooms",
    "PushT": "lewm-pusht",
    "Reacher": "lewm-reacher",
    "Cube": "lewm-cube",
}

DEFAULT_MODEL_ROOT = os.environ.get("PAPER1_DATA_ROOT") or os.environ.get("STABLEWM_HOME")

DEFAULT_CORRUPTIONS = (
    "gaussian_noise:0.01",
    "gaussian_noise:0.02",
    "gaussian_noise:0.03",
    "gaussian_noise:0.04",
    "gaussian_noise:0.05",
    "gaussian_noise:0.06",
    "gaussian_noise:0.07",
    "gaussian_noise:0.08",
)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _jsonable(obj: Any) -> Any:
    return phase0._jsonable(obj)


def _mean_metric(entry: Mapping[str, Any], metric: str) -> float:
    return phase0._mean_metric(entry, metric)


def _safe_quantile(x, q: float) -> float:
    if x.numel() == 0:
        return float("nan")
    return float(phase0.torch.quantile(x.detach().float().cpu(), q))


def _safe_mean(x) -> float:
    if x.numel() == 0:
        return float("nan")
    return float(x.detach().float().mean().cpu())


def corruption_tag(ctype: str, magnitude: float, index: int) -> str:
    if ctype == "gaussian_noise":
        return f"noise_std{magnitude:g}_{index}"
    return f"{ctype}_{magnitude:g}_{index}"


def parse_corruptions(items: Sequence[str]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        if ":" not in item:
            raise ValueError(
                f"Invalid corruption spec {item!r}; expected '<type>:<magnitude>'."
            )
        ctype, value = item.split(":", 1)
        if ctype != "gaussian_noise":
            raise ValueError(
                "Paper 1 ACPC basin diagnostics are Gaussian-noise-only; "
                f"got {ctype!r}. Use separate tooling for non-Gaussian stress tests."
            )
        magnitude = float(value)
        specs.append(
            {
                "type": ctype,
                "magnitude": magnitude,
                "tag": corruption_tag(ctype, magnitude, idx),
            }
        )
    return specs


def _candidate_dirs(
    *,
    task: str,
    subdir: str,
    run_path: str,
    model_roots: Sequence[Path],
) -> list[Path]:
    dirs: list[Path] = []
    if run_path:
        p = Path(run_path).expanduser()
        dirs.append(p)
    for root in model_roots:
        dirs.extend(
            [
                root / TASK_CKPT_ROOTS[task] / "ckpt" / subdir,
                root / subdir,
                root / "ckpt" / subdir,
                root / "checkpoints" / subdir,
            ]
        )
    return list(dict.fromkeys(dirs))


def resolve_epoch10_model_file(
    *,
    task: str,
    subdir: str,
    run_path: str,
    model_roots: Sequence[Path],
) -> tuple[Path | None, list[str], str]:
    tried: list[str] = []
    for directory in _candidate_dirs(
        task=task, subdir=subdir, run_path=run_path, model_roots=model_roots
    ):
        tried.append(str(directory))
        if directory.is_file() and directory.name.endswith("epoch_10_object.ckpt"):
            return directory, tried, "exact_file"
        if not directory.exists() or not directory.is_dir():
            continue
        exact = directory / f"{subdir}_epoch_10_object.ckpt"
        if exact.exists():
            return exact, tried, "exact_name"
        matches = sorted(directory.glob("*epoch_10_object.ckpt"))
        if len(matches) == 1:
            return matches[0], tried, "single_epoch10_match"
        if len(matches) > 1:
            return None, tried, "ambiguous_epoch10_matches"
    return None, tried, "missing_epoch10"


def iter_manifest_rows(
    *,
    method: str,
    tasks: Sequence[str],
    std_keys: Sequence[str],
    eval_file: Path,
) -> Iterable[tuple[str, str, str, Mapping[str, Any]]]:
    data = _load_json(eval_file)
    for task in tasks:
        task_block = data.get(task, {})
        for std_key in std_keys:
            if std_key in task_block:
                yield method, task, std_key, task_block[std_key]


def iter_base_vs_best_rows(
    *,
    method: str,
    tasks: Sequence[str],
    eval_file: Path,
    robust_metric: str,
) -> Iterable[tuple[str, str, str, Mapping[str, Any]]]:
    data = _load_json(eval_file)
    for task in tasks:
        task_block = data.get(task, {})
        if "0.0" not in task_block:
            continue
        yield method, task, "0.0", task_block["0.0"]
        best_std, best_entry = max(
            (
                (std_key, entry)
                for std_key, entry in task_block.items()
                if std_key != "0.0" and robust_metric in entry.get("metrics", {})
            ),
            key=lambda item: (
                _mean_metric(item[1], robust_metric),
                _mean_metric(item[1], "clean"),
            ),
        )
        yield method, task, best_std, best_entry


def make_corrupted_batch(
    batch: Mapping[str, Any],
    *,
    history_size: int,
    spec: Mapping[str, Any],
    seed: int,
    corrupt_goal: bool,
) -> dict[str, Any]:
    return phase0.make_paired_noisy_batch(
        batch,
        history_size=history_size,
        noise_std=float(spec["magnitude"]),
        seed=seed,
        corruption_type=str(spec["type"]),
        corrupt_goal=corrupt_goal,
    )


def _view_pair_stats(views: Sequence[Any]) -> dict[str, float]:
    torch = phase0.torch
    F = phase0.F
    if len(views) < 2:
        return {
            "pair_count": 0.0,
            "l2_median": float("nan"),
            "l2_p90": float("nan"),
            "cos_median": float("nan"),
            "cos_p90": float("nan"),
        }

    l2_parts = []
    cos_parts = []
    for i in range(len(views)):
        a = views[i].reshape(-1, views[i].size(-1))
        an = F.normalize(a, dim=-1, eps=1e-8)
        for j in range(i + 1, len(views)):
            b = views[j].reshape(-1, views[j].size(-1))
            bn = F.normalize(b, dim=-1, eps=1e-8)
            l2_parts.append(torch.linalg.vector_norm(a - b, dim=-1))
            cos_parts.append((1.0 - (an * bn).sum(dim=-1).clamp(-1.0, 1.0)).clamp_min(0.0))

    l2 = torch.cat(l2_parts)
    cos = torch.cat(cos_parts)
    return {
        "pair_count": float(len(l2_parts)),
        "l2_median": _safe_quantile(l2, 0.5),
        "l2_p90": _safe_quantile(l2, 0.9),
        "cos_median": _safe_quantile(cos, 0.5),
        "cos_p90": _safe_quantile(cos, 0.9),
    }


def _prefix(prefix: str, stats: Mapping[str, float]) -> dict[str, float]:
    return {f"{prefix}_{k}": v for k, v in stats.items()}


def _norm(value: float, ref: float) -> float:
    if not math.isfinite(value) or not math.isfinite(ref) or ref <= 0:
        return float("nan")
    return value / ref


def _ratio(num: float, den: float) -> float:
    if not math.isfinite(num) or not math.isfinite(den) or abs(den) <= 1e-12:
        return float("nan")
    return num / den


def run_checkpoint(
    *,
    method: str,
    task: str,
    std_key: str,
    entry: Mapping[str, Any],
    model_file: Path,
    corruption_specs: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    phase0._ensure_runtime_deps()
    torch = phase0.torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        model = phase0.load_model(str(model_file), device)
        history_size = phase0.infer_history_size(model)
        future_steps = max(args.future_steps, args.rollout_horizon + 1)
        batch = phase0.load_dataset_samples(
            dataset_name=phase0.TASK_DATASETS[task],
            state_key=args.state_key,
            n_sequences=args.n_sequences,
            history_size=history_size,
            future_steps=future_steps,
            frameskip=args.frameskip,
            img_size=args.img_size,
            seed=args.seed,
            device=device,
        )

        spaces = phase0.get_model_spaces(model)
        embedding_space = args.embedding_space or spaces["inference_cost_space"]
        clean_outputs = phase0.encode_sequences(model, phase0._clone_batch(batch))
        clean_emb = phase0.get_embedding_space(clean_outputs, embedding_space).detach()
        act_emb = clean_outputs["act_emb"].detach()

        max_steps = min(args.rollout_horizon, max(0, act_emb.size(1) - history_size + 1))
        clean_chain = phase0._autoregressive_rollout(
            model, clean_emb[:, :history_size], act_emb, history_size, max_steps
        )
        clean_pred = clean_chain[:, history_size : history_size + max_steps]

        clean_nn = phase0._clean_nn_dist(clean_emb[:, :history_size])
        transition_ref = phase0._transition_l2_reference(clean_emb, history_size, max_steps)

        encoder_views = [clean_emb[:, :history_size]]
        pred_views = [clean_pred]
        variant_rows: list[dict[str, Any]] = []

        for idx, spec in enumerate(corruption_specs):
            noisy_batch = make_corrupted_batch(
                batch,
                history_size=history_size,
                spec=spec,
                seed=args.seed + 1009 + idx * 37,
                corrupt_goal=args.corrupt_goal,
            )
            noisy_outputs = phase0.encode_sequences(model, phase0._clone_batch(noisy_batch))
            noisy_emb = phase0.get_embedding_space(noisy_outputs, embedding_space).detach()
            noisy_chain = phase0._autoregressive_rollout(
                model,
                noisy_emb[:, :history_size],
                act_emb,
                history_size,
                max_steps,
            )
            noisy_pred = noisy_chain[:, history_size : history_size + max_steps]

            enc_stats = phase0._shift_stats(clean_emb[:, :history_size], noisy_emb[:, :history_size])
            pred_stats = phase0._shift_stats(clean_pred, noisy_pred)
            encoder_views.append(noisy_emb[:, :history_size])
            pred_views.append(noisy_pred)

            variant_rows.append(
                {
                    "tag": spec["tag"],
                    "corruption_type": spec["type"],
                    "magnitude": float(spec["magnitude"]),
                    **_prefix("encoder_to_clean", enc_stats),
                    "encoder_to_clean_l2_norm_by_nn": _norm(enc_stats["l2_median"], clean_nn["l2"]),
                    **_prefix("pred_to_clean", pred_stats),
                    "pred_to_clean_l2_norm_by_transition": _norm(
                        pred_stats["l2_median"], transition_ref
                    ),
                }
            )

        enc_pair = _view_pair_stats(encoder_views)
        pred_pair = _view_pair_stats(pred_views)
        enc_pair_norm = _norm(enc_pair["l2_median"], clean_nn["l2"])
        pred_pair_norm = _norm(pred_pair["l2_median"], transition_ref)

        enc_to_clean_l2 = torch.tensor(
            [row["encoder_to_clean_l2_median"] for row in variant_rows],
            dtype=torch.float32,
        )
        pred_to_clean_l2 = torch.tensor(
            [row["pred_to_clean_l2_median"] for row in variant_rows],
            dtype=torch.float32,
        )
        enc_to_clean_norm = torch.tensor(
            [row["encoder_to_clean_l2_norm_by_nn"] for row in variant_rows],
            dtype=torch.float32,
        )
        pred_to_clean_norm = torch.tensor(
            [row["pred_to_clean_l2_norm_by_transition"] for row in variant_rows],
            dtype=torch.float32,
        )

        clean_success = _mean_metric(entry, "clean")
        pixels_success = _mean_metric(entry, "pixels_std0.08")
        pixels_goal_success = _mean_metric(entry, "pixels_goal_std0.08")

        return {
            "status": "ok",
            "method": method,
            "task": task,
            "std_key": std_key,
            "subdir": entry.get("subdir"),
            "run_path": entry.get("path"),
            "model_file": str(model_file),
            "clean_success": clean_success,
            "pixels_std0.08_success": pixels_success,
            "pixels_goal_std0.08_success": pixels_goal_success,
            "corruption_drop": clean_success - pixels_success,
            "pixels_goal_corruption_drop": clean_success - pixels_goal_success,
            "n_sequences": int(args.n_sequences),
            "embedding_space": embedding_space,
            "history_size": float(history_size),
            "rollout_horizon_requested": float(args.rollout_horizon),
            "rollout_horizon_actual": float(max_steps),
            "corrupt_goal": bool(args.corrupt_goal),
            "corruption_count": len(corruption_specs),
            "view_count": len(corruption_specs) + 1,
            "clean_nn_l2_median": clean_nn["l2"],
            "clean_nn_cos_median": clean_nn["cos"],
            "clean_transition_l2_median": transition_ref,
            "encoder_to_clean_l2_median_over_corruptions": _safe_quantile(
                enc_to_clean_l2, 0.5
            ),
            "encoder_to_clean_l2_p90_over_corruptions": _safe_quantile(
                enc_to_clean_l2, 0.9
            ),
            "encoder_to_clean_l2_norm_by_nn_median": _safe_quantile(
                enc_to_clean_norm, 0.5
            ),
            "encoder_to_clean_l2_norm_by_nn_p90": _safe_quantile(
                enc_to_clean_norm, 0.9
            ),
            "pred_to_clean_l2_median_over_corruptions": _safe_quantile(
                pred_to_clean_l2, 0.5
            ),
            "pred_to_clean_l2_p90_over_corruptions": _safe_quantile(
                pred_to_clean_l2, 0.9
            ),
            "pred_to_clean_l2_norm_by_transition_median": _safe_quantile(
                pred_to_clean_norm, 0.5
            ),
            "pred_to_clean_l2_norm_by_transition_p90": _safe_quantile(
                pred_to_clean_norm, 0.9
            ),
            **_prefix("encoder_view_pair", enc_pair),
            **_prefix("pred_view_pair", pred_pair),
            "encoder_view_pair_l2_norm_by_nn": enc_pair_norm,
            "pred_view_pair_l2_norm_by_transition": pred_pair_norm,
            "basin_contraction_pair_norm": _ratio(pred_pair_norm, enc_pair_norm),
            "basin_contraction_to_clean_norm_median": _ratio(
                _safe_quantile(pred_to_clean_norm, 0.5),
                _safe_quantile(enc_to_clean_norm, 0.5),
            ),
            "variant_rows": variant_rows,
        }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run Paper 1 Gaussian-noise ACPC basin diagnostics.")
    p.add_argument("--methods", nargs="+", default=["LeWM"], choices=sorted(phase0.METHOD_EVALS))
    p.add_argument("--tasks", nargs="+", default=list(phase0.TASKS), choices=list(phase0.TASKS))
    p.add_argument("--std-keys", nargs="+", default=list(phase0.STD_KEYS))
    p.add_argument("--evals-lewm", default=phase0.METHOD_EVALS["LeWM"])
    p.add_argument("--evals-pldm", default=phase0.METHOD_EVALS["PLDM"])
    p.add_argument(
        "--base-vs-best",
        action="store_true",
        help="Run only each task's baseline plus robust-metric best non-baseline checkpoint.",
    )
    p.add_argument("--robust-metric", default="pixels_std0.08")
    p.add_argument(
        "--model-root",
        action="append",
        default=[DEFAULT_MODEL_ROOT] if DEFAULT_MODEL_ROOT else [],
        help="Root containing lewm-{task}/ckpt/<subdir> checkpoint directories. Defaults to PAPER1_DATA_ROOT or STABLEWM_HOME when set.",
    )
    p.add_argument("--out", default="assets/paper1_data/acpc_basin_diagnostics.json")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=None)

    p.add_argument(
        "--corruptions",
        nargs="+",
        default=list(DEFAULT_CORRUPTIONS),
        help=(
            "Eval-time same-state Gaussian-noise perturbation views. Default "
            "is the dense 0.01..0.08 grid matching the training sweep family."
        ),
    )
    p.add_argument("--n-sequences", type=int, default=100)
    p.add_argument("--future-steps", type=int, default=9)
    p.add_argument("--rollout-horizon", type=int, default=8)
    p.add_argument("--corrupt-goal", dest="corrupt_goal", action="store_true")
    p.set_defaults(corrupt_goal=False)

    p.add_argument("--state-key", default=None)
    p.add_argument("--frameskip", type=int, default=5)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--embedding-space", choices=["raw", "normalized"], default=None)
    p.add_argument("--seed", type=int, default=3072)
    p.add_argument("--device", default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    eval_files = {
        "LeWM": Path(args.evals_lewm),
        "PLDM": Path(args.evals_pldm),
    }
    model_roots = [Path(p).expanduser() for p in args.model_root]
    corruption_specs = parse_corruptions(args.corruptions)
    rows: list[dict[str, Any]] = []

    manifest: list[tuple[str, str, str, Mapping[str, Any]]] = []
    for method in args.methods:
        if args.base_vs_best:
            manifest.extend(
                iter_base_vs_best_rows(
                    method=method,
                    tasks=args.tasks,
                    eval_file=eval_files[method],
                    robust_metric=args.robust_metric,
                )
            )
        else:
            manifest.extend(
                iter_manifest_rows(
                    method=method,
                    tasks=args.tasks,
                    std_keys=args.std_keys,
                    eval_file=eval_files[method],
                )
            )
    if args.limit is not None:
        manifest = manifest[: args.limit]

    total = len(manifest)
    for row_idx, (method, task, std_key, entry) in enumerate(manifest, start=1):
        model_file, tried, resolution = resolve_epoch10_model_file(
            task=task,
            subdir=str(entry.get("subdir", "")),
            run_path=str(entry.get("path", "")),
            model_roots=model_roots,
        )
        if args.dry_run or model_file is None:
            status = "dry_run" if model_file is not None else f"skipped_{resolution}"
            rows.append(
                {
                    "status": status,
                    "method": method,
                    "task": task,
                    "std_key": std_key,
                    "subdir": entry.get("subdir"),
                    "run_path": entry.get("path"),
                    "model_file": str(model_file) if model_file else None,
                    "model_resolution": resolution,
                    "model_search_dirs": tried,
                    "clean_success": _mean_metric(entry, "clean"),
                    "pixels_std0.08_success": _mean_metric(entry, "pixels_std0.08"),
                    "pixels_goal_std0.08_success": _mean_metric(entry, "pixels_goal_std0.08"),
                }
            )
            continue

        try:
            print(
                f"[paper1_acpc_basin] ({row_idx}/{total}) "
                f"running {method} {task} std={std_key} -> {model_file}",
                flush=True,
            )
            rows.append(
                run_checkpoint(
                    method=method,
                    task=task,
                    std_key=std_key,
                    entry=entry,
                    model_file=model_file,
                    corruption_specs=corruption_specs,
                    args=args,
                )
            )
            print(
                f"[paper1_acpc_basin] ({row_idx}/{total}) "
                f"finished {method} {task} std={std_key}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - record per-row failure.
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
            "schema_version": "paper1-acpc-basin-0.1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "method": args.methods[0] if len(args.methods) == 1 else None,
            "methods": list(args.methods),
            "tasks": list(args.tasks),
            "std_keys": list(args.std_keys),
            "base_vs_best": bool(args.base_vs_best),
            "robust_metric": str(args.robust_metric),
            "corruptions": corruption_specs,
            "corrupt_goal": bool(args.corrupt_goal),
            "dry_run": bool(args.dry_run),
            "note": (
                "Basin contraction compares same-state visual-view spread before "
                "and after action-conditioned rollout.  The default corruption "
                "set is Gaussian noise on the observation history only, matching "
                "the training sweep family while keeping the goal clean."
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
    print(f"[paper1_acpc_basin] wrote {out}")
    print("[paper1_acpc_basin] status counts:", counts)
    print(f"[paper1_acpc_basin] corruptions: {', '.join(args.corruptions)}")


if __name__ == "__main__":
    main()
