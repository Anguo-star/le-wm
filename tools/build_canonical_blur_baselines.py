"""Aggregate clean-checkpoint blur evals for Paper 1.

This is an eval-only cross-corruption sanity check, not a training sweep.
It reads LeWM and PLDM clean-trained baseline checkpoints, aggregates the
three evaluation seeds for Gaussian-blur corruptions, and writes a canonical
JSON artifact consumed by Appendix G.

Run::

    python3 -m tools.build_canonical_blur_baselines \\
        --root /home/ag/dataset/ag_data/data/world_model/quentinll \\
        --out assets/paper1_data/canonical_blur_baselines_20260523.json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import statistics
from pathlib import Path


TASKS = ("TwoRoom", "PushT", "Reacher", "Cube")
METHODS = ("LeWM", "PLDM")
SEEDS = (42, 43, 44)
BLUR_KERNELS = (3, 7, 11, 15)
BLUR_SCOPES = ("pixels", "goal", "pixels_goal")

TASK_TO_ENV = {
    "TwoRoom": "lewm-tworooms",
    "PushT": "lewm-pusht",
    "Reacher": "lewm-reacher",
    "Cube": "lewm-cube",
}
TASK_TO_PREFIX = {
    "TwoRoom": "tworoom",
    "PushT": "pusht",
    "Reacher": "reacher",
    "Cube": "cube",
}
METHOD_SUFFIX = {
    "LeWM": "lewm_20260430",
    "PLDM": "pldm_baseline",
}

_ARRAY_RE = re.compile(r"\b(?:np\.)?array\((?:[^()]|\([^()]*\))*\)", re.DOTALL)


def _parse_success_rate(metrics_path: Path) -> float:
    text = metrics_path.read_text(errors="replace")
    last = None
    i = 0
    while i < len(text):
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        for j in range(i, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[i : j + 1]
                    if "success_rate" in candidate:
                        last = candidate
                    i = j + 1
                    break
        else:
            break
    if last is None:
        raise ValueError(f"no success_rate dict in {metrics_path}")
    return float(ast.literal_eval(_ARRAY_RE.sub("None", last))["success_rate"])


def _aggregate(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
        "values": [round(v, 4) for v in values],
    }


def _read_condition(eval_results: Path, condition: str) -> dict:
    values = []
    missing = []
    for seed in SEEDS:
        fp = eval_results / f"{condition}_seed{seed}_metrics.txt"
        if not fp.exists():
            missing.append(fp.name)
            continue
        values.append(_parse_success_rate(fp))
    if missing:
        raise FileNotFoundError(f"{eval_results}: missing {condition} files: {missing}")
    return _aggregate(values)


def _worst_pixels_goal_blur(blur: dict[str, dict]) -> dict:
    rows = []
    for kernel in BLUR_KERNELS:
        cond = f"pixels_goal_blur_ks{kernel}"
        rows.append((blur[cond]["mean"], cond))
    worst_mean, worst_condition = min(rows)
    return {
        "condition": worst_condition,
        "kernel_size": int(worst_condition.rsplit("ks", 1)[1]),
        "summary": blur[worst_condition],
    }


def build(root: Path, out_path: Path, schema_path: Path | None = None) -> dict:
    baselines: dict[str, dict] = {method: {} for method in METHODS}
    for method in METHODS:
        for task in TASKS:
            prefix = TASK_TO_PREFIX[task]
            ckpt_dir = root / TASK_TO_ENV[task] / "ckpt" / f"{prefix}_{METHOD_SUFFIX[method]}"
            eval_results = ckpt_dir / "eval_results"
            clean = _read_condition(eval_results, "clean")
            blur = {}
            for scope in BLUR_SCOPES:
                for kernel in BLUR_KERNELS:
                    condition = f"{scope}_blur_ks{kernel}"
                    blur[condition] = _read_condition(eval_results, condition)
            worst = _worst_pixels_goal_blur(blur)
            baselines[method][task] = {
                "path": str(ckpt_dir.resolve()),
                "subdir": ckpt_dir.name,
                "clean": clean,
                "blur": blur,
                "worst_pixels_goal_blur": worst,
                "clean_to_worst_pixels_goal_blur_drop": clean["mean"] - worst["summary"]["mean"],
            }

    payload = {
        "metadata": {
            "schema_version": "blur-baselines-1.0",
            "scope": "Clean-trained LeWM and PLDM baseline evals under Gaussian blur.",
            "methods": list(METHODS),
            "tasks": list(TASKS),
            "blur_scopes": list(BLUR_SCOPES),
            "blur_kernel_sizes": list(BLUR_KERNELS),
            "evaluation_seeds": list(SEEDS),
            "trajectories_per_seed": 100,
            "success_rate_unit": "percent",
            "std_convention": "population standard deviation across evaluation seeds (ddof=0)",
            "status": "eval-only cross-corruption sanity check; not a blur-training sweep",
        },
        "baselines": baselines,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    if schema_path is not None:
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "Paper 1 canonical clean-baseline blur evals",
            "type": "object",
            "required": ["metadata", "baselines"],
            "properties": {
                "metadata": {"type": "object"},
                "baselines": {"type": "object"},
            },
        }
        schema_path.parent.mkdir(parents=True, exist_ok=True)
        schema_path.write_text(json.dumps(schema, indent=2, sort_keys=True))

    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        default="/home/ag/dataset/ag_data/data/world_model/quentinll",
        help="dataset root containing lewm-{cube,pusht,reacher,tworooms}/",
    )
    ap.add_argument(
        "--out",
        default="assets/paper1_data/canonical_blur_baselines_20260523.json",
    )
    ap.add_argument(
        "--schema-out",
        default="assets/paper1_data/canonical_blur_baselines_20260523.schema.json",
    )
    args = ap.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    out_path = Path(args.out)
    schema_path = Path(args.schema_out)
    if not out_path.is_absolute():
        out_path = repo_root / out_path
    if not schema_path.is_absolute():
        schema_path = repo_root / schema_path
    payload = build(Path(args.root), out_path, schema_path)
    print(f"wrote {out_path}")
    print(f"wrote {schema_path}")
    for method, by_task in payload["baselines"].items():
        for task, entry in by_task.items():
            worst = entry["worst_pixels_goal_blur"]
            drop = entry["clean_to_worst_pixels_goal_blur_drop"]
            print(
                f"  {method}/{task}: clean={entry['clean']['mean']:.2f}, "
                f"worst={worst['condition']} {worst['summary']['mean']:.2f}, "
                f"drop={drop:.2f}"
            )


if __name__ == "__main__":
    main()
