"""Aggregate full PLDM five-layer diagnostics for Paper 1.

The earlier PLDM canonical diagnostic artifact stores only the two predictor
metrics used by the cross-checkpoint correlation tables. This release artifact
keeps the full diagnostics-summary rows for every PLDM checkpoint and a compact
base-vs-representative table used by Appendix F.

Run::

    python3 -m tools.build_canonical_full_diagnostics_pldm \\
        --root "$PAPER1_DATA_ROOT" \\
        --out assets/paper1_data/canonical_full_diagnostics_pldm_20260523.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from tools.paper1_paths import paper1_data_root, portable_path


TASKS = ("TwoRoom", "PushT", "Reacher", "Cube")
EXPECTED_CONFIGS = ("0.0", "0.01", "0.02", "0.03", "0.04", "0.05", "0.06", "0.07", "0.08")
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
REPRESENTATIVE_STD_BY_TASK = {
    "TwoRoom": "0.06",
    "PushT": "0.03",
    "Reacher": "0.03",
    "Cube": "0.04",
}
SUMMARY_METRICS = (
    "clean_effective_rank",
    "clean_nn_cos_dist_median",
    "transition_resolution_ratio_l2",
    "transition_resolution_ratio_cos",
    "id_probe_r2",
    "action_mean_pred_shift_norm",
    "predictor_target_to_nn_cos_ratio_at_max_std",
    "predictor_rollout_T8_l2",
)


def _std_key_from_subdir(subdir: str) -> str | None:
    if subdir.endswith("_pldm_baseline"):
        return "0.0"
    m = re.search(r"_pldm_noise_0to(\d+)_p1(?:_\d{8})?$", subdir)
    if not m:
        return None
    return f"0.{int(m.group(1)):02d}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return value


def _read_summary(ckpt_dir: Path) -> dict[str, Any]:
    fp = ckpt_dir / "eval_results" / "diagnostics" / "diagnostics_summary.json"
    payload = json.loads(fp.read_text())
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError(f"expected one-row diagnostics summary in {fp}")
    row = _json_safe(payload[0])
    if not isinstance(row, dict):
        raise ValueError(f"diagnostics summary row is not a dict in {fp}")
    return row


def build(root: Path, out_path: Path, schema_path: Path | None = None) -> dict:
    diagnostics: dict[str, dict[str, dict[str, Any]]] = {task: {} for task in TASKS}

    for task in TASKS:
        ckpt_root = root / TASK_TO_ENV[task] / "ckpt"
        prefix = TASK_TO_PREFIX[task]
        for ckpt_dir in sorted(ckpt_root.iterdir()):
            if not ckpt_dir.is_dir() or not ckpt_dir.name.startswith(f"{prefix}_pldm_"):
                continue
            std_key = _std_key_from_subdir(ckpt_dir.name)
            if std_key is None:
                continue
            row = _read_summary(ckpt_dir)
            previous = diagnostics[task].get(std_key)
            if previous is not None and ckpt_dir.name <= previous["subdir"]:
                continue
            diagnostics[task][std_key] = {
                "path": portable_path(ckpt_dir, root),
                "subdir": ckpt_dir.name,
                "diagnostics_summary": row,
            }

    missing = []
    for task in TASKS:
        got = set(diagnostics[task])
        want = set(EXPECTED_CONFIGS)
        if got != want:
            missing.append(f"{task}: missing {sorted(want - got)}, extra {sorted(got - want)}")
    if missing:
        raise RuntimeError("; ".join(missing))

    rep_values = {}
    for task in TASKS:
        base = diagnostics[task]["0.0"]["diagnostics_summary"]
        rep_key = REPRESENTATIVE_STD_BY_TASK[task]
        rep = diagnostics[task][rep_key]["diagnostics_summary"]
        rep_values[task] = {
            "representative_std": rep_key,
            "base_subdir": diagnostics[task]["0.0"]["subdir"],
            "representative_subdir": diagnostics[task][rep_key]["subdir"],
            "base": {metric: base[metric] for metric in SUMMARY_METRICS},
            "representative": {metric: rep[metric] for metric in SUMMARY_METRICS},
        }

    payload = {
        "metadata": {
            "schema_version": "pldm-full-diagnostics-1.0",
            "scope": "Full PLDM diagnostics-summary rows for 4 tasks x 9 checkpoints.",
            "source": "per-ckpt eval_results/diagnostics/diagnostics_summary.json",
            "tasks": list(TASKS),
            "configs": list(EXPECTED_CONFIGS),
            "summary_metric_order": list(SUMMARY_METRICS),
            "representative_selection": (
                "Representative rows use the PLDM pixels 0.08 clean-goal point-best checkpoint "
                "per task, except where explicitly interpreted as a diagnostic representative."
            ),
        },
        "diagnostics_by_task": diagnostics,
        "representative_std_by_task": REPRESENTATIVE_STD_BY_TASK,
        "representative_diagnostics": {
            "metric_order": list(SUMMARY_METRICS),
            "values": rep_values,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    if schema_path is not None:
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "Paper 1 canonical PLDM full diagnostics",
            "type": "object",
            "required": ["metadata", "diagnostics_by_task", "representative_diagnostics"],
            "properties": {
                "metadata": {"type": "object"},
                "diagnostics_by_task": {"type": "object"},
                "representative_diagnostics": {"type": "object"},
            },
        }
        schema_path.parent.mkdir(parents=True, exist_ok=True)
        schema_path.write_text(json.dumps(schema, indent=2, sort_keys=True))

    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        default=str(paper1_data_root()),
    )
    ap.add_argument(
        "--out",
        default="assets/paper1_data/canonical_full_diagnostics_pldm_20260523.json",
    )
    ap.add_argument(
        "--schema-out",
        default="assets/paper1_data/canonical_full_diagnostics_pldm_20260523.schema.json",
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
    reps = payload["representative_diagnostics"]["values"]
    for task in TASKS:
        row = reps[task]
        base = row["base"]
        rep = row["representative"]
        print(
            f"  {task}: std={row['representative_std']}, "
            f"rank {base['clean_effective_rank']:.2f}->{rep['clean_effective_rank']:.2f}, "
            f"T8 {base['predictor_rollout_T8_l2']:.2f}->{rep['predictor_rollout_T8_l2']:.2f}"
        )


if __name__ == "__main__":
    main()
