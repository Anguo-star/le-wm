"""Aggregate PLDM per-ckpt diagnostics into a canonical JSON file.

Mirrors the ``predictor_metrics_by_task`` block of
``assets/paper1_data/canonical_diagnostics_20260517.json`` (the LeWM
source-of-truth) so the paper's cross-checkpoint correlation analysis
can consume both files via the same loader.

For each (task, std_max) we read::

    <ckpt>/eval_results/diagnostics/diagnostics_summary.json

and pull the two metrics with full coverage across all checkpoints:

    predictor_target_to_nn_cos_ratio_at_max_std   ("fragility ratio")
    predictor_rollout_T8_l2                       ("multi-step drift")

The LeWM canonical uses the suffix ``predictor_rollout_T8_l2_at_max_std``;
PLDM's per-ckpt diagnostic-summary records the same quantity under
``predictor_rollout_T8_l2``. We store it under the LeWM-canonical name
for downstream uniformity.

Run::

    python -m tools.build_canonical_diagnostics_pldm \\
        --root "$PAPER1_DATA_ROOT" \\
        --out  assets/paper1_data/canonical_diagnostics_pldm_<DATE>.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from tools.paper1_paths import paper1_data_root


TASKS = ("TwoRoom", "PushT", "Reacher", "Cube")
TASK_TO_ENV = {
    "TwoRoom": "lewm-tworooms",
    "PushT":   "lewm-pusht",
    "Reacher": "lewm-reacher",
    "Cube":    "lewm-cube",
}
TASK_TO_PREFIX = {
    "TwoRoom": "tworoom",
    "PushT":   "pusht",
    "Reacher": "reacher",
    "Cube":    "cube",
}
# Diagnostic injection ceiling (max std) — same across all four tasks
DIAG_MAX_STD = 0.1


def _std_key_from_subdir(subdir: str) -> str | None:
    """``noise_0toNNN`` ↔ ``std_max = 0.NN`` (verified against per-ckpt config.yaml)."""
    if subdir.endswith("_pldm_baseline"):
        return "0.0"
    m = re.search(r"_pldm_noise_0to(\d+)_p1(?:_\d{8})?$", subdir)
    if not m:
        return None
    n = int(m.group(1))
    return f"0.{n:02d}"


def _read_one(eval_results: Path) -> dict | None:
    fp = eval_results / "diagnostics" / "diagnostics_summary.json"
    if not fp.exists():
        return None
    payload = json.loads(fp.read_text())
    if not isinstance(payload, list) or not payload:
        return None
    row = payload[0]
    frag = row.get("predictor_target_to_nn_cos_ratio_at_max_std")
    drift = row.get("predictor_rollout_T8_l2")
    if frag is None or drift is None:
        return None
    return {
        "diagnostic_max_std": DIAG_MAX_STD,
        "predictor_rollout_T8_l2_at_max_std": float(drift),
        "predictor_target_to_nn_cos_ratio_at_max_std": float(frag),
    }


def build(root: Path, out_path: Path) -> dict:
    canonical: dict = {
        "metadata": {
            "schema_version": "pldm-1.0",
            "diagnostic_max_std": DIAG_MAX_STD,
            "metric_names": [
                "predictor_target_to_nn_cos_ratio_at_max_std",
                "predictor_rollout_T8_l2_at_max_std",
            ],
            "source": "per-ckpt eval_results/diagnostics/diagnostics_summary.json",
        },
        "predictor_metrics_by_task": {t: {} for t in TASKS},
    }
    missing: list[str] = []
    for task in TASKS:
        ckpt_root = root / TASK_TO_ENV[task] / "ckpt"
        if not ckpt_root.is_dir():
            missing.append(f"{task}: no ckpt root {ckpt_root}")
            continue
        prefix = TASK_TO_PREFIX[task]
        for ckpt_dir in sorted(ckpt_root.iterdir()):
            if not ckpt_dir.is_dir():
                continue
            if not ckpt_dir.name.startswith(f"{prefix}_pldm_"):
                continue
            std_key = _std_key_from_subdir(ckpt_dir.name)
            if std_key is None:
                continue
            row = _read_one(ckpt_dir / "eval_results")
            if row is None:
                missing.append(f"{task} {ckpt_dir.name}: no diagnostics_summary")
                continue
            previous = canonical["predictor_metrics_by_task"][task].get(std_key)
            if previous is not None and ckpt_dir.name <= previous["subdir"]:
                continue
            row["subdir"] = ckpt_dir.name
            canonical["predictor_metrics_by_task"][task][std_key] = row
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(canonical, indent=2, sort_keys=True))
    return {"canonical": canonical, "missing": missing}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        default=str(paper1_data_root()),
    )
    ap.add_argument(
        "--out",
        default="assets/paper1_data/canonical_diagnostics_pldm_20260522.json",
    )
    args = ap.parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = repo_root / out_path
    report = build(Path(args.root), out_path)
    print(f"wrote {out_path}")
    pmbt = report["canonical"]["predictor_metrics_by_task"]
    for task, by_std in pmbt.items():
        print(f"  {task}: {len(by_std)} ckpts ({sorted(by_std.keys())})")
    if report["missing"]:
        print("\nMissing / incomplete:")
        for line in report["missing"]:
            print(f"  - {line}")


if __name__ == "__main__":
    main()
