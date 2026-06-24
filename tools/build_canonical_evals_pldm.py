"""Aggregate PLDM noise-sweep eval results into a canonical JSON file.

Mirrors the schema of assets/paper1_data/canonical_evals_20260517.json
(the LeWM source-of-truth) so the paper figures and tables can consume
both files via the same loader.

Walks four task directories under ``$STABLEWM_HOME``:

    lewm-{cube,pusht,reacher,tworooms}/ckpt/<task>_pldm_{baseline,noise_0to00*_p1}/eval_results/

For each (task, std_max) checkpoint, reads every per-seed
``<cond>_seed{42,43,44}_metrics.txt`` file, extracts the trailing
``success_rate`` value from the metrics dict, and aggregates over the
three seeds. The population std (matching LeWM's convention) is stored.

Conditions covered: clean, goal/pixels/pixels_goal at std ∈ {0.03, 0.05, 0.08}.

Run::

    python -m tools.build_canonical_evals_pldm \\
        --root /home/ag/dataset/ag_data/data/world_model/quentinll \\
        --out  assets/paper1_data/canonical_evals_pldm_<DATE>.json
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import statistics
from pathlib import Path


# task → env-dir-name × std_key parser
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

# Conditions in the same order as the LeWM canonical so the schemas align
CONDITIONS = [
    "clean",
    "goal_std0.03",
    "pixels_std0.03",
    "pixels_goal_std0.03",
    "goal_std0.05",
    "pixels_std0.05",
    "pixels_goal_std0.05",
    "goal_std0.08",
    "pixels_std0.08",
    "pixels_goal_std0.08",
]
SEEDS = (42, 43, 44)

_ARRAY_RE = re.compile(r"\b(?:np\.)?array\((?:[^()]|\([^()]*\))*\)", re.DOTALL)


def _parse_success_rate(metrics_path: Path) -> float:
    """Pull ``success_rate`` out of the metrics dict in a metrics.txt file.

    The file format ends with ``metrics: { ... 'success_rate': N.N, ...}``
    followed by ``episode_successes: array([...])``. Some Reacher / Cube
    runs append a second balanced dict (``ROBUST_CEM`` JSON), so we keep
    only candidate dicts that contain a ``success_rate`` key. We strip
    ``array(...)`` literals before ``ast.literal_eval``.
    """
    text = metrics_path.read_text(errors="replace")
    last = None
    i, n = 0, len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        for j in range(i, n):
            c = text[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[i:j + 1]
                    if "success_rate" in candidate:
                        last = candidate
                    i = j + 1
                    break
        else:
            break
    if last is None:
        raise ValueError(f"no success_rate dict in {metrics_path}")
    cleaned = _ARRAY_RE.sub("None", last)
    d = ast.literal_eval(cleaned)
    return float(d["success_rate"])


def _aggregate_seeds(values: list[float]) -> dict:
    """Compute the per-condition stats, matching the LeWM convention.

    Uses population std (ddof=0) over the seed values, identical to
    ``statistics.pstdev`` and to the LeWM canonical JSON entries.
    """
    return {
        "n": len(values),
        "mean": sum(values) / len(values) if values else float("nan"),
        "std": statistics.pstdev(values) if len(values) >= 2 else 0.0,
        "values": [round(v, 4) for v in values],
    }


def _std_key_from_subdir(subdir: str) -> str | None:
    """Map a PLDM ckpt subdir name to its training std_max as a string.

    The repo convention is ``noise_0toNNN`` ↔ ``std_max = 0.NN`` (verified
    against the per-ckpt training ``config.yaml`` --- e.g. ``0to001 ↔ 0.01``,
    ``0to008 ↔ 0.08``).

    ``..._pldm_baseline``          → ``"0.0"``
    ``..._pldm_noise_0to001_p1``           → ``"0.01"``
    ``..._pldm_noise_0to004_p1_20260522``  → ``"0.04"``
    """
    if subdir.endswith("_pldm_baseline"):
        return "0.0"
    m = re.search(r"_pldm_noise_0to(\d+)_p1(?:_\d{8})?$", subdir)
    if not m:
        return None
    # Drop leading zeros, then format as 0.NN
    n = int(m.group(1))
    return f"0.{n:02d}"


def _collect_one_ckpt(eval_results: Path) -> dict[str, dict] | None:
    """Read per-seed metrics for every condition in one ckpt's eval_results.

    Returns ``None`` if not even the clean condition has full 3-seed coverage.
    """
    out: dict[str, dict] = {}
    for cond in CONDITIONS:
        values: list[float] = []
        for s in SEEDS:
            fp = eval_results / f"{cond}_seed{s}_metrics.txt"
            if not fp.exists():
                continue
            try:
                values.append(_parse_success_rate(fp))
            except ValueError:
                continue
        if values:
            out[cond] = _aggregate_seeds(values)
    if "clean" not in out or out["clean"]["n"] < len(SEEDS):
        return None
    return out


def build(root: Path, out_path: Path) -> dict:
    """Walk all four tasks' PLDM dirs and emit the canonical JSON."""
    canonical: dict[str, dict[str, dict]] = {t: {} for t in TASKS}
    missing: list[str] = []
    for task in TASKS:
        env_dir = root / TASK_TO_ENV[task]
        ckpt_root = env_dir / "ckpt"
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
            eval_results = ckpt_dir / "eval_results"
            if not eval_results.is_dir():
                missing.append(f"{task} {ckpt_dir.name}: no eval_results")
                continue
            metrics = _collect_one_ckpt(eval_results)
            if metrics is None:
                missing.append(f"{task} {ckpt_dir.name}: incomplete clean")
                continue
            previous = canonical[task].get(std_key)
            if previous is not None and ckpt_dir.name <= previous["subdir"]:
                continue
            canonical[task][std_key] = {
                "path": str(ckpt_dir.resolve()),
                "subdir": ckpt_dir.name,
                "metrics": metrics,
            }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(canonical, indent=2, sort_keys=True))
    return {"canonical": canonical, "missing": missing}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        default="/home/ag/dataset/ag_data/data/world_model/quentinll",
        help="dataset root that contains lewm-{cube,pusht,reacher,tworooms}/",
    )
    ap.add_argument(
        "--out",
        default="assets/paper1_data/canonical_evals_pldm_20260522.json",
        help="output JSON path (relative to repo root unless absolute)",
    )
    args = ap.parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    root = Path(args.root)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = repo_root / out_path
    report = build(root, out_path)
    print(f"wrote {out_path}")
    for task, by_std in report["canonical"].items():
        print(f"  {task}: {len(by_std)} ckpts ({sorted(by_std.keys())})")
    if report["missing"]:
        print("\nMissing / incomplete:")
        for line in report["missing"]:
            print(f"  - {line}")


if __name__ == "__main__":
    main()
