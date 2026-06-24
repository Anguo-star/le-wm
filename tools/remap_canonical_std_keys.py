"""One-shot fixer: remap the std_max keys in the LeWM canonical JSON files
from the historical (×10-too-small) labels to the actual training std_max.

Background: the LeWM canonical_evals_20260517.json and
canonical_diagnostics_20260517.json files were generated under the wrong
assumption that the directory suffix ``noise_0toNNN`` meant ``std_max =
0.00N``. The per-ckpt training ``config.yaml`` shows the actual values are
``std_max = 0.NN`` (e.g. ``0to001 ↔ 0.01``, ``0to008 ↔ 0.08``). This script
remaps the keys in place; the underlying eval / diagnostic values are
unchanged.

Mapping:
    ``"0.0"``  → ``"0.0"``       (unchanged)
    ``"0.001"`` → ``"0.01"``
    ``"0.002"`` → ``"0.02"``
    ...
    ``"0.008"`` → ``"0.08"``

Run::

    python -m tools.remap_canonical_std_keys
"""
from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "assets" / "paper1_data"


_KEY_MAP = {f"0.00{i}": f"0.0{i}" for i in range(1, 10)}
_KEY_MAP["0.0"] = "0.0"


def _remap(d: dict) -> dict:
    """Return a copy of ``d`` whose std-max-shaped string keys are remapped.

    Any key not in the table is preserved as-is.
    """
    return {_KEY_MAP.get(k, k): v for k, v in d.items()}


def fix_evals(path: Path) -> int:
    """Rewrite canonical_evals*.json with corrected keys per task."""
    data = json.loads(path.read_text())
    changed = 0
    for task in ("TwoRoom", "PushT", "Reacher", "Cube"):
        if task not in data:
            continue
        before = sorted(data[task].keys())
        data[task] = _remap(data[task])
        after = sorted(data[task].keys())
        if before != after:
            changed += 1
    path.write_text(json.dumps(data, indent=2, sort_keys=True))
    return changed


def fix_diagnostics(path: Path) -> int:
    """Rewrite canonical_diagnostics*.json with corrected keys per task."""
    data = json.loads(path.read_text())
    changed = 0
    pmbt = data.get("predictor_metrics_by_task", {})
    for task in ("TwoRoom", "PushT", "Reacher", "Cube"):
        if task not in pmbt:
            continue
        before = sorted(pmbt[task].keys())
        pmbt[task] = _remap(pmbt[task])
        after = sorted(pmbt[task].keys())
        if before != after:
            changed += 1
    # table3 representative diagnostics: each {metric: {base: ..., representative: ...}}
    # no std keys at the top level, but the 'representative' row carries a
    # representative_std_max field. Update if present.
    t3 = data.get("table3_representative_diagnostics", {})
    for k, v in t3.items():
        if not isinstance(v, dict):
            continue
        for entry_key, entry in v.items():
            if not isinstance(entry, dict):
                continue
            old = entry.get("representative_std_max")
            if old in _KEY_MAP and old != _KEY_MAP[old]:
                entry["representative_std_max"] = _KEY_MAP[old]
                changed += 1
    data["predictor_metrics_by_task"] = pmbt
    path.write_text(json.dumps(data, indent=2, sort_keys=True))
    return changed


def main() -> None:
    targets = [
        ("evals", DATA_DIR / "canonical_evals_20260517.json"),
        ("diagnostics", DATA_DIR / "canonical_diagnostics_20260517.json"),
    ]
    for label, p in targets:
        if not p.exists():
            print(f"  skip (missing): {p}")
            continue
        fn = fix_evals if label == "evals" else fix_diagnostics
        n = fn(p)
        print(f"  remapped {n} task entries in {p.relative_to(REPO_ROOT)}")
    print("done.")


if __name__ == "__main__":
    main()
