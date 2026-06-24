"""
regen_diagnostics_summary.py — Rebuild diagnostics_summary.json from the
already-saved per-probe JSON artifacts in a diagnostics dir, without
re-running probes and without needing torch/pandas/sklearn.

Inputs (any subset; missing files just yield missing fields):
    geometry_summary.json
    latent_geometry_summary.json
    noise_sensitivity.json
    predictor_sensitivity.json
    task_resolution.json
    latent_noise_sensitivity.json
    action_effect.json

Output (overwrites):
    diagnostics_summary.json    list[dict] keyed by 'model'

Use to repair files damaged by older skip-some-probes runs of
run_full_diagnostics (now fixed to merge in place; this tool exists to
heal already-damaged files post-hoc).

Usage:
    python -m tools.repr_analysis.regen_diagnostics_summary <dir> [<dir> ...]
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any


def _load(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _is_nan(x):
    return isinstance(x, float) and math.isnan(x)


def regen(d_dir: Path) -> dict:
    geom = _load(d_dir / "geometry_summary.json") or []
    lgeom = _load(d_dir / "latent_geometry_summary.json") or []
    noise = _load(d_dir / "noise_sensitivity.json") or []
    predictor = _load(d_dir / "predictor_sensitivity.json") or []
    resolution = _load(d_dir / "task_resolution.json") or []
    latent_noise = _load(d_dir / "latent_noise_sensitivity.json") or []
    action_effect = _load(d_dir / "action_effect.json") or []

    by_label: dict[str, dict[str, Any]] = {}

    # Old full-suite runs sometimes labeled the rollup with the full ckpt subdir
    # (e.g. 'tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_dim64') while
    # later backfills used short canonical labels ('SWM-base'). Normalize so
    # both contributions land in the same entry.
    import re as _re

    def _norm(label: str) -> str:
        if not isinstance(label, str):
            return label
        low = label.lower()
        m = _re.search(r"_(lewm|swm)(?:_[\w]+?)?_noise_(0to\d+)(?:_p\d+)?", low)
        if m:
            method = "LeWM" if m.group(1) == "lewm" else "SWM"
            return f"{method}-{m.group(2)}-p1"
        if "swm" in low and "noise" not in low:
            return "SWM-base"
        if low.endswith("_lewm") or "lewm_20260430" in low or _re.search(r"_lewm$", low):
            return "LeWM-base"
        return label

    # ---- geometry_summary (precomputed by summarize_noise_geometry) ----
    for r in geom:
        label = _norm(r.get("model"))
        if not label:
            continue
        by_label.setdefault(label, {"model": label})
        for src, dst in [
            ("robust_radius_std", "noise_robust_radius_std"),
            ("noise_angle_slope_deg_per_std", "noise_angle_slope_deg_per_std"),
            ("clean_nn_cos_dist_median", "clean_nn_cos_dist_median"),
            ("clean_effective_rank", "clean_effective_rank"),
            ("geometry_flag", "geometry_flag"),
            ("recommendation", "recommendation"),
        ]:
            v = r.get(src)
            if v is not None and not _is_nan(v):
                by_label[label][dst] = v

    # ---- noise_sensitivity: cka at max std, goal scope ----
    for r in noise:
        if r.get("frame_scope") != "goal":
            continue
        label = _norm(r.get("model"))
        if not label:
            continue
        by_label.setdefault(label, {"model": label})
        std = r.get("std", -1.0)
        if std > by_label[label].get("_cka_max_std", -1.0):
            by_label[label]["_cka_max_std"] = std
            cka = r.get("cka_linear_clean_vs_noisy")
            if cka is not None and not _is_nan(cka):
                by_label[label]["cka_linear_at_max_std"] = cka

    # ---- predictor_sensitivity: rollout drift at largest std ----
    for r in predictor:
        label = _norm(r.get("model"))
        if not label:
            continue
        by_label.setdefault(label, {"model": label})
        std = r.get("std", -1.0)
        if std > by_label[label].get("_pred_max_std", -1.0):
            by_label[label]["_pred_max_std"] = std
            for T in (8, 4, 2, 1):
                key = f"rollout_T{T}_l2_median"
                v = r.get(key)
                if v is not None and not _is_nan(v):
                    by_label[label][f"predictor_rollout_T{T}_l2"] = v
                    by_label[label]["predictor_target_to_nn_cos_ratio_at_max_std"] = (
                        r.get("target_to_nn_cos_ratio")
                    )
                    break

    # ---- task_resolution ----
    for r in resolution:
        label = _norm(r.get("model"))
        if not label:
            continue
        by_label.setdefault(label, {"model": label})
        for k in ("transition_resolution_ratio_cos", "transition_resolution_ratio_l2",
                 "id_probe_r2", "id_probe_r2_min", "lidar_rank"):
            v = r.get(k)
            if v is not None and not _is_nan(v):
                by_label[label][k] = v

    # ---- latent_geometry_summary (already split by frame_scope inside) ----
    for r in lgeom:
        label = _norm(r.get("model"))
        if not label:
            continue
        by_label.setdefault(label, {"model": label})
        scope = r.get("frame_scope", "")
        if scope == "goal":
            v = r.get("cost_surface_slope_z")
            if v is not None and not _is_nan(v):
                by_label[label]["latent_cost_surface_slope_z"] = v
            v = r.get("noise_geometry")
            if v is not None:
                by_label[label]["latent_noise_geometry"] = v
        elif scope == "history":
            for src, dst in [
                ("robust_radius_z", "latent_robust_radius_z"),
                ("rollout_angle_slope_deg_per_std_z", "latent_rollout_angle_slope_per_std_z"),
                ("rollout_l2_slope_per_std_z", "latent_rollout_l2_slope_per_std_z"),
            ]:
                v = r.get(src)
                if v is not None and not _is_nan(v):
                    by_label[label][dst] = v

    # ---- latent_noise_sensitivity: history-scope rollout drift at max std ----
    for r in latent_noise:
        if r.get("frame_scope") != "history":
            continue
        label = _norm(r.get("model"))
        if not label:
            continue
        by_label.setdefault(label, {"model": label})
        std = r.get("std", -1.0)
        if std > by_label[label].get("_lat_pred_max_std", -1.0):
            by_label[label]["_lat_pred_max_std"] = std
            for T in (8, 4, 2, 1):
                key = f"rollout_T{T}_l2_median"
                v = r.get(key)
                if v is not None and not _is_nan(v):
                    by_label[label][f"latent_predictor_rollout_T{T}_l2_history"] = v
                    break

    # ---- action_effect ----
    for r in action_effect:
        label = _norm(r.get("model"))
        if not label:
            continue
        by_label.setdefault(label, {"model": label})
        for src, dst in [
            ("mean_pred_shift_norm", "action_mean_pred_shift_norm"),
            ("action_perturb_pred_shift_corr", "action_perturb_pred_shift_corr"),
            ("interpolation_endpoint_shift", "action_interpolation_endpoint_shift"),
            ("interpolation_monotonicity", "action_interpolation_monotonicity"),
        ]:
            v = r.get(src)
            if v is not None and not _is_nan(v):
                by_label[label][dst] = v

    # Strip bookkeeping
    for s in by_label.values():
        for k in list(s.keys()):
            if k.startswith("_"):
                s.pop(k)

    rollup = list(by_label.values())
    out = d_dir / "diagnostics_summary.json"
    with out.open("w") as f:
        json.dump(rollup, f, indent=2)
    print(f"[regen] {out}  ({len(rollup)} model entries, "
          f"{sum(len(r) for r in rollup)} fields)")
    return {"path": str(out), "n_entries": len(rollup)}


def main(argv: list[str]) -> None:
    if not argv:
        print("usage: python -m tools.repr_analysis.regen_diagnostics_summary <diagnostics_dir> [...]")
        sys.exit(1)
    for raw in argv:
        d = Path(raw).resolve()
        if not d.is_dir():
            print(f"[skip] not a directory: {d}")
            continue
        regen(d)


if __name__ == "__main__":
    main(sys.argv[1:])
