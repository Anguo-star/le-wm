"""
run_full_diagnostics.py — Single entry that runs noise sensitivity,
predictor sensitivity, and task resolution diagnostics for one or more
checkpoints, and saves a unified output directory.

This is the script `run_trainer.sh` calls after training. It produces a
self-contained diagnostic report per checkpoint that downstream
correlation analysis (P0.4 / P0.7) consumes.

Layout (per --save-dir):

    <save_dir>/
        noise_sensitivity.csv / .json
        geometry_summary.csv  / .json
        noise_ratio_curve_goal.png  (if --plot)
        noise_angle_curve_goal.png  (if --plot)
        geometry_tradeoff_goal.png  (if --plot)
        predictor_sensitivity.csv / .json
        task_resolution.csv / .json
        latent_noise_sensitivity.csv / .json   (P5 — encoder-decoupled)
        latent_geometry_summary.csv / .json     (P5)
        latent_noise_ratio_curve_goal.png       (if --plot)
        latent_noise_angle_curve_goal.png       (if --plot)
        action_effect.csv / .json
        diagnostics_summary.json   (one-line per-checkpoint roll-up)

CLI mirrors `noise_sensitivity.py` for backwards compatibility:

    python -m tools.repr_analysis.run_full_diagnostics \
        --model swm=/path/to/model_object.ckpt \
        --dataset tworoom \
        --stds 0.0 0.005 0.01 0.02 0.03 0.05 \
        --save-dir <results_dir>/diagnostics \
        --plot
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch

from tools.repr_analysis.analyze_repr import to_serializable
from tools.repr_analysis.noise_sensitivity import (
    format_noise_table,
    plot_geometry_tradeoff,
    plot_noise_curves,
    run_noise_sensitivity,
    summarize_noise_geometry,
)
from tools.repr_analysis.predictor_sensitivity import (
    format_predictor_table,
    run_predictor_sensitivity,
)
from tools.repr_analysis.task_resolution import (
    format_resolution_table,
    run_task_resolution,
)
from tools.repr_analysis.action_effect import (
    format_action_effect_table,
    run_action_effect,
)
from tools.repr_analysis.latent_noise_sensitivity import (
    format_latent_noise_table,
    plot_latent_noise_curves,
    run_latent_noise_sensitivity,
    summarize_latent_noise_geometry,
)


def _parse_model_specs(specs: Sequence[str]) -> Dict[str, str]:
    models: Dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Model spec must be label=/path/to/ckpt, got: {spec}")
        label, ckpt = spec.split("=", 1)
        models[label.strip()] = ckpt.strip()
    return models


def _save_rows(save_dir: Path, name: str, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with (save_dir / f"{name}.json").open("w") as f:
        json.dump(to_serializable(list(rows)), f, indent=2)
    with (save_dir / f"{name}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summarize_noise_to_predictor_to_resolution(
    *,
    noise_rows,
    predictor_rows,
    resolution_rows,
    latent_noise_rows=None,
    action_effect_rows=None,
) -> list[Dict[str, Any]]:
    """One-line per-checkpoint roll-up combining the diagnostic families."""
    summary: Dict[str, Dict[str, Any]] = {}

    geometry = summarize_noise_geometry(noise_rows, frame_scope="goal")
    if not geometry.empty:
        for _, row in geometry.iterrows():
            label = row["model"]
            summary.setdefault(label, {"model": label})
            summary[label].update({
                "noise_robust_radius_std": float(row["robust_radius_std"]),
                "noise_angle_slope_deg_per_std": float(row["noise_angle_slope_deg_per_std"]),
                "clean_nn_cos_dist_median": float(row["clean_nn_cos_dist_median"]),
                "clean_effective_rank": float(row.get("clean_effective_rank", float("nan"))),
                "geometry_flag": str(row.get("geometry_flag", "")),
                "recommendation": str(row.get("recommendation", "")),
            })

    # CKA at the largest std we measured (worst-case alignment loss)
    for r in noise_rows:
        if r["frame_scope"] != "goal":
            continue
        label = r["model"]
        summary.setdefault(label, {"model": label})
        cka = r.get("cka_linear_clean_vs_noisy", float("nan"))
        std = r["std"]
        cur_max = summary[label].get("_cka_max_std", -1.0)
        if std > cur_max:
            summary[label]["_cka_max_std"] = float(std)
            summary[label]["cka_linear_at_max_std"] = float(cka)

    # Predictor: rollout drift @ T=8 (or smallest available) with largest std
    for r in predictor_rows:
        label = r["model"]
        summary.setdefault(label, {"model": label})
        std = r["std"]
        cur_max = summary[label].get("_pred_max_std", -1.0)
        if std > cur_max:
            summary[label]["_pred_max_std"] = float(std)
            for T in (8, 4, 2, 1):
                key = f"rollout_T{T}_l2_median"
                if r.get(key) is not None and r[key] == r[key]:  # not NaN
                    summary[label][f"predictor_rollout_T{T}_l2"] = float(r[key])
                    summary[label]["predictor_target_to_nn_cos_ratio_at_max_std"] = float(
                        r.get("target_to_nn_cos_ratio", float("nan"))
                    )
                    break

    # Task resolution
    for r in resolution_rows:
        label = r["model"]
        summary.setdefault(label, {"model": label})
        summary[label].update({
            "transition_resolution_ratio_cos": float(r.get("transition_resolution_ratio_cos", float("nan"))),
            "transition_resolution_ratio_l2": float(r.get("transition_resolution_ratio_l2", float("nan"))),
            "id_probe_r2": float(r.get("id_probe_r2", float("nan"))),
            "id_probe_r2_min": float(r.get("id_probe_r2_min", float("nan"))),
            "lidar_rank": float(r.get("lidar_rank", float("nan"))),
        })

    # Latent-noise (P5): encoder-decoupled robust radius / cost slope.
    if latent_noise_rows:
        # Goal scope: cost surface slope (perturbing goal latent affects cost)
        try:
            latent_geom_goal = summarize_latent_noise_geometry(
                latent_noise_rows, frame_scope="goal"
            )
        except Exception:
            latent_geom_goal = None
        if latent_geom_goal is not None and not latent_geom_goal.empty:
            for _, row in latent_geom_goal.iterrows():
                label = row["model"]
                summary.setdefault(label, {"model": label})
                summary[label].update({
                    "latent_cost_surface_slope_z": float(
                        row.get("cost_surface_slope_z", float("nan"))
                    ),
                    "latent_noise_geometry": str(row.get("noise_geometry", "")),
                })
        # History scope: robust radius and predictor slopes (rollout drift is
        # meaningful when history tokens are perturbed; goal scope gives NaN
        # because single-step predictor does not consume the goal token).
        try:
            latent_geom_hist = summarize_latent_noise_geometry(
                latent_noise_rows, frame_scope="history"
            )
        except Exception:
            latent_geom_hist = None
        if latent_geom_hist is not None and not latent_geom_hist.empty:
            for _, row in latent_geom_hist.iterrows():
                label = row["model"]
                summary.setdefault(label, {"model": label})
                summary[label].update({
                    "latent_robust_radius_z": float(row.get("robust_radius_z", float("nan"))),
                    # NOTE: predictor_angle_slope_deg_per_std_z and predictor_l2_slope_per_std_z
                    # are excluded because _open_loop_target_shift mixes clean/noisy windows
                    # when only a subset of tokens are perturbed, making the median uninformative.
                    # Use rollout_angle/l2_slope_per_std_z instead (autoregressive init from
                    # the perturbed slice, so all steps are genuinely noisy).
                    "latent_rollout_angle_slope_per_std_z": float(
                        row.get("rollout_angle_slope_deg_per_std_z", float("nan"))
                    ),
                    "latent_rollout_l2_slope_per_std_z": float(
                        row.get("rollout_l2_slope_per_std_z", float("nan"))
                    ),
                })
        # Pick predictor rollout drift @ T=8 from the largest std (worst-case
        # encoder-decoupled smoothness), parallel to predictor_sensitivity.
        for r in latent_noise_rows:
            if r.get("frame_scope") != "history":
                continue
            label = r["model"]
            summary.setdefault(label, {"model": label})
            std = r["std"]
            cur_max = summary[label].get("_lat_pred_max_std", -1.0)
            if std > cur_max:
                summary[label]["_lat_pred_max_std"] = float(std)
                for T in (8, 4, 2, 1):
                    key = f"rollout_T{T}_l2_median"
                    if r.get(key) is not None and r[key] == r[key]:
                        summary[label][f"latent_predictor_rollout_T{T}_l2_history"] = float(r[key])
                        break

    # Action-effect probe: action perturbation -> predictor shift.
    if action_effect_rows:
        for r in action_effect_rows:
            label = r["model"]
            summary.setdefault(label, {"model": label})
            summary[label].update({
                "action_mean_pred_shift_norm": float(
                    r.get("mean_pred_shift_norm", float("nan"))
                ),
                "action_perturb_pred_shift_corr": float(
                    r.get("action_perturb_pred_shift_corr", float("nan"))
                ),
                "action_interpolation_endpoint_shift": float(
                    r.get("interpolation_endpoint_shift", float("nan"))
                ),
                "action_interpolation_monotonicity": float(
                    r.get("interpolation_monotonicity", float("nan"))
                ),
            })

    # Drop bookkeeping fields
    for s in summary.values():
        for k in list(s.keys()):
            if k.startswith("_"):
                s.pop(k)

    return list(summary.values())


def run_full_diagnostics(
    *,
    models: Mapping[str, str],
    dataset: str = "tworoom",
    stds: Sequence[float] = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08),
    rollout_steps: Sequence[int] = (1, 2, 4, 8),
    state_key: str | None = None,
    n_sequences: int = 256,
    future_steps: int = 8,
    frameskip: int = 1,
    img_size: int = 224,
    embedding_space: str | None = None,
    seed: int = 3072,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_dir: str | Path | None = None,
    plot: bool = False,
    skip_noise: bool = False,
    skip_predictor: bool = False,
    skip_resolution: bool = False,
    skip_latent_noise: bool = False,
    skip_action_effect: bool = False,
    predictor_history_noise_only: bool = True,
    latent_noise_geometry: str = "auto",
    latent_noise_std_mode: str = "relative",
    latent_noise_n_samples: int = 1,
    action_effect_n_trials: int = 128,
    action_effect_interp_steps: int = 16,
    action_effect_perturb_scale: float = 0.5,
    corruption_type: str = "gaussian_noise",
    log=print,
) -> Dict[str, Any]:
    """Run the full diagnostic suite.

    ``corruption_type`` selects the family of pixel-space perturbation
    used by the noise / predictor probes. Defaults to ``gaussian_noise``
    --- the original behaviour --- so existing callers see no change.
    Setting it to ``gaussian_blur`` or ``resize`` swaps the injection
    class; the ``stds`` sequence is reinterpreted as the corresponding
    magnitude (kernel size in pixels, or downscale factor). The
    latent-noise probe is intrinsically z-space Gaussian and is
    auto-skipped when ``corruption_type != 'gaussian_noise'`` (it would
    be ill-defined otherwise).
    """
    # (Module-level docstring above; per-call helper docstring follows.)
    # This is the notebook/API counterpart of the CLI. It returns all raw rows
    # plus formatted tables and the one-line roll-up used by P0 correlation work.
    # When `save_dir` is provided it also writes the same artifacts as the CLI.
    if corruption_type not in ("gaussian_noise", "gaussian_blur", "resize"):
        raise ValueError(
            f"Unsupported corruption_type='{corruption_type}'. "
            "Expected one of: gaussian_noise, gaussian_blur, resize."
        )
    if corruption_type != "gaussian_noise" and not skip_latent_noise:
        if log is not None:
            log(
                "[diagnostics] note: latent_noise_sensitivity probes z-space "
                "Gaussian noise and is not defined for "
                f"corruption_type={corruption_type!r}; skipping it."
            )
        skip_latent_noise = True

    output_dir = Path(save_dir) if save_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    common = dict(
        models=models,
        dataset=dataset,
        state_key=state_key,
        n_sequences=n_sequences,
        future_steps=future_steps,
        frameskip=frameskip,
        img_size=img_size,
        embedding_space=embedding_space,
        seed=seed,
        device=device,
    )

    noise_rows: list = []
    predictor_rows: list = []
    resolution_rows: list = []
    latent_noise_rows: list = []
    action_effect_rows: list = []
    geometry = None
    latent_geometry = None

    if not skip_noise:
        if log is not None:
            log(f"==[diagnostics] running noise_sensitivity (corruption_type={corruption_type}) ==")
        noise_rows = run_noise_sensitivity(stds=stds, corruption_type=corruption_type, **common)
        if output_dir is not None:
            _save_rows(output_dir, "noise_sensitivity", noise_rows)
        try:
            geometry = summarize_noise_geometry(noise_rows, frame_scope="goal")
            if output_dir is not None:
                geometry.to_csv(output_dir / "geometry_summary.csv", index=False)
                with (output_dir / "geometry_summary.json").open("w") as f:
                    json.dump(to_serializable(geometry.to_dict(orient="records")), f, indent=2)
        except Exception as e:
            if log is not None:
                log(f"[diagnostics] geometry summary skipped: {e}")
        try:
            if log is not None:
                log(format_noise_table(noise_rows, frame_scope="goal").to_string(index=False))
        except Exception as e:
            if log is not None:
                log(f"[diagnostics] noise table format skipped: {e}")
        if plot and output_dir is not None:
            try:
                fig = plot_noise_curves(noise_rows, frame_scope="goal")
                fig.savefig(output_dir / "noise_ratio_curve_goal.png", dpi=200, bbox_inches="tight")
                fig = plot_noise_curves(
                    noise_rows,
                    frame_scope="goal",
                    metric="noise_angle_deg_median",
                )
                fig.savefig(output_dir / "noise_angle_curve_goal.png", dpi=200, bbox_inches="tight")
                fig = plot_geometry_tradeoff(
                    summarize_noise_geometry(noise_rows, frame_scope="goal")
                )
                fig.savefig(output_dir / "geometry_tradeoff_goal.png", dpi=200, bbox_inches="tight")
            except Exception as e:
                if log is not None:
                    log(f"[diagnostics] plotting skipped: {e}")

    if not skip_predictor:
        if log is not None:
            log(f"==[diagnostics] running predictor_sensitivity (corruption_type={corruption_type}) ==")
        predictor_rows = run_predictor_sensitivity(
            stds=stds,
            rollout_steps=rollout_steps,
            history_noise_only=predictor_history_noise_only,
            corruption_type=corruption_type,
            **common,
        )
        if output_dir is not None:
            _save_rows(output_dir, "predictor_sensitivity", predictor_rows)
        try:
            if log is not None:
                log(format_predictor_table(predictor_rows).to_string(index=False))
        except Exception as e:
            if log is not None:
                log(f"[diagnostics] predictor table format skipped: {e}")

    if not skip_resolution:
        if log is not None:
            log("==[diagnostics] running task_resolution ==")
        resolution_rows = run_task_resolution(**common)
        if output_dir is not None:
            _save_rows(output_dir, "task_resolution", resolution_rows)
        try:
            if log is not None:
                log(format_resolution_table(resolution_rows).to_string(index=False))
        except Exception as e:
            if log is not None:
                log(f"[diagnostics] resolution table format skipped: {e}")

    if not skip_latent_noise:
        if log is not None:
            log("==[diagnostics] running latent_noise_sensitivity ==")
        latent_noise_rows = run_latent_noise_sensitivity(
            stds=stds,
            rollout_steps=rollout_steps,
            noise_geometry=latent_noise_geometry,
            std_mode=latent_noise_std_mode,
            n_noise_samples=latent_noise_n_samples,
            **common,
        )
        if output_dir is not None:
            _save_rows(output_dir, "latent_noise_sensitivity", latent_noise_rows)
        try:
            latent_geometry = summarize_latent_noise_geometry(
                latent_noise_rows, frame_scope="goal"
            )
            if output_dir is not None and latent_geometry is not None:
                latent_geometry.to_csv(
                    output_dir / "latent_geometry_summary.csv", index=False
                )
                with (output_dir / "latent_geometry_summary.json").open("w") as f:
                    json.dump(
                        to_serializable(latent_geometry.to_dict(orient="records")),
                        f, indent=2,
                    )
        except Exception as e:
            if log is not None:
                log(f"[diagnostics] latent geometry summary skipped: {e}")
        try:
            if log is not None:
                log(format_latent_noise_table(latent_noise_rows, frame_scope="goal").to_string(index=False))
        except Exception as e:
            if log is not None:
                log(f"[diagnostics] latent noise table format skipped: {e}")
        if plot and output_dir is not None:
            try:
                fig = plot_latent_noise_curves(latent_noise_rows, frame_scope="goal")
                fig.savefig(output_dir / "latent_noise_ratio_curve_goal.png",
                            dpi=200, bbox_inches="tight")
                fig = plot_latent_noise_curves(
                    latent_noise_rows, frame_scope="goal",
                    metric="target_angle_deg_median",
                )
                fig.savefig(output_dir / "latent_noise_angle_curve_goal.png",
                            dpi=200, bbox_inches="tight")
            except Exception as e:
                if log is not None:
                    log(f"[diagnostics] latent noise plotting skipped: {e}")

    if not skip_action_effect:
        if log is not None:
            log("==[diagnostics] running action_effect ==")
        action_effect_rows = run_action_effect(
            n_trials=action_effect_n_trials,
            interp_steps=action_effect_interp_steps,
            perturb_scale=action_effect_perturb_scale,
            **common,
        )
        if output_dir is not None:
            _save_rows(output_dir, "action_effect", action_effect_rows)
        try:
            if log is not None:
                log(format_action_effect_table(action_effect_rows).to_string(index=False))
        except Exception as e:
            if log is not None:
                log(f"[diagnostics] action_effect table format skipped: {e}")

    rollup = _summarize_noise_to_predictor_to_resolution(
        noise_rows=noise_rows,
        predictor_rows=predictor_rows,
        resolution_rows=resolution_rows,
        latent_noise_rows=latent_noise_rows,
        action_effect_rows=action_effect_rows,
    )
    if output_dir is not None:
        # Merge with any pre-existing diagnostics_summary.json instead of
        # overwriting. When the user runs only a subset of sub-probes (e.g.
        # to back-fill action_effect alone), preserving fields from earlier
        # full-suite runs prevents data loss.
        summary_path = output_dir / "diagnostics_summary.json"
        merged = to_serializable(rollup)
        if summary_path.is_file():
            try:
                prior = json.loads(summary_path.read_text())
            except Exception:
                prior = None
            if prior:
                # Both prior and merged are list[dict] keyed by 'model'.
                if isinstance(prior, list) and isinstance(merged, list):
                    by_label = {r.get("model"): dict(r) for r in prior if isinstance(r, dict)}
                    for r in merged:
                        if not isinstance(r, dict):
                            continue
                        label = r.get("model")
                        if label in by_label:
                            by_label[label].update(r)
                        else:
                            by_label[label] = dict(r)
                    merged = list(by_label.values())
        with summary_path.open("w") as f:
            json.dump(merged, f, indent=2)

    try:
        noise_table = format_noise_table(noise_rows, frame_scope="goal") if noise_rows else None
    except Exception:
        noise_table = None
    try:
        predictor_table = format_predictor_table(predictor_rows) if predictor_rows else None
    except Exception:
        predictor_table = None
    try:
        resolution_table = format_resolution_table(resolution_rows) if resolution_rows else None
    except Exception:
        resolution_table = None
    try:
        latent_noise_table = (
            format_latent_noise_table(latent_noise_rows, frame_scope="goal")
            if latent_noise_rows else None
        )
    except Exception:
        latent_noise_table = None
    try:
        action_effect_table = (
            format_action_effect_table(action_effect_rows)
            if action_effect_rows else None
        )
    except Exception:
        action_effect_table = None

    if log is not None and output_dir is not None:
        log(f"\n[diagnostics] saved unified output to: {output_dir}")
        log("[diagnostics] per-checkpoint roll-up:")
        for s in rollup:
            log(json.dumps(s, indent=2, default=str))

    return {
        "noise_rows": noise_rows,
        "geometry_summary": geometry,
        "predictor_rows": predictor_rows,
        "resolution_rows": resolution_rows,
        "latent_noise_rows": latent_noise_rows,
        "latent_geometry_summary": latent_geometry,
        "action_effect_rows": action_effect_rows,
        "diagnostics_summary": rollup,
        "noise_table": noise_table,
        "predictor_table": predictor_table,
        "resolution_table": resolution_table,
        "latent_noise_table": latent_noise_table,
        "action_effect_table": action_effect_table,
        "save_dir": output_dir,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run full latent-geometry diagnostic suite.")
    p.add_argument("--model", action="append", required=True,
                   help="Model spec as label=/path/to/model_object.ckpt. Repeat for comparisons.")
    p.add_argument("--dataset", default="tworoom")
    p.add_argument("--stds", type=float, nargs="+",
                   default=[0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08])
    p.add_argument("--rollout-steps", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--state-key", default=None)
    p.add_argument("--n-sequences", type=int, default=256)
    p.add_argument("--future-steps", type=int, default=8)
    p.add_argument("--frameskip", type=int, default=1)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--embedding-space", default=None, choices=[None, "raw", "normalized"])
    p.add_argument("--seed", type=int, default=3072)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-dir", required=True)
    p.add_argument("--plot", action="store_true", help="Save diagnostic PNG plots.")
    p.add_argument("--skip-noise", action="store_true")
    p.add_argument("--skip-predictor", action="store_true")
    p.add_argument("--skip-resolution", action="store_true")
    p.add_argument("--skip-latent-noise", action="store_true",
                   help="Skip P5 latent-space noise probing.")
    p.add_argument("--skip-action-effect", action="store_true",
                   help="Skip action-effect probe (action perturbation -> predictor shift).")
    p.add_argument("--action-effect-n-trials", type=int, default=128)
    p.add_argument("--action-effect-interp-steps", type=int, default=16)
    p.add_argument("--action-effect-perturb-scale", type=float, default=0.5)
    p.add_argument("--predictor-history-noise-only", action="store_true", default=True,
                   help="Predictor diagnostic adds noise only to history frames (default).")
    p.add_argument("--predictor-full-noise",
                   dest="predictor_history_noise_only", action="store_false",
                   help="Predictor diagnostic adds noise to all frames including goal.")
    p.add_argument("--latent-noise-geometry", default="auto",
                   choices=["auto", "ambient", "tangent"],
                   help="Latent-noise geometry. `auto` uses tangent for SWM normalized space and ambient otherwise.")
    p.add_argument("--latent-noise-std-mode", default="relative",
                   choices=["relative", "absolute"],
                   help="`relative` scales std by per-token clean norm "
                        "(comparable across LeWM/SWM).")
    p.add_argument("--latent-noise-n-samples", type=int, default=1,
                   help="Independent noise samples averaged per (std, scope).")
    p.add_argument("--corruption-type", default="gaussian_noise",
                   choices=["gaussian_noise", "gaussian_blur", "resize"],
                   help="Pixel-space corruption family for the noise / predictor "
                        "probes. The ``stds`` list is reinterpreted as the family's "
                        "magnitude (kernel_size in px for gaussian_blur; factor for "
                        "resize). Latent-noise probe is auto-skipped when "
                        "corruption_type is not gaussian_noise.")
    return p


def main():
    args = build_parser().parse_args()
    # Re-route the save dir when corruption_type != gaussian_noise so
    # blur and resize diagnostic outputs do not overwrite the canonical
    # gaussian_noise diagnostic JSONs. The caller can still override by
    # passing a save_dir that already carries the desired suffix.
    save_dir = args.save_dir
    if args.corruption_type != "gaussian_noise" and not save_dir.rstrip("/").endswith(
        f"_{args.corruption_type}"
    ):
        save_dir = save_dir.rstrip("/") + f"_{args.corruption_type}"
    run_full_diagnostics(
        models=_parse_model_specs(args.model),
        dataset=args.dataset,
        stds=args.stds,
        rollout_steps=args.rollout_steps,
        state_key=args.state_key,
        n_sequences=args.n_sequences,
        future_steps=args.future_steps,
        frameskip=args.frameskip,
        img_size=args.img_size,
        embedding_space=args.embedding_space,
        seed=args.seed,
        device=args.device,
        save_dir=save_dir,
        plot=args.plot,
        skip_noise=args.skip_noise,
        skip_predictor=args.skip_predictor,
        skip_resolution=args.skip_resolution,
        skip_latent_noise=args.skip_latent_noise,
        skip_action_effect=args.skip_action_effect,
        predictor_history_noise_only=args.predictor_history_noise_only,
        latent_noise_geometry=args.latent_noise_geometry,
        latent_noise_std_mode=args.latent_noise_std_mode,
        latent_noise_n_samples=args.latent_noise_n_samples,
        action_effect_n_trials=args.action_effect_n_trials,
        action_effect_interp_steps=args.action_effect_interp_steps,
        action_effect_perturb_scale=args.action_effect_perturb_scale,
        corruption_type=args.corruption_type,
    )


if __name__ == "__main__":
    main()
