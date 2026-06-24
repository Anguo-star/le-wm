"""
action_effect.py — Diagnose how strongly action perturbations move the predicted
latent. Wraps `analyze_action_effect` from analyze_repr.py with the same
batch-once / iterate-models pattern as task_resolution / noise_sensitivity, so
it can be plugged into `run_full_diagnostics.py` and produced automatically by
`run_trainer.sh` after each training.

Per-checkpoint indicators (`analyze_action_effect`):
    - `mean_pred_shift_norm`: mean L2 shift of single-step prediction when the
      last action token is perturbed by `perturb_scale × per-dim std`.
    - `action_perturb_pred_shift_corr`: Pearson correlation between action
      perturbation magnitude and resulting prediction shift; >0 means the
      predictor is action-sensitive on average.
    - `interpolation_monotonicity`: along an action interpolation path,
      fraction of consecutive steps where prediction distance from the start
      is non-decreasing; >0.8 means the predictor responds smoothly to action
      gradients.
    - `interpolation_endpoint_shift`: distance between predictions at the two
      endpoints of the interpolation; magnitude check on the interpolation
      itself.

Notebook use:

    from tools.repr_analysis.action_effect import run_action_effect
    rows = run_action_effect(
        models={"swm": "/path/...", "lewm": "/path/..."},
        dataset="tworoom",
    )
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch

from tools.repr_analysis.analyze_repr import (
    analyze_action_effect,
    encode_sequences,
    infer_history_size,
    load_dataset_samples,
    load_model,
    to_serializable,
)


@torch.no_grad()
def analyze_model_action_effect(
    *,
    label: str,
    ckpt: str,
    batch: Mapping[str, torch.Tensor],
    n_trials: int = 128,
    interp_steps: int = 16,
    perturb_scale: float = 0.5,
    device: str = "cuda",
) -> Dict[str, Any]:
    model = load_model(ckpt, device)
    outputs = encode_sequences(
        model,
        {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()},
    )
    metrics = analyze_action_effect(
        model=model,
        outputs=outputs,
        n_trials=n_trials,
        interp_steps=interp_steps,
        perturb_scale=perturb_scale,
    )
    return {
        "model": label,
        "ckpt": ckpt,
        "history_size": int(infer_history_size(model)),
        **metrics,
    }


def run_action_effect(
    *,
    models: Mapping[str, str],
    dataset: str = "tworoom",
    state_key: str | None = None,
    n_sequences: int = 256,
    history_size: int | None = None,
    future_steps: int = 8,
    frameskip: int = 1,
    img_size: int = 224,
    n_trials: int = 128,
    interp_steps: int = 16,
    perturb_scale: float = 0.5,
    embedding_space: str | None = None,  # accepted for API symmetry; unused
    seed: int = 3072,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> list[Dict[str, Any]]:
    del embedding_space  # action_effect uses each model's rollout space
    if not models:
        raise ValueError("models must contain at least one label -> checkpoint path.")

    first_ckpt = next(iter(models.values()))
    first_model = load_model(first_ckpt, device)
    H = history_size or infer_history_size(first_model)
    del first_model

    batch = load_dataset_samples(
        dataset_name=dataset,
        state_key=state_key,
        n_sequences=n_sequences,
        history_size=H,
        future_steps=future_steps,
        frameskip=frameskip,
        img_size=img_size,
        seed=seed,
        device=device,
    )

    rows = []
    for label, ckpt in models.items():
        rows.append(
            analyze_model_action_effect(
                label=label,
                ckpt=ckpt,
                batch=batch,
                n_trials=n_trials,
                interp_steps=interp_steps,
                perturb_scale=perturb_scale,
                device=device,
            )
        )
    return rows


def format_action_effect_table(rows: Sequence[Mapping[str, Any]]):
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("format_action_effect_table requires pandas.") from exc

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    columns = [c for c in [
        "model",
        "n_trials", "n_action_pairs", "n_interp_anchors",
        "perturb_scale",
        "mean_action_perturb_norm",
        "mean_pred_shift_norm",
        "action_perturb_pred_shift_corr",
        "interpolation_endpoint_shift",
        "interpolation_endpoint_shift_std",
        "interpolation_monotonicity",
        "interpolation_monotonicity_std",
    ] if c in df.columns]
    df = df[columns].sort_values("model").reset_index(drop=True)
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].round(4)
    return df


def _parse_model_specs(specs: Sequence[str]) -> Dict[str, str]:
    models: Dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Model spec must be label=/path/to/ckpt, got: {spec}")
        label, ckpt = spec.split("=", 1)
        models[label.strip()] = ckpt.strip()
    return models


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Diagnose predictor action sensitivity.")
    p.add_argument("--model", action="append", required=True,
                   help="Model spec as label=/path/to/model_object.ckpt. Repeat for comparisons.")
    p.add_argument("--dataset", default="tworoom")
    p.add_argument("--state-key", default=None)
    p.add_argument("--n-sequences", type=int, default=256)
    p.add_argument("--future-steps", type=int, default=8)
    p.add_argument("--frameskip", type=int, default=1)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--n-trials", type=int, default=128)
    p.add_argument("--interp-steps", type=int, default=16)
    p.add_argument("--perturb-scale", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=3072)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-dir", default=None)
    return p


def main():
    args = build_parser().parse_args()
    rows = run_action_effect(
        models=_parse_model_specs(args.model),
        dataset=args.dataset,
        state_key=args.state_key,
        n_sequences=args.n_sequences,
        future_steps=args.future_steps,
        frameskip=args.frameskip,
        img_size=args.img_size,
        n_trials=args.n_trials,
        interp_steps=args.interp_steps,
        perturb_scale=args.perturb_scale,
        seed=args.seed,
        device=args.device,
    )
    print(format_action_effect_table(rows).to_string(index=False))

    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        with (save_dir / "action_effect.json").open("w") as f:
            json.dump(to_serializable(rows), f, indent=2)
        if rows:
            with (save_dir / "action_effect.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)


if __name__ == "__main__":
    main()
