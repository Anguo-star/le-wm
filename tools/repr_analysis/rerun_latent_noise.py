"""Rerun latent-noise diagnostic for all models after bugfix."""

import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.paper1_paths import task_dir
from tools.repr_analysis.latent_noise_sensitivity import run_latent_noise_sensitivity


def _read_models_from_csv(csv_path: Path) -> dict[str, str]:
    models = {}
    with csv_path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["model"]
            ckpt = row["ckpt"]
            if label not in models:
                models[label] = ckpt
    return models


def rerun_task(task_dir: Path, dataset: str, out_dir: Path, device: str = "cuda"):
    # Set STABLEWM_HOME so HDF5Dataset finds the correct data files
    os.environ["STABLEWM_HOME"] = str(task_dir)

    # Collect models from main dir
    main_csv = task_dir / "repr_analysis" / "latent_noise_diagnostics" / "latent_noise_sensitivity.csv"
    models = _read_models_from_csv(main_csv) if main_csv.exists() else {}

    # Collect from new_baselines subdirs
    new_base = task_dir / "repr_analysis" / "p03_diagnostics_new_baselines"
    if new_base.exists():
        for sub in new_base.iterdir():
            if sub.is_dir():
                csv_path = sub / "latent_noise_sensitivity.csv"
                if csv_path.exists():
                    models.update(_read_models_from_csv(csv_path))

    if not models:
        print(f"No models found for {task_dir.name}")
        return

    print(f"Rerunning {len(models)} models for {task_dir.name}...")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = run_latent_noise_sensitivity(
        models=models,
        dataset=dataset,
        stds=[0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08],
        rollout_steps=[1, 2, 4, 8],
        frameskip=5,
        n_sequences=256,
        future_steps=8,
        seed=3072,
        device=device,
    )

    # Write CSV
    if rows:
        keys = list(rows[0].keys())
        with (out_dir / "latent_noise_sensitivity.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        with (out_dir / "latent_noise_sensitivity.json").open("w") as f:
            json.dump(rows, f, indent=2)

    print(f"  Wrote {out_dir}/latent_noise_sensitivity.csv ({len(rows)} rows)")


if __name__ == "__main__":
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rerun_task(
        task_dir("lewm-tworooms"),
        "tworoom",
        task_dir("lewm-tworooms") / "repr_analysis" / "latent_noise_diagnostics",
        device,
    )
    rerun_task(
        task_dir("lewm-pusht"),
        "pusht_expert_train",
        task_dir("lewm-pusht") / "repr_analysis" / "latent_noise_diagnostics",
        device,
    )
