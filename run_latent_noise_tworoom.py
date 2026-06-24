import os

from pathlib import Path
from tools.paper1_paths import task_dir
from tools.repr_analysis.latent_noise_sensitivity import run_latent_noise_sensitivity

TASK_ROOT = task_dir("lewm-tworooms")
os.environ.setdefault("STABLEWM_HOME", str(TASK_ROOT))

ROOT = TASK_ROOT / "ckpt"

MODELS = {
    'LeWM-base': str(ROOT / 'tworoom_lewm/tworoom_lewm_epoch_9_object.ckpt'),
    'LeWM-fixed-std': str(ROOT / 'tworoom_lewm_noise_std_0_005/tworoom_lewm_noise_std_0_005_epoch_9_object.ckpt'),
    'LeWM-perframe-p05': str(ROOT / 'tworoom_lewm_noise_0to005_p05/tworoom_lewm_noise_0to005_p05_epoch_9_object.ckpt'),
    'LeWM-perframe-p1': str(ROOT / 'tworoom_lewm_noise_0to005_p1/tworoom_lewm_noise_0to005_p1_epoch_9_object.ckpt'),
    'SWM-base': str(ROOT / 'tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_dim64_20260425/tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_dim64_20260425_epoch_9_object.ckpt'),
    'SWM-fixed-std': str(ROOT / 'tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_std0_005_dim64/tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_std0_005_dim64_epoch_9_object.ckpt'),
    'SWM-perframe-p05': str(ROOT / 'tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to005_p05_dim64/tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to005_p05_dim64_epoch_9_object.ckpt'),
    'SWM-perframe-p1': str(ROOT / 'tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to005_p1_dim64/tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to005_p1_dim64_epoch_9_object.ckpt'),
}

# Run LeWM models with ambient noise geometry
lewm_models = {k: v for k, v in MODELS.items() if k.startswith('LeWM')}
print(f"Running TwoRoom LeWM models: {list(lewm_models.keys())}")
rows_lewm = run_latent_noise_sensitivity(
    models=lewm_models,
    dataset='tworoom',
    frameskip=5,
    noise_geometry='ambient',
    n_sequences=256,
    stds=[0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08],
    rollout_steps=[1, 2, 4, 8],
)

# Run SWM models with tangent noise geometry (spherical)
swm_models = {k: v for k, v in MODELS.items() if k.startswith('SWM')}
print(f"Running TwoRoom SWM models: {list(swm_models.keys())}")
rows_swm = run_latent_noise_sensitivity(
    models=swm_models,
    dataset='tworoom',
    frameskip=5,
    noise_geometry='tangent',
    n_sequences=256,
    stds=[0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08],
    rollout_steps=[1, 2, 4, 8],
)

import csv
import json
from tools.repr_analysis.analyze_repr import to_serializable

out_dir = TASK_ROOT / "repr_analysis" / "latent_noise_diagnostics"
out_dir.mkdir(parents=True, exist_ok=True)

all_rows = rows_lewm + rows_swm
if all_rows:
    keys = list(all_rows[0].keys())
    with open(out_dir / 'latent_noise_sensitivity.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: to_serializable(v) for k, v in row.items()})
    print(f"Saved CSV: {out_dir / 'latent_noise_sensitivity.csv'} ({len(all_rows)} rows)")

print("Done.")
