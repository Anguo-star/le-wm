# Paper 1 Reproduction Guide

This repository contains the code, configs, canonical result artifacts, figure
scripts, and reproducibility metadata for:

> A Diagnostic Study of Gaussian Visual Robustness in JEPA Latent World Models

Large HDF5 datasets and model checkpoints are intentionally not tracked in git.
Set one local prefix and keep all paths below it.

## 1. Environment

```bash
uv venv --python=3.10
source .venv/bin/activate
uv pip install -r requirements.txt
python -c "import torch, stable_worldmodel, stable_pretraining, sklearn; print('paper1 env ok')"
```

The LeWM, SWM, PLDM, eval, and diagnostics paths use packages installed from
`requirements.txt`. The DINO-WM / PreJEPA wrapper also reuses helper functions
from the upstream `stable-worldmodel` training script. For that baseline:

```bash
git clone https://github.com/galilai-group/stable-worldmodel external/stable-worldmodel
export STABLE_WORLDMODEL_REPO="$PWD/external/stable-worldmodel"
```

If you skip the DINO-WM / PreJEPA baseline, this checkout is not needed.

## 2. Data And Checkpoint Layout

Use a prefix that contains the task folders:

```bash
export PAPER1_DATA_ROOT=/path/to/paper1-data
export STABLEWM_HOME="$PAPER1_DATA_ROOT"
```

Expected task layout:

```text
$PAPER1_DATA_ROOT/
  lewm-pusht/
    pusht_expert_train.h5
    ckpt/
  lewm-tworooms/
    tworoom.h5
    ckpt/
  lewm-reacher/
    reacher.h5
    ckpt/
  lewm-cube/
    ogbench/cube_single_expert.h5
    ckpt/
```

The HDF5 resolver also accepts a `datasets/` subdirectory, for example
`$PAPER1_DATA_ROOT/lewm-pusht/datasets/pusht_expert_train.h5`.

Canonical JSON artifacts store checkpoint paths relative to `PAPER1_DATA_ROOT`.
The released aggregate files live in `assets/paper1_data/`; see
`DATA_MANIFEST.md` for task roots, checkpoint names, and SHA-256 values.

## 3. Fast Sanity Checks

These checks do not require model checkpoints:

```bash
python -m tools.check_paper1_consistency
python -m tools.paper1_figs --out-dir assets/paper1_figs
python -m tools.paper1_acpc_basin --dry-run --out /tmp/acpc_basin_dry.json
bash run_phase0_acpc.sh --dry-run
```

The manuscript source and arXiv packaging files are maintained outside this
public code branch. The checks above validate the released artifacts and the
code paths needed to regenerate figures and diagnostics.

## 4. Paper 1 LeWM Training Sweep

`run_trainer.sh` is the main one-shot entry point: train, evaluate clean and
corrupted conditions, then run the diagnostics suite.

Example PushT run at `std_max=0.08`:

```bash
export PAPER1_DATA_ROOT=/path/to/paper1-data
export STABLEWM_HOME="$PAPER1_DATA_ROOT"

dataset_name=pusht \
trainer_file=train.py \
config=lewm \
output_model_name=lewm_noise_0to008_p1 \
num_eval=300 \
image_noise_std_min=0.0 \
image_noise_std_max=0.08 \
image_noise_noise_prob=1.0 \
eval_corruption_stds="0.0 0.03 0.05 0.08" \
bash run_trainer.sh
```

The script maps `dataset_name` to the correct task folder:

| `dataset_name` | Train data group | Task folder |
|---|---|---|
| `tworoom` | `tworoom` | `lewm-tworooms` |
| `pusht` | `pusht` | `lewm-pusht` |
| `reacher` | `dmc` | `lewm-reacher` |
| `cube` | `ogb` | `lewm-cube` |

By default, runs without `SWANLAB_API_KEY` use `logger_backend=none`, so public
reproduction does not require an online logging account. To enable SwanLab:

```bash
export SWANLAB_API_KEY=...
logger_backend=swanlab bash run_trainer.sh
```

For train-only smoke tests:

```bash
post_train_eval_mode=none num_eval=1 ... bash run_trainer.sh
```

For eval-only from an existing checkpoint:

```bash
skip_train=1 \
ckpt_override="$PAPER1_DATA_ROOT/lewm-pusht/ckpt/<run>/<run>_epoch_10_object.ckpt" \
dataset_name=pusht trainer_file=train.py config=lewm \
output_model_name=<run> num_eval=300 \
bash run_trainer.sh
```

## 5. External Baselines

PLDM uses the local wrapper and the same noise pipeline:

```bash
dataset_name=pusht \
trainer_file=train_pldm.py \
config=pldm \
output_model_name=pldm_noise_0to008_p1 \
num_eval=300 \
image_noise_std_min=0.0 \
image_noise_std_max=0.08 \
image_noise_noise_prob=1.0 \
bash run_trainer.sh
```

DINO-WM / PreJEPA needs `STABLE_WORLDMODEL_REPO` as described above. Direct
single-run examples are in `BASELINES.md`.

## 6. Diagnostics And Figure Regeneration

Main release checks:

```bash
python -m tools.check_paper1_consistency
python -m tools.paper1_figs --out-dir assets/paper1_figs
```

Checkpoint-loading diagnostics:

```bash
python -m tools.paper1_acpc_basin \
  --model-root "$PAPER1_DATA_ROOT" \
  --out assets/paper1_data/acpc_basin_diagnostics.json

python -m tools.paper1_acpc_basin \
  --methods PLDM \
  --model-root "$PAPER1_DATA_ROOT" \
  --out assets/paper1_data/acpc_basin_diagnostics_pldm.json

PAPER1_DATA_ROOT="$PAPER1_DATA_ROOT" bash run_phase0_acpc.sh --lewm-only
```

Qualitative PushT figures:

```bash
STABLEWM_HOME="$PAPER1_DATA_ROOT" python -m tools.paper1_selective_contraction \
  --plot-clusters --plot-tasks PushT \
  --n-sequences 128 --cluster-anchor-count 16 \
  --view-stds 0.0 0.01 0.04 0.08 \
  --cluster-perturb-repeats 6 \
  --cluster-out-dir assets/paper1_figs \
  --cluster-envelope ellipse --cluster-envelope-coverage 0.90

STABLEWM_HOME="$PAPER1_DATA_ROOT" python -m tools.paper1_selective_contraction \
  --plot-atlas --plot-tasks PushT \
  --atlas-out-dir assets/paper1_figs
```

More detailed tool notes are in `tools/README_paper1.md`.
