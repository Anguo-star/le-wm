# External baseline training (PLDM, DINO-WM)

This repo's primary results are on **LeWM** (`train.py`). For the paper's
external-baseline comparison, two additional vision-only JEPA-style world
models are supported here with the **same image-noise augmentation pipeline**
as LeWM:

- **PLDM** (Predictive Latent Dynamics Model) — `train_pldm.py`
- **PreJEPA / DINO-WM** (no-prop by default, with-prop variant available) —
  `train_prejepa.py`

Both training scripts are thin wrappers around the upstream
`stable-worldmodel/scripts/train/` originals. The wrapper duplicates only
the Hydra entry point; model code, losses, and forwards are imported from
the upstream package.

`train_pldm.py` uses components available from the installed
`stable_worldmodel` package. `train_prejepa.py` additionally imports helper
functions from the upstream script file. Before running DINO-WM / PreJEPA,
either clone the upstream repo into `external/stable-worldmodel` or set
`STABLE_WORLDMODEL_REPO`:

```bash
git clone https://github.com/galilai-group/stable-worldmodel external/stable-worldmodel
export STABLE_WORLDMODEL_REPO="$PWD/external/stable-worldmodel"
```

## Why two scripts?

LeWM (`jepa.py::JEPA.encode`) and PLDM (upstream `pldm.py::PLDM.encode`)
both operate **vision-only** — they read only `info["pixels"]` and ignore
proprio even if it is loaded in the data pipeline. DINO-WM (PreJEPA) by
default includes a proprio auxiliary encoder.

For this paper's visual-OOD-robustness framing, the apples-to-apples
comparison is **vision-only**: PLDM is native vision-only; DINO-WM has to be
configured with `wm.encoding = {action: 10}` (no `proprio`). The
with-proprio configuration is preserved as an appendix sensitivity check.

| Model | proprio in `encode()`? | Used in this paper |
|---|---|---|
| LeWM (`train.py`) | no | main |
| PLDM (`train_pldm.py`) | no | main external baseline |
| PreJEPA / DINO-WM no-prop (`train_prejepa.py`) | no | main external baseline |
| PreJEPA / DINO-WM with-prop (override) | yes | appendix sensitivity check |

## Quick reference

### PLDM

```bash
# clean (no noise) baseline
python train_pldm.py exp_name=pusht_pldm
python train_pldm.py exp_name=tworoom_pldm data=tworoom_baseline

# noise sweep (override image_noise at CLI)
python train_pldm.py exp_name=pusht_pldm_noise_0to006_p1 \
    image_noise.std_max=0.06 image_noise.noise_prob=1.0
python train_pldm.py exp_name=pusht_pldm_noise_0to008_p1 \
    image_noise.std_max=0.08 image_noise.noise_prob=1.0
```

### DINO-WM (no-prop, main)

```bash
# clean
python train_prejepa.py exp_name=pusht_dinowm_noprop \
    dataset_name=pusht_expert_train

python train_prejepa.py exp_name=tworoom_dinowm_noprop \
    dataset_name=tworoom

# noise sweep
python train_prejepa.py exp_name=pusht_dinowm_noprop_noise_0to006_p1 \
    dataset_name=pusht_expert_train \
    image_noise.std_max=0.06 image_noise.noise_prob=1.0
```

### DINO-WM (with-prop, appendix)

```bash
python train_prejepa.py exp_name=pusht_dinowm_prop \
    dataset_name=pusht_expert_train \
    +wm.encoding.proprio=10
```

## Recommended sweep plan

For the visual-OOD comparison in the paper, the cheapest informative run set is:

| Task | Model | Configurations | # ckpts |
|---|---|---|---|
| PushT | PLDM | base (no noise) + std_max ∈ {0.02, 0.05, 0.08} | 4 |
| PushT | DINO-WM no-prop | base + std_max ∈ {0.02, 0.05, 0.08} | 4 |
| TwoRoom | PLDM | base + std_max ∈ {0.02, 0.05, 0.08} | 4 |
| TwoRoom | DINO-WM no-prop | base + std_max ∈ {0.02, 0.05, 0.08} | 4 |

Total: **16 training runs** (1 GPU × ~2–3 h each). Each run uses
3 seeds × 100 evaluation trajectories for protocol parity with LeWM.

Appendix sensitivity check (with-prop DINO-WM): 2 tasks × 4 configurations = 8 runs.

## Evaluation

`eval.py` is checkpoint-format agnostic. Once a PLDM or PreJEPA model is
saved via `save_pretrained`, `swm.policy.AutoCostModel('<task>/<exp_name>')`
loads it directly. Visual-noise corruption is applied via the same
`eval.corruption.std` field used for LeWM evaluation:

```bash
python eval.py --config-name=pusht.yaml policy=<task>/<exp_name> \
    eval.corruption.std=0.08 \
    eval.corruption.targets=[pixels,goal]
```

No baseline-specific eval changes are required; this is the design payoff
of keeping the training pipeline identical in noise plumbing.
