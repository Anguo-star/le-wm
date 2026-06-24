# Spherical World Model (SWM) — V0 Experiment

An experimental variant of [LeWorldModel](README.md) that replaces Euclidean representations and SIGReg with spherical (L2-normalised) representations and a simple spread loss.

## Motivation

LeWM achieves competitive control performance but scores only 87% on Two-Room, below simpler baselines (100%). A plausible cause: SIGReg forces embeddings toward an isotropic Gaussian, which over-constrains low-intrinsic-dimension environments like Two-Room (2D state space). This experiment asks whether replacing the Euclidean geometry with spherical geometry improves performance on that benchmark.

Full motivation and experimental design: [`plan_v2.md`](plan_v2.md).

## What changes from LeWM

| Component | LeWM | SWM (V0) |
|---|---|---|
| Encoder projector | MLP + BatchNorm → ℝ^d | `nn.Linear` → L2 norm → S^{d-1} |
| Predictor projector | MLP + BatchNorm → ℝ^d | `nn.Linear` → L2 norm → S^{d-1} |
| Prediction loss | MSE: `‖pred − tgt‖²` | Cosine distance: `1 − pred·tgt` |
| Anti-collapse regulariser | SIGReg (Cramer-Wold / Epps-Pulley) | Spread loss: mean pairwise cosine similarity |
| Planning cost | MSE | Cosine distance |

Everything else is identical: ViT-Tiny encoder, ARPredictor (ViT-S), action Embedder, CEM planner, data pipeline.

## New files

```
train_swm.py                  # training entry point for SWM
config/train/swm.yaml         # SWM training config (Two-Room by default)
```

Modified files (additive — LeWM code is unchanged):

```
jepa.py     # + SphericalJEPA class
module.py   # + cosine_pred_loss(), spread_loss()
```

## Installation

Same as LeWM — see the [main README](README.md#using-the-code):

```bash
uv venv --python=3.10
source .venv/bin/activate
uv pip install stable-worldmodel[train,env]
```

## Data

Same datasets as LeWM. See the [main README](README.md#data) for download and placement instructions.

For Two-Room (the primary benchmark for this experiment):

```bash
# download tworoom.tar.zst from https://huggingface.co/collections/quentinll/lewm
tar --zstd -xvf tworoom.tar.zst
# place tworoom.h5 under $STABLEWM_HOME (default: ~/.stable-wm/)
```

## Training

Before training, set your WandB `entity` in `config/train/swm.yaml`:

```yaml
wandb:
  config:
    entity: your_entity
```

Train SWM on Two-Room:

```bash
python train_swm.py data=tworoom
```

Train on other environments (same datasets as LeWM):

```bash
python train_swm.py data=pusht
python train_swm.py data=dmc
python train_swm.py data=ogb
```

Key hyperparameters to tune in `config/train/swm.yaml`:

| Parameter | Default | Notes |
|---|---|---|
| `wm.embed_dim` | 64 | Sphere dimension; try 128 or 192 if underfitting |
| `loss.spread.weight` | 0.1 | λ; increase if representations collapse |
| `optimizer.lr` | 5e-5 | Same as LeWM |

Planning / evaluation can now use a different space from the training
regularizer. The default branch is unchanged:

```yaml
wm:
  inference:
    rollout_state_space: normalized
    cost_space: normalized
    cost_type: cosine
```

This matches the original spherical SWM.

Training now also records which latent space the predictor consumes during
teacher-forced training. By default:

```yaml
loss:
  pred:
    context_space: ${loss.pred.space}
```

That means new runs are self-consistent by default: if prediction is trained in
raw space, predictor context is also fed from raw space unless you override it.

Uniformity regularization now also supports masked pair selection:

```yaml
loss:
  regularizer:
    type: uniformity
  uniformity:
    t: 2.0
    mode: all_pairs        # all_pairs | cross_window | temporal_masked
    temporal_exclusion: 0  # only used by temporal_masked
```

Mode semantics:

- `all_pairs`: current behaviour; every non-identical pair contributes
- `cross_window`: only pairs from different batch items contribute
- `temporal_masked`: same-window pairs with `|Δt| <= temporal_exclusion` are excluded

This is useful when you want to stop the anti-collapse loss from pushing apart
nearby temporal states that the predictor is supposed to keep aligned.

For a hybrid `exp b2` style setup, keep the regularizer on normalized embeddings
but score plans in raw predictor space:

```yaml
loss:
  pred:
    type: mse
    space: raw
    context_space: raw
  regularizer:
    type: uniformity
    space: normalized

wm:
  inference:
    rollout_state_space: normalized
    cost_space: raw
    cost_type: mse
```

This is now an explicit hybrid ablation:

- training prediction targets and predictor context are raw
- the anti-collapse regularizer stays on normalized embeddings
- planning still rolls out on normalized states

Use this only if you intentionally want to test whether changing the terminal
planning cost alone helps. It is not the main raw-dynamics baseline.

For a fully raw-consistent setup, align training, rollout, and planning in raw
space and consider removing projector BatchNorm so `emb_raw` does not collapse
into an almost fixed-norm shell:

```yaml
encoder:
  projection_head:
    type: mlp
    norm_fn: none

loss:
  pred:
    type: mse
    space: raw
    context_space: raw
  regularizer:
    type: uniformity
    space: normalized

wm:
  inference:
    rollout_state_space: raw
    cost_space: raw
    cost_type: mse
```

For the current normalized-cosine mainline, a good parallel sweep is:

```bash
python train_swm.py data=pusht \
  loss.regularizer.weight=0.2 \
  loss.uniformity.t=4.0 \
  loss.uniformity.mode=all_pairs

python train_swm.py data=pusht \
  loss.regularizer.weight=0.1 \
  loss.uniformity.t=2.0 \
  loss.uniformity.mode=cross_window

python train_swm.py data=pusht \
  loss.regularizer.weight=0.1 \
  loss.uniformity.t=2.0 \
  loss.uniformity.mode=temporal_masked \
  loss.uniformity.temporal_exclusion=1
```

Checkpoints are saved to `$STABLEWM_HOME/<subdir>/` upon completion.

## Evaluation

Evaluation is identical to LeWM — use the same `eval.py` and configs under `config/eval/`:

```bash
# Two-Room (primary benchmark)
python eval.py --config-name=tworoom.yaml policy=<subdir>/swm

# PushT
python eval.py --config-name=pusht.yaml policy=<subdir>/swm
```

Replace `<subdir>` with the Hydra job ID printed at training start (also the directory name under `$STABLEWM_HOME`). See the [main README](README.md#planning) for the full path convention.

## Results

> Results pending. Run 3–5 seeds per method for statistical comparison.

### Two-Room (primary)

| Method | Success rate | Seeds |
|---|---|---|
| LeWM (baseline) | 87% | — |
| SWM V0 (ours) | — | — |

### PushT

| Method | Success rate | Seeds |
|---|---|---|
| LeWM (baseline) | — | — |
| SWM V0 (ours) | — | — |

### Training diagnostics

> To be filled after initial runs.

- `train/pred_loss` convergence curve
- `train/spread_loss` convergence curve
- Effective rank of embedding matrix over training
- t-SNE / UMAP visualisation of learned embeddings on Two-Room

## Reference

If this work builds on LeWM, please cite the original:

```bibtex
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```
