# Representation Analysis Guide

This document explains how to use the representation-analysis tools added for
LeWM / SWM experiments.

The goal is not just to ask "did the model avoid collapse?", but to answer:

- Does the latent space stay expressive without collapsing?
- Can it predict the next latent accurately?
- Does autoregressive rollout drift quickly under real dataset actions?
- Does the model's own planning cost prefer expert futures over random futures?
- Does action actually produce meaningful latent branching?
- If we choose an external state proxy, does latent geometry roughly agree with it?

## Files

- [analyze_repr.py](analyze_repr.py)
  Single entrypoint for one or many checkpoints. It runs the analysis, saves outputs, and also exposes notebook-friendly table / plotting helpers.
- [plot_repr.py](plot_repr.py)
  Plots one PCA / t-SNE projection export.
- [compare_repr.py](compare_repr.py)
  Draws side-by-side comparison plots from two analysis directories.
- [repr_compare_template.ipynb](repr_compare_template.ipynb)
  Editable Jupyter notebook template that imports only `analyze_repr.py`.
- [noise_sensitivity.py](noise_sensitivity.py)
  Notebook-friendly clean-vs-noisy latent diagnostics for pixel corruption robustness.
- [run_repr_batch_example.sh](run_repr_batch_example.sh)
  Editable shell example that wraps the batch script for multi-model comparison.

## What To Use When

Use `analyze_repr.py` when:
- you want quantitative diagnostics before changing losses
- you want to inspect one model on one environment
- you want JSON outputs for later plotting or comparison
- you want one CLI for both single-model and multi-model analysis

Use `plot_repr.py` when:
- you already ran `analyze_repr.py`
- you want a quick visualization similar to Figure 9 in the paper

Use `repr_compare_template.ipynb` when:
- you want to keep the final comparison loop in Jupyter instead of shell
- you want to call `run_analysis()` directly from notebook cells instead of precomputing everything in bash
- you want to edit the chosen metrics, eval scores, and model order inline
- you want to save compact key-metric tables and plots for reports

Use `noise_sensitivity.py` when:
- eval-time image noise causes a score drop and you want to see whether latent shifts are large relative to clean state separation
- you want to compare SWM spherical/cosine sensitivity against LeWM Euclidean/SIGReg sensitivity on the same sampled images
- you want a compact notebook table for goal-only noise diagnostics before launching more training runs

Notebook dependencies:

- `pandas` for tables
- `matplotlib` for charts
- `torch` plus the project runtime if you want live analysis instead of read-only loading

Use `compare_repr.py` when:
- you want to compare `SIGReg` vs `BN+uniformity`
- you want to compare two checkpoints on the same environment
- you want matched color scales across both plots

## 1. Run Analysis

Basic usage:

```bash
python tools/repr_analysis/analyze_repr.py \
  --ckpt /path/to/model_object.pt \
  --dataset tworoom \
  --future-steps 8 \
  --save-dir /tmp/repr_analysis_tworoom
```

Noise sensitivity usage from a notebook:

```python
from tools.repr_analysis.noise_sensitivity import (
    format_noise_table,
    plot_geometry_tradeoff,
    plot_noise_curves,
    run_noise_sensitivity,
    summarize_noise_geometry,
)

rows = run_noise_sensitivity(
    models={
        "swm": "/path/to/swm/model_object.ckpt",
        "lewm": "/path/to/lewm/model_object.ckpt",
    },
    dataset="tworoom",
    stds=[0.0, 0.05, 0.01, 0.02, 0.03, 0.05],
    n_sequences=256,
)

noise_table = format_noise_table(rows, frame_scope="goal")
geometry_summary = summarize_noise_geometry(rows, frame_scope="goal")

display(noise_table)
display(geometry_summary)

plot_noise_curves(rows, frame_scope="goal")
plot_noise_curves(rows, frame_scope="goal", metric="noise_angle_deg_median")
plot_geometry_tradeoff(geometry_summary)
```

The key column is `noise_to_nn_cos_ratio_median`: values near or above 1 mean the median noisy goal shift is as large as the median nearest-neighbor clean goal separation.

`summarize_noise_geometry()` turns the raw curve into design indicators:

- `robust_radius_std`: interpolated std where `noise_to_nn_cos_ratio_median` reaches 1.
- `noise_angle_slope_deg_per_std`: near-zero angular gain from pixel noise to latent direction shift.
- `clean_nn_cos_dist_median`: clean-state separation, used as a rough resolution proxy.
- `geometry_flag` / `recommendation`: compact rule-of-thumb labels for whether the model is fragile, clustered, or relatively balanced.

CLI usage with saved plots:

```bash
python -m tools.repr_analysis.noise_sensitivity \
  --model swm=/path/to/swm.ckpt \
  --model lewm=/path/to/lewm.ckpt \
  --dataset tworoom \
  --stds 0 0.05 0.01 0.02 0.03 0.05 \
  --save-dir /tmp/noise_geometry \
  --plot
```

This writes `noise_sensitivity.csv`, `geometry_summary.csv`, and three PNGs: ratio curve, angle curve, and robustness/resolution tradeoff map.

With an optional external reference probe:

```bash
python tools/repr_analysis/analyze_repr.py \
  --ckpt /path/to/model_object.pt \
  --dataset pusht \
  --state-key proprio \
  --future-steps 8 \
  --save-dir /tmp/repr_analysis_pusht
```

Useful flags:

- `--ckpt`: object checkpoint path
- `--dataset`: dataset name, e.g. `tworoom`, `pusht`, `ogb`, `dmc`
- `--state-key`: optional external state key used only for the `reference_probe`
- `--n-sequences`: number of sampled sequences, default `128`
- `--future-steps`: future states used for rollout and planning-signal probes, default `8`
- `--max-points`: cap for the optional reference-probe subsample, default `512`
- `--knn-k`: k for neighborhood overlap, default `10`
- `--planning-random-trials`: number of random futures compared against expert futures, default `16`
- `--export-tsne`: export `tsne_projection.json` if `scikit-learn` is installed
- `--tsne-perplexity`: default `30`
- `--save-dir`: output directory for JSON exports

Notes on rigor:
- prediction metrics are evaluated over sliding teacher-forced windows across the sampled sequences
- rollout metrics use real dataset action futures and the model's actual autoregressive rollout branch
- planning metrics use the model's own `get_cost()` and compare expert futures against random futures for the same start-goal pair
- the old state-geometry comparison is now an optional `reference_probe`, not the main verdict
- action-effect correlation is computed over all perturbed samples, not just one mean per trial
- interpolation smoothness is averaged over multiple anchors instead of a single context

## 2. Outputs

`analyze_repr.py` now prints:

- `Embedding`
  - anti-collapse / distribution health
  - norms, per-dim std, inter-sample cosine, effective rank
- `Prediction`
  - one-step latent prediction quality over sliding windows
- `Rollout`
  - autoregressive multi-step drift under real dataset actions
- `Planning`
  - expert-vs-random cost separation using the model's own planner objective
- `Action Effect`
  - whether changing actions creates meaningful latent branching
  - whether action interpolation is smooth
- `Reference Probe` (only if `--state-key` is provided)
  - optional comparison against an external dataset state proxy

It also prints an `Interpretation` section with environment-aware diagnostic hints.

### Saved files

When `--save-dir` is provided:

- `summary.json`
  All scalar metrics and interpretation hints.
- `metric_guide.json`
  Glossary describing what each metric means and how to use it.
- `pca_projection.json`
  2D PCA projection of latent embeddings.
- `tsne_projection.json`
  2D t-SNE projection, only if exported successfully.
- `local_neighbors.json`
  Anchor-by-anchor comparison of nearest neighbors in latent vs reference-proxy space.
  This file is only written when `--state-key` is provided.

## 3. Batch Analyze Multiple Runs

`analyze_repr.py` also supports batch mode.

Example:

```bash
export PAPER1_DATA_ROOT=dataset/ag_data/data/world_model/quentinll
python -m tools.repr_analysis.analyze_repr \
  --dataset "$PAPER1_DATA_ROOT/lewm-pusht/pusht_expert_train" \
  --future-steps 8 \
  --save-dir "$PAPER1_DATA_ROOT/lewm-pusht/repr_analysis/pusht_batch_compare" \
  --plot-projections pca \
  --compare-projections pca \
  --color-dims 0 1 \
  --model "SIGReg=$PAPER1_DATA_ROOT/lewm-pusht/ckpt/pusht_lewm_20260416/pusht_lewm_20260416_epoch_9_object.ckpt" \
  --model "BN+uniformity=$PAPER1_DATA_ROOT/lewm-pusht/ckpt/pusht_swm_v0_mlp_bn_uniform_lambda_0p1_t_2_emb_dim_192_20260417/pusht_swm_v0_mlp_bn_uniform_lambda_0p1_t_2_emb_dim_192_20260417_epoch_9_object.ckpt"
```

Batch outputs:

- one subdirectory per model with the usual `summary.json`, projection JSONs, and optional plots
- `metrics_wide.csv`
  one row per model, one column per metric
- `metrics_long.csv`
  one row per `(model, metric)` pair with metric description fields; this is the easiest file to use for later plotting
- `metrics_table.md`
  human-readable per-section tables
- `batch_summary.json`
  machine-readable aggregate result bundle
- `batch_report.txt`
  the full printed logs and per-model metric dumps
- `pairwise_compare/`
  optional side-by-side comparison plots plus `compare_manifest.json`

### Notebook workflow

If you prefer interactive comparison instead of shell loops, open:

- [repr_compare_template.ipynb](repr_compare_template.ipynb)

and edit:

- `MODEL_SPECS`
- `DATASET`
- `INCLUDE_PLANNING`
- `ANALYSIS_SAVE_DIR`
- `EXPORT_DIR`
- `EVAL_SCORES`
- `MODEL_ORDER`
- `METRICS`

The notebook can:

- run `analyze_repr.run_analysis()` for each checkpoint directly in memory
- optionally save each per-model analysis directory just like the CLI tool
- build compact comparison tables from the returned records
- render bar charts, heatmaps, and projection grids inline

If you set `ANALYSIS_SAVE_DIR`, it also writes the usual per-model `summary.json`,
projection JSONs, and local-neighbor reports.

If your notebook runtime is pointed at an older checkout where `model.get_cost()`
or planner-specific code paths are not compatible with every checkpoint, set
`INCLUDE_PLANNING = False` in the notebook template. The rest of the embedding,
prediction, rollout, and action-effect analysis will still run.

The notebook saves:

- `key_metrics_wide.csv`
- `key_metrics_long.csv`
- `key_metrics_wide.html`
- `key_metric_reference.csv`
- `key_metric_reference.html`
- `key_metrics_bars.png`
- `key_metrics_heatmap.png`
- optional projection grids such as `pca_projection_grid.png`

The bar chart and heatmap now include inline metric notes so the reader can
tell at a glance what each metric measures and what a weak value usually means.

Pairwise compare rules:

- if you pass exactly two models and set `--compare-projections`, the script compares those two automatically
- use `--compare-all-pairs` to render all pairs when you pass more than two models
- use `--compare-pair "A=B"` to compare only specific labels

Example shell wrapper:

```bash
bash tools/repr_analysis/run_repr_batch_example.sh
```

That example script mirrors your current `STABLEWM_HOME / DATASET / EPOCH / model name`
workflow, but now uses a `MODEL_SPECS` array so you can pass 2, 3, or more models in one run.
Each entry supports `label=model_name` or `label=/full/path/to.ckpt`.

## 4. Visualize One Run

Plot one exported projection:

```bash
python tools/repr_analysis/plot_repr.py \
  --input /tmp/repr_analysis_pusht/tsne_projection.json \
  --output /tmp/repr_analysis_pusht/tsne_state0_state1.png \
  --color-dims 0 1 \
  --title "PushT t-SNE"
```

Examples:

```bash
python tools/repr_analysis/plot_repr.py \
  --input /tmp/repr_analysis_tworoom/pca_projection.json \
  --output /tmp/repr_analysis_tworoom/pca_state01.png \
  --color-dims 0 1
```

```bash
python tools/repr_analysis/plot_repr.py \
  --input /tmp/repr_analysis_pusht/tsne_projection.json \
  --output /tmp/repr_analysis_pusht/tsne_state0.png \
  --color-dims 0
```

## 5. Compare Two Runs

Compare `SIGReg` vs `BN+uniformity`:

```bash
python tools/repr_analysis/compare_repr.py \
  --left-dir /tmp/repr_sigreg \
  --right-dir /tmp/repr_bn_uniform \
  --projection tsne \
  --output /tmp/repr_compare_tsne.png \
  --color-dims 0 1 \
  --left-label SIGReg \
  --right-label BN+uniformity \
  --title "PushT Representation Comparison"
```

Compare PCA instead:

```bash
python tools/repr_analysis/compare_repr.py \
  --left-dir /tmp/repr_sigreg \
  --right-dir /tmp/repr_bn_uniform \
  --projection pca \
  --output /tmp/repr_compare_pca.png \
  --color-dims 0 1 \
  --left-label SIGReg \
  --right-label BN+uniformity
```

## 6. How To Read The Results

### Embedding

Good signs:
- `effective_rank` is not tiny
- `inter_sample_cosine_mean` is not near collapse
- per-dim std is not concentrated near zero

Bad signs:
- rank is very low
- cosine similarity stays very high
- one or a few dimensions dominate

### Prediction

Most important metrics:
- `pred_target_cosine_distance_mean`
- `pred_error_mean`

Good signs:
- one-step prediction error is low
- cosine distance is low and stable across windows

Bad signs:
- even teacher-forced one-step prediction is weak
- before worrying about planning, the predictor itself is still inaccurate

Notes:
- prediction metrics are measured over sliding teacher-forced windows across the sampled sequence
- the comparison space follows the model's configured prediction-analysis space automatically

### Rollout

Most important metrics:
- `rollout_cosine_distance_last_mean`
- `rollout_error_growth`

Good signs:
- rollout stays close to the encoded future under real dataset actions
- last-step error is not dramatically worse than first-step error

Bad signs:
- one-step prediction looks fine but multi-step rollout drifts quickly
- `rollout_error_growth` is large, which usually means compounding error is the real bottleneck

Notes:
- rollout metrics use the model's actual autoregressive inference branch, not a simplified teacher-forced surrogate
- this section is often more diagnostic for planning than any external geometry probe

### Planning

Most important metrics:
- `cost_margin_mean`
- `expert_beats_random_rate`
- `expert_beats_best_random_rate`

Good signs:
- expert futures get lower cost than random futures for the same start and goal
- the cost gap is positive and stable across sequences

Bad signs:
- random futures score as well as or better than expert futures
- planner cost has weak or inconsistent separation even when prediction looks okay

Notes:
- planning metrics call the model's own `get_cost()` with the current inference rollout/cost spaces
- they probe whether the cost function itself carries useful control signal, which is usually more important than matching an external state geometry

### Reference Probe

Most important metrics:
- `distance_rank_corr_cross_seq`
- `knn_overlap_cross_seq`
- `latent_reference_step_corr`

Good signs:
- only if the chosen `state_key` is known to be task-meaningful, decent agreement can be reassuring

Bad signs:
- low agreement does **not** automatically mean the world model is bad
- it may only mean the external proxy state is not the right geometry to match

Notes:
- this section is optional and only appears when `--state-key` is provided
- treat it as a probe, not as the final judge of planning usefulness

### Action Effect

Most important metrics:
- `action_perturb_pred_shift_corr`
- `interpolation_monotonicity`

Good signs:
- small action changes produce smooth, structured latent changes

Bad signs:
- action perturbations barely move the latent prediction
- interpolation is jagged or non-monotonic

Notes:
- `action_perturb_pred_shift_corr` is now computed over all perturbed samples, not just one averaged value per trial
- `interpolation_monotonicity` is averaged over several anchors, so it is less sensitive to one lucky or unlucky context

## 7. How To Read t-SNE Correctly

This is important.

`t-SNE` is useful for:
- visualizing local neighborhoods
- checking whether a smooth state sheet becomes fragmented
- reproducing paper-style latent visualizations

`t-SNE` is *not* reliable for:
- judging global distance
- comparing absolute cluster spacing
- deciding that one method is better from visualization alone

Use it together with:
- `rollout_cosine_distance_last_mean`
- `cost_margin_mean`
- `expert_beats_random_rate`

## 8. Environment-Specific Focus

### TwoRoom

Look for:
- smooth latent continuity inside each room
- meaningful connection near the doorway
- no artificial tearing around door-adjacent states

### PushT

Look for:
- a smooth sheet as agent/block x-y positions vary
- no fragmented islands for nearby states
- local neighborhoods that remain physically meaningful

### Cube / OGB

Look for:
- smooth transitions across contact regimes
- no abrupt manifold tears caused by contact changes

### Reacher / DMC

Look for:
- continuity across nearby arm poses
- no latent shortcuts between distant joint configurations

## 9. Dependencies

Main analysis:
- no extra dependency beyond the training environment

Optional:
- `scikit-learn` for `t-SNE`
- `matplotlib` for `plot_repr.py` and `compare_repr.py`

If missing, the scripts fail with explicit messages.

## 10. Suggested Workflow

Recommended workflow before changing losses again:

1. Run `analyze_repr.py` on `SIGReg` and the candidate model.
2. Compare `summary.json` metrics first.
3. Inspect `local_neighbors.json` for anchor-level failures.
4. Plot `PCA` and optionally `t-SNE`.
5. Use `compare_repr.py` for side-by-side visual comparison.
6. Only then decide whether the next change should target:
   - anti-collapse
   - geometry
   - predictor / action conditioning
   - planning objective

This keeps the optimization loop evidence-driven instead of loss-driven.
