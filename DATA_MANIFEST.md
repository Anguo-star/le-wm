# Paper 1 Data Manifest

This manifest documents the released evaluation aggregate for Paper 1:

- Canonical aggregate: `assets/paper1_data/canonical_evals_20260517.json`
- Schema: `assets/paper1_data/canonical_evals_20260517.schema.json`
- Canonical diagnostics: `assets/paper1_data/canonical_diagnostics_20260517.json` (2026-06-10 revision: the TwoRoom/PushT representative diagnostics used by `tab:diag-base-vs-best` were re-extracted from the per-checkpoint diagnostics after an audit found they duplicated the `*_lewm_hetero_default` values; see the JSON `metadata.table3_revision_20260610` note)
- Diagnostics schema: `assets/paper1_data/canonical_diagnostics_20260517.schema.json`
- External baseline sanity check: `assets/paper1_data/canonical_external_baselines_20260520.json`
- External baseline schema: `assets/paper1_data/canonical_external_baselines_20260520.schema.json`
- **PLDM cross-method replication aggregate**: `assets/paper1_data/canonical_evals_pldm_20260522.json` (36 ckpts: 4 tasks × 9 configs)
- **PLDM cross-method replication diagnostics**: `assets/paper1_data/canonical_diagnostics_pldm_20260522.json`
- **PLDM full diagnostics**: `assets/paper1_data/canonical_full_diagnostics_pldm_20260523.json` (full diagnostics-summary rows for the same 36 PLDM ckpts)
- **PLDM full diagnostics schema**: `assets/paper1_data/canonical_full_diagnostics_pldm_20260523.schema.json`
- **Cross-method correlations**: `assets/paper1_data/cross_method_corr_pldm_20260522.json` (within-LeWM / within-PLDM / joint partial Spearman; consumed by the PLDM appendix)
- **PLDM full ACPC basin replication**: `assets/paper1_data/acpc_basin_diagnostics_pldm.json` (36 rows: 4 tasks × 9 PLDM configs)
- **Partial-correlation bootstrap CIs**: `assets/paper1_data/partial_corr_bootstrap_20260523.json` (95% percentile bootstrap intervals for the LeWM, PLDM, and joint partial-correlation tables)
- **Phase-0 paired ACPC diagnostics**: `assets/paper1_data/acpc_phase0_diagnostics.json` (72 rows: LeWM + PLDM, 4 tasks × 9 configs; consumed by the Phase-0 appendix)
- **Gaussian-noise ACPC basin diagnostics**: `assets/paper1_data/acpc_basin_diagnostics.json` (LeWM 36 ckpts, epoch-10 model objects, paired clean/noised views at Gaussian std 0.01..0.08; source for the ACPC basin table)
- **No-noise-baseline blur sanity check**: `assets/paper1_data/canonical_blur_baselines_20260523.json` (LeWM + PLDM baselines trained without input-noise augmentation, 4 tasks, blur eval only)
- **No-noise-baseline blur schema**: `assets/paper1_data/canonical_blur_baselines_20260523.schema.json`
- Scope: 36 LeWM checkpoints = 4 tasks × 9 configs (`base` + `std_max` 0.01..0.08); 36 PLDM checkpoints on the same grid
- Evaluation protocol: **3 evaluation seeds** (`42`, `43`, `44`) × **100 trajectories per seed**
- Important clarification: these are **evaluation seeds**, not 3 independently trained models per configuration

## Release Provenance Notes

- Paper-facing main evidence: `canonical_evals_20260517.json`, `canonical_diagnostics_20260517.json`, `acpc_basin_diagnostics.json`, `canonical_evals_pldm_20260522.json`, `canonical_diagnostics_pldm_20260522.json`, `canonical_full_diagnostics_pldm_20260523.json`, `acpc_basin_diagnostics_pldm.json`, and `partial_corr_bootstrap_20260523.json`.
- Scope-boundary / sanity artifacts: `canonical_blur_baselines_20260523.json` is eval-only blur stress; `acpc_phase0_diagnostics.json` is exploratory paired-readout evidence under observation+goal stress; `target_view_closed_loop_summary.json` is a negative target-view ablation; `canonical_external_baselines_20260520.json` is retained for backward-compatible sanity checks.
- Contamination fix: the 2026-06-10 audit found that the TwoRoom and PushT representative diagnostic rows in `canonical_diagnostics_20260517.json` duplicated heteroscedastic-loss diagnostics. The affected representative fields were re-extracted from the intended per-checkpoint `diagnostics_summary.json` files. The release checker now guards these values so the PushT noise-sweep row cannot regress to the heteroscedastic `rank 76.4 -> 42.9` narrative.
- Manual revision status: no released JSON artifact is hand-edited for paper prose. The only documented corrective revision is the representative-diagnostics re-extraction above, recorded in JSON metadata and checked by `tools/check_paper1_consistency.py`.

| Artifact | Role | SHA-256 |
|---|---|---|
| `canonical_evals_20260517.json` | LeWM evaluation aggregate | `394c21142311e628232e510d0087b17828ab78d973727cff6b049cb50ed98e1a` |
| `canonical_diagnostics_20260517.json` | LeWM diagnostics and representative rows | `8012fd3bc5fb445bd5d00ea78d3c5df30f57c10f69b5175f8479b48370517443` |
| `acpc_basin_diagnostics.json` | LeWM Gaussian ACPC basin | `e0e468d2d7a94666e6bbcf8dafd24f32fe2dade03d66deb5619a86145a8dc521` |
| `canonical_evals_pldm_20260522.json` | PLDM evaluation aggregate | `e9bf3a49b91f3d17151db2cd94c696cd56f53d8c2be9670351f059ebb671df7a` |
| `canonical_diagnostics_pldm_20260522.json` | PLDM predictor diagnostics | `efb15cc8baafd1f80e5d5b67ffce59666ef909f7a4106dd50cc6a7c9fcf4c536` |
| `canonical_full_diagnostics_pldm_20260523.json` | PLDM five-layer diagnostics | `6a5b2ae47b09b4bd6fd6fba87846e7d9484e6beea2cfe24f75380c238c73fc7d` |
| `acpc_basin_diagnostics_pldm.json` | PLDM Gaussian ACPC basin | `dd6aeaa3e793ce09294b049b31ff2f7791c7b83fe0a6a375e6adba806f23e6e2` |
| `partial_corr_bootstrap_20260523.json` | Bootstrap CI aggregate | `e6cbba0893defd152b540150dcf86ee091fbe5cf4061131871228b6a59d51465` |
| `acpc_phase0_diagnostics.json` | Exploratory paired ACPC/PCC/CRA/MAF diagnostics | `9654759b576216b7249a3bf5e2ee7b778318cf4de22babf3395a0757b3e644fd` |
| `target_view_closed_loop_summary.json` | Negative target-view ablation | `04f75ad72543fb98d51304a6dec12ceb1b2dc099e915e24a49862ed3451744d0` |
| `canonical_blur_baselines_20260523.json` | Eval-only blur sanity check | `8e4c18d9f354a585770e6eb389e4ceb1b449eea7a5e7758af40a324878e0700b` |

## Data Semantics

- Unit: every success-rate field is stored in **percentage points** on `[0, 100]`
- Aggregation: each metric stores `values = [seed42, seed43, seed44]`, `mean`, and `std`
- `std` convention: the JSON stores the **population standard deviation** across the 3 evaluation seeds (`ddof = 0`)
- Conditions released per checkpoint:
  - `clean`
  - `goal_std0.03`, `goal_std0.05`, `goal_std0.08`
  - `pixels_std0.03`, `pixels_std0.05`, `pixels_std0.08`
  - `pixels_goal_std0.03`, `pixels_goal_std0.05`, `pixels_goal_std0.08`
- Paper 1 primary corrupted endpoint: `pixels_std0.08` (observation pixels corrupted, goal image kept clean). `pixels_goal_std0.08` is retained as a stronger full-visual-stream stress condition.
- Canonical lookup key for portability: use `subdir`
  - The JSON also stores an absolute local `path`, but downstream tools should not rely on that exact absolute prefix

## Per-Seed File Pattern

For every checkpoint subdirectory below, the raw per-seed metrics are expected at:

`<ckpt>/eval_results/<cond>_seed42_metrics.txt`

`<ckpt>/eval_results/<cond>_seed43_metrics.txt`

`<ckpt>/eval_results/<cond>_seed44_metrics.txt`

where `<cond>` is one of:

- `clean`
- `goal_std0.03`, `goal_std0.05`, `goal_std0.08`
- `pixels_std0.03`, `pixels_std0.05`, `pixels_std0.08`
- `pixels_goal_std0.03`, `pixels_goal_std0.05`, `pixels_goal_std0.08`

## Task Roots

Set `DATA_ROOT` to the machine-local prefix that contains the released
`lewm-*` task directories. Keep that prefix outside the manifest and commands;
the released task roots below stay relative to `DATA_ROOT` so the same
artifacts can be moved between machines.

| Task | Local ckpt root |
|---|---|
| TwoRoom | `$DATA_ROOT/lewm-tworooms/ckpt` |
| PushT | `$DATA_ROOT/lewm-pusht/ckpt` |
| Reacher | `$DATA_ROOT/lewm-reacher/ckpt` |
| Cube | `$DATA_ROOT/lewm-cube/ckpt` |

## Released Checkpoints

| Task | `std_max` | Canonical `subdir` | Raw per-seed file pattern |
|---|---:|---|---|
| TwoRoom | 0.0 | `tworoom_lewm_20260430` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| TwoRoom | 0.01 | `tworoom_lewm_noise_0to001_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| TwoRoom | 0.02 | `tworoom_lewm_noise_0to002_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| TwoRoom | 0.03 | `tworoom_lewm_noise_0to003_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| TwoRoom | 0.04 | `tworoom_lewm_noise_0to004_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| TwoRoom | 0.05 | `tworoom_lewm_noise_0to005_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| TwoRoom | 0.06 | `tworoom_lewm_noise_0to006_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| TwoRoom | 0.07 | `tworoom_lewm_noise_0to007_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| TwoRoom | 0.08 | `tworoom_lewm_noise_0to008_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| PushT | 0.0 | `pusht_lewm_20260430` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| PushT | 0.01 | `pusht_lewm_noise_0to001_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| PushT | 0.02 | `pusht_lewm_noise_0to002_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| PushT | 0.03 | `pusht_lewm_noise_0to003_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| PushT | 0.04 | `pusht_lewm_noise_0to004_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| PushT | 0.05 | `pusht_lewm_noise_0to005_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| PushT | 0.06 | `pusht_lewm_noise_0to006_p1_20260507` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| PushT | 0.07 | `pusht_lewm_noise_0to007_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| PushT | 0.08 | `pusht_lewm_noise_0to008_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Reacher | 0.0 | `reacher_lewm_20260430` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Reacher | 0.01 | `reacher_lewm_noise_0to001_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Reacher | 0.02 | `reacher_lewm_noise_0to002_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Reacher | 0.03 | `reacher_lewm_noise_0to003_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Reacher | 0.04 | `reacher_lewm_noise_0to004_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Reacher | 0.05 | `reacher_lewm_noise_0to005_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Reacher | 0.06 | `reacher_lewm_noise_0to006_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Reacher | 0.07 | `reacher_lewm_noise_0to007_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Reacher | 0.08 | `reacher_lewm_noise_0to008_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Cube | 0.0 | `cube_lewm_20260430` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Cube | 0.01 | `cube_lewm_noise_0to001_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Cube | 0.02 | `cube_lewm_noise_0to002_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Cube | 0.03 | `cube_lewm_noise_0to003_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Cube | 0.04 | `cube_lewm_noise_0to004_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Cube | 0.05 | `cube_lewm_noise_0to005_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Cube | 0.06 | `cube_lewm_noise_0to006_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Cube | 0.07 | `cube_lewm_noise_0to007_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |
| Cube | 0.08 | `cube_lewm_noise_0to008_p1` | `<subdir>/eval_results/<cond>_seed{42,43,44}_metrics.txt` |

## Consumer Notes

- `tools/paper1_figs.py` treats `assets/paper1_data/canonical_evals_20260517.json` as the source of truth for the script-generated main evaluation figures and tables.
- `tools/paper1_figs.py` treats `assets/paper1_data/canonical_diagnostics_20260517.json` as the source of truth for predictor metrics, the representative diagnostic table (`tab:diag-base-vs-best`), and the PushT fragility scatter (`fig:scatter`).
- The canonical diagnostics release stores:
  - 4 tasks × 9 ckpts of `predictor_target_to_nn_cos_ratio_at_max_std`
  - 4 tasks × 9 ckpts of `predictor_rollout_T8_l2_at_max_std`
  - finalized representative diagnostic values used by `tab:diag-base-vs-best`
  - published LeWM cross-checkpoint correlation and partial-correlation values
- The historical external baseline release stores one PushT PLDM run trained without input-noise augmentation (`pusht_pldm_baseline`) for backward compatibility. The primary PLDM release is now the full 36-checkpoint aggregate in `canonical_evals_pldm_20260522.json`.
- The PLDM aggregate uses the same condition names, success-rate units, 3 evaluation seeds, and population-std convention as the LeWM aggregate. It is used only for the PLDM cross-method appendix, not for the main LeWM-only sweep and correlation tables.
- `canonical_full_diagnostics_pldm_20260523.json` stores the full `diagnostics_summary.json` row for every PLDM checkpoint and a compact base-vs-representative table used by the PLDM appendix. It is interpreted as a mechanism-boundary check: PLDM replicates the task-level fragility/recovery signature but does not reuse LeWM's exact compression-chain profile.
- `acpc_basin_diagnostics_pldm.json` stores the full PLDM 4 tasks × 9 configs Gaussian ACPC basin replication. It uses the same same-state clean/noised view protocol as `acpc_basin_diagnostics.json`; the PLDM appendix reports a compact baseline-vs-pixels-0.08-point-best summary from this full artifact.
- `partial_corr_bootstrap_20260523.json` stores the 95% percentile bootstrap CIs for the LeWM, PLDM, and joint partial-correlation claims quoted in the main text and PLDM appendix. It is generated by `tools/build_partial_corr_bootstrap.py` from the canonical eval/diagnostic JSONs.
- `acpc_phase0_diagnostics.json` stores the Phase-0 paired ACPC diagnostics (ACPC-1/H, PCC, CRA, MAF, ADM action-distance proxy, SPRR) for the LeWM and PLDM full sweep. It is an exploratory diagnostic artifact for the Phase-0 appendix, not a method-result file or a robustness-predictor benchmark.
- `acpc_basin_diagnostics.json` stores the paired Gaussian-noise ACPC basin diagnostic for all 36 LeWM canonical checkpoints. For each checkpoint it uses clean plus noised views at std 0.01..0.08, all rolled out under the same recorded action sequence. The main summary fields are `encoder_view_pair_l2_norm_by_nn`, `pred_view_pair_l2_norm_by_transition`, and `basin_contraction_pair_norm`. This artifact is intentionally Gaussian-noise-only to match the training sweep family; blur/resize are not mixed into the ACPC basin evidence.
- Some released JSON rows retain the absolute checkpoint paths from the machine that produced the artifact. Treat those fields as historical provenance only. Portable reruns should resolve checkpoints from `DATA_ROOT` plus the relative task roots above.
- `canonical_blur_baselines_20260523.json` stores blur evals of LeWM/PLDM baselines trained without input-noise augmentation for kernel sizes 3/7/11/15 on `pixels`, `goal`, and `pixels_goal`. This is an eval-only cross-corruption sanity check for the blur appendix; it is not a blur-training sweep and is not mixed into the Gaussian-noise canonical tables.
- `tools/check_paper1_consistency.py` verifies:
  - the JSON exists
  - the released structure is 4 tasks × 9 configs
  - each config contains `clean`, `pixels_std0.05`, `pixels_std0.08`, `pixels_goal_std0.05`, and `pixels_goal_std0.08` with `mean`/`std`
  - the stored `mean`/`std` agree with the 3 released seed values
  - the historical external PLDM sanity-check aggregate is trained without input-noise augmentation and recomputes its reported means/stds/drop
  - the full PLDM aggregate is 4 tasks × 9 configs, contains required eval metrics, and recomputes means/stds from the three released seed values
  - the PLDM full-diagnostics aggregate is 4 tasks × 9 configs and contains the required five-layer diagnostic fields
  - the bootstrap CI aggregate exists, has the expected n=9 / n=18 scopes, and reproduces the headline PushT CI values quoted in the paper
  - the Phase-0 ACPC aggregate is 2 methods × 4 tasks × 9 configs, all rows are `ok`, and the paired diagnostic fields used by the Phase-0 appendix are finite
  - the ACPC basin artifact covers LeWM 4 tasks × 9 configs, contains only Gaussian-noise std 0.01..0.08 variants, and stores finite encoder/prediction basin radii
  - the PLDM ACPC basin replication covers the full 4 tasks × 9 configs grid, all `ok`
  - the blur sanity-check aggregate covers 2 methods × 4 tasks × 12 blur conditions and recomputes means/stds/worst-blur drops from the three seed values
