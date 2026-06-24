#!/usr/bin/env bash
set -euo pipefail

# Generic driver for batch representation analysis.
# Edit the variables below, then run:
#   bash tools/repr_analysis/run_repr_batch_example.sh
#
# This file must stay on Unix LF line endings. CRLF will break `bash`.
#
# MODEL_SPECS entries support two forms:
#   1. "label=model_name"
#      -> resolves to:
#         ${STABLEWM_HOME}/ckpt/${model_name}/${model_name}_epoch_${EPOCH}_object.ckpt
#   2. "label=/full/path/to/model_epoch_x_object.ckpt"
#      -> uses the full path directly

PYTHON_BIN="${PYTHON_BIN:-python3}"

# STABLEWM_HOME=/opt/huawei/explorer-env/dataset/ag_data/data/world_model/quentinll/lewm-tworooms
# DATASET=tworoom

STABLEWM_HOME="${STABLEWM_HOME:-/opt/huawei/explorer-env/dataset/ag_data/data/world_model/quentinll/lewm-pusht}"
DATASET="${DATASET:-pusht_expert_train}"
STATE_KEY="${STATE_KEY:-proprio}"
EPOCH="${EPOCH:-9}"
DATASET_PATH="${DATASET_PATH:-${STABLEWM_HOME}/${DATASET}}"
SAVE_TAG="${SAVE_TAG:-repr_batch_epoch_${EPOCH}}"
BATCH_SAVE_DIR="${BATCH_SAVE_DIR:-${STABLEWM_HOME}/repr_analysis/${SAVE_TAG}}"

PLOT_PROJECTIONS=(${PLOT_PROJECTIONS:-pca})
COMPARE_PROJECTIONS=(${COMPARE_PROJECTIONS:-pca})
COLOR_DIMS=(${COLOR_DIMS:-0 1})

# Optional flags:
#   EXPORT_TSNE=1           -> pass --export-tsne
#   COMPARE_ALL_PAIRS=1     -> compare every model pair
#   COMPARE_PAIRS='A=B C=D' -> compare only specific labels
EXPORT_TSNE="${EXPORT_TSNE:-0}"
COMPARE_ALL_PAIRS="${COMPARE_ALL_PAIRS:-0}"
COMPARE_PAIRS=(${COMPARE_PAIRS:-})

MODEL_SPECS=(
  "SIGReg=pusht_lewm_20260416"
  "BN+uniformity=pusht_swm_v0_mlp_bn_uniform_lambda_0p1_t_2_emb_dim_192_20260417"
  # "exp-b=pusht_swm_exp_b_20260418"
  # "exp-c=/abs/path/to/custom_exp_c_epoch_9_object.ckpt"
)

if [ "${#MODEL_SPECS[@]}" -lt 1 ]; then
  echo "[run_repr_batch_example] MODEL_SPECS is empty." >&2
  exit 1
fi

resolve_ckpt_path() {
  local model_ref="$1"
  if [[ "${model_ref}" == *.ckpt ]]; then
    printf '%s\n' "${model_ref}"
  else
    printf '%s\n' "${STABLEWM_HOME}/ckpt/${model_ref}/${model_ref}_epoch_${EPOCH}_object.ckpt"
  fi
}

cmd=(
  "${PYTHON_BIN}" -m tools.repr_analysis.analyze_repr
  --dataset "${DATASET_PATH}"
  --state-key "${STATE_KEY}"
  --save-dir "${BATCH_SAVE_DIR}"
  --color-dims "${COLOR_DIMS[@]}"
)

if [ "${#PLOT_PROJECTIONS[@]}" -gt 0 ]; then
  cmd+=(--plot-projections "${PLOT_PROJECTIONS[@]}")
fi

if [ "${#COMPARE_PROJECTIONS[@]}" -gt 0 ]; then
  cmd+=(--compare-projections "${COMPARE_PROJECTIONS[@]}")
fi

if [ "${EXPORT_TSNE}" = "1" ]; then
  cmd+=(--export-tsne)
fi

if [ "${COMPARE_ALL_PAIRS}" = "1" ]; then
  cmd+=(--compare-all-pairs)
fi

for pair in "${COMPARE_PAIRS[@]}"; do
  cmd+=(--compare-pair "${pair}")
done

for spec in "${MODEL_SPECS[@]}"; do
  if [[ "${spec}" != *"="* ]]; then
    echo "[run_repr_batch_example] invalid MODEL_SPECS entry: ${spec}" >&2
    exit 1
  fi

  label="${spec%%=*}"
  model_ref="${spec#*=}"
  ckpt_path="$(resolve_ckpt_path "${model_ref}")"
  cmd+=(--model "${label}=${ckpt_path}")
done

printf '[run_repr_batch_example] save_dir=%s\n' "${BATCH_SAVE_DIR}"
printf '[run_repr_batch_example] models=%s\n' "${#MODEL_SPECS[@]}"
printf '[run_repr_batch_example] command:'
for token in "${cmd[@]}"; do
  printf ' %q' "${token}"
done
printf '\n'

"${cmd[@]}"
