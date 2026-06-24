#!/usr/bin/env bash
# Backfill action_effect probe for canonical 32 (8 LeWM/SWM × 4 tasks) +
# LeWM noise sweep 20 (5 noise levels × 4 tasks) = 52 ckpts.
#
# Skips the four sub-probes already on disk (noise / predictor / resolution /
# latent_noise) and only writes action_effect.{csv,json} +
# diagnostics_summary.json (the latter is rewritten with the action_effect
# fields appended).
#
# Run from the repo root:
#     STABLEWM_HOME=<root> bash tools/repr_analysis/backfill_action_effect.sh
#
# `STABLEWM_HOME` should be the directory that contains lewm-{tworooms,
# pusht,reacher,cube}/ckpt/. Optional env vars:
#     gpu                    GPU id to use (default 0)
#     dry_run=1              print commands instead of running them
#     only_tasks="tworoom pusht"  restrict tasks
#     only_method="lewm"     restrict to lewm or swm

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${SCRIPT_DIR}"

: "${STABLEWM_HOME:?Set STABLEWM_HOME to the dir containing lewm-<task>/}"
gpu="${gpu:-0}"
only_tasks="${only_tasks:-tworoom pusht reacher cube}"
only_method="${only_method:-both}"

# (task_name, sub_dir, hdf5_dataset_name, lewm_prefix, swm_prefix)
declare -A SUBDIR=(
    [tworoom]=tworooms [pusht]=pusht [reacher]=reacher [cube]=cube
)
declare -A H5NAME=(
    [tworoom]=tworoom
    [pusht]=pusht_expert_train
    [reacher]=reacher
    [cube]=ogbench/cube_single_expert
)
# canonical 8 ckpt subdir prefixes (LeWM-base uses task-specific date suffix; SWM-base + noise variants share a fixed naming pattern)
declare -A LEWM_BASE=(
    [tworoom]=tworoom_lewm_20260430
    [pusht]=pusht_lewm_20260430
    [reacher]=reacher_lewm_20260430
    [cube]=cube_lewm_20260430
)
declare -A SWM_BASE_SUFFIX=(
    [tworoom]=swm_mlp_bn_uniform_w02_t2_temporal_masked_2_dim64
    [pusht]=swm_mlp_bn_uniform_w02_t2_temporal_masked_2_dim64
    [reacher]=swm_mlp_bn_uniform_w02_t2_temporal_masked_2_dim64
    [cube]=swm_mlp_bn_uniform_w02_t2_temporal_masked_2_dim64
)
declare -A SWM_NOISE_SUFFIX=(
    [tworoom]=swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise
    [pusht]=swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise
    [reacher]=swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise
    [cube]=swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise
)
LEWM_NOISE_LEVELS="0to001 0to002 0to003 0to004 0to005 0to006 0to007 0to008"
SWM_NOISE_LEVELS="0to001 0to002 0to005"

run_one() {
    local label="$1" ckpt="$2" task="$3"
    local results_dir="$(dirname "${ckpt}")/eval_results"
    local save_dir="${results_dir}/diagnostics"
    if [ ! -f "${ckpt}" ]; then
        echo "[skip] missing ckpt: ${ckpt}"
        return
    fi
    if [ -f "${save_dir}/action_effect.json" ]; then
        echo "[skip] already done: ${save_dir}/action_effect.json"
        return
    fi
    local cmd=(
        python -m tools.repr_analysis.run_full_diagnostics
        --skip-noise --skip-predictor --skip-resolution --skip-latent-noise
        --model "${label}=${ckpt}"
        --dataset "${H5NAME[$task]}"
        --frameskip 5
        --save-dir "${save_dir}"
    )
    echo "[run] ${label} (${task}) -> ${save_dir}"
    if [ "${dry_run:-0}" = "1" ]; then
        printf '   %q ' "${cmd[@]}"; echo
    else
        CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}"
    fi
}

ckpt_path() {
    local task="$1" subdir="$2"
    echo "${STABLEWM_HOME}/lewm-${SUBDIR[$task]}/ckpt/${subdir}/${subdir}_epoch_10_object.ckpt"
}

for task in ${only_tasks}; do
    [ -z "${SUBDIR[$task]:-}" ] && { echo "[warn] unknown task $task"; continue; }

    if [ "${only_method}" != "swm" ]; then
        # LeWM-base
        run_one "LeWM-base" "$(ckpt_path "$task" "${LEWM_BASE[$task]}")" "$task"
        # LeWM noise sweep (canonical 0to001/0to002/0to005 + new 0to003/0to004/0to006/0to007/0to008)
        for lvl in ${LEWM_NOISE_LEVELS}; do
            run_one "LeWM-${lvl}-p1" \
                    "$(ckpt_path "$task" "${task}_lewm_noise_${lvl}_p1")" \
                    "$task"
        done
    fi

    if [ "${only_method}" != "lewm" ]; then
        # SWM-base
        run_one "SWM-base" \
                "$(ckpt_path "$task" "${task}_${SWM_BASE_SUFFIX[$task]}")" \
                "$task"
        # SWM noise sweep (canonical 0to001/0to002/0to005)
        for lvl in ${SWM_NOISE_LEVELS}; do
            run_one "SWM-${lvl}-p1" \
                    "$(ckpt_path "$task" "${task}_${SWM_NOISE_SUFFIX[$task]}_${lvl}_p1_dim64")" \
                    "$task"
        done
    fi
done

echo "[done] action_effect backfill"
