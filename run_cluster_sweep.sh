#!/usr/bin/env bash

# One-shot multi-node sweep launcher for the current paper matrix.
#
# This script does not change training semantics. Each worker process claims a
# subset of jobs and runs the existing run_trainer.sh once per job.
#
# On the ModelArts/Volcano-style launcher used by AReaL_v1/run_trainer_mtp.sh,
# the following env vars are detected automatically:
#   MA_NUM_HOSTS, MA_NUM_GPUS, VC_TASK_INDEX, VC_WORKER_HOSTS
#
# Typical SLURM usage from inside an allocation:
#   cluster_tag=20260505 cluster_seeds="3072 3073 3074" \
#     srun --nodes=4 --ntasks=4 --ntasks-per-node=1 bash run_cluster_sweep.sh
#
# Or submit/run this script once inside an allocated SLURM job; by default it
# auto-launches one worker per node via srun when possible:
#   cluster_tag=20260505 cluster_seeds="3072 3073 3074" bash run_cluster_sweep.sh
#
# Manual split, useful for dry runs or non-SLURM launchers:
#   cluster_num_nodes=4 cluster_node_rank=0 cluster_dry_run=1 bash run_cluster_sweep.sh
#
# Key env/CLI knobs (CLI accepts key=value and exports them):
#   cluster_datasets="tworoom pusht reacher cube"
#   cluster_methods="lewm swm"
#   cluster_seeds="3072"
#   cluster_tag="$(date +%Y%m%d_%H%M%S)"  # set empty to reuse bare model names
#   cluster_num_eval=150
#   cluster_post_train_eval_mode=full|clean|none
#   cluster_skip_eval_sweep=1
#   cluster_skip_diagnostics=1
#   cluster_skip_existing=1
#   cluster_dry_run=1

set -u
set -o pipefail

for arg in "$@"; do
    case "${arg}" in
        --help|-h)
            sed -n '1,42p' "$0"
            exit 0
            ;;
        *=*)
            export "${arg}"
            ;;
        *)
            echo "[cluster] unknown argument: ${arg}" >&2
            echo "[cluster] pass options as key=value, or use --help" >&2
            exit 2
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}" || exit 1

cluster_suite="${cluster_suite:-p03}"
cluster_datasets="${cluster_datasets:-tworoom pusht reacher cube}"
cluster_methods="${cluster_methods:-lewm swm}"
cluster_seeds="${cluster_seeds:-${seed:-3072}}"
cluster_num_eval="${cluster_num_eval:-150}"
if [ "${cluster_tag+x}" != "x" ]; then
    cluster_tag="$(date +%Y%m%d_%H%M%S)"
fi
cluster_append_seed="${cluster_append_seed:-auto}"
cluster_dry_run="${cluster_dry_run:-0}"
cluster_fail_fast="${cluster_fail_fast:-0}"
cluster_skip_existing="${cluster_skip_existing:-0}"
cluster_auto_launch="${cluster_auto_launch:-1}"

cluster_datasets="${cluster_datasets//,/ }"
cluster_methods="${cluster_methods//,/ }"
cluster_seeds="${cluster_seeds//,/ }"

if [ "${cluster_suite}" != "p03" ]; then
    echo "[cluster] unsupported cluster_suite=${cluster_suite}; supported: p03" >&2
    exit 2
fi

if [ -z "${STABLEWM_HOME:-}" ]; then
    echo "[cluster] STABLEWM_HOME is required" >&2
    exit 2
fi

if [ "${cluster_dry_run}" != "1" ] && [ -z "${SWANLAB_API_KEY:-}" ]; then
    echo "[cluster] SWANLAB_API_KEY is required for real runs" >&2
    exit 2
fi

if [ ! -f "${SCRIPT_DIR}/run_trainer.sh" ]; then
    echo "[cluster] run_trainer.sh not found next to ${BASH_SOURCE[0]}" >&2
    exit 2
fi

detect_num_nodes() {
    if [ -n "${cluster_num_nodes:-}" ]; then
        printf '%s\n' "${cluster_num_nodes}"
    elif [ -n "${MA_NUM_HOSTS:-}" ]; then
        printf '%s\n' "${MA_NUM_HOSTS}"
    elif [ -n "${SLURM_JOB_NUM_NODES:-}" ]; then
        printf '%s\n' "${SLURM_JOB_NUM_NODES}"
    elif [ -n "${SLURM_NNODES:-}" ]; then
        printf '%s\n' "${SLURM_NNODES}"
    elif [ -n "${OMPI_COMM_WORLD_SIZE:-}" ]; then
        printf '%s\n' "${OMPI_COMM_WORLD_SIZE}"
    elif [ -n "${PMI_SIZE:-}" ]; then
        printf '%s\n' "${PMI_SIZE}"
    else
        printf '1\n'
    fi
}

detect_node_rank() {
    if [ -n "${cluster_node_rank:-}" ]; then
        printf '%s\n' "${cluster_node_rank}"
    elif [ -n "${VC_TASK_INDEX:-}" ]; then
        printf '%s\n' "${VC_TASK_INDEX}"
    elif [ -n "${SLURM_NODEID:-}" ]; then
        printf '%s\n' "${SLURM_NODEID}"
    elif [ -n "${SLURM_PROCID:-}" ]; then
        printf '%s\n' "${SLURM_PROCID}"
    elif [ -n "${OMPI_COMM_WORLD_RANK:-}" ]; then
        printf '%s\n' "${OMPI_COMM_WORLD_RANK}"
    elif [ -n "${PMI_RANK:-}" ]; then
        printf '%s\n' "${PMI_RANK}"
    else
        printf '0\n'
    fi
}

detect_gpus_per_node() {
    if [ -n "${cluster_gpus_per_node:-}" ]; then
        printf '%s\n' "${cluster_gpus_per_node}"
    elif [ -n "${MA_NUM_GPUS:-}" ]; then
        printf '%s\n' "${MA_NUM_GPUS}"
    elif [ -n "${SLURM_GPUS_ON_NODE:-}" ] && [[ "${SLURM_GPUS_ON_NODE}" =~ ^[0-9]+$ ]]; then
        printf '%s\n' "${SLURM_GPUS_ON_NODE}"
    elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        local visible="${CUDA_VISIBLE_DEVICES//,/ }"
        local count=0
        local device_id
        for device_id in ${visible}; do
            if [ -n "${device_id}" ]; then
                count=$((count + 1))
            fi
        done
        printf '%s\n' "${count}"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l | tr -d ' '
    else
        printf '0\n'
    fi
}

detect_worker_hosts() {
    if [ -n "${cluster_worker_hosts:-}" ]; then
        printf '%s\n' "${cluster_worker_hosts}"
    elif [ -n "${VC_WORKER_HOSTS:-}" ]; then
        printf '%s\n' "${VC_WORKER_HOSTS}"
    elif [ -n "${SLURM_JOB_NODELIST:-}" ]; then
        printf '%s\n' "${SLURM_JOB_NODELIST}"
    elif [ -n "${SLURM_NODELIST:-}" ]; then
        printf '%s\n' "${SLURM_NODELIST}"
    else
        printf '\n'
    fi
}

detect_master_addr() {
    if [ -n "${cluster_master_addr:-}" ]; then
        printf '%s\n' "${cluster_master_addr}"
    elif [ -n "${VC_WORKER_HOSTS:-}" ]; then
        printf '%s\n' "${VC_WORKER_HOSTS%%,*}"
    elif [ -n "${MASTER_ADDR:-}" ]; then
        printf '%s\n' "${MASTER_ADDR}"
    else
        hostname
    fi
}

resolve_host_ip() {
    local host="$1"
    local resolved=""
    if [ -z "${host}" ]; then
        printf '\n'
        return 0
    fi
    if command -v getent >/dev/null 2>&1; then
        resolved="$(getent hosts "${host}" | awk 'NR==1 {print $1; exit}')"
    fi
    if [ -z "${resolved}" ] && command -v python3 >/dev/null 2>&1; then
        resolved="$(python3 -c "import socket; print(socket.gethostbyname('${host}'))" 2>/dev/null || true)"
    fi
    printf '%s\n' "${resolved:-${host}}"
}

detect_device() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        printf 'cuda\n'
    elif command -v npu-smi >/dev/null 2>&1; then
        printf 'npu\n'
    else
        printf 'none\n'
    fi
}

detected_num_nodes="$(detect_num_nodes)"

if [ "${cluster_auto_launch}" = "1" ] \
    && [ "${cluster_worker:-0}" != "1" ] \
    && [ -z "${SLURM_PROCID:-}" ] \
    && [ -z "${MA_NUM_HOSTS:-}" ] \
    && [ -n "${SLURM_JOB_ID:-}" ] \
    && command -v srun >/dev/null 2>&1 \
    && [ "${detected_num_nodes}" -gt 1 ]; then
    echo "[cluster] auto-launching ${detected_num_nodes} workers via srun"
    export cluster_worker=1
    srun --nodes="${detected_num_nodes}" \
        --ntasks="${detected_num_nodes}" \
        --ntasks-per-node=1 \
        bash "${SCRIPT_DIR}/run_cluster_sweep.sh"
    exit $?
fi

cluster_num_nodes="$(detect_num_nodes)"
cluster_node_rank="$(detect_node_rank)"
cluster_gpus_per_node="$(detect_gpus_per_node)"
cluster_worker_hosts="$(detect_worker_hosts)"
cluster_master_addr="$(detect_master_addr)"
cluster_master_ip="$(resolve_host_ip "${cluster_master_addr}")"
cluster_device="$(detect_device)"

if ! [[ "${cluster_num_nodes}" =~ ^[0-9]+$ ]] || [ "${cluster_num_nodes}" -lt 1 ]; then
    echo "[cluster] invalid cluster_num_nodes=${cluster_num_nodes}" >&2
    exit 2
fi
if ! [[ "${cluster_node_rank}" =~ ^[0-9]+$ ]] || [ "${cluster_node_rank}" -ge "${cluster_num_nodes}" ]; then
    echo "[cluster] invalid cluster_node_rank=${cluster_node_rank} for cluster_num_nodes=${cluster_num_nodes}" >&2
    exit 2
fi
if ! [[ "${cluster_gpus_per_node}" =~ ^[0-9]+$ ]]; then
    echo "[cluster] invalid cluster_gpus_per_node=${cluster_gpus_per_node}" >&2
    exit 2
fi
cluster_total_gpus=$((cluster_gpus_per_node * cluster_num_nodes))

cluster_log_dir="${cluster_log_dir:-${STABLEWM_HOME%/}/cluster_sweep_logs/${cluster_suite}_${cluster_tag:-untagged}}"
mkdir -p "${cluster_log_dir}" || exit 1

list_has() {
    local needle="$1"
    shift
    local item
    for item in "$@"; do
        if [ "${item}" = "all" ] || [ "${item}" = "${needle}" ]; then
            return 0
        fi
    done
    return 1
}

dataset_dirname_for() {
    case "$1" in
        tworoom) printf 'tworooms\n' ;;
        pusht) printf 'pusht\n' ;;
        cube) printf 'cube\n' ;;
        reacher) printf 'reacher\n' ;;
        *) return 1 ;;
    esac
}

tagged_name() {
    local base="$1"
    local seed_value="$2"
    local seed_count="$3"
    local name="${base}"

    if [ -n "${cluster_tag}" ]; then
        name="${name}_${cluster_tag}"
    fi

    case "${cluster_append_seed}" in
        1|true|yes|on)
            name="${name}_s${seed_value}"
            ;;
        auto)
            if [ "${seed_count}" -gt 1 ]; then
                name="${name}_s${seed_value}"
            fi
            ;;
    esac

    printf '%s\n' "${name}"
}

JOBS=()

add_job() {
    local label="$1"
    local method="$2"
    local dataset="$3"
    local trainer_file="$4"
    local config_name="$5"
    local output_suffix="$6"
    local extra_env="${7:-}"

    if ! list_has "${dataset}" ${cluster_datasets}; then
        return 0
    fi
    if ! list_has "${method}" ${cluster_methods}; then
        return 0
    fi

    JOBS+=("${label}|${dataset}|${trainer_file}|${config_name}|${output_suffix}|${extra_env}")
}

add_lewm() {
    local dataset="$1"
    local seed_value="$2"
    local seed_count="$3"
    local label_suffix="$4"
    local name_suffix="$5"
    local extra_env="${6:-}"
    local output_suffix
    output_suffix="$(tagged_name "lewm${name_suffix}" "${seed_value}" "${seed_count}")"
    add_job "${dataset}:lewm${label_suffix}:seed${seed_value}" \
        "lewm" "${dataset}" "train.py" "lewm" "${output_suffix}" "${extra_env}"
}

add_swm() {
    local dataset="$1"
    local seed_value="$2"
    local seed_count="$3"
    local label_suffix="$4"
    local name_mid="$5"
    local extra_env="${6:-}"
    local output_suffix
    local swm_base="swm_mlp_bn_uniform_w02_t2_temporal_masked_2${name_mid}_dim64"
    local swm_env="encoder_projection_head_type=mlp loss_regularizer_type=uniformity loss_regularizer_weight=0.2 loss_regularizer_t=2.0 loss_uniformity_mode=temporal_masked loss_uniformity_temporal_exclusion=2 wm_embed_dim=64"
    if [ -n "${extra_env}" ]; then
        swm_env="${swm_env} ${extra_env}"
    fi
    output_suffix="$(tagged_name "${swm_base}" "${seed_value}" "${seed_count}")"
    add_job "${dataset}:swm${label_suffix}:seed${seed_value}" \
        "swm" "${dataset}" "train_swm.py" "swm" "${output_suffix}" "${swm_env}"
}

add_seed_matrix() {
    local seed_value="$1"
    local seed_count="$2"
    local fixed_std="image_noise_std_min=0.005 image_noise_std_max=0.005 image_noise_noise_prob=1.0"
    local n001_p05="image_noise_std_min=0.0 image_noise_std_max=0.001 image_noise_noise_prob=0.5"
    local n001_p1="image_noise_std_min=0.0 image_noise_std_max=0.001 image_noise_noise_prob=1.0"
    local n002_p05="image_noise_std_min=0.0 image_noise_std_max=0.002 image_noise_noise_prob=0.5"
    local n002_p1="image_noise_std_min=0.0 image_noise_std_max=0.002 image_noise_noise_prob=1.0"
    local n005_p05="image_noise_std_min=0.0 image_noise_std_max=0.005 image_noise_noise_prob=0.5"
    local n005_p1="image_noise_std_min=0.0 image_noise_std_max=0.005 image_noise_noise_prob=1.0"

    add_lewm "tworoom" "${seed_value}" "${seed_count}" "-base" "" ""
    add_lewm "tworoom" "${seed_value}" "${seed_count}" "-fixed-std" "_noise_std_0_005" "${fixed_std}"
    add_lewm "tworoom" "${seed_value}" "${seed_count}" "-perframe-0to005-p05" "_noise_0to005_p05" "${n005_p05}"
    add_lewm "tworoom" "${seed_value}" "${seed_count}" "-perframe-0to005-p1" "_noise_0to005_p1" "${n005_p1}"
    add_swm "tworoom" "${seed_value}" "${seed_count}" "-base" "" ""
    add_swm "tworoom" "${seed_value}" "${seed_count}" "-fixed-std" "_noise_std0_005" "${fixed_std}"
    add_swm "tworoom" "${seed_value}" "${seed_count}" "-perframe-0to005-p05" "_noise_0to005_p05" "${n005_p05}"
    add_swm "tworoom" "${seed_value}" "${seed_count}" "-perframe-0to005-p1" "_noise_0to005_p1" "${n005_p1}"

    add_lewm "pusht" "${seed_value}" "${seed_count}" "-base" "" ""
    add_lewm "pusht" "${seed_value}" "${seed_count}" "-fixed-std" "_noise_std_0_005" "${fixed_std}"
    add_lewm "pusht" "${seed_value}" "${seed_count}" "-perframe-0to001-p1" "_noise_0to001_p1" "${n001_p1}"
    add_lewm "pusht" "${seed_value}" "${seed_count}" "-perframe-0to002-p1" "_noise_0to002_p1" "${n002_p1}"
    add_lewm "pusht" "${seed_value}" "${seed_count}" "-perframe-0to005-p1" "_noise_0to005_p1" "${n005_p1}"
    add_swm "pusht" "${seed_value}" "${seed_count}" "-base" "" ""
    add_swm "pusht" "${seed_value}" "${seed_count}" "-fixed-std" "_noise_std0_005" "${fixed_std}"
    add_swm "pusht" "${seed_value}" "${seed_count}" "-perframe-0to001-p05" "_noise_0to001_p05" "${n001_p05}"
    add_swm "pusht" "${seed_value}" "${seed_count}" "-perframe-0to001-p1" "_noise_0to001_p1" "${n001_p1}"
    add_swm "pusht" "${seed_value}" "${seed_count}" "-perframe-0to002-p05" "_noise_0to002_p05" "${n002_p05}"
    add_swm "pusht" "${seed_value}" "${seed_count}" "-perframe-0to002-p1" "_noise_0to002_p1" "${n002_p1}"

    for dataset in reacher cube; do
        add_lewm "${dataset}" "${seed_value}" "${seed_count}" "-base" "" ""
        add_lewm "${dataset}" "${seed_value}" "${seed_count}" "-perframe-0to002-p05" "_noise_0to002_p05" "${n002_p05}"
        add_lewm "${dataset}" "${seed_value}" "${seed_count}" "-perframe-0to002-p1" "_noise_0to002_p1" "${n002_p1}"
        add_lewm "${dataset}" "${seed_value}" "${seed_count}" "-perframe-0to005-p05" "_noise_0to005_p05" "${n005_p05}"
        add_lewm "${dataset}" "${seed_value}" "${seed_count}" "-perframe-0to005-p1" "_noise_0to005_p1" "${n005_p1}"
        add_swm "${dataset}" "${seed_value}" "${seed_count}" "-base" "" ""
        add_swm "${dataset}" "${seed_value}" "${seed_count}" "-perframe-0to002-p05" "_noise_0to002_p05" "${n002_p05}"
        add_swm "${dataset}" "${seed_value}" "${seed_count}" "-perframe-0to002-p1" "_noise_0to002_p1" "${n002_p1}"
        add_swm "${dataset}" "${seed_value}" "${seed_count}" "-perframe-0to005-p05" "_noise_0to005_p05" "${n005_p05}"
        add_swm "${dataset}" "${seed_value}" "${seed_count}" "-perframe-0to005-p1" "_noise_0to005_p1" "${n005_p1}"
    done
}

read -r -a seed_values <<< "${cluster_seeds}"
if [ "${#seed_values[@]}" -eq 0 ]; then
    echo "[cluster] cluster_seeds is empty" >&2
    exit 2
fi

for seed_value in "${seed_values[@]}"; do
    add_seed_matrix "${seed_value}" "${#seed_values[@]}"
done

total_jobs="${#JOBS[@]}"
if [ "${total_jobs}" -eq 0 ]; then
    echo "[cluster] no jobs selected; check cluster_datasets/cluster_methods" >&2
    exit 2
fi

summary_file="${cluster_log_dir}/node_${cluster_node_rank}_summary.tsv"
printf 'job_index\tstatus\tlabel\tdataset\toutput_model_name\tlog\n' > "${summary_file}"

normalize_dataset_root() {
    local dataset="$1"
    local dirname
    dirname="$(dataset_dirname_for "${dataset}")" || return 1
    if [[ "$(basename "${STABLEWM_HOME}")" == lewm-* ]]; then
        printf '%s/lewm-%s\n' "$(dirname "${STABLEWM_HOME}")" "${dirname}"
    else
        printf '%s/lewm-%s\n' "${STABLEWM_HOME%/}" "${dirname}"
    fi
}

should_export_cluster_var() {
    local name="$1"
    [ "${!name+x}" = "x" ]
}

export_optional_runtime_vars() {
    if should_export_cluster_var cluster_eval_gpus; then export eval_gpus="${cluster_eval_gpus}"; fi
    if should_export_cluster_var cluster_post_train_eval_mode; then export post_train_eval_mode="${cluster_post_train_eval_mode}"; fi
    if should_export_cluster_var cluster_eval_corruption_stds; then export eval_corruption_stds="${cluster_eval_corruption_stds}"; fi
    if should_export_cluster_var cluster_eval_corruption_apply_to; then export eval_corruption_apply_to="${cluster_eval_corruption_apply_to}"; fi
    if should_export_cluster_var cluster_noise_table_stds; then export noise_table_stds="${cluster_noise_table_stds}"; fi
    if should_export_cluster_var cluster_diagnostic_rollout_steps; then export diagnostic_rollout_steps="${cluster_diagnostic_rollout_steps}"; fi
    if should_export_cluster_var cluster_frameskip; then export frameskip="${cluster_frameskip}"; fi
    if should_export_cluster_var cluster_eval_epoch; then export eval_epoch="${cluster_eval_epoch}"; fi
    if should_export_cluster_var cluster_diagnostic_dataset_name; then export diagnostic_dataset_name="${cluster_diagnostic_dataset_name}"; fi

    if should_export_cluster_var cluster_skip_eval_sweep; then export skip_eval_sweep="${cluster_skip_eval_sweep}"; fi
    if should_export_cluster_var cluster_skip_diagnostics; then export skip_diagnostics="${cluster_skip_diagnostics}"; fi
    if should_export_cluster_var cluster_skip_noise_table; then export skip_noise_table="${cluster_skip_noise_table}"; fi
    if should_export_cluster_var cluster_diagnostic_skip_predictor; then export diagnostic_skip_predictor="${cluster_diagnostic_skip_predictor}"; fi
    if should_export_cluster_var cluster_diagnostic_skip_resolution; then export diagnostic_skip_resolution="${cluster_diagnostic_skip_resolution}"; fi
}

run_job() {
    local job_index="$1"
    local spec="$2"
    local label dataset trainer_file config_name output_suffix extra_env
    IFS='|' read -r label dataset trainer_file config_name output_suffix extra_env <<< "${spec}"

    local dataset_root
    dataset_root="$(normalize_dataset_root "${dataset}")" || return 1
    local final_model_name="${dataset}_${output_suffix}"
    local results_summary="${dataset_root}/ckpt/${final_model_name}/eval_results/summary.txt"
    local log_file="${cluster_log_dir}/node_${cluster_node_rank}_job_${job_index}_${label//[:\/]/_}.log"

    if [ "${cluster_skip_existing}" = "1" ] && [ -f "${results_summary}" ]; then
        echo "[cluster] skip existing ${label} -> ${final_model_name}"
        printf '%s\tskipped\t%s\t%s\t%s\t%s\n' \
            "${job_index}" "${label}" "${dataset}" "${final_model_name}" "${log_file}" >> "${summary_file}"
        return 0
    fi

    if [ "${cluster_dry_run}" = "1" ]; then
        echo "[cluster][dry-run] job ${job_index}/${total_jobs}: ${label}"
        echo "  dataset_name=${dataset} trainer_file=${trainer_file} config=${config_name} output_model_name=${output_suffix} seed=${label##*seed}"
        echo "  extra_env=${extra_env:-<none>}"
        printf '%s\tdry-run\t%s\t%s\t%s\t%s\n' \
            "${job_index}" "${label}" "${dataset}" "${final_model_name}" "${log_file}" >> "${summary_file}"
        return 0
    fi

    echo "[cluster] start job ${job_index}/${total_jobs}: ${label} -> ${final_model_name}"
    local rc
    (
        unset encoder_projection_head_type
        unset loss_regularizer_type loss_regularizer_weight loss_regularizer_scope
        unset loss_regularizer_t loss_uniformity_mode loss_uniformity_temporal_exclusion
        unset loss_temporal_hinge_weight loss_temporal_hinge_margin loss_temporal_hinge_squared
        unset loss_temporal_hinge_dynamic_enabled loss_temporal_hinge_dynamic_base_margin
        unset loss_temporal_hinge_dynamic_min_margin loss_temporal_hinge_dynamic_max_margin
        unset loss_inverse_dynamics_weight loss_transition_distance_weight
        unset loss_pred_space loss_pred_type loss_rollout_weight loss_rollout_steps
        unset wm_embed_dim wm_inference_rollout_state_space wm_inference_cost_space wm_inference_cost_type
        unset image_noise_std_min image_noise_std_max image_noise_noise_prob image_noise_apply_to_val
        unset eval_gpus post_train_eval_mode eval_corruption_stds eval_corruption_apply_to
        unset noise_table_stds diagnostic_rollout_steps frameskip eval_epoch diagnostic_dataset_name
        unset skip_eval_sweep skip_diagnostics skip_noise_table diagnostic_skip_predictor diagnostic_skip_resolution

        export dataset_name="${dataset}"
        export trainer_file="${trainer_file}"
        export config="${config_name}"
        export output_model_name="${output_suffix}"
        export num_eval="${cluster_num_eval}"
        export seed="${label##*seed}"

        export_optional_runtime_vars

        local extra_vars=()
        if [ -n "${extra_env}" ]; then
            read -r -a extra_vars <<< "${extra_env}"
        fi
        local kv
        for kv in "${extra_vars[@]}"; do
            export "${kv}"
        done

        bash "${SCRIPT_DIR}/run_trainer.sh"
    ) > "${log_file}" 2>&1
    rc=$?

    if [ "${rc}" -eq 0 ]; then
        echo "[cluster] done  job ${job_index}/${total_jobs}: ${label}"
        printf '%s\tdone\t%s\t%s\t%s\t%s\n' \
            "${job_index}" "${label}" "${dataset}" "${final_model_name}" "${log_file}" >> "${summary_file}"
    else
        echo "[cluster] FAIL  job ${job_index}/${total_jobs}: ${label} (rc=${rc}; log=${log_file})"
        printf '%s\tfailed:%s\t%s\t%s\t%s\t%s\n' \
            "${job_index}" "${rc}" "${label}" "${dataset}" "${final_model_name}" "${log_file}" >> "${summary_file}"
    fi

    return "${rc}"
}

echo "[cluster] suite=${cluster_suite} jobs=${total_jobs} seeds=${cluster_seeds}"
echo "[cluster] worker_hosts=${cluster_worker_hosts:-<unknown>}"
echo "[cluster] master_addr=${cluster_master_addr:-<unknown>} master_ip=${cluster_master_ip:-<unknown>}"
echo "[cluster] node_rank=${cluster_node_rank}/${cluster_num_nodes} gpus_per_node=${cluster_gpus_per_node} total_gpus=${cluster_total_gpus} device=${cluster_device}"
echo "[cluster] log_dir=${cluster_log_dir}"
echo "[cluster] datasets=${cluster_datasets} methods=${cluster_methods}"

failures=0
assigned=0
for job_index in "${!JOBS[@]}"; do
    if [ $((job_index % cluster_num_nodes)) -ne "${cluster_node_rank}" ]; then
        continue
    fi
    assigned=$((assigned + 1))
    if ! run_job "${job_index}" "${JOBS[${job_index}]}"; then
        failures=$((failures + 1))
        if [ "${cluster_fail_fast}" = "1" ]; then
            break
        fi
    fi
done

echo "[cluster] node ${cluster_node_rank} assigned=${assigned} failures=${failures} summary=${summary_file}"

if [ "${failures}" -gt 0 ]; then
    exit 1
fi
