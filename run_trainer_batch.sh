#!/usr/bin/env bash

# ==========================================
# Multi-node batch wrapper around run_trainer.sh.
#
# 思路：
#   - 任一受支持的 env var 都可以传入用逗号分隔的多个值；
#   - 每个 env var 要么传 1 个值（所有节点共享），要么传 N 个值（N == NNODES，
#     每个节点取自己那一份），否则报错退出；
#   - NNODES 必须等于"任意 env var 中传入值数量"的最大值；
#   - 解析完后按本节点 NODE_RANK 选一份 value，覆盖原 env var，
#     直接 exec run_trainer.sh，复用其全部逻辑。
#
# 节点环境变量沿用 AReaL_v1/run_trainer_mtp.sh 的约定：
#   MA_NUM_HOSTS    总节点数
#   VC_TASK_INDEX   当前节点 rank (0-based)
#
# 用法示例（4 节点云上任务，仅 image_noise_std_max 在节点间扫）：
#   image_noise_std_max="0.01,0.02,0.03,0.04" \
#   dataset_name=tworoom trainer_file=train_swm.py config=swm \
#   output_model_name=sweep_stdmax num_eval=50 \
#   bash run_trainer_batch.sh
#
# eval_corruption_apply_to 在云平台上建议传数字，避免 '+' 或逗号被平台解析：
#   1=pixels, 2=goal, 3=pixels+goal, 4=pixels 和 pixels+goal, 5=all。
# 例如旧写法 eval_corruption_apply_to="pixels,pixels+goal" 在 batch 入口应写成
#   eval_corruption_apply_to=4
# ==========================================

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NNODES="${MA_NUM_HOSTS:-1}"
NODE_RANK="${VC_TASK_INDEX:-0}"

echo "[batch] NNODES=${NNODES}  NODE_RANK=${NODE_RANK}"

# 所有可被 sweep 的 env var 名称。来自 run_trainer.sh 头部注释里列出的入参。
# 加入新 env var 时在此追加即可，run_trainer_batch.sh 其余逻辑不需改动。
SWEEP_VARS=(
    # 基础必填
    dataset_name trainer_file config output_model_name num_eval seed
    # 训练相关
    encoder_projection_head_type encoder_projection_head_norm_fn encoder_projection_head_hidden_dim
    loss_sigreg_weight
    loss_sigreg_warmup_type loss_sigreg_warmup_mode loss_sigreg_warmup_epochs
    loss_sigreg_warmup_decay_epochs loss_sigreg_warmup_num_proj loss_sigreg_warmup_weight
    loss_regularizer_type loss_regularizer_weight loss_regularizer_scope loss_regularizer_t
    loss_uniformity_mode loss_uniformity_temporal_exclusion
    loss_temporal_hinge_weight loss_temporal_hinge_margin loss_temporal_hinge_squared
    loss_temporal_hinge_dynamic_enabled loss_temporal_hinge_dynamic_base_margin
    loss_temporal_hinge_dynamic_min_margin loss_temporal_hinge_dynamic_max_margin
    loss_inverse_dynamics_weight loss_transition_distance_weight
    loss_target_stop_grad
    loss_pred_space pred_target target_view loss_pred_target_view loss_pred_type loss_rollout_weight loss_rollout_steps
    loss_hetero_enabled loss_hetero_mode loss_hetero_probe_weight loss_hetero_logvar_hidden_dim
    loss_hetero_s_min loss_hetero_s_max loss_hetero_tau_floor
    loss_generic_latent_consistency_enabled loss_snap_acpc_enabled loss_paired_view_control_enabled
    loss_in_forward_noise_control_enabled
    loss_action_gate_enabled loss_action_gate_mode loss_action_gate_intervention
    loss_action_gate_delta_scale loss_action_gate_num_delta_samples
    loss_action_gate_warmup_epochs loss_action_gate_ema_momentum
    loss_action_gate_w_min loss_action_gate_w_max
    loss_adaptive_consistency_enabled loss_adaptive_consistency_weight
    loss_adaptive_consistency_noise_std_min loss_adaptive_consistency_noise_std_max
    loss_adaptive_consistency_noise_prob loss_adaptive_consistency_distance
    loss_adaptive_consistency_detach_origin loss_adaptive_consistency_detach_clean
    wm_embed_dim wm_inference_rollout_state_space wm_inference_cost_space wm_inference_cost_type
    image_noise_std_min image_noise_std_max image_noise_noise_prob image_noise_apply_to_val
    # eval / 诊断
    frameskip eval_gpus eval_epoch post_train_eval_mode eval_corruption_type eval_corruption_stds eval_corruption_apply_to
    eval_blur_kernel_sizes eval_resize_factors eval_seeds eval_base_seed
    diagnostic_corruption_type
    noise_table_stds diagnostic_rollout_steps diagnostic_dataset_name
    skip_eval_sweep skip_noise_table skip_diagnostics
    diagnostic_skip_predictor diagnostic_skip_resolution
    diagnostic_skip_latent_noise diagnostic_skip_action_effect
    run_cross_check_correlations
    logger_backend swanlab_enabled
)

split_values_for_var() {
    local var_name="$1"
    local raw_value="$2"

    # Back-compat: eval_corruption_apply_to historically allowed comma-separated
    # mode lists (for example "pixels,pixels+goal"). That comma conflicts with
    # this batch wrapper's per-node sweep syntax, so string mode lists are kept
    # as one value. Numeric codes can still be swept across nodes, e.g. "1,3".
    if [ "${var_name}" = "eval_corruption_apply_to" ]; then
        case "${raw_value}" in
            *[A-Za-z+_-]*)
                printf '%s\n' "${raw_value}"
                return 0
                ;;
        esac
    fi

    local values=()
    IFS=',' read -ra values <<< "${raw_value}"
    printf '%s\n' "${values[@]}"
}

# 1) 第一遍扫描：解析每个变量的 split values，记录最大长度。
declare -A var_values_csv  # 保存原始 csv，便于后面 split
max_len=1
for v in "${SWEEP_VARS[@]}"; do
    raw="${!v-}"
    if [ -z "${raw}" ]; then
        continue
    fi
    var_values_csv[$v]="${raw}"
    # 计算 split 后的元素数量
    mapfile -t arr < <(split_values_for_var "${v}" "${raw}")
    n=${#arr[@]}
    if [ "$n" -gt "$max_len" ]; then
        max_len=$n
    fi
done

echo "[batch] max sweep length detected: ${max_len}"

# 2) 校验 NNODES 必须等于 max_len。
if [ "${NNODES}" -ne "${max_len}" ]; then
    echo "[batch][error] NNODES (${NNODES}) must equal the max sweep length (${max_len})."
    echo "[batch][error] 修复方法：调整云上节点数，或调整传入数组长度。"
    exit 1
fi

# 2.5) 当存在跨节点 sweep 时（max_len > 1），output_model_name 必须也按节点
#      数展开成 NNODES 个互不相同的值，否则多个节点会写同一个 ckpt 目录互相
#      覆盖结果。
if [ "${max_len}" -gt 1 ]; then
    omn_csv="${var_values_csv[output_model_name]:-}"
    if [ -z "${omn_csv}" ]; then
        echo "[batch][error] output_model_name 未设置，但检测到跨节点 sweep（max_len=${max_len}）。"
        echo "[batch][error] 必须为每个节点指定一个不同的 output_model_name（逗号分隔 ${NNODES} 个值）。"
        exit 1
    fi
    IFS=',' read -ra omn_arr <<< "${omn_csv}"
    if [ "${#omn_arr[@]}" -ne "${NNODES}" ]; then
        echo "[batch][error] output_model_name 只传入了 ${#omn_arr[@]} 个值，但跨节点 sweep 要求 ${NNODES} 个互不相同的值。"
        echo "[batch][error]   raw value: ${omn_csv}"
        echo "[batch][error] 否则多个节点会写到同一个 ckpt 目录互相覆盖。"
        exit 1
    fi
    # 检查是否互不相同
    declare -A _omn_seen=()
    for name in "${omn_arr[@]}"; do
        if [ -n "${_omn_seen[$name]:-}" ]; then
            echo "[batch][error] output_model_name 的 ${NNODES} 个值中出现重复：'${name}'。"
            echo "[batch][error] 每个节点必须有唯一的输出路径。"
            exit 1
        fi
        _omn_seen[$name]=1
    done
fi

# 3) 校验每个变量的值数量必须是 1 或 NNODES。
for v in "${!var_values_csv[@]}"; do
    mapfile -t arr < <(split_values_for_var "${v}" "${var_values_csv[$v]}")
    n=${#arr[@]}
    if [ "$n" -ne 1 ] && [ "$n" -ne "$NNODES" ]; then
        echo "[batch][error] env var '${v}' has ${n} values; must be 1 or ${NNODES}."
        echo "[batch][error]   raw value: ${var_values_csv[$v]}"
        exit 1
    fi
done

# 4) 按 NODE_RANK 选本节点要用的值，覆盖 env var。
echo "[batch] resolved per-node overrides for rank=${NODE_RANK}:"
for v in "${!var_values_csv[@]}"; do
    mapfile -t arr < <(split_values_for_var "${v}" "${var_values_csv[$v]}")
    n=${#arr[@]}
    if [ "$n" -eq 1 ]; then
        picked="${arr[0]}"
    else
        picked="${arr[$NODE_RANK]}"
    fi
    export "$v=${picked}"
    echo "  ${v}=${picked}"
done

# 5) 直接 exec run_trainer.sh，所有 env vars 已经被覆盖成本节点的值。
echo "[batch] launching run_trainer.sh on rank ${NODE_RANK} ..."
exec bash "${SCRIPT_DIR}/run_trainer.sh"
