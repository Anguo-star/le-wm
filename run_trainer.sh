#!/usr/bin/env bash

# ==========================================
# Train + Eval sweep + Noise table，一站式
# ==========================================
#
# 必填 env vars (与旧脚本一致):
#   dataset_name              tworoom | pusht | cube | reacher
#   trainer_file              train.py | train_swm.py
#   config                    swm | lewm (also accepts swm.yaml | lewm.yaml)
#   output_model_name         模型名后缀（最终落盘 `${dataset_name}_${output_model_name}`）
#   num_eval                  eval 总 episode 数量（多 seed 时会平分到每个 seed）
#   STABLEWM_HOME             checkpoint 根目录
#
# 可选 Hydra overrides (与旧脚本一致，留空则不下发):
#   encoder_projection_head_type, encoder_projection_head_norm_fn,
#   encoder_projection_head_hidden_dim,
#   loss_sigreg_weight (lewm anti-collapse, default 0.09),
#   loss_sigreg_warmup_type (lewm-only; none | wasserstein. Adds a scale-aware
#                            sliced-Wasserstein companion to SIGReg; intended
#                            for BN-free training where SIGReg gets stuck; see
#                            notes_lewm_bn_removal.md §3.1/§5),
#   loss_sigreg_warmup_mode (replace | add_decay; default replace.
#                            replace   = Wass replaces SIGReg during warmup,
#                                        only SIGReg after (legacy, has a
#                                        phase switch that perturbs pred_loss).
#                            add_decay = SIGReg always on; Wass at full weight
#                                        during warmup_epochs, then linear
#                                        decay over decay_epochs to 0 (smooth)),
#   loss_sigreg_warmup_epochs (epochs at full Wass weight),
#   loss_sigreg_warmup_decay_epochs (linear-decay epochs after warmup;
#                                     add_decay only, ignored for replace),
#   loss_sigreg_warmup_num_proj (random projections for the Wasserstein term),
#   loss_sigreg_warmup_weight (Wass term weight; empty reuses loss_sigreg_weight),
#   loss_regularizer_*, loss_uniformity_*,
#   loss_temporal_hinge_*, loss_inverse_dynamics_weight,
#   loss_transition_distance_weight, loss_pred_*, loss_rollout_*,
#   loss_target_stop_grad (lewm-only, SimSiam-style stop-grad on target),
#   loss_hetero_enabled (lewm-only sigma head; mode=loss for hetero NLL,
#                         mode=probe for detached sigma_probe_loss),
#   loss_hetero_mode, loss_hetero_probe_weight,
#   loss_hetero_logvar_hidden_dim, loss_hetero_s_min, loss_hetero_s_max,
#   loss_hetero_tau_floor,
#   loss_generic_latent_consistency_enabled (lewm-only GLC baseline; when true,
#                                            image noise is applied in-forward
#                                            as clean/noisy paired views),
#   loss_snap_acpc_enabled (lewm-only one-step action-conditioned predictive
#                           consistency baseline; paired clean/noisy views,
#                           default off),
#   loss_paired_view_control_enabled (lewm-only paired clean/noisy no-aux
#                                     diagnostic; bypasses TransformDataset but
#                                     adds no auxiliary loss),
#   loss_in_forward_noise_control_enabled (lewm-only noisy-only diagnostic;
#                                          bypasses TransformDataset and applies
#                                          configured image noise in forward),
#   loss_action_gate_enabled (lewm-only logging-only adaptive resolution
#                             gate; rides on loss_hetero_mode=probe),
#   loss_action_gate_mode (full | sigma_only; sigma_only skips A_t perturb
#                          and sets critical=gS*0.5, requires hetero_mode=probe),
#   loss_action_gate_intervention (none | shuffle_sigma | shuffle_action |
#                                  random_gate | constant_w; causal-necessity
#                                  controls, see plan_adaptive_resolution.md
#                                  §3.8.1 P0-2),
#   loss_action_gate_delta_scale, loss_action_gate_num_delta_samples,
#   loss_action_gate_warmup_epochs, loss_action_gate_ema_momentum,
#   loss_action_gate_w_min, loss_action_gate_w_max,
#   loss_adaptive_consistency_enabled, loss_adaptive_consistency_weight,
#   loss_adaptive_consistency_noise_std_min/max,
#   loss_adaptive_consistency_noise_prob, loss_adaptive_consistency_distance,
#   loss_adaptive_consistency_detach_origin
#   (legacy loss_adaptive_consistency_detach_clean is still accepted),
#   pred_target               prediction target view:
#                              perturbed = target from configured perturbed future view,
#                              origin = target from unperturbed/original future view.
#                              target_view/loss_pred_target_view are aliases.
#   seed, wm_embed_dim, wm_inference_*, image_noise_std_min/max/apply_to_val
#
# 新增 env vars:
#   image_noise_noise_prob    每帧加噪概率 (默认 1.0；<1 制造 unperturbed+noisy 混合)
#   post_train_eval_mode      训练后执行模式：full | origin | none
#                              full  = eval sweep + full diagnostics（默认，旧行为）
#                              origin = 只跑 unperturbed eval，不跑 corruption sweep/diagnostics
#                              none  = 不跑 eval/diagnostics
#   eval_corruption_type      eval sweep 损坏家族：gaussian_noise (默认) | gaussian_blur | resize
#                              gaussian_noise → 用 eval_corruption_stds 作为 std 列表
#                              gaussian_blur  → 用 eval_blur_kernel_sizes 作为核大小列表（奇数像素）
#                              resize         → 用 eval_resize_factors 作为 factor 列表
#                              输出文件名 tag 跟 std-类不冲突：
#                                gaussian_noise → ..._std<X>_seed<N>          (历史不变)
#                                gaussian_blur  → ..._blur_ks<K>_seed<N>
#                                resize         → ..._rs_factor<X>_seed<N>
#   eval_corruption_stds      eval sweep 噪声列表（gaussian_noise），空格分隔
#                              默认 "0.0 0.03 0.05 0.08"
#                              传 "" 跳过 eval sweep（仍跑 noise table）
#   eval_blur_kernel_sizes    eval sweep 模糊核大小列表（gaussian_blur 时使用）
#                              默认 "1 3 7 15"（1 = unperturbed / no-op；偶数会向上取奇）
#   eval_resize_factors       eval sweep resize factor 列表（resize 时使用）
#                              默认 "1.0 0.75 0.5 0.25"（1.0 = unperturbed）
#   eval_corruption_apply_to  eval sweep 加噪目标。推荐在训练平台上传数字，避免 '+'
#                              或逗号被平台解析错误：
#                                1 = pixels（paper primary: observation-only,
#                                    unperturbed goal；默认）
#                                2 = goal
#                                3 = pixels+goal（同一 eval 同时加噪 observation 与 goal）
#                                4 = pixels,pixels+goal（同时跑 primary 和 auxiliary stress）
#                                5 = pixels,goal,pixels+goal（全模式）
#                              旧字符串仍兼容：pixels / goal / pixels+goal /
#                              pixels_goal / "pixels,pixels+goal"。
#   frameskip                 数据加载 frameskip；默认 5（与训练 data config 一致）
#   eval_gpus                 GPU id 列表，空格分隔；默认自动探测全部
#   noise_table_stds          诊断扫的 std；默认 0.0~0.10 一组（仍由本字段控制）
#   diagnostic_rollout_steps  predictor 自回归 rollout 步数；默认 "1 2 4 8"
#   skip_eval_sweep           设 1 跳过 eval sweep
#   skip_noise_table          legacy 名，等价于 skip_diagnostics
#   skip_diagnostics          设 1 跳过整套诊断（noise/predictor/resolution）
#   diagnostic_skip_predictor 设 1 仅跳过 predictor_sensitivity
#   diagnostic_skip_resolution 设 1 仅跳过 task_resolution
#   diagnostic_skip_latent_noise 设 1 仅跳过 latent_noise_sensitivity (P5)
#   diagnostic_skip_action_effect 设 1 仅跳过 action_effect probe
#   # 注：分布投影（PCA / t-SNE）已在 tools/repr_analysis/analyze_repr.py 与
#   # plot_repr.py 中提供，独立调用即可，不进 run_full_diagnostics 默认链。
#   run_cross_check_correlations=1 训练 + 诊断完成后，对当前任务再跑一次
#                                  P0.5b cross-check（LeWM/SWM 配对 + within-method
#                                  + partial|std + partial|method + pairS + top-bot），
#                                  缺失 ckpt 自动跳过；输出 cross_check_corr.json
#                                  到 ${results_dir}/diagnostics/。
#   eval_epoch                用于 eval 的 epoch 编号；默认读取训练 config 的 trainer.max_epochs
#   eval_seeds                eval sweep 的 seed 数量；默认 3。每个 seed 跑 num_eval/eval_seeds 次
#   eval_base_seed            首 seed；不传则读取 config/eval/<dataset_name>.yaml 顶层 seed；后续 seed = base+1, base+2, ...
#
# 用法示例：
#   dataset_name=tworoom trainer_file=train_swm.py config=swm \
#     output_model_name=perframe_0to05_p1 num_eval=50 \
#     image_noise_std_min=0.0 image_noise_std_max=0.05 image_noise_noise_prob=1.0 \
#     eval_corruption_stds="0.0 0.05 0.08" \
#     bash run_trainer.sh
#
# 在结果目录会得到：
#   eval_results/<label>.log              每个 eval 的完整 stdout
#   eval_results/<label>_results.txt      该 eval 的 metrics 文本
#   eval_results/diagnostics/             noise + predictor + task_resolution
#       noise_sensitivity.{csv,json}
#       geometry_summary.{csv,json}
#       predictor_sensitivity.{csv,json}
#       task_resolution.{csv,json}
#       action_effect.{csv,json}
#       diagnostics_summary.json          per-checkpoint 一行 roll-up
#       *.png                             curves & geometry tradeoff plots
#   eval_results/summary.txt              所有 eval + diagnostics 的摘要
# ==========================================

set -u  # treat unset vars as errors after the unsets below
set -o pipefail

# 切到脚本所在目录，确保所有相对路径（config/、tools/ 等）一致解析，
# 避免从其它 cwd 调用本脚本时 diagnostics 读不到 train data config。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ---------- 1. Hydra 参数构建 (沿用原脚本) ----------
CMD_ARGS=()
add_override() {
    local key="$1"
    local value="${2:-}"
    if [ -n "$value" ]; then
        CMD_ARGS+=("$key=$value")
    fi
}

normalize_eval_corruption_apply_to() {
    local raw="${1:-1}"
    local compact="${raw//[[:space:]]/}"
    local token
    local out=()

    normalize_one_apply_mode() {
        case "$1" in
            1|pixel|pixels|obs|observation)
                echo "pixels"
                ;;
            2|goal)
                echo "goal"
                ;;
            3|both|pixels+goal|pixels_goal|pixels-goal|pixelsgoal|all_streams)
                echo "pixels+goal"
                ;;
            *)
                return 1
                ;;
        esac
    }

    case "$compact" in
        ""|1|pixel|pixels|obs|observation)
            echo "pixels"
            return 0
            ;;
        2|goal)
            echo "goal"
            return 0
            ;;
        3|both|pixels+goal|pixels_goal|pixels-goal|pixelsgoal|all_streams)
            echo "pixels+goal"
            return 0
            ;;
        4|primary_aux|primary+aux|primary_auxiliary|pixels,pixels+goal|pixels,pixels_goal)
            echo "pixels,pixels+goal"
            return 0
            ;;
        5|all)
            echo "pixels,goal,pixels+goal"
            return 0
            ;;
    esac

    IFS=',' read -ra _apply_tokens <<< "$compact"
    for token in "${_apply_tokens[@]}"; do
        local mode
        if ! mode="$(normalize_one_apply_mode "$token")"; then
            echo "[eval] invalid eval_corruption_apply_to token '${token}' in '${raw}'" >&2
            echo "[eval] use numeric codes: 1=pixels, 2=goal, 3=pixels+goal, 4=pixels plus pixels+goal, 5=all" >&2
            return 1
        fi
        out+=("$mode")
    done

    local IFS=','
    echo "${out[*]}"
}

# Backward-compatible Hydra config name. Older callers may pass `lewm.yaml`;
# internally we need the stem for both config/train/<name>.yaml lookup and
# `--config-name=<name>`.
config_name="${config##*/}"
config_name="${config_name%.yaml}"
config_name="${config_name%.yml}"

# data: hydra data group; dataset_dirname: STABLEWM_HOME/lewm-<dirname>;
# default_h5_name: 真实 HDF5 dataset 名称（必须与 config/train/data/<data>.yaml
# 和 config/eval/<dataset_name>.yaml 中的 name 一致）。在此显式列出避免依赖
# cwd 的 grep/sed 解析。
case "${dataset_name}" in
    tworoom) data="tworoom"; dataset_dirname="tworooms"; default_h5_name="tworoom"                    ;;
    pusht)   data="pusht";   dataset_dirname="pusht";    default_h5_name="pusht_expert_train"         ;;
    cube)    data="ogb";     dataset_dirname="cube";     default_h5_name="ogbench/cube_single_expert" ;;
    reacher) data="dmc";     dataset_dirname="reacher";  default_h5_name="reacher"                    ;;
    *) echo "错误: 未知的 dataset_name '${dataset_name}'"; exit 1 ;;
esac

# 从训练数据配置读取默认 frameskip，支持环境变量覆盖
_dataset_cfg="${SCRIPT_DIR}/config/train/data/${data}.yaml"
if [ -f "${_dataset_cfg}" ]; then
    _default_frameskip=$(grep -m1 '^[[:space:]]*frameskip:' "${_dataset_cfg}" | sed 's/.*:[[:space:]]*\([0-9]*\).*/\1/')
    frameskip="${frameskip:-${_default_frameskip:-5}}"
    # 双保险：若 yaml 中的 name 与 case 里硬编码不一致，提示一下
    _yaml_h5_name=$(grep -m1 '^[[:space:]]*name:' "${_dataset_cfg}" | sed 's/.*:[[:space:]]*\([^[:space:]]*\).*/\1/')
    if [ -n "${_yaml_h5_name}" ] && [ "${_yaml_h5_name}" != "${default_h5_name}" ]; then
        echo "[warn] config/train/data/${data}.yaml name=${_yaml_h5_name} 与脚本内 default_h5_name=${default_h5_name} 不一致，使用脚本值；请同步两处"
    fi
else
    frameskip="${frameskip:-5}"
fi
diagnostic_dataset_name="${diagnostic_dataset_name:-${default_h5_name}}"

# Eval 默认使用训练 config 中的 trainer.max_epochs 对应 checkpoint。
_train_cfg="${SCRIPT_DIR}/config/train/${config_name}.yaml"
if [ -f "${_train_cfg}" ]; then
    _config_max_epochs=$(awk '
        /^[^[:space:]]/ { in_trainer=($1=="trainer:") }
        in_trainer && /^[[:space:]]*max_epochs:/ {
            sub(/#.*/, "")
            sub(/.*:[[:space:]]*/, "")
            print
            exit
        }
    ' "${_train_cfg}")
else
    echo "[eval] training config not found: ${_train_cfg}"
    exit 1
fi
if [ -z "${_config_max_epochs}" ]; then
    echo "[eval] trainer.max_epochs not found in ${_train_cfg}"
    exit 1
fi
eval_epoch="${eval_epoch:-${_config_max_epochs}}"
echo "[eval] using checkpoint epoch ${eval_epoch} (trainer.max_epochs from config/train/${config_name}.yaml)"

output_model_name="${dataset_name}_${output_model_name}"

add_override "data" "${data}"
add_override "data.dataset.frameskip" "${frameskip}"
add_override "seed" "${seed:-}"
add_override "output_model_name" "${output_model_name}"
add_override "subdir" "ckpt/${output_model_name}"
add_override "encoder.projection_head.type" "${encoder_projection_head_type:-}"
add_override "encoder.projection_head.norm_fn" "${encoder_projection_head_norm_fn:-}"
add_override "encoder.projection_head.hidden_dim" "${encoder_projection_head_hidden_dim:-}"
add_override "loss.sigreg.weight" "${loss_sigreg_weight:-}"
add_override "loss.sigreg.warmup.type" "${loss_sigreg_warmup_type:-}"
add_override "loss.sigreg.warmup.mode" "${loss_sigreg_warmup_mode:-}"
add_override "loss.sigreg.warmup.epochs" "${loss_sigreg_warmup_epochs:-}"
add_override "loss.sigreg.warmup.decay_epochs" "${loss_sigreg_warmup_decay_epochs:-}"
add_override "loss.sigreg.warmup.num_proj" "${loss_sigreg_warmup_num_proj:-}"
add_override "loss.sigreg.warmup.weight" "${loss_sigreg_warmup_weight:-}"
add_override "loss.regularizer.type" "${loss_regularizer_type:-}"
add_override "loss.regularizer.weight" "${loss_regularizer_weight:-}"
add_override "loss.regularizer.scope" "${loss_regularizer_scope:-}"
add_override "loss.rollout.weight" "${loss_rollout_weight:-}"
add_override "loss.rollout.steps" "${loss_rollout_steps:-}"
add_override "loss.uniformity.t" "${loss_regularizer_t:-}"
add_override "loss.uniformity.mode" "${loss_uniformity_mode:-}"
add_override "loss.uniformity.temporal_exclusion" "${loss_uniformity_temporal_exclusion:-}"
add_override "loss.temporal_hinge.weight" "${loss_temporal_hinge_weight:-}"
add_override "loss.temporal_hinge.margin" "${loss_temporal_hinge_margin:-}"
add_override "loss.temporal_hinge.squared" "${loss_temporal_hinge_squared:-}"
add_override "loss.temporal_hinge.dynamic.enabled" "${loss_temporal_hinge_dynamic_enabled:-}"
add_override "loss.temporal_hinge.dynamic.base_margin" "${loss_temporal_hinge_dynamic_base_margin:-}"
add_override "loss.temporal_hinge.dynamic.min_margin" "${loss_temporal_hinge_dynamic_min_margin:-}"
add_override "loss.temporal_hinge.dynamic.max_margin" "${loss_temporal_hinge_dynamic_max_margin:-}"
add_override "loss.inverse_dynamics.weight" "${loss_inverse_dynamics_weight:-}"
add_override "loss.transition_distance.weight" "${loss_transition_distance_weight:-}"
add_override "loss.pred.space" "${loss_pred_space:-}"
add_override "loss.pred.target_view" "${pred_target:-${target_view:-${loss_pred_target_view:-}}}"
add_override "loss.target_stop_grad" "${loss_target_stop_grad:-}"
add_override "loss.pred.type" "${loss_pred_type:-}"
add_override "loss.hetero.enabled" "${loss_hetero_enabled:-}"
add_override "loss.hetero.mode" "${loss_hetero_mode:-}"
add_override "loss.hetero.probe_weight" "${loss_hetero_probe_weight:-}"
add_override "loss.hetero.logvar_hidden_dim" "${loss_hetero_logvar_hidden_dim:-}"
add_override "loss.hetero.s_min" "${loss_hetero_s_min:-}"
add_override "loss.hetero.s_max" "${loss_hetero_s_max:-}"
add_override "loss.hetero.tau_floor" "${loss_hetero_tau_floor:-}"
add_override "loss.generic_latent_consistency.enabled" "${loss_generic_latent_consistency_enabled:-}"
add_override "loss.snap_acpc.enabled" "${loss_snap_acpc_enabled:-}"
add_override "loss.paired_view_control.enabled" "${loss_paired_view_control_enabled:-}"
add_override "loss.in_forward_noise_control.enabled" "${loss_in_forward_noise_control_enabled:-}"
add_override "loss.action_gate.enabled" "${loss_action_gate_enabled:-}"
add_override "loss.action_gate.mode" "${loss_action_gate_mode:-}"
add_override "loss.action_gate.intervention" "${loss_action_gate_intervention:-}"
add_override "loss.action_gate.delta_scale" "${loss_action_gate_delta_scale:-}"
add_override "loss.action_gate.num_delta_samples" "${loss_action_gate_num_delta_samples:-}"
add_override "loss.action_gate.warmup_epochs" "${loss_action_gate_warmup_epochs:-}"
add_override "loss.action_gate.ema_momentum" "${loss_action_gate_ema_momentum:-}"
add_override "loss.action_gate.w_min" "${loss_action_gate_w_min:-}"
add_override "loss.action_gate.w_max" "${loss_action_gate_w_max:-}"
add_override "loss.adaptive_consistency.enabled" "${loss_adaptive_consistency_enabled:-}"
add_override "loss.adaptive_consistency.weight" "${loss_adaptive_consistency_weight:-}"
add_override "loss.adaptive_consistency.noise_std_min" "${loss_adaptive_consistency_noise_std_min:-}"
add_override "loss.adaptive_consistency.noise_std_max" "${loss_adaptive_consistency_noise_std_max:-}"
add_override "loss.adaptive_consistency.noise_prob" "${loss_adaptive_consistency_noise_prob:-}"
add_override "loss.adaptive_consistency.distance" "${loss_adaptive_consistency_distance:-}"
add_override "loss.adaptive_consistency.detach_origin" "${loss_adaptive_consistency_detach_origin:-${loss_adaptive_consistency_detach_clean:-}}"
add_override "wm.embed_dim" "${wm_embed_dim:-}"
add_override "wm.inference.rollout_state_space" "${wm_inference_rollout_state_space:-}"
add_override "wm.inference.cost_space" "${wm_inference_cost_space:-}"
add_override "wm.inference.cost_type" "${wm_inference_cost_type:-}"
add_override "image_noise.std_min" "${image_noise_std_min:-}"
add_override "image_noise.std_max" "${image_noise_std_max:-}"
add_override "image_noise.noise_prob" "${image_noise_noise_prob:-}"
add_override "image_noise.apply_to_val" "${image_noise_apply_to_val:-}"

# ---------- 2. 训练（支持 skip_train=1 跳过） ----------
if [ "${skip_train:-0}" = "1" ]; then
    echo "[train] skipped (skip_train=1)"
else
    swanlab login -k "${SWANLAB_API_KEY}"
    # Defensive: if STABLEWM_HOME already points to a lewm-* subdir, go up one level first
    if [[ "$(basename "$STABLEWM_HOME")" == lewm-* ]]; then
        export STABLEWM_HOME="$(dirname "$STABLEWM_HOME")/lewm-${dataset_dirname}"
    else
        export STABLEWM_HOME="${STABLEWM_HOME}/lewm-${dataset_dirname}"
    fi

    echo "==================================================="
    echo "[train] starting ${trainer_file} for ${output_model_name}"
    echo "==================================================="
    python ${trainer_file} --config-name="${config_name}" \
        logger_backend=swanlab \
        swanlab.enabled=True \
        "${CMD_ARGS[@]}"

    train_status=$?
    if [ $train_status -ne 0 ]; then
        echo "[train] failed with status ${train_status}; skipping eval sweep"
        exit $train_status
    fi
fi

# Defensive: if STABLEWM_HOME already points to a lewm-* subdir, go up one level first
if [[ "$(basename "$STABLEWM_HOME")" == lewm-* ]]; then
    export STABLEWM_HOME="$(dirname "$STABLEWM_HOME")/lewm-${dataset_dirname}"
else
    export STABLEWM_HOME="${STABLEWM_HOME}/lewm-${dataset_dirname}"
fi

# ---------- 3. Eval / Noise 通用准备 ----------
if [ -n "${ckpt_override:-}" ]; then
    ckpt_abs="${ckpt_override}"
    ckpt_rel="$(basename "${ckpt_abs}" _object.ckpt)"
    results_dir="${STABLEWM_HOME}/ckpt/${output_model_name}/eval_results"
else
    ckpt_rel="ckpt/${output_model_name}/${output_model_name}_epoch_${eval_epoch}"
    ckpt_abs="${STABLEWM_HOME}/${ckpt_rel}_object.ckpt"
    results_dir="${STABLEWM_HOME}/ckpt/${output_model_name}/eval_results"
fi
mkdir -p "${results_dir}"

post_train_eval_mode="${post_train_eval_mode:-full}"
case "${post_train_eval_mode}" in
    full|origin|none) ;;
    *)
        echo "[eval] post_train_eval_mode must be one of: full, origin, none; got '${post_train_eval_mode}'"
        exit 1
        ;;
esac
echo "[eval] post_train_eval_mode=${post_train_eval_mode}"

if [ ! -f "${ckpt_abs}" ]; then
    echo "[eval] checkpoint not found: ${ckpt_abs}"
    echo "[eval] aborting downstream steps"
    exit 1
fi

# GPU 探测
if [ "${post_train_eval_mode}" != "none" ]; then
    if [ -z "${eval_gpus:-}" ]; then
        if command -v nvidia-smi >/dev/null 2>&1; then
            eval_gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | tr '\n' ' ')
        else
            eval_gpus="0"
        fi
    fi
    read -ra gpu_array <<< "${eval_gpus}"
    n_gpus=${#gpu_array[@]}
    echo "[gpu] using GPUs: ${gpu_array[*]} (count=${n_gpus})"
else
    gpu_array=()
    n_gpus=0
    echo "[eval] GPU detection skipped (post_train_eval_mode=none)"
fi

run_eval_sweep=0
run_diagnostics=0
case "${post_train_eval_mode}" in
    full)
        [ "${skip_eval_sweep:-0}" != "1" ] && run_eval_sweep=1
        [ "${skip_diagnostics:-${skip_noise_table:-0}}" != "1" ] && run_diagnostics=1
        ;;
    origin)
        run_eval_sweep=1
        eval_corruption_stds="0.0"
        run_diagnostics=0
        ;;
    none)
        run_eval_sweep=0
        run_diagnostics=0
        ;;
esac

if [ "${run_eval_sweep}" = "0" ]; then
    if [ "${post_train_eval_mode}" = "none" ]; then
        echo "[eval sweep] skipped (post_train_eval_mode=none)"
    else
        echo "[eval sweep] skipped (skip_eval_sweep=1)"
    fi
fi

# ---------- 4. Eval Sweep ----------
# 多 seed 支持：
#   eval_seeds          每个 (std, mode) 组合下要跑的不同 seed 数量；默认 3。
#                       =1 时退化为单 seed 行为（与旧脚本一致）。
#   eval_base_seed      第一个 seed，后续 seed 依次 +1；默认 42（与 LeWM eval
#                       config 中的默认 seed 对齐）。
#   每个 seed 实际跑的 episode 数 = num_eval / eval_seeds（向下取整）。
#   不能整除时打印 warning，并丢弃余数，保证每个 seed 样本数一致便于聚合。
if [ "${run_eval_sweep}" = "1" ]; then
eval_seeds="${eval_seeds:-3}"

if ! [[ "${eval_seeds}" =~ ^[0-9]+$ ]] || [ "${eval_seeds}" -lt 1 ]; then
    echo "[eval] eval_seeds 必须是 >=1 的整数，got '${eval_seeds}'"
    exit 1
fi

# eval_base_seed 未传时从 config/eval/<dataset_name>.yaml 中读取顶层 seed 字段。
if [ -z "${eval_base_seed:-}" ]; then
    _eval_cfg="${SCRIPT_DIR}/config/eval/${dataset_name}.yaml"
    if [ ! -f "${_eval_cfg}" ]; then
        echo "[eval] eval config not found: ${_eval_cfg}"
        exit 1
    fi
    eval_base_seed=$(awk '
        /^[[:space:]]/ { next }       # 仅匹配顶层（行首无空格）的 seed:
        /^seed:/ {
            sub(/#.*/, "")
            sub(/.*:[[:space:]]*/, "")
            print
            exit
        }
    ' "${_eval_cfg}")
    if [ -z "${eval_base_seed}" ]; then
        echo "[eval] 无法在 ${_eval_cfg} 中找到顶层 seed 字段；请显式设置 eval_base_seed"
        exit 1
    fi
fi
if ! [[ "${eval_base_seed}" =~ ^[0-9]+$ ]]; then
    echo "[eval] eval_base_seed 必须是非负整数，got '${eval_base_seed}'"
    exit 1
fi

per_seed_num_eval=$(( num_eval / eval_seeds ))
if [ $(( per_seed_num_eval * eval_seeds )) -ne "${num_eval}" ]; then
    echo "[eval][warn] num_eval=${num_eval} 不能被 eval_seeds=${eval_seeds} 整除，"
    echo "[eval][warn]   每个 seed 跑 ${per_seed_num_eval} 次，余数 $(( num_eval - per_seed_num_eval * eval_seeds )) 被丢弃。"
fi
echo "[eval] seeds=${eval_seeds} (base=${eval_base_seed})  per-seed num_eval=${per_seed_num_eval}"
fi

run_one_eval() {
    local job="$1"
    local gpu="$2"
    # Job tuple format: label|magnitude|mode|seed|ctype
    # ctype ∈ {gaussian_noise, gaussian_blur, resize}; magnitude is
    # reused as std / kernel_size / factor depending on ctype.
    IFS='|' read -ra parts <<< "$job"
    local label="${parts[0]}"
    local mag="${parts[1]}"
    local mode="${parts[2]}"
    local seed="${parts[3]}"
    local ctype="${parts[4]:-gaussian_noise}"

    local args=(
        "--config-name=${dataset_name}.yaml"
        "policy=${ckpt_rel}"
        "seed=${seed}"
        "eval.num_eval=${per_seed_num_eval}"
        "output.filename=${results_dir}/${label}_metrics.txt"
    )
    if [ "$mode" != "none" ]; then
        args+=("eval.corruption.type=${ctype}")
        case "$ctype" in
            gaussian_noise) args+=("eval.corruption.std=${mag}") ;;
            gaussian_blur)  args+=("eval.corruption.kernel_size=${mag}") ;;
            resize)         args+=("eval.corruption.factor=${mag}") ;;
            *) echo "[eval] unknown corruption type '${ctype}'"; return 1 ;;
        esac
        local apply_list="${mode//+/,}"
        args+=("eval.corruption.apply_to=[${apply_list}]")
    fi

    echo "[eval] start  gpu=${gpu} label=${label} ctype=${ctype} mag=${mag} mode=${mode} seed=${seed}"
    CUDA_VISIBLE_DEVICES=${gpu} python eval.py "${args[@]}" \
        > "${results_dir}/${label}.log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "[eval] done   gpu=${gpu} label=${label}"
    else
        echo "[eval] FAIL   gpu=${gpu} label=${label} (rc=${rc}; see ${results_dir}/${label}.log)"
    fi
}

if [ "${run_eval_sweep}" = "1" ]; then
    # Eval corruption family. Default is the existing pixel-noise sweep
    # (`gaussian_noise`). Choose `gaussian_blur` or `resize` to run an
    # alternative corruption family on the same checkpoint; the output
    # filenames are tagged so they do not collide with the noise sweep.
    eval_corruption_type="${eval_corruption_type:-gaussian_noise}"
    eval_corruption_apply_to_raw="${eval_corruption_apply_to:-1}"
    if ! eval_corruption_apply_to="$(normalize_eval_corruption_apply_to "${eval_corruption_apply_to_raw}")"; then
        exit 1
    fi
    echo "[eval] eval_corruption_apply_to=${eval_corruption_apply_to} (raw=${eval_corruption_apply_to_raw})"

    # Per-type magnitude lists (only the one matching eval_corruption_type
    # is consumed; the others are ignored). For `gaussian_noise` we keep
    # the existing `eval_corruption_stds` name for backward compatibility;
    # for the new families we use suggestive names.
    eval_corruption_stds="${eval_corruption_stds-0.0 0.03 0.05 0.08}"
    eval_blur_kernel_sizes="${eval_blur_kernel_sizes-3 7 11 15}"    # odd px; kernel_size=1 is the no-op but omitted from defaults to avoid shell-word-split issues on cloud launchers
    eval_resize_factors="${eval_resize_factors-1.0 0.75 0.5 0.25}"  # 1.0 = unperturbed

    # 构造 (label, magnitude, mode, seed, ctype) 任务列表。多 seed 时 label 后缀 _seed${seed}，
    # 单 seed (eval_seeds=1) 时不加后缀以保持向后兼容（产出的文件名跟旧脚本一致）。
    # 文件命名约定（避免与 noise sweep 撞名）：
    #   gaussian_noise: <apply_to>_std<X>_seed<N>      (历史不变)
    #   gaussian_blur:  <apply_to>_blur_ks<K>_seed<N>
    #   resize:         <apply_to>_rs_factor<X>_seed<N>
    seed_suffix() {
        if [ "${eval_seeds}" -gt 1 ]; then echo "_seed$1"; else echo ""; fi
    }

    case "${eval_corruption_type}" in
        gaussian_noise) _sweep_mags="${eval_corruption_stds}"   ; _tag_prefix="std"       ; _zero_mag="0.0" ;;
        gaussian_blur)  _sweep_mags="${eval_blur_kernel_sizes}" ; _tag_prefix="blur_ks"   ; _zero_mag="1"   ;;
        resize)         _sweep_mags="${eval_resize_factors}"    ; _tag_prefix="rs_factor" ; _zero_mag="1.0" ;;
        *) echo "[eval] unknown eval_corruption_type='${eval_corruption_type}'"; exit 1 ;;
    esac

    jobs=()
    for mag in $_sweep_mags; do
        # is_origin: gaussian_noise → mag == 0.0 ; gaussian_blur → mag == 1 ; resize → mag == 1.0
        is_origin=$(awk -v m="$mag" -v z="$_zero_mag" 'BEGIN{print (m+0==z+0)?1:0}')
        for ((s=0; s<eval_seeds; s++)); do
            cur_seed=$(( eval_base_seed + s ))
            suf=$(seed_suffix "${cur_seed}")
            if [ "$is_origin" = "1" ]; then
                jobs+=("origin${suf}|${_zero_mag}|none|${cur_seed}|${eval_corruption_type}")
            else
                IFS=',' read -ra modes <<< "${eval_corruption_apply_to}"
                for mode in "${modes[@]}"; do
                    local_label="$(echo "${mode}" | tr '+' '_')_${_tag_prefix}${mag}${suf}"
                    jobs+=("${local_label}|${mag}|${mode}|${cur_seed}|${eval_corruption_type}")
                done
            fi
        done
    done

    total=${#jobs[@]}
    echo "==================================================="
    echo "[eval sweep] ${total} jobs across ${n_gpus} GPUs"
    echo "==================================================="

    i=0
    while [ $i -lt $total ]; do
        pids=()
        for ((k=0; k<n_gpus && i<total; k++)); do
            run_one_eval "${jobs[$i]}" "${gpu_array[$k]}" &
            pids+=($!)
            ((i++))
        done
        for pid in "${pids[@]}"; do
            wait "$pid" || true
        done
    done
fi

# ---------- 5. Full Latent-Geometry Diagnostics ----------
# Unified entry: noise_sensitivity + predictor_sensitivity + task_resolution
# + latent_noise_sensitivity + action_effect.
# Output dir: ${results_dir}/diagnostics/
#   noise_sensitivity.{csv,json}, geometry_summary.{csv,json}, *.png
#   predictor_sensitivity.{csv,json}
#   task_resolution.{csv,json}
#   latent_noise_sensitivity.{csv,json}, latent_geometry_summary.{csv,json}
#   action_effect.{csv,json}
#   diagnostics_summary.json   (per-checkpoint roll-up; consumed by P0.7
#                                and by §6 P0.5b cross_check_correlations)
#
# 表征分布投影（PCA/t-SNE）放在 tools/repr_analysis/analyze_repr.py 和
# plot_repr.py 里按需独立调用，不进默认 diagnostics 链。
#
# Backward-compat env vars:
#   skip_noise_table=1         skips the entire diagnostics suite (legacy name)
#   skip_diagnostics=1         same as above (preferred)
#   noise_table_stds           still used; passed as --stds
#   diagnostic_rollout_steps   default "1 2 4 8"
#   diagnostic_skip_predictor=1 / diagnostic_skip_resolution=1
#   diagnostic_skip_latent_noise=1 / diagnostic_skip_action_effect=1
#                              per-tool overrides
if [ "${run_diagnostics}" = "1" ]; then
    diagnostic_rollout_steps="${diagnostic_rollout_steps:-1 2 4 8}"
    # Diagnostic corruption family. Defaults to the same family the eval
    # sweep uses (``eval_corruption_type``) so a blur eval run also gets
    # a blur diagnostic; explicit override is also supported.
    # When non-default, run_full_diagnostics.py auto-appends a
    # ``_<corruption_type>`` suffix to the save dir to keep blur/resize
    # outputs from overwriting the canonical gaussian diagnostic files.
    diagnostic_corruption_type="${diagnostic_corruption_type:-${eval_corruption_type:-gaussian_noise}}"

    # Per-type default sweep magnitudes for the noise / predictor sensitivity
    # probes. Each family needs different scales because the magnitudes have
    # different units (std / kernel_size px / factor). Users can override via
    # ``noise_table_stds`` (kept as the variable name for back-compat).
    case "${diagnostic_corruption_type}" in
        gaussian_noise)
            noise_table_stds="${noise_table_stds:-0.0 0.005 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1}"
            ;;
        gaussian_blur)
            # kernel_size list (odd px). 1 = identity; ramp up to a heavy blur.
            noise_table_stds="${noise_table_stds:-1 3 5 7 11 15 21 31}"
            ;;
        resize)
            # factor list. 1.0 = identity; ramp down to a near-blob.
            noise_table_stds="${noise_table_stds:-1.0 0.8 0.6 0.4 0.25 0.125}"
            ;;
        *) echo "[diagnostics] unknown diagnostic_corruption_type='${diagnostic_corruption_type}'"; exit 1 ;;
    esac

    diag_args=(
        "--model" "${output_model_name}=${ckpt_abs}"
        "--dataset" "${diagnostic_dataset_name}"
        "--stds" ${noise_table_stds}
        "--rollout-steps" ${diagnostic_rollout_steps}
        "--frameskip" "${frameskip}"
        "--save-dir" "${results_dir}/diagnostics"
        "--plot"
        "--corruption-type" "${diagnostic_corruption_type}"
    )
    [ "${diagnostic_skip_predictor:-0}" = "1" ] && diag_args+=("--skip-predictor")
    [ "${diagnostic_skip_resolution:-0}" = "1" ] && diag_args+=("--skip-resolution")
    [ "${diagnostic_skip_latent_noise:-0}" = "1" ] && diag_args+=("--skip-latent-noise")
    [ "${diagnostic_skip_action_effect:-0}" = "1" ] && diag_args+=("--skip-action-effect")

    echo "==================================================="
    echo "[diagnostics] running full suite on ${ckpt_abs}"
    echo "==================================================="
    CUDA_VISIBLE_DEVICES=${gpu_array[0]} python -m tools.repr_analysis.run_full_diagnostics \
        "${diag_args[@]}" 2>&1 | tee "${results_dir}/diagnostics.log"
else
    if [ "${post_train_eval_mode}" = "origin" ]; then
        echo "[diagnostics] skipped (post_train_eval_mode=origin)"
    elif [ "${post_train_eval_mode}" = "none" ]; then
        echo "[diagnostics] skipped (post_train_eval_mode=none)"
    else
        echo "[diagnostics] skipped (skip_diagnostics=1)"
    fi
fi

# ---------- 6. Summary ----------
summary_file="${results_dir}/summary.txt"
{
    echo "===== ${output_model_name} eval summary ====="
    echo "ckpt: ${ckpt_abs}"
    echo "dataset: ${dataset_name}    num_eval: ${num_eval}    epoch: ${eval_epoch}    post_train_eval_mode: ${post_train_eval_mode}"
    echo
    echo "----- eval metrics (per-seed raw) -----"
    for log in "${results_dir}"/*.log; do
        [ -e "$log" ] || continue
        base=$(basename "$log" .log)
        [ "$base" = "noise_table" ] && continue
        [ "$base" = "diagnostics" ] && continue
        echo
        echo "== ${base} =="
        # 抓 metrics line（dict 形式）
        if grep -m1 "^{" "$log" >/dev/null 2>&1; then
            grep "^{" "$log" | tail -1
        else
            grep -i "metrics\|success" "$log" | tail -3
        fi
    done

    echo
    echo "----- eval metrics (aggregated across seeds) -----"
    # 把 *_seed<N>.log 按去掉 seed 后缀的 group key 聚合，
    # 输出 mean / std / sem，并把同一份结果写到 eval_summary.csv 供下游消费。
    python3 - "${results_dir}" <<'PYEOF'
import ast, glob, os, re, statistics, sys, csv

results_dir = sys.argv[1]
groups = {}  # group_key -> list of (seed_or_None, metrics_dict)


def _last_balanced_dict(text: str):
    """Return the last balanced {...} substring, scanning naively. Brace
    counting works for our eval logs (no '{' or '}' inside strings/arrays)."""
    last = None
    i, n = 0, len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        for j in range(i, n):
            c = text[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    last = text[i:j + 1]
                    i = j + 1
                    break
        else:
            break
    return last


# numpy `array([...])` literals (and a generic `np.array(...)`) are not
# accepted by ast.literal_eval; replace them with None before parsing so the
# numeric metrics survive.
_ARRAY_RE = re.compile(r"\b(?:np\.)?array\((?:[^()]|\([^()]*\))*\)", re.DOTALL)

for log in sorted(glob.glob(os.path.join(results_dir, "*.log"))):
    base = os.path.basename(log)[:-4]
    if base in ("noise_table", "diagnostics"):
        continue
    m = re.match(r"^(.*?)(?:_seed(\d+))?$", base)
    group_key, seed = m.group(1), m.group(2)
    with open(log, "r", errors="replace") as f:
        text = f.read()
    candidate = _last_balanced_dict(text)
    if candidate is None or "success_rate" not in candidate:
        continue
    cleaned = _ARRAY_RE.sub("None", candidate)
    try:
        d = ast.literal_eval(cleaned)
    except Exception:
        continue
    if not isinstance(d, dict):
        continue
    groups.setdefault(group_key, []).append((seed, d))

csv_rows = [("group", "n_seeds", "seeds", "metric", "mean", "std", "sem", "values")]
for group_key in sorted(groups):
    runs = groups[group_key]
    seeds_str = ",".join(s if s else "-" for s, _ in runs)
    n = len(runs)
    print(f"\n== {group_key}  (n_seeds={n}, seeds={seeds_str}) ==")
    keys = sorted({k for _, d in runs for k in d.keys()})
    for k in keys:
        vals = [d[k] for _, d in runs if k in d and isinstance(d[k], (int, float)) and not isinstance(d[k], bool)]
        if not vals:
            print(f"  {k}: <non-numeric or missing>")
            continue
        mean = sum(vals) / len(vals)
        std = statistics.pstdev(vals) if len(vals) >= 2 else 0.0
        sem = std / (len(vals) ** 0.5) if len(vals) >= 2 else 0.0
        print(f"  {k}: mean={mean:.4f}  std={std:.4f}  sem={sem:.4f}  n={len(vals)}  raw={['%.4f' % v for v in vals]}")
        csv_rows.append((group_key, n, seeds_str, k, f"{mean:.6f}", f"{std:.6f}", f"{sem:.6f}",
                         ";".join(f"{v:.6f}" for v in vals)))

csv_path = os.path.join(results_dir, "eval_summary.csv")
with open(csv_path, "w", newline="") as f:
    csv.writer(f).writerows(csv_rows)
print(f"\n[aggregated CSV] {csv_path}")
PYEOF
    if [ -f "${results_dir}/diagnostics/geometry_summary.csv" ]; then
        echo
        echo "----- geometry summary -----"
        cat "${results_dir}/diagnostics/geometry_summary.csv"
    fi
    if [ -f "${results_dir}/diagnostics/diagnostics_summary.json" ]; then
        echo
        echo "----- diagnostics roll-up -----"
        cat "${results_dir}/diagnostics/diagnostics_summary.json"
    fi
} > "${summary_file}"

# ---------- 6b. Cross-check correlations (optional) ----------
# Multi-dimensional confound check on the canonical 8 ckpts/task already on
# disk under STABLEWM_HOME's parent directory. Tolerates missing ckpts —
# rows with absent diagnostics_summary.json are silently skipped, so it's
# safe to run after each training even if the sweep is incomplete.
if [ "${run_cross_check_correlations:-0}" = "1" ] && [ "${run_diagnostics}" = "1" ]; then
    cross_check_root="$(dirname "${STABLEWM_HOME}")"
    cross_check_out="${results_dir}/diagnostics/cross_check_corr.json"
    echo "==================================================="
    echo "[cross_check] running on ${cross_check_root} -> ${cross_check_out}"
    echo "==================================================="
    STABLEWM_HOME="${cross_check_root}" \
        python -m tools.repr_analysis.cross_check_correlations \
        --out "${cross_check_out}" \
        2>&1 | tee "${results_dir}/diagnostics/cross_check.log" || \
        echo "[cross_check] non-fatal failure (likely missing ckpts in sweep)"
fi
if [ "${run_cross_check_correlations:-0}" = "1" ] && [ "${run_diagnostics}" != "1" ]; then
    echo "[cross_check] skipped (diagnostics not run; post_train_eval_mode=${post_train_eval_mode})"
fi

echo "==================================================="
echo "[done] artifacts in:"
echo "  ${results_dir}/"
echo "  - per-eval logs:    *.log"
echo "  - per-eval metrics: *_metrics.txt"
if [ "${run_diagnostics}" = "1" ]; then
    echo "  - diagnostics:      diagnostics/"
    echo "      * noise_sensitivity.csv / .json"
    echo "      * geometry_summary.csv / .json"
    echo "      * predictor_sensitivity.csv / .json"
    echo "      * task_resolution.csv / .json"
    echo "      * latent_noise_sensitivity.csv / .json + latent_geometry_summary.csv / .json"
    echo "      * action_effect.csv / .json"
    echo "      * diagnostics_summary.json"
    echo "      * cross_check_corr.json (when run_cross_check_correlations=1)"
    echo "      * noise_ratio_curve_goal.png"
    echo "      * noise_angle_curve_goal.png"
    echo "      * geometry_tradeoff_goal.png"
else
    echo "  - diagnostics:      skipped"
fi
echo "  - summary:          summary.txt"
echo "==================================================="

# ---------- 7. Cleanup ----------
rm -rf "${STABLEWM_HOME}/ckpt/${output_model_name}"/*.mp4
