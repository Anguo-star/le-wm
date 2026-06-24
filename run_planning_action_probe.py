"""
Standalone script to run planning probe + action probe for all models,
saving results as planning_action_probe.json per-task.
"""
import os
import sys
import json
from pathlib import Path

import torch

from tools.paper1_paths import paper1_data_root
from tools.repr_analysis.analyze_repr import (
    analyze_planning_signal,
    analyze_action_effect,
    encode_sequences,
    infer_history_size,
    load_dataset_samples,
    load_model,
    to_serializable,
)


def run_probes_for_task(task_name, models, dataset, frameskip=5, n_sequences=256, future_steps=8, seed=3072):
    """Run planning + action probes for all models in a task."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset once using first model's history_size
    first_model = load_model(next(iter(models.values())), device)
    history_size = infer_history_size(first_model)
    del first_model
    torch.cuda.empty_cache()

    batch = load_dataset_samples(
        state_key=None,
        dataset_name=dataset,
        n_sequences=n_sequences,
        history_size=history_size,
        future_steps=future_steps,
        frameskip=frameskip,
        img_size=224,
        seed=seed,
        device=device,
    )

    results = {}
    for label, ckpt in models.items():
        print(f"[{task_name}] Running probes for {label} ...")
        model = load_model(ckpt, device)

        # Encode batch to get embeddings needed by action_effect probe
        outputs = encode_sequences(model, batch)

        # Planning probe
        try:
            planning = analyze_planning_signal(
                model=model,
                outputs=outputs,
                history_size=history_size,
                future_steps=future_steps,
                random_action_trials=30,
                seed=seed,
            )
        except Exception as e:
            print(f"  Planning probe failed: {e}")
            planning = {"error": str(e)}

        # Action probe
        try:
            action = analyze_action_effect(
                model=model,
                outputs=outputs,
                n_trials=128,
                interp_steps=16,
                perturb_scale=0.5,
            )
        except Exception as e:
            print(f"  Action probe failed: {e}")
            action = {"error": str(e)}

        results[label] = {
            "model": label,
            "ckpt": ckpt,
            "planning": planning,
            "action": action,
        }

        del model
        torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    ROOT = paper1_data_root()

    # TwoRoom models
    tw_models = {
        'LeWM-base': str(ROOT / 'lewm-tworooms/ckpt/tworoom_lewm/tworoom_lewm_epoch_9_object.ckpt'),
        'LeWM-fixed-std': str(ROOT / 'lewm-tworooms/ckpt/tworoom_lewm_noise_std_0_005/tworoom_lewm_noise_std_0_005_epoch_9_object.ckpt'),
        'LeWM-perframe-p05': str(ROOT / 'lewm-tworooms/ckpt/tworoom_lewm_noise_0to005_p05/tworoom_lewm_noise_0to005_p05_epoch_9_object.ckpt'),
        'LeWM-perframe-p1': str(ROOT / 'lewm-tworooms/ckpt/tworoom_lewm_noise_0to005_p1/tworoom_lewm_noise_0to005_p1_epoch_9_object.ckpt'),
        'SWM-base': str(ROOT / 'lewm-tworooms/ckpt/tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_dim64_20260425/tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_dim64_20260425_epoch_9_object.ckpt'),
        'SWM-fixed-std': str(ROOT / 'lewm-tworooms/ckpt/tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_std0_005_dim64/tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_std0_005_dim64_epoch_9_object.ckpt'),
        'SWM-perframe-p05': str(ROOT / 'lewm-tworooms/ckpt/tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to005_p05_dim64/tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to005_p05_dim64_epoch_9_object.ckpt'),
        'SWM-perframe-p1': str(ROOT / 'lewm-tworooms/ckpt/tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to005_p1_dim64/tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to005_p1_dim64_epoch_9_object.ckpt'),
    }

    os.environ['STABLEWM_HOME'] = str(ROOT / 'lewm-tworooms')
    tw_results = run_probes_for_task('TwoRoom', tw_models, 'tworoom', frameskip=5)
    tw_out = ROOT / 'lewm-tworooms/repr_analysis/p03_diagnostics/planning_action_probe.json'
    with open(tw_out, 'w') as f:
        json.dump(to_serializable(tw_results), f, indent=2)
    print(f"Saved TwoRoom: {tw_out}")

    # PushT models
    pt_models = {
        'LeWM-fixed-std': str(ROOT / 'lewm-pusht/ckpt/pusht_lewm_noise_std_0_005/pusht_lewm_noise_std_0_005_epoch_9_object.ckpt'),
        'LeWM-perframe-0to001-p1': str(ROOT / 'lewm-pusht/ckpt/pusht_lewm_noise_0to001_p1/pusht_lewm_noise_0to001_p1_epoch_9_object.ckpt'),
        'LeWM-perframe-0to002-p1': str(ROOT / 'lewm-pusht/ckpt/pusht_lewm_noise_0to002_p1/pusht_lewm_noise_0to002_p1_epoch_9_object.ckpt'),
        'LeWM-perframe-0to005-p1': str(ROOT / 'lewm-pusht/ckpt/pusht_lewm_noise_0to005_p1/pusht_lewm_noise_0to005_p1_epoch_9_object.ckpt'),
        'SWM-fixed-std': str(ROOT / 'lewm-pusht/ckpt/pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_std0_005_dim64/pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_std0_005_dim64_epoch_9_object.ckpt'),
        'SWM-perframe-0to001-p05': str(ROOT / 'lewm-pusht/ckpt/pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to001_p05_dim64/pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to001_p05_dim64_epoch_9_object.ckpt'),
        'SWM-perframe-0to001-p1': str(ROOT / 'lewm-pusht/ckpt/pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to001_p1_dim64/pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to001_p1_dim64_epoch_9_object.ckpt'),
        'SWM-perframe-0to002-p05': str(ROOT / 'lewm-pusht/ckpt/pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to002_p05_dim64/pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to002_p05_dim64_epoch_9_object.ckpt'),
        'SWM-perframe-0to002-p1': str(ROOT / 'lewm-pusht/ckpt/pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to002_p1_dim64/pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to002_p1_dim64_epoch_9_object.ckpt'),
    }

    os.environ['STABLEWM_HOME'] = str(ROOT / 'lewm-pusht')
    pt_results = run_probes_for_task('PushT', pt_models, 'pusht_expert_train', frameskip=5)
    pt_out = ROOT / 'lewm-pusht/repr_analysis/p03_diagnostics/planning_action_probe.json'
    with open(pt_out, 'w') as f:
        json.dump(to_serializable(pt_results), f, indent=2)
    print(f"Saved PushT: {pt_out}")
