"""Run eval for models missing results.txt, parallelized across GPUs."""

import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tools.paper1_paths import repo_root, task_dir

TASKS = {
    "tworoom": {
        "stablewm_home": str(task_dir("lewm-tworooms")),
        "config_name": "tworoom",
        "models": [
            ("LeWM-perframe-p05", "ckpt/tworoom_lewm_noise_0to005_p05/tworoom_lewm_noise_0to005_p05_epoch_9", 94.0),
            ("LeWM-perframe-p1", "ckpt/tworoom_lewm_noise_0to005_p1/tworoom_lewm_noise_0to005_p1_epoch_9", 94.0),
            ("SWM-perframe-p05", "ckpt/tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to005_p05_dim64/tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to005_p05_dim64_epoch_9", 87.3),
            ("SWM-perframe-p1", "ckpt/tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to005_p1_dim64/tworoom_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to005_p1_dim64_epoch_9", 86.7),
        ],
    },
    "pusht": {
        "stablewm_home": str(task_dir("lewm-pusht")),
        "config_name": "pusht",
        "models": [
            ("LeWM-perframe-0to001-p1", "ckpt/pusht_lewm_noise_0to001_p1/pusht_lewm_noise_0to001_p1_epoch_9", 87.3),
            ("LeWM-perframe-0to002-p1", "ckpt/pusht_lewm_noise_0to002_p1/pusht_lewm_noise_0to002_p1_epoch_9", 89.3),
            ("LeWM-perframe-0to005-p1", "ckpt/pusht_lewm_noise_0to005_p1/pusht_lewm_noise_0to005_p1_epoch_9", 82.0),
            ("SWM-perframe-0to001-p05", "ckpt/pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to001_p05_dim64/pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to001_p05_dim64_epoch_9", 78.0),
            ("SWM-perframe-0to001-p1", "ckpt/pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to001_p1_dim64/pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to001_p1_dim64_epoch_9", 87.3),
            ("SWM-perframe-0to002-p05", "ckpt/pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to002_p05_dim64/pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to002_p05_dim64_epoch_9", 78.7),
            ("SWM-perframe-0to002-p1", "ckpt/pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to002_p1_dim64/pusht_swm_mlp_bn_uniform_w02_t2_temporal_masked_2_noise_0to002_p1_dim64_epoch_9", 81.3),
        ],
    },
}

EVAL_TIMEOUT = 12000  # ~3.3 hours per model (covers slow PushT)


def run_single_eval(task_name, label, policy_path, expected,
                    config_name, stablewm_home, gpu_id):
    results_dir = Path(stablewm_home) / policy_path
    if not results_dir.is_dir():
        results_dir = results_dir.parent
    if results_dir.is_dir():
        for old in results_dir.glob("*_results.txt"):
            old.unlink()
            print(f"  [{task_name}/{label}] Removed old {old.name}")
        for old in results_dir.glob("*.mp4"):
            old.unlink()

    log_file = results_dir / "eval_run.log"
    cmd = [
        sys.executable, "-u", "eval.py",
        f"--config-name={config_name}.yaml",
        f"policy={policy_path}",
        "eval.num_eval=150",
    ]
    env = {
        **os.environ,
        "STABLEWM_HOME": stablewm_home,
        "MUJOCO_GL": "egl",
        "CUDA_VISIBLE_DEVICES": str(gpu_id),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }

    print(f"  [{task_name}/{label}] Starting on GPU {gpu_id} (timeout={EVAL_TIMEOUT}s)...")
    start = time.time()
    try:
        with open(log_file, "w") as logf:
            proc = subprocess.run(
                cmd,
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
                timeout=EVAL_TIMEOUT,
                cwd=repo_root(),
            )
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        return {
            "status": "timeout",
            "task": task_name,
            "label": label,
            "gpu": gpu_id,
            "elapsed": elapsed,
        }

    elapsed = time.time() - start

    # Read success rate from log
    actual = None
    if log_file.exists():
        log_text = log_file.read_text()
        m = re.search(r"'success_rate':\s*([0-9.]+)", log_text)
        actual = float(m.group(1)) if m else None

    return {
        "status": "success" if proc.returncode == 0 and actual is not None else "failed",
        "task": task_name,
        "label": label,
        "gpu": gpu_id,
        "elapsed": elapsed,
        "returncode": proc.returncode,
        "actual_score": actual,
        "expected_score": expected,
        "log_file": str(log_file),
    }


def main():
    n_gpus = 8
    try:
        import torch
        n_gpus = torch.cuda.device_count()
    except Exception:
        pass
    print(f"Using {n_gpus} GPUs, timeout={EVAL_TIMEOUT}s")

    jobs = []
    gpu_counter = 0
    for task_name, task_cfg in TASKS.items():
        for label, policy_path, expected in task_cfg["models"]:
            gpu_id = gpu_counter % n_gpus
            gpu_counter += 1
            jobs.append((task_name, label, policy_path, expected,
                         task_cfg["config_name"], task_cfg["stablewm_home"], gpu_id))

    results = []
    with ThreadPoolExecutor(max_workers=n_gpus) as executor:
        futures = {executor.submit(run_single_eval, *job): job for job in jobs}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                result = {"status": "exception", "error": str(e), "label": futures[future][1]}
            results.append(result)
            status = result.get("status", "?")
            label = result.get("label", "?")
            task = result.get("task", "?")
            actual = result.get("actual_score")
            expected = result.get("expected_score")
            elapsed = result.get("elapsed", 0)
            logf = result.get("log_file", "")

            if status == "success" and actual is not None:
                match = "MATCH" if abs(actual - expected) < 0.5 else "MISMATCH"
                print(f"[{task}/{label}] SUCCESS: score={actual:.2f} (expected={expected}, {match}, elapsed={elapsed:.0f}s)")
            elif status == "timeout":
                print(f"[{task}/{label}] TIMEOUT after {elapsed:.0f}s")
            else:
                print(f"[{task}/{label}] FAILED: rc={result.get('returncode','?')}, elapsed={elapsed:.0f}s")
                if logf:
                    print(f"  log: {logf}")

    out_path = repo_root() / "assets" / "paper1_data" / "missing_eval_results_v2.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out_path}")

    success = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") != "success"]
    print(f"Summary: {len(success)} succeeded, {len(failed)} failed/timeout out of {len(results)} total")


if __name__ == "__main__":
    main()
