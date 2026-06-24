# SwanLab Metrics Reading Notes

This note records the reliable way to read SwanLab run metrics from this repo.
Do not commit API keys or paste them into scripts. Prefer passing the key through
`SWANLAB_API_KEY`.

## Quick Start

Use the SDK, not front-end chart URLs or ad-hoc HTTP headers. The front-end APIs
often return `401` / `403` even when the same key works through the SDK.

```bash
export SWANLAB_API_KEY='...'
/tmp/swanlab-read/bin/python -c 'import swanlab; print(swanlab.__version__)'
```

If `/tmp/swanlab-read` is missing, create a throwaway venv under `/tmp` and
install `swanlab`.

## Find Runs

Use the public project path shown in SwanLab URLs:

```python
import os
import swanlab

api = swanlab.Api(api_key=os.environ["SWANLAB_API_KEY"])

for run in api.runs(path="qunteam/worldmodels"):
    name = run.name or ""
    if any(s in name.lower() for s in ["hetero", "probe", "gate", "fixbug"]):
        print(name, run.id, run.state, run.url)
```

Known adaptive-resolution runs:

| Run name | Run id |
|---|---|
| `tworoom_lewm_hetero_default` | `gps6asjv22tmflag9af5m` |
| `pusht_lewm_hetero_default` | `tge50bhmtws06xc7n4wtq` |
| `tworoom_lewm_hetero_probe_default` | `75qiqru0ttwmyy7pwigly` |
| `pusht_lewm_hetero_probe_default` | `jgqsw29zji110j3gczu03` |
| `tworoom_lewm_hetero_probe_default_action_gate` | `awokxbepmodp2shcqmynr` |
| `pusht_lewm_hetero_probe_default_action_gate` | `oezw5j3w0uh3ydxnan63c` |
| `tworoom_lewm_hetero_probe_default_action_gate_fixbug` | `oub19krd3fbecaav7bgie` |
| `pusht_lewm_hetero_probe_default_action_gate_fixbug` | `pare2urey6j6nucr9209m` |

Fetch one run directly by id:

```python
run = api.run(path="qunteam/worldmodels/pare2urey6j6nucr9209m")
print(run.name, run.id, run.state, run.url)
```

## Discover Metric Keys First

Always list columns before reading metrics. `run.metrics(keys=[...])` can fail
the whole request on a missing key with `404 Not Found`, and probing many missing
keys one by one is slow.

```python
def list_metric_keys(run):
    columns, _response = run._client.get(f"/experiment/{run.id}/column")
    return [item["key"] for item in columns.get("list", [])]

keys = list_metric_keys(run)
for key in keys:
    print(key)
```

In current SwanLab, the no-params `/column` request is the most reliable path.
Passing `params={"index": 1, "size": 50}` has returned `400 Bad Request` on some
runs. If the first page looks truncated, use the UI or a small query by exact
key family rather than assuming pagination parameters are stable.

Lightning metric keys are stage-prefixed:

```text
fit/hetero_s_logerr_corr
validate/hetero_s_logerr_corr_epoch
fit/pred_loss_mse_equiv
validate/pred_loss_mse_equiv_epoch
fit/adaptive_corr_sigma_action
validate/adaptive_corr_sigma_action_epoch
```

Unprefixed keys such as `hetero_s_logerr_corr` usually return `404`.

## Safe Metric Reader

Use this pattern for day-to-day analysis. It discovers available columns,
filters requested keys to existing keys, then reads only those keys.

```python
import os
import swanlab

api = swanlab.Api(api_key=os.environ["SWANLAB_API_KEY"])
run = api.run(path="qunteam/worldmodels/pare2urey6j6nucr9209m")

wanted = [
    "validate/hetero_s_logerr_corr_epoch",
    "validate/hetero_err_mean_epoch",
    "validate/pred_loss_epoch",
    "validate/pred_loss_mse_equiv_epoch",
    "validate/sigma_probe_loss_epoch",
    "validate/sigreg_loss_epoch",
    "validate/loss_epoch",
    "fit/adaptive_action_sensitivity_mean",
    "fit/adaptive_action_sensitivity_cv_mean",
    "fit/adaptive_action_sensitivity_cv_high_A",
    "fit/adaptive_gA_mean",
    "fit/adaptive_gS_mean",
    "fit/adaptive_critical_mean",
    "fit/adaptive_weight_mean",
    "fit/adaptive_weight_q10",
    "fit/adaptive_weight_q90",
    "fit/adaptive_corr_sigma_action",
    "fit/adaptive_in_warmup",
]

available = set(list_metric_keys(run))
keys = [k for k in wanted if k in available]
missing = [k for k in wanted if k not in available]

print("RUN", run.name, run.id, run.state)
print("MISSING", missing)

if keys:
    df = run.metrics(keys=keys)
    for key in keys:
        series = df[key].dropna() if key in df.columns else []
        if len(series) == 0:
            print("EMPTY", key)
            continue
        tail_n = min(100, len(series))
        print(
            key,
            "n=", len(series),
            "first=", float(series.iloc[0]),
            "last=", float(series.iloc[-1]),
            "min=", float(series.min()),
            "max=", float(series.max()),
            "tail_mean=", float(series.tail(tail_n).mean()),
        )
```

## Chart Id To Metric Key

Chart links look like:

```text
https://swanlab.cn/@qunteam/worldmodels/chart/default/<run_id>/<chart_id>
```

Use the authenticated SDK client to map a chart id to its metric key:

```python
chart_id = "kxbr3K_K"
info, _response = run._client.get(f"/experiment/{run.id}/chart/{chart_id}/info")
print(info["title"])
print([axis["key"] for axis in info["config"]["yAxis"]])
```

Known TwoRoom hetero chart ids:

| Chart id | Metric key |
|---|---|
| `kxbr3K_K` | `fit/hetero_s_mean` |
| `-CpThSeH` | `fit/hetero_s_std` |
| `AN8W6K2c` | `fit/hetero_weight_q10` |
| `en8Z6wwj` | `fit/hetero_weight_q90` |
| `hIYga6BL` | `fit/hetero_s_logerr_corr` |

## Direct API Pitfalls

Avoid direct front-end calls unless the SDK path is unavailable:

```text
GET /api/experiment/<run_id>/chart/<chart_id>/info
GET /api/project/<workspace>/<project>/runs/metrics
GET /api/experiment/<run_id>/column/csv?key=<metric>
```

Observed failures:

| Status | Meaning in this workflow | Usual fix |
|---|---|---|
| `400 Bad Request` | Column pagination params are not accepted for this run/API version | Use no-params `/column` first |
| `401 Unauthorized` | Key not accepted by this endpoint/header form | Use `swanlab.Api(api_key=...)` |
| `403 Forbidden` | Endpoint sees no project READ role | Use SDK authenticated client or confirm workspace access |
| `404 Not Found` | Metric key is wrong, missing, or the run never logged that stage/key | List columns and request only existing keys |

Do not pass a large guessed key list directly to `run.metrics`. A single
missing key can fail the whole call.

## Interpretation Checklist

For sigma-conditioned JEPA runs, read these first:

| Metric | What to check |
|---|---|
| `validate/hetero_s_logerr_corr_epoch` | Whether sigma tracks prediction error on validation. Stable positive values mean the head learned difficulty. |
| `fit/hetero_s_logerr_corr` | Step-level training version, if logged. |
| `validate/hetero_s_std_epoch` and `validate/hetero_s_abs_max_epoch` | Whether sigma is non-constant and whether it hits clamp bounds. |
| `validate/hetero_weight_q10_q90_ratio_epoch` | For hetero-loss runs only. Very small values indicate strong hard/easy gradient imbalance. |
| `validate/pred_loss_mse_equiv_epoch` vs `validate/pred_loss_epoch` | Whether true MSE keeps improving or hetero loss only improves through weighting. |
| `fit/adaptive_corr_sigma_action` | Whether sigma and action sensitivity are partially independent. Weak/moderate correlation is expected. |
| `fit/adaptive_action_sensitivity_cv_high_A` | Whether high-action-sensitivity states are stable enough for a useful gate. |
| `fit/adaptive_weight_q10`, `fit/adaptive_weight_q90` | Whether the proposed consistency weights have nontrivial spread. |

For PushT, combine SwanLab curves with local representation diagnostics:

```text
clean_nn_cos_dist_median
clean_effective_rank
transition_resolution_ratio_cos
transition_resolution_ratio_l2
id_probe_r2
action_mean_pred_shift_norm
```

If hetero reweighting is strong while these resolution metrics collapse, the
model is likely downweighting hard-but-task-critical transitions.
