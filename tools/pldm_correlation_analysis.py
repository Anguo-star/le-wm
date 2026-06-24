"""Cross-method (LeWM ↔ PLDM) correlation analysis for the
invariance–resolution paper.

For each of the four tasks (TwoRoom, PushT, Reacher, Cube) we compute:

1.  Within-PLDM Spearman ρ(metric, clean) / ρ(metric, observation-noise drop)
    and the partial Spearman conditioned on std_max. n=9 for every task.
2.  Joint LeWM+PLDM Spearman and partial-on-(std_max, method dummy).
    This is n=18 for every task.

Outputs JSON for direct paper-table inclusion and prints a human-readable
summary.

Run::

    python -m tools.pldm_correlation_analysis \\
        --evals-lewm assets/paper1_data/canonical_evals_20260517.json \\
        --evals-pldm assets/paper1_data/canonical_evals_pldm_20260522.json \\
        --diag-lewm  assets/paper1_data/canonical_diagnostics_20260517.json \\
        --diag-pldm  assets/paper1_data/canonical_diagnostics_pldm_20260522.json \\
        --out        assets/paper1_data/cross_method_corr_pldm_20260522.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


# --------------------------------------------------------------------------- #
# rank-based statistics: same conventions as paper1_figs._spearman
# --------------------------------------------------------------------------- #
def _avg_ranks(xs: Sequence[float]) -> list[float]:
    n = len(xs)
    order = sorted(range(n), key=lambda i: xs[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(xs, ys))
    dx = sum((xi - mx) ** 2 for xi in xs) ** 0.5
    dy = sum((yi - my) ** 2 for yi in ys) ** 0.5
    denom = max(dx * dy, 1e-12)
    return num / denom


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 3 or len(xs) != len(ys):
        return float("nan")
    return _pearson(_avg_ranks(xs), _avg_ranks(ys))


def partial_spearman_one(
    xs: Sequence[float], ys: Sequence[float], zs: Sequence[float]
) -> float:
    """Partial Spearman of x and y given a single covariate z."""
    if len(xs) < 4 or not (len(xs) == len(ys) == len(zs)):
        return float("nan")
    rxz = spearman(xs, zs)
    ryz = spearman(ys, zs)
    rxy = spearman(xs, ys)
    denom = ((1 - rxz ** 2) * (1 - ryz ** 2)) ** 0.5
    if denom < 1e-9:
        return float("nan")
    return (rxy - rxz * ryz) / denom


def partial_spearman_two(
    xs: Sequence[float], ys: Sequence[float],
    z1: Sequence[float], z2: Sequence[float],
) -> float:
    """Partial Spearman of x and y given two covariates (z1, z2).

    Iterates the single-covariate residualisation: first remove z1, then
    z2. Stable for the small n=18 case we care about because the
    covariates we use (std_max, method-dummy) are not highly collinear.
    """
    # Residualise xs and ys against z1 in rank space, then partial on z2.
    def _resid_against(a: Sequence[float], b: Sequence[float]) -> list[float]:
        ra = _avg_ranks(a)
        rb = _avg_ranks(b)
        mb = sum(rb) / len(rb)
        ma = sum(ra) / len(ra)
        cov = sum((bi - mb) * (ai - ma) for ai, bi in zip(ra, rb))
        varb = sum((bi - mb) ** 2 for bi in rb)
        slope = cov / max(varb, 1e-12)
        intercept = ma - slope * mb
        return [ai - (slope * bi + intercept) for ai, bi in zip(ra, rb)]

    if len(xs) < 4:
        return float("nan")
    xs1 = _resid_against(xs, z1)
    ys1 = _resid_against(ys, z1)
    z2_r1 = _resid_against(z2, z1)
    # partial Spearman conditioned on z2 (after z1 already removed)
    return partial_spearman_one(xs1, ys1, z2_r1)


# --------------------------------------------------------------------------- #
# Data assembly
# --------------------------------------------------------------------------- #
TASKS = ("TwoRoom", "PushT", "Reacher", "Cube")
FRAG_KEY = "predictor_target_to_nn_cos_ratio_at_max_std"
DRIFT_KEY = "predictor_rollout_T8_l2_at_max_std"
ROBUST_EVAL_KEY = "pixels_std0.08"


def assemble_rows(
    evals_lewm: dict, evals_pldm: dict,
    diag_lewm: dict, diag_pldm: dict,
    task: str,
    method: str,  # 'LeWM' or 'PLDM'
) -> list[dict]:
    evals = evals_lewm if method == "LeWM" else evals_pldm
    diag = diag_lewm if method == "LeWM" else diag_pldm
    pmbt = diag["predictor_metrics_by_task"][task]
    rows = []
    for std_key in sorted(evals[task].keys(), key=float):
        if std_key not in pmbt:
            continue
        try:
            m = evals[task][std_key]["metrics"]
            clean = m["clean"]["mean"]
            px08 = m[ROBUST_EVAL_KEY]["mean"]
        except KeyError:
            continue
        d = pmbt[std_key]
        rows.append({
            "std_max": float(std_key),
            "method": method,
            "method_dummy": 0 if method == "LeWM" else 1,
            "clean": clean,
            "px08": px08,
            "drop": clean - px08,
            "frag": d[FRAG_KEY],
            "drift": d.get(DRIFT_KEY, float("nan")),
        })
    return rows


def task_block(
    evals_lewm: dict, evals_pldm: dict,
    diag_lewm: dict, diag_pldm: dict, task: str,
) -> dict:
    lewm_rows = assemble_rows(evals_lewm, evals_pldm, diag_lewm, diag_pldm, task, "LeWM")
    pldm_rows = assemble_rows(evals_lewm, evals_pldm, diag_lewm, diag_pldm, task, "PLDM")

    def _within(rows: list[dict], metric: str) -> dict:
        xs = [r[metric] for r in rows]
        clean = [r["clean"] for r in rows]
        px08 = [r["px08"] for r in rows]
        drop = [r["drop"] for r in rows]
        std = [r["std_max"] for r in rows]
        return {
            "n": len(rows),
            "rho_metric_clean": spearman(xs, clean),
            "rho_metric_px08": spearman(xs, px08),
            "rho_metric_drop": spearman(xs, drop),
            "partial_metric_clean_on_std": partial_spearman_one(xs, clean, std),
            "partial_metric_px08_on_std": partial_spearman_one(xs, px08, std),
            "partial_metric_drop_on_std": partial_spearman_one(xs, drop, std),
        }

    def _joint(metric: str) -> dict:
        rows = lewm_rows + pldm_rows
        xs = [r[metric] for r in rows]
        clean = [r["clean"] for r in rows]
        px08 = [r["px08"] for r in rows]
        drop = [r["drop"] for r in rows]
        std = [r["std_max"] for r in rows]
        meth = [float(r["method_dummy"]) for r in rows]
        return {
            "n": len(rows),
            "rho_metric_clean": spearman(xs, clean),
            "rho_metric_px08": spearman(xs, px08),
            "rho_metric_drop": spearman(xs, drop),
            "partial_metric_clean_on_std_method": partial_spearman_two(xs, clean, std, meth),
            "partial_metric_px08_on_std_method": partial_spearman_two(xs, px08, std, meth),
            "partial_metric_drop_on_std_method": partial_spearman_two(xs, drop, std, meth),
        }

    return {
        "within_lewm": {
            "frag": _within(lewm_rows, "frag"),
            "drift": _within(lewm_rows, "drift"),
        },
        "within_pldm": {
            "frag": _within(pldm_rows, "frag"),
            "drift": _within(pldm_rows, "drift"),
        },
        "joint": {
            "frag": _joint("frag"),
            "drift": _joint("drift"),
        },
        "rows": {"lewm": lewm_rows, "pldm": pldm_rows},
    }


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def _fmt(x: float) -> str:
    return "  --" if x != x else f"{x:+.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evals-lewm", default="assets/paper1_data/canonical_evals_20260517.json")
    ap.add_argument("--evals-pldm", default="assets/paper1_data/canonical_evals_pldm_20260522.json")
    ap.add_argument("--diag-lewm",  default="assets/paper1_data/canonical_diagnostics_20260517.json")
    ap.add_argument("--diag-pldm",  default="assets/paper1_data/canonical_diagnostics_pldm_20260522.json")
    ap.add_argument("--out",        default="assets/paper1_data/cross_method_corr_pldm_20260522.json")
    args = ap.parse_args()
    repo_root = Path(__file__).resolve().parent.parent

    def load(p: str) -> dict:
        path = Path(p)
        if not path.is_absolute():
            path = repo_root / path
        return json.loads(path.read_text())

    evals_lewm = load(args.evals_lewm)
    evals_pldm = load(args.evals_pldm)
    diag_lewm = load(args.diag_lewm)
    diag_pldm = load(args.diag_pldm)

    blocks = {
        t: task_block(evals_lewm, evals_pldm, diag_lewm, diag_pldm, t)
        for t in TASKS
    }

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = repo_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(blocks, indent=2, sort_keys=True))
    print(f"wrote {out_path}\n")

    # Human-readable summary
    print("=" * 96)
    print(f"{'Task':<10}{'n_lewm':>8}{'n_pldm':>8}  ||  "
          f"{'rho_clean(lewm)':>18}{'rho_clean(pldm)':>18}{'rho_clean(joint)':>20}")
    print(f"{'':<10}{'':>8}{'':>8}  ||  "
          f"{'p|std(lewm)':>18}{'p|std(pldm)':>18}{'p|std,m(joint)':>20}")
    print(f"{'':<10}{'':>8}{'':>8}  ||  "
          f"{'rho_drop(lewm)':>18}{'rho_drop(pldm)':>18}{'rho_drop(joint)':>20}")
    print(f"{'':<10}{'':>8}{'':>8}  ||  "
          f"{'p|std(lewm)':>18}{'p|std(pldm)':>18}{'p|std,m(joint)':>20}")
    print("=" * 96)
    for t in TASKS:
        b = blocks[t]
        wl, wp, j = b["within_lewm"]["frag"], b["within_pldm"]["frag"], b["joint"]["frag"]
        n_l, n_p = wl["n"], wp["n"]
        print(f"{t:<10}{n_l:>8}{n_p:>8}  ||  "
              f"{_fmt(wl['rho_metric_clean']):>18}{_fmt(wp['rho_metric_clean']):>18}{_fmt(j['rho_metric_clean']):>20}")
        print(f"{'':<10}{'':>8}{'':>8}  ||  "
              f"{_fmt(wl['partial_metric_clean_on_std']):>18}{_fmt(wp['partial_metric_clean_on_std']):>18}{_fmt(j['partial_metric_clean_on_std_method']):>20}")
        print(f"{'':<10}{'':>8}{'':>8}  ||  "
              f"{_fmt(wl['rho_metric_drop']):>18}{_fmt(wp['rho_metric_drop']):>18}{_fmt(j['rho_metric_drop']):>20}")
        print(f"{'':<10}{'':>8}{'':>8}  ||  "
              f"{_fmt(wl['partial_metric_drop_on_std']):>18}{_fmt(wp['partial_metric_drop_on_std']):>18}{_fmt(j['partial_metric_drop_on_std_method']):>20}")
        print("-" * 96)


if __name__ == "__main__":
    main()
