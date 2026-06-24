"""Bootstrap 95% confidence intervals for the released partial Spearman
correlation artifacts used by Paper 1.

For each (scope, task, metric, outcome, conditioning) cell that the
paper reports we:

1.  Compute the point estimate using the *same* rank-based and
    partial-correlation code path as
    ``tools.pldm_correlation_analysis`` (the source of truth for
    Tables 6, 7 and 16). The point estimates emitted here therefore
    match the paper to numerical precision.
2.  Bootstrap with replacement, B = 1000 iterations by default. Each
    iteration resamples checkpoint rows independently within the
    scope (within-LeWM n = 9, within-PLDM n = 9, or joint
    LeWM+PLDM n = 18).
3.  Report the percentile 95% confidence interval
    ``[percentile 2.5, percentile 97.5]`` over the bootstrap samples
    that returned a finite value. Iterations that produce NaN — e.g.
    when a resample contains no method variance in the joint case, or
    when rank ties collapse the partial-correlation denominator on a
    saturated task — are counted and excluded from the CI.

Output schema (one block per task, mirroring the layout used by
``cross_method_corr_pldm_*.json`` but with a ``"ci"`` and
``"n_valid"`` field next to every numeric quantity)::

    {
      "metadata": {"n_bootstrap": 1000, "seed": 42, ...},
      "by_task": {
        "TwoRoom": {
          "within_lewm": {
            "n": 9,
            "frag":  {"rho_metric_clean": {"point": ..., "ci": [lo, hi], "n_valid": ...}, ...},
            "drift": {...}
          },
          "within_pldm": {"n": 9, ...},
          "joint": {"n": 18, ...}
        },
        ...
      }
    }

Run::

    python -m tools.build_partial_corr_bootstrap \\
        --out assets/paper1_data/partial_corr_bootstrap_20260523.json

Optional flags:
    --n-bootstrap 1000    # number of bootstrap iterations (default 1000)
    --seed 42             # RNG seed (default 42)
    --ci-low  2.5         # lower percentile (default 2.5)
    --ci-high 97.5        # upper percentile (default 97.5)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import random
from pathlib import Path
from typing import Sequence

# Reuse the paper's published rank/partial-correlation implementations so
# the point estimates here are byte-for-byte identical to Tables 6/7/16.
from tools.pldm_correlation_analysis import (  # noqa: E402
    TASKS,
    assemble_rows,
    partial_spearman_one,
    partial_spearman_two,
    spearman,
)


# --------------------------------------------------------------------------- #
# Cells we evaluate.
# --------------------------------------------------------------------------- #
METRICS = ("frag", "drift")
OUTCOMES = ("clean", "px08", "drop")


def _percentile(values: Sequence[float], pct: float) -> float:
    """Linear-interpolation percentile (matches ``numpy.percentile``)."""
    vs = sorted(values)
    n = len(vs)
    if n == 0:
        return float("nan")
    k = (n - 1) * pct / 100.0
    lo = int(math.floor(k))
    hi = min(lo + 1, n - 1)
    frac = k - lo
    return vs[lo] * (1.0 - frac) + vs[hi] * frac


def _is_finite(v) -> bool:
    """Return True iff v is a real, finite number. Bootstrap resamples
    occasionally push ranks into degenerate configurations (a sample
    with all rows the same, all-LeWM in the joint case, etc.); when
    the partial-correlation formula divides by a near-zero denominator
    in such configurations, floating-point can return ``complex`` or
    NaN. We treat both as invalid and skip them."""
    if isinstance(v, complex):
        return False
    return v == v and not math.isinf(v)


def _bootstrap_indices(n: int, rng: random.Random) -> list[int]:
    return [rng.randrange(n) for _ in range(n)]


def _safe(fn, *args):
    """Call partial-correlation function and convert any
    numerical exception (TypeError on complex compare, ValueError on
    sqrt of negative, ZeroDivisionError on collinear z) into NaN."""
    try:
        return fn(*args)
    except (TypeError, ValueError, ZeroDivisionError):
        return float("nan")


def _within_scope_cell(
    rows: list[dict], metric: str, outcome: str,
    n_bootstrap: int, ci_low: float, ci_high: float, rng: random.Random,
) -> dict:
    """Compute unconditional + partial-on-std cells for one
    (rows, metric, outcome) triple, both point and bootstrap."""
    xs_all = [r[metric] for r in rows]
    ys_all = [r[outcome] for r in rows]
    zs_all = [r["std_max"] for r in rows]

    rho_point = spearman(xs_all, ys_all)
    partial_point = partial_spearman_one(xs_all, ys_all, zs_all)

    rho_bs: list[float] = []
    partial_bs: list[float] = []
    n = len(rows)
    for _ in range(n_bootstrap):
        idx = _bootstrap_indices(n, rng)
        xs = [xs_all[i] for i in idx]
        ys = [ys_all[i] for i in idx]
        zs = [zs_all[i] for i in idx]
        rho = _safe(spearman, xs, ys)
        partial = _safe(partial_spearman_one, xs, ys, zs)
        if _is_finite(rho):
            rho_bs.append(rho)
        if _is_finite(partial):
            partial_bs.append(partial)

    return {
        f"rho_metric_{outcome}": _summary(rho_point, rho_bs, ci_low, ci_high, n_bootstrap),
        f"partial_metric_{outcome}_on_std": _summary(
            partial_point, partial_bs, ci_low, ci_high, n_bootstrap,
        ),
    }


def _joint_scope_cell(
    rows: list[dict], metric: str, outcome: str,
    n_bootstrap: int, ci_low: float, ci_high: float, rng: random.Random,
) -> dict:
    """Joint LeWM+PLDM cell: unconditional + partial-on-(std_max, method)."""
    xs_all = [r[metric] for r in rows]
    ys_all = [r[outcome] for r in rows]
    zs_all = [r["std_max"] for r in rows]
    ms_all = [float(r["method_dummy"]) for r in rows]

    rho_point = spearman(xs_all, ys_all)
    partial_point = partial_spearman_two(xs_all, ys_all, zs_all, ms_all)

    rho_bs: list[float] = []
    partial_bs: list[float] = []
    n = len(rows)
    for _ in range(n_bootstrap):
        idx = _bootstrap_indices(n, rng)
        xs = [xs_all[i] for i in idx]
        ys = [ys_all[i] for i in idx]
        zs = [zs_all[i] for i in idx]
        ms = [ms_all[i] for i in idx]
        rho = _safe(spearman, xs, ys)
        partial = _safe(partial_spearman_two, xs, ys, zs, ms)
        if _is_finite(rho):
            rho_bs.append(rho)
        if _is_finite(partial):
            partial_bs.append(partial)

    return {
        f"rho_metric_{outcome}": _summary(rho_point, rho_bs, ci_low, ci_high, n_bootstrap),
        f"partial_metric_{outcome}_on_std_method": _summary(
            partial_point, partial_bs, ci_low, ci_high, n_bootstrap,
        ),
    }


def _summary(
    point: float, bootstrap_samples: list[float],
    ci_low: float, ci_high: float, n_bootstrap: int,
) -> dict:
    n_valid = len(bootstrap_samples)
    # Suppress the CI when bootstrap diverged on most iterations (rank-tie
    # saturation on TwoRoom under partial-on-std; method-collinearity on
    # the joint case when a resample is all-LeWM or all-PLDM). The
    # downstream paper script consults n_valid before quoting a CI.
    if n_valid >= max(50, n_bootstrap // 4):
        ci = [
            _percentile(bootstrap_samples, ci_low),
            _percentile(bootstrap_samples, ci_high),
        ]
    else:
        ci = None
    return {
        "point": point if _is_finite(point) else None,
        "ci": ci,
        "n_valid": n_valid,
    }


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build_block(
    task: str,
    evals_lewm: dict, evals_pldm: dict,
    diag_lewm: dict, diag_pldm: dict,
    n_bootstrap: int, ci_low: float, ci_high: float, rng: random.Random,
) -> dict:
    lewm_rows = assemble_rows(evals_lewm, evals_pldm, diag_lewm, diag_pldm, task, "LeWM")
    pldm_rows = assemble_rows(evals_lewm, evals_pldm, diag_lewm, diag_pldm, task, "PLDM")
    joint_rows = lewm_rows + pldm_rows

    def _scope(rows: list[dict], scope_kind: str) -> dict:
        out: dict[str, dict] = {"n": len(rows)}
        for metric in METRICS:
            cells: dict[str, dict] = {}
            for outcome in OUTCOMES:
                if scope_kind == "joint":
                    cells.update(_joint_scope_cell(
                        rows, metric, outcome, n_bootstrap, ci_low, ci_high, rng,
                    ))
                else:
                    cells.update(_within_scope_cell(
                        rows, metric, outcome, n_bootstrap, ci_low, ci_high, rng,
                    ))
            out[metric] = cells
        return out

    return {
        "within_lewm": _scope(lewm_rows, "within"),
        "within_pldm": _scope(pldm_rows, "within"),
        "joint": _scope(joint_rows, "joint"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--evals-lewm",
                    default="assets/paper1_data/canonical_evals_20260517.json")
    ap.add_argument("--evals-pldm",
                    default="assets/paper1_data/canonical_evals_pldm_20260522.json")
    ap.add_argument("--diag-lewm",
                    default="assets/paper1_data/canonical_diagnostics_20260517.json")
    ap.add_argument("--diag-pldm",
                    default="assets/paper1_data/canonical_diagnostics_pldm_20260522.json")
    ap.add_argument("--out",
                    default="assets/paper1_data/partial_corr_bootstrap_20260523.json")
    ap.add_argument("--n-bootstrap", type=int, default=1000,
                    help="bootstrap iterations per cell (default 1000)")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed (default 42)")
    ap.add_argument("--ci-low", type=float, default=2.5,
                    help="lower percentile (default 2.5)")
    ap.add_argument("--ci-high", type=float, default=97.5,
                    help="upper percentile (default 97.5)")
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

    rng = random.Random(args.seed)

    by_task = {
        t: build_block(
            t, evals_lewm, evals_pldm, diag_lewm, diag_pldm,
            args.n_bootstrap, args.ci_low, args.ci_high, rng,
        )
        for t in TASKS
    }

    payload = {
        "metadata": {
            "schema_version": "1.0",
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
            "ci_low_pct": args.ci_low,
            "ci_high_pct": args.ci_high,
            "metrics": list(METRICS),
            "outcomes": list(OUTCOMES),
            "robust_eval_metric": "pixels_std0.08",
            "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "sources": {
                "evals_lewm": args.evals_lewm,
                "evals_pldm": args.evals_pldm,
                "diag_lewm": args.diag_lewm,
                "diag_pldm": args.diag_pldm,
            },
        },
        "by_task": by_task,
    }

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = repo_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {out_path}\n")

    _print_summary(by_task, args.ci_low, args.ci_high)


# --------------------------------------------------------------------------- #
# Human-readable summary
# --------------------------------------------------------------------------- #
def _fmt_cell(cell: dict) -> str:
    p = cell["point"]
    ci = cell["ci"]
    if p is None:
        return "      n/a"
    pt = f"{p:+.2f}"
    if ci is None:
        return f"{pt} [tie]"
    return f"{pt} [{ci[0]:+.2f}, {ci[1]:+.2f}]"


def _print_summary(by_task: dict, ci_low: float, ci_high: float) -> None:
    print("=" * 104)
    print(f"Partial Spearman ρ point estimate and {ci_high - ci_low:.0f}% bootstrap CI")
    print(f"(metric = fragility ratio; outcome = observation-noise drop; partial out of std_max")
    print(f" within-method, std_max + method-dummy for joint)")
    print("=" * 104)
    headline = f"{'Task':<10}{'within-LeWM':>30}{'within-PLDM':>30}{'joint LeWM+PLDM':>32}"
    print(headline)
    print("-" * 104)
    for task in TASKS:
        b = by_task[task]
        wl = b["within_lewm"]["frag"]["partial_metric_drop_on_std"]
        wp = b["within_pldm"]["frag"]["partial_metric_drop_on_std"]
        j = b["joint"]["frag"]["partial_metric_drop_on_std_method"]
        print(f"{task:<10}{_fmt_cell(wl):>30}{_fmt_cell(wp):>30}{_fmt_cell(j):>32}")
    print()
    print("PushT detailed (Table 7 layout, fragility ratio):")
    print("-" * 104)
    b = by_task["PushT"]["within_lewm"]["frag"]
    rows = [
        ("ρ(metric, clean) uncond.",       b["rho_metric_clean"]),
        ("ρ(metric, clean) | std_max",     b["partial_metric_clean_on_std"]),
        ("ρ(metric, pixels 0.08) uncond.",   b["rho_metric_px08"]),
        ("ρ(metric, pixels 0.08) | std_max", b["partial_metric_px08_on_std"]),
        ("ρ(metric, obs-noise drop) uncond.",    b["rho_metric_drop"]),
        ("ρ(metric, obs-noise drop) | std_max",  b["partial_metric_drop_on_std"]),
    ]
    for label, cell in rows:
        print(f"  {label:<38}{_fmt_cell(cell):>34}")


if __name__ == "__main__":
    main()
