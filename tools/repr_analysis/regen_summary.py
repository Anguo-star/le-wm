"""Regenerate diagnostics_summary.json from existing CSV artifacts."""

import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.repr_analysis.run_full_diagnostics import _summarize_noise_to_predictor_to_resolution
from tools.repr_analysis.latent_noise_sensitivity import summarize_latent_noise_geometry
from tools.repr_analysis.analyze_repr import to_serializable


def _csv_to_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = {}
            for k, v in row.items():
                if v == "" or v.lower() == "nan":
                    d[k] = float("nan")
                else:
                    try:
                        d[k] = float(v)
                    except ValueError:
                        d[k] = v
            rows.append(d)
    return rows


def _merge_csvs(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        if p.exists():
            rows.extend(_csv_to_rows(p))
    return rows


def regen(task_dir: Path, out_path: Path | None = None):
    p03 = task_dir / "repr_analysis" / "p03_diagnostics"
    latent = task_dir / "repr_analysis" / "latent_noise_diagnostics"
    new_base = task_dir / "repr_analysis" / "p03_diagnostics_new_baselines"

    noise_paths = [p03 / "noise_sensitivity.csv"]
    pred_paths = [p03 / "predictor_sensitivity.csv"]
    res_paths = [p03 / "task_resolution.csv"]
    latent_paths = [latent / "latent_noise_sensitivity.csv"]

    if new_base.exists():
        for sub in new_base.iterdir():
            if sub.is_dir():
                n = sub / "noise_sensitivity.csv"
                p = sub / "predictor_sensitivity.csv"
                r = sub / "task_resolution.csv"
                l = sub / "latent_noise_sensitivity.csv"
                if n.exists():
                    noise_paths.append(n)
                if p.exists():
                    pred_paths.append(p)
                if r.exists():
                    res_paths.append(r)
                if l.exists():
                    latent_paths.append(l)

    noise_rows = _merge_csvs(noise_paths)
    predictor_rows = _merge_csvs(pred_paths)
    resolution_rows = _merge_csvs(res_paths)
    latent_noise_rows = _merge_csvs(latent_paths)

    summary = _summarize_noise_to_predictor_to_resolution(
        noise_rows=noise_rows,
        predictor_rows=predictor_rows,
        resolution_rows=resolution_rows,
        latent_noise_rows=latent_noise_rows,
    )

    if out_path is None:
        out_path = p03 / "diagnostics_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_path} ({len(summary)} models)")
    for s in summary:
        print(f"  - {s['model']}")

    # Also regenerate latent_geometry_summary.csv/.json from the new data
    for scope_name, frame_scope in [("goal", "goal"), ("history", "history")]:
        geom = summarize_latent_noise_geometry(latent_noise_rows, frame_scope=frame_scope)
        if geom is not None and not geom.empty:
            csv_path = latent / f"latent_geometry_summary_{scope_name}.csv"
            json_path = latent / f"latent_geometry_summary_{scope_name}.json"
            geom.to_csv(csv_path, index=False)
            json_path.write_text(json.dumps(to_serializable(geom.to_dict(orient="records")), indent=2))
            print(f"  Wrote {csv_path} ({len(geom)} rows, scope={frame_scope})")


if __name__ == "__main__":
    for task in ["lewm-tworooms", "lewm-pusht"]:
        task_dir = Path(f"/opt/huawei/explorer-env/dataset/ag_data/data/world_model/quentinll/{task}")
        regen(task_dir)
