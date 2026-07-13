"""Aggregate per-run results.json files into Table-2-style summaries.

Scans a logs directory for ``results.json`` files, groups them by experiment
name, and reports per-seed and mean +/- std segment-wise F1 (overall, average
class-wise, and per-class). Writes a CSV and a Markdown table.

Usage:
    python -m src.aggregate_results --logs_dir logs --out_dir results
"""

import argparse
import glob
import json
import os
from collections import defaultdict
from statistics import mean, pstdev
from typing import Dict, List

from src.utils.sed_eval_utils import URBAN_SED_CLASSES


def load_runs(logs_dir: str) -> List[Dict]:
    runs = []
    for path in glob.glob(os.path.join(logs_dir, "**", "results.json"), recursive=True):
        try:
            with open(path) as f:
                runs.append(json.load(f))
        except Exception as e:  # noqa: BLE001
            print(f"[warn] could not read {path}: {e}")
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", default="logs")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    runs = load_runs(args.logs_dir)
    if not runs:
        print(f"No results.json found under {args.logs_dir}")
        return

    by_exp: Dict[str, List[Dict]] = defaultdict(list)
    for r in runs:
        by_exp[r.get("experiment_name", "unknown")].append(r)

    rows = []
    for exp, exp_runs in sorted(by_exp.items()):
        seeds = [r.get("seed") for r in exp_runs]
        overall = [r["test"]["overall_f1_mean"] for r in exp_runs]
        avg_cw = [r["test"]["average_class_f1_mean"] for r in exp_runs]
        row = {
            "experiment": exp,
            "n_seeds": len(exp_runs),
            "seeds": seeds,
            "overall_f1_mean": mean(overall),
            "overall_f1_std": pstdev(overall) if len(overall) > 1 else 0.0,
            "average_class_f1_mean": mean(avg_cw),
            "average_class_f1_std": pstdev(avg_cw) if len(avg_cw) > 1 else 0.0,
        }
        for c in URBAN_SED_CLASSES:
            vals = [r["test"]["class_wise_mean"][c] for r in exp_runs]
            row[c] = mean(vals)
        rows.append(row)

    # CSV
    import csv

    csv_path = os.path.join(args.out_dir, "table2.csv")
    fields = (
        ["experiment", "n_seeds", "overall_f1_mean", "overall_f1_std",
         "average_class_f1_mean", "average_class_f1_std"] + URBAN_SED_CLASSES
    )
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    # Markdown
    md_path = os.path.join(args.out_dir, "table2.md")
    with open(md_path, "w") as f:
        f.write("# URBAN-SED segment-wise F1 (%)\n\n")
        f.write("| Experiment | Seeds | Overall F1 | Avg class F1 |\n")
        f.write("|---|---|---|---|\n")
        for row in rows:
            f.write(
                f"| {row['experiment']} | {row['n_seeds']} | "
                f"{row['overall_f1_mean']:.2f} ± {row['overall_f1_std']:.2f} | "
                f"{row['average_class_f1_mean']:.2f} ± {row['average_class_f1_std']:.2f} |\n"
            )
        f.write("\n## Class-wise mean F1\n\n")
        f.write("| Class | " + " | ".join(r["experiment"] for r in rows) + " |\n")
        f.write("|---" * (len(rows) + 1) + "|\n")
        for c in URBAN_SED_CLASSES:
            f.write(f"| {c} | " + " | ".join(f"{r[c]:.2f}" for r in rows) + " |\n")

    print(f"Wrote {csv_path} and {md_path}")
    for row in rows:
        print(f"{row['experiment']:12s} overall={row['overall_f1_mean']:.2f}±"
              f"{row['overall_f1_std']:.2f}  avg_class={row['average_class_f1_mean']:.2f}")


if __name__ == "__main__":
    main()
