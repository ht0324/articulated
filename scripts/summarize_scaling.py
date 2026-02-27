"""Summarize scaling sweep outputs into tabular files.

This script scans a sweep run directory for `summary.json` files (emitted by
`scripts/run_scaling_sweep.py`), flattens records to CSV, and writes an
aggregated (mean/std/count by architecture and train_size) CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize scaling sweep outputs")
    parser.add_argument(
        "--runs-root",
        type=Path,
        required=True,
        help="Root directory containing per-run summary.json files",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path for all runs (default: <runs-root>/all_runs.csv)",
    )
    parser.add_argument(
        "--aggregate-csv",
        type=Path,
        default=None,
        help=(
            "Output CSV path for aggregated metrics "
            "(default: <runs-root>/aggregated_by_arch_train_size.csv)"
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def flatten_summary(record: dict[str, Any]) -> dict[str, Any]:
    out = dict(record)

    representation = out.pop("representation", None)
    if isinstance(representation, dict):
        for key, value in representation.items():
            out[f"repr_{key}"] = value

    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["no_rows"])
        return

    keys: set[str] = set()
    for row in rows:
        keys.update(row.keys())
    fieldnames = sorted(keys)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _std(xs: list[float], mean: float) -> float:
    if len(xs) <= 1:
        return 0.0
    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
    return var**0.5


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "completed":
            continue

        arch = str(row.get("architecture", "unknown"))
        train_size = int(row.get("train_size", -1))
        grouped[(arch, train_size)].append(row)

    metrics = [
        "val_loss",
        "val_acc",
        "repr_pca_95_dims",
        "repr_pca_pc1_var",
        "repr_participation_ratio",
        "repr_so2_max_corr_theta1",
        "repr_so2_max_corr_theta2",
        "repr_so2_top3_mean_corr_theta1",
        "repr_so2_top3_mean_corr_theta2",
    ]

    aggregated: list[dict[str, Any]] = []
    for (arch, train_size), group in sorted(grouped.items(), key=lambda x: x[0]):
        row: dict[str, Any] = {
            "architecture": arch,
            "train_size": train_size,
            "n_runs": len(group),
        }

        for metric in metrics:
            values = []
            for rec in group:
                val = to_float(rec.get(metric))
                if val is not None:
                    values.append(val)

            if values:
                mean = _mean(values)
                row[f"{metric}_mean"] = mean
                row[f"{metric}_std"] = _std(values, mean)
            else:
                row[f"{metric}_mean"] = None
                row[f"{metric}_std"] = None

        aggregated.append(row)

    return aggregated


def main() -> None:
    args = parse_args()

    output_csv = args.output_csv or (args.runs_root / "all_runs.csv")
    aggregate_csv = args.aggregate_csv or (
        args.runs_root / "aggregated_by_arch_train_size.csv"
    )

    summary_paths = sorted(args.runs_root.rglob("summary.json"))
    if not summary_paths:
        print(f"No summary.json files found under {args.runs_root}")
        return

    rows = [flatten_summary(load_json(path)) for path in summary_paths]
    write_csv(output_csv, rows)

    aggregated_rows = aggregate_rows(rows)
    write_csv(aggregate_csv, aggregated_rows)

    status_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        status_counts[str(row.get("status", "unknown"))] += 1

    print(f"Scanned summaries: {len(rows)}")
    print("Status counts:")
    for status in sorted(status_counts):
        print(f"  {status}: {status_counts[status]}")
    print(f"Wrote all-run CSV: {output_csv}")
    print(f"Wrote aggregate CSV: {aggregate_csv}")


if __name__ == "__main__":
    main()
