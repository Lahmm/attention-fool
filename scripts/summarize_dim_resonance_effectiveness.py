#!/usr/bin/env python3
"""Summarize DIM-resonance transfer effectiveness CSVs.

The runner writes one transfer_eval CSV per adversarial directory via
record_experiment.py.  This script collects the latest row for each configured
variant, ranks average ASR, and writes both machine-readable CSV and a short
Chinese conclusion.
"""
import argparse
import csv
from pathlib import Path

VARIANTS = (
    "dim_mi_noaug",
    "reference_djf",
    "dim_resonance_only",
    "dim_resonance_djf",
    "fft_lowboost_only",
    "fft_lowboost_djf",
    "dim_adjoint_echo_only",
    "dim_adjoint_echo_djf",
)
MODEL_COLS = (
    "deit_base_patch16_224",
    "beit_base_patch16_224",
    "swin_tiny_patch4_window7_224",
    "pvt_v2_b2",
    "cait_s24_224",
    "levit_256",
    "pit_s_224",
    "crossvit_15_240",
)


def csv_path_for_variant(repo: Path, root: Path, variant: str) -> Path:
    adv_dir = (root / variant).resolve()
    rel = adv_dir.relative_to(repo.resolve()).as_posix()
    stem = "_".join(part for part in rel.replace("/", "_").split("_") if part)
    return repo / "outputs" / "csv" / f"{stem}.csv"


def latest_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else None


def f(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/attack/lazyagg/dim_resonance_effectiveness")
    parser.add_argument("--output-dir", default="outputs/analysis")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    root = (repo / args.root).resolve() if not Path(args.root).is_absolute() else Path(args.root).resolve()
    output_dir = repo / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    missing = []
    for variant in VARIANTS:
        path = csv_path_for_variant(repo, root, variant)
        row = latest_row(path)
        if row is None:
            missing.append((variant, path.as_posix()))
            continue
        item = {"variant": variant, "csv_path": path.relative_to(repo).as_posix()}
        item["avg"] = f(row, "avg")
        for model in MODEL_COLS:
            item[model] = f(row, model)
        rows.append(item)

    rows.sort(key=lambda item: (-1.0 if item["avg"] is None else -item["avg"], item["variant"]))
    reference = next((item for item in rows if item["variant"] == "reference_djf"), None)
    for item in rows:
        item["delta_vs_reference_djf"] = None if reference is None or item["avg"] is None else item["avg"] - reference["avg"]

    out_csv = output_dir / "dim_resonance_effectiveness_summary.csv"
    fields = ["variant", "avg", "delta_vs_reference_djf", *MODEL_COLS, "csv_path"]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in rows:
            writer.writerow({key: item.get(key) for key in fields})

    best = rows[0] if rows else None
    md = ["# DIM Resonance Effectiveness Summary", ""]
    md.append(f"Root: `{root.relative_to(repo).as_posix() if root.is_relative_to(repo) else root.as_posix()}`")
    md.append("")
    if best is None:
        md.append("No completed transfer-eval CSVs were found.")
    else:
        md.append(f"Best avg ASR: `{best['variant']}` = `{best['avg']:.6f}`.")
        if reference is not None and best["delta_vs_reference_djf"] is not None:
            md.append(f"Delta vs `reference_djf`: `{best['delta_vs_reference_djf']:+.6f}`.")
        md.append("")
        md.append("## Ranking")
        for idx, item in enumerate(rows, 1):
            delta = item["delta_vs_reference_djf"]
            delta_text = "n/a" if delta is None else f"{delta:+.6f}"
            md.append(f"{idx}. `{item['variant']}` avg=`{item['avg']:.6f}` delta_vs_reference=`{delta_text}`")
    if missing:
        md.append("")
        md.append("## Missing")
        for variant, path in missing:
            md.append(f"- `{variant}`: `{path}`")

    out_md = output_dir / "dim_resonance_effectiveness_summary.md"
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {out_csv}")
    print(f"wrote {out_md}")
    if best is not None:
        print(f"best={best['variant']} avg={best['avg']:.6f}")


if __name__ == "__main__":
    main()
