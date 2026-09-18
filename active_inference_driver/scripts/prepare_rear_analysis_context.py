"""Prepare a non-destructive context for the released rear-end analysis.

``Analysis_rear_end.py`` assumes its input contains at least two values of
``EA_mode``.  A full-model-only run makes that column constant, so the script
removes it and later accesses it unconditionally.  The paper's released
archive naturally includes ablations and does not hit this edge case.  This
helper copies native full results into an analysis-only directory and adds one
already-run no-evidence experiment solely to retain that metadata column.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import pandas as pd


def _append_setup(full_path: Path, contrast_path: Path, index: int) -> None:
    full = pd.read_excel(full_path, index_col=0, keep_default_na=False)
    contrast = pd.read_excel(contrast_path, index_col=0, keep_default_na=False)
    if full_path.name == "Setups_simple_rear_end.xlsx" and "EA_mode" not in full.columns:
        full["EA_mode"] = "Surprise"
    row = contrast.iloc[0].reindex(full.columns)
    # The released saver serializes Python ``None`` as an empty Excel cell,
    # while the released analysis recognizes the literal string below.
    if "EA_mode" in row.index:
        row["EA_mode"] = "None"
    row.name = index
    pd.concat((full, row.to_frame().T)).to_excel(full_path, float_format="%.12g")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-results", type=Path, required=True)
    parser.add_argument("--contrast-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.full_results / "Results_rear_end"
    contrast = args.contrast_results / "Results_rear_end"
    target = args.output_dir / "Results_rear_end"
    if args.output_dir.exists():
        raise RuntimeError(f"analysis context already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    shutil.copytree(source, target)
    existing_indices = [int(path.name.split("_")[-1]) for path in target.glob("Exp_*")]
    added_index = max(existing_indices) + 1
    shutil.copytree(contrast / "Exp_0", target / f"Exp_{added_index}")
    old_folder = target / f"Exp_{added_index}"
    new_folder = target / f"Exp_{added_index}"
    old_file = old_folder / "Exp_0.pkl"
    old_file.rename(new_folder / f"Exp_{added_index}.pkl")
    for filename in ("Setups_rear_end.xlsx", "Setups_simple_rear_end.xlsx"):
        contrast_setup = contrast / filename
        if not contrast_setup.exists():
            contrast_setup = contrast / "Setups_rear_end.xlsx"
        _append_setup(target / filename, contrast_setup, added_index)
    manifest = {
        "full_native_experiments": 28,
        "added_contrast_experiment": added_index,
        "contrast": "no_evidence Exp_0; metadata-only contrast needed by unmodified author analysis",
        "analysis_outputs_must_use": "the full-model row identified by EA_mode=Surprise",
    }
    (args.output_dir / "analysis_context.json").write_text(json.dumps(manifest, indent=2),
                                                           encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
