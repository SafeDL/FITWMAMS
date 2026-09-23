"""Validate the reproduction catalog without importing heavy model runtimes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = Path(__file__).with_name("registry.json")


def load_registry(path: Path = REGISTRY) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _repo_path(value: object, label: str, issues: list[str]) -> Path | None:
    if not isinstance(value, str) or not value:
        issues.append(f"{label}: expected a non-empty repository-relative path")
        return None
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        issues.append(f"{label}: path must stay inside the repository: {value!r}")
        return None
    return ROOT / path


def audit_registry(registry: dict[str, Any] | None = None) -> list[str]:
    data = load_registry() if registry is None else registry
    issues: list[str] = []
    if data.get("schema_version") != 1:
        issues.append("schema_version: expected 1")

    seen_ids: set[str] = set()
    implementation_paths: set[str] = set()
    for group in ("reproductions", "evaluation_tools"):
        entries = data.get(group)
        if not isinstance(entries, list):
            issues.append(f"{group}: expected a list")
            continue
        for index, entry in enumerate(entries):
            label = f"{group}[{index}]"
            if not isinstance(entry, dict):
                issues.append(f"{label}: expected an object")
                continue
            item_id = entry.get("id")
            if not isinstance(item_id, str) or not item_id:
                issues.append(f"{label}.id: expected a non-empty string")
            elif item_id in seen_ids:
                issues.append(f"{label}.id: duplicate id {item_id!r}")
            else:
                seen_ids.add(item_id)

            implementation_value = entry.get("implementation_path")
            implementation = _repo_path(
                implementation_value, f"{label}.implementation_path", issues
            )
            catalog = _repo_path(entry.get("catalog_path"), f"{label}.catalog_path", issues)
            if isinstance(implementation_value, str):
                if implementation_value in implementation_paths:
                    issues.append(
                        f"{label}.implementation_path: duplicate {implementation_value!r}"
                    )
                implementation_paths.add(implementation_value)
            if implementation is not None and not implementation.is_dir():
                issues.append(f"{label}.implementation_path: missing directory")
            if implementation is not None:
                canonical_kind = "models" if group == "reproductions" else "evaluation"
                if implementation.is_symlink():
                    issues.append(f"{label}.implementation_path: source must be physical")
                try:
                    implementation.relative_to(ROOT / "reproduction" / canonical_kind)
                except ValueError:
                    issues.append(
                        f"{label}.implementation_path: source must be under "
                        f"reproduction/{canonical_kind}"
                    )
            if catalog is not None:
                if not catalog.is_dir():
                    issues.append(f"{label}.catalog_path: missing directory or link")
                elif implementation is not None and implementation.is_dir():
                    if catalog.resolve() != implementation.resolve():
                        issues.append(f"{label}.catalog_path: does not resolve to implementation")

            if group == "reproductions":
                paper = _repo_path(entry.get("paper"), f"{label}.paper", issues)
                if paper is not None and not paper.is_file():
                    issues.append(f"{label}.paper: missing paper")
                if not entry.get("status"):
                    issues.append(f"{label}.status: missing evidence status")

    for index, entry in enumerate(data.get("reference_only", [])):
        label = f"reference_only[{index}]"
        if not isinstance(entry, dict):
            issues.append(f"{label}: expected an object")
            continue
        path = _repo_path(entry.get("path"), f"{label}.path", issues)
        if path is not None and not path.exists():
            issues.append(f"{label}.path: missing source snapshot")

    classification = data.get("top_level_classification")
    classified: list[str] = []
    if not isinstance(classification, dict):
        issues.append("top_level_classification: expected an object")
    else:
        for role, names in classification.items():
            if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
                issues.append(f"top_level_classification.{role}: expected a string list")
                continue
            classified.extend(names)
        duplicates = sorted({name for name in classified if classified.count(name) > 1})
        if duplicates:
            issues.append(f"top_level_classification: multiply classified: {duplicates}")
        actual = {path.name for path in ROOT.iterdir() if path.is_dir() and not path.name.startswith(".")}
        missing = sorted(actual - set(classified))
        stale = sorted(set(classified) - actual)
        if missing:
            issues.append(f"top_level_classification: unclassified directories: {missing}")
        if stale:
            issues.append(f"top_level_classification: missing classified directories: {stale}")

    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    args = parser.parse_args()
    registry = load_registry()
    issues = audit_registry(registry)
    summary = {
        "ok": not issues,
        "reproduction_families": len(registry.get("reproductions", [])),
        "evaluation_tools": len(registry.get("evaluation_tools", [])),
        "issues": issues,
    }
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif issues:
        print("Reproduction catalog audit failed:")
        for issue in issues:
            print(f"- {issue}")
    else:
        print(
            "Reproduction catalog OK: "
            f"{summary['reproduction_families']} model families, "
            f"{summary['evaluation_tools']} evaluation tools."
        )
    return int(bool(issues))


if __name__ == "__main__":
    raise SystemExit(main())
