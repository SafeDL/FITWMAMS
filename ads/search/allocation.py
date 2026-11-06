"""Decompose existing search costs and collision novelty without new rollouts."""

import csv
import hashlib
import json
from collections import Counter, defaultdict

from ads.common.setup import REPO_ROOT as ROOT

JOURNAL = (
    ROOT
    / "results/ads/runs/search/search/diffusion_planner/20260926/guided/round_1/search.jsonl"
)
COMPARISON = ROOT / "results/ads/reports/final/search_comparison.json"


def run():
    rows = [json.loads(line) for line in JOURNAL.read_text().splitlines()]
    catalogs = json.loads(COMPARISON.read_text())["catalogs"]
    main = next(v for k, v in catalogs.items() if float(k) == 1)
    assignments = main["methods"]["guided"]["assignments"]
    groups = defaultdict(Counter)
    group_cells = defaultdict(set)
    seen_inputs, seen_traces, seen_cells = set(), set(), set()
    for row in rows:
        collision = bool(row["collision"])
        identity, trace = row["instance_hash"], row["trajectory_hash"]
        cell = assignments.get(identity)
        new_input = collision and identity not in seen_inputs
        new_trace = collision and trace not in seen_traces
        new_cell = collision and cell is not None and cell not in seen_cells
        labels = {
            "execution_kind": str(row["execution_kind"]),
            "proposal_group": str(row.get("proposal_block", "prior")),
            "risk_depth": str(row.get("path_depth", 0)),
            "ancestor": row["root_sample_id"],
        }
        for kind, label in labels.items():
            key = (kind, label)
            groups[key].update(
                {
                    "executions": 1,
                    "collision_executions": int(collision),
                    "new_collision_inputs": int(new_input),
                    "new_collision_traces": int(new_trace),
                    "first_collision_cells": int(new_cell),
                    "accepted_proposals": int(row.get("accepted", False)),
                    "unchanged_parent_execution": int(
                        row.get("unchanged_execution", False)
                    ),
                }
            )
            if collision and cell is not None:
                group_cells[key].add(cell)
        if collision:
            seen_inputs.add(identity)
            seen_traces.add(trace)
            if cell is not None:
                seen_cells.add(cell)
    assert len(rows) == 5000 and len(seen_inputs) == 798 and len(seen_traces) == 706
    assert len(seen_cells) == 175
    table = [
        {
            "grouping": kind,
            "group": name,
            **dict(counts),
            "occupied_collision_cells": len(group_cells[(kind, name)]),
        }
        for (kind, name), counts in sorted(groups.items())
    ]
    output = ROOT / "results/ads/reports/search_allocation_diagnosis.csv"
    with output.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    ancestors = sorted(
        (r for r in table if r["grouping"] == "ancestor"),
        key=lambda r: -r["executions"],
    )
    result = {
        "scope": "existing 5000-search execution journal only; no new search or repair",
        "journal_sha256": hashlib.sha256(JOURNAL.read_bytes()).hexdigest(),
        "comparison_sha256": hashlib.sha256(COMPARISON.read_bytes()).hexdigest(),
        "executions": len(rows),
        "collision_inputs": len(seen_inputs),
        "physical_collision_traces": len(seen_traces),
        "common_collision_cells": len(seen_cells),
        "by_execution_kind": [r for r in table if r["grouping"] == "execution_kind"],
        "by_proposal_group": [r for r in table if r["grouping"] == "proposal_group"],
        "by_risk_depth": [r for r in table if r["grouping"] == "risk_depth"],
        "top_ten_ancestors_by_execution_cost": ancestors[:10],
        "all_ancestor_execution_counts": str(output),
        "most_used_ancestor_execution_fraction": ancestors[0]["executions"] / len(rows),
        "top_ten_ancestor_execution_fraction": sum(
            r["executions"] for r in ancestors[:10]
        )
        / len(rows),
        "interpretation": "First-cell counts depend on chronological execution order; occupied cells can overlap groups. Exact trace novelty is separate from behavior-cell novelty.",
        "new_world_executions": 0,
        "new_optimizer_updates": 0,
    }
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ["top_ten_ancestors_by_execution_cost", "by_risk_depth"]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
