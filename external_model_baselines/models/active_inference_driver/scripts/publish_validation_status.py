"""Publish one explicit status boundary across native and adapter evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-audit", type=Path, required=True)
    parser.add_argument("--bridge-audit", type=Path, required=True)
    parser.add_argument("--official-wrapper-audit", type=Path)
    parser.add_argument("--official-bridge-audit", type=Path)
    parser.add_argument("--highd-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    native, bridge, highd = _read(args.native_audit), _read(args.bridge_audit), _read(args.highd_audit)
    status: dict[str, object] = {
        "native_paper_human_metric": {
            "status": "passed" if native["pre_registered_engineering_gate"]["passes"] else "failed",
            "audit": str(args.native_audit),
            "method": native["analysis"],
        },
        "longitudinal_adapter_25hz_bridge": {
            "status": "passed" if bridge["passes_all_one_tick"] else "failed",
            "audit": str(args.bridge_audit),
            "reason": "independent longitudinal-only adapter does not meet all native timing comparisons",
        },
        "highd_external_domain": {
            "status": "evaluated",
            "audit": str(args.highd_audit),
            "event_coverage": highd["event_coverage"],
        },
        "claim_boundary": "Native human reproduction, the independent longitudinal adapter, and the external official-POMDP wrapper are separate claims. highD is an external-domain evaluation, not the paper's human source-data replication.",
    }
    if args.official_wrapper_audit and args.official_bridge_audit:
        official_step, official_bridge = _read(args.official_wrapper_audit), _read(args.official_bridge_audit)
        status["official_pomdp_external_25hz_wrapper"] = {
            "status": "passed" if official_step["passes_native_step_equivalence"]
            and official_bridge["passes_all_conditions_one_native_tick"] else "failed",
            "step_equivalence_audit": str(args.official_wrapper_audit),
            "bridge_audit": str(args.official_bridge_audit),
            "reason": "Official POMDP is imported from the pinned external checkout; actions are held for five 25 Hz ticks.",
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("human_metric_reproduction.json", "validation_status.json"):
        (args.output_dir / name).write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
