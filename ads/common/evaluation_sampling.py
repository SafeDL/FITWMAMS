"""Seal three ADS-stage holdout cohorts after every policy is frozen."""

import hashlib

import numpy as np

from ads.common.instances import GeneratedSource, json_hash

FORMAL_NAMESPACE = "sealed_method_c_holdouts_v2"
COHORTS = ("tail_generated", "empirical_natural", "natural_generated")


def named_stream(*parts):
    integers = [
        int.from_bytes(hashlib.sha256(str(part).encode()).digest()[:4], "little")
        for part in parts
    ]
    return int(np.random.SeedSequence(integers).generate_state(1)[0])


def source_stream(config, protocol_hash, purpose, *parts):
    if purpose == "final_test":
        parts = (
            FORMAL_NAMESPACE,
            config.get("run_kind", "formal"),
            protocol_hash,
            *parts,
        )
    else:
        parts = (config.get("run_kind", "formal"), purpose, *parts)
    return named_stream(*parts)


def final_source(protocol, cohort="tail_generated", evaluation_assets=None):
    if cohort not in ("tail_generated", "natural_generated"):
        raise ValueError("empirical Test uses a frozen manifest, not a generator")
    config = protocol["settings"]
    assets = (
        protocol["assets"]
        if cohort == "tail_generated"
        else evaluation_assets["natural_world"]
    )
    return GeneratedSource(
        None,
        assets,
        master_seed=config["master_seed"],
        purpose="final_test",
        stream=source_stream(config, json_hash(protocol), "final_test", cohort),
        source_kind=cohort,
    )


def expected_final_roots(protocol, cohort="tail_generated", evaluation_assets=None):
    if cohort == "empirical_natural":
        return set(evaluation_assets["empirical_roots"])
    source = final_source(protocol, cohort, evaluation_assets)
    return {
        source.prior_identity(index)[1]
        for index in range(protocol["settings"]["evaluation"]["instances"])
    }


def final_sampling_spec(protocol, evaluation_assets=None):
    config = protocol["settings"]
    cohorts = COHORTS if evaluation_assets is not None else ("tail_generated",)
    return {
        "schema": "ads_three_holdouts_v2",
        "namespace": FORMAL_NAMESPACE,
        "master_seed": config["master_seed"],
        "purpose": "final_test",
        "source_independence": "ADS-stage instances; full highD tail training retained",
        "cohorts": {
            cohort: {
                "instances": config["evaluation"]["instances"],
                "root_ids_sha256": json_hash(
                    sorted(expected_final_roots(protocol, cohort, evaluation_assets))
                ),
            }
            for cohort in cohorts
        },
    }


def verify_sampling_spec(frozen, protocol):
    if frozen.get("evaluation_sampling") != final_sampling_spec(
        protocol, frozen.get("evaluation_assets")
    ):
        raise ValueError("final sampling stream differs from the sealed freeze")


def verify_final_roots(
    instances, protocol, cohort="tail_generated", evaluation_assets=None
):
    roots = [instance.root_sample_id for instance in instances]
    if len(roots) != len(set(roots)) or set(roots) != expected_final_roots(
        protocol, cohort, evaluation_assets
    ):
        raise ValueError("final cohort differs from the sealed sampling roots")
    if any(
        instance.source_kind != cohort or instance.purpose != "final_test"
        for instance in instances
    ):
        raise ValueError("final cohort contains a different source or purpose")
