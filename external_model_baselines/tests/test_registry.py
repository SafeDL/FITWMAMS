from external_model_baselines.audit import audit_registry, load_registry


def test_registry_and_catalog_links_are_consistent() -> None:
    assert audit_registry() == []


def test_only_executable_paper_adaptations_are_models() -> None:
    registry = load_registry()
    model_ids = {entry["id"] for entry in registry["model_baselines"]}
    tool_ids = {entry["id"] for entry in registry["evaluation_tools"]}

    assert model_ids == {
        "active_inference_driver",
        "bayesian_ma_idm",
        "dynamic_ar_idm",
        "multi_regime_bidm",
        "trafficbots",
    }
    assert tool_ids == {
        "counterfactual_response",
        "driver_reproduction",
    }
    assert model_ids.isdisjoint(tool_ids)
