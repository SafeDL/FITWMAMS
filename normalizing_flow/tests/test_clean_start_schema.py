from __future__ import annotations

import numpy as np

from normalizing_flow.src.features import (
    EGO_FEATURES,
    SLOT_FEATURES,
    SLOT_NAMES,
    build_feature_schema,
    feature_valid_from_slot_mask,
    zero_inactive_slot_features,
)


def test_clean_start_schema_excludes_all_future_action_features() -> None:
    schema = build_feature_schema("clean_start")
    assert schema.feature_mode == "clean_start"
    assert schema.trajectory_features == ()
    assert schema.num_features == len(EGO_FEATURES) + len(SLOT_NAMES) * len(SLOT_FEATURES)
    assert not any("_1s_" in name for name in schema.feature_names)

    slot_mask = np.asarray([[True, False, True, False, False, False]], dtype=bool)
    valid = feature_valid_from_slot_mask({"feature_names": schema.feature_names}, slot_mask)
    raw = np.ones((1, schema.num_features), dtype=np.float32)
    masked = zero_inactive_slot_features(raw, slot_mask)
    assert valid.shape == raw.shape
    assert int(valid.sum()) == len(EGO_FEATURES) + 2 * len(SLOT_FEATURES)
    assert np.all(masked[:, len(EGO_FEATURES) + len(SLOT_FEATURES):2 * len(SLOT_FEATURES) + len(EGO_FEATURES)] == 0.0)


def test_legacy_schema_still_has_future_action_summary_features() -> None:
    schema = build_feature_schema()
    assert schema.feature_mode == "legacy_future_action_summary"
    assert schema.trajectory_features
    assert any("_1s_" in name for name in schema.feature_names)
