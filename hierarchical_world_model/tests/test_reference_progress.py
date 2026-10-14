from __future__ import annotations

import torch

from hierarchical_world_model.src.reference import (
    gather_progress_preview,
    project_reference_progress,
    rebase_soft_preview,
)


def test_spatial_progress_holds_when_vehicle_stops_behind_plan_clock() -> None:
    reference = torch.zeros(1, 10, 6, 2)
    reference[0, :, :, 0] = torch.arange(1, 11)[:, None]
    starts = torch.zeros(1, 6, dtype=torch.long)
    current = torch.zeros(1, 6, 2)
    starts = project_reference_progress(current, reference, starts, nominal_index=0)
    assert not starts.any()
    current[..., 0] = 1.0
    starts = project_reference_progress(current, reference, starts, nominal_index=1)
    assert torch.equal(starts, torch.ones_like(starts))
    stalled = project_reference_progress(current, reference, starts, nominal_index=7)
    assert torch.equal(stalled, starts)


def test_progress_preview_is_actor_specific_and_padded_at_tail() -> None:
    reference = torch.zeros(1, 5, 6, 2)
    reference[0, :, :, 0] = torch.arange(1, 6)[:, None]
    starts = torch.tensor(((0, 1, 2, 3, 4, 4),))
    preview, base = gather_progress_preview(
        reference, torch.zeros(1, 6, 2), starts, preview_frames=3
    )
    assert preview.shape == (1, 3, 6, 2)
    assert preview[0, 0, 0, 0] == 1.0
    assert preview[0, 0, 3, 0] == 4.0
    assert torch.equal(preview[0, :, 5, 0], torch.full((3,), 5.0))
    assert base[0, 0, 0] == 0.0
    assert base[0, 3, 0] == 3.0


def test_vectorized_progress_matches_local_search_for_each_actor() -> None:
    torch.manual_seed(17)
    reference = torch.randn(3, 31, 6, 2).cumsum(dim=1)
    current = torch.randn(3, 6, 2)
    previous = torch.randint(0, 20, (3, 6))
    actual = project_reference_progress(
        current, reference, previous, nominal_index=14, local_window=8
    )
    expected = previous.clone()
    for batch in range(3):
        for actor in range(6):
            lower = max(int(previous[batch, actor]) - 1, 0)
            upper = min(31, max(14 + 8, lower + 1))
            distance = torch.linalg.vector_norm(
                reference[batch, lower:upper, actor] - current[batch, actor], dim=-1
            )
            nearest = lower + int(distance.argmin())
            expected[batch, actor] = max(int(previous[batch, actor]), min(nearest + 1, 30))
    torch.testing.assert_close(actual, expected)


def test_longitudinal_reference_offset_does_not_snap_back_to_old_timeline() -> None:
    current = torch.zeros(1, 6, 6)
    current[..., 0] = 5.0
    reference = torch.zeros(1, 10, 6, 2)
    base = torch.zeros(1, 6, 2)
    rebased = rebase_soft_preview(
        current, reference, base,
        longitudinal_endpoint_offset_weight=1.0,
        lateral_endpoint_offset_weight=0.0,
    )
    torch.testing.assert_close(rebased[:, -1, :, 0], torch.full((1, 6), 5.0))
