"""True-size oriented-box geometry for the common evaluator."""

from __future__ import annotations

import numpy as np


def heading_from_velocity(states: np.ndarray) -> np.ndarray:
    values = np.asarray(states)
    return np.arctan2(
        values[..., 3],
        np.where(np.abs(values[..., 2]) < 1.0e-6, 1.0e-6, values[..., 2]),
    )


def oriented_box_corners(
    centers: np.ndarray,
    headings: np.ndarray,
    lengths_m: np.ndarray,
    widths_m: np.ndarray,
) -> np.ndarray:
    """Return counter-clockwise corners with shape ``[...,4,2]``."""
    center = np.asarray(centers, np.float64)
    heading = np.asarray(headings, np.float64)
    length = np.asarray(lengths_m, np.float64)
    width = np.asarray(widths_m, np.float64)
    if center.shape[-1] != 2 or center.shape[:-1] != heading.shape:
        raise ValueError("centers and headings do not align")
    forward = np.stack((np.cos(heading), np.sin(heading)), axis=-1)
    left = np.stack((-np.sin(heading), np.cos(heading)), axis=-1)
    local = np.asarray(((1, 1), (-1, 1), (-1, -1), (1, -1)), np.float64)
    return (
        center[..., None, :]
        + local[:, :1] * length[..., None, None] * 0.5 * forward[..., None, :]
        + local[:, 1:] * width[..., None, None] * 0.5 * left[..., None, :]
    )


def _intervals(corners: np.ndarray, axes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    projection = np.einsum("...pd,...ad->...ap", corners, axes)
    return projection.min(axis=-1), projection.max(axis=-1)


def oriented_boxes_overlap(
    corners_a: np.ndarray,
    corners_b: np.ndarray,
    *,
    contact_tolerance_m: float = 1.0e-6,
) -> np.ndarray:
    """Separating-axis intersection; edge-only contact is not a collision."""
    a, b = np.asarray(corners_a, np.float64), np.asarray(corners_b, np.float64)
    if a.shape[-2:] != (4, 2) or b.shape[-2:] != (4, 2):
        raise ValueError("corners must end in [4,2]")
    edges = np.concatenate(
        (a[..., 1:3, :] - a[..., :2, :], b[..., 1:3, :] - b[..., :2, :]), axis=-2
    )
    axes = np.stack((-edges[..., 1], edges[..., 0]), axis=-1)
    axes /= np.maximum(np.linalg.norm(axes, axis=-1, keepdims=True), 1.0e-12)
    amin, amax = _intervals(a, axes)
    bmin, bmax = _intervals(b, axes)
    overlap = np.minimum(amax, bmax) - np.maximum(amin, bmin)
    return np.all(overlap > float(contact_tolerance_m), axis=-1)


def collision_matrix(
    states: np.ndarray,
    valid: np.ndarray,
    lengths_m: np.ndarray,
    widths_m: np.ndarray,
    *,
    contact_tolerance_m: float = 1.0e-6,
) -> np.ndarray:
    """Return symmetric pair collisions for arbitrary leading time axes."""
    state, present = np.asarray(states), np.asarray(valid, bool)
    n = state.shape[-2]
    if state.shape[-1] < 4 or present.shape != state.shape[:-1]:
        raise ValueError("states/valid must align and states need x,y,vx,vy")
    # Pairwise OBB SAT without materializing four corners and four axes for
    # every pair.  This matters for full K x 149-frame benchmark batches.
    length = np.broadcast_to(np.asarray(lengths_m, state.dtype), state.shape[:-1])
    width = np.broadcast_to(np.asarray(widths_m, state.dtype), state.shape[:-1])
    heading = heading_from_velocity(state).astype(state.dtype, copy=False)
    forward = np.stack((np.cos(heading), np.sin(heading)), axis=-1)
    left_axis = np.stack((-np.sin(heading), np.cos(heading)), axis=-1)
    delta = state[..., None, :, :2] - state[..., :, None, :2]
    fi, fj = forward[..., :, None, :], forward[..., None, :, :]
    li, lj = left_axis[..., :, None, :], left_axis[..., None, :, :]
    hli, hlj = length[..., :, None] * 0.5, length[..., None, :] * 0.5
    hwi, hwj = width[..., :, None] * 0.5, width[..., None, :] * 0.5

    def dot(a, b):
        return np.sum(a * b, axis=-1)

    ff, fl = np.abs(dot(fi, fj)), np.abs(dot(fi, lj))
    lf, ll = np.abs(dot(li, fj)), np.abs(dot(li, lj))
    on_fi = np.abs(dot(delta, fi)) < (hli + hlj * ff + hwj * fl - contact_tolerance_m)
    on_li = np.abs(dot(delta, li)) < (hwi + hlj * lf + hwj * ll - contact_tolerance_m)
    on_fj = np.abs(dot(delta, fj)) < (hlj + hli * ff + hwi * lf - contact_tolerance_m)
    on_lj = np.abs(dot(delta, lj)) < (hwj + hli * fl + hwi * ll - contact_tolerance_m)
    overlap = on_fi & on_li & on_fj & on_lj
    pair_valid = present[..., :, None] & present[..., None, :]
    overlap &= pair_valid
    diagonal = np.eye(n, dtype=bool).reshape((1,) * (overlap.ndim - 2) + (n, n))
    return overlap & ~diagonal


def straight_road_offroad(
    states: np.ndarray,
    valid: np.ndarray,
    lengths_m: np.ndarray,
    widths_m: np.ndarray,
    map_polylines: np.ndarray,
    map_valid: np.ndarray,
    *,
    margin_m: float = 0.2,
) -> np.ndarray:
    """HighD straight-road corner test using lane centers and recorded widths."""
    state, present = np.asarray(states), np.asarray(valid, bool)
    lines, line_valid = np.asarray(map_polylines), np.asarray(map_valid, bool)
    lane_mask = line_valid.any(-1)
    if not lane_mask.any():
        return np.zeros_like(present)
    centers = np.divide(
        (lines[..., 1] * line_valid).sum(-1), line_valid.sum(-1).clip(1)
    )[lane_mask]
    lane_widths = np.divide(
        (lines[..., 4] * line_valid).sum(-1), line_valid.sum(-1).clip(1)
    )[lane_mask]
    lower = float(np.min(centers - lane_widths * 0.5)) - float(margin_m)
    upper = float(np.max(centers + lane_widths * 0.5)) + float(margin_m)
    length = np.broadcast_to(np.asarray(lengths_m), state.shape[:-1])
    width = np.broadcast_to(np.asarray(widths_m), state.shape[:-1])
    corners = oriented_box_corners(
        state[..., :2], heading_from_velocity(state), length, width
    )
    outside = (corners[..., 1] < lower).any(-1) | (corners[..., 1] > upper).any(-1)
    return outside & present
