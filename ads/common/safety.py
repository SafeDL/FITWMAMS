"""Initial defects are measured before any ADS action and remain in the cohort."""

import torch

from hierarchical_world_model.src.scene_geometry import paired_rectangle_separation


def initial_safety(event):
    states, dims = (
        torch.as_tensor(event.initial_states),
        torch.as_tensor(event.vehicle_dimensions),
    )
    heading = torch.atan2(states[:, 3], states[:, 2])
    collision = npc_collision = False
    for i in range(7):
        for j in range(i + 1, 7):
            if event.valid[i] and event.valid[j]:
                overlap = (
                    float(
                        paired_rectangle_separation(
                            states[i, :2],
                            heading[i],
                            dims[i],
                            states[j, :2],
                            heading[j],
                            dims[j],
                        )
                    )
                    <= 0
                )
                collision |= bool(overlap and i == 0)
                npc_collision |= bool(overlap and i != 0)
    offroad = bool(
        event.geometry.offroad(
            event.initial_states[:1, :2],
            vehicle_width_m=event.vehicle_dimensions[:1, 1],
        )[0]
    )
    return collision, offroad, npc_collision
