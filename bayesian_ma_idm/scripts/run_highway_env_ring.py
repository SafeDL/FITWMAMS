#!/usr/bin/env python3
"""Run the paper's ring-road MA-IDM experiment through local highway-env.

This is a faithful *organisation* of Zhang & Sun Fig. 10, not a claim that
their 5 Hz, 20-driver posterior and our native-25-Hz full-highD posterior are
numerically identical.  The road, vehicle integration and collision handling
are highway-env's; the longitudinal driver is :class:`MAIDMVehicle`.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


def _imports():
    from highway_env.road.lane import CircularLane
    from highway_env.road.road import Road, RoadNetwork
    from bayesian_ma_idm.src.highway_env_driver import MAIDMVehicle
    return CircularLane, Road, RoadNetwork, MAIDMVehicle


def make_ring(radius: float, seed: int):
    CircularLane, Road, RoadNetwork, _ = _imports()

    class PeriodicCircularLane(CircularLane):
        """CircularLane whose longitudinal coordinate is always [0, length)."""
        def local_coordinates(self, position: np.ndarray) -> tuple[float, float]:
            delta = np.asarray(position) - self.center
            phi = np.arctan2(delta[1], delta[0])
            phase = (self.direction * (phi - self.start_phase)) % (2 * np.pi)
            longitudinal = phase * self.radius
            lateral = self.direction * (self.radius - np.linalg.norm(delta))
            return float(longitudinal), float(lateral)

    class CyclicRoad(Road):
        """highway-env Road with wrap-around leader selection on its one ring lane."""
        def neighbour_vehicles(self, vehicle, lane_index=None):
            lane_index = lane_index or vehicle.lane_index
            lane = self.network.get_lane(lane_index)
            own, _ = lane.local_coordinates(vehicle.position)
            front = rear = None; front_gap = rear_gap = np.inf
            for other in self.vehicles:
                if other is vehicle:
                    continue
                longitudinal, lateral = lane.local_coordinates(other.position)
                if abs(lateral) > lane.width / 2 + 1:
                    continue
                forward = (longitudinal - own) % lane.length
                backward = (own - longitudinal) % lane.length
                if 1.e-8 < forward < front_gap:
                    front, front_gap = other, forward
                if 1.e-8 < backward < rear_gap:
                    rear, rear_gap = other, backward
            return front, rear

    network = RoadNetwork()
    # highway-env's CircularLane length is positive when clockwise=True with
    # increasing phases; the direction itself is immaterial to this symmetric
    # single-lane experiment.
    network.add_lane("ring", "ring", PeriodicCircularLane(np.zeros(2), radius, 0., 2 * np.pi,
                                                              clockwise=True, width=4., speed_limit=40.))
    return CyclicRoad(network=network, np_random=np.random.default_rng(seed), record_history=False), ("ring", "ring", 0)


def apply_protocol(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve the documented paper-text or public-source ring protocol.

    The source repository's ``config.py`` and ``simulation_ring.py`` disagree
    with the published Fig. 10 text.  A source-code run must select all of its
    coupled choices together instead of exposing a misleading hybrid.
    """
    if args.protocol == "source_code":
        args.vehicles, args.radius = 32, 137.
        args.fixed_profile = "donor_ring"
        args.update_mode, args.idm_semantics = "source_sequential", "donor_ring"
    return args


def source_fixed_iid_sigma(trace_path: str | Path) -> float:
    """Return the exact noise quantity used by the public fixed-IDM branch.

    ``simulation_ring.py`` samples ``tr.posterior.s_a.mean(...)``.  It does
    *not* use ``Config.sim_eps_sigma`` despite that value being present in the
    repository configuration.  The original trace pickle was not released, so
    source-code protocol uses the identically specified B-model NUTS rerun.
    """
    import arviz as az

    trace = az.from_netcdf(trace_path)
    if "s_a" not in trace.posterior:
        raise ValueError(f"B-model trace has no s_a: {trace_path}")
    return float(np.asarray(trace.posterior["s_a"], dtype=float).mean())


def run(args: argparse.Namespace) -> Path:
    args = apply_protocol(args)
    _, _, _, MAIDMVehicle = _imports()
    fixed_noise_source = None
    if args.mode == "fixed_idm":
        alpha = 1.5 if args.fixed_profile == "paper_recommended" else .73
        fixed_sigma = args.fixed_iid_sigma
        if args.protocol == "source_code":
            fixed_sigma = source_fixed_iid_sigma(args.source_b_trace)
            fixed_noise_source = "mean s_a from identically specified local B-model NUTS rerun"
        else:
            fixed_noise_source = "command-line fixed_iid_sigma"
        MAIDMVehicle.configure_fixed_iid(np.asarray([33.3, 2., 1.6, alpha, 1.67]), fixed_sigma)
    else:
        MAIDMVehicle.configure_posterior(args.posterior)
        MAIDMVehicle.configure_source_code_gp_protocol(args.protocol == "source_code")
    MAIDMVehicle.configure_idm_semantics(args.idm_semantics)
    road, lane_index = make_ring(args.radius, args.seed)
    lane = road.network.get_lane(lane_index)
    for index in range(args.vehicles):
        longitudinal = lane.length * index / args.vehicles
        vehicle = MAIDMVehicle(road, lane.position(longitudinal, 0.), heading=lane.heading_at(longitudinal),
                               speed=args.initial_speed, target_lane_index=lane_index, enable_lane_change=False)
        road.vehicles.append(vehicle)
    dt, frames = 1 / 25., int(round(args.seconds * 25))
    record_stride = 5
    records = {key: [] for key in ("longitudinal", "speed", "acceleration", "parameters")}
    for frame in range(frames):
        if args.update_mode == "synchronous":
            road.act(); road.step(dt)
        else:
            # The public ring script updates vehicle 0, then 1, ..., so the
            # final vehicle observes vehicle 0's just-updated state.  Preserve
            # that ordering while retaining highway-env vehicle/road/collision
            # objects.  This is a donor-code comparison mode, not the default
            # multi-agent scheduling semantics.
            for vehicle in road.vehicles:
                vehicle.act(); vehicle.step(dt)
            for index, vehicle in enumerate(road.vehicles):
                for other in road.vehicles[index + 1:]:
                    vehicle.handle_collisions(other, dt)
        if frame % record_stride == 0:
            records["longitudinal"].append([lane.local_coordinates(vehicle.position)[0] for vehicle in road.vehicles])
            records["speed"].append([vehicle.speed for vehicle in road.vehicles])
            records["acceleration"].append([vehicle.action["acceleration"] for vehicle in road.vehicles])
    values = {key: np.asarray(value, dtype=float) for key, value in records.items() if key != "parameters"}
    values["parameters"] = np.asarray([vehicle.ma_parameters for vehicle in road.vehicles])
    values.update({"dt_s": np.asarray(dt), "record_dt_s": np.asarray(record_stride * dt), "radius_m": np.asarray(args.radius)})
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    archive = output / "highway_env_ma_idm_ring.npz"; np.savez_compressed(archive, **values)
    time = np.arange(len(values["speed"])) * record_stride * dt
    # Paper Fig. 10 uses periodic road coordinates and colours trajectory
    # segments by speed.  Do not unwrap: the apparent diagonal discontinuities
    # at x=0 are part of the circular-road time--space representation.
    fig, axis = plt.subplots(figsize=(10, 3.8))
    norm = plt.Normalize(0., 18.)
    last = None
    for vehicle in range(args.vehicles):
        points = np.column_stack((time, values["longitudinal"][:, vehicle])).reshape(-1, 1, 2)
        segments = np.concatenate((points[:-1], points[1:]), axis=1)
        # Never join a vehicle's exit at the periodic seam to its re-entry.
        seam = np.abs(np.diff(values["longitudinal"][:, vehicle])) > lane.length / 2.
        segments = segments[~seam]
        speed = values["speed"][:-1, vehicle][~seam]
        last = LineCollection(segments, cmap="jet_r", norm=norm, linewidth=0.72)
        last.set_array(speed)
        axis.add_collection(last)
    panel = "a" if args.mode == "fixed_idm" else "b"
    label = "fixed IDM + iid noise" if args.mode == "fixed_idm" else "MA-IDM"
    axis.set(xlim=(0., args.seconds), ylim=(0., lane.length), xlabel="time [s]", ylabel="space [m]",
             title=f"highway-env {label} ring road (Fig. 10({panel}) analogue)")
    colorbar = fig.colorbar(last, ax=axis, pad=.02)
    colorbar.set_label("speed [m/s]")
    fig.tight_layout()
    fig.savefig(output / "fig10_highway_env_ring.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    direct_draws = MAIDMVehicle.posterior is not None and "driver_parameter_draws" in MAIDMVehicle.posterior
    if args.mode == "fixed_idm":
        model_name = ("fixed Table-I recommended IDM + iid action noise"
                      if args.fixed_profile == "paper_recommended"
                      else "fixed public-ring-script IDM (alpha=.73) + iid action noise")
        parameter_sampling = ("fixed Table-I recommended parameters"
                              if args.fixed_profile == "paper_recommended"
                              else "fixed public ring-script parameters (alpha=.73)")
    else:
        model_name = "MA-IDM hierarchical population draw"
        parameter_sampling = ("direct retained PyMC draw x fitted driver" if direct_draws
                              else "population log-normal approximation")
    report = {"engine": "local highway-env", "protocol": args.protocol, "model": model_name, "native_fps": 25,
              "driver_action_fps": 5, "vehicles": args.vehicles, "radius_m": args.radius, "seconds": args.seconds,
              "initial_speed_mps": args.initial_speed, "driver_mode": args.mode,
              "fixed_profile": args.fixed_profile if args.mode == "fixed_idm" else None,
              "fixed_iid_sigma_mps2": (float(fixed_sigma) if args.mode == "fixed_idm" else None),
              "fixed_iid_noise_source": fixed_noise_source,
              "posterior": None if args.mode == "fixed_idm" else str(args.posterior),
              "lane_changes": "disabled: paper experiment is single-lane car following", "crashed_vehicles": int(sum(vehicle.crashed for vehicle in road.vehicles)),
              "speed_min_mps": float(np.min(values["speed"])), "speed_max_mps": float(np.max(values["speed"])),
              "stopped_share_speed_le_0_1_mps": float(np.mean(values["speed"] <= .1)),
              "parameter_sampling": parameter_sampling,
              "idm_semantics": args.idm_semantics,
              "update_mode": args.update_mode,
              "plant_integration": MAIDMVehicle.PLANT_INTEGRATION,
              "ma_gp_history": (None if args.mode == "fixed_idm" else
                                {"kind": "causal finite-memory SE-GP", "memory_s": MAIDMVehicle.GP_MEMORY_SECONDS,
                                 "audit": "finite-memory GP is covered by the retained kernel tests",
                                 "hyperparameters": ("posterior global mean + 2ell zero context (source_code protocol)"
                                                     if MAIDMVehicle.source_code_gp_protocol else "joint per-vehicle posterior draw")}),
              "paper_analogue": f"Fig. 10({panel}) stochastic ring simulation; posterior/data differ from original and are not table-level reproduction",
              "source_code_note": ("source commit 7520b15 config.py: 32 vehicles, radius 137 m; simulation_ring.py: alpha=.73, sequential update"
                                   "; fixed-IDM action noise is posterior s_a, not Config.sim_eps_sigma"
                                   if args.protocol == "source_code" else None)}
    (output / "highway_env_ring_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return archive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--posterior", default=str(ROOT / "bayesian_ma_idm/evidence/paper_reference/nuts/author_reference_ma_idm_posterior.npz"))
    parser.add_argument("--source-b-trace", default=str(ROOT / "bayesian_ma_idm/evidence/paper_reference/b_nuts/author_reference_b_trace.nc"),
                        help="B-model posterior used by public simulation_ring.py's fixed-IDM s_a noise path")
    parser.add_argument("--output-dir", default=str(ROOT / "bayesian_ma_idm/evidence/paper_reference/highway_env"))
    parser.add_argument("--seconds", type=float, default=3000.)
    parser.add_argument("--vehicles", type=int, default=37)
    parser.add_argument("--radius", type=float, default=128.)
    parser.add_argument("--initial-speed", type=float, default=11.6)
    parser.add_argument("--seed", type=int, default=1116)
    parser.add_argument("--mode", choices=("ma_idm", "fixed_idm"), default="ma_idm")
    parser.add_argument("--protocol", choices=("custom", "source_code"), default="custom",
                        help="custom/paper-text choices, or coupled public simulation_ring.py config choices")
    parser.add_argument("--fixed-iid-sigma", type=float, default=.333,
                        help="custom protocol IID action noise; source_code protocol instead uses B posterior s_a")
    parser.add_argument("--fixed-profile", choices=("paper_recommended", "donor_ring"), default="paper_recommended",
                        help="Table-I alpha=1.5, or the public ring-script's alpha=.73")
    parser.add_argument("--update-mode", choices=("synchronous", "source_sequential"), default="synchronous",
                        help="highway-env simultaneous actions, or public ring-script vehicle-order update")
    parser.add_argument("--idm-semantics", choices=("equation", "donor_ring"), default="donor_ring",
                        help="paper Eq. (2), or the author's ring-script max(dynamic gap, 0) variant")
    print(run(parser.parse_args()))


if __name__ == "__main__":
    main()
