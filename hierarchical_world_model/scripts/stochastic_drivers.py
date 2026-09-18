#!/usr/bin/env python3
"""Inspect or execute retained stochastic drivers through the CIH-WM boundary."""

from __future__ import annotations

import argparse
import json

from hierarchical_world_model.src.stochastic_drivers import (
    LongitudinalObservation,
    create_driver_session,
    verify_driver_assets,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inventory")
    rollout = sub.add_parser("rollout")
    rollout.add_argument("--model", required=True)
    rollout.add_argument("--frames", type=int, default=150)
    rollout.add_argument("--seed", type=int, default=0)
    rollout.add_argument("--style-id", type=int, default=1)
    rollout.add_argument("--allow-unaccepted", action="store_true")
    rollout.add_argument("--official-source-dir")
    rollout.add_argument("--official-following-dir")
    rollout.add_argument("--official-device", default="cpu")
    args = parser.parse_args()
    if args.command == "inventory":
        print(json.dumps(verify_driver_assets(), indent=2))
        return

    session = create_driver_session(
        args.model, seed=args.seed, style_id=args.style_id,
        allow_unaccepted=args.allow_unaccepted,
        official_source_dir=args.official_source_dir,
        official_following_dir=args.official_following_dir,
        official_device=args.official_device,
    )
    dt = 0.04
    leader_x, leader_v = 30.0, 18.0
    follower_x, follower_v = 0.0, 18.0
    initial = LongitudinalObservation(leader_x - follower_x - 4.8, follower_v, leader_v)
    session.reset(initial, seed=args.seed)
    decisions: list[dict[str, float | int]] = []
    for _ in range(args.frames):
        observation = LongitudinalObservation(
            leader_x - follower_x - 4.8, follower_v, leader_v
        )
        command = session.step(observation)
        if command.updated:
            decisions.append({
                "frame": command.native_frame,
                "requested_acceleration_mps2": command.requested_acceleration_mps2,
                "applied_acceleration_mps2": command.acceleration_mps2,
            })
        follower_x += follower_v * dt + 0.5 * command.acceleration_mps2 * dt**2
        follower_v = max(0.0, follower_v + command.acceleration_mps2 * dt)
        leader_x += leader_v * dt
    print(json.dumps({
        "model": args.model,
        "native_frames": args.frames,
        "decisions": decisions,
        "terminal_gap_m": leader_x - follower_x - 4.8,
        "terminal_follower_speed_mps": follower_v,
    }, indent=2))


if __name__ == "__main__":
    main()
