"""Persist a failing episode and tensor metadata without changing its execution."""

import json
import time
from pathlib import Path

import numpy as np
import torch


def capture_failure(error, output):
    directory = Path(output) / "failures" / str(time.time_ns())
    directory.mkdir(parents=True)
    frames, values = [], {}
    environment, action = None, None
    tb = error.__traceback__
    while tb is not None:
        frame = tb.tb_frame
        local = frame.f_locals
        frames.append(
            {
                "file": frame.f_code.co_filename,
                "line": tb.tb_lineno,
                "function": frame.f_code.co_name,
                "bytecode_offset": tb.tb_lasti,
            }
        )
        owner = local.get("self")
        if owner is not None and type(owner).__name__ in (
            "PolicyEnvironment",
            "RepairEnvironment",
            "TeacherEnvironment",
        ):
            environment = owner
        if frame.f_code.co_name == "step" and "ego_action" in local:
            action = np.asarray(local["ego_action"])
        tb = tb.tb_next
    if environment is not None and environment.event is not None:
        instance = next(
            i
            for i in environment.instances
            if i.event.event_id == environment.event.event_id
        )
        instance.save(directory / "instance.npz")
        actions = list(environment.physics_controls)
        if action is not None:
            actions.append(action)
        np.save(
            directory / "actions.npy", np.asarray(actions, np.float32).reshape(-1, 2)
        )
        values["reference_index"] = int(environment.world.reference_index)
        values["instance_hash"] = instance.instance_hash
        controller = environment.world.controller
        state = {
            k: v.detach().cpu().numpy()
            for k, v in controller.snapshot_runtime().items()
            if torch.is_tensor(v)
        }
        np.savez_compressed(directory / "controller.npz", **state)
    (directory / "failure.json").write_text(
        json.dumps({"error": str(error), "frames": frames, **values}, indent=2)
    )
    return str(directory)
