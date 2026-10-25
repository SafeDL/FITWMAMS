"""Decision-indexed policy randomness, independent of the searchable world prior."""

import hashlib
import random
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class PolicyRandomContext:
    algorithm: str
    seed: int = 20260926
    mode: str = "search"
    instance_hash: str = ""

    def seed_for(self, decision, use="native_sampling"):
        instance = (
            self.instance_hash
            if self.mode in ("development", "final_test", "training")
            else "common"
        )
        material = f"ads_policy_v1:{self.seed}:{self.algorithm}:{self.mode}:{instance}:{decision}:{use}"
        return int.from_bytes(
            hashlib.sha256(material.encode()).digest()[:8], "little"
        ) % (2**63 - 1)

    @property
    def identity(self):
        return hashlib.sha256(
            str(self.seed_for(0, "stream_identity")).encode()
        ).hexdigest()

    @contextmanager
    def step(self, decision, device="cpu", use="native_sampling"):
        device = torch.device(device)
        devices = (
            [device.index if device.index is not None else torch.cuda.current_device()]
            if device.type == "cuda"
            else []
        )
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        with torch.random.fork_rng(devices=devices):
            seed = self.seed_for(decision, use)
            torch.random.default_generator.manual_seed(seed)
            for index in devices:
                torch.cuda.default_generators[index].manual_seed(seed)
            np.random.seed(seed % (2**32))
            random.seed(seed)
            try:
                yield
            finally:
                np.random.set_state(numpy_state)
                random.setstate(python_state)
