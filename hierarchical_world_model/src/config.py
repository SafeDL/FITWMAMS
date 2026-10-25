"""Physical execution contract for the traffic world."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .core.utils import load_yaml


@dataclass(frozen=True)
class WorldModelConfig:
    history_frames: int = 25
    execute_frames: int = 1
    dt_s: float = 0.04
    min_acceleration_mps2: float = -8.0
    max_acceleration_mps2: float = 4.0
    max_yaw_rate_rps: float = 0.6

    def to_dict(self):
        return asdict(self)


def load_protocol_config(path: str | Path) -> dict[str, Any]:
    config = load_yaml(path)
    root = Path(__file__).resolve().parents[2]
    config["paths"] = {
        name: str((root / value).resolve()) for name, value in config["paths"].items()
    }
    return config
