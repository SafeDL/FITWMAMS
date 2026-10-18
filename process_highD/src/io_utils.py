"""
io_utils.py — I/O 工具
=======================
YAML 配置加载、路径管理等通用 I/O 辅助函数。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """加载 YAML 配置文件并返回字典。

    Parameters
    ----------
    config_path : str
        YAML 文件的路径。

    Returns
    -------
    dict
        解析后的配置字典。
    """
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


def json_ready(value: Any) -> Any:
    """将 NumPy 数值与非有限浮点数转换为 JSON 可保存的值。"""
    if isinstance(value, dict):
        return {str(key): json_ready(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def resolve_data_path(raw_dir: str, config_path: str | Path) -> Path:
    """相对路径始终以配置文件所在目录为基准。"""
    raw = Path(raw_dir)
    if raw.is_absolute():
        return raw
    return (Path(config_path).resolve().parent / raw).resolve()


def resolve_recording_ids(
    raw_dir: str | Path,
    recordings_cfg: dict[str, Any],
) -> list[int]:
    """按唯一配置中的 include/exclude 选择录像。"""
    include = recordings_cfg["include"]
    exclude = {int(value) for value in recordings_cfg["exclude"]}
    raw_path = Path(raw_dir)
    if include == "all":
        ids = {
            int(p.stem.split("_")[0])
            for p in raw_path.glob("*_tracks.csv")
            if p.stem.split("_")[0].isdigit()
        }
    else:
        ids = {int(value) for value in include}

    return sorted(ids - exclude)
