"""
io_utils.py — I/O 工具
=======================
YAML 配置加载、路径管理等通用 I/O 辅助函数。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict[str, Any]:
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
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger.info("已加载配置: %s", config_path)
    return cfg


def ensure_dir(path: str | Path) -> Path:
    """若目录不存在则创建并返回 Path 对象。"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_data_path(raw_dir: str, config_path: str | None = None) -> Path:
    """将 raw_dir 解析为绝对路径。

    如果 raw_dir 是相对路径，则基于 config_path 所在目录进行解析。
    """
    raw = Path(raw_dir)
    if raw.is_absolute():
        return raw
    if config_path is not None:
        base = Path(config_path).resolve().parent
    else:
        base = Path.cwd()
    return (base / raw).resolve()


def _parse_recording_id_set(value: Any) -> set[int] | str:
    """Parse recording IDs from YAML-friendly forms."""
    if value is None:
        return set()
    if isinstance(value, str):
        value = value.strip()
        if value.lower() == "all":
            return "all"
        if not value:
            return set()
        return {int(x.strip()) for x in value.split(",") if x.strip()}
    if isinstance(value, int):
        return {int(value)}
    return {int(x) for x in value}


def resolve_recording_ids(
    raw_dir: str | Path,
    recordings_cfg: dict[str, Any] | None = None,
) -> list[int]:
    """Resolve recording IDs from config only.

    `include` accepts "all", a single ID, a comma-separated string, or a YAML list.
    `exclude` accepts the same ID forms except "all".
    """
    recordings_cfg = recordings_cfg or {}
    include = _parse_recording_id_set(recordings_cfg.get("include", "all"))
    exclude = _parse_recording_id_set(recordings_cfg.get("exclude", []))
    if exclude == "all":
        raise ValueError("recordings.exclude cannot be 'all'")

    raw_path = Path(raw_dir)
    if include == "all":
        ids = {
            int(p.stem.split("_")[0])
            for p in raw_path.glob("*_tracks.csv")
            if p.stem.split("_")[0].isdigit()
        }
    else:
        ids = include

    return sorted(ids - exclude)
