# HighwayEnv 仿真依赖

本目录保留 [Farama HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv) 的环境源码，沿用 [LICENSE](LICENSE)。本工程使用其道路、车辆动力学、碰撞检测和渲染功能。

| 目录 | 内容 |
| --- | --- |
| `highway_env/` | 道路、车辆、环境、动作与观测实现。 |
| `tests/` | 上游环境测试。 |
| `docs/`、`scripts/` | 上游说明和使用示例。 |

本工程的交通演化入口在 [hierarchical_world_model](../hierarchical_world_model/README.md)，学习型策略的 5 Hz 决策与 25 Hz 执行接口在 [ads](../ads/README.md)。论文运行使用这些接口组合道路和车辆，不直接采用通用 `highway-v0` 的默认任务配置。

使用现有 `tread` 环境。上游 API 和环境示例见[官方文档](https://highway-env.farama.org/)，依赖声明见 `pyproject.toml`；通用环境示例不作为本文的实验协议。
