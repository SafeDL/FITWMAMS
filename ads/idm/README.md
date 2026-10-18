# IDM 自车评测

使用 HighwayEnv 的原生 IDM 自车测试交通世界模型。代码提供 Monte Carlo、子集模拟、
固定 Test 场景评测及结果回放；自车不由 NPC 策略接管。

| 位置 | 内容 |
| --- | --- |
| `src/idm_policy.py` | IDM 自车参数接口；参数来自 [`tools/idm_ego.yaml`](../../tools/idm_ego.yaml)。 |
| `src/world_subset_runner.py` | 本项目世界模型的采样和评测。 |
| `src/trafficbots_*.py` | TrafficBots 对照模型适配。 |
| `src/multi_world_runner.py` | 多模型运行、协议核对与结果汇总。 |
| `scripts/` | 运行、对比、回放和可复现性检查入口。 |

两个测试空间分别由以下配置指定：

- `configs/world_subset_idm.yaml`：EVT 尾部分布 → Flow → Diffusion → 跟踪与 NPC 接管，
  生成完整场景；不读取真实未来 `K_GT`。
- `configs/world_factual_k_gt.yaml`：固定 Test 的真实条件与 `K_GT`，分析事实条件下的闭环表现。
  自车由 IDM 驾驶，因此该运行不等同于日志自车的事实回放。

两者通过 [`hierarchical_world_model/config.yaml`](../../hierarchical_world_model/config.yaml)
加载当前组件。TrafficBots 无 `K` 输入，默认未启用，不与固定 `K_GT` 结果混作同一测试空间。

从仓库根目录运行：

```bash
conda activate tread
python -m ads.idm.scripts.run_world_models_idm
python -m ads.idm.scripts.run_empirical_test_sweep
```

前者默认运行配置中启用的模型及两种估计器；后者遍历固定 Test 条件。
完成运行后，可用 `compare_world_models_idm` 汇总、`render_world_models_idm` 生成 GIF，
`audit_subset_replays` 检查保存案例能否精确重放。
输出按配置写入 `results/idm_ams/`，目录名沿用既有结果协议。

估计的概率以 EVT 尾部场景分布为总体，不能解释为现实道路事故率。
更换世界模型、桥接或 NPC 后需要重算；当前新版完整主链的 Monte Carlo 和子集模拟尚未完成。
桥接与 Test 验证见[世界模型说明](../../hierarchical_world_model/README.md)。
