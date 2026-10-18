# 交通世界模型结果

当前结果按用途组织，不再按桥接开发版本堆叠目录。主线为 Flow／Diffusion 计划、普通跟踪、计划适用性一次交权及共享 NPC；具体定义见[方法与入口](../../hierarchical_world_model/README.md)。

| 位置 | 内容 |
| --- | --- |
| `model_assets/` | 当前 IDM 人群、MOBIL／纵向修正、换道模型、经验误差库和发布选择；本目录的执行参数统一从这里读取。 |
| `diffusion_plans/` | 冻结 Diffusion 的 Train／Val／Test 计划缓存，保留物理场景身份及顺序。 |
| `bridge/` | 当前桥接的标定、自然 Test、干预、耗时及诊断报告。 |
| `bridge/playbacks/` | 7 个配对 GIF 和示例清单；只输出 GIF。 |
| `npc_response/` | NPC 人群与自主交通分项证据；该批结果早于当前纵向切入修正，不能作为最终组合策略的最新验证。 |

## 当前方法证据

| 问题 | 原始报告与范围 |
| --- | --- |
| 经验轨迹是否充分利用数据 | [误差库标定](bridge/calibration.json)：42 段 Train 原录像、26,687,683 个物理历史—未来窗口；跟踪误差覆盖 72,771 个 Train 片段。 |
| NPC 输入修正是否拟合完成 | [纵向标定](bridge/npc_calibration.json)：4,837,720 个 Train 状态拟合、919,154 个 Val 状态选择；IDM 人群及 MOBIL 主体冻结。 |
| 门限如何确定 | [自然数据选择](bridge/selection.json)：完整 Train／Val；后续跟驰候选收紧沿用原门限，不按 ADS 事故数调节。 |
| 自然回放是否基本保持 | [Test 报告](bridge/test_report.json)和[接管记录](bridge/test_admissions.json)：代表全部 10,151 个场景，三组随机实现均有 7 个场景接管。当前结果由全量运行及后续 98 场景回归合成，其余 10,053 场景因候选只减少而复用。 |
| 外部控制改变时是否响应 | [干预报告](bridge/interventions.json)、[选样协议](bridge/intervention_cases.json)：51 个预选案例、204 个回合；脚本改变主车实际控制，桥接看不到未来脚本。 |
| 是否可实时执行 | [CPU 耗时](bridge/latency.json)：七车世界步 P99 8.59 ms，8,493 步无 40 ms 超时，不含 ADS 计算。 |
| 跟踪器为何这样选择 | [跟踪标定与事实重建](bridge/tracking.json)，保留完整候选比较。 |

自然 NPC ADE 是执行位置与同物理时刻日志位置的平均欧氏距离，不含主车；无 NPC 的场景不按零误差加入均值。接管后的运动可以合理偏离日志，干预和生成场景的偏差不当作事实真值误差。接管理由、NPC 响应与 ADS 失效分别评价，事故不是桥接优化目标。

[诊断记录](bridge/diagnostics.json)合并保留原 12 条事故的配对记录、既有生成场景行为复核、历史未通过的 95% 时机门槛及跟驰规则修订依据。这里是开发证据，不能当作最新版本的完整生成场景 ADS 矩阵。自然 Test 已参与诊断，也不称为全新盲测。

## 文件与复现

`natural_records.json.gz` 保存完整自然评测的逐回合记录；`intervention_traces.npz` 将 204 个回合的逐帧数组合并为一个文件，GIF 从它生成。[来源记录](bridge/provenance.json)保存整理前的协议、源码哈希和场景复用依据；其中旧路径是历史标识，不是当前运行依赖。当前验证状态见 [verification.json](bridge/verification.json)，整理前后数量和等价性检查见 [organization.json](organization.json)。

标定和评测入口会重建临时缓存到 `bridge/.cache/`；自然评测完成后将回合文件合并并清理该阶段缓存。历史距离接管、误差包络、旧经验判据的源码快照、剂量扫描、重复权重与回合文件已删除。参数和数值计算保持不变，删除的是旧实现及重复存储。
