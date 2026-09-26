# 世界模型评测结果

这是唯一维护的世界模型评测包：Flow → Diffusion → HiQR → 在线 MA-IDM → HighwayEnv。
全 Test 指标使用 10,151 条 highD 轨迹；生成场景结果单独报告，不与 Test 指标混算。

## 核心报告

- `factual_test.json`：全 Test 事实轨迹重建。
- `factual_hiqr_only_test.json`：同物理条件、同 ADS/计划的事后配对消融，用于误差和碰撞归因；不是在线“无干预世界”。
- `ads_test.json`、`ads_right_lane_test.json`、`dose_order_test.json`：ADS 制动/左右换道及非零纵向动作剂量评估。
- `lane_meta_actions.json`、`generated_ads_sweep.json`：生成世界中的换道语义及 ADS META 动作压力测试。
- `npc_to_npc_highd_test.json`、`npc_to_npc_sweep.json`：观测 Test 子集与受控探针中的 NPC-NPC 响应。
- `ma_idm_parameter_distribution_audit.json`：观测数据标定参数的分布审计。
- `factual_tail_audit.json`、`dose_inversion_audit.json`、`left_added_pair_audit.json`、`spatial_path_scene_audit.json` 等：解释边界场景和归因的审计记录。
- `plans_full/` 与 `runtime_assets_manifest.json`：可重放的 10,151 条 Test Diffusion 计划、生成协议和运行资产哈希。

代表性动画：[`ads_demo.gif`](ads_demo.gif)、[`right_lane_meta_demo.gif`](right_lane_meta_demo.gif)、
[`npc_to_npc_highd_demo.gif`](npc_to_npc_highd_demo.gif)、
[`npc_autonomous_lane_demo.gif`](npc_autonomous_lane_demo.gif) 和
[`ads_cutin_pass_demo.gif`](ads_cutin_pass_demo.gif)。其余专项 GIF 保留为回归/边界诊断，
不代表额外方法分支。

碰撞按原始仿真事件报告，不设“零碰撞”门槛；ADS 主动切入、NPC 响应和 NPC-NPC 接触分开归因。
指标口径、结论及限制见 [`doc/Traffic_World_Model_Convergence.md`](../../../doc/Traffic_World_Model_Convergence.md)。
完整一致性检查：`conda run -n tread python hierarchical_world_model/scripts/verify_online_artifacts.py`。
本目录原名为 `online_closed_loop/`；迁移只改变工件位置，不改变原始指标和协议标识。
`generated_ads_sweep.json` 保留运行时脚本哈希 `evaluation_script_sha256_at_run`；
当前 `evaluation_script_sha256` 对应改名后的脚本；除默认输出目录外，
其导入路径也随共享包更名而变化。

`world_model/` 更名为 `traffic_components/` 后，导入和运行指纹路径随之变化。
[`package_rename_manifest.json`](package_rename_manifest.json) 保留原始与当前哈希；
各 JSON 中的当前运行哈希只证明改名后的代码一致性，指标与 GIF 未重新计算。
