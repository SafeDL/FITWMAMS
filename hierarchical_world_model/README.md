# 分层交通行为世界模型

当前运行主链是 **Flow → Diffusion → HiQR → 在线 MA-IDM NPC 响应 → HighwayEnv**。
HiQR 编码器、统一动力学和规范序列接口由
[`traffic_components/`](../traffic_components/) 提供；该目录是本主链的共享基础组件，不是另一条世界模型分支。
`HierarchicalWorldSampler.create_world()` 与 `rollout_world()` 默认只执行一次世界：
HiQR 每帧给出高精度动作，在线控制器从已实现的车辆状态选择风险最高的候选前车；
只在预测间隙不安全且 HiQR 尚未充分减速时，以固定 episode 后验驾驶人参数
对该 NPC 加制动修正；已实现的前车加速度突变可引出提前减速或加速。
当 NPC 已明显落后冻结计划、仍在追近同车道慢速 ADS 时，在线层还会抑制
HiQR 的正向追赶加速度；这个条件本身不强迫 IDM 制动。
若 NPC 已有 Diffusion 生成的相邻车道意图，车道语义控制器按地图车道中心和
当前已实现的目标车道前后车位置决定延后或完成该机动；纵向受干预后，横向目标可沿
生成路径按已实现的纵向位置推进，不再强行追赶原计划时间戳。Diffusion 生成的约束计划仍是条件信息，
安全的既定换道可避免原车道间隙造成的误刹车；若 ADS 已在向目标车道横移，
则利用其已实现的横向速度预留交汇空间，继续执行必要的纵向响应。
当已实现的 ADS 或 NPC 前车急刹、原车道短时净间隙即将不安全、且没有既定
Diffusion 换道意图时，控制器还会检查地图相邻车道在当前、预计入道和恢复时刻
的前后间隙；若安全则自主锁定目标车道中心并完成换道。新机动期间纵向动作
由该车固定 episode 参数的 MA-IDM 执行，动作变化率沿用 12 m/s³ 约束；
状态快照包含目标车道承诺，避免到达中心后被旧 HiQR 保持车道轨迹拉回。
但没有预跑无干预世界，也不把其未来状态/动作或真实未来状态喂给在线控制器。
不受影响的 NPC 原样执行 HiQR 动作。ADS 策略在每个决策边界读取当前观测并给出动作；
可用 `OnlineAccelerationWindowPolicy` 和 `OnlineSemanticLaneChangePolicy` 做完整加速度
区间及闭环换道测试。`rollout_world()` 同时返回逐帧车辆对碰撞和越界证据，
便于把 ADS 策略、NPC 响应和物理事件按同一时间轴核查。
`controller=None` 只保留为明确的 HiQR 消融开关，不是默认环境。

旧场景条件化差分协议与其运行脚本、结果已移除。新协议在完整 10,151 条 test 的日志 ADS 事实重建上得到
ADE/FDE **0.035606/0.027590 m**，全部有限；同一 100 m/s 数值速度上限下的
纯 HiQR 对照为 **0.035517/0.027576 m**，不启用旧手工响应适配器。
原先 50 m/s 的积分上限低于 highD test
约 64 m/s 的实测最高车速，旧报告 0.04002/0.03635 m 仅作历史记录，不再作为同物理条件对照。
生成式 GPU 149 帧冒烟测试也已通过。
535 个地图确认的 NPC 相邻换道计划中，终态完成率 **93.64%（501/535）**；
安全换道消除了第 46,684 行的误刹车，同时第 6,944 行的慢速 ADS 横移安全门槛
在 −8/−6 m/s² 干预下消除了新增重叠。
事实场景中，NPC 对另一 NPC 前车的在线纵向修正涉及 **1 个场景、103 帧**，
全部事实场景均无矩形重叠。
新自主换道在全 10,151 条无干预事实测试中触发 **0 帧**，因此事实
ADE/FDE 与原版完全一致。
另有 36 个受控 HighwayEnv NPC→NPC 制动测试：20/30 m 的 24 个近距场景均在
第二帧向制动方向响应，全部 36 个场景无 NPC–NPC 重叠或越界，9 组间距/速度
下的四档平均动作均按制动强度排序。报告见
[`npc_to_npc_sweep.json`](../results/hierarchical_world_model/evaluation/npc_to_npc_sweep.json)。
进一步从完整 10,151 条 Test 筛出 936 条 NPC 前后车场景，分别强制前车执行
−8/−6/−4/−2 m/s² 一秒制动，后车仍由完整在线模型驱动。首秒后车制动方向率
依次为 853/936、768/936、618/936、263/936；四档逐场景效应在
0.05 m/s² 容差下 935/936 有序，全部 3,744 次运行无 NPC–NPC 重叠。
原始车辆重叠另有 3 次，均涉及日志 ADS 与被强制制动的 NPC 前车，未被掩盖。
在 −8/−6/−4/−2 m/s² 档，目标后车分别有 **60/33/6/0** 次自主提出新换道；
固定 149 帧窗口内分别有 **54/14/1/0** 次达到地图目标车道终态阈值，
其余多在窗口后段启动。全部 3,744 次运行的目标后车均无越界或 NPC–NPC 重叠。
对其中 30 次窗口末未完成的自主换道另做同一 HighwayEnv 世界续跑：前 149 帧
与原离线结果位置 ADE 0.000088 m、启动帧一致，后 75 帧使用明示的终端计划外推
与 ADS 零加速度/零转向；30/30 于第 224 帧前完成，无重叠或后车越界。
这验证的是外推尾段下的安全收敛，不把尾段当作 Diffusion 的预测精度。
已提供 `hierarchical_world_model.src.continuation.append_future_reference(world, tail_xy)`
作为计划耗尽后的显式续跑入口；调用方须预先采样足够长的响应随机过程并提供新的
`[batch,frames,6,2]` 计划。当前内建 Diffusion 仍只生成 149 帧，诊断脚本的
线性尾段只是一个可复核的后续计划提供者，不是默认长时规划器。
`HighwayEnvClosedLoopWorld.snapshot()`/`restore()` 现在一并保存和恢复动态
追加的参考计划；边界前快照可用于独立 ADS 分叉，单元测试已覆盖不同尾段的
分叉后恢复与同尾段逐步回放一致。滚动 Diffusion 重规划仍需补齐。
报告和动画见
[`late_autonomous_lane_continuation.json`](../results/hierarchical_world_model/evaluation/late_autonomous_lane_continuation.json)
及 [`late_autonomous_lane_continuation.gif`](../results/hierarchical_world_model/evaluation/late_autonomous_lane_continuation.gif)。
逐场景报告和真实 Test 动画分别见
[`npc_to_npc_highd_test.json`](../results/hierarchical_world_model/evaluation/npc_to_npc_highd_test.json)
及 [`npc_to_npc_highd_demo.gif`](../results/hierarchical_world_model/evaluation/npc_to_npc_highd_demo.gif)。
自主新换道的第 83,340 行另见
[`npc_autonomous_lane_demo.gif`](../results/hierarchical_world_model/evaluation/npc_autonomous_lane_demo.gif)：
NPC 3 急刹后 NPC 4 完成地图相邻换道，终态中心误差约 0.049 m，且无重叠。
逐场景 ADE 的 P99 为 **0.08013 m**，最大为 **0.446 m**（第 67,927 行）。
原来最差场景的 10.923 m 误差由错误截速导致，已消除；前车横向离开导致的
第 61,899 行误刹车也已修正。剩余长尾仍与换道附近的交互控制有关。
全 test 筛选后的 1,851 个同车道后车和 1,000 个左目标车道后车也已做六档纵向／
完整左换道独立闭环、HiQR-only 安全消融及逐场景剂量排序，结果在
[`evaluation/`](../results/hierarchical_world_model/evaluation/)。
近距后车的方向率为 93.6%–97.4%；ADS −8/−6 m/s² 档分别有
**131/40** 个目标后车自主换道且原始重叠均为 0。
六档逐场景严格单调率为 **67.9%**，0.05 m/s² 容差下为 **97.6%**；
地图目标车道左换道在线分支有 84/1,000 个原始重叠，相对同 ADS 动作纯 HiQR 对照的
417 个重叠避免 333 个、未新增重叠；ADS 到真实车道中心的终态完成率为 100%。自主换道已具备紧急响应入口，但更广泛的
择道/超车策略仍需扩展。
当 ADS 正横向进入目标车道、但按当前速度 NPC 将在入道前完整通过时，
地图约束的通过判据保留 HiQR 动作，不把 ADS 误作必须跟随的前车。
当前通过判据使用真实地图车道中心、已实现 ADS 状态和当帧 HiQR 动作，
并保持已确认的通过承诺；另一 NPC 真正需要更强制动时仍优先处理。
五条历史回归场景在新地图路径下均未产生相对纯 HiQR 的新增重叠。对应逐帧对照见
[`ads_cutin_pass_demo.gif`](../results/hierarchical_world_model/evaluation/ads_cutin_pass_demo.gif)。
此前逐场景逆序审计发现少数强制制动分支存在 HiQR 追赶计划与在线制动修正相抵的现象；
正向追赶抑制曾将 0.05 m/s² 容差单调率由 96.4% 提至 98.2%；
加入自主换道后完整队列为 **97.6%**，保持原车道的 1,720 场景为 **98.2%**，
但最大逆序仍约 0.99 m/s²。因此不能仅以均值方向率判断响应合理性。16 条 highD 场景上
HighwayEnv 与离线执行器位置差异 ADE 为 0.000034 m。另在一条 ADS −8 m/s²
干预场景中，NPC 有 96 帧启用空间路径模式，两种执行器的模式逐帧一致、
位置差异 ADE 为 0.000293 m；另在原高速度失真场景，两个执行器的
位置差异 ADE 为 0.000050 m。这些都只是执行器抽样一致性。
另有 1,024 个 Flow 生成场景的八策略 HighwayEnv 审核：全部数值有限且无越界；
采用换道入口速度反馈的 ADS 左换道 1,024/1,024 到达地图目标车道中心，最大误差
0.060 m。左换道仍有 27 个矩形重叠场景，均为 ADS 切入暴露；强制 +2/+4 m/s²
分别出现 264/561 个仿真碰撞场景，隔离 ADS 追尾暴露后仍有 54/102 个几何重叠。
因此当前是可运行 ADS 测试环境候选，不能称作已发布的可信安全评测环境。
第 27,342 行曾因较近的目标车道 NPC 遮蔽急刹 ADS 而发生重叠；当前多前车风险
选择已在完整 1,851 场景强制 −8 m/s² 队列中将原始重叠恢复为 0，并保留
[`npc_ads_brake_lane_demo.gif`](../results/hierarchical_world_model/evaluation/npc_ads_brake_lane_demo.gif)
供逐帧核查。尚有 34/535 个事实换道、4/25 个 −8 m/s² 空间路径换道未达到
当前终态完成阈值。事实未完成的 34 个中，32 个日志轨迹在同一时间窗口也未完成；
另外 2 个在线结果只略超 0.75 m 阈值。因此不能靠强制提前换道抬高事实完成率，
但干预后的 4 个换道仍需检查其后续收敛。安全间隙打开后的收尾控制使
ADS −8 m/s² 队列中 535 个计划换道的终态完成数由 492 增至 497，
空间路径模式由 16/25 增至 21/25，整组仍无原始车辆重叠。
纵向正向追赶抑制的第 32,028 行可视化见
[`npc_plan_pursuit_demo.gif`](../results/hierarchical_world_model/evaluation/npc_plan_pursuit_demo.gif)。

当前在线协议的可复核入口：

```bash
conda run -n tread python hierarchical_world_model/scripts/evaluate_online_factual.py
conda run -n tread python hierarchical_world_model/scripts/evaluate_online_factual.py --hiqr-only --output results/hierarchical_world_model/evaluation/factual_hiqr_only_test.json
conda run -n tread python hierarchical_world_model/scripts/audit_online_factual_tail.py
conda run -n tread python hierarchical_world_model/scripts/evaluate_online_ads.py
conda run -n tread python hierarchical_world_model/scripts/evaluate_online_dose_order.py
conda run -n tread python hierarchical_world_model/scripts/evaluate_generated_online_ads.py
conda run -n tread python hierarchical_world_model/scripts/render_online_ads_demo.py
conda run -n tread python hierarchical_world_model/scripts/render_online_npc_lane_demo.py
conda run -n tread python hierarchical_world_model/scripts/render_online_npc_to_npc_demo.py
conda run -n tread python hierarchical_world_model/scripts/evaluate_online_npc_to_npc_sweep.py
conda run -n tread python hierarchical_world_model/scripts/evaluate_online_npc_to_npc_highd.py
conda run -n tread python hierarchical_world_model/scripts/audit_online_late_lane_continuation.py
conda run -n tread python hierarchical_world_model/scripts/render_online_npc_to_npc_highd_demo.py
conda run -n tread python hierarchical_world_model/scripts/render_online_npc_to_npc_highd_demo.py --test-row 83340 --output results/hierarchical_world_model/evaluation/npc_autonomous_lane_demo.gif
conda run -n tread python hierarchical_world_model/scripts/render_online_ads_cutin_pass_demo.py
conda run -n tread python hierarchical_world_model/scripts/render_online_npc_follow_demo.py
conda run -n tread python hierarchical_world_model/scripts/evaluate_online_highway_parity.py
conda run -n tread python hierarchical_world_model/scripts/audit_online_spatial_path.py
conda run -n tread python hierarchical_world_model/scripts/verify_online_artifacts.py
```

前者按完整 10,151 条 test 做带日志 ADS 动作的事实重建审核；后者在 Flow 生成场景中
分别运行在线 ADS 保持动作、-8 m/s² 制动和闭环左换道，并以工程既有道路/车辆风格
绘制 GIF。三种策略均独立启动单次闭环，不互相提供未来参考轨迹。
NPC→NPC 制动 GIF 是单独的受控合成 HighwayEnv 探针，不作为 highD test 指标。

项目级证据、分协议比较及未完成的可信性验收见
[`doc/Traffic_World_Model_Convergence.md`](../doc/Traffic_World_Model_Convergence.md)。

## 随机驾驶人复现接入

`src/stochastic_drivers/` 是论文复现模型进入本项目的唯一纵向边界。它把分层世界模型
的 `[x,y,vx,vy,ax,ay]` 状态转换为跟驰观测，在25 Hz plant内保持5 Hz动作，并支持
NumPy driver的随机状态快照/恢复。限幅不反写GP、AR或regime状态。

```bash
python -m hierarchical_world_model.scripts.stochastic_drivers inventory
python -m hierarchical_world_model.scripts.stochastic_drivers rollout --model ma_idm
```

默认只允许证据状态可用的B-IDM、MA-IDM、稳定MAP Dynamic-AR和需要外部固定版本源码的
官方Active Inference wrapper。Multi-regime和独立纵向Active适配仍可用于研究诊断，但必须
显式传入 `--allow-unaccepted`，不能静默进入自动驾驶测试。

同数据比较使用218-pair共享cohort：recording 25的全部182个事件训练，recording 26/36
的全部36个合格事件评测，结果位于
`results/driver_reproduction/matched_metrics.json`。各模型的论文原生复现仍
单独保留；共同评测不会覆盖论文特有的推断结论。

## 历史分支边界

旧 CIH 响应候选和 50 m/s 截速事实结果不属于当前方案，其结果目录已从活动结果树移除。
仍被独立反事实响应测试复用的 validation reaction-event 数据位于
`external_model_baselines/evaluation/counterfactual_response/evidence/`；它不是世界模型运行资产。
