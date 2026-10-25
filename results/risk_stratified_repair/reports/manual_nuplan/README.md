# nuPlan 人工评价交接

Diffusion Planner 的 highway 正式比较已结束，原始与修复权重已经交付。本目录只保存人工评价材料；尚未运行城市仿真，nuPlan 指标由用户手动生成，单独保存，不并入 highD 统计或用于重新挑选权重。

## 已交付文件

本文模型使用 `diffusion_planner/guided/20260926/`：

| 文件 | 用途 | SHA256 |
| --- | --- | --- |
| `initial/model.pth` | 官方初始部署参数的原生等价导出 | `33b04d5a4eac837b6c441dbcb72afe99dbfbd8b638e484ba48b25bf96170555e` |
| `repaired/model.pth` | 与 highway 最终评价完全相同的本文模型 | `8c1e4b5e664f91f6bb00926533cd6c953a1301f8a7568faf7d4760ec11d44383` |
| `initial/args.json`、`repaired/args.json` | 官方架构与归一化配置 | 与正式导出逐字节一致 |
| `handoff.json` | 原始／修复来源、哈希、共同场景及指标目录 | 文件内记录完整关联 |

`monte_carlo`、`subset` 的对照模型也分别保存一组交接。三组共用初始参数与场景清单；本文交付使用预先指定的 guided 分支，不依据最终测试改选其他模型。四份正式导出均经官方 `DiffusionPlanner.initialize`、EMA 加载、276 个状态张量和原生推理核对，参数一致，输出最大差为零。验收见工程 `results/risk_stratified_repair/formal/diffusion_planner/native_acceptance/`。

`scenarios.json` 固定官方 Val14 的 1,118 个不同 token、14 类场景及种子 `20260926`；SHA256 为 `87b184d569c85830f9b46d8001c59d8e4fb77bed4d254901c82d1a32b026a3a0`。`preparation/config/scenario_filter/manual_evaluation.yaml` 保留每类最多 100 个、15 秒间隔、有效目标筛选及 `shuffle=false`。实际运行数取决于这些原生规则；两次运行都保存实际 token 列表，并核对一致，不把 1,118 当作已完成的仿真次数。

## 手动运行 DP

在现有 `tread` 运行。先令 `ADS_ROLE=initial`，再令其为 `repaired`，分别执行相同命令；每次输出保存到交接记录指定的 `native_metrics/<role>`。默认使用 CPU，若改用 GPU，两次保持相同设备和配置。原有 `run_closed_loop.sh` 为 CaRL 专用，不用于 DP。

```bash
conda activate tread
source /media/hp/DATA/nuplan_workspace/env.sh
export PYTHONPATH="/home/hp/FITWMAMS:$PYTHONPATH"
export USE_TF=0
ADS_ROLE=repaired
ADS_PAIR="$NUPLAN_WORKSPACE/FITWMAMS_results/risk_stratified_repair_manual_nuplan/diffusion_planner/guided/20260926"
ADS_CHECKPOINT="$ADS_PAIR/$ADS_ROLE/model.pth"
ADS_ARGS="$ADS_PAIR/$ADS_ROLE/args.json"
python "$NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py" \
  +simulation=closed_loop_reactive_agents \
  planner=diffusion_planner \
  "planner.diffusion_planner.config.args_file=$ADS_ARGS" \
  "planner.diffusion_planner.ckpt_path=$ADS_CHECKPOINT" \
  planner.diffusion_planner.enable_ema=true \
  planner.diffusion_planner.device=cpu \
  scenario_builder=nuplan \
  "scenario_builder.data_root=$NUPLAN_DATA_ROOT/nuplan-v1.1/splits/trainval" \
  scenario_builder.include_cameras=false \
  scenario_builder.max_workers=1 \
  scenario_filter=manual_evaluation \
  scenario_filter.shuffle=false \
  worker=sequential \
  run_metric=true \
  seed=20260926 \
  'hydra.searchpath=[file:///home/hp/FITWMAMS/results/risk_stratified_repair_manual_nuplan/preparation/config,file:///home/hp/FITWMAMS/ads/configs,pkg://carl_nuplan.planning.script.config.common,pkg://nuplan.planning.script.config.common,pkg://nuplan.planning.script.experiments]' \
  "group=$ADS_PAIR/native_metrics/$ADS_ROLE"
```

评价其他研究对照时，把 `ADS_PAIR` 中的 `guided` 改为相应方法名。原始／修复比较保留相同 token、筛选配置、设备、seed、失败记录和原生指标；加载失败或未完成场景单独记录。部署不加载保持教师，不按数据集切换模型权重。

本工程已核对原生权重和正式交接路径的 CLI 配置，见 `preparation/final_configuration_acceptance.json`；`--cfg job` 未执行仿真。完整结果见工程 `doc/risk_stratified_repair.md` 第 8.5 节。手动结果留在本目录，不反向用于调整已封存的 highway 模型。

## CaRL 状态

CaRL 两轮先导未获得通过开发准入的增强版本，本阶段没有 CaRL 正式交接对。`development/carl/handoff/` 中是最终保留的官方参数等价导出，不能标为增强权重。原生 `PPOPlanner`、`closed_loop_reactive_agents_action`、`OneStageController` 的配置与加载验收仍保存在 `preparation/carl_*`，没有执行城市闭环。当前请先评价上述 DP 交付。
