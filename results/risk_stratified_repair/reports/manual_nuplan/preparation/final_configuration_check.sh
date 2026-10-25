#!/usr/bin/env bash
set -e
source /media/hp/DATA/nuplan_workspace/env.sh
export PYTHONPATH="/home/hp/FITWMAMS:$PYTHONPATH"
export USE_TF=0
ADS_ROLE=repaired
ADS_PAIR="$NUPLAN_WORKSPACE/FITWMAMS_results/risk_stratified_repair_manual_nuplan/diffusion_planner/guided/20260926"
ADS_CHECKPOINT="$ADS_PAIR/$ADS_ROLE/model.pth"
ADS_ARGS="$ADS_PAIR/$ADS_ROLE/args.json"
/home/hp/anaconda3/envs/tread/bin/python "$NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py" \
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
  "group=$ADS_PAIR/native_metrics/$ADS_ROLE" --cfg job
