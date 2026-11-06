# nuPlan 共同场景

`common_scenarios.json` 固定 `scenarios.json` 的 SHA256；场景过滤配置保存在 `preparation/config/`。原始和安全增强版使用相同清单，城市闭环由用户手动执行，指标不并入 highD 结果。

```bash
conda activate tread
cd /media/hp/DATA/nuplan_workspace
source env.sh
ADS_PAIR=/home/hp/FITWMAMS/results/ads/runs/safety_enhanced/manual_nuplan
ADS_ROLE=enhanced
PLANNER=diffusion_planner PLANNER_DEVICE=cuda \
  bash run_closed_loop.sh "diffusion_$ADS_ROLE" 2 "$ADS_PAIR/$ADS_ROLE/model.pth"
```

交付目录使用 `initial` 和 `enhanced` 两个角色；执行时将 `ADS_ROLE` 分别设为这两个值。参数 `2` 用于加载检查，需要完整评价时由用户改为 `all`。`args.json` 与原生 `model.pth` 同目录，训练状态 `policy.pth` 不直接交给 nuPlan。
