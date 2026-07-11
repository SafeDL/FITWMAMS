# Cut-in 论文实验

本目录只基于已有结果生成后处理后的 cut-in 论文产物。
本次没有重新执行 cut-in diffusion 训练、EVT 拟合或 subset simulation。

## 输入

- `event_scores`: `results/highd_events/cutin_event_scores.csv`
- `event_cache_summary`: `results/highd_events/cutin_event_cache_summary.json`
- `subset_summary`: `IDM_subset/results/cutin/latent_subset_summary.json`
- `subset_level_stats`: `IDM_subset/results/cutin/latent_subset_level_stats.csv`
- `subset_samples`: `IDM_subset/results/cutin/latent_subset_samples.npz`
- `monte_carlo_summary`: `IDM_subset/results/monte_carlo_cutin/latent_monte_carlo_summary.json`
- `cutin_diffusion_dataset`: `results/diffusion_natural/cutin/dataset.npz`
- `evt_model`: `results/highd_cutin_tail/evt/cutin_peak_evt_model.json`
- `evt_summary`: `results/highd_cutin_tail/evt/cutin_peak_evt_summary.json`
- `exposure_summary`: `results/highd_cutin_tail/exposure/highd_cutin_exposure_summary.json`
- `tail_condition_distribution`: `results/highd_cutin_tail/contexts/scenario_condition_distribution.npz`
- `tail_contexts`: `results/highd_cutin_tail/contexts/tail_contexts.npz`
- `tail_generated_scenarios`: `results/highd_cutin_tail/generated/diffusion_generated_scenarios.npz`
- `tail_generated_summary`: `results/highd_cutin_tail/generated/diffusion_generated_scenarios_summary.json`
- `tail_distribution_similarity_summary`: `results/highd_cutin_tail/generated/figures/distribution_similarity_summary.json`

## 生成产物

- `results/paper_experiments/cutin/cutin_safety_threshold_inverse_calibration.png`
- `results/paper_experiments/cutin/cutin_gpd_diagnostic_panel.png`
- `results/paper_experiments/cutin/cutin_tail_diffusion_generalization_panel.png`
- `results/paper_experiments/cutin/cutin_subset_level_score_histograms.png`

## 复用的已有产物

- 复用已有产物：`cutin_tail_diffusion_generalization_panel: results/highd_cutin_tail/contexts/scenario_condition_distribution.npz`
- 复用已有产物：`cutin_tail_diffusion_generalization_panel: results/highd_cutin_tail/contexts/tail_contexts.npz`
- 复用已有产物：`cutin_tail_diffusion_generalization_panel: results/highd_cutin_tail/generated/diffusion_generated_scenarios.npz`
- 复用已有产物：`cutin_tail_diffusion_generalization_panel: results/diffusion_natural/cutin/dataset.npz`
- 复用已有产物：`cutin_subset_level_score_histograms: IDM_subset/results/cutin/latent_subset_samples.npz`

## 跳过的产物

- 无

## 解读说明

- 所有论文图都使用共享 TREAD 论文样式：300 dpi 导出、Times 兼容衬线字体，以及 STIX/LaTeX 风格数学渲染。
- 主 exposure 分母为 `all_vehicle_km`。
- ADS intensity 定义为 `conditional exceedance probability x highD tail peak exposure rate`。
- 这些概率以 highD cutin tail scenario-condition distribution 为条件，不是无条件道路事故率。
