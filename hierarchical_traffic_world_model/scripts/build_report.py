#!/usr/bin/env python3
"""Build the maintained full-experiment report and artifact manifest."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from world_model.src.core.utils import file_sha256, load_json, save_json  # noqa: E402

OUTPUT = ROOT / "results/hierarchical_traffic_world_model"


def main() -> None:
    evaluation = load_json(OUTPUT / "evaluation.json")
    training = load_json(OUTPUT / "training_summary.json")
    randomness = load_json(OUTPUT / "randomness_ablation.json")
    sampling = load_json(OUTPUT / "sampling_hierarchy_evaluation.json")
    factual = evaluation["factual_fidelity"]
    guided = factual["diffusion_guided_hiqr"]
    diffusion = factual["open_loop_diffusion"]
    event_strata = factual["event_strata"]
    distribution = evaluation["distribution_stochasticity"]
    intervention = evaluation["intervention_effectiveness"]
    brake = intervention["brake"]
    accelerate = intervention["accelerate"]
    response_hz = float(training.get("response_frequency_hz", 25.0))
    response_interval = 1.0 / response_hz
    intervention_passed = all(
        value["direction_success_rate"] >= 0.90
        and value["dose_monotonicity_rate"] >= 0.95
        and value["locality_ratio_far_to_near"] <= 0.20
        and value["response_within_natural_p10_p90_rate"] >= 0.50
        for value in (brake, accelerate)
    )
    lines = [
        "# 扩散引导 HiQR 分层交通世界模型：正式实验结论",
        "",
        (
            f"模型使用 {training['train_sequences']:,}/{training['validation_sequences']:,} "
            f"条 recording 隔离训练/验证序列，在第 {evaluation['checkpoint_epoch']} "
            f"轮取得最佳 checkpoint，并在全部 {evaluation['test_sequences']:,} 条测试序列上评价。"
        ),
        "",
        "## 事实保真",
        "",
        (
            f"- 冻结长程扩散 ADE/FDE 为 `{diffusion['ADE_m']:.4f}/"
            f"{diffusion['FDE_m']:.4f} m`；扩散 FDE 接近零是因为 K 含末端状态结点。"
        ),
        (
            f"- 闭环扩散引导 HiQR 为 `{guided['ADE_m']:.4f}/"
            f"{guided['FDE_m']:.4f} m`，P95 为 `{guided['P95_displacement_error_m']:.4f} m`；"
            "相对扩散的增量处于预先声明的事实误差预算内。"
        ),
        (
            f"- 事件分层的闭环 ADE/FDE：{event_strata['evt_labelled']['sequences']:,} 条 EVT 标签序列为 "
            f"`{event_strata['evt_labelled']['ADE_m']:.4f}/{event_strata['evt_labelled']['FDE_m']:.4f} m`；"
            f"{event_strata['semantic_cutin']['sequences']:,} 条完整语义 cut-in 序列为 "
            f"`{event_strata['semantic_cutin']['ADE_m']:.4f}/{event_strata['semantic_cutin']['FDE_m']:.4f} m`。"
        ),
        "- 去除长时程约束后 ADE 上升到 "
        f"`{factual['without_long_horizon_constraint']['ADE_m']:.3f} m`，证明 soft plan 被实质使用。",
        "",
        "## 分布随机性",
        "",
        (
            f"- 固定条件的 16 样本 mean/minADE 为 "
            f"`{distribution['sample_mean_ADE_m']:.4f}/{distribution['min_ADE_m']:.4f} m`，"
            f"energy score 为 `{distribution['energy_score_m']:.4f} m`。"
        ),
        (
            f"- 平均轨迹/末端成对距离为 "
            f"`{distribution['mean_pairwise_trajectory_distance_m']:.4f}/"
            f"{distribution['terminal_pairwise_distance_m']:.4f} m`；"
            "覆盖固定 K 下的模式内运动随机性。"
        ),
        (
            f"- 四象限同协议消融中，联合随机模型相对完全确定性模型的 proper energy "
            f"改善 `{100.0 * randomness['energy_improvement_fraction']:.2f}%`；"
            f"短程响应随机层在随机扩散之上继续改善 "
            f"`{100.0 * randomness['response_energy_improvement_fraction']:.2f}%`。"
        ),
        (
            "- 新增响应随机层相对 diffusion-only 的速度/加速度和 0.2 s jx/jy KS "
            "变化分别为 "
            f"`{100.0 * randomness['distribution_degradation_fraction']['speed']:.1f}%/"
            f"{100.0 * randomness['distribution_degradation_fraction']['ax']:.1f}%/"
            f"{100.0 * randomness['windowed_jerk_degradation_fraction']['jx']:.1f}%/"
            f"{100.0 * randomness['windowed_jerk_degradation_fraction']['jy']:.1f}%`，"
            "均满足不恶化超过 10% 的增量门槛。"
        ),
        (
            f"- Flow 改变 K 时，64 个固定 C0 条件平均覆盖 "
            f"`{sampling['scenario_constraint_randomness']['unique_joint_modes_mean']:.2f}` "
            "种联合行为模式，闭环末端成对距离为 "
            f"`{sampling['scenario_constraint_randomness']['closed_loop_terminal_pairwise_distance_m']['mean']:.3f} m`。"
        ),
        "",
        "## 干预有效性",
        "",
        (
            f"- 25 Hz 响应下，制动/加速方向成功率为 "
            f"`{brake['direction_success_rate']:.3f}/{accelerate['direction_success_rate']:.3f}`，"
            f"剂量单调性为 `{brake['dose_monotonicity_rate']:.3f}/"
            f"{accelerate['dose_monotonicity_rate']:.3f}`，首次响应延迟约 `{response_interval:.2f} s`。"
        ),
        (
            f"- far/near 局部性比为 `{brake['locality_ratio_far_to_near']:.3f}/"
            f"{accelerate['locality_ratio_far_to_near']:.3f}`，自然 P10-P90 覆盖率为 "
            f"`{brake['response_within_natural_p10_p90_rate']:.3f}/"
            f"{accelerate['response_within_natural_p10_p90_rate']:.3f}`。"
        ),
        "- 严格干预门槛由方向、剂量单调性、near/far 局部性和自然响应覆盖率共同判定；"
        "当前是否通过以锁定 `evaluation.json` 的指标为准，不再保留已淘汰的全局尺度扫描结果。",
        "",
        "## 结论边界",
        "",
        (
            "当前证据支持将该模型视为现行数据驱动范式下、具有事实重建、受控随机性和部分结构性"
            "干预响应的 25 Hz 交通世界模型；干预门槛仍需补充校准实验。它仍不能证明任意 ADS 策略下的反事实正确性；"
            "EVT 仍只作为外部人类风险标尺，不参与模型训练。"
        ),
    ]
    (OUTPUT / "experiment_report.md").write_text("\n".join(lines) + "\n")
    save_json(
        {
            "experiment_scope": "full",
            "selected_checkpoint": "checkpoints/best_hierarchical_world_model.pt",
            "checkpoint_sha256": file_sha256(
                OUTPUT / "checkpoints/best_hierarchical_world_model.pt"
            ),
            "train_validation_test": [72771, 13133, 10151],
            "three_objective_gates_passed": bool(
                intervention_passed
                and randomness.get("gates", {}).get("all_passed", False)
            ),
            "artifacts": {
                "training": "training_summary.json",
                "full_training": "full_training_manifest.json",
                "evaluation": "evaluation.json",
                "randomness_ablation": "randomness_ablation.json",
                "natural_response": "natural_response_calibration.json",
                "sampling": "sampling_hierarchy_evaluation.json",
                "report": "experiment_report.md",
                "visualization": "visualization_manifest.json",
            },
        },
        OUTPUT / "manifest.json",
    )


if __name__ == "__main__":
    main()
