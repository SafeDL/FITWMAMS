# QR-WM 长尾驾驶事件分析

## 协议

本结果由 `world_model/scripts/evaluate_long_tail_reproduction.py` 生成。评测使用 328 个 held-out highD EVT-tail 条件、5 秒闭环重建和 32 条分支（1 条确定性、31 条随机）。所有模型共享日志一秒历史、冻结 B0、道路图和逐响应 ego 回放；CAT-TopK 的 START 额外使用归档未来动作摘要，因此被标记为信息条件不对称的参考基线。

QR-WM checkpoint：`best_qr_world_model.pt`  
SHA-256：`caee850ff455a8f41cc196ac6c7f6979b86ff5f57566ad1f3f92f21db02be754`

## 与基线的统一结果

| 模型 | 确定性 FDE（m） | minFDE@32（m） | 交通特征 Fréchet | RBF-MMD |
| --- | ---: | ---: | ---: | ---: |
| RAMP-WM | 0.7519 | 0.7463 | 15.1449 | 0.3101 |
| FIRM-WM | 0.7098 | **0.4730** | 4.1594 | 0.1105 |
| Semi-Markov WM | 0.9278 | 0.8808 | **0.7199** | 0.0122 |
| CAT-TopK（信息不对称） | 0.7793 | 0.5704 | 4.0504 | **0.0040** |
| QR-WM | **0.6387** | 0.5240 | 0.7476 | 0.0081 |

QR-WM 的确定性 FDE 在五个结果中最低。其多分支 minFDE@32 低于 RAMP、Semi-Markov 和 CAT-TopK，但高于 FIRM；交通特征 Fréchet 仅高于 Semi-Markov，明显低于 RAMP、FIRM 和 CAT-TopK。CAT-TopK 使用额外的未来动作摘要，因此不应与其他模型作严格信息对称比较。

## QR-WM 细节

- 5 秒 ADE：0.1616 m；5 秒末端 FDE：0.6288 m。
- `minFDE ≤ 1 m` 覆盖率：89.94%；`minFDE ≤ 2 m`：98.48%；`minFDE ≤ 5 m`：100%。
- 生成轨迹与真实轨迹的 RBF-MMD：0.0081；交通特征 Fréchet：0.7476。
- 生成和真实轨迹的碰撞 episode rate 都为 0。
- 确定性跟驰误差：gap MAE 0.1695 m、TTC MAE 0.0557 s、DRAC MAE 0.0052 m/s²、相对速度 MAE 0.1578 m/s。
- 制动响应曲线 MAE：0.0726 m/s²。

## 产物

- `metrics.json`：完整数值、协议和 checkpoint 哈希。
- `figures/`：重建样例、轨迹分支、运动学、交互、分布及长尾事件分解图。
- `event_playbacks/`：`high_risk_following`、`hard_braking` 与 `close_interaction` 的 GIF 回放。
- `../overview/`：五模型汇总 JSON 与总览图。

该评测是给定条件下的闭环重建与生成分布分析，不是无条件自然驾驶场景生成评测。
