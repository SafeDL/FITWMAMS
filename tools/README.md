# tools：跨模块公共工具

`tools/` 存放 TREAD 主线工程共同使用的轻量工具。凡是 `process_highD/`、
`diffusion/`、`IDM_subset/` 中出现重复实现，且语义不属于某个单独模块的
函数，都应优先放在这里。

## 当前内容

```text
tools/
├── io.py                 # resolve_path、load_npz、write_json、write_csv
├── evt.py                # POT/GPD EVT tail model、return level 和 S_EVT 标定
├── risk.py               # y_long 计算、闭环风险和 EVT risk_score 标定
├── highd_longitudinal.py # highD following 事件重建与共享 y_long 计算
├── highd_cutin.py        # highD cut-in 事件重建、raw context 和风险评分
├── highd_exposure.py     # 暴露量、独立峰值和 tail rate 汇总
├── context.py            # context NPZ 读取和单条 context 组装
├── normalization.py      # numpy / torch 归一化与反归一化
├── diffusion_adapter.py  # frozen diffusion prior 的共享适配器
├── plot_style.py         # 论文图样式、标签、坐标轴和 paper artifact helpers
└── idm_ego.yaml          # process_highD/subset 共用 IDM ego 参数
```

## 使用原则

- 风险评分统一从 `tools/risk.py` 引入，避免各工程维护不同公式。
- EVT 模型统一从 `tools/evt.py` 引入，避免 highD 拟合和闭环仿真使用不同尾部映射。
- NPZ、JSON、CSV 和配置路径解析统一使用 `tools/io.py`。
- 学术绘图和 paper artifact manifest/README helper 统一使用 `tools/plot_style.py`，
  避免各模块维护不同字体、符号标签、线型编码和后处理记录规则。
- highway-env IDM ego 参数统一放在 `tools/idm_ego.yaml`，供 process_highD 回放和 subset 闭环复用。
- 子模块不应新增仅做转发的兼容入口；调用点应直接 import `tools/` 中的真实实现。
- 不把模块私有训练逻辑、模型结构或脚本默认参数放进 `tools/`。

## 风险评分口径

`tools/risk.py` 提供同一套可配置的纵向风险评分实现：

- `process_highD/` EVT 拟合、长尾筛选和 `IDM_subset/` 闭环仿真都先计算同一个
  `y_long`：`1/TTC`、`1/THW`、
  `1/gap` 和 `DRAC` 的 softmax-pool 聚合，再加 collision、
  near collision 和 hard-brake 配置项。
- 如果配置 `evt.score_space: evt` 且提供 EVT model，`risk_score` 表示
  `S_EVT(y_long) = -log P_EVT(Y_long > y_long)`；否则 `risk_score`
  回退为 raw `y_long`。
如果后续需要调整危险得分公式，应优先修改 `tools/risk.py` 和对应 YAML：
闭环事件验证使用 `closed_loop_risk_scoring`，highD EVT/tail context 使用
`tools/highd_longitudinal.py` 中的共享默认配置。
修改后需要同步更新相关 README。

EVT 模型由 `process_highD/scripts/estimate_following_exposure.py` 和
`process_highD/scripts/estimate_cutin_exposure.py` 生成，保存
`u, xi, beta, z20, z50, z100`、return level 置信区间和暴露量校准的碰撞距离估计。

共享长尾自然驾驶 context 由 `process_highD/scripts/select_following_tail_contexts.py`
和 `process_highD/scripts/select_cutin_tail_contexts.py` 生成。
`extract_highd_events.py` 通过 `tools/highd_longitudinal.py` 一次性生成
`following_event_scores.csv` 和 `following_event_contexts.npz`。
following 的 `Y_long` 在完整不等长原始 following 事件上计算；同一缓存中的
`context_anchor_frame`、`scenario_conditions` 和 `initial_states` 只用于后续长尾
context selection、diffusion 生成和 125 帧回放对齐。
`estimate_following_exposure.py` 和 following/cut-in tail selection 脚本必须读取这些缓存；
缓存缺失会直接报错，不回退 raw highD 重建。
筛选脚本默认在全部有效 following events 的 tail 上构建 context 分布。
`IDM_subset/` 在这批长尾样本下继续做子集模拟。

## 维护边界

`tools/` 只放跨 `process_highD/`、`diffusion/`、`IDM_subset/` 复用的真实实现。当前
`plot_style.py` 中的 manifest、README 和 figure helper 被 `results/build_*_paper_experiments.py`
直接使用，属于共享论文产物工具，不是死代码。不要在这里放只服务单个脚本的私有训练逻辑，
也不要新增仅做 import 转发的兼容模块。
