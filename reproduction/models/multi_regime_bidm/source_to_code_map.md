# 论文到保留实现的映射

| 论文对象 | 当前实现 | 对齐状态 |
|---|---|---|
| §5.1：三维事件特征 K-means style | `src/style.py` | 机制对齐；highD 聚类不沿用论文语义名 |
| §3.2、§5.2：HDP-HSMM | `src/hsmm.py` | documented adaptation：finite Gaussian HSMM + Poisson duration |
| 无自转移的显式持续时间状态 | `src/hsmm.py`、`src/online_filter.py` | 已实现并测试 |
| 式(19)–(25)：style population / regime B-IDM | `src/pymc_stage_b.py` | 部分对齐；层次NUTS使用独立维度稳定化先验，不是LKJ prior |
| pooled B-IDM | `src/pymc_stage_b.py::fit_style_pooled_nuts` | 同style、同观测子集和同NUTS预算独立拟合 |
| IID Gaussian action noise | `src/pymc_stage_b.py`、`src/evaluation.py` | 对齐 |
| 论文离线代表事件仿真 | `scripts/visualize_stage_a.py` | 仅论文风格类比图 |
| highD 因果部署评测 | `src/evaluation.py` | 论文外工程扩展；5 Hz decision / 25 Hz plant；模型间共享创新 |

保留实现只对应 `evidence/dataset`、`evidence/segmentation`、`evidence/posterior` 和 `evidence/heldout`。旧 MAP、smoke、训练 25+26 诊断和局部过滤工件已清理。
