# highD 数据处理

从原始录像提取七车片段，计算安全包络侵入风险 `R_SEI`，用 EVT 筛选长尾事件，再生成各模块共用的数据与场景清单。配置统一在 [config.yaml](config.yaml)，代码目录不保存生成数据。

## 数据口径

- 原始数据：`highD_dataset/Matlab/data/`；25 Hz，每片段 150 个状态点、149 次转移，跨度 5.96 秒。
- 主车为轨迹完整的小汽车，起点固定同车道及左右车道的前后六个槽位；有效背景车覆盖完整窗口，保留实际车辆尺寸。
- 换道片段保留完整过程及至少 1 秒换道前、2 秒换道后轨迹。
- 60 段录像按 42 / 9 / 9 分为 Train / Val / Test，共 96,055 个片段，划分数量为 72,771 / 13,133 / 10,151。
- EVT 在全部片段上统一拟合和筛选，得到 2,964 条尾部事件。`R_SEI` 用于风险排序，不是碰撞概率。

## 运行

在项目根目录依次运行：

```bash
conda activate tread
python process_highD/scripts/extract_highd_natural_segments.py
python process_highD/scripts/fit_evt.py
python normalizing_flow/scripts/prepare.py
python process_highD/scripts/prepare_highd_sequences.py
python process_highD/scripts/build_event_bank.py
python process_highD/scripts/build_manifests.py
```

Flow 的 `prepare.py` 只生成共享场景条件，不启动训练；后续时序缓存依赖这份条件数据。已有缓存直接复用，明确需要重建时给 `prepare_highd_sequences.py` 添加 `--rebuild`。

## 输出

- [results/evt/](../results/evt/README.md)：风险数据、EVT 模型、事件清单、诊断图和原始轨迹回放。
- [results/highd_shared_training_data/](../results/highd_shared_training_data/README.md)：按 `segment_id` 对齐的场景条件与真实时序缓存。

Flow、Diffusion 和世界模型的训练由各自入口完成；事件清单中的固定小样本实验不替代完整 Test 评测。单元测试：`pytest -q process_highD/tests`。
