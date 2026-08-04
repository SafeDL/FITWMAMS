# 全 highD 测试集条件重建

本目录保存五个世界模型在完整 held-out highD 测试集（24,216 条序列）上的已验证、模型原生条件重建报告。每个条件直接使用真实 `C0+B0`、道路图和逐响应 ego 回放；不使用 Flow 采样，因此用于衡量世界模型本身的条件演化能力。

正式命令为：

```bash
python world_model/scripts/test_ramp_world_model.py
python world_model/scripts/test_firm_world_model.py
python world_model/scripts/test_semi_markov_world_model.py
python world_model/scripts/test_cat_topk_world_model.py
python world_model/scripts/test_qr_world_model.py
python world_model/scripts/evaluate_test_conditional_reconstruction.py --mode native
```

`study_manifest.json` 记录每个源报告和 checkpoint 的 SHA-256；`overview/test_conditional_reconstruction_summary.json` 给出可追溯的五秒指标索引。CAT-TopK 使用了归档的未来动作摘要，因而仅作为信息条件不对称的参考基线。

`overview/01_model_native_comparison.png` 是由上述正式汇总直接绘制的五秒指标对比图；斜线的 CAT-TopK 柱明确表示其信息条件不对称。需要单独重绘时执行：

```bash
python world_model/scripts/plot_reconstruction_result_summaries.py --only test
```

`evaluate_test_conditional_reconstruction.py --mode diagnostic --output-dir <独立目录>` 仍可用于单模型、32 分支的深入诊断；它不会作为子集模拟中的标准全量评测，以免保存全部分支造成不必要的内存和运行时间开销。
