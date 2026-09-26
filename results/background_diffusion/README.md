# 背景车扩散模型结果

`manifest.json` 记录当前 118 维条件模型和资产清单；`dataset_contract.json` 定义条件协议。
正式测试指标见 `evaluation_summary.json`，逐序列值见 `evaluation_per_sequence.npz`。
训练曲线、历史和测试图保存在本目录。`checkpoints/best_background_diffusion.pt`
是 EMA 推理权重；[`design_audit/`](design_audit/README.md) 保存条件表达能力和运动基审计；
`playbacks/` 保存由 `playback_manifest.json` 标识的测试场景 GIF。

测试结果使用真实未来稀疏结点作条件，属于条件重建。复核入口见
[`diffusion/README.md`](../../diffusion/README.md)。
