# TrafficBots highD 基线

`evaluation.json` 与 `full_acceptance.json` 记录正式测试和验收，`audit.json`、
`training_manifest.json` 与 `intervention_crn.npz` 保留方法及配对随机数证据。
`checkpoints/best.ckpt` 是正式评测使用的模型权重。

`posterior_resimulation.json` 是单独的场景条件 posterior 诊断，不能当作 prior-only
生成测试。方法和重跑方式见
[`external_model_baselines/models/trafficbots/README.md`](../../../external_model_baselines/models/trafficbots/README.md)。

当前共享缓存的 Test 排序已改变，旧 `audit.json` 的有序身份与之不匹配；保存的配对场景仍属于当前 Test。这里保留原协议对照，后续使用当前数据作方法比较时需重新评测并更新审计。
