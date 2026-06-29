# highD EVT-tail Normalizing Flow Experiment Report

## Selected Baseline

Final selected configuration:

```text
normalizing/scripts/configs/highd_tail_flow_best.yaml
```

Final output directory:

```text
results/highd_tail_flow_best/
```

Final checkpoint:

```text
results/highd_tail_flow_best/checkpoints/best_tail_conditional_maf.pt
```

The selected model keeps the strict EVT-tail target distribution. It uses exact quota sampling over `(mask_pattern, primary_slot)` and latent sampling temperature `1.0295`. No full-natural or near-tail rows were added.

## Training Chain

```text
density_fit:             base strict-tail weighted density fit, max_epochs 240
likelihood_refinement:   resume from density_fit, lr 1e-5, epochs 180
final_refinement:        resume from likelihood_refinement, lr 5e-7, stopped at 230 epochs
ultra_low_lr_probe:      tried lr 1e-7 from selected checkpoint; no val-NLL improvement
```

Checkpoint selection used unweighted strict-tail validation NLL. The selected checkpoint improved validation NLL from `-88.3093` to `-88.3726`; the final `1e-7` continuation did not improve that value.

## Final Metrics

```text
conditional rq-spline MAF    train -147.0504  val -88.3726  test -106.0621
GMM                          train -125.6069  val -43.3915  test  -65.9791
Gaussian                     train    1.4248  val  16.2438  test    6.7505
Copula                       train   30.3386  val  36.5070  test   32.6475
Unconditional RealNVP        train   78.3462  val  81.3035  test   79.1450
```

Full EVT-tail 2209 vs 2209 reproduction:

```text
mean per-feature KS:          0.0928
mean Wasserstein:             0.3126
Pearson corr MAE:             0.0419
mask occupancy L1:            0.0000
primary-slot occupancy L1:    0.0000
invalid_rate:                 0.0000
overlap_rate:                 0.0000
negative_gap_rate:            0.0000
semantic_error_rate:          0.0000
sampling rejection_rate:      0.1421
```

Slot-wise mean KS:

```text
same_front  0.0390
same_rear   0.0726
left_front  0.1392
left_rear   0.1455
right_front 0.0518
right_rear  0.1082
```

## Comparison Against Previous Refinement Baseline

Previous selected baseline is reported here only as a metric reference; the current retained baseline is `results/highd_tail_flow_best/`.

```text
test NLL:                  -105.9444 -> -106.0621
mean per-feature KS:          0.0946 -> 0.0928
mean Wasserstein:             0.3186 -> 0.3126
Pearson corr MAE:             0.0420 -> 0.0419
mask occupancy L1:            0.0000 -> 0.0000
primary-slot occupancy L1:    0.0000 -> 0.0000
sampling rejection_rate:      0.1408 -> 0.1421
```

Slot-wise KS changed as follows:

```text
same_front  0.0388 -> 0.0390
same_rear   0.0746 -> 0.0726
left_front  0.1418 -> 0.1392
left_rear   0.1478 -> 0.1455
right_front 0.0526 -> 0.0518
right_rear  0.1119 -> 0.1082
```

The selected baseline improves density and distribution metrics with a small rejection-rate increase. Physical validity after rejection remains perfect.

## Temperature Calibration

For the selected checkpoint, the relevant temperature scan was:

```text
temperature 1.0290: mean KS 0.0929, mean W 0.3130, corr 0.0419, rejection 0.1421
temperature 1.0295: mean KS 0.0928, mean W 0.3126, corr 0.0419, rejection 0.1421
temperature 1.0335: mean KS 0.0937, mean W 0.3122, corr 0.0415, rejection 0.1431
temperature 1.0345: mean KS 0.0935, mean W 0.3115, corr 0.0415, rejection 0.1431
```

Temperature `1.0295` was selected because it gave the best mean KS while still improving test NLL, Wasserstein, and correlation versus the previous refinement baseline. Higher temperatures reduced Wasserstein/correlation but worsened KS and rejection.

## Rejected Candidates

```text
previous_refinement_baseline + temperature 1.02:
  test NLL -105.9444, mean KS 0.0946, mean W 0.3186, corr 0.0420, rejection 0.1408
likelihood_refinement_checkpoint + temperature 1.031:
  test NLL -105.9911, mean KS 0.0931, mean W 0.3124, corr 0.0419, rejection 0.1428
final_refinement + temperature 1.0345:
  test NLL -106.0621, mean KS 0.0935, mean W 0.3115, corr 0.0415, rejection 0.1431
ultra_low_lr_probe:
  validation NLL did not improve over -88.3726
additional_low_lr_probe:
  validation NLL did not improve over the previous refinement baseline
alternate_seed_long_run:
  test NLL -109.6735, mean KS 0.1078, mean W 0.3787, corr 0.0433, rejection 0.1199
```

`alternate_seed_long_run` was rejected despite much better NLL because generated strict-tail distribution metrics were worse, especially `left_rear`.

## Remaining Issues

The largest remaining KS errors are still lateral acceleration and side-slot one-second action summaries. Longitudinal relative distance still dominates Wasserstein error. Next work should focus on `left_rear` and longitudinal-gap transforms while keeping exact quota sampling and physical validity unchanged.
