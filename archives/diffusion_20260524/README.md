# Naturalistic Action Diffusion Prior

`diffusion/` now contains a clean highD-based natural driving prior for car-following events. Its only job is to learn the natural distribution of future lead-car actions from short ego-lead interaction histories.

The model answers:

```text
Given recent ego-lead interaction history, can we generate natural, smooth,
physically feasible lead-car future actions close to highD statistics?
```

It does not condition on target severity or try to control generated severity. Tail calibration, guided generation, RSS checks, and closed-loop adversarial testing belong in later modules outside this natural-prior training path.

## Current Scope

- Event type: `following`
- Input: ego/lead history states, relative-history stream, and current/history-only context features
- Output: lead-car future action sequence, defaulting to jerk `[jx(t+1), ..., jx(t+H)]`
- Training objective: DDPM noise MSE plus optional `x0` reconstruction L1 and smoothness auxiliary loss
- Sampling: standard unconditional DDPM sampling conditioned only on scene history/context

## Key Files

```text
diffusion/
  scripts/
    configs/natural_following.yaml
    build_natural_dataset.py
    train_natural_diffusion.py
    evaluate_natural_prior.py
    sample_natural_rollouts.py
  src/
    data.py
    features.py
    normalization.py
    types.py
    model.py
    train.py
    kinematics.py
    utils.py
```

## Environment

The base conda environment may not have PyTorch. The existing project environment can be used with:

```bash
conda activate jzm
```

or explicitly:

```bash
/home/hp/anaconda3/envs/jzm/bin/python
```

## Build Dataset

```bash
/home/hp/anaconda3/envs/jzm/bin/python diffusion/scripts/build_natural_dataset.py
```

The default config writes to:

```text
data/diffusion_natural/following/
```

Main dataset arrays:

```text
context_states
future_states
context_features
relative_history
actions
split_index
recording_id
event_id
anchor_frame
ego_length
adv_length
```

`future_states` has shape `[N, horizon_steps, 2, 6]` in the anchor ego local frame.
Actor `0` is the real highD ego future and actor `1` is the real highD lead future.

## Train

```bash
/home/hp/anaconda3/envs/jzm/bin/python diffusion/scripts/train_natural_diffusion.py
```

To force a dataset rebuild before training:

```bash
/home/hp/anaconda3/envs/jzm/bin/python diffusion/scripts/train_natural_diffusion.py --rebuild-dataset
```

Checkpoints:

```text
checkpoints/best.pt
checkpoints/best_noise_mse.pt
checkpoints/last.pt
```

Training also writes the full convergence trace:

```text
training_history.csv
training_history.json
training_summary.json
```

TensorBoard logs stochastic train/validation losses and deterministic fixed-noise validation losses.

## Evaluate Natural Prior

```bash
/home/hp/anaconda3/envs/jzm/bin/python diffusion/scripts/evaluate_natural_prior.py
```

Outputs:

```text
naturalness_summary.json
naturalness_metrics.csv
diversity_summary.json
natural_prior_plots/ax_distribution_real_vs_generated.png
natural_prior_plots/jerk_distribution_real_vs_generated.png
natural_prior_plots/speed_distribution_real_vs_generated.png
natural_prior_plots/example_rollouts.png
```

The evaluator reports:

- validation denoising/reconstruction/smoothness metrics
- acceleration and jerk distribution statistics
- Wasserstein, KS, and histogram L1 distances
- action clipping, speed, jerk, acceleration, and trajectory discontinuity rates
- lead speed, final speed, and displacement statistics against the real highD lead future
- interaction naturalness using real highD ego future with real/generated lead futures: gap, TTC, THW, relative speed, closing speed, collision, and near-collision rates
- multi-sample diversity for repeated contexts

When the action representation is jerk, evaluation first integrates generated jerk to acceleration before calling `integrate_following_actions()`.

## Sample Rollouts

```bash
/home/hp/anaconda3/envs/jzm/bin/python diffusion/scripts/sample_natural_rollouts.py
```

This writes decoded actions, acceleration, and integrated lead trajectories to:

```text
natural_rollouts.npz
natural_rollouts_summary.json
```
