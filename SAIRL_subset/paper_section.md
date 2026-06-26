# SAIRL Policy Evaluation

`SAIRL_subset/` evaluates a SAIRL ego policy under the same long-tail scenario
generator and EVT scoring stack used by the IDM baseline. The adversary action
sequence is decoded from a frozen conditional diffusion model, while the ego vehicle
is controlled by a converted SAIRL policy checkpoint:

```text
SAIRL_subset/weights/sairl/model.npz
```

The policy is loaded for inference only. The TensorFlow-to-NPZ conversion script is
kept only for weight reproducibility and is not part of the normal evaluation run.

## Policy Interface

The current SAIRL adapter implements the reference discrete policy as a PyTorch MLP:

```text
input:  7 vehicles x [presence, x, y, vx, vy] = 35 dims
hidden: 96, 96, 96
output: 5 highway-env discrete MDP actions
```

The retained action set is:

```text
LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER
```

Therefore the SAIRL policy can choose lane-change actions in the two-lane cut-in
environment. Following remains single-lane because the test configuration matches
the IDM baseline.

## Comparable Evaluation

All non-policy components are shared with the IDM baseline:

```text
same highD tail scenario-condition distribution
same deterministic DDIM adversary sampler
same initial-state reconstruction
same EVT/GPD safety-critical threshold
same subset-simulation estimator
same final-level playback mechanism
```

The compressed result JSON records fairness checks against IDM, including the EVT
threshold, subset sample count, and `p0`.

## Current Results

```text
following subset:      p = 0.17946667, se = 0.00110831
following Monte Carlo: p = 0.16855000
cut-in subset:         p = 0.03530000, se = 0.00151126
cut-in Monte Carlo:    p = 0.03610000
```

These values are tail-conditional failure probabilities under the same long-tail
test distribution, not naturalistic full-dataset collision probabilities.
