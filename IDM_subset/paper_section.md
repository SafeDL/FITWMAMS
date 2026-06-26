# IDM Baseline Evaluation

We estimate the safety-critical probability of an IDM-controlled ego vehicle under
highD long-tail driving conditions. A test case is defined by a scenario condition
`c` sampled from the highD tail scenario-condition distribution and a diffusion
latent variable `z ~ N(0, I)`. A frozen conditional diffusion model decodes
`(c, z)` into the adversary vehicle action sequence, and the ego vehicle responds
closed-loop with the IDM parameters in `tools/idm_ego.yaml`.

The same evaluation stack is used for all ADS subsets:

```text
c ~ highD tail scenario-condition distribution
z ~ N(0, I)
adversary actions = deterministic DDIM(c, z)
ego actions = ADS policy(state)
score = S_EVT(Y_sim)
failure = score >= S_EVT(x_c), x_c = 5.0
```

The reported probability is therefore the tail-conditional probability
`P(failure | highD tail scenario condition)`, not the naturalistic full-dataset
collision probability. When the exposure metadata passes reliability checks, the
summary also maps this value to a global highD exposure intensity through the
independent tail peak rate.

## Scenario Setup

Following and cut-in use separate configurations:

```text
following:
  lanes_count = 1
  horizon = 125 steps
  dynamics = kinematic_bicycle
  subset N = 3000, p0 = 0.2
  Monte Carlo N = 200000

cut-in:
  lanes_count = 2
  horizon = 100 steps
  dynamics = point_mass target vehicle
  subset N = 1000, p0 = 0.1
  Monte Carlo N = 10000
```

The frozen diffusion checkpoints are the held-out train/validation/test selected
models under `results/diffusion_natural/*/checkpoints/`. There is no fallback to
older full-data checkpoints.

## Subset Simulation

The implementation uses standard subset simulation. At level `i`, the sampler keeps
the highest-risk `p0` fraction of samples, then uses Metropolis-Hastings proposals in
the joint `(c, z)` space to sample the next conditional level. The final estimate is:

```text
P_hat = p0^level_idx * mean(score >= failure_threshold at final level)
```

Reliability diagnostics include closed-loop evaluation count, unique context count,
unique state fraction, largest context/state share, and MH acceptance rate. These
diagnostics affect interpretation flags, not the probability formula.

## Current Results

```text
following subset:      p = 0.00249067, se = 0.00006763
following Monte Carlo: p = 0.00241000, se = 0.00010964
cut-in subset:         p = 0.00680000, se = 0.00079609
cut-in Monte Carlo:    p = 0.00650000, se = 0.00080360
```

The corresponding artifacts are under `IDM_subset/results/`. Final-level playback
reads the saved subset samples and replays safety-critical final-level cases without
resampling the scenario distribution.
