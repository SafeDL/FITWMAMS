# CIH-WM continuation execution plan

1. Preserve the frozen factual model, diffusion model, and complete mechanism-guided response prior.
2. Keep every background slot, including `same_rear`, in training and evaluation.
3. Repair logged-context supervision, the retention/residual action map, execution projection, influence recovery, PPO masks, and explicit randomness.
4. Validate replay, snapshot/restore, shared-prefix identity, and full all-slot factual fidelity before training.
5. Train one calibrator with 400 supervised updates and at most 160 closed-loop fair-Energy PPO updates.
6. Evaluate the complete validation split with 32 futures per natural event; retain supervised if PPO lacks a positive recording-cluster interval.
7. Run the confirmation test only after every validation gate passes.
8. Record a precise rejection/readiness decision when no eligible candidate remains.
