# CIH-WM retained evidence

- `priors/`: frozen learned behaviour prior and calibrated longitudinal mechanism.
- `evidence/`: immutable highD event reference and human-response support set.
- `candidate_unaccepted/`: the earlier diagnostic run, retained for historical comparison.
- `continuation/`: the preserved current CIH response calibration and its complete formal validation.
- `training/`: the latest full retraining attempt. It is retained as a rejected experiment because its complete validation was worse than `continuation/`.
- `playbacks/`: regenerated qualitative comparison GIFs.  They are not acceptance evidence.
- `response_prior_lineage.json`: source evidence for the frozen mechanism-guided response prior loaded inside CIH-WM.
- `evaluation_index.json`: the consolidated result index for the one CIH-WM method chain.

The frozen factual-transition layer is in `../factual_hiqr/` (the directory name is historical). It consists of relational observation encoding, hierarchical belief filtering, DDIM diffusion-plan conditioning, coordinated jerk decoding, and 25 Hz kinematic integration. Its historical masked-protocol full-test score is ADE 0.040227 m and FDE 0.036870 m. Current code includes `same_rear`; the historical score must not be presented as an all-slot result.

CIH-WM has one method chain: factual transition, direct causal routing, frozen mechanism-guided response prior, constrained calibration, and unified execution. The `continuation/` calibrator remains the effective response component; `training/` does not supersede it.
