# CIH-WM retained evidence

- `priors/`: frozen learned behaviour prior and calibrated longitudinal mechanism.
- `evidence/`: immutable highD event reference and human-response support set.
- `candidate_unaccepted/`: the only full CIH-WM run retained.  It contains one supervised and one policy checkpoint, each with one canonical 256-sequence all-follower diagnostic. It did not pass the direction and `same_rear` non-inferiority gates; it is diagnostic evidence, not a released model.
- `playbacks/`: regenerated qualitative comparison GIFs.  They are not acceptance evidence.

The frozen factual-transition layer is in `../factual_hiqr/` (the directory name is historical). It consists of relational observation encoding, hierarchical belief filtering, DDIM diffusion-plan conditioning, coordinated jerk decoding, and 25 Hz kinematic integration. Its historical masked-protocol full-test score is ADE 0.040227 m and FDE 0.036870 m. Current code includes `same_rear`; the historical score must not be presented as an all-slot result.
