#!/usr/bin/env python3
"""Leave-one-recording-out evaluation; every qualified pair is evaluated once OOF."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from bayesian_ma_idm.src.data import load_ragged_pairs
from bayesian_ma_idm.src.full_evaluation import evaluate_full_population
from bayesian_ma_idm.src.full_model import fit_full_population

def main() -> None:
    p = argparse.ArgumentParser(description="Final B-IDM / Gaussian MA-IDM recording-held-out evaluation.")
    p.add_argument("--model", choices=("b_idm", "ma_idm"), default="ma_idm")
    p.add_argument("--dataset", default=str(ROOT / "bayesian_ma_idm/evidence/highd_cohort/highd_author_full_25hz.npz"))
    p.add_argument("--output-dir", help="Defaults to evidence/population or evidence/personalized.")
    p.add_argument("--posterior-dir", help="Optional directory supplying already fitted fold posteriors.")
    p.add_argument("--futures", type=int, default=64)
    p.add_argument("--personalize-from-prefix", action="store_true")
    p.add_argument("--reuse-existing-posteriors", action="store_true", help="Re-evaluate saved fold posteriors without refitting them.")
    a = p.parse_args()
    _, pairs = load_ragged_pairs(a.dataset)
    recordings = sorted({int(pair["recording_id"]) for pair in pairs})
    default = "personalized" if a.personalize_from_prefix else "population"
    output = Path(a.output_dir or ROOT / "bayesian_ma_idm/evidence" / default)
    output.mkdir(parents=True, exist_ok=True)
    for holdout in recordings:
        train = [recording for recording in recordings if recording != holdout]; fitted = (Path(a.posterior_dir) if a.posterior_dir else output) / f"{a.model}_holdout_{holdout}.npz"; posterior = fitted; report = output / f"{a.model}_holdout_{holdout}.json"
        if not a.reuse_existing_posteriors or not posterior.exists():
            fit_full_population(a.dataset, posterior, model=a.model, training_recordings=train)
        evaluate_full_population(a.dataset, posterior, report, evaluation_recordings=[holdout], futures=a.futures,
                                 personalize_from_prefix=a.personalize_from_prefix)
        print(report)
if __name__ == "__main__": main()
