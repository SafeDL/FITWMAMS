from __future__ import annotations
import argparse
from dynamic_ar_idm.fit import fit_dynamic_map, fit_pymc

parser = argparse.ArgumentParser(description="Fit Dynamic-AR IDM to the full highD cohort")
parser.add_argument("--dataset", default="dynamic_ar_idm/artifacts/full_data/full_highd_25hz.npz")
parser.add_argument("--output", default="dynamic_ar_idm/artifacts/full_posterior/dynamic_ar5_full_posterior.npz")
parser.add_argument("--ar-order", type=int, default=5)
parser.add_argument("--backend", choices=("laplace", "pymc"), default="laplace")
parser.add_argument("--draws", type=int, default=1000)
parser.add_argument("--tune", type=int, default=3000)
parser.add_argument("--chains", type=int, default=4)
parser.add_argument("--cores", type=int, default=1, help="parallel sampling workers for the PyMC backend")
parser.add_argument("--target-accept", type=float, default=.90, help="NUTS target acceptance probability")
parser.add_argument("--sampler-init", default="jitter+adapt_diag_grad", help="PyMC NUTS mass-matrix adaptation")
args = parser.parse_args()
if args.backend == "pymc":
    result = fit_pymc(args.dataset, args.output, ar_order=args.ar_order, draws=args.draws, tune=args.tune,
                      chains=args.chains, cores=args.cores, target_accept=args.target_accept,
                      sampler_init=args.sampler_init)
else:
    result = fit_dynamic_map(args.dataset, args.output, ar_order=args.ar_order, posterior_draws=args.draws)
print(result)
