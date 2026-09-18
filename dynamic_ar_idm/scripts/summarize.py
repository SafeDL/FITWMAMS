"""Write a compact AR(0) vs AR(5) decision artifact from completed evaluations."""
from __future__ import annotations
import argparse, json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--ar0", default="dynamic_ar_idm/artifacts/full_evaluation/natural_metrics_full_ar0.json")
parser.add_argument("--ar5", default="dynamic_ar_idm/artifacts/full_evaluation/natural_metrics_full_stable_map_rho.json")
parser.add_argument("--output", default="dynamic_ar_idm/artifacts/full_evaluation/lag_order_test_full.json")
args = parser.parse_args()
zero, five = (json.loads(Path(path).read_text(encoding="utf-8")) for path in (args.ar0, args.ar5))
metrics = ("a_rmse", "v_rmse", "s_rmse", "a_crps", "v_crps", "s_crps")
comparison = {name: {"ar0": zero["horizons"]["5"][name]["mean"], "ar5": five["horizons"]["5"][name]["mean"],
                     "relative_change": five["horizons"]["5"][name]["mean"] / zero["horizons"]["5"][name]["mean"] - 1.} for name in metrics}
improved = [name for name, value in comparison.items() if value["relative_change"] < 0]
not_improved = [name for name, value in comparison.items() if value["relative_change"] >= 0]
pairs = five.get("pairs", zero.get("pairs", "the selected"))
if not_improved:
    conclusion = "AR(5) improves " + ", ".join(improved) + "; it does not improve " + ", ".join(not_improved) + ". The exception is retained rather than tuned away."
else:
    conclusion = "AR(5) improves all six 5 s RMSE/CRPS metrics under this matched protocol, reproducing the paper's qualitative AR-versus-IID conclusion."
report = {"comparison": f"same {pairs} highD pairs, same 25 Hz plant and rolling origins; AR order is the only intended change", "horizon_s": 5,
          "results": comparison, "improved_metrics": improved, "not_improved_metrics": not_improved,
          "conclusion": conclusion, "inference_caveat": "This is a joint MAP/Laplace approximation; it is not an exact NUTS posterior replication."}
Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
print(args.output)
