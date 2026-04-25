#!/usr/bin/env python3
"""
Read NN-CLI predict JSONs (with `outputs[]` and `logits[]` arrays) for an
in-distribution and an out-of-distribution image set, compute the
free-energy score per sample, and emit a threshold.json.

Free-energy: E(x) = −log Σᵢ exp(zᵢ)
  Lower  E → more in-distribution.
  Higher E → likely OOD.

The threshold is picked at a chosen percentile of the ID distribution
(default 95th). Above the threshold we declare a prediction OOD.
"""
import argparse
import json
import math
from pathlib import Path


def free_energy(logits):
    m = max(logits)
    return -(m + math.log(sum(math.exp(z - m) for z in logits)))


def percentile(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = (len(sorted_vals) - 1) * p / 100
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


def stats(name, energies, percentiles):
    return {
        "n": len(energies),
        "min": min(energies),
        "max": max(energies),
        "mean": sum(energies) / len(energies),
        **{f"p{p}": percentile(energies, p) for p in percentiles},
    }


def round_dict(d, places=4):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = round_dict(v, places)
        elif isinstance(v, float):
            out[k] = round(v, places)
        else:
            out[k] = v
    return out


def load_energies(path):
    data = json.loads(Path(path).read_text())
    if "logits" not in data:
        raise SystemExit(f"{path}: missing 'logits' array — server too old?")
    return [free_energy(l) for l in data["logits"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--id-predict", required=True, help="NN-CLI predict JSON for the ID set")
    ap.add_argument("--ood-predict", required=True, help="NN-CLI predict JSON for the OOD set")
    ap.add_argument("--out", required=True, help="Output threshold.json path")
    ap.add_argument(
        "--id-percentile",
        type=float,
        default=95.0,
        help="ID-distribution percentile used as the OOD threshold (default 95)",
    )
    args = ap.parse_args()

    id_energies = sorted(load_energies(args.id_predict))
    ood_energies = sorted(load_energies(args.ood_predict))

    threshold = percentile(id_energies, args.id_percentile)

    # Threshold-based confusion matrix (E > threshold ⇒ predicted OOD).
    id_accepted = sum(1 for e in id_energies if e <= threshold)
    id_rejected = len(id_energies) - id_accepted
    ood_rejected = sum(1 for e in ood_energies if e > threshold)
    ood_accepted = len(ood_energies) - ood_rejected

    output = {
        "freeEnergyThreshold": threshold,
        "idPercentileUsed": args.id_percentile,
        "rule": "predicted_ood = (free_energy > freeEnergyThreshold)",
        "idStats": stats("id", id_energies, [1, 5, 50, 90, 95, 99]),
        "oodStats": stats("ood", ood_energies, [1, 5, 50, 95, 99]),
        "confusion": {
            "idAccepted": id_accepted,
            "idRejected": id_rejected,
            "oodAccepted": ood_accepted,
            "oodRejected": ood_rejected,
            "idAcceptanceRate": id_accepted / len(id_energies),
            "oodRejectionRate": ood_rejected / len(ood_energies),
        },
    }
    output = round_dict(output, places=4)

    Path(args.out).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
