# OOD Calibration

Picks a free-energy threshold for an image classifier so callers can flag
out-of-distribution inputs (e.g. a photo of a hand instead of a skin
lesion). The script + datasets are model-agnostic; only the resulting
`threshold.json` is tied to a specific trained checkpoint.

## What it computes

For each prediction, NN-Server / NN-CLI now returns the dense head's
pre-activation logits `z`. The **free-energy** score
`E(x) = -log Σ exp(zᵢ)` collapses the whole logit vector into one
number — lower means more in-distribution. Softmax is shift-invariant
(`softmax(z) == softmax(z + c)`), so two very different logit vectors
can produce the same probabilities; `E` recovers the magnitude
information softmax discards.

Calibration computes `E` over a set of in-distribution (ID) images and a
set of out-of-distribution (OOD) images, then picks the threshold at a
chosen percentile of the ID distribution (default 95th).

## Layout

```
NN-CLI/
├── scripts/calibration/                 # generic, reusable
│   ├── README.md
│   ├── fetch-ood.sh                     # downloads DTD + Places365 + makes synthetic
│   ├── make-synthetic-ood.py            # generates black/white/noise/colorblock images
│   ├── run-calibration.sh               # runs NN-CLI predict on both sets
│   └── compute-threshold.py             # reads logits, emits threshold.json
├── extern-datasets/ood/                 # gitignored, populated by fetch-ood.sh
│   ├── dtd/
│   ├── places365-val/
│   └── synthetic/
└── examples/<TASK>/<train-app-N>/output/
    └── threshold.json                   # per-model output
```

## Usage

From the repo root:

```bash
# 1. One-time: fetch OOD images (~3 GB total).
./scripts/calibration/fetch-ood.sh

# 2. Run calibration against a specific trained model.
./scripts/calibration/run-calibration.sh \
    --model examples/ISIC-MILK10k/train-app-9/output/best_model.json \
    --id-images examples/ISIC-MILK10k/train-datasets/isicMILK10k/input \
    --output-dir examples/ISIC-MILK10k/train-app-9/output
```

This writes `threshold.json` next to the model:

```json
{
  "freeEnergyThreshold": 13.40,
  "idPercentileUsed": 95.0,
  "idStats":  { "n": 500, "p50": 11.2, "p95": 13.40, ... },
  "oodStats": { "n": 1500, "p50": 14.85, ... },
  "confusion": {
    "idAcceptanceRate": 0.95,
    "oodRejectionRate": 0.83
  }
}
```

Consumers (e.g. DermaSnap) embed `freeEnergyThreshold` as a constant
and flag predictions with `E > threshold` as low-confidence / likely
OOD.

## Datasets used

- **DTD** — 5 640 texture images. Pure texture, no objects. License: free
  for research. https://www.robots.ox.ac.uk/~vgg/data/dtd/
- **Places365 val_256** — 36 500 scene photos at 256×256. License: CC.
  Approximates "user accidentally took a photo of their living room."
  http://places2.csail.mit.edu/
- **Synthetic** — generated locally: pure black/white/gray, RGB blocks,
  random noise, blurred patterns. Mirrors common user-mistake failure
  modes (e.g. the all-black image that originally motivated this work).

## Caveats

- The ID holdout is sampled from the model's training set, so the
  threshold is **slightly optimistic** — real unseen skin images will
  have somewhat higher `E`. Mitigated by using the 95th percentile
  rather than the max, but a future improvement is to keep a held-out
  validation slice that the model never saw.
- The threshold belongs to the specific trained checkpoint. Retrain
  the model → rerun calibration.
