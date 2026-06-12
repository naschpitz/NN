# NN-CLI — Neural Network CLI Tool

JSON-config-driven CLI for training, predicting, testing, and calibrating neural networks. Supports both ANN and CNN models with CPU/GPU execution.

## Build

```bash
./build.sh            # from repo root
./NN-CLI/build.sh     # standalone
```

Binary: `NN-CLI/build/NN-CLI`. Test: `NN-CLI/build/test_nncli` (pass `--full` for long MNIST train/test).

## Dependencies

- Qt6 Core + Concurrent
- CNN (PUBLIC — transitively provides ANN + OpenCLWrapper)
- ncurses-wide (training terminal UI)
- tcmalloc (optional, lower RSS under heavy augmentation)
- nlohmann/json (header-only, in `libs/nlohmann/`)
- stb (header-only, in `libs/stb/`)

## Modes

| Mode | Description |
|------|-------------|
| `train` | Train a model from JSON config or IDX data |
| `predict` | Run inference on a trained model |
| `test` | Evaluate a trained model on a test set |
| `calibrate` | Compute a free-energy OOD threshold: gather ID/OOD images, run predict to get logits, score with −log Σ exp(z), pick a percentile threshold |

## CLI flags

| Flag | Description |
|------|-------------|
| `--config, -c <path>` | JSON config file (required for train) |
| `--mode, -m <mode>` | `train`, `predict`, `test`, or `calibrate` |
| `--device, -d <cpu|gpu>` | Device override (default: cpu) |
| `--input, -i <path>` | JSON file with batch inputs (predict mode, required) |
| `--input-type <vector|image>` | Input data type (overrides config) |
| `--samples, -s <path>` | Samples JSON file (train/test) |
| `--idx-data <path>` | IDX3 data file (alternative to --samples) |
| `--idx-labels <path>` | IDX1 labels file (requires --idx-data) |
| `--output, -o <path>` | Output file (train: model, predict: result) |
| `--output-type <vector|image>` | Output data type (overrides config) |
| `--log-level, -l <quiet|error|warning|info|debug>` | Log level (default: error) |
| `--gpu-profile` | Enable OpenCL GPU kernel profiling (~12% overhead) |

### Calibrate-mode flags

| Flag | Description |
|------|-------------|
| `--id-images <dir>` | In-distribution image directory (recursed, required) |
| `--ood-dir <dir>` | OOD images directory (default: `<cwd>/extern-datasets/ood`) |
| `--id-sample-count <N>` | ID subsample size (default 500) |
| `--ood-sample-count <N>` | OOD subsample size (default 1500) |
| `--id-percentile <P>` | ID percentile used as threshold (default 95) |
| `--no-fetch` | Don't auto-download OOD even if --ood-dir is empty |

## Architecture

MVC pattern: `App` → `Controller` → `{Model (Runner), View (Window)}`

| Layer | Class | Role |
|-------|-------|------|
| App | `NN-CLI_App` | Entry point, config loading, mode dispatch |
| Controller | `TrainingController`, `PredictController`, `TestController` | Orchestrate training/prediction/test flow |
| Model | `NN-CLI_Runner` (ANN/CNN variants) | Bridge to ANN::Core / CNN::Core |
| View | `NN-CLI_TerminalUI_*` (Panel, Table, Window, ProgressBar) | ncurses terminal UI |
| Observer | `IRunnerObserver` | Model → Controller notifications |

## Config format

JSON config controls: device, layers, training params, I/O, augmentation, validation split. See `examples/` for valid configs (MNIST, ISIC-MILK10k).

## Augmentation

- `augmentationFactor` — N× total samples per class (0 = disabled)
- `balanceAugmentation` — augment minority classes to max class count
- `fullAugmentation` — augment ALL samples every epoch (originals included)
- `autoClassWeights` — inverse-frequency class weights
- `augmentationProbability` — per-transform probability (default 0.5)
- GPU augment via `NN-CLI_GpuAugmenter` (OpenCL-backed)

## Data loading

- IDX format (MNIST) — `NN-CLI_DataLoader`
- Image format (JPEG/PNG) — `NN-CLI_ImageLoader` + stb
- JSON samples file — `NN-CLI_Loader`
- Data splitting — `NN-CLI_DataSplitter`

## Model serialization

- Format: `model.json` + `params.bin` (binary weights)
- Serializer: `NN-CLI_ModelSerializer` (shared with NN-Server)
- Package format: `.nnmodel` = `tar` archive of model.json + params.bin
- Checkpoint files: `trained_E-<epoch>_S-<step>_L-<loss>.nnmodel.tar` (training output) and `checkpoint_E-<epoch>_L-<loss>.nnmodel.tar` (per-epoch)

## Testing

```bash
cd NN-CLI/build && ./test_nncli          # basic tests
cd NN-CLI/build && ./test_nncli --full   # includes long MNIST cases
```

## Gotchas

- `nncli_common` OBJECT library compiles shared sources once (linked into both NN-CLI and test_nncli)
- tcmalloc linked with `--no-as-needed` to prevent modern linkers from stripping it
- Terminal UI uses ncurses-wide (check for `ncursesw` library)
- Example configs in `examples/ISIC-MILK10k/` and `examples/MNIST/`
