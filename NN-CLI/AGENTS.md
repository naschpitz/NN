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
