# NN-Server — Neural Network Inference Server

A QTcpServer-based HTTP inference server for serving **ANN** and **CNN** model predictions over a REST API. NN-Server loads a trained model into a thread-safe pool and exposes a `POST /predict` endpoint that accepts JSON vectors or raw image uploads, returning predictions as JSON or PNG images.

## Architecture

```
HTTP Client
    ↓
HttpServer (QTcpServer + QThreadPool)
    ↓
CorePool (thread-safe pool of ANN::Core<float> / CNN::Core<float>)
    ↓
ANN / CNN predict()
```

### CorePool

The `CorePool` manages pre-loaded ANN/CNN core instances with thread-safe acquire/release semantics:

- **Thread safety**: `QMutex` + `QWaitCondition` — blocks when all cores are busy
- **Each entry** holds an `ANN::Core<float>` or `CNN::Core<float>` plus an `available` flag
- **Acquire**: returns the first available core (blocks if none)
- **Release**: marks a core as available and wakes waiters
- **Pool size** is set via the `poolSize` config field (default: number of CPU cores)

### Components

| Component | Base Class | Role |
|-----------|-----------|------|
| `HttpServer` | `QTcpServer` | Accepts TCP connections, reads HTTP requests |
| `RequestHandler` | `QRunnable` | Dispatched via `QThreadPool`; acquires a Core, runs prediction, releases it |
| `CorePool` | — | Thread-safe pool of pre-loaded ANN/CNN cores |
| `ImageLoader` | — | Decodes images (JPEG, PNG, etc.) to NCHW float arrays |
| `Loader` | — | Loads model JSON, auto-detects ANN vs CNN, creates cores |
| `Logger` | — | Rotating log file with configurable max size |

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check — returns `{"status":"ok"}` |
| `POST` | `/predict` | Run prediction (JSON or image input) |
| *any* | *other* | Returns 404 `{"error":"Not found. Use POST /predict"}` |

### POST /predict — JSON Input

```bash
curl -X POST http://localhost:8080/predict \
     -H "Content-Type: application/json" \
     -d '{"input": [0.1, 0.5, 0.9, 0.3]}'
```

Response:
```json
{
  "output": [0.95, 0.05],
  "logits": [2.94, 0.00]
}
```

### POST /predict — Image Input

```bash
curl -X POST http://localhost:8080/predict \
     -H "Content-Type: image/jpeg" \
     --data-binary @photo.jpg
```

For image-output models, the response is a raw PNG image.

## Configuration

NN-Server is configured through a JSON file. Pass the path with `--config`, or it defaults to `config.json` in the current directory.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `"model"` | string | Yes | — | Path to the trained model `.nnmodel` or `.nnmodel.tar` package |
| `"port"` | integer | No | `8080` | TCP port to listen on (1–65535) |
| `"poolSize"` | integer | No | CPU cores | Number of Core instances in the pool |
| `"maxBodySize"` | integer | No | `10` | Maximum request body size in MB. `0` = unlimited |
| `"maxQueueSize"` | integer | No | `0` (unlimited) | Max concurrent requests; excess rejected with 503 |
| `"logFile"` | string | No | — | Path to the log file |
| `"maxLogSize"` | integer | No | `1` (GB) | Max log file size in GB. `0` = unlimited |

### Example config.json

```json
{
  "model": "trained_model.nnmodel",
  "port": 8080,
  "poolSize": 4,
  "maxBodySize": 20,
  "logFile": "/var/log/nn-server.log",
  "maxLogSize": 2
}
```

## Model Package Format (.nnmodel)

NN-Server loads `.nnmodel` packages (tar archives containing `model.json` + `params.bin`). To create a `.nnmodel` package from a trained model:

```bash
tar -cf model.nnmodel model.json params.bin
```

To extract:

```bash
tar -xf model.nnmodel
```

## Install

NN-Server ships an interactive installer that clones the monorepo, builds the
server (release), interviews you for `config.json`, and optionally installs a
systemd service. Run it with a one-liner:

```bash
curl -fsSL https://raw.githubusercontent.com/naschpitz/NN/main/NN-Server/install.sh | bash
```

The installer:

- Clones `https://github.com/naschpitz/NN.git` and runs `NN-Server/build.sh --release`
- Asks for an install path (default `~/nn-server`), or accepts it as an argument
- Interviews you for each `config.json` field with described defaults (fresh installs only)
- Tracks the install in `~/.config/nn-server/install.state` (version + paths) so
  re-running the script detects the existing install and offers an update when a
  newer release tag is available
- Offers to install/refresh the `nn-server.service` systemd unit

If you prefer to pipe through `less` first, or pass an install path:

```bash
curl -fsSL https://raw.githubusercontent.com/naschpitz/NN/main/NN-Server/install.sh | less
curl -fsSL https://raw.githubusercontent.com/naschpitz/NN/main/NN-Server/install.sh | bash -s -- /opt/nn-server
```

To update an existing install, just re-run the one-liner — the installer detects
the prior install and offers to update (or rebuild) in place.

## Build Instructions

### From repo root

```bash
./build.sh            # static Qt6
./build.sh shared     # shared Qt6
```

### Standalone

```bash
cd NN-Server
./build.sh
```

## Dependencies

| Dependency | Purpose | Included |
|------------|---------|----------|
| [CNN](../CNN/) | Convolutional network implementation | Local (monorepo) |
| [ANN](../ANN/) | Dense network implementation | Transitive (via CNN) |
| [OpenCLWrapper](../OpenCLWrapper/) | OpenCL GPU abstraction | Git submodule |
| Qt6 (Core, Concurrent, Network) | TCP server, threading, file I/O | System |
| nlohmann/json | JSON parsing | Header-only (bundled) |
| stb | Image loading, writing, resizing | Header-only (libs/stb/) |

## License

See [LICENSE.md](LICENSE.md) for details.
