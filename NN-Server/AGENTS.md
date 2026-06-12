# NN-Server — HTTP Inference Server

QTcpServer-based HTTP server for serving ANN/CNN model predictions over REST. Loads a trained model into a thread-safe CorePool and handles concurrent requests.

## Build

```bash
./build.sh            # from repo root
./NN-Server/build.sh  # standalone
```

Binary: `NN-Server/build/NN-Server`. Tests: `test_endpoints` (integration), `test_logger` (unit).

## Dependencies

- Qt6 Core + Network + Concurrent
- CNN (PUBLIC — transitively provides ANN + OpenCLWrapper)
- nlohmann/json (header-only, in `libs/nlohmann/`)
- stb (header-only, in `libs/stb/`)
- NN-CLI sources: `NN-CLI_ModelPackage.cpp`, `NN-CLI_ModelSerializer.cpp`, `NN-CLI_DataType.cpp` (shared with NN-CLI)

## Usage

```bash
./NN-Server/build/NN-Server [config.json]
# Defaults to config.json in current directory if no argument
```

## Configuration

JSON config (fields in config.json or passed as CLI argument):

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `"model"` | string | Yes | — | Path to `.nnmodel` or `.nnmodel.tar` |
| `"port"` | int | No | `8080` | TCP port (1–65535) |
| `"poolSize"` | int | No | CPU cores | Core instances in pool |
| `"maxBodySize"` | int | No | `10` | Max request body in MB (0 = unlimited) |
| `"maxQueueSize"` | int | No | `0` (unlimited) | Max concurrent requests (503 when exceeded) |
| `"logFile"` | string | No | — | Log file path |
| `"maxLogSize"` | int | No | `1` (GB) | Max log file size (0 = unlimited) |

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | `{"status":"ok"}` |
| `POST` | `/predict` | JSON vector or image input → prediction output |
| *any* | *other* | 404 `{"error":"Not found. Use POST /predict"}` |

### POST /predict — JSON

```bash
curl -X POST http://localhost:8080/predict \
     -H "Content-Type: application/json" \
     -d '{"input": [0.1, 0.5, 0.9, 0.3]}'
```

Response: `{"output": [...], "logits": [...]}`

### POST /predict — Image

```bash
curl -X POST http://localhost:8080/predict \
     -H "Content-Type: image/jpeg" \
     --data-binary @photo.jpg
```

Image output models return raw PNG.

## Architecture

```
HTTP Client
    ↓
HttpServer (QTcpServer + QThreadPool)
    ↓
RequestHandler (QRunnable) — acquires Core, runs predict, releases
    ↓
CorePool (thread-safe pool of ANN::Core<float> / CNN::Core<float>)
    ↓
ANN / CNN predict()
```

| Component | Base Class | Role |
|-----------|-----------|------|
| `HttpServer` | `QTcpServer` | Accept TCP connections, read HTTP |
| `RequestHandler` | `QRunnable` | Dispatched via `QThreadPool` |
| `CorePool` | — | Thread-safe pool (QMutex + QWaitCondition) |
| `ImageLoader` | — | Decode JPEG/PNG → NCHW float arrays |
| `Loader` | — | Load model JSON, auto-detect ANN vs CNN |
| `Logger` | — | Rotating log file (circular overwrite) |

## Model package format

`.nnmodel` = `tar` archive containing `model.json` + `params.bin`.
Create: `tar -cf model.nnmodel model.json params.bin`
Extract: `tar -xf model.nnmodel`

## CorePool internals

- Each entry: `ANN::Core<float>` or `CNN::Core<float>` + `available` flag
- `acquire()` — returns first available core (blocks if none)
- `release()` — marks available, wakes waiters via `QWaitCondition`
- Pool size defaults to CPU core count

## Testing

```bash
cd NN-Server/build
./test_endpoints    # integration (starts server, sends HTTP requests)
./test_logger       # unit (tests Logger directly)
```

## Gotchas

- NN-Server reuses NN-CLI's ModelPackage and ModelSerializer directly (cross-component source)
- Model auto-detection: `.nnmodel` JSON specifies `networkType` (ANN/CNN)
- CorePool blocks on `acquire()` when all cores are busy — no request dropping
- Logger uses circular write: when file hits `maxSizeBytes`, seeks to start and overwrites
