# NN-Server — HTTP Inference Server

QTcpServer-based HTTP server for serving ANN/CNN model predictions over REST. Loads a trained model into a thread-safe CorePool and handles concurrent requests.

## Build

```bash
./build.sh            # from repo root
./NN-Server/build.sh  # standalone
```

Binary: `NN-Server/build/NN-Server`. Tests: `test_endpoints` (integration), `test_logger` (unit).

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
