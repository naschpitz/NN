# ANN - Neural Network Library

A C++ neural network library supporting both CPU and GPU (OpenCL) execution.

## Features

- Multi-layer perceptron (MLP) neural networks
- CPU and GPU execution via OpenCL
- Training, predict, and evaluation modes
- JSON-based configuration and model serialization
- Multiple activation functions

## Dependencies

- Qt Core and Qt Concurrent
- [OpenCLWrapper](https://github.com/naschpitz/OpenCLWrapper) (for GPU support)
- [nlohmann/json](https://github.com/nlohmann/json)

## Building

```bash
mkdir build && cd build
cmake ..
make
```

## Configuration File Format

### Train Mode

```json
{
  "mode": "train",
  "device": "cpu",
  "layersConfig": [
    { "numNeurons": 784, "actvFunc": "none" },
    { "numNeurons": 128, "actvFunc": "relu" },
    { "numNeurons": 10, "actvFunc": "sigmoid" }
  ],
  "trainingConfig": {
    "numEpochs": 100,
    "learningRate": 0.01,
    "numThreads": 0
  }
}
```

### Predict Mode

```json
{
  "mode": "predict",
  "device": "gpu",
  "layersConfig": [
    { "numNeurons": 784, "actvFunc": "none" },
    { "numNeurons": 128, "actvFunc": "relu" },
    { "numNeurons": 10, "actvFunc": "sigmoid" }
  ],
  "parameters": {
    "weights": [...],
    "biases": [...]
  }
}
```

### Test Mode

```json
{
  "mode": "test",
  "device": "cpu",
  "layersConfig": [
    { "numNeurons": 784, "actvFunc": "none" },
    { "numNeurons": 128, "actvFunc": "relu" },
    { "numNeurons": 10, "actvFunc": "sigmoid" }
  ],
  "parameters": {
    "weights": [...],
    "biases": [...]
  }
}
```

## Configuration Fields

| Field | Description | Required |
|-------|-------------|----------|
| `mode` | Operation mode: `train`, `predict`, or `test` | Optional (default: `predict`) |
| `device` | Execution device: `cpu` or `gpu` | Optional (default: `cpu`) |
| `layersConfig` | Array of layer definitions | Required |
| `trainingConfig` | Training hyperparameters | Required for `train` mode |
| `parameters` | Trained weights and biases | Required for `predict`/`test` modes |

## Activation Functions

| Function | Description |
|----------|-------------|
| `none` | No activation (use for input layer) |
| `relu` | Rectified Linear Unit |
| `sigmoid` | Sigmoid function |
| `tanh` | Hyperbolic tangent |

## Output File Formats

### Trained Model Output (from training)

After training, the model file contains the network architecture and learned parameters:

```json
{
  "layersConfig": [
    { "numNeurons": 784, "actvFunc": "none" },
    { "numNeurons": 128, "actvFunc": "relu" },
    { "numNeurons": 10, "actvFunc": "sigmoid" }
  ],
  "parameters": {
    "weights": [
      [[...], [...], ...],
      [[...], [...], ...]
    ],
    "biases": [
      [...],
      [...]
    ]
  },
  "trainingMetadata": {
    "startTime": "2025-02-20T10:30:00",
    "endTime": "2025-02-20T10:45:30",
    "durationSeconds": 930.5,
    "durationFormatted": "15m 30.5s",
    "numEpochs": 100,
    "learningRate": 0.01,
    "numSamples": 60000,
    "finalLoss": 0.0234
  }
}
```

### Predict Output

The predict output contains the predict metadata and the network output:

```json
{
  "predictMetadata": {
    "startTime": "2025-02-20T11:00:00",
    "endTime": "2025-02-20T11:00:00",
    "durationSeconds": 0.0012,
    "durationFormatted": "1.2ms"
  },
  "output": [0.01, 0.02, 0.95, 0.01, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0]
}
```

### Test/Evaluation Output

Test mode returns evaluation metrics:

```json
{
  "testMetadata": {
    "startTime": "2025-02-20T11:05:00",
    "endTime": "2025-02-20T11:05:45",
    "durationSeconds": 45.3,
    "durationFormatted": "45.3s"
  },
  "results": {
    "numSamples": 10000,
    "totalLoss": 234.5,
    "averageLoss": 0.02345
  }
}
```

## Input File Formats

### Samples File (for training/testing)

```json
{
  "samples": [
    {
      "input": [0.0, 0.5, 1.0, 0.75, ...],
      "output": [1.0, 0.0, 0.0, ...]
    },
    {
      "input": [1.0, 0.25, 0.0, 0.5, ...],
      "output": [0.0, 1.0, 0.0, ...]
    }
  ]
}
```

### Input File (for predict)

```json
{
  "input": [0.0, 0.5, 1.0, 0.75, ...]
}
```

## License

See [LICENSE.md](LICENSE.md) for details.

