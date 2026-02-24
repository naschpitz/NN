# ANN - Neural Network Library

A C++ neural network library supporting both CPU and GPU (OpenCL) execution.

## Features

- Multi-layer perceptron (MLP) neural networks
- CPU and GPU execution via OpenCL
- Training, predict, and evaluation modes
- Multiple activation functions
- Pure library — accessible entirely via C++ function calls

## Dependencies

- Qt Core and Qt Concurrent
- [OpenCLWrapper](https://github.com/naschpitz/OpenCLWrapper) (for GPU support)

## Building

```bash
mkdir build && cd build
cmake ..
make
```

## API Overview

ANN is a pure library — all interaction is through C++ types. There is no built-in file I/O;
the calling application (e.g. [NN-CLI](https://github.com/naschpitz/NN-CLI)) handles serialization.

```cpp
#include <ANN_Core.hpp>
#include <ANN_CoreConfig.hpp>
#include <ANN_Sample.hpp>

// Configure
ANN::CoreConfig<float> config;
config.modeType    = ANN::ModeType::TRAIN;
config.deviceType  = ANN::DeviceType::CPU;
config.numThreads  = 0;   // 0 = use all available CPU cores
config.numGPUs     = 0;   // 0 = use all available GPUs (GPU mode)
config.layersConfig = makeLayersConfig({{784, ANN::ActvFuncType::NONE},
                                         {128, ANN::ActvFuncType::RELU},
                                         { 10, ANN::ActvFuncType::SIGMOID}});
config.trainingConfig.numEpochs    = 100;
config.trainingConfig.batchSize    = 64;
config.trainingConfig.learningRate = 0.01f;

// Create, train, and query
auto core = ANN::Core<float>::makeCore(config);
core->train(samples);

const auto& params = core->getParameters();   // weights & biases
const auto& meta   = core->getTrainingMetadata();
```

## Activation Functions

| Function | Description |
|----------|-------------|
| `none` | No activation (use for input layer) |
| `relu` | Rectified Linear Unit |
| `sigmoid` | Sigmoid function |
| `tanh` | Hyperbolic tangent |

## Documentation

Comprehensive HTML documentation is available in the [`docs/`](docs/) directory. Open `docs/index.html` in a browser to browse the API reference, architecture overview, mathematical foundations, and GPU implementation details.

## License

See [LICENSE.md](LICENSE.md) for details.

