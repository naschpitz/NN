# File Locations and Test Examples

## File Locations

### Batch Normalization
```
extern/CNN/CNN_BatchNorm.hpp              - Class definition
extern/CNN/CNN_BatchNorm.cpp              - CPU implementation
extern/CNN/CNN_BatchNormParameters.hpp    - Parameter structure
extern/CNN/CNN_LayersConfig.hpp           - BatchNormLayerConfig
extern/CNN/CNN_BatchNorm.cpp.cl           - OpenCL kernels (CPU)
extern/CNN/opencl/CNN_BatchNorm.cpp.cl    - OpenCL kernels (GPU)
extern/CNN/tests/test_layers.cpp          - Tests
```

### Convolution
```
extern/CNN/CNN_Conv2D.hpp                 - Class definition
extern/CNN/CNN_Conv2D.cpp                 - CPU implementation
extern/CNN/CNN_Conv2DParameters.hpp       - Parameter structure
extern/CNN/CNN_LayersConfig.hpp           - ConvLayerConfig
extern/CNN/opencl/CNN_Propagate.cpp.cl    - Forward pass kernel
extern/CNN/opencl/CNN_Backpropagate.cpp.cl - Backward pass kernels
extern/CNN/tests/test_conv2d.cpp          - Tests
```

### Cost Functions
```
extern/CNN/CNN_CostFunctionConfig.hpp     - CNN cost function enum
extern/CNN/extern/ANN/ANN_CostFunctionConfig.hpp - ANN cost function enum
extern/CNN/CNN_Worker.cpp                 - CNN loss calculation
extern/CNN/extern/ANN/ANN_Worker.cpp      - ANN loss calculation
extern/CNN/opencl/ANN_Loss.cpp.cl         - GPU loss kernel
extern/CNN/tests/test_core.cpp            - Cost function tests
```

### Integration
```
extern/CNN/CNN_Worker.hpp                 - Worker class
extern/CNN/CNN_CoreCPUWorker.cpp          - CPU worker implementation
extern/CNN/CNN_Parameters.hpp             - All parameters struct
```

---

## Test Examples

### Batch Norm Test (from test_layers.cpp)

```cpp
static void testBatchNormBackpropagate() {
    // Setup
    CNN::Shape3D shape{1, 1, 4};
    CNN::Tensor3D<double> input(shape);
    input.data = {1.0, 2.0, 3.0, 4.0};
    
    CNN::BatchNormParameters<double> params;
    params.numChannels = 1;
    params.gamma = {2.0};
    params.beta = {0.5};
    params.runningMean = {0.0};
    params.runningVar = {1.0};
    
    CNN::BatchNormLayerConfig config;
    config.epsilon = 0.0;
    config.momentum = 0.1;
    
    // Forward (training)
    std::vector<double> batchMean, batchVar;
    CNN::Tensor3D<double> xNorm;
    auto output = CNN::BatchNorm<double>::propagate(
        input, shape, params, config, true,
        &batchMean, &batchVar, &xNorm
    );
    
    // Create gradient
    CNN::Tensor3D<double> dOutput(shape);
    dOutput.data = {0.1, 0.2, 0.3, 0.4};
    
    // Backward
    std::vector<double> dGamma, dBeta;
    auto dInput = CNN::BatchNorm<double>::backpropagate(
        dOutput, shape, params, config,
        batchMean, batchVar, xNorm, dGamma, dBeta
    );
    
    // Verify shapes
    assert(dInput.shape == shape);
    assert(dGamma.size() == 1);
    assert(dBeta.size() == 1);
}
```

### Conv2D Test (from test_conv2d.cpp)

```cpp
static void testConv2DMultiFilter() {
    // Input: 1 channel, 3x3
    CNN::Tensor3D<double> input({1, 3, 3});
    input.data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    
    // Config: 2 filters, 2x2 kernel, stride 1, valid padding
    CNN::ConvLayerConfig config{2, 2, 2, 1, 1, CNN::SlidingStrategyType::VALID};
    
    // Parameters
    CNN::ConvParameters<double> params;
    params.numFilters = 2;
    params.inputC = 1;
    params.filterH = 2;
    params.filterW = 2;
    params.filters = {1, 1, 1, 1, -1, -1, -1, -1};  // 2 filters
    params.biases = {0.0, 0.0};
    
    // Forward
    auto output = CNN::Conv2D<double>::propagate(input, config, params);
    
    // Verify output shape
    assert(output.shape.c == 2);  // 2 filters
    assert(output.shape.h == 2);  // (3-2)/1+1 = 2
    assert(output.shape.w == 2);
    
    // Backward
    CNN::Tensor3D<double> dOutput(output.shape);
    dOutput.data.assign(4, 0.1);  // 2x2 gradient
    
    std::vector<double> dFilters, dBiases;
    auto dInput = CNN::Conv2D<double>::backpropagate(
        dOutput, input, config, params, dFilters, dBiases
    );
    
    assert(dInput.shape == input.shape);
    assert(dFilters.size() == 8);  // 2 filters × 1 channel × 2×2
    assert(dBiases.size() == 2);
}
```

### Cost Function Test (from test_core.cpp)

```cpp
static void testCostFunctionConfigGetter() {
    ANN::CoreConfig<double> config;
    config.modeType = ANN::ModeType::PREDICT;
    config.deviceType = ANN::DeviceType::CPU;
    config.layersConfig = makeLayersConfig(
        {{2, ANN::ActvFuncType::RELU}, {2, ANN::ActvFuncType::SIGMOID}}
    );
    config.costFunctionConfig.type = ANN::CostFunctionType::WEIGHTED_SQUARED_DIFFERENCE;
    config.costFunctionConfig.weights = {3.0, 0.5};
    
    auto core = ANN::Core<double>::makeCore(config);
    
    const auto& cfc = core->getCostFunctionConfig();
    assert(cfc.type == ANN::CostFunctionType::WEIGHTED_SQUARED_DIFFERENCE);
    assert(cfc.weights.size() == 2);
    assert(cfc.weights[0] == 3.0);
    assert(cfc.weights[1] == 0.5);
}
```

---

## Running Tests

```bash
# Build tests
cd /home/naschpitz/QtProjects/CNN
mkdir -p build && cd build
cmake ..
make

# Run specific test
./test_layers
./test_conv2d
./test_core
```

## Key Test Patterns

1. **Create input tensor** with known values
2. **Setup parameters** (filters, biases, gamma, beta, etc.)
3. **Setup config** (strides, padding, epsilon, momentum)
4. **Forward pass** - verify output shape and values
5. **Create gradient** with known values
6. **Backward pass** - verify gradient shapes
7. **Assert correctness** - check numerical values if possible

