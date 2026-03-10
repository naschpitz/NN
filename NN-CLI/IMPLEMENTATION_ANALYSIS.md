# Implementation Analysis: Batch Norm, Convolution, and Cost Functions

## 1. BATCH NORMALIZATION

### Location
- **Header**: `extern/CNN/CNN_BatchNorm.hpp`
- **Implementation**: `extern/CNN/CNN_BatchNorm.cpp`
- **Parameters**: `extern/CNN/CNN_BatchNormParameters.hpp`
- **Config**: `extern/CNN/CNN_LayersConfig.hpp` (BatchNormLayerConfig)

### API Signatures

```cpp
// Forward pass (training or inference)
static Tensor3D<T> propagate(
    const Tensor3D<T>& input,
    const Shape3D& inputShape,
    BatchNormParameters<T>& params,
    const BatchNormLayerConfig& config,
    bool training = false,
    std::vector<T>* batchMean = nullptr,
    std::vector<T>* batchVar = nullptr,
    Tensor3D<T>* xNormalized = nullptr
);

// Backward pass
static Tensor3D<T> backpropagate(
    const Tensor3D<T>& dOutput,
    const Shape3D& inputShape,
    const BatchNormParameters<T>& params,
    const BatchNormLayerConfig& config,
    const std::vector<T>& batchMean,
    const std::vector<T>& batchVar,
    const Tensor3D<T>& xNormalized,
    std::vector<T>& dGamma,
    std::vector<T>& dBeta
);
```

### Parameters Structure
```cpp
template <typename T>
struct BatchNormParameters {
    std::vector<T> gamma;        // Scale [numChannels]
    std::vector<T> beta;         // Shift [numChannels]
    std::vector<T> runningMean;  // Running mean [numChannels]
    std::vector<T> runningVar;   // Running variance [numChannels]
    ulong numChannels = 0;
};
```

### Config Structure
```cpp
struct BatchNormLayerConfig {
    float epsilon = 1e-5f;   // Numerical stability
    float momentum = 0.1f;   // Running stats update rate
};
```

### Forward Pass Logic
- **Training**: Computes batch mean/var over spatial dims, normalizes, scales (gamma), shifts (beta), updates running stats
- **Inference**: Uses running mean/var instead of batch statistics
- Stores normalized values (xNormalized) for backprop

### Backward Pass Logic
- Computes dGamma and dBeta via reduction over spatial dimensions
- Computes dInput using full batch norm gradient formula: `(gamma * invStd / N) * (N * dOut - db - xn * dg)`

---

## 2. CONVOLUTION (Conv2D)

### Location
- **Header**: `extern/CNN/CNN_Conv2D.hpp`
- **Implementation**: `extern/CNN/CNN_Conv2D.cpp`
- **Parameters**: `extern/CNN/CNN_Conv2DParameters.hpp`
- **Config**: `extern/CNN/CNN_LayersConfig.hpp` (ConvLayerConfig)

### API Signatures

```cpp
// Forward pass
static Tensor3D<T> propagate(
    const Tensor3D<T>& input,
    const ConvLayerConfig& config,
    const ConvParameters<T>& params
);

// Backward pass
static Tensor3D<T> backpropagate(
    const Tensor3D<T>& dOut,
    const Tensor3D<T>& input,
    const ConvLayerConfig& config,
    const ConvParameters<T>& params,
    std::vector<T>& dFilters,
    std::vector<T>& dBiases
);
```

### Parameters Structure
```cpp
template <typename T>
struct ConvParameters {
    std::vector<T> filters;  // Flat: [numFilters][inputC][filterH][filterW]
    std::vector<T> biases;   // [numFilters]
    ulong numFilters = 0;
    ulong inputC = 0;
    ulong filterH = 0;
    ulong filterW = 0;

    // Access: filterAt(f, c, h, w)
};
```

### Config Structure
```cpp
struct ConvLayerConfig {
    ulong numFilters;
    ulong filterH;
    ulong filterW;
    ulong strideY;
    ulong strideX;
    SlidingStrategyType slidingStrategy;  // VALID or SAME
};
```

### Forward Pass Logic
- For each output position (f, oh, ow):
  - Initialize with bias[f]
  - Slide filter over input with stride
  - Apply zero-padding based on SlidingStrategy
  - Accumulate: sum += input[c,ih,iw] * filter[f,c,kh,kw]

### Backward Pass Logic
- **dFilters**: Accumulate dOut[f,oh,ow] * input[c,ih,iw]
- **dBiases**: Sum dOut over spatial dimensions
- **dInput**: Accumulate dOut[f,oh,ow] * filter[f,c,kh,kw]

---

## 3. COST FUNCTIONS

### Location
- **ANN**: `extern/CNN/extern/ANN/ANN_CostFunctionConfig.hpp`
- **CNN**: `extern/CNN/CNN_CostFunctionConfig.hpp`
- **Implementation**: `extern/CNN/extern/ANN/ANN_Worker.cpp` and `extern/CNN/CNN_Worker.cpp`

### Available Cost Functions

```cpp
enum class CostFunctionType : int {
    SQUARED_DIFFERENCE = 0,
    WEIGHTED_SQUARED_DIFFERENCE = 1,
    CROSS_ENTROPY = 2
};
```

### Config Structure
```cpp
template <typename T>
struct CostFunctionConfig {
    CostFunctionType type = CostFunctionType::SQUARED_DIFFERENCE;
    std::vector<T> weights;  // Per-output weights (for WEIGHTED variants)
};
```

### String Conversion Helpers
```cpp
CostFunction::nameToType(const std::string& name)
CostFunction::typeToName(CostFunctionType type)
```

### Loss Calculations

**SQUARED_DIFFERENCE**: `L = Σ(predicted - expected)² / N`

**WEIGHTED_SQUARED_DIFFERENCE**: `L = Σ w_i * (predicted - expected)² / N`

**CROSS_ENTROPY**: `L = -Σ w_i * expected_i * log(max(predicted_i, 1e-7))`

### Backward Pass (Last Layer Gradient)
- **MSE**: `dL/da_j = 2 * w_j * (a_j - y_j)`
- **Cross-Entropy + Softmax**: `dL/da_j = w_j * (a_j - y_j)`

---

## 4. INTEGRATION WITH RUNNER/WORKER

### CNN Worker Usage

**File**: `extern/CNN/CNN_Worker.cpp`

#### Forward Pass Integration
```cpp
// In CoreCPUWorker::propagate()
case LayerType::BATCHNORM: {
    const auto& bn = std::get<BatchNormLayerConfig>(layerConfig.config);
    BatchNormParameters<T> bnParams = this->sharedParams.bnParams[bnIdx];

    if (training) {
        // Store intermediates for backprop
        this->bnBatchMeans.push_back({});
        this->bnBatchVars.push_back({});
        this->bnXNormalized.push_back({});
        current = BatchNorm<T>::propagate(current, current.shape, bnParams, bn,
                                         true, &this->bnBatchMeans.back(),
                                         &this->bnBatchVars.back(),
                                         &this->bnXNormalized.back());
    } else {
        current = BatchNorm<T>::propagate(current, current.shape, bnParams, bn);
    }
    bnIdx++;
    break;
}
```

#### Loss Calculation
```cpp
// In Worker<T>::calculateLoss()
T loss = 0;
switch (this->costFunctionConfig.type) {
case CostFunctionType::CROSS_ENTROPY:
    // L = -sum(w_i * y_i * log(a_i))
    for (ulong i = 0; i < expected.size(); i++) {
        T pred = std::max(predicted[i], epsilon);
        T weight = (!this->costFunctionConfig.weights.empty())
                   ? this->costFunctionConfig.weights[i] : 1;
        loss -= weight * expected[i] * std::log(pred);
    }
    break;

case CostFunctionType::SQUARED_DIFFERENCE:
case CostFunctionType::WEIGHTED_SQUARED_DIFFERENCE:
default:
    // L = sum(w_i * (a_i - y_i)^2) / N
    for (ulong i = 0; i < expected.size(); i++) {
        T diff = predicted[i] - expected[i];
        T weight = (!this->costFunctionConfig.weights.empty())
                   ? this->costFunctionConfig.weights[i] : 1;
        loss += weight * diff * diff;
    }
    loss /= expected.size();
    break;
}
return loss;
```

### Parameter Initialization

**File**: `extern/CNN/CNN_Parameters.hpp`

```cpp
template <typename T>
struct Parameters {
    std::vector<ConvParameters<T>> convParams;      // One per conv layer
    std::vector<BatchNormParameters<T>> bnParams;   // One per batch norm layer
    ANN::Parameters<T> denseParams;                 // Dense layer parameters
};
```

### Testing Through Runner

To test these components in isolation or through the full pipeline:

1. **Create CoreConfig** with desired layers
2. **Set CostFunctionConfig** type and weights
3. **Initialize Parameters** with random/specific values
4. **Call propagate()** for forward pass
5. **Call backpropagate()** for gradient computation
6. **Verify loss** and gradient shapes

Example test pattern from `test_layers.cpp`:
```cpp
// Setup
CNN::Shape3D shape{1, 1, 4};
CNN::Tensor3D<double> input(shape);
CNN::BatchNormParameters<double> params;
CNN::BatchNormLayerConfig config;

// Forward (training)
std::vector<double> batchMean, batchVar;
CNN::Tensor3D<double> xNorm;
auto output = CNN::BatchNorm<double>::propagate(
    input, shape, params, config, true,
    &batchMean, &batchVar, &xNorm
);

// Backward
std::vector<double> dGamma, dBeta;
auto dInput = CNN::BatchNorm<double>::backpropagate(
    dOutput, shape, params, config,
    batchMean, batchVar, xNorm, dGamma, dBeta
);
```

---

## Testing Integration

All three components are tested in:
- `extern/CNN/tests/test_layers.cpp` (BatchNorm, Conv2D)
- `extern/CNN/extern/ANN/tests/test_core.cpp` (Cost functions)

Tests cover:
- Forward/backward pass correctness
- Gradient computation
- Parameter updates
- Integration through full pipeline

