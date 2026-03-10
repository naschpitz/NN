# Quick Reference: Batch Norm, Conv2D, Cost Functions

## Batch Normalization Quick Start

**Headers to include:**
```cpp
#include "CNN_BatchNorm.hpp"
#include "CNN_BatchNormParameters.hpp"
#include "CNN_LayersConfig.hpp"
```

**Forward pass (training):**
```cpp
std::vector<T> batchMean, batchVar;
Tensor3D<T> xNormalized;
Tensor3D<T> output = BatchNorm<T>::propagate(
    input, inputShape, params, config, 
    true,  // training=true
    &batchMean, &batchVar, &xNormalized
);
```

**Forward pass (inference):**
```cpp
Tensor3D<T> output = BatchNorm<T>::propagate(
    input, inputShape, params, config
    // training=false (default)
);
```

**Backward pass:**
```cpp
std::vector<T> dGamma, dBeta;
Tensor3D<T> dInput = BatchNorm<T>::backpropagate(
    dOutput, inputShape, params, config,
    batchMean, batchVar, xNormalized,
    dGamma, dBeta
);
```

---

## Convolution Quick Start

**Headers to include:**
```cpp
#include "CNN_Conv2D.hpp"
#include "CNN_Conv2DParameters.hpp"
#include "CNN_LayersConfig.hpp"
```

**Forward pass:**
```cpp
Tensor3D<T> output = Conv2D<T>::propagate(input, config, params);
```

**Backward pass:**
```cpp
std::vector<T> dFilters, dBiases;
Tensor3D<T> dInput = Conv2D<T>::backpropagate(
    dOutput, input, config, params,
    dFilters, dBiases
);
```

**Access filter weights:**
```cpp
T weight = params.filterAt(filterIdx, channelIdx, heightIdx, widthIdx);
```

---

## Cost Functions Quick Start

**Headers to include:**
```cpp
#include "CNN_CostFunctionConfig.hpp"  // or ANN_CostFunctionConfig.hpp
```

**Available types:**
- `CostFunctionType::SQUARED_DIFFERENCE` (default)
- `CostFunctionType::WEIGHTED_SQUARED_DIFFERENCE`
- `CostFunctionType::CROSS_ENTROPY`

**Setup config:**
```cpp
CostFunctionConfig<double> costConfig;
costConfig.type = CostFunctionType::CROSS_ENTROPY;
costConfig.weights = {1.0, 1.0, 1.0};  // per-output weights
```

**String conversion:**
```cpp
auto type = CostFunction::nameToType("crossEntropy");
auto name = CostFunction::typeToName(CostFunctionType::CROSS_ENTROPY);
```

**Calculate loss (in Worker):**
```cpp
T loss = worker.calculateLoss(predicted, expected);
```

---

## Key Data Structures

### BatchNormParameters<T>
- `gamma`: Scale per channel [numChannels]
- `beta`: Shift per channel [numChannels]
- `runningMean`: Accumulated mean [numChannels]
- `runningVar`: Accumulated variance [numChannels]

### ConvParameters<T>
- `filters`: Flat array [numFilters × inputC × filterH × filterW]
- `biases`: Per-filter bias [numFilters]
- `filterAt(f, c, h, w)`: Access helper

### CostFunctionConfig<T>
- `type`: Which cost function to use
- `weights`: Per-output weights (optional)

---

## Common Patterns

**Initialize BatchNorm params:**
```cpp
params.numChannels = C;
params.gamma.assign(C, 1.0);      // Initialize to 1
params.beta.assign(C, 0.0);       // Initialize to 0
params.runningMean.assign(C, 0.0);
params.runningVar.assign(C, 1.0);
```

**Initialize Conv params:**
```cpp
params.numFilters = numFilters;
params.inputC = inputChannels;
params.filterH = kH;
params.filterW = kW;
params.filters.resize(numFilters * inputC * kH * kW);
params.biases.resize(numFilters);
// Fill with random/specific values
```

**Test pattern:**
```cpp
// 1. Create input tensor
Tensor3D<T> input(shape);
// 2. Setup parameters
ConvParameters<T> params;
// 3. Setup config
ConvLayerConfig config{...};
// 4. Forward pass
auto output = Conv2D<T>::propagate(input, config, params);
// 5. Create gradient
Tensor3D<T> dOutput(output.shape);
// 6. Backward pass
std::vector<T> dFilters, dBiases;
auto dInput = Conv2D<T>::backpropagate(dOutput, input, config, params, dFilters, dBiases);
```

