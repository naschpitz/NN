# Architecture Diagram: Component Integration

## Overall Pipeline

```
Input Tensor
    ↓
┌─────────────────────────────────────────────────────────────┐
│                    CNN Forward Pass                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Conv2D::propagate()  →  ReLU  →  Pool  →  BatchNorm      │
│      ↓                                          ↓           │
│  [numFilters × H × W]                  [C × H × W]         │
│                                                             │
│  Flatten  →  Dense (ANN)  →  Output                        │
│      ↓              ↓              ↓                        │
│  [C×H×W]    [neurons]      [numOutputs]                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
    ↓
Loss Calculation (CostFunction::calculateLoss)
    ↓
    L = Loss value
    ↓
┌─────────────────────────────────────────────────────────────┐
│                   CNN Backward Pass                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  dL/dOutput  ←  Dense (ANN)  ←  Flatten  ←  BatchNorm      │
│      ↓              ↓              ↓              ↓         │
│  [numOutputs]  [neurons]      [C×H×W]    [C×H×W]          │
│                                                             │
│  Pool  ←  ReLU  ←  Conv2D::backpropagate()                 │
│      ↓        ↓              ↓                             │
│  [C×H×W]  [C×H×W]  dFilters, dBiases                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
    ↓
Gradient Accumulation & Parameter Update
```

---

## Component Interaction

```
┌──────────────────────────────────────────────────────────────┐
│                      Worker<T>                               │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  propagate(input, training)                         │   │
│  │  ├─ Conv2D::propagate()                             │   │
│  │  ├─ ReLU::propagate()                               │   │
│  │  ├─ Pool::propagate()                               │   │
│  │  ├─ BatchNorm::propagate()                          │   │
│  │  ├─ Flatten::propagate()                            │   │
│  │  └─ ANN::propagate()                                │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  calculateLoss(predicted, expected)                 │   │
│  │  ├─ CostFunctionType::SQUARED_DIFFERENCE            │   │
│  │  ├─ CostFunctionType::WEIGHTED_SQUARED_DIFFERENCE   │   │
│  │  └─ CostFunctionType::CROSS_ENTROPY                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  backpropagate(dOutput)                             │   │
│  │  ├─ ANN::backpropagate()                            │   │
│  │  ├─ Flatten::backpropagate()                        │   │
│  │  ├─ BatchNorm::backpropagate()                      │   │
│  │  ├─ Pool::backpropagate()                           │   │
│  │  ├─ ReLU::backpropagate()                           │   │
│  │  └─ Conv2D::backpropagate()                         │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Data Flow: Batch Normalization

```
Training Mode:
  Input [C × H × W]
    ↓
  Compute batch mean/var (per channel, over spatial dims)
    ↓
  Normalize: x̂ = (x - μ) / √(σ² + ε)
    ↓
  Scale & Shift: y = γ·x̂ + β
    ↓
  Update running stats: running = (1-m)·running + m·batch
    ↓
  Output [C × H × W]
  Store: batchMean, batchVar, xNormalized (for backprop)

Inference Mode:
  Input [C × H × W]
    ↓
  Normalize using running stats: x̂ = (x - running_μ) / √(running_σ² + ε)
    ↓
  Scale & Shift: y = γ·x̂ + β
    ↓
  Output [C × H × W]

Backward Pass:
  dOutput [C × H × W]
    ↓
  Compute dγ = Σ dOutput · x̂
  Compute dβ = Σ dOutput
    ↓
  Compute dInput = (γ·invStd/N) · (N·dOutput - dβ - x̂·dγ)
    ↓
  dInput [C × H × W]
```

---

## Data Flow: Convolution

```
Forward Pass:
  Input [inputC × inputH × inputW]
  Filters [numFilters × inputC × filterH × filterW]
  Biases [numFilters]
    ↓
  For each output position (f, oh, ow):
    z = bias[f] + Σ_c Σ_kh Σ_kw input[c, oh·sy+kh-py, ow·sx+kw-px] · filter[f,c,kh,kw]
    ↓
  Output [numFilters × outH × outW]

Backward Pass:
  dOutput [numFilters × outH × outW]
  Input [inputC × inputH × inputW]
    ↓
  dFilters = Σ dOutput[f,oh,ow] · input[c, ih, iw]
  dBiases = Σ_oh Σ_ow dOutput[f, oh, ow]
  dInput = Σ_f Σ_kh Σ_kw dOutput[f,oh,ow] · filter[f,c,kh,kw]
    ↓
  dFilters [numFilters × inputC × filterH × filterW]
  dBiases [numFilters]
  dInput [inputC × inputH × inputW]
```

---

## Data Flow: Cost Functions

```
Forward Pass:
  Predicted [numOutputs]
  Expected [numOutputs]
  Weights [numOutputs] (optional)
    ↓
  SQUARED_DIFFERENCE:
    L = (1/N) Σ w_i · (pred_i - exp_i)²
  
  WEIGHTED_SQUARED_DIFFERENCE:
    L = (1/N) Σ w_i · (pred_i - exp_i)²
  
  CROSS_ENTROPY:
    L = -Σ w_i · exp_i · log(max(pred_i, 1e-7))
    ↓
  Loss (scalar)

Backward Pass (Last Layer):
  dOutput [numOutputs]
    ↓
  MSE:
    dL/da_i = 2·w_i·(a_i - y_i)
  
  Cross-Entropy + Softmax:
    dL/da_i = w_i·(a_i - y_i)
    ↓
  dInput [numOutputs]
```

---

## Parameter Storage

```
Parameters<T>
├── convParams: std::vector<ConvParameters<T>>
│   ├── [0]: ConvParameters for layer 0
│   │   ├── filters [numFilters × inputC × filterH × filterW]
│   │   └── biases [numFilters]
│   ├── [1]: ConvParameters for layer 1
│   └── ...
│
├── bnParams: std::vector<BatchNormParameters<T>>
│   ├── [0]: BatchNormParameters for layer 0
│   │   ├── gamma [numChannels]
│   │   ├── beta [numChannels]
│   │   ├── runningMean [numChannels]
│   │   └── runningVar [numChannels]
│   ├── [1]: BatchNormParameters for layer 1
│   └── ...
│
└── denseParams: ANN::Parameters<T>
    ├── weights [layers]
    └── biases [layers]
```

