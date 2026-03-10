# Mathematical Details: Batch Norm, Conv2D, Cost Functions

## Batch Normalization Mathematics

### Forward Pass (Training)

For each channel `c`:

1. **Compute batch statistics** (over spatial dimensions H×W):
   - `μ_c = (1/N) Σ x[c,h,w]` where N = H×W
   - `σ²_c = (1/N) Σ (x[c,h,w] - μ_c)²`

2. **Normalize**:
   - `x̂[c,h,w] = (x[c,h,w] - μ_c) / √(σ²_c + ε)`

3. **Scale and shift**:
   - `y[c,h,w] = γ_c · x̂[c,h,w] + β_c`

4. **Update running statistics** (exponential moving average):
   - `running_mean_c ← (1 - m) · running_mean_c + m · μ_c`
   - `running_var_c ← (1 - m) · running_var_c + m · σ²_c`
   - where m = momentum (default 0.1)

### Forward Pass (Inference)

Uses running statistics instead of batch statistics:
- `y[c,h,w] = γ_c · (x[c,h,w] - running_mean_c) / √(running_var_c + ε) + β_c`

### Backward Pass

For each channel `c`:

1. **Gradient w.r.t. scale and shift**:
   - `dγ_c = Σ dy[c,h,w] · x̂[c,h,w]`
   - `dβ_c = Σ dy[c,h,w]`

2. **Gradient w.r.t. input**:
   - `dx[c,h,w] = (γ_c · invStd / N) · (N · dy[c,h,w] - dβ_c - x̂[c,h,w] · dγ_c)`
   - where invStd = 1 / √(σ²_c + ε)

---

## Convolution Mathematics

### Forward Pass

For each output position (f, oh, ow):

```
z[f,oh,ow] = b_f + Σ_c Σ_kh Σ_kw input[c, oh·s_y+kh-p_y, ow·s_x+kw-p_x] · w[f,c,kh,kw]
```

Where:
- f = filter index
- oh, ow = output height, width
- s_y, s_x = stride
- p_y, p_x = padding
- kh, kw = kernel height, width
- Zero-padding applied outside input bounds

### Backward Pass

1. **Gradient w.r.t. filters**:
   - `dw[f,c,kh,kw] = Σ_oh Σ_ow dz[f,oh,ow] · input[c, oh·s_y+kh-p_y, ow·s_x+kw-p_x]`

2. **Gradient w.r.t. biases**:
   - `db_f = Σ_oh Σ_ow dz[f,oh,ow]`

3. **Gradient w.r.t. input**:
   - `dinput[c,ih,iw] = Σ_f Σ_kh Σ_kw dz[f,oh,ow] · w[f,c,kh,kw]`
   - where (oh, ow) are positions where (ih, iw) participates

### Output Shape Calculation

```
outH = (inputH + 2·padY - filterH) / strideY + 1
outW = (inputW + 2·padX - filterW) / strideX + 1
```

---

## Cost Functions Mathematics

### Squared Difference (MSE)

**Loss**:
```
L = (1/N) Σ_i w_i · (ŷ_i - y_i)²
```

**Gradient (last layer)**:
```
dL/da_i = (2/N) · w_i · (a_i - y_i)
```

### Weighted Squared Difference

Same as above but with per-output weights w_i.

### Cross-Entropy

**Loss**:
```
L = -Σ_i w_i · y_i · log(max(ŷ_i, ε))
```

where ε = 1e-7 (numerical stability)

**Gradient (with softmax)**:
```
dL/da_i = w_i · (a_i - y_i)
```

Note: This assumes softmax activation on output layer.

---

## Implementation Notes

### Batch Norm Storage

During training, the forward pass stores:
- `batchMean`: Per-channel mean
- `batchVar`: Per-channel variance
- `xNormalized`: Normalized values (before scale/shift)

These are **required** for backpropagation.

### Conv Filter Layout

Filters stored as flat vector in order:
```
[f0_c0_h0_w0, f0_c0_h0_w1, ..., f0_c0_h1_w0, ..., f1_c0_h0_w0, ...]
```

Access via `filterAt(f, c, h, w)` helper.

### Numerical Stability

- **Batch Norm**: epsilon = 1e-5 (prevents division by zero)
- **Cross-Entropy**: epsilon = 1e-7 (prevents log(0))

### Memory Layout

All tensors use **NCHW** format (channels-first):
- Tensor3D: [channels][height][width]
- Tensor1D: Flattened version of Tensor3D

### Gradient Accumulation

In backpropagation:
- `dFilters` and `dBiases` are **accumulated** (not reset)
- Caller must initialize to zero before backprop

