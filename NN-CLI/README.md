# NN-CLI

A command-line interface for training, testing, and predicting with Neural Networks (ANN and CNN).

The network type is **auto-detected** from the configuration file: if the config contains a `"layers"` key, it is treated as an ANN; otherwise as a CNN.

## Building

```bash
mkdir build && cd build
cmake ..
make
```

## Usage

```bash
# Training
NN-CLI --config <config_file> --mode train [options]

# Running predict
NN-CLI --config <model_file> --mode predict --input <input_file> [options]

# Testing/evaluation
NN-CLI --config <model_file> --mode test --samples <samples_file> [options]
```

### Options

| Option | Short | Description |
|--------|-------|-------------|
| `--config` | `-c` | Path to JSON configuration/model file (required) |
| `--mode` | `-m` | Mode: `train`, `predict`, or `test` (overrides config file) |
| `--device` | `-d` | Device: `cpu` or `gpu` (overrides config file) |
| `--input` | `-i` | Path to JSON file with input values (predict mode) |
| `--input-type` | | Input data type: `vector` or `image` (overrides config file) |
| `--samples` | `-s` | Path to JSON file with samples (for train/test modes) |
| `--idx-data` | | Path to IDX3 data file (alternative to `--samples`) |
| `--idx-labels` | | Path to IDX1 labels file (requires `--idx-data`) |
| `--output` | `-o` | Output file for saving trained model or prediction result |
| `--output-type` | | Output data type: `vector` or `image` (overrides config file) |
| `--num-epochs` | | Number of training epochs (overrides config file) |
| `--log-level` | `-l` | Log level: `quiet`, `error`, `warning`, `info`, `debug` (default: `error`) |
| `--help` | `-h` | Show help message |

### Modes

- **train**: Train a neural network using `--config` and samples, outputs a trained model file.
- **predict**: Run predict using `--config` (trained model) with one or more inputs in parallel (across threads on CPU, across GPUs on GPU). Output order matches input order.
- **test**: Evaluate a trained model (`--config`) on test samples and report the loss.
- **calibrate**: Pick a free-energy out-of-distribution threshold. Requires `--id-images` and optionally `--ood-dir`. Options: `--id-images` (required, directory of ID images), `--ood-dir` (OOD root, auto-populated if empty), `--id-sample-count` (default 500), `--ood-sample-count` (default 1500), `--id-percentile` (default 95), `--no-fetch` (disable auto-download), `--output` (default: `<config_dir>/threshold.json`)

## ANN Configuration

### ANN Config File (Train Mode)

```json
{
  "mode": "train",
  "device": "cpu",
  "numThreads": 0,
  "numGPUs": 0,
  "progressReports": 1000,
  "saveModelInterval": 10,
  "inputType": "vector",
  "outputType": "vector",
  "layers": [
    { "numNeurons": 784, "actvFunc": "none" },
    { "numNeurons": 128, "actvFunc": "relu" },
    { "numNeurons": 64, "actvFunc": "relu" },
    { "numNeurons": 10, "actvFunc": "sigmoid" }
  ],
  "costFunction": {
    "type": "weightedSquaredDifference",
    "weights": [1.0, 1.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
  },
  "training": {
    "numEpochs": 100,
    "batchSize": 64,
    "learningRate": 0.01,
    "dropoutRate": 0.3,
    "augmentationFactor": 2,
    "balanceAugmentation": true,
    "fullAugmentation": true,
    "autoClassWeights": true,
    "augmentationProbability": 0.5,
    "augmentationTransforms": {
      "horizontalFlip": true,
      "rotation": 15.0,
      "translation": 0.1,
      "brightness": 0.1,
      "contrast": 0.2,
      "gaussianNoise": 0.02
    }
  },
  "test": {
    "batchSize": 64
  }
}
```

### ANN Config File (Predict/Test Mode)

```json
{
  "mode": "predict",
  "device": "gpu",
  "inputType": "vector",
  "outputType": "vector",
  "layers": [
    { "numNeurons": 784, "actvFunc": "none" },
    { "numNeurons": 128, "actvFunc": "relu" },
    { "numNeurons": 64, "actvFunc": "relu" },
    { "numNeurons": 10, "actvFunc": "sigmoid" }
  ],
  "parameters": {
    "weights": [...],
    "biases": [...]
  }
}
```

#### ANN Top-Level Fields

- `mode`: Operation mode (optional, default: `predict`) — *can be overridden by `--mode`*
- `device`: Execution device (optional, default: `cpu`) — *can be overridden by `--device`*
- `numThreads`: Number of CPU threads for CPU mode (optional, default: `0` = all available cores)
- `numGPUs`: Number of GPU devices for GPU mode (optional, default: `0` = all available GPUs)
- `progressReports`: Progress update frequency for all modes (optional, default: `1000`)
- `saveModelInterval`: Save a checkpoint every N epochs during training (optional, default: `10`; `0` = disabled)
- `inputType`: Input data type — `"vector"` (default) or `"image"` — *can be overridden by `--input-type`*
- `outputType`: Output data type — `"vector"` (default) or `"image"` — *can be overridden by `--output-type`*
- `inputShape`: Input image dimensions (`c`, `h`, `w`) — required when `inputType` is `"image"`
- `outputShape`: Output image dimensions (`c`, `h`, `w`) — required when `outputType` is `"image"`

#### ANN Cost Function Configuration (`costFunction`)

Optional object placed between `layers` and `training`. Controls the cost function used during training:

- `type`: `"squaredDifference"` (default) or `"weightedSquaredDifference"`
- `weights`: Array of per-output-neuron weights (required when type is `"weightedSquaredDifference"`). Each weight multiplies the squared difference for the corresponding output neuron, allowing rare classes to receive higher penalty.

If omitted, the default `squaredDifference` loss is used (equivalent to standard MSE).

#### ANN Layers Configuration

- `numNeurons`: Number of neurons in the layer
- `actvFunc`: Activation function (`none`, `relu`, `sigmoid`, `tanh`)

#### ANN Model Configuration

- `numEpochs`: Number of training epochs
- `batchSize`: Mini-batch size (default: 64)
- `learningRate`: Learning rate for gradient descent
- `shuffleSamples`: Shuffle sample order each epoch (default: `true`)
- `dropoutRate`: Dropout probability for hidden layers (default: `0.0` = disabled). Uses inverted dropout — activations are scaled by 1/(1−p) during training, no adjustment at inference
- `augmentationFactor`: Multiply each class by N× using random transforms (default: `0` = disabled). NN-CLI applies transforms before passing samples to the library
- `balanceAugmentation`: Oversample minority classes up to the majority class count (default: `false`). When combined with `augmentationFactor`, the balanced count is also multiplied. Only the oversampled (extra) copies are augmented; the original samples are kept as-is
- `fullAugmentation`: Apply augmentation transforms to **all** training samples every epoch — not just the oversampled copies — for stronger regularisation (default: `false`). Independent of, and composable with, `balanceAugmentation`/`augmentationFactor` (validation samples are never augmented)
- `autoClassWeights`: Auto-compute inverse-frequency class weights and set `weightedSquaredDifference` cost function (default: `false`). Only applies when no manual `costFunction.weights` are specified
- `augmentationProbability`: Probability of applying each enabled transform per augmented sample (default: `0.5` = 50% chance)
- `augmentationTransforms`: Object controlling individual augmentation transforms. Numeric values control intensity; set to `0` to disable. `horizontalFlip` is a boolean (no intensity parameter). Defaults shown below:

  | Transform | Type | Default | Meaning | Disabled |
  |---|---|---|---|---|
  | `horizontalFlip` | `bool` | `true` | Mirror along vertical axis | `false` |
  | `rotation` | `float` | `15.0` | ±15° max rotation | `0` |
  | `translation` | `float` | `0.1` | ±10% max shift | `0` |
  | `brightness` | `float` | `0.1` | ±0.1 max delta | `0` |
  | `contrast` | `float` | `0.2` | range 0.8–1.2× (delta from 1.0) | `0` |
  | `gaussianNoise` | `float` | `0.02` | σ=0.02 noise stddev | `0` |
  | `randomErasing` | `float` | `0.0` | Max area fraction to erase (cutout) | `0` |
  | `hueShift` | `float` | `0.0` | Max hue shift (fraction of 360°) | `0` |
  | `scaling` | `float` | `0.0` | Max zoom deviation from 1.0× | `0` |
  | `elasticDeformation` | `object` | — | Elastic deformation config | `alpha: 0` |
  | `elasticDeformation.alpha` | `float` | `0.0` | Deformation intensity | `0` |
  | `elasticDeformation.sigma` | `float` | `5.0` | Deformation smoothness | — |

- `validation`: Object controlling train/validation split for overfitting detection. A held-out portion of training samples is evaluated after each epoch (or every N epochs) and the validation loss is reported alongside training loss. The split is stratified (preserves class distribution) and deterministic.

  | Field | Type | Default | Description |
  |---|---|---|---|
  | `enabled` | `bool` | `true` | Enable validation split |
  | `autoSize` | `bool` | `true` | Auto-select split ratio based on dataset size |
  | `size` | `float` | `0.15` | Fixed validation fraction (0.0–1.0). Only used when `autoSize` is `false` |
  | `checkInterval` | `int` | `1` | Run validation every N epochs (1 = every epoch) |

  Auto-size ratios: <1k samples → 20%, 1k–10k → 15%, 10k–100k → 10%, 100k–1M → 2%, >1M → 1%.
  To disable validation: `"validation": { "enabled": false }`

- `monitoring`: Object controlling training health monitoring and early stopping. Tracks loss trends and stops training when no progress is being made. When validation is enabled, the monitor uses validation loss; otherwise it uses training loss. Saves the best model to `output/best_model.json` (overwritten on each new best).

  | Field | Type | Default | Description |
  |---|---|---|---|
  | `enabled` | `bool` | `false` | Enable training monitoring |
  | `checkInterval` | `int` | `5` | Evaluate health metrics every N epochs |
  | `patience` | `int` | `20` | Stop after this many consecutive check intervals without improvement |
  | `metrics.lossStagnation.enabled` | `bool` | `true` | Track whether loss is improving |
  | `metrics.lossStagnation.minDelta` | `float` | `0.0001` | Minimum loss decrease to count as improvement |
  | `metrics.lossExplosion.enabled` | `bool` | `true` | Detect loss blowing up (NaN or sudden spike) |
  | `metrics.lossExplosion.threshold` | `float` | `10.0` | Stop if loss exceeds this × best loss |

  Example — enable monitoring with custom patience:
  ```json
  "monitoring": {
      "enabled": true,
      "patience": 30
  }
  ```

  Example — only rotation (strong) and translation, nothing else:

  ```json
  "augmentationTransforms": {
    "horizontalFlip": false,
    "rotation": 30.0,
    "translation": 0.15,
    "brightness": 0,
    "contrast": 0,
    "gaussianNoise": 0
  }
  ```

#### ANN Test Configuration

- `batchSize`: Mini-batch size for test evaluation (default: `64`). Controls how many samples are loaded into memory at once during `--mode test`

## CNN Configuration

### CNN Config File (Train Mode)

```json
{
  "mode": "train",
  "device": "cpu",
  "numThreads": 0,
  "numGPUs": 0,
  "progressReports": 1000,
  "saveModelInterval": 10,
  "inputType": "vector",
  "outputType": "vector",
  "inputShape": { "c": 1, "h": 28, "w": 28 },
  "convolutionalLayers": [
    { "type": "conv", "numFilters": 8, "filterH": 3, "filterW": 3, "strideY": 1, "strideX": 1, "slidingStrategy": "valid" },
    { "type": "relu" },
    { "type": "pool", "poolType": "max", "poolH": 2, "poolW": 2, "strideY": 2, "strideX": 2 },
    { "type": "flatten" }
  ],
  "denseLayers": [
    { "numNeurons": 128, "actvFunc": "relu" },
    { "numNeurons": 10, "actvFunc": "sigmoid" }
  ],
  "costFunction": {
    "type": "weightedSquaredDifference",
    "weights": [1.0, 1.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
  },
  "training": {
    "numEpochs": 10,
    "batchSize": 64,
    "learningRate": 0.01,
    "dropoutRate": 0.3,
    "augmentationFactor": 2,
    "balanceAugmentation": true,
    "fullAugmentation": true,
    "autoClassWeights": true,
    "augmentationProbability": 0.5,
    "augmentationTransforms": {
      "horizontalFlip": true,
      "rotation": 15.0
    }
  },
  "test": {
    "batchSize": 64
  }
}
```

#### CNN Top-Level Fields

- `mode`: Operation mode (optional, default: `predict`) — *can be overridden by `--mode`*
- `device`: Execution device (optional, default: `cpu`) — *can be overridden by `--device`*
- `numThreads`: Number of CPU threads for CPU mode (optional, default: `0` = all available cores)
- `numGPUs`: Number of GPU devices for GPU mode (optional, default: `0` = all available GPUs)
- `progressReports`: Progress update frequency for all modes (optional, default: `1000`)
- `saveModelInterval`: Save a checkpoint every N epochs during training (optional, default: `10`; `0` = disabled)
- `inputType`: Input data type — `"vector"` (default) or `"image"` — *can be overridden by `--input-type`*
- `outputType`: Output data type — `"vector"` (default) or `"image"` — *can be overridden by `--output-type`*
- `inputShape`: Input tensor dimensions (`c` channels, `h` height, `w` width)
- `outputShape`: Output image dimensions (`c`, `h`, `w`) — required when `outputType` is `"image"`

#### CNN Cost Function Configuration (`costFunction`)

Same as ANN — optional object placed between `denseLayers` and `training`:

- `type`: `"squaredDifference"` (default) or `"weightedSquaredDifference"`
- `weights`: Array of per-output-neuron weights (required when type is `"weightedSquaredDifference"`)

#### CNN Layers Configuration (`convolutionalLayers`)

Each layer has a `type` field:

- **conv**: Convolutional layer
  - `numFilters`, `filterH`, `filterW`, `strideY`, `strideX`, `slidingStrategy` (`valid` or `same`)
- **relu**: ReLU activation layer
- **pool**: Pooling layer
  - `poolType` (`max` or `avg`), `poolH`, `poolW`, `strideY`, `strideX`
- **flatten**: Flatten 3D feature maps to 1D vector

#### Dense Layers Configuration (`denseLayers`)

- `numNeurons`: Number of neurons in the layer
- `actvFunc`: Activation function (`relu`, `sigmoid`, `tanh`)

#### CNN Model Configuration

- `numEpochs`: Number of training epochs
- `batchSize`: Mini-batch size (default: 64)
- `learningRate`: Learning rate for gradient descent
- `shuffleSamples`: Shuffle sample order each epoch (default: `true`)
- `dropoutRate`: Dropout probability for dense hidden layers (default: `0.0` = disabled). Convolutional layers are not affected
- `augmentationFactor`: Multiply each class by N× using random image transforms (default: `0` = disabled)
- `balanceAugmentation`: Oversample minority classes up to the majority class count (default: `false`). Only the oversampled copies are augmented; originals are kept as-is
- `fullAugmentation`: Apply augmentation to all training samples every epoch (not just the oversampled copies), for stronger regularisation (default: `false`). Composable with `balanceAugmentation`/`augmentationFactor`
- `autoClassWeights`: Auto-compute inverse-frequency class weights (default: `false`)
- `augmentationProbability`: Probability of applying each enabled transform (default: `0.5`)
- `augmentationTransforms`: Control individual transforms (same fields as ANN — see above for defaults)
- `validation`: Train/validation split for overfitting detection (same fields as ANN — see above for details)
- `monitoring`: Training health monitoring and early stopping (same fields as ANN — see above for details)

#### CNN Test Configuration

- `batchSize`: Mini-batch size for test evaluation (default: `64`). Controls how many samples are loaded into memory at once during `--mode test`

## Model File (output from training)

The trained model file contains the network architecture and learned parameters. This file is generated by `--mode train` and can be used directly with `--config` for `--mode predict` and `--mode test`.

## Model Package Format (.nnmodel)

NN-CLI uses `.nnmodel` packages (tar archives containing `model.json` + `params.bin`) for non-train modes. Plain JSON configs are rejected for predict/test/calibrate — only `.nnmodel` or `.nnmodel.tar` files are accepted.

The `NN-CLI_ModelPackage::isPackage()` function checks for `.nnmodel` or `.nnmodel.tar` extensions.

To create a `.nnmodel` package from a trained model:

```bash
tar -cf model.nnmodel model.json params.bin
```

To extract:

```bash
tar -xf model.nnmodel
```

## Samples File (JSON format)

Training samples with input/output pairs. Values can be numeric vectors or image file paths (when `inputType`/`outputType` is `"image"`):

**Vector format** (default):

```json
{
  "samples": [
    {
      "input": [0.0, 0.5, 1.0, 0.75],
      "output": [1.0, 0.0]
    },
    {
      "input": [1.0, 0.25, 0.0, 0.5],
      "output": [0.0, 1.0]
    }
  ]
}
```

**Image format** (when `inputType` and/or `outputType` is `"image"`):

```json
{
  "samples": [
    {
      "input": "images/cat_01.png",
      "output": [1.0, 0.0]
    },
    {
      "input": "images/dog_01.png",
      "output": [0.0, 1.0]
    }
  ]
}
```

Image paths can be absolute or relative to the samples file location. Images are automatically loaded, resized to match `inputShape` (or `outputShape`), normalised to [0, 1], and converted to NCHW layout.

## Input File (for predict mode)

The input file uses an `"inputs"` array to support batch predictions (one or more inputs in a single run).

**Vector format** (default):

```json
{
  "inputs": [
    [0.0, 0.5, 1.0, 0.75],
    [1.0, 0.25, 0.0, 0.5]
  ]
}
```

**Image format** (when `inputType` is `"image"`):

```json
{
  "inputs": ["photo1.png", "photo2.png"]
}
```

## Predict Output

When `outputType` is `"vector"` (default), the output is a JSON file with an `"outputs"` array (one entry per input) and batch metadata:

```json
{
  "predictMetadata": {
    "startTime": "2026-02-22T10:30:00-03:00",
    "endTime": "2026-02-22T10:30:01-03:00",
    "durationSeconds": 0.123,
    "durationFormatted": "0s",
    "numInputs": 2
  },
  "outputs": [
    [0.95, 0.05],
    [0.10, 0.90]
  ]
}
```

When `outputType` is `"image"`, the prediction outputs are saved as numbered PNG images (0.png, 1.png, ...) inside a folder instead of a JSON file.

## IDX File Format

As an alternative to JSON samples, you can use IDX format files (commonly used for MNIST and similar datasets):

- **IDX3**: Multi-dimensional data (e.g., images)
- **IDX1**: Labels

The data is automatically normalized to 0-1 range and labels are one-hot encoded. For CNN configs, the IDX image data is automatically reshaped to match the `inputShape` specified in the config.

## Examples

### ANN: Training with JSON samples

```bash
NN-CLI --config ann_config.json --mode train --samples training_data.json --output trained_model.json
```

### ANN: Training with IDX files (MNIST)

```bash
NN-CLI --config examples/MNIST/mnist_ann_config.json --mode train --idx-data train-images-idx3-ubyte --idx-labels train-labels-idx1-ubyte
```

### CNN: Training with IDX files (MNIST)

```bash
NN-CLI --config examples/MNIST/mnist_cnn_config.json --mode train --idx-data train-images-idx3-ubyte --idx-labels train-labels-idx1-ubyte
```

### Training on GPU

```bash
NN-CLI --config config.json --mode train --device gpu --samples training_data.json
```

### Running predict

```bash
NN-CLI --config trained_model.json --mode predict --input test_input.json
```

### Testing a model

```bash
NN-CLI --config trained_model.json --mode test --samples test_data.json
```

### Testing with IDX files

```bash
NN-CLI --config trained_model.json --mode test --idx-data t10k-images-idx3-ubyte --idx-labels t10k-labels-idx1-ubyte
```

### Training with image files

```bash
NN-CLI --config config.json --mode train --input-type image --samples image_samples.json
```

### Predicting with image input and output

```bash
NN-CLI --config trained_model.json --mode predict --input-type image --output-type image --input input.json
```

## Image Support

NN-CLI supports image file paths in samples and input JSON files. Images are automatically loaded, resized, normalised to [0, 1], and converted to NCHW layout. Supported read formats: JPEG, PNG, BMP, GIF, TGA, PSD, HDR, PIC. Supported write formats: PNG, JPEG, BMP.

Set `"inputType": "image"` and/or `"outputType": "image"` in the config JSON or use the `--input-type` / `--output-type` CLI options. When using image input for ANN, an `inputShape` with `c`, `h`, `w` must be provided. When using image output, an `outputShape` must be provided.

Image loading uses the [stb](https://github.com/nothings/stb) header-only library (bundled in `libs/stb/`).

## License

See [LICENSE.md](LICENSE.md) for details.

