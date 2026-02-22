# NN-CLI

A command-line interface for training, testing, and predicting with Neural Networks (ANN and CNN).

The network type is **auto-detected** from the configuration file: if the config contains `inputShape` or `cnnLayersConfig`, it is treated as a CNN; otherwise as an ANN.

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
| `--verbose` | `-v` | Print detailed initialization and processing info |
| `--help` | `-h` | Show help message |

### Modes

- **train**: Train a neural network using `--config` and samples, outputs a trained model file.
- **predict**: Run predict using `--config` (trained model) with a single input.
- **test**: Evaluate a trained model (`--config`) on test samples and report the loss (MSE).

## ANN Configuration

### ANN Config File (Train Mode)

```json
{
  "mode": "train",
  "device": "cpu",
  "inputType": "vector",
  "outputType": "vector",
  "layersConfig": [
    { "numNeurons": 784, "actvFunc": "none" },
    { "numNeurons": 128, "actvFunc": "relu" },
    { "numNeurons": 64, "actvFunc": "relu" },
    { "numNeurons": 10, "actvFunc": "sigmoid" }
  ],
  "trainingConfig": {
    "numEpochs": 100,
    "learningRate": 0.01,
    "numThreads": 0
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
  "layersConfig": [
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
- `inputType`: Input data type — `"vector"` (default) or `"image"` — *can be overridden by `--input-type`*
- `outputType`: Output data type — `"vector"` (default) or `"image"` — *can be overridden by `--output-type`*
- `inputShape`: Input image dimensions (`c`, `h`, `w`) — required when `inputType` is `"image"`
- `outputShape`: Output image dimensions (`c`, `h`, `w`) — required when `outputType` is `"image"`

#### ANN Layers Configuration

- `numNeurons`: Number of neurons in the layer
- `actvFunc`: Activation function (`none`, `relu`, `sigmoid`, `tanh`)

#### ANN Training Configuration

- `numEpochs`: Number of training epochs
- `learningRate`: Learning rate for gradient descent
- `numThreads`: Number of CPU threads (default: 0 = all available cores)

## CNN Configuration

### CNN Config File (Train Mode)

```json
{
  "mode": "train",
  "device": "cpu",
  "inputType": "vector",
  "outputType": "vector",
  "inputShape": { "c": 1, "h": 28, "w": 28 },
  "cnnLayersConfig": [
    { "type": "conv", "numFilters": 8, "filterH": 3, "filterW": 3, "strideY": 1, "strideX": 1, "slidingStrategy": "valid" },
    { "type": "relu" },
    { "type": "pool", "poolType": "max", "poolH": 2, "poolW": 2, "strideY": 2, "strideX": 2 },
    { "type": "flatten" }
  ],
  "denseLayersConfig": [
    { "numNeurons": 128, "actvFunc": "relu" },
    { "numNeurons": 10, "actvFunc": "sigmoid" }
  ],
  "trainingConfig": {
    "numEpochs": 10,
    "learningRate": 0.01
  }
}
```

#### CNN Top-Level Fields

- `mode`: Operation mode (optional, default: `predict`) — *can be overridden by `--mode`*
- `device`: Execution device (optional, default: `cpu`) — *can be overridden by `--device`*
- `inputType`: Input data type — `"vector"` (default) or `"image"` — *can be overridden by `--input-type`*
- `outputType`: Output data type — `"vector"` (default) or `"image"` — *can be overridden by `--output-type`*
- `inputShape`: Input tensor dimensions (`c` channels, `h` height, `w` width)
- `outputShape`: Output image dimensions (`c`, `h`, `w`) — required when `outputType` is `"image"`

#### CNN Layers Configuration (`cnnLayersConfig`)

Each layer has a `type` field:

- **conv**: Convolutional layer
  - `numFilters`, `filterH`, `filterW`, `strideY`, `strideX`, `slidingStrategy` (`valid` or `same`)
- **relu**: ReLU activation layer
- **pool**: Pooling layer
  - `poolType` (`max`), `poolH`, `poolW`, `strideY`, `strideX`
- **flatten**: Flatten 3D feature maps to 1D vector

#### Dense Layers Configuration (`denseLayersConfig`)

- `numNeurons`: Number of neurons in the layer
- `actvFunc`: Activation function (`relu`, `sigmoid`, `tanh`)

#### CNN Training Configuration

- `numEpochs`: Number of training epochs
- `learningRate`: Learning rate for gradient descent

## Model File (output from training)

The trained model file contains the network architecture and learned parameters. This file is generated by `--mode train` and can be used directly with `--config` for `--mode predict` and `--mode test`.

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

**Vector format** (default):

```json
{
  "input": [0.0, 0.5, 1.0, 0.75]
}
```

**Image format** (when `inputType` is `"image"`):

```json
{
  "input": "photo.png"
}
```

## Predict Output

When `outputType` is `"image"`, the prediction output is saved as a PNG image file instead of a JSON vector. When `outputType` is `"vector"` (default), the output is a JSON file with the prediction vector and metadata.

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

