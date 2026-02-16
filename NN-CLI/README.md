# ANN-CLI

A command-line interface for training and running Artificial Neural Networks.

## Building

```bash
mkdir build && cd build
cmake ..
make
```

## Usage

```bash
ANN-CLI --config <file> --mode <train|run> [options]
```

### Options

| Option | Short | Description |
|--------|-------|-------------|
| `--config` | `-c` | Path to JSON configuration file (required) |
| `--mode` | `-m` | Mode: `train` or `run` (required) |
| `--type` | `-t` | Core type: `cpu` or `gpu` (default: `cpu`) |
| `--samples` | `-s` | Path to JSON file with training samples |
| `--idx-data` | | Path to IDX3 data file (alternative to `--samples`) |
| `--idx-labels` | | Path to IDX1 labels file (requires `--idx-data`) |
| `--input` | `-i` | Path to JSON file with input values for run mode |
| `--output` | `-o` | Output file for saving trained model |
| `--help` | `-h` | Show help message |

## JSON File Formats

### Configuration File

The configuration file defines the neural network architecture and training parameters.

```json
{
  "layersConfig": [
    { "numNeurons": 784, "actvFunc": "relu" },
    { "numNeurons": 128, "actvFunc": "relu" },
    { "numNeurons": 64, "actvFunc": "relu" },
    { "numNeurons": 10, "actvFunc": "sigmoid" }
  ],
  "trainingConfig": {
    "numEpochs": 100,
    "learningRate": 0.01
  }
}
```

#### Layers Configuration

- `numNeurons`: Number of neurons in the layer
- `actvFunc`: Activation function. Available options:
  - `relu` - Rectified Linear Unit
  - `sigmoid` - Sigmoid function
  - `tanh` - Hyperbolic tangent

#### Training Configuration (optional)

- `numEpochs`: Number of training epochs
- `learningRate`: Learning rate for gradient descent

#### Pre-trained Parameters (optional)

For loading pre-trained models:

```json
{
  "layersConfig": [...],
  "parameters": {
    "weights": [...],
    "biases": [...]
  }
}
```

### Samples File (JSON format)

Training samples with input/output pairs:

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

### Input File (for run mode)

```json
{
  "input": [0.0, 0.5, 1.0, 0.75]
}
```

## IDX File Format

As an alternative to JSON samples, you can use IDX format files (commonly used for MNIST and similar datasets):

- **IDX3**: Multi-dimensional data (e.g., images)
- **IDX1**: Labels

The data is automatically normalized to 0-1 range and labels are one-hot encoded.

## Examples

### Training with JSON samples

```bash
ANN-CLI --config model.json --mode train --samples training_data.json --output trained_model.json
```

### Training with IDX files

```bash
ANN-CLI --config model.json --mode train --idx-data train-images-idx3-ubyte --idx-labels train-labels-idx1-ubyte --output trained_model.json
```

### Training on GPU

```bash
ANN-CLI --config model.json --mode train --type gpu --samples training_data.json
```

### Running inference

```bash
ANN-CLI --config trained_model.json --mode run --input test_input.json
```

## License

See [LICENSE.md](LICENSE.md) for details.

