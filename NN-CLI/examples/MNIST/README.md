# MNIST Handwritten Digit Recognition Example

This example demonstrates training, testing, and running predict on the MNIST dataset using ANN-CLI.

## Overview

The MNIST dataset contains 70,000 handwritten digit images (0-9):
- **Training set**: 60,000 images (28×28 pixels = 784 features)
- **Test set**: 10,000 images

The neural network architecture defined in `mnist_config.json`:
- **Input layer**: 784 neurons (28×28 pixel values, normalized 0-1)
- **Hidden layer 1**: 128 neurons (ReLU activation)
- **Hidden layer 2**: 64 neurons (ReLU activation)
- **Output layer**: 10 neurons (Sigmoid activation, one per digit 0-9)

## Directory Structure

```
MNIST/
├── mnist_config.json           # Network architecture and training config
├── train/
│   ├── train-images.idx3-ubyte # Training images (IDX format)
│   ├── train-labels.idx1-ubyte # Training labels (IDX format)
│   └── output/                 # Trained models saved here
├── test/
│   ├── t10k-images.idx3-ubyte  # Test images (IDX format)
│   └── t10k-labels.idx1-ubyte  # Test labels (IDX format)
└── predict/
    ├── mnist_digit_2_input.json # Sample input for predict
    └── output/                  # Predict results saved here
```

## Usage

### 1. Training

Train the neural network using the MNIST training dataset:

```bash
# Train on GPU (recommended)
ANN-CLI --config mnist_config.json \
        --mode train \
        --device gpu \
        --idx-data train/train-images.idx3-ubyte \
        --idx-labels train/train-labels.idx1-ubyte \
        --output train/output/trained_model.json \
        --verbose

# Train on CPU
ANN-CLI --config mnist_config.json \
        --mode train \
        --device cpu \
        --idx-data train/train-images.idx3-ubyte \
        --idx-labels train/train-labels.idx1-ubyte \
        --output train/output/trained_model.json \
        --verbose
```

Training output will show progress with loss values:
```
Epoch 100/1000 | Sample 60000/60000 | Loss: 0.0689 | ETA: 2m 30s
```

### 2. Testing/Evaluation

Evaluate the trained model on the test dataset:

```bash
# Test on GPU
ANN-CLI --config train/output/trained_model.json \
        --mode test \
        --device gpu \
        --idx-data test/t10k-images.idx3-ubyte \
        --idx-labels test/t10k-labels.idx1-ubyte \
        --verbose

# Test on CPU
ANN-CLI --config train/output/trained_model.json \
        --mode test \
        --device cpu \
        --idx-data test/t10k-images.idx3-ubyte \
        --idx-labels test/t10k-labels.idx1-ubyte \
        --verbose
```

Test output shows evaluation metrics:
```
Test Results:
  Samples evaluated: 10000
  Total loss:        156.7
  Average loss:      0.01567
```

### 3. Running Predict

Run predict on a single input image:

```bash
# Predict on GPU
ANN-CLI --config train/output/trained_model.json \
        --mode predict \
        --device gpu \
        --input predict/mnist_digit_2_input.json \
        --output predict/output/result.json \
        --verbose

# Predict on CPU
ANN-CLI --config train/output/trained_model.json \
        --mode predict \
        --device cpu \
        --input predict/mnist_digit_2_input.json \
        --verbose
```

The output will be a JSON file with the network's prediction:
```json
{
  "predictMetadata": {
    "startTime": "2025-02-20T10:30:00",
    "endTime": "2025-02-20T10:30:00",
    "durationSeconds": 0.0012,
    "durationFormatted": "1.2ms"
  },
  "output": [0.01, 0.02, 0.95, 0.01, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0]
}
```

The output array represents probabilities for digits 0-9. The highest value indicates the predicted digit (in this example, index 2 has value 0.95, so the prediction is digit "2").

## Tips

- **GPU vs CPU**: GPU training is significantly faster (10-100x) for large datasets
- **Learning rate**: The default 0.01 works well; lower values (0.001) may improve accuracy but take longer
- **Epochs**: More epochs generally improve accuracy until overfitting occurs; monitor test loss
- **Progress reports**: Set `progressReports` at the root level of the config to control how often progress is displayed (used across loading, training, testing, and predicting)
- **Model checkpoints**: Set `saveModelInterval` at the root level to save a checkpoint every N epochs during training (default 10; set to 0 to disable)

