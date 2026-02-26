#include "test_helpers.hpp"

static void testMissingConfig() {
  std::cout << "  testMissingConfig... ";

  auto result = runNNCLI({"--mode", "train"});

  CHECK(result.exitCode == 1, "Missing config: exit code 1");
  CHECK(result.stdErr.contains("Error: --config is required."),
        "Missing config: error message");
  std::cout << std::endl;
}

static void testInvalidMode() {
  std::cout << "  testInvalidMode... ";

  auto result = runNNCLI({
    "--config", fixturePath("ann_train_config.json"),
    "--mode", "invalid"
  });

  CHECK(result.exitCode == 1, "Invalid mode: exit code 1");
  CHECK(result.stdErr.contains("Error: Mode must be 'train', 'predict', or 'test'."),
        "Invalid mode: error message");
  std::cout << std::endl;
}

static void testInvalidDevice() {
  std::cout << "  testInvalidDevice... ";

  auto result = runNNCLI({
    "--config", fixturePath("ann_train_config.json"),
    "--mode", "train",
    "--device", "tpu"
  });

  CHECK(result.exitCode == 1, "Invalid device: exit code 1");
  CHECK(result.stdErr.contains("Error: Device must be 'cpu' or 'gpu'."),
        "Invalid device: error message");
  std::cout << std::endl;
}

static void testMissingSamplesANN() {
  std::cout << "  testMissingSamplesANN... ";

  auto result = runNNCLI({
    "--config", fixturePath("ann_train_config.json"),
    "--mode", "train",
    "--device", "cpu"
  });

  CHECK(result.exitCode == 1, "Missing samples ANN: exit code 1");
  CHECK(result.stdErr.contains("requires either --samples (JSON) or --idx-data and --idx-labels (IDX)"),
        "Missing samples ANN: error message");
  std::cout << std::endl;
}

static void testMissingSamplesCNN() {
  std::cout << "  testMissingSamplesCNN... ";

  auto result = runNNCLI({
    "--config", fixturePath("cnn_train_config.json"),
    "--mode", "train",
    "--device", "cpu"
  });

  CHECK(result.exitCode == 1, "Missing samples CNN: exit code 1");
  CHECK(result.stdErr.contains("requires either --samples (JSON) or --idx-data and --idx-labels (IDX)"),
        "Missing samples CNN: error message");
  std::cout << std::endl;
}

static void testPredictWithoutInput() {
  std::cout << "  testPredictWithoutInput... ";

  if (trainedANNModelPath.isEmpty() || !QFile::exists(trainedANNModelPath)) {
    CHECK(false, "Predict without input: skipped — no trained model available (testANNTrainXOR must run first)");
    std::cout << std::endl;
    return;
  }

  // Must use a config with trained parameters so predict path is reached
  auto result = runNNCLI({
    "--config", trainedANNModelPath,
    "--mode", "predict",
    "--device", "cpu"
  });

  CHECK(result.exitCode == 1, "Predict without input: exit code 1");
  CHECK(result.stdErr.contains("--input option is required for predict mode"),
        "Predict without input: error message");
  std::cout << std::endl;
}

static void testIdxWithoutLabels() {
  std::cout << "  testIdxWithoutLabels... ";

  auto result = runNNCLI({
    "--config", fixturePath("ann_train_config.json"),
    "--mode", "train",
    "--device", "cpu",
    "--idx-data", examplePath("MNIST/train/train-images.idx3-ubyte")
  });

  CHECK(result.exitCode == 1, "IDX without labels: exit code 1");
  CHECK(result.stdErr.contains("--idx-labels is required when using --idx-data"),
        "IDX without labels: error message");
  std::cout << std::endl;
}

static void testBothSamplesAndIdx() {
  std::cout << "  testBothSamplesAndIdx... ";

  auto result = runNNCLI({
    "--config", fixturePath("ann_train_config.json"),
    "--mode", "train",
    "--device", "cpu",
    "--samples", fixturePath("ann_train_samples.json"),
    "--idx-data", examplePath("MNIST/train/train-images.idx3-ubyte"),
    "--idx-labels", examplePath("MNIST/train/train-labels.idx1-ubyte")
  });

  CHECK(result.exitCode == 1, "Both samples and IDX: exit code 1");
  CHECK(result.stdErr.contains("Cannot use both --samples and --idx-data"),
        "Both samples and IDX: error message");
  std::cout << std::endl;
}

static void testInvalidActvFuncANN() {
  std::cout << "  testInvalidActvFuncANN... ";

  auto result = runNNCLI({
    "--config", fixturePath("ann_invalid_actvfunc_config.json"),
    "--mode", "train",
    "--device", "cpu",
    "--samples", fixturePath("ann_train_samples.json")
  });

  CHECK(result.exitCode == 1, "Invalid actvFunc ANN: exit code 1");
  CHECK(result.stdErr.contains("Unknown activation function"),
        "Invalid actvFunc ANN: error message contains 'Unknown activation function'");
  std::cout << std::endl;
}

static void testInvalidActvFuncCNN() {
  std::cout << "  testInvalidActvFuncCNN... ";

  auto result = runNNCLI({
    "--config", fixturePath("cnn_invalid_actvfunc_config.json"),
    "--mode", "train",
    "--device", "cpu",
    "--samples", fixturePath("cnn_train_samples.json")
  });

  CHECK(result.exitCode == 1, "Invalid actvFunc CNN: exit code 1");
  CHECK(result.stdErr.contains("Unknown activation function"),
        "Invalid actvFunc CNN: error message contains 'Unknown activation function'");
  std::cout << std::endl;
}

static void testInvalidCostFuncANN() {
  std::cout << "  testInvalidCostFuncANN... ";

  auto result = runNNCLI({
    "--config", fixturePath("ann_invalid_costfunc_config.json"),
    "--mode", "train",
    "--device", "cpu",
    "--samples", fixturePath("ann_train_samples.json")
  });

  CHECK(result.exitCode == 1, "Invalid costFunc ANN: exit code 1");
  CHECK(result.stdErr.contains("Unknown cost function"),
        "Invalid costFunc ANN: error message contains 'Unknown cost function'");
  std::cout << std::endl;
}

void runErrorTests() {
  testMissingConfig();
  testInvalidMode();
  testInvalidDevice();
  testMissingSamplesANN();
  testMissingSamplesCNN();
  testPredictWithoutInput();
  testIdxWithoutLabels();
  testBothSamplesAndIdx();
  testInvalidActvFuncANN();
  testInvalidActvFuncCNN();
  testInvalidCostFuncANN();
}

