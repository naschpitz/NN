#include "test_helpers.hpp"

static void testMissingConfig()
{
  TestScope _t("testMissingConfig");

  auto result = runNNCLI({"--mode", "train"});

  CHECK(result.exitCode == 1, "Missing config: exit code 1");
  CHECK(result.stdErr.contains("Error: --model or --model-package is required."), "Missing config: error message");
}

static void testInvalidMode()
{
  TestScope _t("testInvalidMode");

  auto result = runNNCLI({"--model", fixturePath("ann_train_config.json"), "--mode", "invalid"});

  CHECK(result.exitCode == 1, "Invalid mode: exit code 1");
  CHECK(result.stdErr.contains("Error: Mode must be 'train', 'predict', 'test', or 'calibrate'."),
        "Invalid mode: error message");
}

static void testInvalidDevice()
{
  TestScope _t("testInvalidDevice");

  auto result = runNNCLI({"--model", fixturePath("ann_train_config.json"), "--mode", "train", "--device", "tpu"});

  CHECK(result.exitCode == 1, "Invalid device: exit code 1");
  CHECK(result.stdErr.contains("Error: Device must be 'cpu' or 'gpu'."), "Invalid device: error message");
  std::cout << std::endl;
}

static void testMissingSamples()
{
  TestScope _t("testMissingSamples");

  auto result = runNNCLI({"--model", fixturePath("ann_train_config.json"), "--mode", "train", "--device", "cpu"});

  CHECK(result.exitCode == 1, "Missing samples : exit code 1");
  CHECK(result.stdErr.contains("requires either --samples (JSON) or --idx-data and --idx-labels (IDX)"),
        "Missing samples : error message");
}

static void testMissingSamplesCNN()
{
  TestScope _t("testMissingSamplesCNN");

  auto result = runNNCLI({"--model", fixturePath("cnn_train_config.json"), "--mode", "train", "--device", "cpu"});

  CHECK(result.exitCode == 1, "Missing samples CNN: exit code 1");
  CHECK(result.stdErr.contains("requires either --samples (JSON) or --idx-data and --idx-labels (IDX)"),
        "Missing samples CNN: error message");
}

static void testPredictWithoutInput()
{
  TestScope _t("testPredictWithoutInput");

  if (trainedModelPath.isEmpty() || !QFile::exists(trainedModelPath)) {
    CHECK(false, "Predict without input: skipped — no trained model available (testTrainXOR must run first)");
    return;
  }

  // Must use a config with trained parameters so predict path is reached
  auto result = runNNCLI({"--model-package", trainedModelPath, "--mode", "predict", "--device", "cpu"});

  CHECK(result.exitCode == 1, "Predict without input: exit code 1");
  CHECK(result.stdErr.contains("--input option is required for predict mode"), "Predict without input: error message");
}

static void testIdxWithoutLabels()
{
  TestScope _t("testIdxWithoutLabels");

  auto result = runNNCLI({"--model", fixturePath("ann_train_config.json"), "--mode", "train", "--device", "cpu",
                          "--idx-data", examplePath("MNIST/train/train-images.idx3-ubyte")});

  CHECK(result.exitCode == 1, "IDX without labels: exit code 1");
  CHECK(result.stdErr.contains("--idx-labels is required when using --idx-data"), "IDX without labels: error message");
}

static void testBothSamplesAndIdx()
{
  TestScope _t("testBothSamplesAndIdx");

  auto result =
    runNNCLI({"--model", fixturePath("ann_train_config.json"), "--mode", "train", "--device", "cpu", "--samples",
              fixturePath("ann_train_samples.json"), "--idx-data", examplePath("MNIST/train/train-images.idx3-ubyte"),
              "--idx-labels", examplePath("MNIST/train/train-labels.idx1-ubyte")});

  CHECK(result.exitCode == 1, "Both samples and IDX: exit code 1");
  CHECK(result.stdErr.contains("Cannot use both --samples and --idx-data"), "Both samples and IDX: error message");
}

static void testInvalidActvFunc()
{
  TestScope _t("testInvalidActvFunc");

  auto result = runNNCLI({"--model", fixturePath("ann_invalid_actvfunc_config.json"), "--mode", "train", "--device",
                          "cpu", "--samples", fixturePath("ann_train_samples.json")});

  CHECK(result.exitCode == 1, "Invalid actvFunc : exit code 1");
  CHECK(result.stdErr.contains("Unknown activation function"),
        "Invalid actvFunc : error message contains 'Unknown activation function'");
}

static void testInvalidActvFuncCNN()
{
  TestScope _t("testInvalidActvFuncCNN");

  auto result = runNNCLI({"--model", fixturePath("cnn_invalid_actvfunc_config.json"), "--mode", "train", "--device",
                          "cpu", "--samples", fixturePath("cnn_train_samples.json")});

  CHECK(result.exitCode == 1, "Invalid actvFunc CNN: exit code 1");
  CHECK(result.stdErr.contains("Unknown activation function"),
        "Invalid actvFunc CNN: error message contains 'Unknown activation function'");
}

static void testInvalidCostFunc()
{
  TestScope _t("testInvalidCostFunc");

  auto result = runNNCLI({"--model", fixturePath("ann_invalid_costfunc_config.json"), "--mode", "train", "--device",
                          "cpu", "--samples", fixturePath("ann_train_samples.json")});

  CHECK(result.exitCode == 1, "Invalid costFunc : exit code 1");
  CHECK(result.stdErr.contains("Unknown cost function"),
        "Invalid costFunc : error message contains 'Unknown cost function'");
}

void runErrorTests()
{
  testMissingConfig();
  testInvalidMode();
  testInvalidDevice();
  testMissingSamples();
  testMissingSamplesCNN();
  testPredictWithoutInput();
  testIdxWithoutLabels();
  testBothSamplesAndIdx();
  testInvalidActvFunc();
  testInvalidActvFuncCNN();
  testInvalidCostFunc();
}
