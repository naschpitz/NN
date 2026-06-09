#include "test_helpers.hpp"

#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>

QString trainedModelPath;

//===================================================================================================================//

static void testTrainXOR()
{
  std::cout << "  testTrainXOR... ";

  trainedModelPath = tempDir() + "/ann_xor_model.nnmodel.tar";

  auto result = runNNCLI({"--config", fixturePath("ann_train_config.json"), "--mode", "train", "--device", "cpu",
                          "--samples", fixturePath("ann_train_samples.json"), "--output", trainedModelPath});

  CHECK(result.exitCode == 0, " train XOR: exit code 0");
  CHECK(result.stdOut.contains("Training completed."), " train XOR: 'Training completed.'");
  CHECK(result.stdOut.contains("Model saved to:"), " train XOR: 'Model saved to:'");
  CHECK(QFile::exists(trainedModelPath), " train XOR: model file exists");

  // Clear the path if training failed so downstream tests skip gracefully
  if (result.exitCode != 0 || !QFile::exists(trainedModelPath)) {
    trainedModelPath.clear();
  }

  std::cout << std::endl;
}

//===================================================================================================================//

static void testNetworkDetection()
{
  std::cout << "  testNetworkDetection... ";

  if (trainedModelPath.isEmpty() || !QFile::exists(trainedModelPath)) {
    CHECK(false, " detection: skipped — no trained model available (testTrainXOR must run first)");
    std::cout << std::endl;
    return;
  }

  // Create a temporary predict input compatible with XOR model (2 inputs)
  QString predictInputPath = tempDir() + "/ann_detect_input.json";
  QFile inputFile(predictInputPath);

  if (inputFile.open(QIODevice::WriteOnly)) {
    inputFile.write(R"({"inputs": [[0.0, 1.0]]})");
    inputFile.close();
  }

  auto result = runNNCLI({"--config", trainedModelPath, "--mode", "predict", "--device", "cpu", "--input",
                          predictInputPath, "--output", tempDir() + "/ann_detect_output.json", "--log-level", "info"});

  CHECK(result.exitCode == 0, " detection: exit code 0");
  CHECK(result.stdOut.contains("Network type: "), " detection: stdout contains 'Network type: '");
  std::cout << std::endl;
}

//===================================================================================================================//

static void testModeOverride()
{
  std::cout << "  testModeOverride... ";

  if (trainedModelPath.isEmpty() || !QFile::exists(trainedModelPath)) {
    CHECK(false, " mode override: skipped — no trained model available (testTrainXOR must run first)");
    std::cout << std::endl;
    return;
  }

  // Create a temporary predict input compatible with XOR model (2 inputs)
  QString predictInputPath = tempDir() + "/ann_override_input.json";
  QFile inputFile(predictInputPath);

  if (inputFile.open(QIODevice::WriteOnly)) {
    inputFile.write(R"({"inputs": [[0.0, 1.0]]})");
    inputFile.close();
  }

  QString outputPath = tempDir() + "/ann_override_output.json";

  // Trained model has mode=train; override to predict via CLI
  auto result = runNNCLI({"--config", trainedModelPath, "--mode", "predict", "--device", "cpu", "--input",
                          predictInputPath, "--output", outputPath, "--log-level", "info"});

  CHECK(result.exitCode == 0, " mode override: exit code 0");
  CHECK(result.stdOut.contains("Mode: predict (CLI)"), " mode override: 'Mode: predict (CLI)'");
  std::cout << std::endl;
}

//===================================================================================================================//

static void testTrainWithWeightedLoss()
{
  std::cout << "  testTrainWithWeightedLoss... ";

  QString modelPath = tempDir() + "/ann_weighted_model.nnmodel.tar";

  auto result = runNNCLI({"--config", fixturePath("ann_train_weighted_config.json"), "--mode", "train", "--device",
                          "cpu", "--samples", fixturePath("ann_train_samples.json"), "--output", modelPath});

  CHECK(result.exitCode == 0, " weighted train: exit code 0");
  CHECK(result.stdOut.contains("Training completed."), " weighted train: 'Training completed.'");
  CHECK(result.stdOut.contains("Model saved to:"), " weighted train: 'Model saved to:'");
  CHECK(QFile::exists(modelPath), " weighted train: model file exists");

  // Verify saved model JSON contains costFunctionConfig
  QJsonObject root = readModelJsonFromPackage(modelPath);

  if (!root.isEmpty()) {
    CHECK(root.contains("costFunction"), " weighted train: saved model has 'costFunctionConfig'");

    QJsonObject cfc = root["costFunction"].toObject();
    CHECK(cfc["type"].toString() == "weightedSquaredDifference",
          " weighted train: type is 'weightedSquaredDifference'");
    CHECK(cfc.contains("weights"), " weighted train: has 'weights'");

    QJsonArray weights = cfc["weights"].toArray();
    CHECK(weights.size() == 2, " weighted train: weights has 2 elements");
    CHECK_NEAR(weights[0].toDouble(), 3.0, 1e-6, " weighted train: weight[0] = 3.0");
    CHECK_NEAR(weights[1].toDouble(), 1.0, 1e-6, " weighted train: weight[1] = 1.0");
  } else {
    CHECK(false, " weighted train: failed to read saved model package");
  }

  std::cout << std::endl;
}

// Regression guard for the validation deadlock ( side). With validation enabled, the
// per-epoch validation pass (CoreCPU::test) runs from inside train()'s per-sample callback.
// When train() and test() shared the global QThreadPool, that nested map deadlocked — every
// worker thread parked in a futex at 0% CPU. The fix gives each core its own pool.
// Runner only runs validation when saveModelInterval > 0, so the fixture sets it; the
// net trains for 3 epochs (validation fires from epoch 2 on) under a short timeout, so a
// regression surfaces as a fast failure instead of an indefinite hang. Not --full gated.
static void testTrainValidationNoDeadlock()
{
  std::cout << "  testTrainValidationNoDeadlock... ";

  QString modelPath = tempDir() + "/ann_validation_nodeadlock.nnmodel.tar";

  auto result =
    runNNCLI({"--config", fixturePath("ann_validation_config.json"), "--mode", "train", "--device", "cpu", "--samples",
              fixturePath("ann_validation_samples.json"), "--output", modelPath, "--log-level", "quiet"},
             60000); // 60s deadlock guard — real train takes <1s; a hang trips the timeout

  CHECK(result.exitCode == 0, " validation no-deadlock: training exit code 0 (timeout/-2 = deadlock)");
  CHECK(QFile::exists(modelPath), " validation no-deadlock: trained model file exists");

  std::cout << std::endl;
}

//===================================================================================================================//

void runCPUBasicTests()
{
  testTrainXOR();
  testNetworkDetection();
  testModeOverride();
  testTrainWithWeightedLoss();
  testTrainValidationNoDeadlock();
}
