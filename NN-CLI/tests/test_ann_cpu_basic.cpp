#include "test_helpers.hpp"

#include <QDir>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>

QString trainedModelPath;

//===================================================================================================================//

static void testTrainXOR()
{
  TestScope _t("testTrainXOR");

  QString outDir = tempDir() + "/ann_xor_out";
  QDir(outDir).removeRecursively();
  QDir().mkpath(outDir);

  auto result = runNNCLI({"--model", fixturePath("ann_train_config.json"), "--mode", "train", "--device", "cpu",
                          "--samples", fixturePath("ann_train_samples.json"), "--output", outDir});

  CHECK(result.exitCode == 0, " train XOR: exit code 0");
  CHECK(result.stdOut.contains("Training completed."), " train XOR: 'Training completed.'");
  CHECK(result.stdOut.contains("Model saved to:"), " train XOR: 'Model saved to:'");
  trainedModelPath = findTrainedModel(outDir);
  CHECK(!trainedModelPath.isEmpty(), " train XOR: model file exists");

  // Clear the path if training failed so downstream tests skip gracefully
  if (result.exitCode != 0 || trainedModelPath.isEmpty()) {
    trainedModelPath.clear();
  }
}

//===================================================================================================================//

static void testNetworkDetection()
{
  TestScope _t("testNetworkDetection");

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

  QString outDir = tempDir() + "/ann_detect_out";
  QDir(outDir).removeRecursively();
  QDir().mkpath(outDir);

  auto result = runNNCLI({"--model-package", trainedModelPath, "--mode", "predict", "--device", "cpu", "--input",
                          predictInputPath, "--output", outDir, "--log-level", "info"});

  CHECK(result.exitCode == 0, " detection: exit code 0");
  CHECK(result.stdOut.contains("Network type: "), " detection: stdout contains 'Network type: '");
}

//===================================================================================================================//

static void testModeOverride()
{
  TestScope _t("testModeOverride");

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

  QString outDir = tempDir() + "/ann_override_out";
  QDir(outDir).removeRecursively();
  QDir().mkpath(outDir);

  // Trained model has mode=train; override to predict via CLI
  auto result = runNNCLI({"--model-package", trainedModelPath, "--mode", "predict", "--device", "cpu", "--input",
                          predictInputPath, "--output", outDir, "--log-level", "info"});

  CHECK(result.exitCode == 0, " mode override: exit code 0");
  CHECK(result.stdOut.contains("Mode: predict (CLI)"), " mode override: 'Mode: predict (CLI)'");
}

//===================================================================================================================//

static void testTrainWithWeightedLoss()
{
  TestScope _t("testTrainWithWeightedLoss");

  QString outDir = tempDir() + "/ann_weighted_out";
  QDir(outDir).removeRecursively();
  QDir().mkpath(outDir);

  auto result = runNNCLI({"--model", fixturePath("ann_train_weighted_config.json"), "--mode", "train", "--device",
                          "cpu", "--samples", fixturePath("ann_train_samples.json"), "--output", outDir});

  CHECK(result.exitCode == 0, " weighted train: exit code 0");
  CHECK(result.stdOut.contains("Training completed."), " weighted train: 'Training completed.'");
  CHECK(result.stdOut.contains("Model saved to:"), " weighted train: 'Model saved to:'");
  QString modelPath = findTrainedModel(outDir);
  CHECK(!modelPath.isEmpty(), " weighted train: model file exists");

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
  TestScope _t("testTrainValidationNoDeadlock");

  QString outDir = tempDir() + "/ann_validation_nodeadlock_out";
  QDir(outDir).removeRecursively();
  QDir().mkpath(outDir);

  auto result =
    runNNCLI({"--model", fixturePath("ann_validation_config.json"), "--mode", "train", "--device", "cpu", "--samples",
              fixturePath("ann_validation_samples.json"), "--output", outDir, "--log-level", "quiet"},
             60000); // 60s deadlock guard — real train takes <1s; a hang trips the timeout

  CHECK(result.exitCode == 0, " validation no-deadlock: training exit code 0 (timeout/-2 = deadlock)");
  QString modelPath = findTrainedModel(outDir);
  CHECK(!modelPath.isEmpty(), " validation no-deadlock: trained model file exists");
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
