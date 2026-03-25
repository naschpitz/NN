#include "test_helpers.hpp"

#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>

QString trainedANNModelPath;

//===================================================================================================================//

static void testANNTrainXOR()
{
  std::cout << "  testANNTrainXOR... ";

  trainedANNModelPath = tempDir() + "/ann_xor_model.json";

  auto result = runNNCLI({"--config", fixturePath("ann_train_config.json"), "--mode", "train", "--device", "cpu",
                          "--samples", fixturePath("ann_train_samples.json"), "--output", trainedANNModelPath});

  CHECK(result.exitCode == 0, "ANN train XOR: exit code 0");
  CHECK(result.stdOut.contains("Training completed."), "ANN train XOR: 'Training completed.'");
  CHECK(result.stdOut.contains("Model saved to:"), "ANN train XOR: 'Model saved to:'");
  CHECK(QFile::exists(trainedANNModelPath), "ANN train XOR: model file exists");

  // Clear the path if training failed so downstream tests skip gracefully
  if (result.exitCode != 0 || !QFile::exists(trainedANNModelPath)) {
    trainedANNModelPath.clear();
  }

  std::cout << std::endl;
}

//===================================================================================================================//

static void testANNNetworkDetection()
{
  std::cout << "  testANNNetworkDetection... ";

  if (trainedANNModelPath.isEmpty() || !QFile::exists(trainedANNModelPath)) {
    CHECK(false, "ANN detection: skipped — no trained model available (testANNTrainXOR must run first)");
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

  auto result = runNNCLI({"--config", trainedANNModelPath, "--mode", "predict", "--device", "cpu", "--input",
                          predictInputPath, "--output", tempDir() + "/ann_detect_output.json", "--log-level", "info"});

  CHECK(result.exitCode == 0, "ANN detection: exit code 0");
  CHECK(result.stdOut.contains("Network type: ANN"), "ANN detection: stdout contains 'Network type: ANN'");
  std::cout << std::endl;
}

//===================================================================================================================//

static void testANNModeOverride()
{
  std::cout << "  testANNModeOverride... ";

  if (trainedANNModelPath.isEmpty() || !QFile::exists(trainedANNModelPath)) {
    CHECK(false, "ANN mode override: skipped — no trained model available (testANNTrainXOR must run first)");
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
  auto result = runNNCLI({"--config", trainedANNModelPath, "--mode", "predict", "--device", "cpu", "--input",
                          predictInputPath, "--output", outputPath, "--log-level", "info"});

  CHECK(result.exitCode == 0, "ANN mode override: exit code 0");
  CHECK(result.stdOut.contains("Mode: predict (CLI)"), "ANN mode override: 'Mode: predict (CLI)'");
  std::cout << std::endl;
}

//===================================================================================================================//

static void testANNTrainWithWeightedLoss()
{
  std::cout << "  testANNTrainWithWeightedLoss... ";

  QString modelPath = tempDir() + "/ann_weighted_model.json";

  auto result = runNNCLI({"--config", fixturePath("ann_train_weighted_config.json"), "--mode", "train", "--device",
                          "cpu", "--samples", fixturePath("ann_train_samples.json"), "--output", modelPath});

  CHECK(result.exitCode == 0, "ANN weighted train: exit code 0");
  CHECK(result.stdOut.contains("Training completed."), "ANN weighted train: 'Training completed.'");
  CHECK(result.stdOut.contains("Model saved to:"), "ANN weighted train: 'Model saved to:'");
  CHECK(QFile::exists(modelPath), "ANN weighted train: model file exists");

  // Verify saved model JSON contains costFunctionConfig
  QFile file(modelPath);

  if (file.open(QIODevice::ReadOnly)) {
    QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
    QJsonObject root = doc.object();

    CHECK(root.contains("costFunctionConfig"), "ANN weighted train: saved model has 'costFunctionConfig'");

    QJsonObject cfc = root["costFunctionConfig"].toObject();
    CHECK(cfc["type"].toString() == "weightedSquaredDifference",
          "ANN weighted train: type is 'weightedSquaredDifference'");
    CHECK(cfc.contains("weights"), "ANN weighted train: has 'weights'");

    QJsonArray weights = cfc["weights"].toArray();
    CHECK(weights.size() == 2, "ANN weighted train: weights has 2 elements");
    CHECK_NEAR(weights[0].toDouble(), 3.0, 1e-6, "ANN weighted train: weight[0] = 3.0");
    CHECK_NEAR(weights[1].toDouble(), 1.0, 1e-6, "ANN weighted train: weight[1] = 1.0");

    file.close();
  } else {
    CHECK(false, "ANN weighted train: failed to open saved model file");
  }

  std::cout << std::endl;
}

//===================================================================================================================//

void runANNCPUBasicTests()
{
  testANNTrainXOR();
  testANNNetworkDetection();
  testANNModeOverride();
  testANNTrainWithWeightedLoss();
}
