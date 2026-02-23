#include "test_helpers.hpp"

#include <QDir>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>

// Pretrained MNIST model for predict/test tests
static const char* PRETRAINED_ANN =
    "MNIST/train/output/trained_model_1000_60000_0.06897394359111786.json";

static void testANNNetworkDetection() {
  std::cout << "  testANNNetworkDetection... ";

  auto result = runNNCLI({
    "--config", examplePath(PRETRAINED_ANN),
    "--mode", "predict",
    "--device", "cpu",
    "--input", examplePath("MNIST/predict/mnist_digit_2_input.json"),
    "--output", tempDir() + "/ann_detect_output.json",
    "--log-level", "info"
  });

  CHECK(result.exitCode == 0, "ANN detection: exit code 0");
  CHECK(result.stdOut.contains("Network type: ANN"), "ANN detection: stdout contains 'Network type: ANN'");
  std::cout << std::endl;
}

static void testANNTrainXOR() {
  std::cout << "  testANNTrainXOR... ";

  QString modelPath = tempDir() + "/ann_xor_model.json";

  auto result = runNNCLI({
    "--config", fixturePath("ann_train_config.json"),
    "--mode", "train",
    "--device", "cpu",
    "--samples", fixturePath("ann_train_samples.json"),
    "--output", modelPath
  });

  CHECK(result.exitCode == 0, "ANN train XOR: exit code 0");
  CHECK(result.stdOut.contains("Training completed."), "ANN train XOR: 'Training completed.'");
  CHECK(result.stdOut.contains("Model saved to:"), "ANN train XOR: 'Model saved to:'");
  CHECK(QFile::exists(modelPath), "ANN train XOR: model file exists");
  std::cout << std::endl;
}

static void testANNPredictMNIST() {
  std::cout << "  testANNPredictMNIST... ";

  QString outputPath = tempDir() + "/ann_predict_output.json";

  auto result = runNNCLI({
    "--config", examplePath(PRETRAINED_ANN),
    "--mode", "predict",
    "--device", "cpu",
    "--input", examplePath("MNIST/predict/mnist_digit_2_input.json"),
    "--output", outputPath
  });

  CHECK(result.exitCode == 0, "ANN predict MNIST: exit code 0");
  CHECK(result.stdOut.contains("Predict result saved to:"), "ANN predict MNIST: 'Predict result saved to:'");
  CHECK(QFile::exists(outputPath), "ANN predict MNIST: output file exists");

  // Verify output JSON structure and content
  QFile file(outputPath);
  if (file.open(QIODevice::ReadOnly)) {
    QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
    QJsonObject root = doc.object();

    CHECK(root.contains("predictMetadata"), "ANN predict MNIST: has 'predictMetadata'");
    CHECK(root.contains("outputs"), "ANN predict MNIST: has 'outputs'");

    QJsonArray outputsArray = root["outputs"].toArray();
    CHECK(outputsArray.size() == 1, "ANN predict MNIST: outputs has 1 element (batch of 1)");

    QJsonArray firstOutput = outputsArray[0].toArray();
    CHECK(firstOutput.size() == 10, "ANN predict MNIST: first output has 10 elements");

    // Verify all outputs are valid numbers in [0, 1]
    bool allValid = true;
    for (int i = 0; i < firstOutput.size(); ++i) {
      double v = firstOutput[i].toDouble();
      if (v < 0.0 || v > 1.0) { allValid = false; break; }
    }
    CHECK(allValid, "ANN predict MNIST: all outputs in [0, 1]");

    QJsonObject meta = root["predictMetadata"].toObject();
    CHECK(meta.contains("startTime"), "ANN predict MNIST: metadata has 'startTime'");
    CHECK(meta.contains("endTime"), "ANN predict MNIST: metadata has 'endTime'");
    CHECK(meta.contains("durationSeconds"), "ANN predict MNIST: metadata has 'durationSeconds'");
    CHECK(meta.contains("durationFormatted"), "ANN predict MNIST: metadata has 'durationFormatted'");
    CHECK(meta.contains("numInputs"), "ANN predict MNIST: metadata has 'numInputs'");
    file.close();
  } else {
    CHECK(false, "ANN predict MNIST: failed to open output file");
  }
  std::cout << std::endl;
}

static void testANNTestMNIST() {
  std::cout << "  testANNTestMNIST... ";

  auto result = runNNCLI({
    "--config", examplePath(PRETRAINED_ANN),
    "--mode", "test",
    "--device", "cpu",
    "--idx-data", examplePath("MNIST/test/t10k-images.idx3-ubyte"),
    "--idx-labels", examplePath("MNIST/test/t10k-labels.idx1-ubyte")
  });

  CHECK(result.exitCode == 0, "ANN test MNIST: exit code 0");
  CHECK(result.stdOut.contains("Test Results:"), "ANN test MNIST: 'Test Results:'");
  CHECK(result.stdOut.contains("Samples evaluated: 10000"), "ANN test MNIST: 'Samples evaluated: 10000'");
  CHECK(result.stdOut.contains("Total loss:"), "ANN test MNIST: 'Total loss:'");
  CHECK(result.stdOut.contains("Average loss:"), "ANN test MNIST: 'Average loss:'");
  std::cout << std::endl;
}

static void testANNModeOverride() {
  std::cout << "  testANNModeOverride... ";

  QString outputPath = tempDir() + "/ann_override_output.json";

  // Pretrained model has mode=train; override to predict via CLI
  auto result = runNNCLI({
    "--config", examplePath(PRETRAINED_ANN),
    "--mode", "predict",
    "--device", "cpu",
    "--input", examplePath("MNIST/predict/mnist_digit_2_input.json"),
    "--output", outputPath,
    "--log-level", "info"
  });

  CHECK(result.exitCode == 0, "ANN mode override: exit code 0");
  CHECK(result.stdOut.contains("Mode: predict (CLI)"), "ANN mode override: 'Mode: predict (CLI)'");
  std::cout << std::endl;
}

static void testANNTrainWithWeightedLoss() {
  std::cout << "  testANNTrainWithWeightedLoss... ";

  QString modelPath = tempDir() + "/ann_weighted_model.json";

  auto result = runNNCLI({
    "--config", fixturePath("ann_train_weighted_config.json"),
    "--mode", "train",
    "--device", "cpu",
    "--samples", fixturePath("ann_train_samples.json"),
    "--output", modelPath
  });

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

static void testANNCheckpointParameters() {
  std::cout << "  testANNCheckpointParameters... ";

  // Copy config and samples to tempDir so checkpoints go to tempDir/output/
  // (generateCheckpointPath uses the samples file directory, not the config directory)
  QString configSrc = fixturePath("ann_train_config.json");
  QString configDst = tempDir() + "/ann_ckpt_config.json";
  QFile::remove(configDst);
  QFile::copy(configSrc, configDst);

  QString samplesSrc = fixturePath("ann_train_samples.json");
  QString samplesDst = tempDir() + "/ann_ckpt_samples.json";
  QFile::remove(samplesDst);
  QFile::copy(samplesSrc, samplesDst);

  // Clean up any prior checkpoint output
  QDir(tempDir() + "/output").removeRecursively();

  QString modelPath = tempDir() + "/ann_ckpt_model.json";

  auto result = runNNCLI({
    "--config", configDst,
    "--mode", "train",
    "--device", "cpu",
    "--samples", samplesDst,
    "--output", modelPath
  });

  CHECK(result.exitCode == 0, "ANN checkpoint params: exit code 0");
  CHECK(result.stdOut.contains("Training completed."), "ANN checkpoint params: 'Training completed.'");

  // Find checkpoint files in tempDir/output/
  QDir outputDir(tempDir() + "/output");
  QStringList checkpoints = outputDir.entryList({"checkpoint_epoch_*.json"}, QDir::Files);
  CHECK(!checkpoints.isEmpty(), "ANN checkpoint params: checkpoint files exist");

  if (!checkpoints.isEmpty()) {
    // Parse the first checkpoint and verify parameters are non-empty
    QString checkpointPath = outputDir.filePath(checkpoints.first());
    QFile file(checkpointPath);
    if (file.open(QIODevice::ReadOnly)) {
      QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
      QJsonObject root = doc.object();
      CHECK(root.contains("parameters"), "ANN checkpoint params: has 'parameters'");

      QJsonObject params = root["parameters"].toObject();
      QJsonArray weights = params["weights"].toArray();
      QJsonArray biases = params["biases"].toArray();

      CHECK(!weights.isEmpty(), "ANN checkpoint params: weights non-empty");
      CHECK(!biases.isEmpty(), "ANN checkpoint params: biases non-empty");

      // Verify at least one layer has actual weight data
      if (!weights.isEmpty()) {
        bool hasData = false;
        for (int i = 0; i < weights.size(); ++i) {
          if (weights[i].toArray().size() > 0) { hasData = true; break; }
        }
        CHECK(hasData, "ANN checkpoint params: weights contain actual data");
      }

      file.close();
    } else {
      CHECK(false, "ANN checkpoint params: failed to open checkpoint file");
    }
  }

  // Cleanup checkpoint output dir
  QDir(tempDir() + "/output").removeRecursively();

  std::cout << std::endl;
}

void runANNTests() {
  testANNNetworkDetection();
  testANNTrainXOR();
  testANNPredictMNIST();
  testANNTestMNIST();
  testANNModeOverride();
  testANNTrainWithWeightedLoss();
  testANNCheckpointParameters();
}

