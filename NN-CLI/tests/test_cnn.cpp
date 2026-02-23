#include "test_helpers.hpp"

#include <QDir>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>

// Trained model path shared between CNN tests (train → predict/test)
static QString trainedCNNModelPath;

static void testCNNNetworkDetection() {
  std::cout << "  testCNNNetworkDetection... ";

  // Train with tiny fixture + verbose to check detection
  auto result = runNNCLI({
    "--config", fixturePath("cnn_train_config.json"),
    "--mode", "train",
    "--device", "cpu",
    "--samples", fixturePath("cnn_train_samples.json"),
    "--output", tempDir() + "/cnn_detect_model.json",
    "--log-level", "info"
  });

  CHECK(result.exitCode == 0, "CNN detection: exit code 0");
  CHECK(result.stdOut.contains("Network type: CNN"), "CNN detection: 'Network type: CNN'");
  std::cout << std::endl;
}

static void testCNNTrain() {
  std::cout << "  testCNNTrain... ";

  trainedCNNModelPath = tempDir() + "/cnn_trained_model.json";

  auto result = runNNCLI({
    "--config", fixturePath("cnn_train_config.json"),
    "--mode", "train",
    "--device", "cpu",
    "--samples", fixturePath("cnn_train_samples.json"),
    "--output", trainedCNNModelPath
  });

  CHECK(result.exitCode == 0, "CNN train: exit code 0");
  CHECK(result.stdOut.contains("Training completed."), "CNN train: 'Training completed.'");
  CHECK(result.stdOut.contains("Model saved to:"), "CNN train: 'Model saved to:'");
  CHECK(QFile::exists(trainedCNNModelPath), "CNN train: model file exists");
  std::cout << std::endl;
}

static void testCNNPredict() {
  std::cout << "  testCNNPredict... ";

  if (trainedCNNModelPath.isEmpty() || !QFile::exists(trainedCNNModelPath)) {
    CHECK(false, "CNN predict: skipped — no trained model available");
    std::cout << std::endl;
    return;
  }

  QString outputPath = tempDir() + "/cnn_predict_output.json";

  auto result = runNNCLI({
    "--config", trainedCNNModelPath,
    "--mode", "predict",
    "--device", "cpu",
    "--input", fixturePath("cnn_predict_input.json"),
    "--output", outputPath
  });

  CHECK(result.exitCode == 0, "CNN predict: exit code 0");
  CHECK(result.stdOut.contains("Predict result saved to:"), "CNN predict: 'Predict result saved to:'");
  CHECK(QFile::exists(outputPath), "CNN predict: output file exists");

  // Verify output JSON structure
  QFile file(outputPath);
  if (file.open(QIODevice::ReadOnly)) {
    QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
    QJsonObject root = doc.object();

    CHECK(root.contains("predictMetadata"), "CNN predict: has 'predictMetadata'");
    CHECK(root.contains("outputs"), "CNN predict: has 'outputs'");

    QJsonArray outputsArray = root["outputs"].toArray();
    CHECK(outputsArray.size() == 1, "CNN predict: outputs has 1 element (batch of 1)");

    QJsonArray firstOutput = outputsArray[0].toArray();
    CHECK(firstOutput.size() == 2, "CNN predict: first output has 2 elements");

    QJsonObject meta = root["predictMetadata"].toObject();
    CHECK(meta.contains("startTime"), "CNN predict: metadata has 'startTime'");
    CHECK(meta.contains("durationSeconds"), "CNN predict: metadata has 'durationSeconds'");
    CHECK(meta.contains("numInputs"), "CNN predict: metadata has 'numInputs'");
    file.close();
  } else {
    CHECK(false, "CNN predict: failed to open output file");
  }
  std::cout << std::endl;
}

static void testCNNTest() {
  std::cout << "  testCNNTest... ";

  if (trainedCNNModelPath.isEmpty() || !QFile::exists(trainedCNNModelPath)) {
    CHECK(false, "CNN test: skipped — no trained model available");
    std::cout << std::endl;
    return;
  }

  auto result = runNNCLI({
    "--config", trainedCNNModelPath,
    "--mode", "test",
    "--device", "cpu",
    "--samples", fixturePath("cnn_train_samples.json")
  });

  CHECK(result.exitCode == 0, "CNN test: exit code 0");
  CHECK(result.stdOut.contains("Test Results:"), "CNN test: 'Test Results:'");
  CHECK(result.stdOut.contains("Samples evaluated: 4"), "CNN test: 'Samples evaluated: 4'");
  CHECK(result.stdOut.contains("Total loss:"), "CNN test: 'Total loss:'");
  CHECK(result.stdOut.contains("Average loss:"), "CNN test: 'Average loss:'");
  std::cout << std::endl;
}

static void testCNNTrainWithWeightedLoss() {
  std::cout << "  testCNNTrainWithWeightedLoss... ";

  QString modelPath = tempDir() + "/cnn_weighted_model.json";

  auto result = runNNCLI({
    "--config", fixturePath("cnn_train_weighted_config.json"),
    "--mode", "train",
    "--device", "cpu",
    "--samples", fixturePath("cnn_train_samples.json"),
    "--output", modelPath
  });

  CHECK(result.exitCode == 0, "CNN weighted train: exit code 0");
  CHECK(result.stdOut.contains("Training completed."), "CNN weighted train: 'Training completed.'");
  CHECK(result.stdOut.contains("Model saved to:"), "CNN weighted train: 'Model saved to:'");
  CHECK(QFile::exists(modelPath), "CNN weighted train: model file exists");

  // Verify saved model JSON contains costFunctionConfig
  QFile file(modelPath);
  if (file.open(QIODevice::ReadOnly)) {
    QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
    QJsonObject root = doc.object();

    CHECK(root.contains("costFunctionConfig"), "CNN weighted train: saved model has 'costFunctionConfig'");

    QJsonObject cfc = root["costFunctionConfig"].toObject();
    CHECK(cfc["type"].toString() == "weightedSquaredDifference",
          "CNN weighted train: type is 'weightedSquaredDifference'");
    CHECK(cfc.contains("weights"), "CNN weighted train: has 'weights'");

    QJsonArray weights = cfc["weights"].toArray();
    CHECK(weights.size() == 2, "CNN weighted train: weights has 2 elements");
    CHECK_NEAR(weights[0].toDouble(), 5.0, 1e-6, "CNN weighted train: weight[0] = 5.0");
    CHECK_NEAR(weights[1].toDouble(), 1.0, 1e-6, "CNN weighted train: weight[1] = 1.0");

    file.close();
  } else {
    CHECK(false, "CNN weighted train: failed to open saved model file");
  }
  std::cout << std::endl;
}

static void testCNNCheckpointParameters() {
  std::cout << "  testCNNCheckpointParameters... ";

  // Write a custom config with enough epochs to trigger checkpoints
  // (existing fixture has 5 epochs / interval 10, which produces no checkpoints)
  QString configPath = tempDir() + "/cnn_ckpt_config.json";
  QFile configFile(configPath);
  if (configFile.exists()) configFile.remove();
  if (configFile.open(QIODevice::WriteOnly)) {
    const char* configJson = R"({
  "mode": "train",
  "device": "cpu",
  "progressReports": 0,
  "saveModelInterval": 5,
  "inputShape": { "c": 1, "h": 4, "w": 4 },
  "convolutionalLayersConfig": [
    { "type": "conv", "numFilters": 1, "filterH": 3, "filterW": 3, "strideY": 1, "strideX": 1, "slidingStrategy": "valid" },
    { "type": "relu" },
    { "type": "flatten" }
  ],
  "denseLayersConfig": [
    { "numNeurons": 2, "actvFunc": "sigmoid" }
  ],
  "trainingConfig": {
    "numEpochs": 20,
    "learningRate": 0.1
  }
})";
    configFile.write(configJson);
    configFile.close();
  } else {
    CHECK(false, "CNN checkpoint params: failed to write config file");
    std::cout << std::endl;
    return;
  }

  // Copy samples to tempDir so checkpoints go to tempDir/output/
  // (generateCheckpointPath uses the samples file directory, not the config directory)
  QString samplesSrc = fixturePath("cnn_train_samples.json");
  QString samplesDst = tempDir() + "/cnn_ckpt_samples.json";
  QFile::remove(samplesDst);
  QFile::copy(samplesSrc, samplesDst);

  // Clean up any prior checkpoint output
  QDir(tempDir() + "/output").removeRecursively();

  QString modelPath = tempDir() + "/cnn_ckpt_model.json";

  auto result = runNNCLI({
    "--config", configPath,
    "--mode", "train",
    "--device", "cpu",
    "--samples", samplesDst,
    "--output", modelPath
  });

  CHECK(result.exitCode == 0, "CNN checkpoint params: exit code 0");
  CHECK(result.stdOut.contains("Training completed."), "CNN checkpoint params: 'Training completed.'");

  // Find checkpoint files in tempDir/output/
  QDir outputDir(tempDir() + "/output");
  QStringList checkpoints = outputDir.entryList({"checkpoint_E-*.json"}, QDir::Files);
  CHECK(!checkpoints.isEmpty(), "CNN checkpoint params: checkpoint files exist");

  if (!checkpoints.isEmpty()) {
    QString checkpointPath = outputDir.filePath(checkpoints.first());
    QFile file(checkpointPath);
    if (file.open(QIODevice::ReadOnly)) {
      QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
      QJsonObject root = doc.object();
      CHECK(root.contains("parameters"), "CNN checkpoint params: has 'parameters'");

      QJsonObject params = root["parameters"].toObject();

      // Verify conv parameters are non-empty
      QJsonArray convArr = params["conv"].toArray();
      CHECK(!convArr.isEmpty(), "CNN checkpoint params: conv non-empty");
      if (!convArr.isEmpty()) {
        QJsonObject firstConv = convArr[0].toObject();
        QJsonArray filters = firstConv["filters"].toArray();
        CHECK(!filters.isEmpty(), "CNN checkpoint params: conv[0].filters non-empty");
      }

      // Verify dense parameters are non-empty
      QJsonObject dense = params["dense"].toObject();
      QJsonArray denseWeights = dense["weights"].toArray();
      QJsonArray denseBiases = dense["biases"].toArray();
      CHECK(!denseWeights.isEmpty(), "CNN checkpoint params: dense.weights non-empty");
      CHECK(!denseBiases.isEmpty(), "CNN checkpoint params: dense.biases non-empty");

      file.close();
    } else {
      CHECK(false, "CNN checkpoint params: failed to open checkpoint file");
    }
  }

  // Cleanup checkpoint output dir
  QDir(tempDir() + "/output").removeRecursively();

  std::cout << std::endl;
}

void runCNNTests() {
  testCNNNetworkDetection();
  testCNNTrain();
  testCNNPredict();
  testCNNTest();
  testCNNTrainWithWeightedLoss();
  testCNNCheckpointParameters();
}

