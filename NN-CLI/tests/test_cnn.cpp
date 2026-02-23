#include "test_helpers.hpp"

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

void runCNNTests() {
  testCNNNetworkDetection();
  testCNNTrain();
  testCNNPredict();
  testCNNTest();
}

