#include "test_helpers.hpp"

#include <QDir>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>

#include <json.hpp>

// CNN/ANN library headers for direct instantiation tests
#include <CNN_Core.hpp>
#include <CNN_CoreConfig.hpp>
#include <CNN_CoreGPU.hpp>
#include <CNN_CoreGPUWorker.hpp>
#include <CNN_GPUBufferManager.hpp>
#include <CNN_Sample.hpp>
#include <ANN_CoreGPUWorker.hpp>

static void testCNNSaveLoadPredictConsistency()
{
  std::cout << "  testCNNSaveLoadPredictConsistency... ";

  // Train a CNN with instancenorm, predict, save, load, predict again — outputs must match
  QString configPath = tempDir() + "/cnn_slpc_config.json";
  QFile configFile(configPath);

  if (configFile.exists())
    configFile.remove();

  if (configFile.open(QIODevice::WriteOnly)) {
    const char* configJson = R"({
  "mode": "train",
  "device": "cpu",
  "progressReports": 0,
  "saveModelInterval": 0,
  "inputShape": { "c": 1, "h": 4, "w": 4 },
  "convolutionalLayersConfig": [
    { "type": "conv", "numFilters": 2, "filterH": 3, "filterW": 3, "strideY": 1, "strideX": 1, "slidingStrategy": "valid" },
    { "type": "instancenorm" },
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
    CHECK(false, "CNN save/load predict: failed to write config file");
    std::cout << std::endl;
    return;
  }

  QString modelPath = tempDir() + "/cnn_slpc_model.json";

  // Step 1: Train
  auto trainResult = runNNCLI({"--config", configPath, "--mode", "train", "--device", "cpu", "--samples",
                               fixturePath("cnn_train_samples.json"), "--output", modelPath});

  CHECK(trainResult.exitCode == 0, "CNN save/load predict: train exit code 0");

  if (trainResult.exitCode != 0) {
    std::cout << std::endl;
    return;
  }

  // Step 2: Predict with the trained model (before save/load cycle)
  QString predictOutput1 = tempDir() + "/cnn_slpc_predict1.json";
  auto pred1Result = runNNCLI({"--config", modelPath, "--mode", "predict", "--device", "cpu", "--input",
                               fixturePath("cnn_predict_input.json"), "--output", predictOutput1});

  CHECK(pred1Result.exitCode == 0, "CNN save/load predict: predict1 exit code 0");

  // Step 3: Predict again with the same saved model (simulates loading from disk)
  QString predictOutput2 = tempDir() + "/cnn_slpc_predict2.json";
  auto pred2Result = runNNCLI({"--config", modelPath, "--mode", "predict", "--device", "cpu", "--input",
                               fixturePath("cnn_predict_input.json"), "--output", predictOutput2});

  CHECK(pred2Result.exitCode == 0, "CNN save/load predict: predict2 exit code 0");

  // Step 4: Compare outputs — they must be identical (same model, same input)
  if (pred1Result.exitCode == 0 && pred2Result.exitCode == 0) {
    QFile f1(predictOutput1);
    QFile f2(predictOutput2);

    if (f1.open(QIODevice::ReadOnly) && f2.open(QIODevice::ReadOnly)) {
      QJsonDocument doc1 = QJsonDocument::fromJson(f1.readAll());
      QJsonDocument doc2 = QJsonDocument::fromJson(f2.readAll());

      QJsonArray outputs1 = doc1.object()["outputs"].toArray();
      QJsonArray outputs2 = doc2.object()["outputs"].toArray();

      CHECK(outputs1.size() == outputs2.size(), "CNN save/load predict: same number of outputs");

      if (!outputs1.isEmpty() && !outputs2.isEmpty()) {
        QJsonArray out1 = outputs1[0].toArray();
        QJsonArray out2 = outputs2[0].toArray();

        CHECK(out1.size() == out2.size(), "CNN save/load predict: same output size");

        bool allMatch = true;

        for (int i = 0; i < out1.size() && i < out2.size(); i++) {
          if (std::fabs(out1[i].toDouble() - out2[i].toDouble()) > 1e-6) {
            allMatch = false;
            break;
          }
        }

        CHECK(allMatch, "CNN save/load predict: predictions match after reload");

        // Also verify outputs are not all identical (model should produce non-trivial output)
        if (out1.size() >= 2) {
          bool nonTrivial = std::fabs(out1[0].toDouble() - out1[1].toDouble()) > 1e-6;
          CHECK(nonTrivial, "CNN save/load predict: output values are non-trivial (not all same)");
        }
      }

      f1.close();
      f2.close();
    } else {
      CHECK(false, "CNN save/load predict: failed to open predict output files");
    }
  }

  std::cout << std::endl;
}

static void testCNNSaveLoadPredictConsistencyGPU()
{
  std::cout << "  testCNNSaveLoadPredictConsistencyGPU... ";

  if (!checkGPUAvailable()) {
    std::cout << "(skipped — no GPU)" << std::endl;
    return;
  }

  // Same as CPU test but on GPU
  QString configPath = tempDir() + "/cnn_slpc_gpu_config.json";
  QFile configFile(configPath);

  if (configFile.exists())
    configFile.remove();

  if (configFile.open(QIODevice::WriteOnly)) {
    const char* configJson = R"({
  "mode": "train",
  "device": "gpu",
  "progressReports": 0,
  "saveModelInterval": 0,
  "inputShape": { "c": 1, "h": 4, "w": 4 },
  "convolutionalLayersConfig": [
    { "type": "conv", "numFilters": 2, "filterH": 3, "filterW": 3, "strideY": 1, "strideX": 1, "slidingStrategy": "valid" },
    { "type": "instancenorm" },
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
    CHECK(false, "CNN GPU save/load predict: failed to write config file");
    std::cout << std::endl;
    return;
  }

  QString modelPath = tempDir() + "/cnn_slpc_gpu_model.json";

  // Step 1: Train on GPU
  auto trainResult = runNNCLI({"--config", configPath, "--mode", "train", "--device", "gpu", "--samples",
                               fixturePath("cnn_train_samples.json"), "--output", modelPath});

  CHECK(trainResult.exitCode == 0, "CNN GPU save/load predict: train exit code 0");

  if (trainResult.exitCode != 0) {
    std::cout << std::endl;
    return;
  }

  // Step 2: Predict on GPU with the saved model
  QString predictOutput1 = tempDir() + "/cnn_slpc_gpu_predict1.json";
  auto pred1Result = runNNCLI({"--config", modelPath, "--mode", "predict", "--device", "gpu", "--input",
                               fixturePath("cnn_predict_input.json"), "--output", predictOutput1});

  CHECK(pred1Result.exitCode == 0, "CNN GPU save/load predict: predict1 exit code 0");

  // Step 3: Predict again (fresh load from disk)
  QString predictOutput2 = tempDir() + "/cnn_slpc_gpu_predict2.json";
  auto pred2Result = runNNCLI({"--config", modelPath, "--mode", "predict", "--device", "gpu", "--input",
                               fixturePath("cnn_predict_input.json"), "--output", predictOutput2});

  CHECK(pred2Result.exitCode == 0, "CNN GPU save/load predict: predict2 exit code 0");

  // Step 4: Compare outputs
  if (pred1Result.exitCode == 0 && pred2Result.exitCode == 0) {
    QFile f1(predictOutput1);
    QFile f2(predictOutput2);

    if (f1.open(QIODevice::ReadOnly) && f2.open(QIODevice::ReadOnly)) {
      QJsonDocument doc1 = QJsonDocument::fromJson(f1.readAll());
      QJsonDocument doc2 = QJsonDocument::fromJson(f2.readAll());

      QJsonArray outputs1 = doc1.object()["outputs"].toArray();
      QJsonArray outputs2 = doc2.object()["outputs"].toArray();

      CHECK(outputs1.size() == outputs2.size(), "CNN GPU save/load predict: same number of outputs");

      if (!outputs1.isEmpty() && !outputs2.isEmpty()) {
        QJsonArray out1 = outputs1[0].toArray();
        QJsonArray out2 = outputs2[0].toArray();

        CHECK(out1.size() == out2.size(), "CNN GPU save/load predict: same output size");

        bool allMatch = true;

        for (int i = 0; i < out1.size() && i < out2.size(); i++) {
          if (std::fabs(out1[i].toDouble() - out2[i].toDouble()) > 1e-6) {
            allMatch = false;
            break;
          }
        }

        CHECK(allMatch, "CNN GPU save/load predict: predictions match after reload");
      }

      f1.close();
      f2.close();
    } else {
      CHECK(false, "CNN GPU save/load predict: failed to open predict output files");
    }
  }

  // Step 5: Cross-device check — predict on CPU with GPU-trained model
  QString predictOutputCPU = tempDir() + "/cnn_slpc_gpu_predict_cpu.json";
  auto predCPUResult = runNNCLI({"--config", modelPath, "--mode", "predict", "--device", "cpu", "--input",
                                 fixturePath("cnn_predict_input.json"), "--output", predictOutputCPU});

  CHECK(predCPUResult.exitCode == 0, "CNN GPU save/load predict: CPU predict exit code 0");

  if (pred1Result.exitCode == 0 && predCPUResult.exitCode == 0) {
    QFile fGPU(predictOutput1);
    QFile fCPU(predictOutputCPU);

    if (fGPU.open(QIODevice::ReadOnly) && fCPU.open(QIODevice::ReadOnly)) {
      QJsonDocument docGPU = QJsonDocument::fromJson(fGPU.readAll());
      QJsonDocument docCPU = QJsonDocument::fromJson(fCPU.readAll());

      QJsonArray outputsGPU = docGPU.object()["outputs"].toArray();
      QJsonArray outputsCPU = docCPU.object()["outputs"].toArray();

      if (!outputsGPU.isEmpty() && !outputsCPU.isEmpty()) {
        QJsonArray outGPU = outputsGPU[0].toArray();
        QJsonArray outCPU = outputsCPU[0].toArray();

        bool crossMatch = true;

        for (int i = 0; i < outGPU.size() && i < outCPU.size(); i++) {
          if (std::fabs(outGPU[i].toDouble() - outCPU[i].toDouble()) > 0.01) {
            crossMatch = false;
            break;
          }
        }

        CHECK(crossMatch, "CNN GPU save/load predict: GPU and CPU predictions match (tol=0.01)");
      }

      fGPU.close();
      fCPU.close();
    }
  }

  std::cout << std::endl;
}

void runCNNFeaturesTests2()
{
  testCNNSaveLoadPredictConsistency();
  testCNNSaveLoadPredictConsistencyGPU();
}
