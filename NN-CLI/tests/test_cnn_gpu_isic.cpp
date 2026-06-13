#include "test_helpers.hpp"

#include <QDir>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>

#include <json.hpp>

// CNN/ library headers for direct instantiation tests
#include <CNN_Core.hpp>
#include <CNN_CoreConfig.hpp>
#include <CNN_CoreGPU.hpp>
#include <CNN_CoreGPUWorker.hpp>
#include <CNN_GPUBufferManager.hpp>
#include <CNN_Sample.hpp>
#include <ANN_CoreGPUWorker.hpp>

//             → Conv(8,3x3,same)→BN→ReLU→Conv(8,3x3,same)→BN→ReLU→MaxPool(2x2)
//             → AvgPool(4x4)→Flatten→Dense(4,relu)→Dense(3,softmax)
// With crossEntropy, dropout 0.5, multi-class (3 classes)
static QString writeISICLikeConfig(const QString& path, const QString& device)
{
  QFile f(path);

  if (f.exists())
    f.remove();

  if (!f.open(QIODevice::WriteOnly))
    return {};

  // clang-format off
  QString json = QString(R"({
  "mode": "train",
  "device": "%1",
  "progressReports": 0,
  "saveModelInterval": 0,
  "inputShape": { "c": 1, "h": 16, "w": 16 },
  "convolutionalLayers": [
    { "type": "conv", "numFilters": 4, "filterH": 3, "filterW": 3, "strideY": 1, "strideX": 1, "slidingStrategy": "same" },
    { "type": "instancenorm" },
    { "type": "relu" },
    { "type": "conv", "numFilters": 4, "filterH": 3, "filterW": 3, "strideY": 1, "strideX": 1, "slidingStrategy": "same" },
    { "type": "instancenorm" },
    { "type": "relu" },
    { "type": "pool", "poolType": "max", "poolH": 2, "poolW": 2, "strideY": 2, "strideX": 2 },

    { "type": "conv", "numFilters": 8, "filterH": 3, "filterW": 3, "strideY": 1, "strideX": 1, "slidingStrategy": "same" },
    { "type": "instancenorm" },
    { "type": "relu" },
    { "type": "conv", "numFilters": 8, "filterH": 3, "filterW": 3, "strideY": 1, "strideX": 1, "slidingStrategy": "same" },
    { "type": "instancenorm" },
    { "type": "relu" },
    { "type": "pool", "poolType": "max", "poolH": 2, "poolW": 2, "strideY": 2, "strideX": 2 },

    { "type": "pool", "poolType": "avg", "poolH": 4, "poolW": 4, "strideY": 4, "strideX": 4 },
    { "type": "flatten" }
  ],
  "denseLayers": [
    { "numNeurons": 4, "actvFunc": "relu" },
    { "numNeurons": 3, "actvFunc": "softmax" }
  ],
  "costFunction": {
    "type": "crossEntropy"
  },
  "train": {
    "numEpochs": 100,
    "batchSize": 2,
    "learningRate": 0.005,
    "optimizer": { "type": "adam" },
    "dropoutRate": 0.0
  }
})").arg(device);
  // clang-format on

  f.write(json.toUtf8());
  f.close();
  return path;
}

// Helper: write ISIC-like samples (1x16x16 input, 3-class one-hot output)
static QString writeISICLikeSamples(const QString& path)
{
  QFile f(path);

  if (f.exists())
    f.remove();

  if (!f.open(QIODevice::WriteOnly))
    return {};

  // 6 samples, 3 classes (2 per class), 1x16x16 = 256 values each
  // Use distinct patterns so the model can learn something
  nlohmann::ordered_json samplesJson;
  nlohmann::ordered_json samplesArr = nlohmann::ordered_json::array();

  for (int s = 0; s < 6; s++) {
    nlohmann::ordered_json sample;
    std::vector<float> input(256);

    for (int i = 0; i < 256; i++) {
      // Different patterns per class
      int cls = s / 2;

      if (cls == 0)
        input[i] = static_cast<float>(i) / 255.0f; // gradient
      else if (cls == 1)
        input[i] = static_cast<float>(255 - i) / 255.0f; // reverse gradient
      else
        input[i] = (i % 2 == 0) ? 0.8f : 0.2f; // checkerboard
    }

    sample["input"] = input;

    // One-hot output
    std::vector<float> output(3, 0.0f);
    output[s / 2] = 1.0f;
    sample["output"] = output;
    samplesArr.push_back(sample);
  }

  samplesJson["samples"] = samplesArr;
  std::string jsonStr = samplesJson.dump();
  f.write(QByteArray::fromStdString(jsonStr));
  f.close();
  return path;
}

static void testCNNISICLikeSaveLoadPredict()
{
  std::cout << "  testCNNISICLikeSaveLoadPredict (CPU)... " << std::flush;

  QString configPath = writeISICLikeConfig(tempDir() + "/cnn_isic_config.json", "cpu");
  QString samplesPath = writeISICLikeSamples(tempDir() + "/cnn_isic_samples.json");

  CHECK(!configPath.isEmpty(), "ISIC-like CPU: config written");
  CHECK(!samplesPath.isEmpty(), "ISIC-like CPU: samples written");

  if (configPath.isEmpty() || samplesPath.isEmpty()) {
    std::cout << std::endl;
    return;
  }

  QString modelPath = tempDir() + "/cnn_isic_model.nnmodel.tar";

  // Step 1: Train
  auto trainResult = runNNCLI(
    {"--config", configPath, "--mode", "train", "--device", "cpu", "--samples", samplesPath, "--output", modelPath},
    300000); // 5 min timeout

  CHECK(trainResult.exitCode == 0, "ISIC-like CPU: train exit code 0");

  if (trainResult.exitCode != 0) {
    std::cout << "(train failed)" << std::endl;
    return;
  }

  // Step 2: Verify saved model has all 4 instancenorm layers in the config
  {
    QJsonObject root = readModelJsonFromPackage(modelPath);

    if (!root.isEmpty()) {
      QJsonArray convLayers = root["convolutionalLayers"].toArray();
      int normCount = 0;

      for (int i = 0; i < convLayers.size(); ++i) {
        if (convLayers[i].toObject()["type"].toString() == "instancenorm")
          normCount++;
      }

      CHECK(normCount == 4, "ISIC-like CPU: 4 instancenorm layers in saved model");
    } else {
      CHECK(false, "ISIC-like CPU: failed to read model package");
    }
  }

  // Step 3: Predict on all 6 samples (as predict inputs) — outputs must differ
  // Write predict inputs from the same samples
  QString predictInputPath = tempDir() + "/cnn_isic_predict_inputs.json";
  {
    QFile pf(predictInputPath);

    if (pf.exists())
      pf.remove();

    if (pf.open(QIODevice::WriteOnly)) {
      nlohmann::ordered_json inputsJson;
      nlohmann::ordered_json inputsArr = nlohmann::ordered_json::array();

      for (int s = 0; s < 6; s++) {
        std::vector<float> input(256);
        int cls = s / 2;

        for (int i = 0; i < 256; i++) {
          if (cls == 0)
            input[i] = static_cast<float>(i) / 255.0f;
          else if (cls == 1)
            input[i] = static_cast<float>(255 - i) / 255.0f;
          else
            input[i] = (i % 2 == 0) ? 0.8f : 0.2f;
        }

        inputsArr.push_back(input);
      }

      inputsJson["inputs"] = inputsArr;
      std::string jsonStr = inputsJson.dump();
      pf.write(QByteArray::fromStdString(jsonStr));
      pf.close();
    }
  }

  QString predictOutput1 = tempDir() + "/cnn_isic_predict1.json";
  auto pred1 = runNNCLI({"--config", modelPath, "--mode", "predict", "--device", "cpu", "--input", predictInputPath,
                         "--output", predictOutput1});

  CHECK(pred1.exitCode == 0, "ISIC-like CPU: predict1 exit code 0");

  // Step 4: Predict again (reload from disk) — must match
  QString predictOutput2 = tempDir() + "/cnn_isic_predict2.json";
  auto pred2 = runNNCLI({"--config", modelPath, "--mode", "predict", "--device", "cpu", "--input", predictInputPath,
                         "--output", predictOutput2});

  CHECK(pred2.exitCode == 0, "ISIC-like CPU: predict2 exit code 0");

  if (pred1.exitCode == 0 && pred2.exitCode == 0) {
    QFile f1(predictOutput1);
    QFile f2(predictOutput2);

    if (f1.open(QIODevice::ReadOnly) && f2.open(QIODevice::ReadOnly)) {
      QJsonArray outputs1 = QJsonDocument::fromJson(f1.readAll()).object()["outputs"].toArray();
      QJsonArray outputs2 = QJsonDocument::fromJson(f2.readAll()).object()["outputs"].toArray();

      CHECK(outputs1.size() == 6, "ISIC-like CPU: 6 predict outputs");
      CHECK(outputs1.size() == outputs2.size(), "ISIC-like CPU: same output count");

      // Check predictions match between two loads
      bool allMatch = true;

      for (int s = 0; s < outputs1.size() && s < outputs2.size(); s++) {
        QJsonArray o1 = outputs1[s].toArray();
        QJsonArray o2 = outputs2[s].toArray();

        for (int i = 0; i < o1.size() && i < o2.size(); i++) {
          if (std::fabs(o1[i].toDouble() - o2[i].toDouble()) > 1e-6) {
            allMatch = false;
            break;
          }
        }
      }

      CHECK(allMatch, "ISIC-like CPU: predictions match after reload");

      // Check that NOT all outputs are identical (the "all same" bug)
      if (outputs1.size() >= 2) {
        QJsonArray first = outputs1[0].toArray();
        bool anyDifferent = false;

        for (int s = 1; s < outputs1.size(); s++) {
          QJsonArray other = outputs1[s].toArray();

          for (int i = 0; i < first.size() && i < other.size(); i++) {
            if (std::fabs(first[i].toDouble() - other[i].toDouble()) > 1e-6) {
              anyDifferent = true;
              break;
            }
          }

          if (anyDifferent)
            break;
        }

        CHECK(anyDifferent, "ISIC-like CPU: different inputs produce different outputs (not collapsed)");
      }

      f1.close();
      f2.close();
    } else {
      CHECK(false, "ISIC-like CPU: failed to open predict output files");
    }
  }

  // Step 5: Test mode — verify it runs and produces reasonable results
  auto testResult = runNNCLI({"--config", modelPath, "--mode", "test", "--device", "cpu", "--samples", samplesPath});
  CHECK(testResult.exitCode == 0, "ISIC-like CPU: test exit code 0");
  CHECK(testResult.stdOut.contains("Test Results:"), "ISIC-like CPU: test produces results");
  CHECK(testResult.stdOut.contains("Samples evaluated: 6"), "ISIC-like CPU: test evaluated 6 samples");

  std::cout << std::endl;
}

static void testCNNISICLikeSaveLoadPredictGPU()
{
  std::cout << "  testCNNISICLikeSaveLoadPredictGPU... " << std::flush;

  if (!checkGPUAvailable()) {
    std::cout << "(skipped — no GPU)" << std::endl;
    return;
  }

  QString configPath = writeISICLikeConfig(tempDir() + "/cnn_isic_gpu_config.json", "gpu");
  QString samplesPath = tempDir() + "/cnn_isic_samples.json"; // reuse from CPU test

  // Ensure samples exist (CPU test may not have run)
  if (!QFile::exists(samplesPath))
    samplesPath = writeISICLikeSamples(samplesPath);

  CHECK(!configPath.isEmpty(), "ISIC-like GPU: config written");

  QString modelPath = tempDir() + "/cnn_isic_gpu_model.nnmodel.tar";

  // Step 1: Train on GPU
  auto trainResult = runNNCLI(
    {"--config", configPath, "--mode", "train", "--device", "gpu", "--samples", samplesPath, "--output", modelPath},
    300000);

  CHECK(trainResult.exitCode == 0, "ISIC-like GPU: train exit code 0");

  if (trainResult.exitCode != 0) {
    std::cout << "(train failed)" << std::endl;
    return;
  }

  // Step 2: Verify 4 instancenorm layers in config
  {
    QJsonObject root = readModelJsonFromPackage(modelPath);

    if (!root.isEmpty()) {
      QJsonArray convLayers = root["convolutionalLayers"].toArray();
      int normCount = 0;

      for (int i = 0; i < convLayers.size(); ++i) {
        if (convLayers[i].toObject()["type"].toString() == "instancenorm")
          normCount++;
      }

      CHECK(normCount == 4, "ISIC-like GPU: 4 instancenorm layers in saved model");
    }
  }

  // Step 3: Write predict inputs
  QString predictInputPath = tempDir() + "/cnn_isic_gpu_predict_inputs.json";
  {
    QFile pf(predictInputPath);

    if (pf.exists())
      pf.remove();

    if (pf.open(QIODevice::WriteOnly)) {
      nlohmann::ordered_json inputsJson;
      nlohmann::ordered_json inputsArr = nlohmann::ordered_json::array();

      for (int s = 0; s < 6; s++) {
        std::vector<float> input(256);
        int cls = s / 2;

        for (int i = 0; i < 256; i++) {
          if (cls == 0)
            input[i] = static_cast<float>(i) / 255.0f;
          else if (cls == 1)
            input[i] = static_cast<float>(255 - i) / 255.0f;
          else
            input[i] = (i % 2 == 0) ? 0.8f : 0.2f;
        }

        inputsArr.push_back(input);
      }

      inputsJson["inputs"] = inputsArr;
      std::string jsonStr = inputsJson.dump();
      pf.write(QByteArray::fromStdString(jsonStr));
      pf.close();
    }
  }

  // Step 4: Predict on GPU
  QString predictOutput1 = tempDir() + "/cnn_isic_gpu_predict1.json";
  auto pred1 = runNNCLI({"--config", modelPath, "--mode", "predict", "--device", "gpu", "--input", predictInputPath,
                         "--output", predictOutput1});

  CHECK(pred1.exitCode == 0, "ISIC-like GPU: predict1 exit code 0");

  // Step 5: Predict again (reload)
  QString predictOutput2 = tempDir() + "/cnn_isic_gpu_predict2.json";
  auto pred2 = runNNCLI({"--config", modelPath, "--mode", "predict", "--device", "gpu", "--input", predictInputPath,
                         "--output", predictOutput2});

  CHECK(pred2.exitCode == 0, "ISIC-like GPU: predict2 exit code 0");

  if (pred1.exitCode == 0 && pred2.exitCode == 0) {
    QFile f1(predictOutput1);
    QFile f2(predictOutput2);

    if (f1.open(QIODevice::ReadOnly) && f2.open(QIODevice::ReadOnly)) {
      QJsonArray outputs1 = QJsonDocument::fromJson(f1.readAll()).object()["outputs"].toArray();
      QJsonArray outputs2 = QJsonDocument::fromJson(f2.readAll()).object()["outputs"].toArray();

      CHECK(outputs1.size() == 6, "ISIC-like GPU: 6 predict outputs");

      // Check predictions match between two loads
      bool allMatch = true;

      for (int s = 0; s < outputs1.size() && s < outputs2.size(); s++) {
        QJsonArray o1 = outputs1[s].toArray();
        QJsonArray o2 = outputs2[s].toArray();

        for (int i = 0; i < o1.size() && i < o2.size(); i++) {
          if (std::fabs(o1[i].toDouble() - o2[i].toDouble()) > 1e-6) {
            allMatch = false;
            break;
          }
        }
      }

      CHECK(allMatch, "ISIC-like GPU: predictions match after reload");

      // Check outputs are not all identical
      if (outputs1.size() >= 2) {
        QJsonArray first = outputs1[0].toArray();
        bool anyDifferent = false;

        for (int s = 1; s < outputs1.size(); s++) {
          QJsonArray other = outputs1[s].toArray();

          for (int i = 0; i < first.size() && i < other.size(); i++) {
            if (std::fabs(first[i].toDouble() - other[i].toDouble()) > 1e-6) {
              anyDifferent = true;
              break;
            }
          }

          if (anyDifferent)
            break;
        }

        if (!anyDifferent) {
          std::cerr << "  GPU predict outputs (all identical):" << std::endl;

          for (int s = 0; s < std::min((int)outputs1.size(), 3); s++) {
            QJsonArray o = outputs1[s].toArray();
            std::cerr << "    sample " << s << ": [";

            for (int i = 0; i < o.size(); i++) {
              if (i > 0)
                std::cerr << ", ";

              std::cerr << o[i].toDouble();
            }

            std::cerr << "]" << std::endl;
          }
        }

        CHECK(anyDifferent, "ISIC-like GPU: different inputs produce different outputs (not collapsed)");
      }

      f1.close();
      f2.close();
    } else {
      CHECK(false, "ISIC-like GPU: failed to open predict output files");
    }
  }

  // Step 6: Test mode on GPU (should at least not crash)
  auto testResult = runNNCLI({"--config", modelPath, "--mode", "test", "--device", "gpu", "--samples", samplesPath});
  CHECK(testResult.exitCode == 0, "ISIC-like GPU: test exit code 0");
  CHECK(testResult.stdOut.contains("Test Results:"), "ISIC-like GPU: test produces results");

  std::cout << std::endl;
}

void runCNNGPUISICTests()
{
  testCNNISICLikeSaveLoadPredict();
  testCNNISICLikeSaveLoadPredictGPU();
}
