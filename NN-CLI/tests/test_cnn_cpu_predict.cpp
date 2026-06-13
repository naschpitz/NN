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

static void testCNNMultiInputPredictDiversity()
{
  std::cout << "  testCNNMultiInputPredictDiversity... ";

  // Train a model, then predict on multiple DIFFERENT inputs and verify outputs differ
  // This catches the "all outputs identical" bug
  QString configPath = tempDir() + "/cnn_diversity_config.json";
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
  "convolutionalLayers": [
    { "type": "conv", "numFilters": 2, "filterH": 3, "filterW": 3, "strideY": 1, "strideX": 1, "slidingStrategy": "valid" },
    { "type": "instancenorm" },
    { "type": "relu" },
    { "type": "flatten" }
  ],
  "denseLayers": [
    { "numNeurons": 2, "actvFunc": "sigmoid" }
  ],
  "train": {
    "numEpochs": 20,
    "learningRate": 0.1
  }
})";

    configFile.write(configJson);
    configFile.close();
  } else {
    CHECK(false, "CNN diversity: failed to write config file");
    std::cout << std::endl;
    return;
  }

  // Create a multi-input predict file with 2 different inputs
  QString multiInputPath = tempDir() + "/cnn_diversity_inputs.json";
  QFile multiInputFile(multiInputPath);

  if (multiInputFile.exists())
    multiInputFile.remove();

  if (multiInputFile.open(QIODevice::WriteOnly)) {
    const char* inputJson = R"({
  "inputs": [
    [0.1,0.2,0.3,0.4, 0.5,0.6,0.7,0.8, 0.1,0.2,0.3,0.4, 0.5,0.6,0.7,0.8],
    [0.9,0.8,0.7,0.6, 0.5,0.4,0.3,0.2, 0.9,0.8,0.7,0.6, 0.5,0.4,0.3,0.2]
  ]
})";

    multiInputFile.write(inputJson);
    multiInputFile.close();
  } else {
    CHECK(false, "CNN diversity: failed to write multi-input file");
    std::cout << std::endl;
    return;
  }

  QString modelPath = tempDir() + "/cnn_diversity_model.nnmodel.tar";

  // Train
  auto trainResult = runNNCLI({"--config", configPath, "--mode", "train", "--device", "cpu", "--samples",
                               fixturePath("cnn_train_samples.json"), "--output", modelPath});

  CHECK(trainResult.exitCode == 0, "CNN diversity: train exit code 0");

  if (trainResult.exitCode != 0) {
    std::cout << std::endl;
    return;
  }

  // Predict on multiple inputs
  QString predictOutput = tempDir() + "/cnn_diversity_predict.json";
  auto predResult = runNNCLI({"--config", modelPath, "--mode", "predict", "--device", "cpu", "--input", multiInputPath,
                              "--output", predictOutput});

  CHECK(predResult.exitCode == 0, "CNN diversity: predict exit code 0");

  if (predResult.exitCode == 0) {
    QFile f(predictOutput);

    if (f.open(QIODevice::ReadOnly)) {
      QJsonDocument doc = QJsonDocument::fromJson(f.readAll());
      QJsonArray outputs = doc.object()["outputs"].toArray();

      CHECK(outputs.size() == 2, "CNN diversity: 2 outputs");

      if (outputs.size() == 2) {
        QJsonArray out0 = outputs[0].toArray();
        QJsonArray out1 = outputs[1].toArray();

        // The two different inputs should produce different outputs
        bool outputsDiffer = false;

        for (int i = 0; i < out0.size() && i < out1.size(); i++) {
          if (std::fabs(out0[i].toDouble() - out1[i].toDouble()) > 1e-6) {
            outputsDiffer = true;
            break;
          }
        }

        CHECK(outputsDiffer, "CNN diversity: different inputs produce different outputs");
      }

      f.close();
    } else {
      CHECK(false, "CNN diversity: failed to open predict output file");
    }
  }

  std::cout << std::endl;
}

void runCNNCPUPredictTests()
{
  testCNNMultiInputPredictDiversity();
}
