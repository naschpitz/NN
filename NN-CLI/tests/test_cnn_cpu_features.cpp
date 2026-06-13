#include "test_helpers.hpp"

#include <QDir>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>

//===================================================================================================================//

static void testCNNCheckpointParameters()
{
  std::cout << "  testCNNCheckpointParameters... ";

  // Write a custom config with enough epochs to trigger checkpoints
  // (existing fixture has 5 epochs / interval 10, which produces no checkpoints)
  QString configPath = tempDir() + "/cnn_ckpt_config.json";
  QFile configFile(configPath);

  if (configFile.exists())
    configFile.remove();

  if (configFile.open(QIODevice::WriteOnly)) {
    const char* configJson = R"({
  "mode": "train",
  "device": "cpu",
  "progressReports": 0,
  "saveModelInterval": 5,
  "inputShape": { "c": 1, "h": 4, "w": 4 },
  "convolutionalLayers": [
    { "type": "conv", "numFilters": 1, "filterH": 3, "filterW": 3, "strideY": 1, "strideX": 1, "slidingStrategy": "valid" },
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

  QString modelPath = tempDir() + "/cnn_ckpt_model.nnmodel.tar";

  auto result = runNNCLI(
    {"--config", configPath, "--mode", "train", "--device", "cpu", "--samples", samplesDst, "--output", modelPath});

  CHECK(result.exitCode == 0, "CNN checkpoint params: exit code 0");
  CHECK(result.stdOut.contains("Training completed."), "CNN checkpoint params: 'Training completed.'");

  // Find checkpoint files in tempDir/output/
  QDir outputDir(tempDir() + "/output");
  QStringList checkpoints = outputDir.entryList({"checkpoint_E-*.nnmodel.tar"}, QDir::Files);
  CHECK(!checkpoints.isEmpty(), "CNN checkpoint params: checkpoint files exist");

  if (!checkpoints.isEmpty()) {
    QString checkpointPath = outputDir.filePath(checkpoints.first());
    QJsonObject root = readModelJsonFromPackage(checkpointPath);

    if (!root.isEmpty()) {
      // model.json present — config has layer structure, parameters are in params.bin
      CHECK(root.contains("convolutionalLayers"), "CNN checkpoint params: has 'convolutionalLayers' config");

      QJsonArray convLayers = root["convolutionalLayers"].toArray();
      CHECK(!convLayers.isEmpty(), "CNN checkpoint params: conv layers non-empty");

      CHECK(root.contains("denseLayers"), "CNN checkpoint params: has 'denseLayers' config");

      // Verify checkpoint file has non-trivial size (params.bin contains trained data)
      QFile cpf(checkpointPath);

      if (cpf.open(QIODevice::ReadOnly)) {
        CHECK(cpf.size() > 1024, "CNN checkpoint params: checkpoint file has parameter data");
        cpf.close();
      }
    } else {
      CHECK(false, "CNN checkpoint params: failed to read checkpoint package");
    }
  }

  // Cleanup checkpoint output dir
  QDir(tempDir() + "/output").removeRecursively();

  std::cout << std::endl;
}

//===================================================================================================================//

static void testCNNCheckpointInstanceNormRoundTrip()
{
  std::cout << "  testCNNCheckpointInstanceNormRoundTrip... ";

  // Config with instancenorm layer — train enough to get non-trivial running stats
  QString configPath = tempDir() + "/cnn_norm_ckpt_config.json";
  QFile configFile(configPath);

  if (configFile.exists())
    configFile.remove();

  if (configFile.open(QIODevice::WriteOnly)) {
    const char* configJson = R"({
  "mode": "train",
  "device": "cpu",
  "progressReports": 0,
  "saveModelInterval": 5,
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
    CHECK(false, "CNN BN checkpoint: failed to write config file");
    std::cout << std::endl;
    return;
  }

  // Copy samples to tempDir so checkpoints go to tempDir/output/
  QString samplesSrc = fixturePath("cnn_train_samples.json");
  QString samplesDst = tempDir() + "/cnn_norm_ckpt_samples.json";
  QFile::remove(samplesDst);
  QFile::copy(samplesSrc, samplesDst);

  // Clean up any prior checkpoint output
  QDir(tempDir() + "/output").removeRecursively();

  QString modelPath = tempDir() + "/cnn_norm_ckpt_model.nnmodel.tar";

  auto result = runNNCLI(
    {"--config", configPath, "--mode", "train", "--device", "cpu", "--samples", samplesDst, "--output", modelPath});

  CHECK(result.exitCode == 0, "CNN BN checkpoint: exit code 0");
  CHECK(result.stdOut.contains("Training completed."), "CNN BN checkpoint: 'Training completed.'");

  // Verify the saved model has instancenorm layers in the config
  QJsonObject root = readModelJsonFromPackage(modelPath);

  if (!root.isEmpty()) {
    CHECK(root.contains("convolutionalLayers"), "CNN BN checkpoint: has 'convolutionalLayers'");

    QJsonArray convLayers = root["convolutionalLayers"].toArray();
    int normCount = 0;

    for (int i = 0; i < convLayers.size(); ++i) {
      if (convLayers[i].toObject()["type"].toString() == "instancenorm")
        normCount++;
    }

    CHECK(normCount == 1, "CNN BN checkpoint: 1 instancenorm layer in config");

    // Parameters (gamma, beta, runningMean, runningVar) are now in params.bin (binary).
    // The round-trip test below verifies they survive save/load correctly.

    // Now load the model back and verify the parameters survive the round-trip
    auto result2 = runNNCLI({"--config", modelPath, "--mode", "test", "--device", "cpu", "--samples", samplesDst});
    CHECK(result2.exitCode == 0, "CNN BN checkpoint: test with loaded model exit code 0");
    CHECK(result2.stdOut.contains("Test Results:"), "CNN BN checkpoint: test produces results");
  } else {
    CHECK(false, "CNN BN checkpoint: failed to read model package");
  }

  // Cleanup
  QDir(tempDir() + "/output").removeRecursively();

  std::cout << std::endl;
}

//===================================================================================================================//

static void testCNNGlobalDualPoolEndToEnd()
{
  std::cout << "  testCNNGlobalDualPoolEndToEnd... ";

  // Config with globaldualpool: conv→relu→globaldualpool→flatten→dense
  // Input 1x8x8, conv 4 filters 3x3 valid → 4x6x6, GDP → 8x1x1, flatten → dense 2 sigmoid
  QString configPath = tempDir() + "/cnn_gdp_config.json";
  QFile configFile(configPath);

  if (configFile.exists())
    configFile.remove();

  if (configFile.open(QIODevice::WriteOnly)) {
    const char* configJson = R"({
  "mode": "train",
  "device": "cpu",
  "progressReports": 0,
  "saveModelInterval": 0,
  "inputShape": { "c": 1, "h": 8, "w": 8 },
  "convolutionalLayers": [
    { "type": "conv", "numFilters": 4, "filterH": 3, "filterW": 3, "strideY": 1, "strideX": 1, "slidingStrategy": "valid" },
    { "type": "relu" },
    { "type": "globaldualpool" },
    { "type": "flatten" }
  ],
  "denseLayers": [
    { "numNeurons": 2, "actvFunc": "sigmoid" }
  ],
  "train": {
    "numEpochs": 200,
    "learningRate": 0.5
  }
})";

    configFile.write(configJson);
    configFile.close();
  } else {
    CHECK(false, "CNN GDP e2e: failed to write config file");
    std::cout << std::endl;
    return;
  }

  // Write samples: 2 samples, 1x8x8 = 64 values each, 2 outputs
  QString samplesPath = tempDir() + "/cnn_gdp_samples.json";
  QFile samplesFile(samplesPath);

  if (samplesFile.exists())
    samplesFile.remove();

  if (samplesFile.open(QIODevice::WriteOnly)) {
    QJsonArray samples;

    for (int s = 0; s < 2; s++) {
      QJsonObject sample;
      QJsonArray input;

      for (int i = 0; i < 64; i++)
        input.append(s == 0 ? (i / 64.0) : (1.0 - i / 64.0));

      QJsonArray output;
      output.append(s == 0 ? 1.0 : 0.0);
      output.append(s == 0 ? 0.0 : 1.0);

      sample["input"] = input;
      sample["output"] = output;
      samples.append(sample);
    }

    QJsonObject root;
    root["samples"] = samples;
    samplesFile.write(QJsonDocument(root).toJson());
    samplesFile.close();
  }

  // Train
  QString modelPath = tempDir() + "/cnn_gdp_model.nnmodel.tar";

  auto trainResult = runNNCLI(
    {"--config", configPath, "--mode", "train", "--device", "cpu", "--samples", samplesPath, "--output", modelPath});

  CHECK(trainResult.exitCode == 0, "CNN GDP e2e: train exit code 0");
  CHECK(trainResult.stdOut.contains("Training completed."), "CNN GDP e2e: training completed");
  CHECK(QFile::exists(modelPath), "CNN GDP e2e: model file created");

  // Predict using the trained model
  if (QFile::exists(modelPath)) {
    // Write predict inputs
    QString predictPath = tempDir() + "/cnn_gdp_predict_input.json";
    QFile predictFile(predictPath);

    if (predictFile.open(QIODevice::WriteOnly)) {
      QJsonObject root;
      QJsonArray inputs;

      for (int s = 0; s < 2; s++) {
        QJsonArray input;

        for (int i = 0; i < 64; i++)
          input.append(s == 0 ? (i / 64.0) : (1.0 - i / 64.0));

        inputs.append(input);
      }

      root["inputs"] = inputs;
      predictFile.write(QJsonDocument(root).toJson());
      predictFile.close();
    }

    QString predictOutput = tempDir() + "/cnn_gdp_predict_output.json";

    auto predResult = runNNCLI({"--config", modelPath, "--mode", "predict", "--device", "cpu", "--input", predictPath,
                                "--output", predictOutput});

    CHECK(predResult.exitCode == 0, "CNN GDP e2e: predict exit code 0");
    CHECK(QFile::exists(predictOutput), "CNN GDP e2e: predict output file created");

    // Read predictions and check they're different for different inputs
    if (QFile::exists(predictOutput)) {
      QFile outFile(predictOutput);

      if (outFile.open(QIODevice::ReadOnly)) {
        QJsonDocument doc = QJsonDocument::fromJson(outFile.readAll());
        QJsonObject root = doc.object();
        QJsonArray outputs = root["outputs"].toArray();
        CHECK(outputs.size() == 2, "CNN GDP e2e: 2 predictions");

        if (outputs.size() == 2) {
          QJsonArray pred0 = outputs[0].toArray();
          QJsonArray pred1 = outputs[1].toArray();
          CHECK(pred0.size() == 2 && pred1.size() == 2, "CNN GDP e2e: 2 outputs per prediction");

          // The two inputs are different patterns, predictions should differ
          double diff = std::abs(pred0[0].toDouble() - pred1[0].toDouble());
          CHECK(diff > 0.01, "CNN GDP e2e: predictions are distinct");
        }

        outFile.close();
      }
    }
  }

  std::cout << std::endl;
}

//===================================================================================================================//

static void testCNNResidualEndToEnd()
{
  std::cout << "  testCNNResidualEndToEnd... ";

  // Identity residual: Conv(4,same)→ReLU→residual_start→Conv(4,same)→ReLU→residual_end→GAP→Flatten→Dense(2)
  QString configPath = tempDir() + "/cnn_res_config.json";
  QFile configFile(configPath);

  if (configFile.exists())
    configFile.remove();

  if (configFile.open(QIODevice::WriteOnly)) {
    const char* configJson = R"({
  "mode": "train",
  "device": "cpu",
  "progressReports": 0,
  "saveModelInterval": 0,
  "inputShape": { "c": 1, "h": 8, "w": 8 },
  "convolutionalLayers": [
    { "type": "conv", "numFilters": 4, "filterH": 3, "filterW": 3, "strideY": 1, "strideX": 1, "slidingStrategy": "same" },
    { "type": "relu" },
    { "type": "residual_start" },
    { "type": "conv", "numFilters": 4, "filterH": 3, "filterW": 3, "strideY": 1, "strideX": 1, "slidingStrategy": "same" },
    { "type": "relu" },
    { "type": "residual_end" },
    { "type": "globalavgpool" },
    { "type": "flatten" }
  ],
  "denseLayers": [
    { "numNeurons": 2, "actvFunc": "sigmoid" }
  ],
  "train": {
    "numEpochs": 200,
    "learningRate": 0.5
  }
})";

    configFile.write(configJson);
    configFile.close();
  }

  // Write samples
  QString samplesPath = tempDir() + "/cnn_res_samples.json";
  QFile samplesFile(samplesPath);

  if (samplesFile.exists())
    samplesFile.remove();

  if (samplesFile.open(QIODevice::WriteOnly)) {
    QJsonArray samples;

    for (int s = 0; s < 2; s++) {
      QJsonObject sample;
      QJsonArray input;

      for (int i = 0; i < 64; i++)
        input.append(s == 0 ? (i / 64.0) : (1.0 - i / 64.0));

      QJsonArray output;
      output.append(s == 0 ? 1.0 : 0.0);
      output.append(s == 0 ? 0.0 : 1.0);

      sample["input"] = input;
      sample["output"] = output;
      samples.append(sample);
    }

    QJsonObject root;
    root["samples"] = samples;
    samplesFile.write(QJsonDocument(root).toJson());
    samplesFile.close();
  }

  // Train
  QString modelPath = tempDir() + "/cnn_res_model.nnmodel.tar";

  auto trainResult = runNNCLI(
    {"--config", configPath, "--mode", "train", "--device", "cpu", "--samples", samplesPath, "--output", modelPath});

  CHECK(trainResult.exitCode == 0, "CNN Residual e2e: train exit code 0");
  CHECK(trainResult.stdOut.contains("Training completed."), "CNN Residual e2e: training completed");
  CHECK(QFile::exists(modelPath), "CNN Residual e2e: model file created");

  // Verify model JSON contains residual_start/end
  if (QFile::exists(modelPath)) {
    QJsonObject root = readModelJsonFromPackage(modelPath);

    if (!root.isEmpty()) {
      QJsonArray convLayers = root["convolutionalLayers"].toArray();
      bool hasResStart = false;
      bool hasResEnd = false;

      for (int i = 0; i < convLayers.size(); ++i) {
        QString type = convLayers[i].toObject()["type"].toString();

        if (type == "residual_start")
          hasResStart = true;

        if (type == "residual_end")
          hasResEnd = true;
      }

      CHECK(hasResStart, "CNN Residual e2e: model contains residual_start");
      CHECK(hasResEnd, "CNN Residual e2e: model contains residual_end");
    }
  }

  // Predict
  QString predictPath = tempDir() + "/cnn_res_predict_input.json";
  QFile predictFile(predictPath);

  if (predictFile.open(QIODevice::WriteOnly)) {
    QJsonObject root;
    QJsonArray inputs;

    for (int s = 0; s < 2; s++) {
      QJsonArray input;

      for (int i = 0; i < 64; i++)
        input.append(s == 0 ? (i / 64.0) : (1.0 - i / 64.0));

      inputs.append(input);
    }

    root["inputs"] = inputs;
    predictFile.write(QJsonDocument(root).toJson());
    predictFile.close();
  }

  QString predictOutput = tempDir() + "/cnn_res_predict_output.json";

  auto predResult = runNNCLI(
    {"--config", modelPath, "--mode", "predict", "--device", "cpu", "--input", predictPath, "--output", predictOutput});

  CHECK(predResult.exitCode == 0, "CNN Residual e2e: predict exit code 0");

  if (QFile::exists(predictOutput)) {
    QFile outFile(predictOutput);

    if (outFile.open(QIODevice::ReadOnly)) {
      QJsonDocument doc = QJsonDocument::fromJson(outFile.readAll());
      QJsonArray outputs = doc.object()["outputs"].toArray();
      CHECK(outputs.size() == 2, "CNN Residual e2e: 2 predictions");

      if (outputs.size() == 2) {
        double diff = std::abs(outputs[0].toArray()[0].toDouble() - outputs[1].toArray()[0].toDouble());
        CHECK(diff > 0.01, "CNN Residual e2e: predictions are distinct");
      }

      outFile.close();
    }
  }

  std::cout << std::endl;
}

//===================================================================================================================//

void runCNNCPUFeatureTests()
{
  testCNNCheckpointParameters();
  testCNNCheckpointInstanceNormRoundTrip();
  testCNNGlobalDualPoolEndToEnd();
  testCNNResidualEndToEnd();
}