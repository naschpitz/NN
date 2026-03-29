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

  auto result = runNNCLI(
    {"--config", configPath, "--mode", "train", "--device", "cpu", "--samples", samplesDst, "--output", modelPath});

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
      QJsonArray convArr = params["convolutional"].toArray();
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

  QString modelPath = tempDir() + "/cnn_norm_ckpt_model.json";

  auto result = runNNCLI(
    {"--config", configPath, "--mode", "train", "--device", "cpu", "--samples", samplesDst, "--output", modelPath});

  CHECK(result.exitCode == 0, "CNN BN checkpoint: exit code 0");
  CHECK(result.stdOut.contains("Training completed."), "CNN BN checkpoint: 'Training completed.'");

  // Verify the saved model has instancenorm parameters
  QFile modelFile(modelPath);

  if (modelFile.open(QIODevice::ReadOnly)) {
    QJsonDocument doc = QJsonDocument::fromJson(modelFile.readAll());
    QJsonObject root = doc.object();
    CHECK(root.contains("parameters"), "CNN BN checkpoint: has 'parameters'");

    QJsonObject params = root["parameters"].toObject();
    CHECK(params.contains("instancenorm"), "CNN BN checkpoint: has 'instancenorm' params");

    QJsonArray normArr = params["instancenorm"].toArray();
    CHECK(normArr.size() == 1, "CNN BN checkpoint: 1 instancenorm layer");

    if (!normArr.isEmpty()) {
      QJsonObject bn = normArr[0].toObject();
      CHECK(bn.contains("numChannels"), "CNN BN checkpoint: has 'numChannels'");
      CHECK(bn["numChannels"].toInt() == 2, "CNN BN checkpoint: numChannels == 2");
      CHECK(bn.contains("gamma"), "CNN BN checkpoint: has 'gamma'");
      CHECK(bn.contains("beta"), "CNN BN checkpoint: has 'beta'");
      CHECK(bn.contains("runningMean"), "CNN BN checkpoint: has 'runningMean'");
      CHECK(bn.contains("runningVar"), "CNN BN checkpoint: has 'runningVar'");

      QJsonArray gamma = bn["gamma"].toArray();
      QJsonArray beta = bn["beta"].toArray();
      QJsonArray runningMean = bn["runningMean"].toArray();
      QJsonArray runningVar = bn["runningVar"].toArray();

      CHECK(gamma.size() == 2, "CNN BN checkpoint: gamma has 2 elements");
      CHECK(beta.size() == 2, "CNN BN checkpoint: beta has 2 elements");
      CHECK(runningMean.size() == 2, "CNN BN checkpoint: runningMean has 2 elements");
      CHECK(runningVar.size() == 2, "CNN BN checkpoint: runningVar has 2 elements");

      // After training, running stats should have moved from their initial values (0.0 / 1.0)
      bool meanMoved = std::fabs(runningMean[0].toDouble()) > 1e-6 || std::fabs(runningMean[1].toDouble()) > 1e-6;
      CHECK(meanMoved, "CNN BN checkpoint: runningMean moved from initial 0.0");

      // Now load the model back and verify the parameters survive the round-trip
      auto result2 = runNNCLI({"--config", modelPath, "--mode", "test", "--device", "cpu", "--samples", samplesDst});
      CHECK(result2.exitCode == 0, "CNN BN checkpoint: test with loaded model exit code 0");
      CHECK(result2.stdOut.contains("Test Results:"), "CNN BN checkpoint: test produces results");
    }

    modelFile.close();
  } else {
    CHECK(false, "CNN BN checkpoint: failed to open model file");
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
  "convolutionalLayersConfig": [
    { "type": "conv", "numFilters": 4, "filterH": 3, "filterW": 3, "strideY": 1, "strideX": 1, "slidingStrategy": "valid" },
    { "type": "relu" },
    { "type": "globaldualpool" },
    { "type": "flatten" }
  ],
  "denseLayersConfig": [
    { "numNeurons": 2, "actvFunc": "sigmoid" }
  ],
  "trainingConfig": {
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
  QString modelPath = tempDir() + "/cnn_gdp_model.json";

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

    auto predResult =
      runNNCLI({"--config", modelPath, "--mode", "predict", "--device", "cpu", "--input", predictPath, "--output",
                predictOutput});

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
  "convolutionalLayersConfig": [
    { "type": "conv", "numFilters": 4, "filterH": 3, "filterW": 3, "strideY": 1, "strideX": 1, "slidingStrategy": "same" },
    { "type": "relu" },
    { "type": "residual_start" },
    { "type": "conv", "numFilters": 4, "filterH": 3, "filterW": 3, "strideY": 1, "strideX": 1, "slidingStrategy": "same" },
    { "type": "relu" },
    { "type": "residual_end" },
    { "type": "globalavgpool" },
    { "type": "flatten" }
  ],
  "denseLayersConfig": [
    { "numNeurons": 2, "actvFunc": "sigmoid" }
  ],
  "trainingConfig": {
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
  QString modelPath = tempDir() + "/cnn_res_model.json";

  auto trainResult = runNNCLI(
    {"--config", configPath, "--mode", "train", "--device", "cpu", "--samples", samplesPath, "--output", modelPath});

  CHECK(trainResult.exitCode == 0, "CNN Residual e2e: train exit code 0");
  CHECK(trainResult.stdOut.contains("Training completed."), "CNN Residual e2e: training completed");
  CHECK(QFile::exists(modelPath), "CNN Residual e2e: model file created");

  // Verify model JSON contains residual_start/end
  if (QFile::exists(modelPath)) {
    QFile model(modelPath);

    if (model.open(QIODevice::ReadOnly)) {
      QByteArray data = model.readAll();
      model.close();
      CHECK(data.contains("residual_start"), "CNN Residual e2e: model contains residual_start");
      CHECK(data.contains("residual_end"), "CNN Residual e2e: model contains residual_end");
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

    auto predResult =
      runNNCLI({"--config", modelPath, "--mode", "predict", "--device", "cpu", "--input", predictPath, "--output",
                predictOutput});

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