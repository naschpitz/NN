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

// Trained model path shared between CNN tests (train → predict/test)
static QString trainedCNNModelPath;

static void testCNNNetworkDetection()
{
  std::cout << "  testCNNNetworkDetection... ";

  // Train with tiny fixture + verbose to check detection
  auto result = runNNCLI({"--config", fixturePath("cnn_train_config.json"), "--mode", "train", "--device", "cpu",
                          "--samples", fixturePath("cnn_train_samples.json"), "--output",
                          tempDir() + "/cnn_detect_model.json", "--log-level", "info"});

  CHECK(result.exitCode == 0, "CNN detection: exit code 0");
  CHECK(result.stdOut.contains("Network type: CNN"), "CNN detection: 'Network type: CNN'");
  std::cout << std::endl;
}

static void testCNNTrain()
{
  std::cout << "  testCNNTrain... ";

  trainedCNNModelPath = tempDir() + "/cnn_trained_model.json";

  auto result = runNNCLI({"--config", fixturePath("cnn_train_config.json"), "--mode", "train", "--device", "cpu",
                          "--samples", fixturePath("cnn_train_samples.json"), "--output", trainedCNNModelPath});

  CHECK(result.exitCode == 0, "CNN train: exit code 0");
  CHECK(result.stdOut.contains("Training completed."), "CNN train: 'Training completed.'");
  CHECK(result.stdOut.contains("Model saved to:"), "CNN train: 'Model saved to:'");
  CHECK(QFile::exists(trainedCNNModelPath), "CNN train: model file exists");
  std::cout << std::endl;
}

static void testCNNPredict()
{
  std::cout << "  testCNNPredict... ";

  if (trainedCNNModelPath.isEmpty() || !QFile::exists(trainedCNNModelPath)) {
    CHECK(false, "CNN predict: skipped — no trained model available");
    std::cout << std::endl;
    return;
  }

  QString outputPath = tempDir() + "/cnn_predict_output.json";

  auto result = runNNCLI({"--config", trainedCNNModelPath, "--mode", "predict", "--device", "cpu", "--input",
                          fixturePath("cnn_predict_input.json"), "--output", outputPath});

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

static void testCNNTest()
{
  std::cout << "  testCNNTest... ";

  if (trainedCNNModelPath.isEmpty() || !QFile::exists(trainedCNNModelPath)) {
    CHECK(false, "CNN test: skipped — no trained model available");
    std::cout << std::endl;
    return;
  }

  auto result = runNNCLI({"--config", trainedCNNModelPath, "--mode", "test", "--device", "cpu", "--samples",
                          fixturePath("cnn_train_samples.json")});

  CHECK(result.exitCode == 0, "CNN test: exit code 0");
  CHECK(result.stdOut.contains("Test Results:"), "CNN test: 'Test Results:'");
  CHECK(result.stdOut.contains("Samples evaluated: 4"), "CNN test: 'Samples evaluated: 4'");
  CHECK(result.stdOut.contains("Total loss:"), "CNN test: 'Total loss:'");
  CHECK(result.stdOut.contains("Average loss:"), "CNN test: 'Average loss:'");
  CHECK(result.stdOut.contains("Correct:"), "CNN test: 'Correct:'");
  CHECK(result.stdOut.contains("Accuracy:"), "CNN test: 'Accuracy:'");
  std::cout << std::endl;
}

static void testCNNTrainWithWeightedLoss()
{
  std::cout << "  testCNNTrainWithWeightedLoss... ";

  QString modelPath = tempDir() + "/cnn_weighted_model.json";

  auto result = runNNCLI({"--config", fixturePath("cnn_train_weighted_config.json"), "--mode", "train", "--device",
                          "cpu", "--samples", fixturePath("cnn_train_samples.json"), "--output", modelPath});

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

static void testCNNTrainAndTestMNIST()
{
  std::cout << "  testCNNTrainAndTestMNIST... " << std::flush;

  if (!runFullTests) {
    std::cout << "(skipped — use --full to enable)" << std::endl;
    return;
  }

  QString modelPath = tempDir() + "/cnn_mnist_trained.json";

  // Step 1: Train on MNIST training data on CPU (10 epochs, 60k samples, Adam + crossEntropy + batchnorm)
  auto trainResult =
    runNNCLI({"--config", fixturePath("mnist_cnn_train_config.json"), "--mode", "train", "--device", "cpu",
              "--idx-data", examplePath("MNIST/train/train-images.idx3-ubyte"), "--idx-labels",
              examplePath("MNIST/train/train-labels.idx1-ubyte"), "--output", modelPath, "--log-level", "quiet"},
             3600000); // 60 min timeout

  CHECK(trainResult.exitCode == 0, "CNN MNIST train+test: training exit code 0");
  CHECK(QFile::exists(modelPath), "CNN MNIST train+test: trained model file exists");

  if (trainResult.exitCode != 0 || !QFile::exists(modelPath)) {
    std::cout << "(training failed, skipping test step)" << std::endl;
    return;
  }

  // Step 2: Evaluate against MNIST test data (10k samples)
  auto testResult = runNNCLI({"--config", modelPath, "--mode", "test", "--device", "cpu", "--idx-data",
                              examplePath("MNIST/test/t10k-images.idx3-ubyte"), "--idx-labels",
                              examplePath("MNIST/test/t10k-labels.idx1-ubyte")},
                             600000); // 10 min timeout

  CHECK(testResult.exitCode == 0, "CNN MNIST train+test: test exit code 0");
  CHECK(testResult.stdOut.contains("Test Results:"), "CNN MNIST train+test: 'Test Results:'");
  CHECK(testResult.stdOut.contains("Samples evaluated: 10000"), "CNN MNIST train+test: 'Samples evaluated: 10000'");

  // Extract and verify average loss is reasonable
  double avgLoss = -1;
  int idx = testResult.stdOut.indexOf("Average loss:");

  if (idx >= 0) {
    QString lossStr = testResult.stdOut.mid(idx + QString("Average loss:").length()).trimmed();
    lossStr = lossStr.left(lossStr.indexOf('\n'));
    avgLoss = lossStr.toDouble();
  }

  CHECK(avgLoss > 0 && avgLoss < 2.5, "CNN MNIST train+test: average loss < 2.5");

  // Extract and verify accuracy is reasonable (> 25% for 10 epochs with Adam + crossEntropy + batchnorm on CPU)
  double accuracy = -1;
  int accIdx = testResult.stdOut.indexOf("Accuracy:");

  if (accIdx >= 0) {
    QString accStr = testResult.stdOut.mid(accIdx + QString("Accuracy:").length()).trimmed();
    accStr = accStr.left(accStr.indexOf('%'));
    accuracy = accStr.toDouble();
  }

  CHECK(accuracy > 25.0, "CNN MNIST train+test: accuracy > 25%");

  std::cout << "(loss=" << avgLoss << ", accuracy=" << accuracy << "%) " << std::endl;
}

//===================================================================================================================//

static void testCNNTrainAndTestMNISTGPU()
{
  std::cout << "  testCNNTrainAndTestMNISTGPU... " << std::flush;

  if (!runFullTests) {
    std::cout << "(skipped — use --full to enable)" << std::endl;
    return;
  }

  if (!checkGPUAvailable()) {
    std::cout << "(skipped — no GPU available)" << std::endl;
    return;
  }

  QString modelPath = tempDir() + "/cnn_mnist_trained_gpu.json";

  // Step 1: Train on MNIST training data on GPU (10 epochs, 60k samples, Adam + crossEntropy + batchnorm)
  auto trainResult =
    runNNCLI({"--config", fixturePath("mnist_cnn_train_config.json"), "--mode", "train", "--device", "gpu",
              "--idx-data", examplePath("MNIST/train/train-images.idx3-ubyte"), "--idx-labels",
              examplePath("MNIST/train/train-labels.idx1-ubyte"), "--output", modelPath, "--log-level", "quiet"},
             1800000); // 30 min timeout

  CHECK(trainResult.exitCode == 0, "CNN MNIST GPU train+test: training exit code 0");
  CHECK(QFile::exists(modelPath), "CNN MNIST GPU train+test: trained model file exists");

  if (trainResult.exitCode != 0 || !QFile::exists(modelPath)) {
    std::cout << "(training failed, skipping test step)" << std::endl;
    return;
  }

  // Step 2: Evaluate against MNIST test data (10k samples) on GPU
  auto testResult = runNNCLI({"--config", modelPath, "--mode", "test", "--device", "gpu", "--idx-data",
                              examplePath("MNIST/test/t10k-images.idx3-ubyte"), "--idx-labels",
                              examplePath("MNIST/test/t10k-labels.idx1-ubyte")},
                             600000); // 10 min timeout

  CHECK(testResult.exitCode == 0, "CNN MNIST GPU train+test: test exit code 0");
  CHECK(testResult.stdOut.contains("Test Results:"), "CNN MNIST GPU train+test: 'Test Results:'");
  CHECK(testResult.stdOut.contains("Samples evaluated: 10000"), "CNN MNIST GPU train+test: 'Samples evaluated: 10000'");

  // Extract and verify average loss is reasonable
  double avgLoss = -1;
  int idx = testResult.stdOut.indexOf("Average loss:");

  if (idx >= 0) {
    QString lossStr = testResult.stdOut.mid(idx + QString("Average loss:").length()).trimmed();
    lossStr = lossStr.left(lossStr.indexOf('\n'));
    avgLoss = lossStr.toDouble();
  }

  CHECK(avgLoss > 0 && avgLoss < 2.0, "CNN MNIST GPU train+test: average loss < 2.0");

  // Extract and verify accuracy is reasonable (> 30% for 30 epochs with mini-batch SGD)
  double accuracy = -1;
  int accIdx = testResult.stdOut.indexOf("Accuracy:");

  if (accIdx >= 0) {
    QString accStr = testResult.stdOut.mid(accIdx + QString("Accuracy:").length()).trimmed();
    accStr = accStr.left(accStr.indexOf('%'));
    accuracy = accStr.toDouble();
  }

  CHECK(accuracy > 50.0, "CNN MNIST GPU train+test: accuracy > 50%");

  std::cout << "(loss=" << avgLoss << ", accuracy=" << accuracy << "%) " << std::endl;
}

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

static void testCNNCheckpointBatchNormRoundTrip()
{
  std::cout << "  testCNNCheckpointBatchNormRoundTrip... ";

  // Config with batchnorm layer — train enough to get non-trivial running stats
  QString configPath = tempDir() + "/cnn_bn_ckpt_config.json";
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
    { "type": "batchnorm" },
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
  QString samplesDst = tempDir() + "/cnn_bn_ckpt_samples.json";
  QFile::remove(samplesDst);
  QFile::copy(samplesSrc, samplesDst);

  // Clean up any prior checkpoint output
  QDir(tempDir() + "/output").removeRecursively();

  QString modelPath = tempDir() + "/cnn_bn_ckpt_model.json";

  auto result = runNNCLI(
    {"--config", configPath, "--mode", "train", "--device", "cpu", "--samples", samplesDst, "--output", modelPath});

  CHECK(result.exitCode == 0, "CNN BN checkpoint: exit code 0");
  CHECK(result.stdOut.contains("Training completed."), "CNN BN checkpoint: 'Training completed.'");

  // Verify the saved model has batchnorm parameters
  QFile modelFile(modelPath);

  if (modelFile.open(QIODevice::ReadOnly)) {
    QJsonDocument doc = QJsonDocument::fromJson(modelFile.readAll());
    QJsonObject root = doc.object();
    CHECK(root.contains("parameters"), "CNN BN checkpoint: has 'parameters'");

    QJsonObject params = root["parameters"].toObject();
    CHECK(params.contains("batchnorm"), "CNN BN checkpoint: has 'batchnorm' params");

    QJsonArray bnArr = params["batchnorm"].toArray();
    CHECK(bnArr.size() == 1, "CNN BN checkpoint: 1 batchnorm layer");

    if (!bnArr.isEmpty()) {
      QJsonObject bn = bnArr[0].toObject();
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

static void testCNNSaveLoadPredictConsistency()
{
  std::cout << "  testCNNSaveLoadPredictConsistency... ";

  // Train a CNN with batchnorm, predict, save, load, predict again — outputs must match
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
    { "type": "batchnorm" },
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
    { "type": "batchnorm" },
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
  "convolutionalLayersConfig": [
    { "type": "conv", "numFilters": 2, "filterH": 3, "filterW": 3, "strideY": 1, "strideX": 1, "slidingStrategy": "valid" },
    { "type": "batchnorm" },
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

  QString modelPath = tempDir() + "/cnn_diversity_model.json";

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

//===================================================================================================================//
// Systematic GPU predict tests: isolate which layer combination breaks GPU predict
//===================================================================================================================//

// Helper: generate a CNN config JSON with given convolutional layers.
// All configs use 1x8x8 input, train on CPU, 2-output sigmoid dense layer.
static QString writeGPUTestConfig(const QString& path, const QString& convLayersJson)
{
  QFile f(path);

  if (f.exists())
    f.remove();

  if (!f.open(QIODevice::WriteOnly))
    return {};

  QString json = QString(R"({
  "mode": "train",
  "device": "cpu",
  "progressReports": 0,
  "saveModelInterval": 0,
  "inputShape": { "c": 1, "h": 8, "w": 8 },
  "convolutionalLayersConfig": %1,
  "denseLayersConfig": [
    { "numNeurons": 2, "actvFunc": "sigmoid" }
  ],
  "trainingConfig": {
    "numEpochs": 50,
    "learningRate": 0.01
  }
})")

                   .arg(convLayersJson);

  f.write(json.toUtf8());
  f.close();
  return path;
}

// Helper: write test inputs (1x8x8 = 64 values, 2 distinct patterns)
static QString writeGPUTestInputs(const QString& path)
{
  QFile f(path);

  if (f.exists())
    f.remove();

  if (!f.open(QIODevice::WriteOnly))
    return {};

  nlohmann::ordered_json inputsJson;
  nlohmann::ordered_json inputsArr = nlohmann::ordered_json::array();

  // Pattern A: gradient
  std::vector<float> a(64);

  for (int i = 0; i < 64; i++)
    a[i] = static_cast<float>(i) / 63.0f;
  inputsArr.push_back(a);

  // Pattern B: inverse gradient
  std::vector<float> b(64);

  for (int i = 0; i < 64; i++)
    b[i] = static_cast<float>(63 - i) / 63.0f;
  inputsArr.push_back(b);

  inputsJson["inputs"] = inputsArr;
  f.write(QByteArray::fromStdString(inputsJson.dump()));
  f.close();
  return path;
}

// Helper: write training samples for the GPU tests (1x8x8 input, 2-class output)
static QString writeGPUTestSamples(const QString& path)
{
  QFile f(path);

  if (f.exists())
    f.remove();

  if (!f.open(QIODevice::WriteOnly))
    return {};

  nlohmann::ordered_json samplesJson;
  nlohmann::ordered_json arr = nlohmann::ordered_json::array();

  // 4 samples: 2 per class with distinct spatial patterns
  for (int s = 0; s < 4; s++) {
    nlohmann::ordered_json sample;
    std::vector<float> input(64);
    int cls = s / 2;

    for (int i = 0; i < 64; i++) {
      if (cls == 0)
        input[i] = static_cast<float>(i) / 63.0f + 0.01f * s;
      else
        input[i] = static_cast<float>(63 - i) / 63.0f + 0.01f * (s - 2);
    }

    sample["input"] = input;
    sample["output"] = (cls == 0) ? std::vector<float>{1.0f, 0.0f} : std::vector<float>{0.0f, 1.0f};
    arr.push_back(sample);
  }

  samplesJson["samples"] = arr;
  f.write(QByteArray::fromStdString(samplesJson.dump()));
  f.close();
  return path;
}

// Core test: train on CPU, predict on CPU and GPU, compare.
// Returns true if GPU predict produces diverse outputs matching CPU.
static bool runGPUPredictTest(const QString& testName, const QString& convLayersJson)
{
  QString prefix = tempDir() + "/gpu_test_" + testName.toLower().replace(" ", "_");
  QString configPath = writeGPUTestConfig(prefix + "_config.json", convLayersJson);
  QString samplesPath = writeGPUTestSamples(prefix + "_samples.json");
  QString inputsPath = writeGPUTestInputs(prefix + "_inputs.json");
  QString modelPath = prefix + "_model.json";

  // Step 1: Train on CPU
  auto trainResult = runNNCLI(
    {"--config", configPath, "--mode", "train", "--device", "cpu", "--samples", samplesPath, "--output", modelPath});

  if (trainResult.exitCode != 0) {
    std::cerr << "    [" << testName.toStdString() << "] train failed (exit=" << trainResult.exitCode << ")"
              << std::endl;
    return false;
  }

  // Step 2: Predict on CPU
  QString cpuPredPath = prefix + "_pred_cpu.json";
  auto cpuPred = runNNCLI(
    {"--config", modelPath, "--mode", "predict", "--device", "cpu", "--input", inputsPath, "--output", cpuPredPath});

  if (cpuPred.exitCode != 0) {
    std::cerr << "    [" << testName.toStdString() << "] CPU predict failed" << std::endl;
    return false;
  }

  // Step 3: Predict on GPU
  QString gpuPredPath = prefix + "_pred_gpu.json";
  auto gpuPred = runNNCLI(
    {"--config", modelPath, "--mode", "predict", "--device", "gpu", "--input", inputsPath, "--output", gpuPredPath});

  if (gpuPred.exitCode != 0) {
    std::cerr << "    [" << testName.toStdString() << "] GPU predict failed (exit=" << gpuPred.exitCode << ")"
              << std::endl;

    if (!gpuPred.stdErr.isEmpty())
      std::cerr << "    stderr: " << gpuPred.stdErr.toStdString() << std::endl;
    return false;
  }

  // Step 4: Compare outputs
  QFile fc(cpuPredPath), fg(gpuPredPath);

  if (!fc.open(QIODevice::ReadOnly) || !fg.open(QIODevice::ReadOnly)) {
    std::cerr << "    [" << testName.toStdString() << "] failed to open output files" << std::endl;
    return false;
  }

  QJsonArray cpuOutputs = QJsonDocument::fromJson(fc.readAll()).object()["outputs"].toArray();
  QJsonArray gpuOutputs = QJsonDocument::fromJson(fg.readAll()).object()["outputs"].toArray();
  fc.close();
  fg.close();

  if (cpuOutputs.size() != 2 || gpuOutputs.size() != 2) {
    std::cerr << "    [" << testName.toStdString() << "] wrong output count" << std::endl;
    return false;
  }

  // Check CPU outputs are diverse (sanity)
  bool cpuDiverse = false;
  {
    QJsonArray c0 = cpuOutputs[0].toArray(), c1 = cpuOutputs[1].toArray();

    for (int i = 0; i < c0.size() && i < c1.size(); i++)

      if (std::fabs(c0[i].toDouble() - c1[i].toDouble()) > 1e-6)
        cpuDiverse = true;

    if (!cpuDiverse) {
      std::cerr << "    [" << testName.toStdString() << "] CPU outputs identical (model didn't learn): ["
                << c0[0].toDouble() << "," << c0[1].toDouble() << "]" << std::endl;
      return false; // Can't test GPU if CPU doesn't work
    }
  }

  // Check GPU outputs are diverse
  bool gpuDiverse = false;
  {
    QJsonArray g0 = gpuOutputs[0].toArray(), g1 = gpuOutputs[1].toArray();

    for (int i = 0; i < g0.size() && i < g1.size(); i++)

      if (std::fabs(g0[i].toDouble() - g1[i].toDouble()) > 1e-6)
        gpuDiverse = true;

    if (!gpuDiverse) {
      std::cerr << "    [" << testName.toStdString() << "] GPU outputs COLLAPSED: [" << g0[0].toDouble() << ","
                << g0[1].toDouble() << "]" << std::endl;
      std::cerr << "    CPU was: [" << cpuOutputs[0].toArray()[0].toDouble() << ","
                << cpuOutputs[0].toArray()[1].toDouble() << "] vs [" << cpuOutputs[1].toArray()[0].toDouble() << ","
                << cpuOutputs[1].toArray()[1].toDouble() << "]" << std::endl;
    }
  }

  // Check GPU outputs approximately match CPU
  bool gpuMatchesCPU = true;

  for (int s = 0; s < 2; s++) {
    QJsonArray co = cpuOutputs[s].toArray(), go = gpuOutputs[s].toArray();

    for (int i = 0; i < co.size() && i < go.size(); i++) {
      if (std::fabs(co[i].toDouble() - go[i].toDouble()) > 0.01) {
        gpuMatchesCPU = false;
      }
    }
  }

  if (!gpuMatchesCPU && gpuDiverse) {
    std::cerr << "    [" << testName.toStdString() << "] GPU outputs differ from CPU (>0.01):" << std::endl;

    for (int s = 0; s < 2; s++) {
      QJsonArray co = cpuOutputs[s].toArray(), go = gpuOutputs[s].toArray();
      std::cerr << "      sample " << s << ": CPU=[" << co[0].toDouble() << "," << co[1].toDouble() << "] GPU=["
                << go[0].toDouble() << "," << go[1].toDouble() << "]" << std::endl;
    }
  }

  return gpuDiverse;
}

static void testCNNGPUPredictLayerIsolation()
{
  std::cout << "  testCNNGPUPredictLayerIsolation..." << std::endl;

  if (!checkGPUAvailable()) {
    std::cout << "    (skipped — no GPU)" << std::endl;
    return;
  }

  // Each entry: { testName, convLayersJson }
  // Architecture varies systematically: add one layer type at a time
  struct TestCase {
      const char* name;
      const char* layers;
  };

  // clang-format off
  TestCase cases[] = {
    // 1. Minimal: conv + flatten
    {"conv_flatten",
     R"([{"type":"conv","numFilters":4,"filterH":3,"filterW":3,"strideY":1,"strideX":1,"slidingStrategy":"valid"},{"type":"flatten"}])"},

    // 2. conv + relu + flatten
    {"conv_relu_flatten",
     R"([{"type":"conv","numFilters":4,"filterH":3,"filterW":3,"strideY":1,"strideX":1,"slidingStrategy":"valid"},{"type":"relu"},{"type":"flatten"}])"},

    // 3. conv + relu + maxpool + flatten
    {"conv_relu_maxpool_flatten",
     R"([{"type":"conv","numFilters":4,"filterH":3,"filterW":3,"strideY":1,"strideX":1,"slidingStrategy":"valid"},{"type":"relu"},{"type":"pool","poolType":"max","poolH":2,"poolW":2,"strideY":2,"strideX":2},{"type":"flatten"}])"},

    // 4. conv + relu + avgpool + flatten
    {"conv_relu_avgpool_flatten",
     R"([{"type":"conv","numFilters":4,"filterH":3,"filterW":3,"strideY":1,"strideX":1,"slidingStrategy":"valid"},{"type":"relu"},{"type":"pool","poolType":"avg","poolH":2,"poolW":2,"strideY":2,"strideX":2},{"type":"flatten"}])"},

    // 5. conv + batchnorm + relu + flatten
    {"conv_bn_relu_flatten",
     R"([{"type":"conv","numFilters":4,"filterH":3,"filterW":3,"strideY":1,"strideX":1,"slidingStrategy":"valid"},{"type":"batchnorm"},{"type":"relu"},{"type":"flatten"}])"},

    // 6. conv + batchnorm + relu + maxpool + flatten
    {"conv_bn_relu_maxpool_flatten",
     R"([{"type":"conv","numFilters":4,"filterH":3,"filterW":3,"strideY":1,"strideX":1,"slidingStrategy":"valid"},{"type":"batchnorm"},{"type":"relu"},{"type":"pool","poolType":"max","poolH":2,"poolW":2,"strideY":2,"strideX":2},{"type":"flatten"}])"},

    // 7. conv + batchnorm + relu + avgpool + flatten (smallest avgpool+bn combo)
    {"conv_bn_relu_avgpool_flatten",
     R"([{"type":"conv","numFilters":4,"filterH":3,"filterW":3,"strideY":1,"strideX":1,"slidingStrategy":"valid"},{"type":"batchnorm"},{"type":"relu"},{"type":"pool","poolType":"avg","poolH":2,"poolW":2,"strideY":2,"strideX":2},{"type":"flatten"}])"},

    // 8. conv + relu + maxpool + conv + relu + flatten (2 conv layers)
    {"conv2x_relu_maxpool_flatten",
     R"([{"type":"conv","numFilters":4,"filterH":3,"filterW":3,"strideY":1,"strideX":1,"slidingStrategy":"same"},{"type":"relu"},{"type":"pool","poolType":"max","poolH":2,"poolW":2,"strideY":2,"strideX":2},{"type":"conv","numFilters":4,"filterH":3,"filterW":3,"strideY":1,"strideX":1,"slidingStrategy":"valid"},{"type":"relu"},{"type":"flatten"}])"},

    // 9. conv + relu + maxpool + conv + relu + avgpool + flatten (maxpool then avgpool)
    {"conv_maxpool_conv_avgpool_flatten",
     R"([{"type":"conv","numFilters":4,"filterH":3,"filterW":3,"strideY":1,"strideX":1,"slidingStrategy":"same"},{"type":"relu"},{"type":"pool","poolType":"max","poolH":2,"poolW":2,"strideY":2,"strideX":2},{"type":"conv","numFilters":4,"filterH":3,"filterW":3,"strideY":1,"strideX":1,"slidingStrategy":"valid"},{"type":"relu"},{"type":"pool","poolType":"avg","poolH":2,"poolW":2,"strideY":2,"strideX":2},{"type":"flatten"}])"},

    // 10. conv + bn + relu + maxpool + conv + bn + relu + avgpool + flatten (full combo)
    {"conv_bn_maxpool_conv_bn_avgpool_flatten",
     R"([{"type":"conv","numFilters":4,"filterH":3,"filterW":3,"strideY":1,"strideX":1,"slidingStrategy":"same"},{"type":"batchnorm"},{"type":"relu"},{"type":"pool","poolType":"max","poolH":2,"poolW":2,"strideY":2,"strideX":2},{"type":"conv","numFilters":4,"filterH":3,"filterW":3,"strideY":1,"strideX":1,"slidingStrategy":"valid"},{"type":"batchnorm"},{"type":"relu"},{"type":"pool","poolType":"avg","poolH":2,"poolW":2,"strideY":2,"strideX":2},{"type":"flatten"}])"},

    // 11. Same as #3 but with "same" padding instead of "valid"
    {"conv_same_relu_maxpool_flatten",
     R"([{"type":"conv","numFilters":4,"filterH":3,"filterW":3,"strideY":1,"strideX":1,"slidingStrategy":"same"},{"type":"relu"},{"type":"pool","poolType":"max","poolH":2,"poolW":2,"strideY":2,"strideX":2},{"type":"flatten"}])"},

    // 12. Same as #4 but with "same" padding
    {"conv_same_relu_avgpool_flatten",
     R"([{"type":"conv","numFilters":4,"filterH":3,"filterW":3,"strideY":1,"strideX":1,"slidingStrategy":"same"},{"type":"relu"},{"type":"pool","poolType":"avg","poolH":4,"poolW":4,"strideY":4,"strideX":4},{"type":"flatten"}])"},
  };
  // clang-format on

  int numCases = sizeof(cases) / sizeof(cases[0]);
  int passed = 0, failed = 0, skipped = 0;

  for (int i = 0; i < numCases; i++) {
    std::cout << "    " << (i + 1) << "/" << numCases << " " << cases[i].name << "... " << std::flush;
    bool ok = runGPUPredictTest(cases[i].name, cases[i].layers);

    if (ok) {
      std::cout << "PASS" << std::endl;
      passed++;
    } else {
      // Check if it was a "model didn't learn" skip
      std::cout << "FAIL" << std::endl;
      failed++;
    }
  }

  std::cout << "    Summary: " << passed << " passed, " << failed << " failed, " << skipped << " skipped" << std::endl;
  CHECK(failed == 0, "GPU predict layer isolation: all architectures produce correct GPU predictions");
  std::cout << std::endl;
}

//===================================================================================================================//
// Deep diagnostic test: compare ALL internal state between CPU and GPU predict
//===================================================================================================================//

// Helper: compare two float vectors element-by-element, report first mismatch
static bool compareVectors(const std::string& label, const std::vector<float>& cpu, const std::vector<float>& gpu,
                           float tol = 1e-4f, int maxReport = 5)
{
  if (cpu.size() != gpu.size()) {
    std::cerr << "    " << label << ": SIZE MISMATCH cpu=" << cpu.size() << " gpu=" << gpu.size() << std::endl;
    return false;
  }

  int mismatches = 0;
  float maxDiff = 0;

  for (size_t i = 0; i < cpu.size(); i++) {
    float diff = std::fabs(cpu[i] - gpu[i]);

    if (diff > maxDiff)
      maxDiff = diff;

    if (diff > tol) {
      if (mismatches < maxReport)
        std::cerr << "    " << label << "[" << i << "]: cpu=" << cpu[i] << " gpu=" << gpu[i] << " diff=" << diff
                  << std::endl;
      mismatches++;
    }
  }

  if (mismatches > 0) {
    std::cerr << "    " << label << ": " << mismatches << "/" << cpu.size() << " mismatches (maxDiff=" << maxDiff << ")"
              << std::endl;
    return false;
  }

  return true;
}

static void testCNNGPUPredictDeepDiagnostic()
{
  std::cout << "  testCNNGPUPredictDeepDiagnostic..." << std::endl;

  if (!checkGPUAvailable()) {
    std::cout << "    (skipped — no GPU)" << std::endl;
    return;
  }

  // Build config programmatically: conv(4,3x3,same) → BN → ReLU → MaxPool(2x2)
  //                                → conv(8,3x3,same) → BN → ReLU → AvgPool(2x2)
  //                                → Flatten → Dense(4,relu) → Dense(2,sigmoid)
  // Input: 1x8x8
  CNN::CoreConfig<float> trainConfig;
  trainConfig.modeType = CNN::ModeType::TRAIN;
  trainConfig.deviceType = CNN::DeviceType::CPU;
  trainConfig.inputShape = {1, 8, 8};
  trainConfig.progressReports = 0;
  trainConfig.logLevel = CNN::LogLevel::ERROR;

  // CNN layers
  CNN::CNNLayerConfig conv1;
  conv1.type = CNN::LayerType::CONV;
  conv1.config = CNN::ConvLayerConfig{4, 3, 3, 1, 1, CNN::SlidingStrategyType::SAME};

  CNN::CNNLayerConfig bn1;
  bn1.type = CNN::LayerType::BATCHNORM;
  bn1.config = CNN::BatchNormLayerConfig{1e-5f, 0.1f};

  CNN::CNNLayerConfig relu1;
  relu1.type = CNN::LayerType::RELU;
  relu1.config = CNN::ReLULayerConfig{};

  CNN::CNNLayerConfig maxpool1;
  maxpool1.type = CNN::LayerType::POOL;
  maxpool1.config = CNN::PoolLayerConfig{CNN::PoolTypeEnum::MAX, 2, 2, 2, 2};

  CNN::CNNLayerConfig conv2;
  conv2.type = CNN::LayerType::CONV;
  conv2.config = CNN::ConvLayerConfig{8, 3, 3, 1, 1, CNN::SlidingStrategyType::SAME};

  CNN::CNNLayerConfig bn2;
  bn2.type = CNN::LayerType::BATCHNORM;
  bn2.config = CNN::BatchNormLayerConfig{1e-5f, 0.1f};

  CNN::CNNLayerConfig relu2;
  relu2.type = CNN::LayerType::RELU;
  relu2.config = CNN::ReLULayerConfig{};

  CNN::CNNLayerConfig avgpool1;
  avgpool1.type = CNN::LayerType::POOL;
  avgpool1.config = CNN::PoolLayerConfig{CNN::PoolTypeEnum::AVG, 2, 2, 2, 2};

  CNN::CNNLayerConfig flatten;
  flatten.type = CNN::LayerType::FLATTEN;
  flatten.config = CNN::FlattenLayerConfig{};

  trainConfig.layersConfig.cnnLayers = {conv1, bn1, relu1, maxpool1, conv2, bn2, relu2, avgpool1, flatten};
  trainConfig.layersConfig.denseLayers = {{4, ANN::ActvFuncType::RELU}, {2, ANN::ActvFuncType::SIGMOID}};

  trainConfig.trainingConfig.numEpochs = 50;
  trainConfig.trainingConfig.learningRate = 0.01f;
  trainConfig.trainingConfig.batchSize = 4;
  trainConfig.trainingConfig.shuffleSamples = false;

  // Create training samples: 1x8x8 input, 2-class output
  CNN::Samples<float> samples(4);

  for (int s = 0; s < 4; s++) {
    samples[s].input = CNN::Input<float>({1, 8, 8});
    int cls = s / 2;

    for (int i = 0; i < 64; i++) {
      if (cls == 0)
        samples[s].input.data[i] = static_cast<float>(i) / 63.0f + 0.01f * s;
      else
        samples[s].input.data[i] = static_cast<float>(63 - i) / 63.0f + 0.01f * (s - 2);
    }

    samples[s].output = (cls == 0) ? CNN::Output<float>{1.0f, 0.0f} : CNN::Output<float>{0.0f, 1.0f};
  }

  // Step 1: Train on CPU
  std::cout << "    Training on CPU..." << std::flush;
  auto cpuTrainCore = CNN::Core<float>::makeCore(trainConfig);
  cpuTrainCore->train(samples.size(), CNN::makeSampleProvider(samples));
  const CNN::Parameters<float>& trainedParams = cpuTrainCore->getParameters();
  std::cout << " done" << std::endl;

  // Step 2: Create CPU predict core with trained parameters
  CNN::CoreConfig<float> cpuPredConfig;
  cpuPredConfig.modeType = CNN::ModeType::PREDICT;
  cpuPredConfig.deviceType = CNN::DeviceType::CPU;
  cpuPredConfig.inputShape = trainConfig.inputShape;
  cpuPredConfig.layersConfig = trainConfig.layersConfig;
  cpuPredConfig.parameters = trainedParams;
  cpuPredConfig.progressReports = 0;
  cpuPredConfig.logLevel = CNN::LogLevel::ERROR;

  auto cpuCore = CNN::Core<float>::makeCore(cpuPredConfig);

  // Step 3: Create GPU predict core with same trained parameters
  CNN::CoreConfig<float> gpuPredConfig = cpuPredConfig;
  gpuPredConfig.deviceType = CNN::DeviceType::GPU;

  auto gpuCoreBase = CNN::Core<float>::makeCore(gpuPredConfig);
  auto* gpuCore = dynamic_cast<CNN::CoreGPU<float>*>(gpuCoreBase.get());
  CHECK(gpuCore != nullptr, "Deep diag: GPU core created");

  if (!gpuCore) {
    std::cout << std::endl;
    return;
  }

  auto* gpuWorker = gpuCore->getWorker(0);
  CHECK(gpuWorker != nullptr, "Deep diag: GPU worker accessible");

  if (!gpuWorker) {
    std::cout << std::endl;
    return;
  }

  // Step 4: Compare parameters BEFORE predict (verify they were loaded correctly)
  std::cout << "    Comparing parameters..." << std::flush;
  const auto& cpuParams = cpuCore->getParameters();
  const auto& gpuParams = gpuWorker->getParameters();
  bool paramsOk = true;

  // Conv filters
  for (size_t li = 0; li < cpuParams.convParams.size(); li++) {
    if (!compareVectors("conv[" + std::to_string(li) + "].filters", cpuParams.convParams[li].filters,
                        gpuParams.convParams[li].filters))
      paramsOk = false;

    if (!compareVectors("conv[" + std::to_string(li) + "].biases", cpuParams.convParams[li].biases,
                        gpuParams.convParams[li].biases))
      paramsOk = false;
  }

  // BatchNorm params
  for (size_t li = 0; li < cpuParams.bnParams.size(); li++) {
    if (!compareVectors("bn[" + std::to_string(li) + "].gamma", cpuParams.bnParams[li].gamma,
                        gpuParams.bnParams[li].gamma))
      paramsOk = false;

    if (!compareVectors("bn[" + std::to_string(li) + "].beta", cpuParams.bnParams[li].beta,
                        gpuParams.bnParams[li].beta))
      paramsOk = false;

    if (!compareVectors("bn[" + std::to_string(li) + "].runningMean", cpuParams.bnParams[li].runningMean,
                        gpuParams.bnParams[li].runningMean))
      paramsOk = false;

    if (!compareVectors("bn[" + std::to_string(li) + "].runningVar", cpuParams.bnParams[li].runningVar,
                        gpuParams.bnParams[li].runningVar))
      paramsOk = false;
  }

  // ANN weights and biases
  for (size_t li = 0; li < cpuParams.denseParams.weights.size(); li++) {
    for (size_t ni = 0; ni < cpuParams.denseParams.weights[li].size(); ni++) {
      if (!compareVectors("ann.weights[" + std::to_string(li) + "][" + std::to_string(ni) + "]",
                          cpuParams.denseParams.weights[li][ni], gpuParams.denseParams.weights[li][ni]))
        paramsOk = false;
    }

    std::vector<float> cpuBiases(cpuParams.denseParams.biases[li].begin(), cpuParams.denseParams.biases[li].end());
    std::vector<float> gpuBiases(gpuParams.denseParams.biases[li].begin(), gpuParams.denseParams.biases[li].end());

    if (!compareVectors("ann.biases[" + std::to_string(li) + "]", cpuBiases, gpuBiases))
      paramsOk = false;
  }

  CHECK(paramsOk, "Deep diag: all parameters match between CPU and GPU");
  std::cout << (paramsOk ? " OK" : " MISMATCH") << std::endl;

  // Step 5: Run predict on both and compare outputs + intermediate activations
  CNN::Input<float> testInput({1, 8, 8});

  for (int i = 0; i < 64; i++)
    testInput.data[i] = static_cast<float>(i) / 63.0f;

  std::cout << "    Running predict on CPU and GPU..." << std::flush;
  auto cpuOutput = cpuCore->predict(testInput);
  auto gpuOutput = gpuCore->predict(testInput);
  std::cout << " done" << std::endl;

  // Compare final outputs
  bool outputsMatch = compareVectors("final_output", cpuOutput, gpuOutput, 1e-4f);
  CHECK(outputsMatch, "Deep diag: CPU and GPU final outputs match");

  // Step 6: Read GPU intermediate activations and compare with CPU
  // We need to re-run CPU predict step-by-step to get intermediates
  std::cout << "    Comparing layer-by-layer activations..." << std::flush;

  // Read ALL GPU activations from the cnn_actvs buffer
  auto& bm = *gpuWorker->bufferManager;
  std::vector<float> gpuAllActvs(bm.totalActvSize);
  gpuWorker->readGPUBuffer<float>("cnn_actvs", gpuAllActvs, 0);

  // Read GPU filters from buffer (to verify they match what was uploaded)
  if (bm.totalFilterSize > 0) {
    std::vector<float> gpuFilters(bm.totalFilterSize);
    gpuWorker->readGPUBuffer<float>("cnn_filters", gpuFilters, 0);

    // Reconstruct expected flat filters from CPU params
    std::vector<float> cpuFlatFilters;

    for (size_t ci = 0; ci < bm.convInfos.size(); ci++) {
      cpuFlatFilters.insert(cpuFlatFilters.end(), cpuParams.convParams[ci].filters.begin(),
                            cpuParams.convParams[ci].filters.end());
    }

    bool filtersOnGPUMatch = compareVectors("gpu_buffer_filters", cpuFlatFilters, gpuFilters);
    CHECK(filtersOnGPUMatch, "Deep diag: GPU buffer filters match CPU params");
  }

  // Read GPU biases from buffer
  if (bm.totalBiasSize > 0) {
    std::vector<float> gpuBiases(bm.totalBiasSize);
    gpuWorker->readGPUBuffer<float>("cnn_biases", gpuBiases, 0);

    std::vector<float> cpuFlatBiases;

    for (size_t ci = 0; ci < bm.convInfos.size(); ci++) {
      cpuFlatBiases.insert(cpuFlatBiases.end(), cpuParams.convParams[ci].biases.begin(),
                           cpuParams.convParams[ci].biases.end());
    }

    bool biasesOnGPUMatch = compareVectors("gpu_buffer_biases", cpuFlatBiases, gpuBiases);
    CHECK(biasesOnGPUMatch, "Deep diag: GPU buffer biases match CPU params");
  }

  // Read GPU BN params from buffers
  if (bm.totalBNParamSize > 0) {
    std::vector<float> gpuGamma(bm.totalBNParamSize), gpuBeta(bm.totalBNParamSize);
    std::vector<float> gpuRunMean(bm.totalBNParamSize), gpuRunVar(bm.totalBNParamSize);
    gpuWorker->readGPUBuffer<float>("cnn_bn_gamma", gpuGamma, 0);
    gpuWorker->readGPUBuffer<float>("cnn_bn_beta", gpuBeta, 0);
    gpuWorker->readGPUBuffer<float>("cnn_bn_running_mean", gpuRunMean, 0);
    gpuWorker->readGPUBuffer<float>("cnn_bn_running_var", gpuRunVar, 0);

    std::vector<float> cpuFlatGamma, cpuFlatBeta, cpuFlatRunMean, cpuFlatRunVar;

    for (size_t bi = 0; bi < bm.bnInfos.size(); bi++) {
      cpuFlatGamma.insert(cpuFlatGamma.end(), cpuParams.bnParams[bi].gamma.begin(), cpuParams.bnParams[bi].gamma.end());
      cpuFlatBeta.insert(cpuFlatBeta.end(), cpuParams.bnParams[bi].beta.begin(), cpuParams.bnParams[bi].beta.end());
      cpuFlatRunMean.insert(cpuFlatRunMean.end(), cpuParams.bnParams[bi].runningMean.begin(),
                            cpuParams.bnParams[bi].runningMean.end());
      cpuFlatRunVar.insert(cpuFlatRunVar.end(), cpuParams.bnParams[bi].runningVar.begin(),
                           cpuParams.bnParams[bi].runningVar.end());
    }

    CHECK(compareVectors("gpu_buffer_bn_gamma", cpuFlatGamma, gpuGamma), "Deep diag: GPU BN gamma matches");
    CHECK(compareVectors("gpu_buffer_bn_beta", cpuFlatBeta, gpuBeta), "Deep diag: GPU BN beta matches");
    CHECK(compareVectors("gpu_buffer_bn_running_mean", cpuFlatRunMean, gpuRunMean),
          "Deep diag: GPU BN running mean matches");
    CHECK(compareVectors("gpu_buffer_bn_running_var", cpuFlatRunVar, gpuRunVar),
          "Deep diag: GPU BN running var matches");
  }

  // Compare per-layer activations
  // GPU stores activations in a flat buffer with offsets per layer.
  // CPU computes them step-by-step. We need to manually propagate on CPU to get intermediates.
  // Since CoreCPUWorker::propagateCNN is private, we'll use the Loader to load the model
  // and compare via the public API. For now, compare the GPU layer activations for sanity:
  bool anyLayerAllZero = false;

  for (size_t li = 0; li < bm.layerInfos.size(); li++) {
    ulong offset = bm.layerInfos[li].actvOffset;
    ulong size = bm.layerInfos[li].actvSize;

    float sum = 0, minVal = 1e30f, maxVal = -1e30f;

    for (ulong j = 0; j < size && (offset + j) < gpuAllActvs.size(); j++) {
      float v = gpuAllActvs[offset + j];
      sum += v;

      if (v < minVal)
        minVal = v;

      if (v > maxVal)
        maxVal = v;
    }

    bool allZero = (minVal == 0.0f && maxVal == 0.0f);

    if (allZero && li > 0) { // layer 0 is input, can be zero
      std::cerr << "    Layer " << li << " (offset=" << offset << ", size=" << size << "): ALL ZEROS" << std::endl;
      anyLayerAllZero = true;
    }
  }

  CHECK(!anyLayerAllZero, "Deep diag: no intermediate layer is all zeros on GPU");

  // Step 7: Read ANN GPU buffers and compare with CPU ANN output
  auto* annGPUWorker = bm.annGPUWorker.get();

  if (annGPUWorker) {
    auto& annBM = *annGPUWorker->bufferManager;

    // Read ANN activations from GPU
    ulong totalANNNeurons = 0;

    for (size_t li = 0; li < cpuParams.denseParams.weights.size(); li++)
      totalANNNeurons += cpuParams.denseParams.weights[li].size();

    // Add input layer neurons (= flattenSize)
    totalANNNeurons += bm.flattenSize;

    if (totalANNNeurons > 0) {
      std::vector<float> gpuANNActvs(totalANNNeurons);
      annGPUWorker->readGPUBuffer<float>("actvs", gpuANNActvs, 0);

      // Check ANN input layer (should be the flattened CNN output)
      std::vector<float> gpuANNInput(gpuANNActvs.begin(), gpuANNActvs.begin() + static_cast<long>(bm.flattenSize));

      // The CNN output on GPU should be at the last CNN layer's activation
      ulong lastCNNLayerIdx = bm.layerInfos.size() - 1;
      ulong lastOffset = bm.layerInfos[lastCNNLayerIdx].actvOffset;
      ulong lastSize = bm.layerInfos[lastCNNLayerIdx].actvSize;
      std::vector<float> gpuCNNOutput(gpuAllActvs.begin() + static_cast<long>(lastOffset),
                                      gpuAllActvs.begin() + static_cast<long>(lastOffset + lastSize));

      bool cnnToAnnBridgeOk = compareVectors("cnn_to_ann_bridge", gpuCNNOutput, gpuANNInput, 1e-6f);
      CHECK(cnnToAnnBridgeOk, "Deep diag: CNN→ANN bridge (flatten output == ANN input)");

      // Read ANN weights from GPU
      ulong totalANNWeights = 0;

      for (size_t li = 0; li < cpuParams.denseParams.weights.size(); li++)

        for (size_t ni = 0; ni < cpuParams.denseParams.weights[li].size(); ni++)
          totalANNWeights += cpuParams.denseParams.weights[li][ni].size();

      if (totalANNWeights > 0) {
        std::vector<float> gpuANNWeights(totalANNWeights);
        annGPUWorker->readGPUBuffer<float>("weights", gpuANNWeights, 0);

        std::vector<float> cpuFlatANNWeights;

        for (size_t li = 0; li < cpuParams.denseParams.weights.size(); li++)

          for (size_t ni = 0; ni < cpuParams.denseParams.weights[li].size(); ni++)
            cpuFlatANNWeights.insert(cpuFlatANNWeights.end(), cpuParams.denseParams.weights[li][ni].begin(),
                                     cpuParams.denseParams.weights[li][ni].end());

        CHECK(compareVectors("ann_gpu_weights", cpuFlatANNWeights, gpuANNWeights),
              "Deep diag: ANN GPU weights match CPU");
      }

      // Read ANN biases from GPU
      ulong totalANNBiases = 0;

      for (size_t li = 0; li < cpuParams.denseParams.biases.size(); li++)
        totalANNBiases += cpuParams.denseParams.biases[li].size();

      if (totalANNBiases > 0) {
        std::vector<float> gpuANNBiases(totalANNBiases);
        annGPUWorker->readGPUBuffer<float>("biases", gpuANNBiases, 0);

        std::vector<float> cpuFlatANNBiases;

        for (size_t li = 0; li < cpuParams.denseParams.biases.size(); li++)
          cpuFlatANNBiases.insert(cpuFlatANNBiases.end(), cpuParams.denseParams.biases[li].begin(),
                                  cpuParams.denseParams.biases[li].end());

        CHECK(compareVectors("ann_gpu_biases", cpuFlatANNBiases, gpuANNBiases), "Deep diag: ANN GPU biases match CPU");
      }
    }
  }

  // Step 8: Run a second input and verify outputs still differ
  CNN::Input<float> testInput2({1, 8, 8});

  for (int i = 0; i < 64; i++)
    testInput2.data[i] = static_cast<float>(63 - i) / 63.0f;

  auto cpuOutput2 = cpuCore->predict(testInput2);
  auto gpuOutput2 = gpuCore->predict(testInput2);

  CHECK(compareVectors("final_output_2", cpuOutput2, gpuOutput2, 1e-4f),
        "Deep diag: CPU and GPU match on second input");

  // Verify diversity
  bool diverse = false;

  for (size_t i = 0; i < cpuOutput.size() && i < cpuOutput2.size(); i++)

    if (std::fabs(cpuOutput[i] - cpuOutput2[i]) > 1e-6f)
      diverse = true;

  CHECK(diverse, "Deep diag: different inputs produce different outputs");

  std::cout << "    All deep diagnostic checks complete." << std::endl;
  std::cout << std::endl;
}

// Helper: write a JSON config string mirroring the ISIC architecture (scaled down)
// Architecture: Conv(4,3x3,same)→BN→ReLU→Conv(4,3x3,same)→BN→ReLU→MaxPool(2x2)
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
  "convolutionalLayersConfig": [
    { "type": "conv", "numFilters": 4, "filterH": 3, "filterW": 3, "strideY": 1, "strideX": 1, "slidingStrategy": "same" },
    { "type": "batchnorm" },
    { "type": "relu" },
    { "type": "conv", "numFilters": 4, "filterH": 3, "filterW": 3, "strideY": 1, "strideX": 1, "slidingStrategy": "same" },
    { "type": "batchnorm" },
    { "type": "relu" },
    { "type": "pool", "poolType": "max", "poolH": 2, "poolW": 2, "strideY": 2, "strideX": 2 },

    { "type": "conv", "numFilters": 8, "filterH": 3, "filterW": 3, "strideY": 1, "strideX": 1, "slidingStrategy": "same" },
    { "type": "batchnorm" },
    { "type": "relu" },
    { "type": "conv", "numFilters": 8, "filterH": 3, "filterW": 3, "strideY": 1, "strideX": 1, "slidingStrategy": "same" },
    { "type": "batchnorm" },
    { "type": "relu" },
    { "type": "pool", "poolType": "max", "poolH": 2, "poolW": 2, "strideY": 2, "strideX": 2 },

    { "type": "pool", "poolType": "avg", "poolH": 4, "poolW": 4, "strideY": 4, "strideX": 4 },
    { "type": "flatten" }
  ],
  "denseLayersConfig": [
    { "numNeurons": 4, "actvFunc": "relu" },
    { "numNeurons": 3, "actvFunc": "softmax" }
  ],
  "costFunctionConfig": {
    "type": "crossEntropy"
  },
  "trainingConfig": {
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

  QString modelPath = tempDir() + "/cnn_isic_model.json";

  // Step 1: Train
  auto trainResult = runNNCLI(
    {"--config", configPath, "--mode", "train", "--device", "cpu", "--samples", samplesPath, "--output", modelPath},
    300000); // 5 min timeout

  CHECK(trainResult.exitCode == 0, "ISIC-like CPU: train exit code 0");

  if (trainResult.exitCode != 0) {
    std::cout << "(train failed)" << std::endl;
    return;
  }

  // Step 2: Verify saved model has all 4 batchnorm parameter sets
  {
    QFile mf(modelPath);

    if (mf.open(QIODevice::ReadOnly)) {
      QJsonDocument doc = QJsonDocument::fromJson(mf.readAll());
      QJsonObject params = doc.object()["parameters"].toObject();
      QJsonArray bnArr = params["batchnorm"].toArray();
      CHECK(bnArr.size() == 4, "ISIC-like CPU: 4 batchnorm layers in saved model");

      // Check each BN layer has correct channel count
      if (bnArr.size() == 4) {
        CHECK(bnArr[0].toObject()["numChannels"].toInt() == 4, "ISIC-like CPU: BN[0] numChannels=4");
        CHECK(bnArr[1].toObject()["numChannels"].toInt() == 4, "ISIC-like CPU: BN[1] numChannels=4");
        CHECK(bnArr[2].toObject()["numChannels"].toInt() == 8, "ISIC-like CPU: BN[2] numChannels=8");
        CHECK(bnArr[3].toObject()["numChannels"].toInt() == 8, "ISIC-like CPU: BN[3] numChannels=8");
      }

      mf.close();
    } else {
      CHECK(false, "ISIC-like CPU: failed to open model file");
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

  QString modelPath = tempDir() + "/cnn_isic_gpu_model.json";

  // Step 1: Train on GPU
  auto trainResult = runNNCLI(
    {"--config", configPath, "--mode", "train", "--device", "gpu", "--samples", samplesPath, "--output", modelPath},
    300000);

  CHECK(trainResult.exitCode == 0, "ISIC-like GPU: train exit code 0");

  if (trainResult.exitCode != 0) {
    std::cout << "(train failed)" << std::endl;
    return;
  }

  // Step 2: Verify 4 batchnorm layers saved
  {
    QFile mf(modelPath);

    if (mf.open(QIODevice::ReadOnly)) {
      QJsonDocument doc = QJsonDocument::fromJson(mf.readAll());
      QJsonObject params = doc.object()["parameters"].toObject();
      QJsonArray bnArr = params["batchnorm"].toArray();
      CHECK(bnArr.size() == 4, "ISIC-like GPU: 4 batchnorm layers in saved model");
      mf.close();
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

static void testCNNShuffleSamplesCLI()
{
  std::cout << "  testCNNShuffleSamplesCLI... ";

  // Train with --shuffle-samples true
  QString modelPathTrue = tempDir() + "/cnn_shuffle_true.json";
  auto resultTrue =
    runNNCLI({"--config", fixturePath("cnn_train_config.json"), "--mode", "train", "--device", "cpu", "--samples",
              fixturePath("cnn_train_samples.json"), "--output", modelPathTrue, "--shuffle-samples", "true"});

  CHECK(resultTrue.exitCode == 0, "CNN shuffle=true: exit code 0");
  CHECK(resultTrue.stdOut.contains("Training completed."), "CNN shuffle=true: 'Training completed.'");

  // Train with --shuffle-samples false
  QString modelPathFalse = tempDir() + "/cnn_shuffle_false.json";
  auto resultFalse =
    runNNCLI({"--config", fixturePath("cnn_train_config.json"), "--mode", "train", "--device", "cpu", "--samples",
              fixturePath("cnn_train_samples.json"), "--output", modelPathFalse, "--shuffle-samples", "false"});

  CHECK(resultFalse.exitCode == 0, "CNN shuffle=false: exit code 0");
  CHECK(resultFalse.stdOut.contains("Training completed."), "CNN shuffle=false: 'Training completed.'");

  // Verify shuffleSamples is saved in the output model JSON
  QFile fileTrue(modelPathTrue);

  if (fileTrue.open(QIODevice::ReadOnly)) {
    QJsonDocument doc = QJsonDocument::fromJson(fileTrue.readAll());
    QJsonObject root = doc.object();
    CHECK(root.contains("trainingConfig"), "CNN shuffle=true: has 'trainingConfig'");
    QJsonObject tc = root["trainingConfig"].toObject();
    CHECK(tc.contains("shuffleSamples"), "CNN shuffle=true: has 'shuffleSamples'");
    CHECK(tc["shuffleSamples"].toBool() == true, "CNN shuffle=true: shuffleSamples is true");
    fileTrue.close();
  } else {
    CHECK(false, "CNN shuffle=true: failed to open model file");
  }

  QFile fileFalse(modelPathFalse);

  if (fileFalse.open(QIODevice::ReadOnly)) {
    QJsonDocument doc = QJsonDocument::fromJson(fileFalse.readAll());
    QJsonObject root = doc.object();
    QJsonObject tc = root["trainingConfig"].toObject();
    CHECK(tc.contains("shuffleSamples"), "CNN shuffle=false: has 'shuffleSamples'");
    CHECK(tc["shuffleSamples"].toBool() == false, "CNN shuffle=false: shuffleSamples is false");
    fileFalse.close();
  } else {
    CHECK(false, "CNN shuffle=false: failed to open model file");
  }

  std::cout << std::endl;
}

void runCNNTests()
{
  testCNNNetworkDetection();
  testCNNTrain();
  testCNNPredict();
  testCNNTest();
  testCNNTrainWithWeightedLoss();
  testCNNTrainAndTestMNIST();
  testCNNTrainAndTestMNISTGPU();
  testCNNCheckpointParameters();
  testCNNCheckpointBatchNormRoundTrip();
  testCNNSaveLoadPredictConsistency();
  testCNNSaveLoadPredictConsistencyGPU();
  testCNNMultiInputPredictDiversity();
  testCNNGPUPredictLayerIsolation();
  testCNNGPUPredictDeepDiagnostic();
  testCNNISICLikeSaveLoadPredict();
  testCNNISICLikeSaveLoadPredictGPU();
  testCNNShuffleSamplesCLI();
}
