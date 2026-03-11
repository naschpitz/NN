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

  // Step 1: Train on MNIST training data on GPU (10 epochs, 60k samples, Adam + crossEntropy + instancenorm)
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

void runCNNFeaturesTests()
{
  testCNNTrainAndTestMNISTGPU();
  testCNNCheckpointParameters();
  testCNNCheckpointInstanceNormRoundTrip();
}
