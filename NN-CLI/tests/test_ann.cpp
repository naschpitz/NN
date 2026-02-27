#include "test_helpers.hpp"

#include <QDir>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>

// Trained model paths shared between chained tests
QString trainedANNModelPath;                    // XOR model — used by detection/override/error tests
static QString trainedANNMNISTModelPath;        // MNIST model — used by --full predict/test tests

static void testANNNetworkDetection() {
  std::cout << "  testANNNetworkDetection... ";

  if (trainedANNModelPath.isEmpty() || !QFile::exists(trainedANNModelPath)) {
    CHECK(false, "ANN detection: skipped — no trained model available (testANNTrainXOR must run first)");
    std::cout << std::endl;
    return;
  }

  // Create a temporary predict input compatible with XOR model (2 inputs)
  QString predictInputPath = tempDir() + "/ann_detect_input.json";
  QFile inputFile(predictInputPath);
  if (inputFile.open(QIODevice::WriteOnly)) {
    inputFile.write(R"({"inputs": [[0.0, 1.0]]})");
    inputFile.close();
  }

  auto result = runNNCLI({
    "--config", trainedANNModelPath,
    "--mode", "predict",
    "--device", "cpu",
    "--input", predictInputPath,
    "--output", tempDir() + "/ann_detect_output.json",
    "--log-level", "info"
  });

  CHECK(result.exitCode == 0, "ANN detection: exit code 0");
  CHECK(result.stdOut.contains("Network type: ANN"), "ANN detection: stdout contains 'Network type: ANN'");
  std::cout << std::endl;
}

static void testANNTrainXOR() {
  std::cout << "  testANNTrainXOR... ";

  trainedANNModelPath = tempDir() + "/ann_xor_model.json";

  auto result = runNNCLI({
    "--config", fixturePath("ann_train_config.json"),
    "--mode", "train",
    "--device", "cpu",
    "--samples", fixturePath("ann_train_samples.json"),
    "--output", trainedANNModelPath
  });

  CHECK(result.exitCode == 0, "ANN train XOR: exit code 0");
  CHECK(result.stdOut.contains("Training completed."), "ANN train XOR: 'Training completed.'");
  CHECK(result.stdOut.contains("Model saved to:"), "ANN train XOR: 'Model saved to:'");
  CHECK(QFile::exists(trainedANNModelPath), "ANN train XOR: model file exists");

  // Clear the path if training failed so downstream tests skip gracefully
  if (result.exitCode != 0 || !QFile::exists(trainedANNModelPath)) {
    trainedANNModelPath.clear();
  }

  std::cout << std::endl;
}

static void testANNPredictMNIST() {
  std::cout << "  testANNPredictMNIST... " << std::flush;

  if (!runFullTests) {
    std::cout << "(skipped — use --full to enable)" << std::endl;
    return;
  }

  if (trainedANNMNISTModelPath.isEmpty() || !QFile::exists(trainedANNMNISTModelPath)) {
    CHECK(false, "ANN predict MNIST: skipped — no trained MNIST model available (testANNTrainAndTestMNIST must run first)");
    std::cout << std::endl;
    return;
  }

  QString outputPath = tempDir() + "/ann_predict_output.json";

  auto result = runNNCLI({
    "--config", trainedANNMNISTModelPath,
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
  std::cout << "  testANNTestMNIST... " << std::flush;

  if (!runFullTests) {
    std::cout << "(skipped — use --full to enable)" << std::endl;
    return;
  }

  if (trainedANNMNISTModelPath.isEmpty() || !QFile::exists(trainedANNMNISTModelPath)) {
    CHECK(false, "ANN test MNIST: skipped — no trained MNIST model available (testANNTrainAndTestMNIST must run first)");
    std::cout << std::endl;
    return;
  }

  auto result = runNNCLI({
    "--config", trainedANNMNISTModelPath,
    "--mode", "test",
    "--device", "cpu",
    "--idx-data", examplePath("MNIST/test/t10k-images.idx3-ubyte"),
    "--idx-labels", examplePath("MNIST/test/t10k-labels.idx1-ubyte")
  }, 600000);  // 10 min timeout

  CHECK(result.exitCode == 0, "ANN test MNIST: exit code 0");
  CHECK(result.stdOut.contains("Test Results:"), "ANN test MNIST: 'Test Results:'");
  CHECK(result.stdOut.contains("Samples evaluated: 10000"), "ANN test MNIST: 'Samples evaluated: 10000'");
  CHECK(result.stdOut.contains("Total loss:"), "ANN test MNIST: 'Total loss:'");
  CHECK(result.stdOut.contains("Average loss:"), "ANN test MNIST: 'Average loss:'");
  CHECK(result.stdOut.contains("Correct:"), "ANN test MNIST: 'Correct:'");
  CHECK(result.stdOut.contains("Accuracy:"), "ANN test MNIST: 'Accuracy:'");
  std::cout << std::endl;
}

static void testANNModeOverride() {
  std::cout << "  testANNModeOverride... ";

  if (trainedANNModelPath.isEmpty() || !QFile::exists(trainedANNModelPath)) {
    CHECK(false, "ANN mode override: skipped — no trained model available (testANNTrainXOR must run first)");
    std::cout << std::endl;
    return;
  }

  // Create a temporary predict input compatible with XOR model (2 inputs)
  QString predictInputPath = tempDir() + "/ann_override_input.json";
  QFile inputFile(predictInputPath);
  if (inputFile.open(QIODevice::WriteOnly)) {
    inputFile.write(R"({"inputs": [[0.0, 1.0]]})");
    inputFile.close();
  }

  QString outputPath = tempDir() + "/ann_override_output.json";

  // Trained model has mode=train; override to predict via CLI
  auto result = runNNCLI({
    "--config", trainedANNModelPath,
    "--mode", "predict",
    "--device", "cpu",
    "--input", predictInputPath,
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

static void testANNTrainAndTestMNIST() {
  std::cout << "  testANNTrainAndTestMNIST... " << std::flush;

  if (!runFullTests) {
    std::cout << "(skipped — use --full to enable)" << std::endl;
    return;
  }

  trainedANNMNISTModelPath = tempDir() + "/ann_mnist_trained.json";

  // Step 1: Train on MNIST training data on CPU (30 epochs, 60k samples, all cores)
  auto trainResult = runNNCLI({
    "--config", fixturePath("mnist_ann_train_config.json"),
    "--mode", "train",
    "--device", "cpu",
    "--idx-data", examplePath("MNIST/train/train-images.idx3-ubyte"),
    "--idx-labels", examplePath("MNIST/train/train-labels.idx1-ubyte"),
    "--output", trainedANNMNISTModelPath,
    "--log-level", "quiet"
  }, 1800000);  // 30 min timeout

  CHECK(trainResult.exitCode == 0, "ANN MNIST train+test: training exit code 0");
  CHECK(QFile::exists(trainedANNMNISTModelPath), "ANN MNIST train+test: trained model file exists");

  if (trainResult.exitCode != 0 || !QFile::exists(trainedANNMNISTModelPath)) {
    trainedANNMNISTModelPath.clear();
    std::cout << "(training failed, skipping test step)" << std::endl;
    return;
  }

  // Step 2: Evaluate against MNIST test data (10k samples)
  auto testResult = runNNCLI({
    "--config", trainedANNMNISTModelPath,
    "--mode", "test",
    "--device", "cpu",
    "--idx-data", examplePath("MNIST/test/t10k-images.idx3-ubyte"),
    "--idx-labels", examplePath("MNIST/test/t10k-labels.idx1-ubyte")
  }, 600000);  // 10 min timeout

  CHECK(testResult.exitCode == 0, "ANN MNIST train+test: test exit code 0");
  CHECK(testResult.stdOut.contains("Test Results:"), "ANN MNIST train+test: 'Test Results:'");
  CHECK(testResult.stdOut.contains("Samples evaluated: 10000"), "ANN MNIST train+test: 'Samples evaluated: 10000'");

  // Extract and verify average loss is reasonable
  double avgLoss = -1;
  int idx = testResult.stdOut.indexOf("Average loss:");
  if (idx >= 0) {
    QString lossStr = testResult.stdOut.mid(idx + QString("Average loss:").length()).trimmed();
    lossStr = lossStr.left(lossStr.indexOf('\n'));
    avgLoss = lossStr.toDouble();
  }
  CHECK(avgLoss > 0 && avgLoss < 0.5, "ANN MNIST train+test: average loss < 0.5");

  // Extract and verify accuracy is reasonable (> 30% for 30 epochs with mini-batch SGD)
  double accuracy = -1;
  int accIdx = testResult.stdOut.indexOf("Accuracy:");
  if (accIdx >= 0) {
    QString accStr = testResult.stdOut.mid(accIdx + QString("Accuracy:").length()).trimmed();
    accStr = accStr.left(accStr.indexOf('%'));
    accuracy = accStr.toDouble();
  }
  CHECK(accuracy > 30.0, "ANN MNIST train+test: accuracy > 30%");

  std::cout << "(loss=" << avgLoss << ", accuracy=" << accuracy << "%) " << std::endl;
}

//===================================================================================================================//

static void testANNTrainAndTestMNISTGPU() {
  std::cout << "  testANNTrainAndTestMNISTGPU... " << std::flush;

  if (!runFullTests) {
    std::cout << "(skipped — use --full to enable)" << std::endl;
    return;
  }

  if (!checkGPUAvailable()) {
    std::cout << "(skipped — no GPU available)" << std::endl;
    return;
  }

  QString modelPath = tempDir() + "/ann_mnist_trained_gpu.json";

  // Step 1: Train on MNIST training data on GPU (30 epochs, 60k samples, all GPUs)
  auto trainResult = runNNCLI({
    "--config", fixturePath("mnist_ann_train_config.json"),
    "--mode", "train",
    "--device", "gpu",
    "--idx-data", examplePath("MNIST/train/train-images.idx3-ubyte"),
    "--idx-labels", examplePath("MNIST/train/train-labels.idx1-ubyte"),
    "--output", modelPath,
    "--log-level", "quiet"
  }, 1800000);  // 30 min timeout

  CHECK(trainResult.exitCode == 0, "ANN MNIST GPU train+test: training exit code 0");
  CHECK(QFile::exists(modelPath), "ANN MNIST GPU train+test: trained model file exists");

  if (trainResult.exitCode != 0 || !QFile::exists(modelPath)) {
    std::cout << "(training failed, skipping test step)" << std::endl;
    return;
  }

  // Step 2: Evaluate against MNIST test data (10k samples) on GPU
  auto testResult = runNNCLI({
    "--config", modelPath,
    "--mode", "test",
    "--device", "gpu",
    "--idx-data", examplePath("MNIST/test/t10k-images.idx3-ubyte"),
    "--idx-labels", examplePath("MNIST/test/t10k-labels.idx1-ubyte")
  }, 600000);  // 10 min timeout

  CHECK(testResult.exitCode == 0, "ANN MNIST GPU train+test: test exit code 0");
  CHECK(testResult.stdOut.contains("Test Results:"), "ANN MNIST GPU train+test: 'Test Results:'");
  CHECK(testResult.stdOut.contains("Samples evaluated: 10000"), "ANN MNIST GPU train+test: 'Samples evaluated: 10000'");

  // Extract and verify average loss is reasonable
  double avgLoss = -1;
  int idx = testResult.stdOut.indexOf("Average loss:");
  if (idx >= 0) {
    QString lossStr = testResult.stdOut.mid(idx + QString("Average loss:").length()).trimmed();
    lossStr = lossStr.left(lossStr.indexOf('\n'));
    avgLoss = lossStr.toDouble();
  }
  CHECK(avgLoss > 0 && avgLoss < 0.5, "ANN MNIST GPU train+test: average loss < 0.5");

  // Extract and verify accuracy is reasonable (> 30% for 30 epochs with mini-batch SGD)
  double accuracy = -1;
  int accIdx = testResult.stdOut.indexOf("Accuracy:");
  if (accIdx >= 0) {
    QString accStr = testResult.stdOut.mid(accIdx + QString("Accuracy:").length()).trimmed();
    accStr = accStr.left(accStr.indexOf('%'));
    accuracy = accStr.toDouble();
  }
  CHECK(accuracy > 30.0, "ANN MNIST GPU train+test: accuracy > 30%");

  std::cout << "(loss=" << avgLoss << ", accuracy=" << accuracy << "%) " << std::endl;
}

//===================================================================================================================//

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
  QStringList checkpoints = outputDir.entryList({"checkpoint_E-*.json"}, QDir::Files);
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

static void testANNShuffleSamplesCLI() {
  std::cout << "  testANNShuffleSamplesCLI... ";

  // Train with --shuffle-samples true
  QString modelPathTrue = tempDir() + "/ann_shuffle_true.json";
  auto resultTrue = runNNCLI({
    "--config", fixturePath("ann_train_config.json"),
    "--mode", "train",
    "--device", "cpu",
    "--samples", fixturePath("ann_train_samples.json"),
    "--output", modelPathTrue,
    "--shuffle-samples", "true"
  });

  CHECK(resultTrue.exitCode == 0, "ANN shuffle=true: exit code 0");
  CHECK(resultTrue.stdOut.contains("Training completed."), "ANN shuffle=true: 'Training completed.'");

  // Train with --shuffle-samples false
  QString modelPathFalse = tempDir() + "/ann_shuffle_false.json";
  auto resultFalse = runNNCLI({
    "--config", fixturePath("ann_train_config.json"),
    "--mode", "train",
    "--device", "cpu",
    "--samples", fixturePath("ann_train_samples.json"),
    "--output", modelPathFalse,
    "--shuffle-samples", "false"
  });

  CHECK(resultFalse.exitCode == 0, "ANN shuffle=false: exit code 0");
  CHECK(resultFalse.stdOut.contains("Training completed."), "ANN shuffle=false: 'Training completed.'");

  // Verify shuffleSamples is saved in the output model JSON
  QFile fileTrue(modelPathTrue);
  if (fileTrue.open(QIODevice::ReadOnly)) {
    QJsonDocument doc = QJsonDocument::fromJson(fileTrue.readAll());
    QJsonObject root = doc.object();
    CHECK(root.contains("trainingConfig"), "ANN shuffle=true: has 'trainingConfig'");
    QJsonObject tc = root["trainingConfig"].toObject();
    CHECK(tc.contains("shuffleSamples"), "ANN shuffle=true: has 'shuffleSamples'");
    CHECK(tc["shuffleSamples"].toBool() == true, "ANN shuffle=true: shuffleSamples is true");
    fileTrue.close();
  } else {
    CHECK(false, "ANN shuffle=true: failed to open model file");
  }

  QFile fileFalse(modelPathFalse);
  if (fileFalse.open(QIODevice::ReadOnly)) {
    QJsonDocument doc = QJsonDocument::fromJson(fileFalse.readAll());
    QJsonObject root = doc.object();
    QJsonObject tc = root["trainingConfig"].toObject();
    CHECK(tc.contains("shuffleSamples"), "ANN shuffle=false: has 'shuffleSamples'");
    CHECK(tc["shuffleSamples"].toBool() == false, "ANN shuffle=false: shuffleSamples is false");
    fileFalse.close();
  } else {
    CHECK(false, "ANN shuffle=false: failed to open model file");
  }

  std::cout << std::endl;
}

static void testANNShuffleSamplesInvalidValue() {
  std::cout << "  testANNShuffleSamplesInvalidValue... ";

  auto result = runNNCLI({
    "--config", fixturePath("ann_train_config.json"),
    "--mode", "train",
    "--device", "cpu",
    "--samples", fixturePath("ann_train_samples.json"),
    "--output", tempDir() + "/ann_shuffle_invalid.json",
    "--shuffle-samples", "maybe"
  });

  CHECK(result.exitCode == 1, "ANN shuffle=invalid: exit code 1");
  CHECK(result.stdErr.contains("--shuffle-samples must be 'true' or 'false'"),
        "ANN shuffle=invalid: error message");
  std::cout << std::endl;
}

//===================================================================================================================//

static void testANNTrainWithDropout() {
  std::cout << "  testANNTrainWithDropout... ";

  QString configPath = fixturePath("ann_train_dropout_config.json");
  QString samplesPath = fixturePath("ann_train_samples.json");
  QString outputPath = tempDir() + "/ann_dropout_model.json";

  auto result = runNNCLI({
    "--config", configPath,
    "--samples", samplesPath,
    "--output", outputPath
  });

  CHECK(result.exitCode == 0, "ANN dropout training exits 0");
  CHECK(QFile::exists(outputPath), "ANN dropout model file created");

  // Verify the model JSON contains dropoutRate in trainingConfig
  QFile file(outputPath);
  file.open(QIODevice::ReadOnly);
  auto json = QJsonDocument::fromJson(file.readAll()).object();
  file.close();

  auto tc = json["trainingConfig"].toObject();
  CHECK(tc.contains("dropoutRate"), "dropoutRate saved in model JSON");
  CHECK(std::abs(tc["dropoutRate"].toDouble() - 0.3) < 0.01,
        "dropoutRate value is 0.3");

  std::cout << std::endl;
}

//===================================================================================================================//

static void testANNTrainWithAugmentation() {
  std::cout << "  testANNTrainWithAugmentation... ";

  QString configPath = fixturePath("ann_train_augment_config.json");
  QString samplesPath = fixturePath("ann_train_samples.json");
  QString outputPath = tempDir() + "/ann_augment_model.json";

  auto result = runNNCLI({
    "--config", configPath,
    "--samples", samplesPath,
    "--output", outputPath
  });

  CHECK(result.exitCode == 0, "ANN augmentation training exits 0");
  CHECK(QFile::exists(outputPath), "ANN augmented model file created");

  // Verify auto class weights were applied
  QFile file(outputPath);
  file.open(QIODevice::ReadOnly);
  auto json = QJsonDocument::fromJson(file.readAll()).object();
  file.close();

  auto cfc = json["costFunctionConfig"].toObject();
  CHECK(cfc["type"].toString() == "weightedSquaredDifference",
        "auto class weights set cost function to weightedSquaredDifference");
  CHECK(cfc.contains("weights"), "auto class weights present in model");

  std::cout << std::endl;
}

//===================================================================================================================//

static void testANNDropoutRateParsing() {
  std::cout << "  testANNDropoutRateParsing... ";

  // Verify that dropoutRate=0 (default) is not saved in model JSON
  QString configPath = fixturePath("ann_train_config.json");
  QString samplesPath = fixturePath("ann_train_samples.json");
  QString outputPath = tempDir() + "/ann_no_dropout_model.json";

  auto result = runNNCLI({
    "--config", configPath,
    "--samples", samplesPath,
    "--output", outputPath
  });

  CHECK(result.exitCode == 0, "ANN no-dropout training exits 0");

  QFile file(outputPath);
  file.open(QIODevice::ReadOnly);
  auto json = QJsonDocument::fromJson(file.readAll()).object();
  file.close();

  auto tc = json["trainingConfig"].toObject();
  CHECK(!tc.contains("dropoutRate"), "dropoutRate not saved when 0.0 (default)");

  std::cout << std::endl;
}

//===================================================================================================================//

void runANNTests() {
  // Train XOR first — downstream tests use its output model
  testANNTrainXOR();
  testANNNetworkDetection();
  testANNModeOverride();
  testANNTrainWithWeightedLoss();
  testANNCheckpointParameters();
  testANNShuffleSamplesCLI();
  testANNShuffleSamplesInvalidValue();
  testANNTrainWithDropout();
  testANNTrainWithAugmentation();
  testANNDropoutRateParsing();
  // MNIST tests (--full only): train first, then predict/test using trained model
  testANNTrainAndTestMNIST();
  testANNTrainAndTestMNISTGPU();
  testANNPredictMNIST();
  testANNTestMNIST();
}

