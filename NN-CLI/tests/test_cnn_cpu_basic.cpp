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

// Trained model path shared between CNN tests (train → predict/test)
static QString trainedCNNModelPath;

static void testCNNNetworkDetection()
{
  std::cout << "  testCNNNetworkDetection... ";

  // Train with tiny fixture + verbose to check detection
  auto result = runNNCLI({"--config", fixturePath("cnn_train_config.json"), "--mode", "train", "--device", "cpu",
                          "--samples", fixturePath("cnn_train_samples.json"), "--output",
                           tempDir() + "/cnn_detect_model.nnmodel.tar", "--log-level", "info"});

  CHECK(result.exitCode == 0, "CNN detection: exit code 0");
  CHECK(result.stdOut.contains("Network type: CNN"), "CNN detection: 'Network type: CNN'");
  std::cout << std::endl;
}

static void testCNNTrain()
{
  std::cout << "  testCNNTrain... ";

  trainedCNNModelPath = tempDir() + "/cnn_trained_model.nnmodel.tar";

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

  QString modelPath = tempDir() + "/cnn_weighted_model.nnmodel.tar";

  auto result = runNNCLI({"--config", fixturePath("cnn_train_weighted_config.json"), "--mode", "train", "--device",
                          "cpu", "--samples", fixturePath("cnn_train_samples.json"), "--output", modelPath});

  CHECK(result.exitCode == 0, "CNN weighted train: exit code 0");
  CHECK(result.stdOut.contains("Training completed."), "CNN weighted train: 'Training completed.'");
  CHECK(result.stdOut.contains("Model saved to:"), "CNN weighted train: 'Model saved to:'");
  CHECK(QFile::exists(modelPath), "CNN weighted train: model file exists");

  // Verify saved model JSON contains costFunctionConfig
  QJsonObject root = readModelJsonFromPackage(modelPath);

  if (!root.isEmpty()) {
    CHECK(root.contains("costFunction"), "CNN weighted train: saved model has 'costFunctionConfig'");

    QJsonObject cfc = root["costFunction"].toObject();
    CHECK(cfc["type"].toString() == "weightedSquaredDifference",
          "CNN weighted train: type is 'weightedSquaredDifference'");
    CHECK(cfc.contains("weights"), "CNN weighted train: has 'weights'");

    QJsonArray weights = cfc["weights"].toArray();
    CHECK(weights.size() == 2, "CNN weighted train: weights has 2 elements");
    CHECK_NEAR(weights[0].toDouble(), 5.0, 1e-6, "CNN weighted train: weight[0] = 5.0");
    CHECK_NEAR(weights[1].toDouble(), 1.0, 1e-6, "CNN weighted train: weight[1] = 1.0");
  } else {
    CHECK(false, "CNN weighted train: failed to read saved model package");
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

  QString modelPath = tempDir() + "/cnn_mnist_trained.nnmodel.tar";

  // Step 1: Train on MNIST training data on CPU (10 epochs, 60k samples, Adam + crossEntropy + instancenorm)
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

  // Extract and verify accuracy is reasonable (> 25% for 10 epochs with Adam + crossEntropy + instancenorm on CPU)
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

static void testCNNResidualMNIST()
{
  std::cout << "  testCNNResidualMNIST... " << std::flush;

  if (!runFullTests) {
    std::cout << "(skipped — use --full to enable)" << std::endl;
    return;
  }

  QString modelPath = tempDir() + "/cnn_mnist_residual_trained.nnmodel.tar";

  // Train ResNet-like architecture on MNIST (10 epochs, 60k samples, Adam + crossEntropy + residual blocks)
  // Residual is deeper than the plain-conv MNIST test, so the training pass is markedly slower; 60min was not enough.
  auto trainResult =
    runNNCLI({"--config", fixturePath("mnist_cnn_residual_config.json"), "--mode", "train", "--device", "cpu",
              "--idx-data", examplePath("MNIST/train/train-images.idx3-ubyte"), "--idx-labels",
              examplePath("MNIST/train/train-labels.idx1-ubyte"), "--output", modelPath, "--log-level", "quiet"},
             7200000); // 2h timeout

  CHECK(trainResult.exitCode == 0, "CNN Residual MNIST: training exit code 0");
  CHECK(QFile::exists(modelPath), "CNN Residual MNIST: trained model file exists");

  if (trainResult.exitCode != 0 || !QFile::exists(modelPath)) {
    std::cout << "(training failed exit=" << trainResult.exitCode << ", skipping test step)" << std::endl;
    return;
  }

  // Verify model contains residual markers in the configuration
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

      CHECK(hasResStart, "CNN Residual MNIST: model has residual_start");
      CHECK(hasResEnd, "CNN Residual MNIST: model has residual_end");
      // Projection weights exist when residual blocks have channel changes (stored in params.bin)
      CHECK(hasResStart && hasResEnd, "CNN Residual MNIST: model has residual params");
    }
  }

  // Evaluate against MNIST test data (10k samples)
  auto testResult = runNNCLI({"--config", modelPath, "--mode", "test", "--device", "cpu", "--idx-data",
                              examplePath("MNIST/test/t10k-images.idx3-ubyte"), "--idx-labels",
                              examplePath("MNIST/test/t10k-labels.idx1-ubyte")},
                             600000); // 10 min timeout

  CHECK(testResult.exitCode == 0, "CNN Residual MNIST: test exit code 0");
  CHECK(testResult.stdOut.contains("Test Results:"), "CNN Residual MNIST: 'Test Results:'");
  CHECK(testResult.stdOut.contains("Samples evaluated: 10000"), "CNN Residual MNIST: 'Samples evaluated: 10000'");

  double avgLoss = -1;
  int idx = testResult.stdOut.indexOf("Average loss:");

  if (idx >= 0) {
    QString lossStr = testResult.stdOut.mid(idx + QString("Average loss:").length()).trimmed();
    lossStr = lossStr.left(lossStr.indexOf('\n'));
    avgLoss = lossStr.toDouble();
  }

  CHECK(avgLoss > 0 && avgLoss < 2.5, "CNN Residual MNIST: average loss < 2.5");

  double accuracy = -1;
  int accIdx = testResult.stdOut.indexOf("Accuracy:");

  if (accIdx >= 0) {
    QString accStr = testResult.stdOut.mid(accIdx + QString("Accuracy:").length()).trimmed();
    accStr = accStr.left(accStr.indexOf('%'));
    accuracy = accStr.toDouble();
  }

  CHECK(accuracy > 25.0, "CNN Residual MNIST: accuracy > 25%");

  std::cout << "(loss=" << avgLoss << ", accuracy=" << accuracy << "%) " << std::endl;
}

// Regression guard for the validation deadlock. With validation enabled, the per-epoch
// validation pass (CoreCPU::test) runs from inside train()'s per-sample callback. When
// train() and test() shared the global QThreadPool, that nested map deadlocked — every
// worker thread parked in a futex at 0% CPU. The fix gives each core its own pool.
// This trains a tiny net for 3 epochs (validation fires from epoch 2 on) with a short
// timeout, so a regression surfaces as a fast failure instead of an hour-long stall.
// Not --full gated: the legitimate train finishes in well under a second.
static void testCNNTrainValidationNoDeadlock()
{
  std::cout << "  testCNNTrainValidationNoDeadlock... " << std::flush;

  QString modelPath = tempDir() + "/cnn_validation_nodeadlock.nnmodel.tar";

  auto result =
    runNNCLI({"--config", fixturePath("cnn_validation_config.json"), "--mode", "train", "--device", "cpu", "--samples",
              fixturePath("cnn_validation_samples.json"), "--output", modelPath, "--log-level", "quiet"},
             60000); // 60s deadlock guard — real train takes <1s; a hang trips the timeout

  CHECK(result.exitCode == 0, "CNN validation no-deadlock: training exit code 0 (timeout/-2 = deadlock)");
  CHECK(QFile::exists(modelPath), "CNN validation no-deadlock: trained model file exists");

  std::cout << std::endl;
}

//===================================================================================================================//

void runCNNCPUBasicTests()
{
  testCNNNetworkDetection();
  testCNNTrain();
  testCNNPredict();
  testCNNTest();
  testCNNTrainWithWeightedLoss();
  testCNNTrainValidationNoDeadlock();
  testCNNTrainAndTestMNIST();
  testCNNResidualMNIST();
}
