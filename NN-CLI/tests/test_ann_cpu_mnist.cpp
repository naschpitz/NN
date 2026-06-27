#include "test_helpers.hpp"

#include <QDir>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>

static QString trainedMNISTModelPath;

//===================================================================================================================//

static void testTrainAndTestMNIST()
{
  TestScope _t("testTrainAndTestMNIST");

  if (!runFullTests) {
    std::cout << "(skipped — use --full to enable)" << std::endl;
    return;
  }

  trainedMNISTModelPath = tempDir() + "/ann_mnist_trained.nnmodel.tar";

  // Step 1: Train on MNIST training data on CPU (10 epochs, 60k samples, Adam + crossEntropy)
  QString outDir = tempDir() + "/ann_mnist_out";
  QDir(outDir).removeRecursively();
  QDir().mkpath(outDir);
  auto trainResult =
    runNNCLI({"--model", fixturePath("mnist_ann_train_config.json"), "--mode", "train", "--device", "cpu", "--idx-data",
              examplePath("MNIST/train/train-images.idx3-ubyte"), "--idx-labels",
              examplePath("MNIST/train/train-labels.idx1-ubyte"), "--output", outDir, "--log-level", "quiet"},
             3600000); // 60 min timeout

  CHECK(trainResult.exitCode == 0, " MNIST train+test: training exit code 0");
  trainedMNISTModelPath = findTrainedModel(outDir);
  CHECK(!trainedMNISTModelPath.isEmpty(), " MNIST train+test: trained model file exists");

  if (trainResult.exitCode != 0 || trainedMNISTModelPath.isEmpty()) {
    trainedMNISTModelPath.clear();
    std::cout << "(training failed, skipping test step)" << std::endl;
    return;
  }

  // Step 2: Evaluate against MNIST test data (10k samples)
  auto testResult = runNNCLI({"--model-package", trainedMNISTModelPath, "--mode", "test", "--device", "cpu",
                              "--idx-data", examplePath("MNIST/test/t10k-images.idx3-ubyte"), "--idx-labels",
                              examplePath("MNIST/test/t10k-labels.idx1-ubyte")},
                             600000); // 10 min timeout

  CHECK(testResult.exitCode == 0, " MNIST train+test: test exit code 0");
  CHECK(testResult.stdOut.contains("Test Results:"), " MNIST train+test: 'Test Results:'");
  CHECK(testResult.stdOut.contains("Samples evaluated: 10000"), " MNIST train+test: 'Samples evaluated: 10000'");

  // Extract and verify average loss is reasonable
  double avgLoss = -1;
  int idx = testResult.stdOut.indexOf("Average loss:");

  if (idx >= 0) {
    QString lossStr = testResult.stdOut.mid(idx + QString("Average loss:").length()).trimmed();
    lossStr = lossStr.left(lossStr.indexOf('\n'));
    avgLoss = lossStr.toDouble();
  }

  CHECK(avgLoss > 0 && avgLoss < 2.0, " MNIST train+test: average loss < 2.0");

  // Extract and verify accuracy is reasonable (> 30% for 10 epochs with Adam + crossEntropy)
  double accuracy = -1;
  int accIdx = testResult.stdOut.indexOf("Accuracy:");

  if (accIdx >= 0) {
    QString accStr = testResult.stdOut.mid(accIdx + QString("Accuracy:").length()).trimmed();
    accStr = accStr.left(accStr.indexOf('%'));
    accuracy = accStr.toDouble();
  }

  CHECK(accuracy > 30.0, " MNIST train+test: accuracy > 30%");

  std::cout << "(loss=" << avgLoss << ", accuracy=" << accuracy << "%) " << std::endl;
}

//===================================================================================================================//

static void testPredictMNIST()
{
  TestScope _t("testPredictMNIST");

  if (!runFullTests) {
    std::cout << "(skipped — use --full to enable)" << std::endl;
    return;
  }

  if (trainedMNISTModelPath.isEmpty() || !QFile::exists(trainedMNISTModelPath)) {
    CHECK(false, " predict MNIST: skipped — no trained MNIST model available (testTrainAndTestMNIST must run first)");
    std::cout << std::endl;
    return;
  }

  QString outDir = tempDir() + "/ann_predict_out";
  QDir(outDir).removeRecursively();
  QDir().mkpath(outDir);

  auto result = runNNCLI({"--model-package", trainedMNISTModelPath, "--mode", "predict", "--device", "cpu", "--input",
                          examplePath("MNIST/predict/mnist_digit_2_input.json"), "--output", outDir});

  CHECK(result.exitCode == 0, " predict MNIST: exit code 0");
  CHECK(result.stdOut.contains("Predict result saved to:"), " predict MNIST: 'Predict result saved to:'");
  QString outputPath = predictJsonPath(outDir, examplePath("MNIST/predict/mnist_digit_2_input.json"));
  CHECK(QFile::exists(outputPath), " predict MNIST: output file exists");

  // Verify output JSON structure and content
  QFile file(outputPath);

  if (file.open(QIODevice::ReadOnly)) {
    QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
    QJsonObject root = doc.object();

    CHECK(root.contains("predictMetadata"), " predict MNIST: has 'predictMetadata'");
    CHECK(root.contains("outputs"), " predict MNIST: has 'outputs'");

    QJsonArray outputsArray = root["outputs"].toArray();
    CHECK(outputsArray.size() == 1, " predict MNIST: outputs has 1 element (batch of 1)");

    QJsonArray firstOutput = outputsArray[0].toArray();
    CHECK(firstOutput.size() == 10, " predict MNIST: first output has 10 elements");

    // Verify all outputs are valid numbers in [0, 1]
    bool allValid = true;

    for (int i = 0; i < firstOutput.size(); ++i) {
      double v = firstOutput[i].toDouble();

      if (v < 0.0 || v > 1.0) {
        allValid = false;
        break;
      }
    }

    CHECK(allValid, " predict MNIST: all outputs in [0, 1]");

    QJsonObject meta = root["predictMetadata"].toObject();
    CHECK(meta.contains("startTime"), " predict MNIST: metadata has 'startTime'");
    CHECK(meta.contains("endTime"), " predict MNIST: metadata has 'endTime'");
    CHECK(meta.contains("durationSeconds"), " predict MNIST: metadata has 'durationSeconds'");
    CHECK(meta.contains("durationFormatted"), " predict MNIST: metadata has 'durationFormatted'");
    CHECK(meta.contains("numInputs"), " predict MNIST: metadata has 'numInputs'");
    file.close();
  } else {
    CHECK(false, " predict MNIST: failed to open output file");
  }
}

//===================================================================================================================//

static void testTestMNIST()
{
  TestScope _t("testTestMNIST");

  if (!runFullTests) {
    std::cout << "(skipped — use --full to enable)" << std::endl;
    return;
  }

  if (trainedMNISTModelPath.isEmpty() || !QFile::exists(trainedMNISTModelPath)) {
    CHECK(false, " test MNIST: skipped — no trained MNIST model available (testTrainAndTestMNIST must run first)");
    std::cout << std::endl;
    return;
  }

  auto result = runNNCLI({"--model-package", trainedMNISTModelPath, "--mode", "test", "--device", "cpu", "--idx-data",
                          examplePath("MNIST/test/t10k-images.idx3-ubyte"), "--idx-labels",
                          examplePath("MNIST/test/t10k-labels.idx1-ubyte")},
                         600000); // 10 min timeout

  CHECK(result.exitCode == 0, " test MNIST: exit code 0");
  CHECK(result.stdOut.contains("Test Results:"), " test MNIST: 'Test Results:'");
  CHECK(result.stdOut.contains("Samples evaluated: 10000"), " test MNIST: 'Samples evaluated: 10000'");
  CHECK(result.stdOut.contains("Total loss:"), " test MNIST: 'Total loss:'");
  CHECK(result.stdOut.contains("Average loss:"), " test MNIST: 'Average loss:'");
  CHECK(result.stdOut.contains("Correct:"), " test MNIST: 'Correct:'");
  CHECK(result.stdOut.contains("Accuracy:"), " test MNIST: 'Accuracy:'");
}

//===================================================================================================================//

void runCPUMNISTTests()
{
  testTrainAndTestMNIST();
  testPredictMNIST();
  testTestMNIST();
}