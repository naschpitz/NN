#include "test_helpers.hpp"

#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>

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
    CHECK(root.contains("training"), "CNN shuffle=true: has 'trainingConfig'");
    QJsonObject tc = root["training"].toObject();
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
    QJsonObject tc = root["training"].toObject();
    CHECK(tc.contains("shuffleSamples"), "CNN shuffle=false: has 'shuffleSamples'");
    CHECK(tc["shuffleSamples"].toBool() == false, "CNN shuffle=false: shuffleSamples is false");
    fileFalse.close();
  } else {
    CHECK(false, "CNN shuffle=false: failed to open model file");
  }

  std::cout << std::endl;
}

void runCNNCPUShuffleTests()
{
  testCNNShuffleSamplesCLI();
}
