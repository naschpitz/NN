#include "test_helpers.hpp"

#include <QDir>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>

static void testANNCheckpointParameters()
{
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

  auto result = runNNCLI(
    {"--config", configDst, "--mode", "train", "--device", "cpu", "--samples", samplesDst, "--output", modelPath});

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
          if (weights[i].toArray().size() > 0) {
            hasData = true;
            break;
          }
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

static void testANNShuffleSamplesCLI()
{
  std::cout << "  testANNShuffleSamplesCLI... ";

  // Train with --shuffle-samples true
  QString modelPathTrue = tempDir() + "/ann_shuffle_true.json";
  auto resultTrue =
    runNNCLI({"--config", fixturePath("ann_train_config.json"), "--mode", "train", "--device", "cpu", "--samples",
              fixturePath("ann_train_samples.json"), "--output", modelPathTrue, "--shuffle-samples", "true"});

  CHECK(resultTrue.exitCode == 0, "ANN shuffle=true: exit code 0");
  CHECK(resultTrue.stdOut.contains("Training completed."), "ANN shuffle=true: 'Training completed.'");

  // Train with --shuffle-samples false
  QString modelPathFalse = tempDir() + "/ann_shuffle_false.json";
  auto resultFalse =
    runNNCLI({"--config", fixturePath("ann_train_config.json"), "--mode", "train", "--device", "cpu", "--samples",
              fixturePath("ann_train_samples.json"), "--output", modelPathFalse, "--shuffle-samples", "false"});

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

static void testANNShuffleSamplesInvalidValue()
{
  std::cout << "  testANNShuffleSamplesInvalidValue... ";

  auto result = runNNCLI({"--config", fixturePath("ann_train_config.json"), "--mode", "train", "--device", "cpu",
                          "--samples", fixturePath("ann_train_samples.json"), "--output",
                          tempDir() + "/ann_shuffle_invalid.json", "--shuffle-samples", "maybe"});

  CHECK(result.exitCode == 1, "ANN shuffle=invalid: exit code 1");
  CHECK(result.stdErr.contains("--shuffle-samples must be 'true' or 'false'"), "ANN shuffle=invalid: error message");
  std::cout << std::endl;
}

//===================================================================================================================//

static void testANNTrainWithDropout()
{
  std::cout << "  testANNTrainWithDropout... ";

  QString configPath = fixturePath("ann_train_dropout_config.json");
  QString samplesPath = fixturePath("ann_train_samples.json");
  QString outputPath = tempDir() + "/ann_dropout_model.json";

  auto result = runNNCLI({"--config", configPath, "--samples", samplesPath, "--output", outputPath});

  CHECK(result.exitCode == 0, "ANN dropout training exits 0");
  CHECK(QFile::exists(outputPath), "ANN dropout model file created");

  // Verify the model JSON contains dropoutRate in trainingConfig
  QFile file(outputPath);
  file.open(QIODevice::ReadOnly);
  auto json = QJsonDocument::fromJson(file.readAll()).object();
  file.close();

  auto tc = json["trainingConfig"].toObject();
  CHECK(tc.contains("dropoutRate"), "dropoutRate saved in model JSON");
  CHECK(std::abs(tc["dropoutRate"].toDouble() - 0.3) < 0.01, "dropoutRate value is 0.3");

  std::cout << std::endl;
}

//===================================================================================================================//

static void testANNTrainWithAugmentation()
{
  std::cout << "  testANNTrainWithAugmentation... ";

  QString configPath = fixturePath("ann_train_augment_config.json");
  QString samplesPath = fixturePath("ann_train_samples.json");
  QString outputPath = tempDir() + "/ann_augment_model.json";

  auto result = runNNCLI({"--config", configPath, "--samples", samplesPath, "--output", outputPath});

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

static void testANNDropoutRateParsing()
{
  std::cout << "  testANNDropoutRateParsing... ";

  // Verify that dropoutRate=0 (default) is not saved in model JSON
  QString configPath = fixturePath("ann_train_config.json");
  QString samplesPath = fixturePath("ann_train_samples.json");
  QString outputPath = tempDir() + "/ann_no_dropout_model.json";

  auto result = runNNCLI({"--config", configPath, "--samples", samplesPath, "--output", outputPath});

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

static void testANNImageNetworkDetection()
{
  std::cout << "  testANNImageNetworkDetection... ";

  // An ANN config with inputType "image" and inputShape must still be detected as ANN, not CNN.
  // Train with the ann_image_train_config fixture (which has layersConfig + inputShape + inputType "image")
  // using image samples, and verify the log says "Network type: ANN".
  QString modelPath = tempDir() + "/ann_image_detect_model.json";

  auto result =
    runNNCLI({"--config", fixturePath("ann_image_train_config.json"), "--mode", "train", "--device", "cpu", "--samples",
              fixturePath("ann_image_train_samples.json"), "--output", modelPath, "--log-level", "info"});

  CHECK(result.exitCode == 0, "ANN image detection: exit code 0");
  CHECK(result.stdOut.contains("Network type: ANN"), "ANN image detection: detected as ANN (not CNN)");
  CHECK(!result.stdOut.contains("Network type: CNN"), "ANN image detection: NOT detected as CNN");
  CHECK(result.stdOut.contains("Input type: image"), "ANN image detection: input type is image");
  CHECK(result.stdOut.contains("Training completed."), "ANN image train: training completed");
  CHECK(QFile::exists(modelPath), "ANN image train: model file exists");

  std::cout << std::endl;
}

//===================================================================================================================//

void runANNCPUFeatureTests()
{
  testANNCheckpointParameters();
  testANNShuffleSamplesCLI();
  testANNShuffleSamplesInvalidValue();
  testANNTrainWithDropout();
  testANNTrainWithAugmentation();
  testANNDropoutRateParsing();
  testANNImageNetworkDetection();
}
