#include "test_helpers.hpp"

#include <QDir>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>

static void testCheckpointParameters()
{
  std::cout << "  testCheckpointParameters... ";

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

  CHECK(result.exitCode == 0, " checkpoint params: exit code 0");
  CHECK(result.stdOut.contains("Training completed."), " checkpoint params: 'Training completed.'");

  // Find checkpoint files in tempDir/output/
  QDir outputDir(tempDir() + "/output");
  QStringList checkpoints = outputDir.entryList({"checkpoint_E-*.json"}, QDir::Files);
  CHECK(!checkpoints.isEmpty(), " checkpoint params: checkpoint files exist");

  if (!checkpoints.isEmpty()) {
    // Parse the first checkpoint and verify parameters are non-empty
    QString checkpointPath = outputDir.filePath(checkpoints.first());
    QFile file(checkpointPath);

    if (file.open(QIODevice::ReadOnly)) {
      QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
      QJsonObject root = doc.object();
      CHECK(root.contains("parameters"), " checkpoint params: has 'parameters'");

      QJsonObject params = root["parameters"].toObject();
      QJsonArray weights = params["weights"].toArray();
      QJsonArray biases = params["biases"].toArray();

      CHECK(!weights.isEmpty(), " checkpoint params: weights non-empty");
      CHECK(!biases.isEmpty(), " checkpoint params: biases non-empty");

      // Verify at least one layer has actual weight data
      if (!weights.isEmpty()) {
        bool hasData = false;

        for (int i = 0; i < weights.size(); ++i) {
          if (weights[i].toArray().size() > 0) {
            hasData = true;
            break;
          }
        }

        CHECK(hasData, " checkpoint params: weights contain actual data");
      }

      file.close();
    } else {
      CHECK(false, " checkpoint params: failed to open checkpoint file");
    }
  }

  // Cleanup checkpoint output dir
  QDir(tempDir() + "/output").removeRecursively();

  std::cout << std::endl;
}

//===================================================================================================================//

static void testTrainWithDropout()
{
  std::cout << "  testTrainWithDropout... ";

  QString configPath = fixturePath("ann_train_dropout_config.json");
  QString samplesPath = fixturePath("ann_train_samples.json");
  QString outputPath = tempDir() + "/ann_dropout_model.json";

  auto result = runNNCLI({"--config", configPath, "--samples", samplesPath, "--output", outputPath});

  CHECK(result.exitCode == 0, " dropout training exits 0");
  CHECK(QFile::exists(outputPath), " dropout model file created");

  // Verify the model JSON contains dropoutRate in trainingConfig
  QFile file(outputPath);
  file.open(QIODevice::ReadOnly);
  auto json = QJsonDocument::fromJson(file.readAll()).object();
  file.close();

  auto tc = json["training"].toObject();
  CHECK(tc.contains("dropoutRate"), "dropoutRate saved in model JSON");
  CHECK(std::abs(tc["dropoutRate"].toDouble() - 0.3) < 0.01, "dropoutRate value is 0.3");

  std::cout << std::endl;
}

//===================================================================================================================//

static void testTrainWithAugmentation()
{
  std::cout << "  testTrainWithAugmentation... ";

  QString configPath = fixturePath("ann_train_augment_config.json");
  QString samplesPath = fixturePath("ann_train_samples.json");
  QString outputPath = tempDir() + "/ann_augment_model.json";

  auto result = runNNCLI({"--config", configPath, "--samples", samplesPath, "--output", outputPath});

  CHECK(result.exitCode == 0, " augmentation training exits 0");
  CHECK(QFile::exists(outputPath), " augmented model file created");

  // Verify auto class weights were applied
  QFile file(outputPath);
  file.open(QIODevice::ReadOnly);
  auto json = QJsonDocument::fromJson(file.readAll()).object();
  file.close();

  auto cfc = json["costFunction"].toObject();
  CHECK(cfc["type"].toString() == "weightedSquaredDifference",
        "auto class weights set cost function to weightedSquaredDifference");
  CHECK(cfc.contains("weights"), "auto class weights present in model");

  std::cout << std::endl;
}

//===================================================================================================================//

static void testDropoutRateParsing()
{
  std::cout << "  testDropoutRateParsing... ";

  // Verify that dropoutRate=0 (default) is not saved in model JSON
  QString configPath = fixturePath("ann_train_config.json");
  QString samplesPath = fixturePath("ann_train_samples.json");
  QString outputPath = tempDir() + "/ann_no_dropout_model.json";

  auto result = runNNCLI({"--config", configPath, "--samples", samplesPath, "--output", outputPath});

  CHECK(result.exitCode == 0, " no-dropout training exits 0");

  QFile file(outputPath);
  file.open(QIODevice::ReadOnly);
  auto json = QJsonDocument::fromJson(file.readAll()).object();
  file.close();

  auto tc = json["training"].toObject();
  CHECK(tc.contains("dropoutRate"), "dropoutRate always saved for complete snapshot");

  std::cout << std::endl;
}

//===================================================================================================================//

static void testImageNetworkDetection()
{
  std::cout << "  testImageNetworkDetection... ";

  // An  config with inputType "image" and inputShape must still be detected as , not CNN.
  // Train with the ann_image_train_config fixture (which has layersConfig + inputShape + inputType "image")
  // using image samples, and verify the log says "Network type: ".
  QString modelPath = tempDir() + "/ann_image_detect_model.json";

  auto result =
    runNNCLI({"--config", fixturePath("ann_image_train_config.json"), "--mode", "train", "--device", "cpu", "--samples",
              fixturePath("ann_image_train_samples.json"), "--output", modelPath, "--log-level", "info"});

  CHECK(result.exitCode == 0, " image detection: exit code 0");
  CHECK(result.stdOut.contains("Network type: "), " image detection: detected as  (not CNN)");
  CHECK(!result.stdOut.contains("Network type: CNN"), " image detection: NOT detected as CNN");
  CHECK(result.stdOut.contains("Input type: image"), " image detection: input type is image");
  CHECK(result.stdOut.contains("Training completed."), " image train: training completed");
  CHECK(QFile::exists(modelPath), " image train: model file exists");

  std::cout << std::endl;
}

//===================================================================================================================//

void runCPUFeatureTests()
{
  testCheckpointParameters();
  testTrainWithDropout();
  testTrainWithAugmentation();
  testDropoutRateParsing();
  testImageNetworkDetection();
}
