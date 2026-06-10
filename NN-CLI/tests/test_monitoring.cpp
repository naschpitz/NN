#include "test_helpers.hpp"
#include "../NN-CLI_ANNLoader.hpp"

#include "Common/Common_TrainingMonitor.hpp"

#include <optional>

using namespace NN_CLI;

//===================================================================================================================//

static void testMonitoringConfigParsing()
{
  std::cout << "  testMonitoringConfigParsing... ";

  QString configPath = tempDir() + "/monitoring_config_test.json";
  QFile file(configPath);
  file.open(QIODevice::WriteOnly);
  file.write(R"({
    "mode": "train",
    "layers": [{"numNeurons": 4, "actvFunc": "relu"}],
    "training": {
      "numEpochs": 100,
      "learningRate": 0.01,
      "monitoring": {
        "enabled": true,
        "checkInterval": 10,
        "patience": 30,
        "metrics": {
          "lossStagnation": { "enabled": true, "minDelta": 0.001 },
          "lossExplosion": { "enabled": false, "threshold": 5.0 }
        }
      }
    }
  })");

  file.close();

  auto config = ANNLoader::loadConfig(configPath.toStdString());

  CHECK(config.trainingConfig.monitoringConfig.enabled == true, "monitoring enabled");
  CHECK(config.trainingConfig.monitoringConfig.checkInterval == 10, "checkInterval");
  CHECK(config.trainingConfig.monitoringConfig.patience == 30, "patience");
  CHECK_NEAR(config.trainingConfig.monitoringConfig.metrics.lossStagnation.minDelta, 0.001f, 0.0001f, "minDelta");
  CHECK(config.trainingConfig.monitoringConfig.metrics.lossExplosion.enabled == false, "explosion disabled");
  CHECK_NEAR(config.trainingConfig.monitoringConfig.metrics.lossExplosion.threshold, 5.0f, 0.01f,
             "explosion threshold");

  std::cout << "PASS" << std::endl;
}

//===================================================================================================================//

static void testMonitoringConfigDefaults()
{
  std::cout << "  testMonitoringConfigDefaults... ";

  QString configPath = tempDir() + "/monitoring_defaults_test.json";
  QFile file(configPath);
  file.open(QIODevice::WriteOnly);
  file.write(R"({
    "mode": "train",
    "layers": [{"numNeurons": 4, "actvFunc": "relu"}],
    "training": {
      "numEpochs": 10,
      "learningRate": 0.01
    }
  })");

  file.close();

  auto config = ANNLoader::loadConfig(configPath.toStdString());

  CHECK(config.trainingConfig.monitoringConfig.enabled == false, "default disabled");
  CHECK(config.trainingConfig.monitoringConfig.checkInterval == 5, "default checkInterval");
  CHECK(config.trainingConfig.monitoringConfig.patience == 20, "default patience");

  std::cout << "PASS" << std::endl;
}

//===================================================================================================================//

static void testTrainingMonitorStagnation()
{
  std::cout << "  testTrainingMonitorStagnation... ";

  Common::MonitoringConfig config;
  config.enabled = true;
  config.checkInterval = 1;
  config.patience = 3;
  config.metrics.lossStagnation.enabled = true;
  config.metrics.lossStagnation.minDelta = 0.01f;
  config.metrics.lossExplosion.enabled = false;

  Common::TrainingMonitor<float> monitor(config);

  // Epoch 1: loss = 1.0 — new best
  CHECK(monitor.checkEpoch(1, 1.0f) == false, "epoch 1 no stop");
  CHECK(monitor.isNewBest() == true, "epoch 1 is new best");
  CHECK(monitor.getBestEpoch() == 1, "best epoch is 1");

  // Epoch 2: loss = 0.5 — improvement
  CHECK(monitor.checkEpoch(2, 0.5f) == false, "epoch 2 no stop");
  CHECK(monitor.isNewBest() == true, "epoch 2 is new best");

  // Epoch 3: loss = 0.5 — no improvement (patience 1/3)
  CHECK(monitor.checkEpoch(3, 0.5f) == false, "epoch 3 no stop");
  CHECK(monitor.isNewBest() == false, "epoch 3 not new best");

  // Epoch 4: loss = 0.5 — no improvement (patience 2/3)
  CHECK(monitor.checkEpoch(4, 0.5f) == false, "epoch 4 no stop");

  // Epoch 5: loss = 0.5 — no improvement (patience 3/3 → stop)
  CHECK(monitor.checkEpoch(5, 0.5f) == true, "epoch 5 stops");
  CHECK(monitor.getStopReason().find("stagnation") != std::string::npos, "stop reason mentions stagnation");

  std::cout << "PASS" << std::endl;
}

//===================================================================================================================//

static void testTrainingMonitorExplosion()
{
  std::cout << "  testTrainingMonitorExplosion... ";

  Common::MonitoringConfig config;
  config.enabled = true;
  config.checkInterval = 1;
  config.patience = 100;
  config.metrics.lossStagnation.enabled = false;
  config.metrics.lossExplosion.enabled = true;
  config.metrics.lossExplosion.threshold = 5.0f;

  Common::TrainingMonitor<float> monitor(config);

  // Epoch 1: loss = 1.0
  CHECK(monitor.checkEpoch(1, 1.0f) == false, "epoch 1 no stop");

  // Epoch 2: loss = 4.0 — high but under 5x
  CHECK(monitor.checkEpoch(2, 4.0f) == false, "epoch 2 no stop (4x)");

  // Epoch 3: loss = 6.0 — over 5x best (1.0 × 5 = 5.0)
  CHECK(monitor.checkEpoch(3, 6.0f) == true, "epoch 3 stops (6x best)");
  CHECK(monitor.getStopReason().find("explosion") != std::string::npos ||
          monitor.getStopReason().find("Explosion") != std::string::npos,
        "stop reason mentions explosion");

  std::cout << "PASS" << std::endl;
}

//===================================================================================================================//

static void testTrainingMonitorWithValidationLoss()
{
  std::cout << "  testTrainingMonitorWithValidationLoss... ";

  Common::MonitoringConfig config;
  config.enabled = true;
  config.checkInterval = 1;
  config.patience = 2;
  config.metrics.lossStagnation.enabled = true;
  config.metrics.lossStagnation.minDelta = 0.01f;
  config.metrics.lossExplosion.enabled = false;

  Common::TrainingMonitor<float> monitor(config);

  // Training loss drops but validation loss is what matters
  CHECK(monitor.checkEpoch(1, 2.0f, std::optional<float>(1.0f)) == false, "epoch 1");
  CHECK(monitor.isNewBest() == true, "epoch 1 new best (val=1.0)");

  // Training loss drops further but validation loss stagnates
  CHECK(monitor.checkEpoch(2, 1.0f, std::optional<float>(1.0f)) == false, "epoch 2 no stop (patience 1/2)");
  CHECK(monitor.isNewBest() == false, "epoch 2 not new best (val unchanged)");

  CHECK(monitor.checkEpoch(3, 0.5f, std::optional<float>(1.0f)) == true, "epoch 3 stops (patience 2/2)");

  // Best should still be epoch 1
  CHECK(monitor.getBestEpoch() == 1, "best epoch is 1 (based on validation loss)");

  std::cout << "PASS" << std::endl;
}

//===================================================================================================================//

void runMonitoringTests()
{
  testMonitoringConfigParsing();
  testMonitoringConfigDefaults();
  testTrainingMonitorStagnation();
  testTrainingMonitorExplosion();
  testTrainingMonitorWithValidationLoss();
}
