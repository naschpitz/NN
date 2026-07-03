#include "test_helpers.hpp"

#include <QDir>
#include <QJsonArray>
#include <QJsonObject>
#include <QJsonDocument>

#include <vector>

//===================================================================================================================//

// Regression guard for the ANN GPU validation parameter-sync bug. ANN::Core previously had no
// syncParametersToGPU, so the separate validationCore's GPU weight buffers were frozen at
// construction — every epoch evaluated the same epoch-0 weights and valLoss stayed EXACTLY
// constant. The fix added syncParametersToGPU (mirroring CNN), called each epoch after
// setParameters. This test trains a tiny ANN on GPU with validation enabled and asserts valLoss
// decreases across epochs (the frozen-params bug would leave it perfectly flat). GPU-gated.
static void testGPUValidationLossDecreases()
{
  TestScope _t("testGPUValidationLossDecreases");

  if (!checkGPUAvailable()) {
    std::cout << "(skipped — no GPU available)" << std::endl;
    return;
  }

  QString outDir = tempDir() + "/ann_gpu_validation_out";
  QDir(outDir).removeRecursively();
  QDir().mkpath(outDir);

  auto result =
    runNNCLI({"--model", fixturePath("ann_gpu_validation_config.json"), "--mode", "train", "--device", "gpu",
              "--samples", fixturePath("ann_validation_samples.json"), "--output", outDir, "--log-level", "quiet"},
             120000);

  CHECK(result.exitCode == 0, "GPU validation: training exit code 0");
  QString modelPath = findTrainedModel(outDir);
  CHECK(!modelPath.isEmpty(), "GPU validation: trained model file exists");

  if (result.exitCode != 0 || modelPath.isEmpty()) {
    std::cout << "(training failed, skipping valLoss check)" << std::endl;
    return;
  }

  QJsonObject root = readModelJsonFromPackage(modelPath);
  CHECK(!root.isEmpty(), "GPU validation: model.json readable");

  QJsonArray epochs = root.value("trainMetadata").toObject().value("epochs").toArray();

  std::vector<double> valLosses;

  for (const QJsonValue& epochRecord : epochs) {
    QJsonObject rec = epochRecord.toObject();

    if (rec.value("hasValLoss").toBool(false))
      valLosses.push_back(rec.value("valLoss").toDouble());
  }

  CHECK(valLosses.size() >= 2, "GPU validation: at least 2 epochs with valLoss recorded");

  if (valLosses.size() >= 2) {
    double firstValLoss = valLosses.front();
    double lastValLoss = valLosses.back();
    CHECK(lastValLoss < firstValLoss,
          "GPU validation: valLoss must decrease across epochs (frozen-params bug would leave it flat)");
    std::cout << "(valLoss " << firstValLoss << " -> " << lastValLoss << ")" << std::endl;
  }
}

//===================================================================================================================//

void runGPUValidationTests()
{
  testGPUValidationLossDecreases();
}
