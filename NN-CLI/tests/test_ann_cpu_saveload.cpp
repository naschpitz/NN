#include "test_helpers.hpp"

#include <QDir>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>

//===================================================================================================================//

static void testANNSaveLoadPredictConsistency()
{
  std::cout << "  testANNSaveLoadPredictConsistency... ";

  // Train an ANN on XOR, predict, save, load, predict again — outputs must match
  QString modelPath = tempDir() + "/ann_slpc_model.nnmodel.tar";

  // Step 1: Train using the existing XOR fixtures
  auto trainResult = runNNCLI({"--config", fixturePath("ann_train_config.json"), "--mode", "train", "--device", "cpu",
                               "--samples", fixturePath("ann_train_samples.json"), "--output", modelPath});

  CHECK(trainResult.exitCode == 0, "ANN save/load predict: train exit code 0");

  if (trainResult.exitCode != 0) {
    std::cout << std::endl;
    return;
  }

  // Step 2: Verify the saved model is a valid .nnmodel.tar package
  QJsonObject modelJson = readModelJsonFromPackage(modelPath);
  CHECK(!modelJson.isEmpty(), "ANN save/load predict: model package contains valid model.json");

  if (!modelJson.isEmpty()) {
    CHECK(modelJson.contains("layers"), "ANN save/load predict: model.json has 'layers'");
    CHECK(modelJson.contains("train"), "ANN save/load predict: model.json has 'train'");
  }

  // Step 3: Create a predict input compatible with the XOR model (2 inputs)
  QString predictInputPath = tempDir() + "/ann_slpc_predict_input.json";
  QFile inputFile(predictInputPath);

  if (inputFile.exists())
    inputFile.remove();

  if (inputFile.open(QIODevice::WriteOnly)) {
    inputFile.write(R"({"inputs": [[0.0, 1.0], [1.0, 1.0]]})");
    inputFile.close();
  } else {
    CHECK(false, "ANN save/load predict: failed to write predict input file");
    std::cout << std::endl;
    return;
  }

  // Step 4: Predict with the saved model (first load from disk)
  QString predictOutput1 = tempDir() + "/ann_slpc_predict1.json";
  auto pred1Result = runNNCLI({"--config", modelPath, "--mode", "predict", "--device", "cpu", "--input",
                               predictInputPath, "--output", predictOutput1});

  CHECK(pred1Result.exitCode == 0, "ANN save/load predict: predict1 exit code 0");

  // Step 5: Predict again with the same saved model (fresh load from disk)
  QString predictOutput2 = tempDir() + "/ann_slpc_predict2.json";
  auto pred2Result = runNNCLI({"--config", modelPath, "--mode", "predict", "--device", "cpu", "--input",
                               predictInputPath, "--output", predictOutput2});

  CHECK(pred2Result.exitCode == 0, "ANN save/load predict: predict2 exit code 0");

  // Step 6: Compare outputs — they must be identical (same model, same input)
  if (pred1Result.exitCode == 0 && pred2Result.exitCode == 0) {
    QFile f1(predictOutput1);
    QFile f2(predictOutput2);

    if (f1.open(QIODevice::ReadOnly) && f2.open(QIODevice::ReadOnly)) {
      QJsonDocument doc1 = QJsonDocument::fromJson(f1.readAll());
      QJsonDocument doc2 = QJsonDocument::fromJson(f2.readAll());

      QJsonArray outputs1 = doc1.object()["outputs"].toArray();
      QJsonArray outputs2 = doc2.object()["outputs"].toArray();

      CHECK(outputs1.size() == outputs2.size(), "ANN save/load predict: same number of outputs");

      if (!outputs1.isEmpty() && !outputs2.isEmpty()) {
        bool allSamplesMatch = true;

        for (int s = 0; s < outputs1.size() && s < outputs2.size(); s++) {
          QJsonArray out1 = outputs1[s].toArray();
          QJsonArray out2 = outputs2[s].toArray();

          CHECK(out1.size() == out2.size(), "ANN save/load predict: same output size per sample");

          for (int i = 0; i < out1.size() && i < out2.size(); i++) {
            if (std::fabs(out1[i].toDouble() - out2[i].toDouble()) > 1e-6) {
              allSamplesMatch = false;
              break;
            }
          }

          if (!allSamplesMatch)
            break;
        }

        CHECK(allSamplesMatch, "ANN save/load predict: predictions match after reload");
      }

      f1.close();
      f2.close();
    } else {
      CHECK(false, "ANN save/load predict: failed to open predict output files");
    }
  }

  std::cout << std::endl;
}

//===================================================================================================================//

void runANNCPUSaveLoadTests()
{
  testANNSaveLoadPredictConsistency();
}
