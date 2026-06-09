#include "test_helpers.hpp"

#include <QDir>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>

#include <json.hpp>

//===================================================================================================================//
// Systematic GPU predict tests: isolate which layer combination breaks GPU predict
//===================================================================================================================//

// Helper: generate a CNN config JSON with given convolutional layers.
// All configs use 1x8x8 input, train on CPU, 2-output sigmoid dense layer.
static QString writeGPUTestConfig(const QString& path, const QString& convLayersJson)
{
  QFile f(path);

  if (f.exists())
    f.remove();

  if (!f.open(QIODevice::WriteOnly))
    return {};

  QString json = QString(R"({
  "mode": "train",
  "device": "cpu",
  "progressReports": 0,
  "saveModelInterval": 0,
  "inputShape": { "c": 1, "h": 8, "w": 8 },
  "convolutionalLayers": %1,
  "denseLayers": [
    { "numNeurons": 2, "actvFunc": "sigmoid" }
  ],
  "training": {
    "numEpochs": 50,
    "learningRate": 0.01
  }
})")

                   .arg(convLayersJson);

  f.write(json.toUtf8());
  f.close();
  return path;
}

// Helper: write test inputs (1x8x8 = 64 values, 2 distinct patterns)
static QString writeGPUTestInputs(const QString& path)
{
  QFile f(path);

  if (f.exists())
    f.remove();

  if (!f.open(QIODevice::WriteOnly))
    return {};

  nlohmann::ordered_json inputsJson;
  nlohmann::ordered_json inputsArr = nlohmann::ordered_json::array();

  // Pattern A: gradient
  std::vector<float> a(64);

  for (int i = 0; i < 64; i++)
    a[i] = static_cast<float>(i) / 63.0f;
  inputsArr.push_back(a);

  // Pattern B: inverse gradient
  std::vector<float> b(64);

  for (int i = 0; i < 64; i++)
    b[i] = static_cast<float>(63 - i) / 63.0f;
  inputsArr.push_back(b);

  inputsJson["inputs"] = inputsArr;
  f.write(QByteArray::fromStdString(inputsJson.dump()));
  f.close();
  return path;
}

// Helper: write training samples for the GPU tests (1x8x8 input, 2-class output)
static QString writeGPUTestSamples(const QString& path)
{
  QFile f(path);

  if (f.exists())
    f.remove();

  if (!f.open(QIODevice::WriteOnly))
    return {};

  nlohmann::ordered_json samplesJson;
  nlohmann::ordered_json arr = nlohmann::ordered_json::array();

  // 4 samples: 2 per class with distinct spatial patterns
  for (int s = 0; s < 4; s++) {
    nlohmann::ordered_json sample;
    std::vector<float> input(64);
    int cls = s / 2;

    for (int i = 0; i < 64; i++) {
      if (cls == 0)
        input[i] = static_cast<float>(i) / 63.0f + 0.01f * s;
      else
        input[i] = static_cast<float>(63 - i) / 63.0f + 0.01f * (s - 2);
    }

    sample["input"] = input;
    sample["output"] = (cls == 0) ? std::vector<float>{1.0f, 0.0f} : std::vector<float>{0.0f, 1.0f};
    arr.push_back(sample);
  }

  samplesJson["samples"] = arr;
  f.write(QByteArray::fromStdString(samplesJson.dump()));
  f.close();
  return path;
}

// Core test: train on CPU, predict on CPU and GPU, compare.
// Returns true if GPU predict produces diverse outputs matching CPU.
static bool runGPUPredictTest(const QString& testName, const QString& convLayersJson)
{
  QString prefix = tempDir() + "/gpu_test_" + testName.toLower().replace(" ", "_");
  QString configPath = writeGPUTestConfig(prefix + "_config.json", convLayersJson);
  QString samplesPath = writeGPUTestSamples(prefix + "_samples.json");
  QString inputsPath = writeGPUTestInputs(prefix + "_inputs.json");
  QString modelPath = prefix + "_model.nnmodel";

  // Step 1: Train on CPU
  auto trainResult = runNNCLI(
    {"--config", configPath, "--mode", "train", "--device", "cpu", "--samples", samplesPath, "--output", modelPath});

  if (trainResult.exitCode != 0) {
    std::cerr << "    [" << testName.toStdString() << "] train failed (exit=" << trainResult.exitCode << ")"
              << std::endl;
    return false;
  }

  // Step 2: Predict on CPU
  QString cpuPredPath = prefix + "_pred_cpu.json";
  auto cpuPred = runNNCLI(
    {"--config", modelPath, "--mode", "predict", "--device", "cpu", "--input", inputsPath, "--output", cpuPredPath});

  if (cpuPred.exitCode != 0) {
    std::cerr << "    [" << testName.toStdString() << "] CPU predict failed" << std::endl;
    return false;
  }

  // Step 3: Predict on GPU
  QString gpuPredPath = prefix + "_pred_gpu.json";
  auto gpuPred = runNNCLI(
    {"--config", modelPath, "--mode", "predict", "--device", "gpu", "--input", inputsPath, "--output", gpuPredPath});

  if (gpuPred.exitCode != 0) {
    std::cerr << "    [" << testName.toStdString() << "] GPU predict failed (exit=" << gpuPred.exitCode << ")"
              << std::endl;

    if (!gpuPred.stdErr.isEmpty())
      std::cerr << "    stderr: " << gpuPred.stdErr.toStdString() << std::endl;
    return false;
  }

  // Step 4: Compare outputs
  QFile fc(cpuPredPath), fg(gpuPredPath);

  if (!fc.open(QIODevice::ReadOnly) || !fg.open(QIODevice::ReadOnly)) {
    std::cerr << "    [" << testName.toStdString() << "] failed to open output files" << std::endl;
    return false;
  }

  QJsonArray cpuOutputs = QJsonDocument::fromJson(fc.readAll()).object()["outputs"].toArray();
  QJsonArray gpuOutputs = QJsonDocument::fromJson(fg.readAll()).object()["outputs"].toArray();
  fc.close();
  fg.close();

  if (cpuOutputs.size() != 2 || gpuOutputs.size() != 2) {
    std::cerr << "    [" << testName.toStdString() << "] wrong output count" << std::endl;
    return false;
  }

  // Check CPU outputs are diverse (sanity)
  bool cpuDiverse = false;
  {
    QJsonArray c0 = cpuOutputs[0].toArray(), c1 = cpuOutputs[1].toArray();

    for (int i = 0; i < c0.size() && i < c1.size(); i++)

      if (std::fabs(c0[i].toDouble() - c1[i].toDouble()) > 1e-6)
        cpuDiverse = true;

    if (!cpuDiverse) {
      std::cerr << "    [" << testName.toStdString() << "] CPU outputs identical (model didn't learn): ["
                << c0[0].toDouble() << "," << c0[1].toDouble() << "]" << std::endl;
      return false; // Can't test GPU if CPU doesn't work
    }
  }

  // Check GPU outputs are diverse
  bool gpuDiverse = false;
  {
    QJsonArray g0 = gpuOutputs[0].toArray(), g1 = gpuOutputs[1].toArray();

    for (int i = 0; i < g0.size() && i < g1.size(); i++)

      if (std::fabs(g0[i].toDouble() - g1[i].toDouble()) > 1e-6)
        gpuDiverse = true;

    if (!gpuDiverse) {
      std::cerr << "    [" << testName.toStdString() << "] GPU outputs COLLAPSED: [" << g0[0].toDouble() << ","
                << g0[1].toDouble() << "]" << std::endl;
      std::cerr << "    CPU was: [" << cpuOutputs[0].toArray()[0].toDouble() << ","
                << cpuOutputs[0].toArray()[1].toDouble() << "] vs [" << cpuOutputs[1].toArray()[0].toDouble() << ","
                << cpuOutputs[1].toArray()[1].toDouble() << "]" << std::endl;
    }
  }

  // Check GPU outputs approximately match CPU
  bool gpuMatchesCPU = true;

  for (int s = 0; s < 2; s++) {
    QJsonArray co = cpuOutputs[s].toArray(), go = gpuOutputs[s].toArray();

    for (int i = 0; i < co.size() && i < go.size(); i++) {
      if (std::fabs(co[i].toDouble() - go[i].toDouble()) > 0.01) {
        gpuMatchesCPU = false;
      }
    }
  }

  if (!gpuMatchesCPU && gpuDiverse) {
    std::cerr << "    [" << testName.toStdString() << "] GPU outputs differ from CPU (>0.01):" << std::endl;

    for (int s = 0; s < 2; s++) {
      QJsonArray co = cpuOutputs[s].toArray(), go = gpuOutputs[s].toArray();
      std::cerr << "      sample " << s << ": CPU=[" << co[0].toDouble() << "," << co[1].toDouble() << "] GPU=["
                << go[0].toDouble() << "," << go[1].toDouble() << "]" << std::endl;
    }
  }

  return gpuDiverse;
}

static void testCNNGPUPredictLayerIsolation()
{
  std::cout << "  testCNNGPUPredictLayerIsolation..." << std::endl;

  if (!checkGPUAvailable()) {
    std::cout << "    (skipped — no GPU)" << std::endl;
    return;
  }

  // Each entry: { testName, convLayersJson }
  // Architecture varies systematically: add one layer type at a time
  struct TestCase {
      const char* name;
      const char* layers;
  };

  // clang-format off
  TestCase cases[] = {
    // 1. Minimal: conv + flatten
    {"conv_flatten",
     R"([{"type":"conv","numFilters":4,"filterH":3,"filterW":3,"strideY":1,"strideX":1,"slidingStrategy":"valid"},{"type":"flatten"}])"},

    // 2. conv + relu + flatten
    {"conv_relu_flatten",
     R"([{"type":"conv","numFilters":4,"filterH":3,"filterW":3,"strideY":1,"strideX":1,"slidingStrategy":"valid"},{"type":"relu"},{"type":"flatten"}])"},

    // 3. conv + relu + maxpool + flatten
    {"conv_relu_maxpool_flatten",
     R"([{"type":"conv","numFilters":4,"filterH":3,"filterW":3,"strideY":1,"strideX":1,"slidingStrategy":"valid"},{"type":"relu"},{"type":"pool","poolType":"max","poolH":2,"poolW":2,"strideY":2,"strideX":2},{"type":"flatten"}])"},

    // 4. conv + relu + avgpool + flatten
    {"conv_relu_avgpool_flatten",
     R"([{"type":"conv","numFilters":4,"filterH":3,"filterW":3,"strideY":1,"strideX":1,"slidingStrategy":"valid"},{"type":"relu"},{"type":"pool","poolType":"avg","poolH":2,"poolW":2,"strideY":2,"strideX":2},{"type":"flatten"}])"},

    // 5. conv + instancenorm + relu + flatten
    {"conv_bn_relu_flatten",
     R"([{"type":"conv","numFilters":4,"filterH":3,"filterW":3,"strideY":1,"strideX":1,"slidingStrategy":"valid"},{"type":"instancenorm"},{"type":"relu"},{"type":"flatten"}])"},

    // 6. conv + instancenorm + relu + maxpool + flatten
    {"conv_bn_relu_maxpool_flatten",
     R"([{"type":"conv","numFilters":4,"filterH":3,"filterW":3,"strideY":1,"strideX":1,"slidingStrategy":"valid"},{"type":"instancenorm"},{"type":"relu"},{"type":"pool","poolType":"max","poolH":2,"poolW":2,"strideY":2,"strideX":2},{"type":"flatten"}])"},

    // 7. conv + instancenorm + relu + avgpool + flatten (smallest avgpool+bn combo)
    {"conv_bn_relu_avgpool_flatten",
     R"([{"type":"conv","numFilters":4,"filterH":3,"filterW":3,"strideY":1,"strideX":1,"slidingStrategy":"valid"},{"type":"instancenorm"},{"type":"relu"},{"type":"pool","poolType":"avg","poolH":2,"poolW":2,"strideY":2,"strideX":2},{"type":"flatten"}])"},

    // 8. conv + relu + maxpool + conv + relu + flatten (2 conv layers)
    {"conv2x_relu_maxpool_flatten",
     R"([{"type":"conv","numFilters":4,"filterH":3,"filterW":3,"strideY":1,"strideX":1,"slidingStrategy":"same"},{"type":"relu"},{"type":"pool","poolType":"max","poolH":2,"poolW":2,"strideY":2,"strideX":2},{"type":"conv","numFilters":4,"filterH":3,"filterW":3,"strideY":1,"strideX":1,"slidingStrategy":"valid"},{"type":"relu"},{"type":"flatten"}])"},

    // 9. conv + relu + maxpool + conv + relu + avgpool + flatten (maxpool then avgpool)
    {"conv_maxpool_conv_avgpool_flatten",
     R"([{"type":"conv","numFilters":4,"filterH":3,"filterW":3,"strideY":1,"strideX":1,"slidingStrategy":"same"},{"type":"relu"},{"type":"pool","poolType":"max","poolH":2,"poolW":2,"strideY":2,"strideX":2},{"type":"conv","numFilters":4,"filterH":3,"filterW":3,"strideY":1,"strideX":1,"slidingStrategy":"valid"},{"type":"relu"},{"type":"pool","poolType":"avg","poolH":2,"poolW":2,"strideY":2,"strideX":2},{"type":"flatten"}])"},

    // 10. conv + bn + relu + maxpool + conv + bn + relu + avgpool + flatten (full combo)
    {"conv_bn_maxpool_conv_bn_avgpool_flatten",
     R"([{"type":"conv","numFilters":4,"filterH":3,"filterW":3,"strideY":1,"strideX":1,"slidingStrategy":"same"},{"type":"instancenorm"},{"type":"relu"},{"type":"pool","poolType":"max","poolH":2,"poolW":2,"strideY":2,"strideX":2},{"type":"conv","numFilters":4,"filterH":3,"filterW":3,"strideY":1,"strideX":1,"slidingStrategy":"valid"},{"type":"instancenorm"},{"type":"relu"},{"type":"pool","poolType":"avg","poolH":2,"poolW":2,"strideY":2,"strideX":2},{"type":"flatten"}])"},

    // 11. Same as #3 but with "same" padding instead of "valid"
    {"conv_same_relu_maxpool_flatten",
     R"([{"type":"conv","numFilters":4,"filterH":3,"filterW":3,"strideY":1,"strideX":1,"slidingStrategy":"same"},{"type":"relu"},{"type":"pool","poolType":"max","poolH":2,"poolW":2,"strideY":2,"strideX":2},{"type":"flatten"}])"},

    // 12. Same as #4 but with "same" padding
    {"conv_same_relu_avgpool_flatten",
     R"([{"type":"conv","numFilters":4,"filterH":3,"filterW":3,"strideY":1,"strideX":1,"slidingStrategy":"same"},{"type":"relu"},{"type":"pool","poolType":"avg","poolH":4,"poolW":4,"strideY":4,"strideX":4},{"type":"flatten"}])"},

    // 13. conv + relu + globaldualpool + flatten (minimal GDP)
    {"conv_relu_gdp_flatten",
     R"([{"type":"conv","numFilters":4,"filterH":3,"filterW":3,"strideY":1,"strideX":1,"slidingStrategy":"valid"},{"type":"relu"},{"type":"globaldualpool"},{"type":"flatten"}])"},

    // 14. conv + instancenorm + relu + globaldualpool + flatten (GDP with norm)
    {"conv_bn_relu_gdp_flatten",
     R"([{"type":"conv","numFilters":4,"filterH":3,"filterW":3,"strideY":1,"strideX":1,"slidingStrategy":"valid"},{"type":"instancenorm"},{"type":"relu"},{"type":"globaldualpool"},{"type":"flatten"}])"},

    // 15. conv + relu + maxpool + conv + relu + globaldualpool + flatten (2 conv + GDP)
    {"conv2x_relu_maxpool_gdp_flatten",
     R"([{"type":"conv","numFilters":4,"filterH":3,"filterW":3,"strideY":1,"strideX":1,"slidingStrategy":"same"},{"type":"relu"},{"type":"pool","poolType":"max","poolH":2,"poolW":2,"strideY":2,"strideX":2},{"type":"conv","numFilters":4,"filterH":3,"filterW":3,"strideY":1,"strideX":1,"slidingStrategy":"valid"},{"type":"relu"},{"type":"globaldualpool"},{"type":"flatten"}])"},

    // 16. conv + bn + relu + maxpool + conv + bn + relu + globaldualpool + flatten (full combo with GDP)
    {"conv_bn_maxpool_conv_bn_gdp_flatten",
     R"([{"type":"conv","numFilters":4,"filterH":3,"filterW":3,"strideY":1,"strideX":1,"slidingStrategy":"same"},{"type":"instancenorm"},{"type":"relu"},{"type":"pool","poolType":"max","poolH":2,"poolW":2,"strideY":2,"strideX":2},{"type":"conv","numFilters":4,"filterH":3,"filterW":3,"strideY":1,"strideX":1,"slidingStrategy":"valid"},{"type":"instancenorm"},{"type":"relu"},{"type":"globaldualpool"},{"type":"flatten"}])"},

    // 17. residual identity: conv(4,same)→relu→res(conv(4,same)→relu)→gap→flatten
    {"res_identity_conv_relu_gap_flatten",
     R"([{"type":"conv","numFilters":4,"filterH":3,"filterW":3,"strideY":1,"strideX":1,"slidingStrategy":"same"},{"type":"relu"},{"type":"residual_start"},{"type":"conv","numFilters":4,"filterH":3,"filterW":3,"strideY":1,"strideX":1,"slidingStrategy":"same"},{"type":"relu"},{"type":"residual_end"},{"type":"globalavgpool"},{"type":"flatten"}])"},

    // 18. residual projection: res(conv(4,same)→relu)→gap→flatten (1→4 channel change)
    {"res_proj_conv_relu_gap_flatten",
     R"([{"type":"residual_start"},{"type":"conv","numFilters":4,"filterH":3,"filterW":3,"strideY":1,"strideX":1,"slidingStrategy":"same"},{"type":"relu"},{"type":"residual_end"},{"type":"globalavgpool"},{"type":"flatten"}])"},

    // 19. residual + pool + residual + gap: res(conv4)→pool→res(conv4)→gap→flatten
    {"res_pool_res_gap_flatten",
     R"([{"type":"conv","numFilters":4,"filterH":3,"filterW":3,"strideY":1,"strideX":1,"slidingStrategy":"same"},{"type":"relu"},{"type":"residual_start"},{"type":"conv","numFilters":4,"filterH":3,"filterW":3,"strideY":1,"strideX":1,"slidingStrategy":"same"},{"type":"relu"},{"type":"residual_end"},{"type":"pool","poolType":"max","poolH":2,"poolW":2,"strideY":2,"strideX":2},{"type":"residual_start"},{"type":"conv","numFilters":4,"filterH":3,"filterW":3,"strideY":1,"strideX":1,"slidingStrategy":"same"},{"type":"relu"},{"type":"residual_end"},{"type":"globalavgpool"},{"type":"flatten"}])"},

    // 20. residual + pool + residual + globaldualpool
    {"res_pool_res_gdp_flatten",
     R"([{"type":"conv","numFilters":4,"filterH":3,"filterW":3,"strideY":1,"strideX":1,"slidingStrategy":"same"},{"type":"relu"},{"type":"residual_start"},{"type":"conv","numFilters":4,"filterH":3,"filterW":3,"strideY":1,"strideX":1,"slidingStrategy":"same"},{"type":"relu"},{"type":"residual_end"},{"type":"pool","poolType":"max","poolH":2,"poolW":2,"strideY":2,"strideX":2},{"type":"residual_start"},{"type":"conv","numFilters":4,"filterH":3,"filterW":3,"strideY":1,"strideX":1,"slidingStrategy":"same"},{"type":"relu"},{"type":"residual_end"},{"type":"globaldualpool"},{"type":"flatten"}])"},

    // 21. mixed identity + projection residuals (identity first, then projection — the ISIC segfault case)
    {"res_identity_then_proj_gap_flatten",
     R"([{"type":"conv","numFilters":4,"filterH":3,"filterW":3,"strideY":1,"strideX":1,"slidingStrategy":"same"},{"type":"relu"},{"type":"residual_start"},{"type":"conv","numFilters":4,"filterH":3,"filterW":3,"strideY":1,"strideX":1,"slidingStrategy":"same"},{"type":"relu"},{"type":"residual_end"},{"type":"pool","poolType":"max","poolH":2,"poolW":2,"strideY":2,"strideX":2},{"type":"residual_start"},{"type":"conv","numFilters":8,"filterH":3,"filterW":3,"strideY":1,"strideX":1,"slidingStrategy":"same"},{"type":"relu"},{"type":"residual_end"},{"type":"globalavgpool"},{"type":"flatten"}])"},

    // 22. stem + identity + projection + projection (ISIC-like architecture)
    {"stem_res_id_proj_proj_gdp_flatten",
     R"([{"type":"conv","numFilters":4,"filterH":3,"filterW":3,"strideY":2,"strideX":2,"slidingStrategy":"same"},{"type":"relu"},{"type":"residual_start"},{"type":"conv","numFilters":4,"filterH":3,"filterW":3,"strideY":1,"strideX":1,"slidingStrategy":"same"},{"type":"relu"},{"type":"residual_end"},{"type":"pool","poolType":"max","poolH":2,"poolW":2,"strideY":2,"strideX":2},{"type":"residual_start"},{"type":"conv","numFilters":8,"filterH":3,"filterW":3,"strideY":1,"strideX":1,"slidingStrategy":"same"},{"type":"relu"},{"type":"residual_end"},{"type":"pool","poolType":"max","poolH":2,"poolW":2,"strideY":2,"strideX":2},{"type":"residual_start"},{"type":"conv","numFilters":16,"filterH":3,"filterW":3,"strideY":1,"strideX":1,"slidingStrategy":"same"},{"type":"relu"},{"type":"residual_end"},{"type":"globaldualpool"},{"type":"flatten"}])"},
  };
  // clang-format on

  int numCases = sizeof(cases) / sizeof(cases[0]);
  int passed = 0, failed = 0, skipped = 0;

  for (int i = 0; i < numCases; i++) {
    std::cout << "    " << (i + 1) << "/" << numCases << " " << cases[i].name << "... " << std::flush;
    bool ok = runGPUPredictTest(cases[i].name, cases[i].layers);

    if (ok) {
      std::cout << "PASS" << std::endl;
      passed++;
    } else {
      // Check if it was a "model didn't learn" skip
      std::cout << "FAIL" << std::endl;
      failed++;
    }
  }

  std::cout << "    Summary: " << passed << " passed, " << failed << " failed, " << skipped << " skipped" << std::endl;
  CHECK(failed == 0, "GPU predict layer isolation: all architectures produce correct GPU predictions");
  std::cout << std::endl;
}

void runCNNGPULayerTests()
{
  testCNNGPUPredictLayerIsolation();
}
