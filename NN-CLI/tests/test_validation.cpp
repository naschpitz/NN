#include "test_helpers.hpp"
#include "../NN-CLI_DataSplitter.hpp"
#include "../NN-CLI_Loader.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <vector>

using namespace NN_CLI;

//===================================================================================================================//

// Helper: create output vectors for N samples across numClasses.
// Distribution: class 0 gets 50% of samples, rest split evenly.
static std::vector<std::vector<float>> makeImbalancedOutputs(ulong count, ulong numClasses = 3)
{
  std::vector<std::vector<float>> outputs(count);

  for (ulong i = 0; i < count; i++) {
    outputs[i].assign(numClasses, 0.0f);

    if (i < count / 2)
      outputs[i][0] = 1.0f; // 50% class 0
    else
      outputs[i][1 + (i % (numClasses - 1))] = 1.0f; // rest split among 1..N-1
  }

  return outputs;
}

//===================================================================================================================//

static void testAutoValSizeThresholds()
{
  std::cout << "  testAutoValSizeThresholds... ";

  // Exact anchor points
  CHECK_NEAR(DataSplitter::computeAutoValSize(100), 0.20f, 0.001f, "100 → 20%");
  CHECK_NEAR(DataSplitter::computeAutoValSize(1000), 0.15f, 0.001f, "1k → 15%");
  CHECK_NEAR(DataSplitter::computeAutoValSize(10000), 0.10f, 0.001f, "10k → 10%");
  CHECK_NEAR(DataSplitter::computeAutoValSize(100000), 0.05f, 0.001f, "100k → 5%");
  CHECK_NEAR(DataSplitter::computeAutoValSize(1000000), 0.01f, 0.001f, "1M → 1%");

  // Smooth interpolation: midpoints should be between anchors, not cliff edges
  float mid1k10k = DataSplitter::computeAutoValSize(3162); // ~sqrt(1k*10k)
  CHECK(mid1k10k > 0.10f && mid1k10k < 0.15f, "3.2k is between 10% and 15%");

  float mid10k100k = DataSplitter::computeAutoValSize(31623); // ~sqrt(10k*100k)
  CHECK(mid10k100k > 0.05f && mid10k100k < 0.10f, "31.6k is between 5% and 10%");

  float mid100k1M = DataSplitter::computeAutoValSize(316228); // ~sqrt(100k*1M)
  CHECK(mid100k1M > 0.01f && mid100k1M < 0.05f, "316k is between 1% and 5%");

  // Monotonically decreasing
  CHECK(DataSplitter::computeAutoValSize(500) > DataSplitter::computeAutoValSize(5000), "500 > 5k ratio");
  CHECK(DataSplitter::computeAutoValSize(5000) > DataSplitter::computeAutoValSize(50000), "5k > 50k ratio");
  CHECK(DataSplitter::computeAutoValSize(50000) > DataSplitter::computeAutoValSize(500000), "50k > 500k ratio");

  // Clamping at boundaries
  CHECK_NEAR(DataSplitter::computeAutoValSize(10), 0.20f, 0.001f, "tiny → clamped to 20%");
  CHECK_NEAR(DataSplitter::computeAutoValSize(10000000), 0.01f, 0.001f, "10M → clamped to 1%");

  std::cout << "PASS" << std::endl;
}

//===================================================================================================================//

static void testStratifiedSplitBasic()
{
  std::cout << "  testStratifiedSplitBasic... ";

  // 100 samples, 3 classes: 50 class-0, 25 class-1, 25 class-2
  auto outputs = makeImbalancedOutputs(100, 3);
  DataSplit split = DataSplitter::stratifiedSplit(outputs, 0.20f);

  // Total should be preserved
  CHECK(split.trainIndices.size() + split.validationIndices.size() == 100, "total preserved");

  // Roughly 20% validation
  CHECK(split.validationIndices.size() >= 18 && split.validationIndices.size() <= 22, "~20% validation");

  // No duplicates
  std::vector<ulong> all;
  all.insert(all.end(), split.trainIndices.begin(), split.trainIndices.end());
  all.insert(all.end(), split.validationIndices.begin(), split.validationIndices.end());
  std::sort(all.begin(), all.end());

  for (ulong i = 1; i < all.size(); i++) {
    CHECK(all[i] != all[i - 1], "no duplicate indices");
  }

  std::cout << "PASS" << std::endl;
}

//===================================================================================================================//

static void testStratifiedSplitPreservesClassDistribution()
{
  std::cout << "  testStratifiedSplitPreservesClassDistribution... ";

  ulong numClasses = 4;
  ulong count = 200;
  auto outputs = makeImbalancedOutputs(count, numClasses);

  DataSplit split = DataSplitter::stratifiedSplit(outputs, 0.20f);

  // Count class distribution in train and val
  std::map<ulong, ulong> trainCounts, validationCounts, totalCounts;

  for (ulong idx : split.trainIndices) {
    const auto& out = outputs[idx];
    ulong cls = static_cast<ulong>(std::distance(out.begin(), std::max_element(out.begin(), out.end())));
    trainCounts[cls]++;
  }

  for (ulong idx : split.validationIndices) {
    const auto& out = outputs[idx];
    ulong cls = static_cast<ulong>(std::distance(out.begin(), std::max_element(out.begin(), out.end())));
    validationCounts[cls]++;
  }

  for (ulong i = 0; i < count; i++) {
    ulong cls =
      static_cast<ulong>(std::distance(outputs[i].begin(), std::max_element(outputs[i].begin(), outputs[i].end())));
    totalCounts[cls]++;
  }

  // Every class present in total should be present in both splits
  for (const auto& [cls, total] : totalCounts) {
    CHECK(trainCounts.count(cls) > 0, "class present in train split");
    CHECK(validationCounts.count(cls) > 0, "class present in validation split");

    // Val should be roughly 20% of each class (within ±2 samples due to rounding)
    ulong expectedVal = static_cast<ulong>(std::round(total * 0.20f));
    ulong actualVal = validationCounts[cls];
    CHECK(actualVal >= expectedVal - 2 && actualVal <= expectedVal + 2,
          "val count proportional for class " + std::to_string(cls));
  }

  std::cout << "PASS" << std::endl;
}

//===================================================================================================================//

static void testStratifiedSplitDeterministic()
{
  std::cout << "  testStratifiedSplitDeterministic... ";

  auto outputs = makeImbalancedOutputs(100, 3);

  DataSplit split1 = DataSplitter::stratifiedSplit(outputs, 0.15f, 42);
  DataSplit split2 = DataSplitter::stratifiedSplit(outputs, 0.15f, 42);

  CHECK(split1.trainIndices == split2.trainIndices, "train indices identical with same seed");
  CHECK(split1.validationIndices == split2.validationIndices, "val indices identical with same seed");

  // Different seed should produce different split
  DataSplit split3 = DataSplitter::stratifiedSplit(outputs, 0.15f, 99);
  CHECK(split3.validationIndices != split1.validationIndices, "different seed → different split");

  std::cout << "PASS" << std::endl;
}

//===================================================================================================================//

static void testStratifiedSplitEdgeCases()
{
  std::cout << "  testStratifiedSplitEdgeCases... ";

  // Single sample
  std::vector<std::vector<float>> single = {{1.0f, 0.0f}};
  DataSplit splitSingle = DataSplitter::stratifiedSplit(single, 0.20f);
  CHECK(splitSingle.trainIndices.size() + splitSingle.validationIndices.size() == 1, "single sample total preserved");

  // Zero ratio — all training
  auto outputs = makeImbalancedOutputs(50, 2);
  DataSplit splitZero = DataSplitter::stratifiedSplit(outputs, 0.0f);
  CHECK(splitZero.validationIndices.empty(), "zero ratio → no validation");
  CHECK(splitZero.trainIndices.size() == 50, "zero ratio → all training");

  // 100% ratio — all validation
  DataSplit splitAll = DataSplitter::stratifiedSplit(outputs, 1.0f);
  CHECK(splitAll.trainIndices.empty(), "100% ratio → no training");
  CHECK(splitAll.validationIndices.size() == 50, "100% ratio → all validation");

  std::cout << "PASS" << std::endl;
}

//===================================================================================================================//

static void testValidationConfigParsing()
{
  std::cout << "  testValidationConfigParsing... ";

  // Create a temp config file with validationConfig
  QString configPath = tempDir() + "/val_config_test.json";
  QFile file(configPath);
  file.open(QIODevice::WriteOnly);
  file.write(R"({
    "layers": [{"numNeurons": 4, "actvFunc": "relu"}],
    "train": {
      "numEpochs": 10,
      "learningRate": 0.01,
      "validation": {
        "enabled": false,
        "autoSize": false,
        "size": 0.25,
        "checkInterval": 5
      }
    }
  })");

  file.close();

  AugmentationConfig config = Loader::loadAugmentationConfig(configPath.toStdString());

  CHECK(config.validationConfig.enabled == false, "enabled parsed correctly");
  CHECK(config.validationConfig.autoSize == false, "autoSize parsed correctly");
  CHECK_NEAR(config.validationConfig.size, 0.25f, 0.001f, "size parsed correctly");
  CHECK(config.validationConfig.checkInterval == 5, "checkInterval parsed correctly");

  std::cout << "PASS" << std::endl;
}

//===================================================================================================================//

static void testValidationConfigDefaults()
{
  std::cout << "  testValidationConfigDefaults... ";

  // Config without validationConfig — should use defaults
  QString configPath = tempDir() + "/val_config_defaults.json";
  QFile file(configPath);
  file.open(QIODevice::WriteOnly);
  file.write(R"({
    "layers": [{"numNeurons": 4, "actvFunc": "relu"}],
    "train": {
      "numEpochs": 10,
      "learningRate": 0.01
    }
  })");

  file.close();

  AugmentationConfig config = Loader::loadAugmentationConfig(configPath.toStdString());

  CHECK(config.validationConfig.enabled == true, "default enabled is true");
  CHECK(config.validationConfig.autoSize == true, "default autoSize is true");
  CHECK_NEAR(config.validationConfig.size, 0.15f, 0.001f, "default size is 0.15");
  CHECK(config.validationConfig.checkInterval == 1, "default checkInterval is 1");

  std::cout << "PASS" << std::endl;
}

//===================================================================================================================//

void runValidationTests()
{
  testAutoValSizeThresholds();
  testStratifiedSplitBasic();
  testStratifiedSplitPreservesClassDistribution();
  testStratifiedSplitDeterministic();
  testStratifiedSplitEdgeCases();
  testValidationConfigParsing();
  testValidationConfigDefaults();
}
