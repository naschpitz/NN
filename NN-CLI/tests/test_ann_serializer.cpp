#include "test_helpers.hpp"

#include <QFile>

#include <ANN_Core.hpp>
#include <ANN_CoreConfig.hpp>
#include <ANN_LayersConfig.hpp>
#include <ANN_ActvFunc.hpp>

#include "NN-CLI_ModelSerializer.hpp"

#include <cmath>
#include <string>

//===================================================================================================================//

//
// Tests for the ANN binary parameter serializer.
//
// The ANN serializer (saveANNParametersBinary / loadANNParametersBinary) writes
// and reads weights/biases for all layers, including the input-layer placeholder
// at index 0. These tests verify:
//   - Every weight and bias value survives a binary save→load round-trip
//   - Predictions from a freshly-built core match those from a loaded core
//   - Binary format header validation (wrong magic, version, type → throws)
//

//--- Helpers --//

static std::vector<char> saveAndReadBackANN(const ANN::Core<float>& core, const QString& filename)
{
  QString binPath = tempDir() + "/" + filename;
  NN_CLI::ModelSerializer::saveANNParametersBinary(binPath.toStdString(), core);

  QFile binFile(binPath);

  if (!binFile.open(QIODevice::ReadOnly))
    return {};

  QByteArray binData = binFile.readAll();
  binFile.close();
  return std::vector<char>(binData.constData(), binData.constData() + binData.size());
}

static int verifyANNParams(const ANN::Parameters<float>& original, const ANN::Parameters<float>& loaded,
                           const std::string& label)
{
  int checks = 0;

  CHECK(loaded.weights.size() == original.weights.size(), label + ": weights layer count");
  CHECK(loaded.biases.size() == original.biases.size(), label + ": biases layer count");
  checks += 2;

  if (loaded.weights.size() != original.weights.size())
    return checks;

  for (size_t layer = 0; layer < original.weights.size(); ++layer) {
    if (original.weights[layer].empty())
      continue;

    for (size_t neuron = 0; neuron < original.weights[layer].size(); ++neuron) {
      for (size_t input = 0; input < original.weights[layer][neuron].size(); ++input) {
        CHECK_NEAR(loaded.weights[layer][neuron][input], original.weights[layer][neuron][input], 1e-6f,
                   label + ": w[" + std::to_string(layer) + "][" + std::to_string(neuron) + "][" +
                     std::to_string(input) + "]");
        checks++;
      }
    }

    for (size_t neuron = 0; neuron < original.biases[layer].size(); ++neuron) {
      CHECK_NEAR(loaded.biases[layer][neuron], original.biases[layer][neuron], 1e-6f,
                 label + ": b[" + std::to_string(layer) + "][" + std::to_string(neuron) + "]");
      checks++;
    }
  }

  return checks;
}

static void fillANNParams(ANN::Parameters<float>& params, const std::vector<ulong>& layerSizes)
{
  params.weights.resize(layerSizes.size());
  params.biases.resize(layerSizes.size());

  // Input layer (index 0): empty
  params.weights[0] = {};
  params.biases[0] = {};

  for (size_t i = 1; i < layerSizes.size(); ++i) {
    ulong numNeurons = layerSizes[i];
    ulong numInputs = layerSizes[i - 1];

    params.weights[i].resize(numNeurons);
    params.biases[i].resize(numNeurons);

    for (ulong j = 0; j < numNeurons; ++j) {
      params.weights[i][j].resize(numInputs);

      for (ulong k = 0; k < numInputs; ++k)
        params.weights[i][j][k] = static_cast<float>(100 * i + 10 * j) + static_cast<float>(k) * 0.1f;
      params.biases[i][j] = static_cast<float>(1000 * i + j) + 0.5f;
    }
  }
}

static ANN::LayersConfig makeANNLayers(std::initializer_list<std::pair<ulong, ANN::ActvFuncType>> layers)
{
  ANN::LayersConfig config;

  for (const auto& [numNeurons, actvFunc] : layers)
    config.push_back({numNeurons, actvFunc});

  return config;
}

//--- Tests --//

//
// Exhaustive binary round-trip: every weight and bias value verified.
//
static void testANNBinaryParamsExhaustive()
{
  TestScope _t("testANNBinaryParamsExhaustive");

  ANN::CoreConfig<float> config;
  config.modeType = Common::ModeType::PREDICT;
  config.deviceType = Common::DeviceType::CPU;
  config.layersConfig =
    makeANNLayers({{4, ANN::ActvFuncType::RELU}, {3, ANN::ActvFuncType::SIGMOID}, {2, ANN::ActvFuncType::SOFTMAX}});

  fillANNParams(config.parameters, {4, 3, 2});

  auto core = ANN::Core<float>::makeCore(config);
  CHECK(core != nullptr, "ANN core created");

  const auto& original = core->getParameters();

  auto data = saveAndReadBackANN(*core, "ann_serializer_exhaustive.bin");
  CHECK(!data.empty(), "params.bin written");

  if (data.empty())
    return;

  ANN::CoreConfig<float> loadedConfig;
  loadedConfig.modeType = Common::ModeType::PREDICT;
  loadedConfig.deviceType = Common::DeviceType::CPU;
  loadedConfig.layersConfig = config.layersConfig;

  NN_CLI::ModelSerializer::loadANNParametersBinary(data, loadedConfig, loadedConfig.layersConfig);

  int checks = verifyANNParams(original, loadedConfig.parameters, "ann-exhaustive");
  std::cout << "  (" << checks << " individual checks)" << std::endl;
}

//
// Larger architecture (5 layers) to stress-test the binary format.
//
static void testANNBinaryLargeArchitecture()
{
  TestScope _t("testANNBinaryLargeArchitecture");

  ANN::CoreConfig<float> config;
  config.modeType = Common::ModeType::PREDICT;
  config.deviceType = Common::DeviceType::CPU;
  config.layersConfig = makeANNLayers({{10, ANN::ActvFuncType::RELU},
                                       {8, ANN::ActvFuncType::RELU},
                                       {6, ANN::ActvFuncType::TANH},
                                       {4, ANN::ActvFuncType::RELU},
                                       {2, ANN::ActvFuncType::SOFTMAX}});

  fillANNParams(config.parameters, {10, 8, 6, 4, 2});

  auto core = ANN::Core<float>::makeCore(config);
  CHECK(core != nullptr, "ANN core created");

  const auto& original = core->getParameters();

  auto data = saveAndReadBackANN(*core, "ann_serializer_large.bin");
  CHECK(!data.empty(), "params.bin written");

  if (data.empty())
    return;

  ANN::CoreConfig<float> loadedConfig;
  loadedConfig.modeType = Common::ModeType::PREDICT;
  loadedConfig.deviceType = Common::DeviceType::CPU;
  loadedConfig.layersConfig = config.layersConfig;

  NN_CLI::ModelSerializer::loadANNParametersBinary(data, loadedConfig, loadedConfig.layersConfig);

  // Build core from loaded config and verify via getParameters()
  auto loadedCore = ANN::Core<float>::makeCore(loadedConfig);
  CHECK(loadedCore != nullptr, "loaded ANN core created");

  int checks = verifyANNParams(original, loadedCore->getParameters(), "ann-large");
  std::cout << "  (" << checks << " individual checks)" << std::endl;
}

//
// Predict consistency: coreA (original) vs coreB (from binary round-trip).
// Five different inputs, verify every output and logit matches.
//
static void testANNBinaryPredictConsistency()
{
  TestScope _t("testANNBinaryPredictConsistency");

  ANN::CoreConfig<float> config;
  config.modeType = Common::ModeType::PREDICT;
  config.deviceType = Common::DeviceType::CPU;
  config.layersConfig =
    makeANNLayers({{5, ANN::ActvFuncType::RELU}, {4, ANN::ActvFuncType::RELU}, {2, ANN::ActvFuncType::SOFTMAX}});

  fillANNParams(config.parameters, {5, 4, 2});

  auto coreA = ANN::Core<float>::makeCore(config);
  CHECK(coreA != nullptr, "coreA created");

  auto data = saveAndReadBackANN(*coreA, "ann_serializer_predict.bin");
  CHECK(!data.empty(), "params.bin written");

  if (data.empty())
    return;

  ANN::CoreConfig<float> loadedConfig;
  loadedConfig.modeType = Common::ModeType::PREDICT;
  loadedConfig.deviceType = Common::DeviceType::CPU;
  loadedConfig.layersConfig = config.layersConfig;

  NN_CLI::ModelSerializer::loadANNParametersBinary(data, loadedConfig, loadedConfig.layersConfig);

  auto coreB = ANN::Core<float>::makeCore(loadedConfig);
  CHECK(coreB != nullptr, "coreB created");

  for (int trial = 0; trial < 5; ++trial) {
    ANN::Input<float> input;
    input.resize(5);

    for (int i = 0; i < 5; ++i)
      input[i] = static_cast<float>(i) * 0.1f * static_cast<float>(trial + 1);

    auto resultA = coreA->predict(input);
    auto resultB = coreB->predict(input);

    CHECK(resultA.output.size() == 2, "output size = 2");

    if (resultA.output.size() == 2 && resultB.output.size() == 2) {
      CHECK_NEAR(resultA.output[0], resultB.output[0], 1e-5f, "output[0] trial " + std::to_string(trial));
      CHECK_NEAR(resultA.output[1], resultB.output[1], 1e-5f, "output[1] trial " + std::to_string(trial));

      if (resultA.logits.size() == 2 && resultB.logits.size() == 2) {
        CHECK_NEAR(resultA.logits[0], resultB.logits[0], 1e-5f, "logits[0] trial " + std::to_string(trial));
        CHECK_NEAR(resultA.logits[1], resultB.logits[1], 1e-5f, "logits[1] trial " + std::to_string(trial));
      }
    }
  }
}

//
// Binary format header validation: corrupted data must throw.
//
static void testANNBinaryCorruptionThrows()
{
  TestScope _t("testANNBinaryCorruptionThrows");

  // Build a valid binary first
  ANN::CoreConfig<float> config;
  config.modeType = Common::ModeType::PREDICT;
  config.deviceType = Common::DeviceType::CPU;
  config.layersConfig = makeANNLayers({{2, ANN::ActvFuncType::RELU}, {1, ANN::ActvFuncType::SIGMOID}});
  fillANNParams(config.parameters, {2, 1});

  auto core = ANN::Core<float>::makeCore(config);
  auto validData = saveAndReadBackANN(*core, "ann_serializer_corrupt.bin");
  CHECK(!validData.empty(), "valid binary created");

  if (validData.empty())
    return;

  // Test 1: truncated data (too small for header)
  {
    std::vector<char> truncated(validData.begin(), validData.begin() + 10);
    bool threw = false;

    try {
      ANN::CoreConfig<float> lc;
      lc.layersConfig = config.layersConfig;
      NN_CLI::ModelSerializer::loadANNParametersBinary(truncated, lc, lc.layersConfig);
    } catch (...) {
      threw = true;
    }

    CHECK(threw, "truncated data (10 bytes) throws");
  }

  // Test 2: wrong magic
  {
    std::vector<char> bad = validData;
    bad[0] = 0;
    bad[1] = 0;
    bad[2] = 0;
    bad[3] = 0; // overwrite magic
    bool threw = false;

    try {
      ANN::CoreConfig<float> lc;
      lc.layersConfig = config.layersConfig;
      NN_CLI::ModelSerializer::loadANNParametersBinary(bad, lc, lc.layersConfig);
    } catch (...) {
      threw = true;
    }

    CHECK(threw, "wrong magic throws");
  }

  // Test 3: wrong version
  {
    std::vector<char> bad = validData;
    bad[6] = 99; // version byte
    bool threw = false;

    try {
      ANN::CoreConfig<float> lc;
      lc.layersConfig = config.layersConfig;
      NN_CLI::ModelSerializer::loadANNParametersBinary(bad, lc, lc.layersConfig);
    } catch (...) {
      threw = true;
    }

    CHECK(threw, "wrong version throws");
  }

  // Test 4: wrong model type (CNN instead of ANN)
  {
    std::vector<char> bad = validData;
    bad[8] = 1; // BINARY_MODEL_CNN instead of BINARY_MODEL_ANN
    bool threw = false;

    try {
      ANN::CoreConfig<float> lc;
      lc.layersConfig = config.layersConfig;
      NN_CLI::ModelSerializer::loadANNParametersBinary(bad, lc, lc.layersConfig);
    } catch (...) {
      threw = true;
    }

    CHECK(threw, "wrong model type (CNN instead of ANN) throws");
  }
}

//===================================================================================================================//

void runANNSerializerTests()
{
  testANNBinaryParamsExhaustive();
  testANNBinaryLargeArchitecture();
  testANNBinaryPredictConsistency();
  testANNBinaryCorruptionThrows();
}
