#include "test_helpers.hpp"

#include <QDir>
#include <QFile>

#include <CNN_Core.hpp>
#include <CNN_CoreConfig.hpp>
#include <CNN_LayersConfig.hpp>
#include <CNN_SlidingStrategy.hpp>
#include <ANN_ActvFunc.hpp>
#include <ANN_Core.hpp>

#include "NN-CLI_ModelSerializer.hpp"
#include "NN-CLI_ModelPackage.hpp"
#include "NN-CLI_IOConfig.hpp"
#include "NN-CLI_AugmentationConfig.hpp"

#include <cmath>
#include <string>

//===================================================================================================================//

//
// Background:
//   The CNN binary serializer writes dense (ANN) parameters as a flat sequence of
//   BLOCK_ANN_WEIGHTS + BLOCK_ANN_BIASES pairs. The ANN core stores parameters with
//   an extra entry at index 0 — a placeholder for the input layer that has no weights.
//   So a CNN with N dense layers produces N+1 pairs in the binary.
//
//   loadCNNParametersBinary() previously read only layersConfig.denseLayers.size()
//   pairs (= N, without the input-layer placeholder). This left the last dense layer
//   unread. Worse, ANN_CoreCPU::initializeParameters() then saw
//   weights.size() (N) != numLayers (N+1) and discarded ALL loaded weights,
//   re-initializing them with a fixed random seed (std::mt19937 gen(42)).
//
//   These tests verify a save → load round-trip preserves ALL dense layer weights
//   and that predictions are identical before and after serialization.
//

//--- Helpers --//

//
// Save a CNN core's parameters to a temp binary file, read the bytes back.
// Returns empty vector on failure.
//
static std::vector<char> saveAndReadBack(const CNN::Core<float>& core, const QString& filename)
{
  QString binPath = tempDir() + "/" + filename;
  NN_CLI::ModelSerializer::saveCNNParametersBinary(binPath.toStdString(), core);

  QFile binFile(binPath);

  if (!binFile.open(QIODevice::ReadOnly))
    return {};

  QByteArray binData = binFile.readAll();
  binFile.close();
  return std::vector<char>(binData.constData(), binData.constData() + binData.size());
}

//
// Verify every weight and bias value in a loaded dense parameter set matches the original.
// Returns the number of individual value checks performed.
//
static int verifyAllDenseParams(const ANN::Parameters<float>& original, const ANN::Parameters<float>& loaded,
                                const std::string& label)
{
  int checks = 0;

  CHECK(loaded.weights.size() == original.weights.size(), label + ": weights entry count matches (got " +
                                                            std::to_string(loaded.weights.size()) + ", expected " +
                                                            std::to_string(original.weights.size()) + ")");
  checks++;

  CHECK(loaded.biases.size() == original.biases.size(), label + ": biases entry count matches (got " +
                                                          std::to_string(loaded.biases.size()) + ", expected " +
                                                          std::to_string(original.biases.size()) + ")");
  checks++;

  if (loaded.weights.size() != original.weights.size())
    return checks;

  for (size_t layer = 0; layer < original.weights.size(); ++layer) {
    // Skip the input-layer placeholder (index 0) — it's empty
    if (original.weights[layer].empty())
      continue;

    CHECK(loaded.weights[layer].size() == original.weights[layer].size(),
          label + ": weights[" + std::to_string(layer) + "] neuron count matches");
    checks++;

    if (loaded.weights[layer].size() != original.weights[layer].size())
      continue;

    for (size_t neuron = 0; neuron < original.weights[layer].size(); ++neuron) {
      CHECK(loaded.weights[layer][neuron].size() == original.weights[layer][neuron].size(),
            label + ": weights[" + std::to_string(layer) + "][" + std::to_string(neuron) + "] input count matches");
      checks++;

      if (loaded.weights[layer][neuron].size() != original.weights[layer][neuron].size())
        continue;

      for (size_t input = 0; input < original.weights[layer][neuron].size(); ++input) {
        CHECK_NEAR(loaded.weights[layer][neuron][input], original.weights[layer][neuron][input], 1e-6f,
                   label + ": w[" + std::to_string(layer) + "][" + std::to_string(neuron) + "][" +
                     std::to_string(input) + "]");
        checks++;
      }
    }
  }

  for (size_t layer = 0; layer < original.biases.size(); ++layer) {
    if (original.biases[layer].empty())
      continue;

    CHECK(loaded.biases[layer].size() == original.biases[layer].size(),
          label + ": biases[" + std::to_string(layer) + "] count matches");
    checks++;

    if (loaded.biases[layer].size() != original.biases[layer].size())
      continue;

    for (size_t neuron = 0; neuron < original.biases[layer].size(); ++neuron) {
      CHECK_NEAR(loaded.biases[layer][neuron], original.biases[layer][neuron], 1e-6f,
                 label + ": b[" + std::to_string(layer) + "][" + std::to_string(neuron) + "]");
      checks++;
    }
  }

  return checks;
}

//
// Fill dense weights with a deterministic pattern that is very unlikely to match
// random init (seed 42). Values are large and distinctive.
//
static void fillDenseWeights(ANN::Parameters<float>& params, const std::vector<ulong>& layerSizes)
{
  // layerSizes = [flattenSize, dense1, dense2, ...]
  // params.weights has layerSizes.size() entries
  // params.weights[0] is empty (input layer)
  // params.weights[i] is [layerSizes[i]][layerSizes[i-1]]

  params.weights.resize(layerSizes.size());
  params.biases.resize(layerSizes.size());

  params.weights[0] = {};
  params.biases[0] = {};

  for (size_t i = 1; i < layerSizes.size(); ++i) {
    ulong numNeurons = layerSizes[i];
    ulong numInputs = layerSizes[i - 1];

    params.weights[i].resize(numNeurons);
    params.biases[i].resize(numNeurons);

    for (ulong j = 0; j < numNeurons; ++j) {
      params.weights[i][j].resize(numInputs);

      for (ulong k = 0; k < numInputs; ++k) {
        // Distinctive pattern: 100 * layerIdx + 10 * neuron + k * 0.1
        params.weights[i][j][k] = static_cast<float>(100 * i + 10 * j) + static_cast<float>(k) * 0.1f;
      }

      params.biases[i][j] = static_cast<float>(1000 * i + j) + 0.5f;
    }
  }
}

//--- Tests --//

//
// TEST 1: Two dense layers — exhaustive verification of EVERY weight and bias value.
// This is the minimal case that triggers the bug (N=2 → reads 2, should read 3).
//
static void testCNNDenseParamsExhaustive()
{
  TestScope _t("testCNNDenseParamsExhaustive");

  CNN::CoreConfig<float> config;
  config.modeType = Common::ModeType::PREDICT;
  config.deviceType = Common::DeviceType::CPU;
  config.inputShape = {1, 4, 4};
  config.progressReports = 0;

  config.layersConfig.cnnLayers.push_back(
    {CNN::LayerType::CONV, CNN::ConvLayerConfig{1, 3, 3, 1, 1, CNN::SlidingStrategyType::VALID}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::RELU, CNN::ReLULayerConfig{}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::FLATTEN, CNN::FlattenLayerConfig{}});

  config.layersConfig.denseLayers.push_back({3, ANN::ActvFuncType::SIGMOID});
  config.layersConfig.denseLayers.push_back({2, ANN::ActvFuncType::SOFTMAX});

  // Conv output: 1x2x2 = 4 after flatten
  fillDenseWeights(config.parameters.denseParams, {4, 3, 2});

  auto core = CNN::Core<float>::makeCore(config);
  CHECK(core != nullptr, "core created");

  // Extract the original params from the core (they should match what we set)
  const auto& originalDense = core->getParameters().denseParams;

  auto data = saveAndReadBack(*core, "cnn_dense_exhaustive.bin");
  CHECK(!data.empty(), "params.bin written and read back");

  if (data.empty())
    return;

  CNN::CoreConfig<float> loadedConfig;
  loadedConfig.modeType = Common::ModeType::PREDICT;
  loadedConfig.deviceType = Common::DeviceType::CPU;
  loadedConfig.inputShape = config.inputShape;
  loadedConfig.progressReports = 0;
  loadedConfig.layersConfig = config.layersConfig;

  NN_CLI::ModelSerializer::loadCNNParametersBinary(data, loadedConfig, loadedConfig.layersConfig);

  int checks = verifyAllDenseParams(originalDense, loadedConfig.parameters.denseParams, "exhaustive");
  std::cout << "  (" << checks << " individual weight/bias checks)" << std::endl;
}

//
// TEST 2: Single dense layer — minimum case (N=1 → bug reads 1, should read 2).
//
static void testCNNSingleDenseLayer()
{
  TestScope _t("testCNNSingleDenseLayer");

  CNN::CoreConfig<float> config;
  config.modeType = Common::ModeType::PREDICT;
  config.deviceType = Common::DeviceType::CPU;
  config.inputShape = {1, 4, 4};
  config.progressReports = 0;

  config.layersConfig.cnnLayers.push_back(
    {CNN::LayerType::CONV, CNN::ConvLayerConfig{1, 3, 3, 1, 1, CNN::SlidingStrategyType::VALID}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::RELU, CNN::ReLULayerConfig{}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::FLATTEN, CNN::FlattenLayerConfig{}});

  config.layersConfig.denseLayers.push_back({2, ANN::ActvFuncType::SOFTMAX});

  // Conv output: 1x2x2 = 4 after flatten
  fillDenseWeights(config.parameters.denseParams, {4, 2});

  auto core = CNN::Core<float>::makeCore(config);
  CHECK(core != nullptr, "core created");

  const auto& originalDense = core->getParameters().denseParams;

  auto data = saveAndReadBack(*core, "cnn_dense_single.bin");
  CHECK(!data.empty(), "params.bin written and read back");

  if (data.empty())
    return;

  CNN::CoreConfig<float> loadedConfig;
  loadedConfig.modeType = Common::ModeType::PREDICT;
  loadedConfig.deviceType = Common::DeviceType::CPU;
  loadedConfig.inputShape = config.inputShape;
  loadedConfig.progressReports = 0;
  loadedConfig.layersConfig = config.layersConfig;

  NN_CLI::ModelSerializer::loadCNNParametersBinary(data, loadedConfig, loadedConfig.layersConfig);

  verifyAllDenseParams(originalDense, loadedConfig.parameters.denseParams, "single-dense");
}

//
// TEST 3: Three dense layers — larger architecture, ensures no off-by-one with more layers.
//
static void testCNNThreeDenseLayers()
{
  TestScope _t("testCNNThreeDenseLayers");

  CNN::CoreConfig<float> config;
  config.modeType = Common::ModeType::PREDICT;
  config.deviceType = Common::DeviceType::CPU;
  config.inputShape = {1, 6, 6};
  config.progressReports = 0;

  config.layersConfig.cnnLayers.push_back(
    {CNN::LayerType::CONV, CNN::ConvLayerConfig{2, 3, 3, 1, 1, CNN::SlidingStrategyType::VALID}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::INSTANCENORM, CNN::NormLayerConfig{}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::RELU, CNN::ReLULayerConfig{}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::FLATTEN, CNN::FlattenLayerConfig{}});

  config.layersConfig.denseLayers.push_back({8, ANN::ActvFuncType::RELU});
  config.layersConfig.denseLayers.push_back({4, ANN::ActvFuncType::RELU});
  config.layersConfig.denseLayers.push_back({2, ANN::ActvFuncType::SOFTMAX});

  // Conv output: 2x4x4 = 32 after flatten
  fillDenseWeights(config.parameters.denseParams, {32, 8, 4, 2});

  auto core = CNN::Core<float>::makeCore(config);
  CHECK(core != nullptr, "core created");

  const auto& originalDense = core->getParameters().denseParams;

  auto data = saveAndReadBack(*core, "cnn_dense_three.bin");
  CHECK(!data.empty(), "params.bin written and read back");

  if (data.empty())
    return;

  CNN::CoreConfig<float> loadedConfig;
  loadedConfig.modeType = Common::ModeType::PREDICT;
  loadedConfig.deviceType = Common::DeviceType::CPU;
  loadedConfig.inputShape = config.inputShape;
  loadedConfig.progressReports = 0;
  loadedConfig.layersConfig = config.layersConfig;

  NN_CLI::ModelSerializer::loadCNNParametersBinary(data, loadedConfig, loadedConfig.layersConfig);

  verifyAllDenseParams(originalDense, loadedConfig.parameters.denseParams, "three-dense");
}

//
// TEST 4: Predict consistency with multiple random inputs.
// Creates coreA (original) and coreB (from save→load round-trip),
// runs predictions on 5 different inputs, verifies ALL outputs match.
//
static void testCNNDensePredictMultipleInputs()
{
  TestScope _t("testCNNDensePredictMultipleInputs");

  CNN::CoreConfig<float> config;
  config.modeType = Common::ModeType::PREDICT;
  config.deviceType = Common::DeviceType::CPU;
  config.inputShape = {2, 5, 5};
  config.progressReports = 0;

  config.layersConfig.cnnLayers.push_back(
    {CNN::LayerType::CONV, CNN::ConvLayerConfig{3, 3, 3, 1, 1, CNN::SlidingStrategyType::SAME}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::INSTANCENORM, CNN::NormLayerConfig{}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::RELU, CNN::ReLULayerConfig{}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::FLATTEN, CNN::FlattenLayerConfig{}});

  config.layersConfig.denseLayers.push_back({6, ANN::ActvFuncType::RELU});
  config.layersConfig.denseLayers.push_back({2, ANN::ActvFuncType::SOFTMAX});

  // Conv output: 3x5x5 = 75 after flatten
  fillDenseWeights(config.parameters.denseParams, {75, 6, 2});

  auto coreA = CNN::Core<float>::makeCore(config);
  CHECK(coreA != nullptr, "coreA created");

  auto data = saveAndReadBack(*coreA, "cnn_dense_multi_input.bin");
  CHECK(!data.empty(), "params.bin written and read back");

  if (data.empty())
    return;

  CNN::CoreConfig<float> loadedConfig;
  loadedConfig.modeType = Common::ModeType::PREDICT;
  loadedConfig.deviceType = Common::DeviceType::CPU;
  loadedConfig.inputShape = config.inputShape;
  loadedConfig.progressReports = 0;
  loadedConfig.layersConfig = config.layersConfig;

  NN_CLI::ModelSerializer::loadCNNParametersBinary(data, loadedConfig, loadedConfig.layersConfig);

  auto coreB = CNN::Core<float>::makeCore(loadedConfig);
  CHECK(coreB != nullptr, "coreB created from loaded params");

  // Run 5 different inputs
  for (int testIdx = 0; testIdx < 5; ++testIdx) {
    CNN::Input<float> input(config.inputShape);
    input.data.resize(config.inputShape.size());
    // Different pattern for each test
    for (size_t i = 0; i < input.data.size(); ++i)
      input.data[i] = static_cast<float>(i) * 0.01f * static_cast<float>(testIdx + 1);

    auto resultA = coreA->predict(input);
    auto resultB = coreB->predict(input);

    CHECK(resultA.output.size() == 2, "output size = 2 for test " + std::to_string(testIdx));

    if (resultA.output.size() == 2 && resultB.output.size() == 2) {
      CHECK_NEAR(resultA.output[0], resultB.output[0], 1e-5f, "output[0] matches for input " + std::to_string(testIdx));
      CHECK_NEAR(resultA.output[1], resultB.output[1], 1e-5f, "output[1] matches for input " + std::to_string(testIdx));

      if (resultA.logits.size() == 2 && resultB.logits.size() == 2) {
        CHECK_NEAR(resultA.logits[0], resultB.logits[0], 1e-5f,
                   "logits[0] matches for input " + std::to_string(testIdx));
        CHECK_NEAR(resultA.logits[1], resultB.logits[1], 1e-5f,
                   "logits[1] matches for input " + std::to_string(testIdx));
      }
    }
  }

  std::cout << "  (5 inputs × 4 checks each = 20 prediction checks)" << std::endl;
}

//
// TEST 5: Verify the ANN sub-core's getParameters() returns the trained weights
// (not just the config struct). This checks that the weights actually made it
// into the running ANN core and weren't silently discarded by initializeParameters().
//
static void testCNNAnnCorePreservesWeights()
{
  TestScope _t("testCNNAnnCorePreservesWeights");

  CNN::CoreConfig<float> config;
  config.modeType = Common::ModeType::PREDICT;
  config.deviceType = Common::DeviceType::CPU;
  config.inputShape = {1, 4, 4};
  config.progressReports = 0;

  config.layersConfig.cnnLayers.push_back(
    {CNN::LayerType::CONV, CNN::ConvLayerConfig{1, 3, 3, 1, 1, CNN::SlidingStrategyType::VALID}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::RELU, CNN::ReLULayerConfig{}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::FLATTEN, CNN::FlattenLayerConfig{}});

  config.layersConfig.denseLayers.push_back({3, ANN::ActvFuncType::SIGMOID});
  config.layersConfig.denseLayers.push_back({2, ANN::ActvFuncType::SOFTMAX});

  // Conv output: 1x2x2 = 4 after flatten
  fillDenseWeights(config.parameters.denseParams, {4, 3, 2});

  // Build original core
  auto coreA = CNN::Core<float>::makeCore(config);
  CHECK(coreA != nullptr, "coreA created");

  const auto& originalDense = coreA->getParameters().denseParams;

  // Save → load → build coreB
  auto data = saveAndReadBack(*coreA, "cnn_dense_ann_core.bin");
  CHECK(!data.empty(), "params.bin written");

  if (data.empty())
    return;

  CNN::CoreConfig<float> loadedConfig;
  loadedConfig.modeType = Common::ModeType::PREDICT;
  loadedConfig.deviceType = Common::DeviceType::CPU;
  loadedConfig.inputShape = config.inputShape;
  loadedConfig.progressReports = 0;
  loadedConfig.layersConfig = config.layersConfig;

  NN_CLI::ModelSerializer::loadCNNParametersBinary(data, loadedConfig, loadedConfig.layersConfig);

  auto coreB = CNN::Core<float>::makeCore(loadedConfig);
  CHECK(coreB != nullptr, "coreB created from loaded params");

  // CRITICAL CHECK: coreB's getParameters() must return the SAME dense weights as coreA.
  // If initializeParameters() discarded them, the values would be random (seed 42).
  const auto& coreBDense = coreB->getParameters().denseParams;

  verifyAllDenseParams(originalDense, coreBDense, "ann-core-getParameters");
}

//
// TEST 6: Verify that conv/norm params are NOT corrupted by the dense fix.
// The fix changes how many dense blocks are read; we must ensure conv/norm/residual
// blocks before the dense section are still read correctly.
//
static void testCNNConvNormParamsPreserved()
{
  TestScope _t("testCNNConvNormParamsPreserved");

  CNN::CoreConfig<float> config;
  config.modeType = Common::ModeType::PREDICT;
  config.deviceType = Common::DeviceType::CPU;
  config.inputShape = {1, 6, 6};
  config.progressReports = 0;

  config.layersConfig.cnnLayers.push_back(
    {CNN::LayerType::CONV, CNN::ConvLayerConfig{2, 3, 3, 1, 1, CNN::SlidingStrategyType::SAME}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::INSTANCENORM, CNN::NormLayerConfig{}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::RELU, CNN::ReLULayerConfig{}});
  config.layersConfig.cnnLayers.push_back(
    {CNN::LayerType::CONV, CNN::ConvLayerConfig{3, 3, 3, 1, 1, CNN::SlidingStrategyType::SAME}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::INSTANCENORM, CNN::NormLayerConfig{}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::RELU, CNN::ReLULayerConfig{}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::FLATTEN, CNN::FlattenLayerConfig{}});

  config.layersConfig.denseLayers.push_back({2, ANN::ActvFuncType::SOFTMAX});

  // Conv output: 3x6x6 = 108 after flatten
  fillDenseWeights(config.parameters.denseParams, {108, 2});

  auto coreA = CNN::Core<float>::makeCore(config);
  CHECK(coreA != nullptr, "coreA created");

  const auto& origParams = coreA->getParameters();

  auto data = saveAndReadBack(*coreA, "cnn_dense_conv_norm.bin");
  CHECK(!data.empty(), "params.bin written");

  if (data.empty())
    return;

  CNN::CoreConfig<float> loadedConfig;
  loadedConfig.modeType = Common::ModeType::PREDICT;
  loadedConfig.deviceType = Common::DeviceType::CPU;
  loadedConfig.inputShape = config.inputShape;
  loadedConfig.progressReports = 0;
  loadedConfig.layersConfig = config.layersConfig;

  NN_CLI::ModelSerializer::loadCNNParametersBinary(data, loadedConfig, loadedConfig.layersConfig);

  auto coreB = CNN::Core<float>::makeCore(loadedConfig);
  CHECK(coreB != nullptr, "coreB created");

  const auto& loadedParams = coreB->getParameters();

  // Verify conv params
  CHECK(loadedParams.convParams.size() == origParams.convParams.size(), "conv params count matches");

  if (loadedParams.convParams.size() == origParams.convParams.size()) {
    for (size_t i = 0; i < origParams.convParams.size(); ++i) {
      CHECK(loadedParams.convParams[i].filters.size() == origParams.convParams[i].filters.size(),
            "conv[" + std::to_string(i) + "] filter count matches");
      CHECK(loadedParams.convParams[i].biases.size() == origParams.convParams[i].biases.size(),
            "conv[" + std::to_string(i) + "] bias count matches");

      // Spot-check a few filter values
      if (!origParams.convParams[i].filters.empty()) {
        CHECK_NEAR(loadedParams.convParams[i].filters[0], origParams.convParams[i].filters[0], 1e-6f,
                   "conv[" + std::to_string(i) + "] filter[0] preserved");
        CHECK_NEAR(loadedParams.convParams[i].filters.back(), origParams.convParams[i].filters.back(), 1e-6f,
                   "conv[" + std::to_string(i) + "] filter[last] preserved");
      }

      if (!origParams.convParams[i].biases.empty()) {
        CHECK_NEAR(loadedParams.convParams[i].biases[0], origParams.convParams[i].biases[0], 1e-6f,
                   "conv[" + std::to_string(i) + "] bias[0] preserved");
      }
    }
  }

  // Verify norm params (gamma, beta, runningMean, runningVar)
  CHECK(loadedParams.normParams.size() == origParams.normParams.size(), "norm params count matches");

  if (loadedParams.normParams.size() == origParams.normParams.size()) {
    for (size_t i = 0; i < origParams.normParams.size(); ++i) {
      if (!origParams.normParams[i].gamma.empty()) {
        CHECK_NEAR(loadedParams.normParams[i].gamma[0], origParams.normParams[i].gamma[0], 1e-6f,
                   "norm[" + std::to_string(i) + "] gamma[0] preserved");
        CHECK_NEAR(loadedParams.normParams[i].beta[0], origParams.normParams[i].beta[0], 1e-6f,
                   "norm[" + std::to_string(i) + "] beta[0] preserved");
      }
    }
  }

  // Verify dense params too
  verifyAllDenseParams(origParams.denseParams, loadedParams.denseParams, "conv-norm-dense");
}

//
// TEST 7: Full package round-trip (model.json + params.bin in a .nnmodel.tar).
// This exercises the complete save/load path used by NN-Server.
//
static void testCNNFullPackageRoundTrip()
{
  TestScope _t("testCNNFullPackageRoundTrip");

  CNN::CoreConfig<float> config;
  config.modeType = Common::ModeType::PREDICT;
  config.deviceType = Common::DeviceType::CPU;
  config.inputShape = {1, 4, 4};
  config.progressReports = 0;

  config.layersConfig.cnnLayers.push_back(
    {CNN::LayerType::CONV, CNN::ConvLayerConfig{2, 3, 3, 1, 1, CNN::SlidingStrategyType::VALID}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::INSTANCENORM, CNN::NormLayerConfig{}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::RELU, CNN::ReLULayerConfig{}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::FLATTEN, CNN::FlattenLayerConfig{}});

  config.layersConfig.denseLayers.push_back({4, ANN::ActvFuncType::RELU});
  config.layersConfig.denseLayers.push_back({2, ANN::ActvFuncType::SOFTMAX});

  // Conv output: 2x2x2 = 8 after flatten
  fillDenseWeights(config.parameters.denseParams, {8, 4, 2});

  auto coreA = CNN::Core<float>::makeCore(config);
  CHECK(coreA != nullptr, "coreA created");

  // Save to full package
  QString pkgPath = tempDir() + "/cnn_dense_fullpkg.nnmodel.tar";

  NN_CLI::IOConfig ioConfig;
  NN_CLI::AugmentationConfig augConfig;
  NN_CLI::ValidationMetadata valMeta;

  NN_CLI::ModelSerializer::saveCNNModelToPackage(pkgPath.toStdString(), *coreA, config, ioConfig, augConfig, valMeta);

  CHECK(QFile::exists(pkgPath), "package file created");

  // Read binary params from the package
  std::string jsonStr = NN_CLI::ModelPackage::readJsonFromPackage(pkgPath.toStdString());
  std::vector<char> binData = NN_CLI::ModelPackage::readBinaryFromPackage(pkgPath.toStdString());

  CHECK(!jsonStr.empty(), "model.json extracted from package");
  CHECK(!binData.empty(), "params.bin extracted from package");

  if (binData.empty())
    return;

  // Load params from the extracted binary
  CNN::CoreConfig<float> loadedConfig;
  loadedConfig.modeType = Common::ModeType::PREDICT;
  loadedConfig.deviceType = Common::DeviceType::CPU;
  loadedConfig.inputShape = config.inputShape;
  loadedConfig.progressReports = 0;
  loadedConfig.layersConfig = config.layersConfig;

  NN_CLI::ModelSerializer::loadCNNParametersBinary(binData, loadedConfig, loadedConfig.layersConfig);

  auto coreB = CNN::Core<float>::makeCore(loadedConfig);
  CHECK(coreB != nullptr, "coreB created from package-loaded params");

  // Verify predictions match
  CNN::Input<float> input(config.inputShape);
  input.data.resize(config.inputShape.size());

  for (size_t i = 0; i < input.data.size(); ++i)
    input.data[i] = static_cast<float>(i) * 0.07f;

  auto resultA = coreA->predict(input);
  auto resultB = coreB->predict(input);

  CHECK(resultA.output.size() == 2, "coreA output = 2");
  CHECK(resultB.output.size() == 2, "coreB output = 2");

  if (resultA.output.size() == 2 && resultB.output.size() == 2) {
    CHECK_NEAR(resultA.output[0], resultB.output[0], 1e-5f, "package round-trip output[0]");
    CHECK_NEAR(resultA.output[1], resultB.output[1], 1e-5f, "package round-trip output[1]");
    CHECK_NEAR(resultA.logits[0], resultB.logits[0], 1e-5f, "package round-trip logits[0]");
    CHECK_NEAR(resultA.logits[1], resultB.logits[1], 1e-5f, "package round-trip logits[1]");
  }

  // Also verify dense params from the package-loaded core
  const auto& origDense = coreA->getParameters().denseParams;
  const auto& loadedDense = coreB->getParameters().denseParams;
  verifyAllDenseParams(origDense, loadedDense, "fullpkg");
}

//===================================================================================================================//

void runCNNDenseRoundTripTests()
{
  testCNNDenseParamsExhaustive();
  testCNNSingleDenseLayer();
  testCNNThreeDenseLayers();
  testCNNDensePredictMultipleInputs();
  testCNNAnnCorePreservesWeights();
  testCNNConvNormParamsPreserved();
  testCNNFullPackageRoundTrip();
}
