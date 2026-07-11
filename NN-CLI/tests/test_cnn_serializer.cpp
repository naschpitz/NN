#include "test_helpers.hpp"

#include <QFile>

#include <CNN_Core.hpp>
#include <CNN_CoreConfig.hpp>
#include <CNN_LayersConfig.hpp>
#include <CNN_SlidingStrategy.hpp>
#include <ANN_ActvFunc.hpp>

#include "NN-CLI_ModelSerializer.hpp"

#include <cmath>
#include <string>

//===================================================================================================================//

//
// Tests for the CNN binary parameter serializer — every parameter type.
//
// Covers: conv (filters + biases), norm (gamma + beta + runningMean + runningVar),
// residual projections (weights + biases), dense (via test_cnn_dense_roundtrip.cpp).
// Also tests BatchNorm models and binary format robustness.
//

//--- Helpers (shared with test_cnn_dense_roundtrip.cpp pattern) --//

static std::vector<char> saveAndReadBackCNN(const CNN::Core<float>& core, const QString& filename)
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

static void fillConvParams(CNN::ConvParameters<float>& cp, ulong numFilters, ulong inputC, ulong filterH, ulong filterW)
{
  cp.numFilters = numFilters;
  cp.inputC = inputC;
  cp.filterH = filterH;
  cp.filterW = filterW;

  ulong filterSize = numFilters * inputC * filterH * filterW;
  cp.filters.resize(filterSize);
  cp.biases.resize(numFilters);

  for (ulong i = 0; i < filterSize; ++i)
    cp.filters[i] = static_cast<float>(i % 200) * 0.01f + 0.1f;

  for (ulong i = 0; i < numFilters; ++i)
    cp.biases[i] = static_cast<float>(i) * 0.1f;
}

static void fillNormParams(CNN::NormParameters<float>& np, ulong numChannels)
{
  np.numChannels = numChannels;
  np.gamma.resize(numChannels);
  np.beta.resize(numChannels);
  np.runningMean.resize(numChannels);
  np.runningVar.resize(numChannels);

  for (ulong i = 0; i < numChannels; ++i) {
    np.gamma[i] = 1.5f + static_cast<float>(i) * 0.1f;
    np.beta[i] = 0.3f + static_cast<float>(i) * 0.05f;
    np.runningMean[i] = 2.0f + static_cast<float>(i) * 0.2f;
    np.runningVar[i] = 0.8f + static_cast<float>(i) * 0.03f;
  }
}

static void fillResidualParams(CNN::ResidualParameters<float>& rp, ulong inC, ulong outC)
{
  rp.inC = inC;
  rp.outC = outC;

  ulong numWeights = outC * inC;
  rp.weights.resize(numWeights);
  rp.biases.resize(outC);

  for (ulong i = 0; i < numWeights; ++i)
    rp.weights[i] = static_cast<float>(i % 100) * 0.05f + 0.5f;

  for (ulong i = 0; i < outC; ++i)
    rp.biases[i] = static_cast<float>(i) * 0.2f;
}

static void fillDenseWeights(ANN::Parameters<float>& params, const std::vector<ulong>& layerSizes)
{
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

      for (ulong k = 0; k < numInputs; ++k)
        params.weights[i][j][k] = static_cast<float>(100 * i + 10 * j) + static_cast<float>(k) * 0.1f;
      params.biases[i][j] = static_cast<float>(1000 * i + j) + 0.5f;
    }
  }
}

//
// Verify all conv params (filters + biases) — every value.
//
static int verifyConvParams(const std::vector<CNN::ConvParameters<float>>& original,
                            const std::vector<CNN::ConvParameters<float>>& loaded, const std::string& label)
{
  int checks = 0;
  CHECK(loaded.size() == original.size(), label + ": conv layer count");
  checks++;

  if (loaded.size() != original.size())
    return checks;

  for (size_t i = 0; i < original.size(); ++i) {
    CHECK(loaded[i].filters.size() == original[i].filters.size(),
          label + ": conv[" + std::to_string(i) + "] filter size");
    checks++;

    if (loaded[i].filters.size() == original[i].filters.size()) {
      for (size_t j = 0; j < original[i].filters.size(); ++j) {
        CHECK_NEAR(loaded[i].filters[j], original[i].filters[j], 1e-6f,
                   label + ": conv[" + std::to_string(i) + "] filter[" + std::to_string(j) + "]");
        checks++;
      }
    }

    CHECK(loaded[i].biases.size() == original[i].biases.size(), label + ": conv[" + std::to_string(i) + "] bias size");
    checks++;

    if (loaded[i].biases.size() == original[i].biases.size()) {
      for (size_t j = 0; j < original[i].biases.size(); ++j) {
        CHECK_NEAR(loaded[i].biases[j], original[i].biases[j], 1e-6f,
                   label + ": conv[" + std::to_string(i) + "] bias[" + std::to_string(j) + "]");
        checks++;
      }
    }
  }

  return checks;
}

//
// Verify all norm params (gamma, beta, runningMean, runningVar) — every value.
//
static int verifyNormParams(const std::vector<CNN::NormParameters<float>>& original,
                            const std::vector<CNN::NormParameters<float>>& loaded, const std::string& label)
{
  int checks = 0;
  CHECK(loaded.size() == original.size(), label + ": norm layer count");
  checks++;

  if (loaded.size() != original.size())
    return checks;

  for (size_t i = 0; i < original.size(); ++i) {
    const std::string base = label + ": norm[" + std::to_string(i) + "]";

    for (size_t j = 0; j < original[i].gamma.size(); ++j) {
      CHECK_NEAR(loaded[i].gamma[j], original[i].gamma[j], 1e-6f, base + " gamma[" + std::to_string(j) + "]");
      checks++;
    }

    for (size_t j = 0; j < original[i].beta.size(); ++j) {
      CHECK_NEAR(loaded[i].beta[j], original[i].beta[j], 1e-6f, base + " beta[" + std::to_string(j) + "]");
      checks++;
    }

    // CRITICAL: verify runningMean and runningVar — previously never tested!
    for (size_t j = 0; j < original[i].runningMean.size(); ++j) {
      CHECK_NEAR(loaded[i].runningMean[j], original[i].runningMean[j], 1e-6f,
                 base + " runningMean[" + std::to_string(j) + "]");
      checks++;
    }

    for (size_t j = 0; j < original[i].runningVar.size(); ++j) {
      CHECK_NEAR(loaded[i].runningVar[j], original[i].runningVar[j], 1e-6f,
                 base + " runningVar[" + std::to_string(j) + "]");
      checks++;
    }
  }

  return checks;
}

//
// Verify all residual projection params (weights + biases) — every value.
//
static int verifyResidualParams(const std::vector<CNN::ResidualParameters<float>>& original,
                                const std::vector<CNN::ResidualParameters<float>>& loaded, const std::string& label)
{
  int checks = 0;
  CHECK(loaded.size() == original.size(), label + ": residual count");
  checks++;

  if (loaded.size() != original.size())
    return checks;

  for (size_t i = 0; i < original.size(); ++i) {
    const std::string base = label + ": res[" + std::to_string(i) + "]";

    CHECK(loaded[i].inC == original[i].inC, base + " inC");
    CHECK(loaded[i].outC == original[i].outC, base + " outC");
    checks += 2;

    for (size_t j = 0; j < original[i].weights.size(); ++j) {
      CHECK_NEAR(loaded[i].weights[j], original[i].weights[j], 1e-6f, base + " weights[" + std::to_string(j) + "]");
      checks++;
    }

    for (size_t j = 0; j < original[i].biases.size(); ++j) {
      CHECK_NEAR(loaded[i].biases[j], original[i].biases[j], 1e-6f, base + " biases[" + std::to_string(j) + "]");
      checks++;
    }
  }

  return checks;
}

//--- Tests --//

//
// Conv params exhaustive round-trip: multiple conv layers, all filter/bias values.
//
static void testCNNConvParamsExhaustive()
{
  TestScope _t("testCNNConvParamsExhaustive");

  CNN::CoreConfig<float> config;
  config.modeType = Common::ModeType::PREDICT;
  config.deviceType = Common::DeviceType::CPU;
  config.inputShape = {2, 6, 6};
  config.progressReports = 0;

  // Two conv layers with different sizes
  config.layersConfig.cnnLayers.push_back(
    {CNN::LayerType::CONV, CNN::ConvLayerConfig{3, 3, 3, 1, 1, CNN::SlidingStrategyType::SAME}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::RELU, CNN::ReLULayerConfig{}});
  config.layersConfig.cnnLayers.push_back(
    {CNN::LayerType::CONV, CNN::ConvLayerConfig{5, 3, 3, 1, 1, CNN::SlidingStrategyType::VALID}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::RELU, CNN::ReLULayerConfig{}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::FLATTEN, CNN::FlattenLayerConfig{}});

  config.layersConfig.denseLayers.push_back({2, ANN::ActvFuncType::SOFTMAX});

  // Set known conv params
  config.parameters.convParams.resize(2);
  fillConvParams(config.parameters.convParams[0], 3, 2, 3, 3);
  fillConvParams(config.parameters.convParams[1], 5, 3, 3, 3);

  // Conv1 output: 3x6x6=108, Conv2 output: 5x4x4=80 after flatten
  fillDenseWeights(config.parameters.denseParams, {80, 2});

  auto core = CNN::Core<float>::makeCore(config);
  CHECK(core != nullptr, "core created");

  const auto& origParams = core->getParameters();

  auto data = saveAndReadBackCNN(*core, "cnn_serializer_conv.bin");
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

  int checks = verifyConvParams(origParams.convParams, loadedConfig.parameters.convParams, "conv");
  std::cout << "  (" << checks << " individual checks)" << std::endl;
}

//
// InstanceNorm params exhaustive round-trip: gamma, beta, runningMean, runningVar.
// CRITICAL: runningMean and runningVar were NEVER verified in any previous test.
//
static void testCNNInstanceNormParamsExhaustive()
{
  TestScope _t("testCNNInstanceNormParamsExhaustive");

  CNN::CoreConfig<float> config;
  config.modeType = Common::ModeType::PREDICT;
  config.deviceType = Common::DeviceType::CPU;
  config.inputShape = {1, 5, 5};
  config.progressReports = 0;

  config.layersConfig.cnnLayers.push_back(
    {CNN::LayerType::CONV, CNN::ConvLayerConfig{4, 3, 3, 1, 1, CNN::SlidingStrategyType::SAME}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::INSTANCENORM, CNN::NormLayerConfig{}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::RELU, CNN::ReLULayerConfig{}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::FLATTEN, CNN::FlattenLayerConfig{}});

  config.layersConfig.denseLayers.push_back({2, ANN::ActvFuncType::SOFTMAX});

  config.parameters.convParams.resize(1);
  fillConvParams(config.parameters.convParams[0], 4, 1, 3, 3);

  config.parameters.normParams.resize(1);
  fillNormParams(config.parameters.normParams[0], 4);

  // Conv output: 4x5x5=100 after flatten
  fillDenseWeights(config.parameters.denseParams, {100, 2});

  auto core = CNN::Core<float>::makeCore(config);
  CHECK(core != nullptr, "core created");

  const auto& origParams = core->getParameters();

  auto data = saveAndReadBackCNN(*core, "cnn_serializer_inorm.bin");
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

  int nChecks = verifyNormParams(origParams.normParams, loadedConfig.parameters.normParams, "inorm");
  int cChecks = verifyConvParams(origParams.convParams, loadedConfig.parameters.convParams, "inorm-conv");
  std::cout << "  (" << nChecks << " norm + " << cChecks << " conv checks)" << std::endl;
}

//
// BatchNorm params exhaustive round-trip.
// BatchNorm uses running stats at inference (unlike InstanceNorm), so verifying
// these are loaded correctly is critical for model accuracy.
//
static void testCNNBatchNormParamsExhaustive()
{
  TestScope _t("testCNNBatchNormParamsExhaustive");

  CNN::CoreConfig<float> config;
  config.modeType = Common::ModeType::PREDICT;
  config.deviceType = Common::DeviceType::CPU;
  config.inputShape = {1, 5, 5};
  config.progressReports = 0;

  config.layersConfig.cnnLayers.push_back(
    {CNN::LayerType::CONV, CNN::ConvLayerConfig{4, 3, 3, 1, 1, CNN::SlidingStrategyType::SAME}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::BATCHNORM, CNN::NormLayerConfig{}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::RELU, CNN::ReLULayerConfig{}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::FLATTEN, CNN::FlattenLayerConfig{}});

  config.layersConfig.denseLayers.push_back({2, ANN::ActvFuncType::SOFTMAX});

  config.parameters.convParams.resize(1);
  fillConvParams(config.parameters.convParams[0], 4, 1, 3, 3);

  config.parameters.normParams.resize(1);
  fillNormParams(config.parameters.normParams[0], 4);

  fillDenseWeights(config.parameters.denseParams, {100, 2});

  auto core = CNN::Core<float>::makeCore(config);
  CHECK(core != nullptr, "core created");

  const auto& origParams = core->getParameters();

  auto data = saveAndReadBackCNN(*core, "cnn_serializer_bnorm.bin");
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

  int nChecks = verifyNormParams(origParams.normParams, loadedConfig.parameters.normParams, "bnorm");
  std::cout << "  (" << nChecks << " norm checks)" << std::endl;

  // Also verify predictions match (BatchNorm uses running stats at inference)
  auto coreB = CNN::Core<float>::makeCore(loadedConfig);
  CHECK(coreB != nullptr, "coreB created");

  CNN::Input<float> input(config.inputShape);
  input.data.resize(config.inputShape.size());

  for (size_t i = 0; i < input.data.size(); ++i)
    input.data[i] = static_cast<float>(i) * 0.01f;

  auto resultA = core->predict(input);
  auto resultB = coreB->predict(input);

  CHECK(resultA.output.size() == 2, "output size = 2");

  if (resultA.output.size() == 2 && resultB.output.size() == 2) {
    CHECK_NEAR(resultA.output[0], resultB.output[0], 1e-5f, "bnorm predict output[0]");
    CHECK_NEAR(resultA.output[1], resultB.output[1], 1e-5f, "bnorm predict output[1]");
  }
}

//
// Residual block with channel projection (4→8) round-trip.
// Verifies projection weights and biases survive serialization.
//
static void testCNNResidualProjectionRoundTrip()
{
  TestScope _t("testCNNResidualProjectionRoundTrip");

  CNN::CoreConfig<float> config;
  config.modeType = Common::ModeType::PREDICT;
  config.deviceType = Common::DeviceType::CPU;
  config.inputShape = {1, 6, 6};
  config.progressReports = 0;

  // Stem: 1→4 channels
  config.layersConfig.cnnLayers.push_back(
    {CNN::LayerType::CONV, CNN::ConvLayerConfig{4, 3, 3, 1, 1, CNN::SlidingStrategyType::SAME}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::RELU, CNN::ReLULayerConfig{}});

  // Residual block: 4→8 channels (projection needed)
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::RESIDUAL_START, CNN::ResidualStartConfig{}});
  config.layersConfig.cnnLayers.push_back(
    {CNN::LayerType::CONV, CNN::ConvLayerConfig{8, 3, 3, 1, 1, CNN::SlidingStrategyType::SAME}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::INSTANCENORM, CNN::NormLayerConfig{}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::RELU, CNN::ReLULayerConfig{}});
  config.layersConfig.cnnLayers.push_back(
    {CNN::LayerType::CONV, CNN::ConvLayerConfig{8, 3, 3, 1, 1, CNN::SlidingStrategyType::SAME}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::INSTANCENORM, CNN::NormLayerConfig{}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::RELU, CNN::ReLULayerConfig{}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::RESIDUAL_END, CNN::ResidualEndConfig{}});

  config.layersConfig.cnnLayers.push_back({CNN::LayerType::FLATTEN, CNN::FlattenLayerConfig{}});

  config.layersConfig.denseLayers.push_back({2, ANN::ActvFuncType::SOFTMAX});

  // Set known params for ALL types
  config.parameters.convParams.resize(3);
  fillConvParams(config.parameters.convParams[0], 4, 1, 3, 3); // stem
  fillConvParams(config.parameters.convParams[1], 8, 4, 3, 3); // res block conv 1
  fillConvParams(config.parameters.convParams[2], 8, 8, 3, 3); // res block conv 2

  config.parameters.normParams.resize(2);
  fillNormParams(config.parameters.normParams[0], 8); // res block norm 1
  fillNormParams(config.parameters.normParams[1], 8); // res block norm 2

  // Residual projection: 4→8 channels
  config.parameters.residualParams.resize(1);
  fillResidualParams(config.parameters.residualParams[0], 4, 8);

  // After residual block: 8x6x6 = 288 after flatten
  fillDenseWeights(config.parameters.denseParams, {288, 2});

  auto core = CNN::Core<float>::makeCore(config);
  CHECK(core != nullptr, "core created with residual projection");

  const auto& origParams = core->getParameters();

  auto data = saveAndReadBackCNN(*core, "cnn_serializer_residual.bin");
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

  int rChecks = verifyResidualParams(origParams.residualParams, loadedConfig.parameters.residualParams, "res");
  int cChecks = verifyConvParams(origParams.convParams, loadedConfig.parameters.convParams, "res-conv");
  int nChecks = verifyNormParams(origParams.normParams, loadedConfig.parameters.normParams, "res-norm");
  std::cout << "  (" << rChecks << " residual + " << cChecks << " conv + " << nChecks << " norm checks)" << std::endl;

  // Verify predictions match
  auto coreB = CNN::Core<float>::makeCore(loadedConfig);
  CHECK(coreB != nullptr, "coreB created");

  CNN::Input<float> input(config.inputShape);
  input.data.resize(config.inputShape.size());

  for (size_t i = 0; i < input.data.size(); ++i)
    input.data[i] = static_cast<float>(i) * 0.005f;

  auto resultA = core->predict(input);
  auto resultB = coreB->predict(input);

  CHECK(resultA.output.size() == 2, "output size = 2");

  if (resultA.output.size() == 2 && resultB.output.size() == 2) {
    CHECK_NEAR(resultA.output[0], resultB.output[0], 1e-5f, "residual predict output[0]");
    CHECK_NEAR(resultA.output[1], resultB.output[1], 1e-5f, "residual predict output[1]");
  }
}

//
// Combined model: conv + InstanceNorm + residual(projection) + BatchNorm + dense.
// All parameter types in one model — verify everything survives.
//
static void testCNNCombinedModelRoundTrip()
{
  TestScope _t("testCNNCombinedModelRoundTrip");

  CNN::CoreConfig<float> config;
  config.modeType = Common::ModeType::PREDICT;
  config.deviceType = Common::DeviceType::CPU;
  config.inputShape = {1, 6, 6};
  config.progressReports = 0;

  // Stem
  config.layersConfig.cnnLayers.push_back(
    {CNN::LayerType::CONV, CNN::ConvLayerConfig{4, 3, 3, 1, 1, CNN::SlidingStrategyType::SAME}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::INSTANCENORM, CNN::NormLayerConfig{}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::RELU, CNN::ReLULayerConfig{}});

  // Residual block (4→8, projection)
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::RESIDUAL_START, CNN::ResidualStartConfig{}});
  config.layersConfig.cnnLayers.push_back(
    {CNN::LayerType::CONV, CNN::ConvLayerConfig{8, 3, 3, 1, 1, CNN::SlidingStrategyType::SAME}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::BATCHNORM, CNN::NormLayerConfig{}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::RELU, CNN::ReLULayerConfig{}});
  config.layersConfig.cnnLayers.push_back(
    {CNN::LayerType::CONV, CNN::ConvLayerConfig{8, 3, 3, 1, 1, CNN::SlidingStrategyType::SAME}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::BATCHNORM, CNN::NormLayerConfig{}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::RELU, CNN::ReLULayerConfig{}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::RESIDUAL_END, CNN::ResidualEndConfig{}});

  config.layersConfig.cnnLayers.push_back({CNN::LayerType::FLATTEN, CNN::FlattenLayerConfig{}});

  config.layersConfig.denseLayers.push_back({4, ANN::ActvFuncType::RELU});
  config.layersConfig.denseLayers.push_back({2, ANN::ActvFuncType::SOFTMAX});

  // Set ALL params with known values
  config.parameters.convParams.resize(3);
  fillConvParams(config.parameters.convParams[0], 4, 1, 3, 3);
  fillConvParams(config.parameters.convParams[1], 8, 4, 3, 3);
  fillConvParams(config.parameters.convParams[2], 8, 8, 3, 3);

  config.parameters.normParams.resize(3);
  fillNormParams(config.parameters.normParams[0], 4); // stem InstanceNorm
  fillNormParams(config.parameters.normParams[1], 8); // res block BatchNorm 1
  fillNormParams(config.parameters.normParams[2], 8); // res block BatchNorm 2

  config.parameters.residualParams.resize(1);
  fillResidualParams(config.parameters.residualParams[0], 4, 8);

  // 8x6x6 = 288 after flatten
  fillDenseWeights(config.parameters.denseParams, {288, 4, 2});

  auto coreA = CNN::Core<float>::makeCore(config);
  CHECK(coreA != nullptr, "coreA created");

  const auto& origParams = coreA->getParameters();

  auto data = saveAndReadBackCNN(*coreA, "cnn_serializer_combined.bin");
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

  int total = 0;
  total += verifyConvParams(origParams.convParams, loadedParams.convParams, "combined-conv");
  total += verifyNormParams(origParams.normParams, loadedParams.normParams, "combined-norm");
  total += verifyResidualParams(origParams.residualParams, loadedParams.residualParams, "combined-res");

  // Predict consistency on 3 inputs
  for (int trial = 0; trial < 3; ++trial) {
    CNN::Input<float> input(config.inputShape);
    input.data.resize(config.inputShape.size());

    for (size_t i = 0; i < input.data.size(); ++i)
      input.data[i] = static_cast<float>(i) * 0.01f * static_cast<float>(trial + 1);

    auto resultA = coreA->predict(input);
    auto resultB = coreB->predict(input);

    if (resultA.output.size() == 2 && resultB.output.size() == 2) {
      CHECK_NEAR(resultA.output[0], resultB.output[0], 1e-5f, "combined output[0] trial " + std::to_string(trial));
      CHECK_NEAR(resultA.output[1], resultB.output[1], 1e-5f, "combined output[1] trial " + std::to_string(trial));
      total += 2;
    }
  }

  std::cout << "  (" << total << " total checks)" << std::endl;
}

//
// Binary format robustness: corrupted data must throw exceptions.
//
static void testCNNBinaryCorruptionThrows()
{
  TestScope _t("testCNNBinaryCorruptionThrows");

  // Build a valid binary
  CNN::CoreConfig<float> config;
  config.modeType = Common::ModeType::PREDICT;
  config.deviceType = Common::DeviceType::CPU;
  config.inputShape = {1, 4, 4};
  config.progressReports = 0;
  config.layersConfig.cnnLayers.push_back(
    {CNN::LayerType::CONV, CNN::ConvLayerConfig{2, 3, 3, 1, 1, CNN::SlidingStrategyType::VALID}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::RELU, CNN::ReLULayerConfig{}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::FLATTEN, CNN::FlattenLayerConfig{}});
  config.layersConfig.denseLayers.push_back({2, ANN::ActvFuncType::SOFTMAX});
  fillDenseWeights(config.parameters.denseParams, {2, 2});

  auto core = CNN::Core<float>::makeCore(config);
  auto validData = saveAndReadBackCNN(*core, "cnn_serializer_corrupt.bin");
  CHECK(!validData.empty(), "valid binary created");

  if (validData.empty())
    return;

  auto tryLoad = [&](std::vector<char>& data) -> bool {
    try {
      CNN::CoreConfig<float> lc;
      lc.inputShape = config.inputShape;
      lc.layersConfig = config.layersConfig;
      NN_CLI::ModelSerializer::loadCNNParametersBinary(data, lc, lc.layersConfig);
      return false; // did not throw
    } catch (...) {
      return true; // threw as expected
    }
  };

  // Truncated (too small for header)
  {
    auto truncated = std::vector<char>(validData.begin(), validData.begin() + 10);
    CHECK(tryLoad(truncated), "truncated data (10 bytes) throws");
  }

  // Wrong magic
  {
    auto bad = validData;
    bad[0] = bad[1] = bad[2] = bad[3] = 0;
    CHECK(tryLoad(bad), "wrong magic throws");
  }

  // Wrong version
  {
    auto bad = validData;
    bad[6] = 99;
    CHECK(tryLoad(bad), "wrong version throws");
  }

  // Wrong model type (ANN instead of CNN)
  {
    auto bad = validData;
    bad[8] = 0; // BINARY_MODEL_ANN instead of BINARY_MODEL_CNN
    CHECK(tryLoad(bad), "wrong model type (ANN instead of CNN) throws");
  }

  // Truncated mid-stream (cut off last dense layer data)
  {
    auto bad = std::vector<char>(validData.begin(), validData.begin() + validData.size() - 20);
    CHECK(tryLoad(bad), "truncated mid-stream throws");
  }
}

//===================================================================================================================//

void runCNNSerializerTests()
{
  testCNNConvParamsExhaustive();
  testCNNInstanceNormParamsExhaustive();
  testCNNBatchNormParamsExhaustive();
  testCNNResidualProjectionRoundTrip();
  testCNNCombinedModelRoundTrip();
  testCNNBinaryCorruptionThrows();
}
