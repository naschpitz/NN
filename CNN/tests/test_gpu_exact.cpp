#include "test_helpers.hpp"

//===================================================================================================================//

static void testGPUExactForwardBackwardCrossEntropy()
{
  TestScope _t("testGPUExactForwardBackwardCrossEntropy");

  // Same hand-computed network as CPU test, but on GPU with float.
  // 1x3x3 → Conv(1 filter 2x2, stride=1, valid) → ReLU → Flatten(4) → Dense(2, softmax)
  // Cross-entropy cost, SGD lr=1.0, 1 epoch, 1 sample
  CNN::CoreConfig<float> config;
  config.modeType = Common::ModeType::TRAIN;
  config.deviceType = Common::DeviceType::GPU;
  config.inputShape = {1, 3, 3};
  config.logLevel = Common::LogLevel::ERROR;
  config.numThreads = 1;
  config.numGPUs = 1;

  CNN::CNNLayerConfig convLayer;
  convLayer.type = CNN::LayerType::CONV;
  convLayer.config = CNN::ConvLayerConfig{1, 2, 2, 1, 1, CNN::SlidingStrategyType::VALID};

  CNN::CNNLayerConfig reluLayer;
  reluLayer.type = CNN::LayerType::RELU;
  reluLayer.config = CNN::ReLULayerConfig{};

  CNN::CNNLayerConfig flattenLayer;
  flattenLayer.type = CNN::LayerType::FLATTEN;
  flattenLayer.config = CNN::FlattenLayerConfig{};

  config.layersConfig.cnnLayers = {convLayer, reluLayer, flattenLayer};
  config.layersConfig.denseLayers = {{2, ANN::ActvFuncType::SOFTMAX}};

  // Preset conv parameters
  CNN::ConvParameters<float> initConv;
  initConv.numFilters = 1;
  initConv.inputC = 1;
  initConv.filterH = 2;
  initConv.filterW = 2;
  initConv.filters = {0.1f, -0.2f, 0.3f, -0.1f};
  initConv.biases = {0.0f};
  config.parameters.convParams = {initConv};

  // Preset  dense parameters
  ANN::Parameters<float> denseParams;
  denseParams.weights.resize(2);
  denseParams.biases.resize(2);
  denseParams.weights[0] = {};
  denseParams.biases[0] = {};
  denseParams.weights[1] = {{0.1f, -0.2f, 0.3f, -0.1f}, {0.2f, 0.1f, -0.3f, 0.2f}};
  denseParams.biases[1] = {0.0f, 0.0f};
  config.parameters.denseParams = denseParams;

  config.costFunctionConfig.type = Common::CostFunctionType::CROSS_ENTROPY;
  config.trainConfig.numEpochs = 1;
  config.trainConfig.learningRate = 1.0f;
  config.trainConfig.shuffleSamples = false;
  config.progressReports = 0;

  // Input: 1x3x3, target: [1, 0]
  CNN::Samples<float> samples(1);
  samples[0].input = CNN::Tensor3D<float>({1, 3, 3});
  samples[0].input.data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f};
  samples[0].output = {1.0f, 0.0f};

  auto core = CNN::Core<float>::makeCore(config);
  core->train(samples.size(), CNN::makeSampleProvider(samples));

  const CNN::Parameters<float>& p = core->getParameters();

  // Conv filter — GPU float values verified against CPU double (tolerance 1e-4 for float)
  CHECK_NEAR(p.convParams[0].filters[0], 0.11000499999958335f, 1e-4, "GPU exact conv filt[0]");
  CHECK_NEAR(p.convParams[0].filters[1], -0.19499750000020832f, 1e-4, "GPU exact conv filt[1]");
  CHECK_NEAR(p.convParams[0].filters[2], 0.2949975000002083f, 1e-4, "GPU exact conv filt[2]");
  CHECK_NEAR(p.convParams[0].filters[3], -0.11000499999958332f, 1e-4, "GPU exact conv filt[3]");
  CHECK_NEAR(p.convParams[0].biases[0], -0.050024999997916653f, 1e-4, "GPU exact conv bias");

  // Dense weights ( layer 1)
  CHECK_NEAR(p.denseParams.weights[1][0][0], 0.12000999999916667f, 1e-4, "GPU exact dw[0][0]");
  CHECK_NEAR(p.denseParams.weights[1][0][1], -0.17498750000104168f, 1e-4, "GPU exact dw[0][1]");
  CHECK_NEAR(p.denseParams.weights[1][0][2], 0.33501749999854163f, 1e-4, "GPU exact dw[0][2]");
  CHECK_NEAR(p.denseParams.weights[1][0][3], -0.059980000001666679f, 1e-4, "GPU exact dw[0][3]");
  CHECK_NEAR(p.denseParams.biases[1][0], 0.50024999997916675f, 1e-4, "GPU exact db[0]");

  CHECK_NEAR(p.denseParams.weights[1][1][0], 0.17999000000083334f, 1e-4, "GPU exact dw[1][0]");
  CHECK_NEAR(p.denseParams.weights[1][1][1], 0.074987500001041679f, 1e-4, "GPU exact dw[1][1]");
  CHECK_NEAR(p.denseParams.weights[1][1][2], -0.33501749999854163f, 1e-4, "GPU exact dw[1][2]");
  CHECK_NEAR(p.denseParams.weights[1][1][3], 0.15998000000166668f, 1e-4, "GPU exact dw[1][3]");
  CHECK_NEAR(p.denseParams.biases[1][1], -0.50024999997916664f, 1e-4, "GPU exact db[1]");
}

//===================================================================================================================//

static void testGPUExactForwardBackwardSquaredDifference()
{
  TestScope _t("testGPUExactForwardBackwardSquaredDifference");

  // 1x3x3 → Conv(1 filter 2x2, stride=1, valid) → ReLU → Flatten(4) → Dense(1, sigmoid)
  // Squared-difference cost, SGD lr=1.0, 1 epoch, 1 sample
  CNN::CoreConfig<float> config;
  config.modeType = Common::ModeType::TRAIN;
  config.deviceType = Common::DeviceType::GPU;
  config.inputShape = {1, 3, 3};
  config.logLevel = Common::LogLevel::ERROR;
  config.numThreads = 1;
  config.numGPUs = 1;

  CNN::CNNLayerConfig convLayer;
  convLayer.type = CNN::LayerType::CONV;
  convLayer.config = CNN::ConvLayerConfig{1, 2, 2, 1, 1, CNN::SlidingStrategyType::VALID};

  CNN::CNNLayerConfig reluLayer;
  reluLayer.type = CNN::LayerType::RELU;
  reluLayer.config = CNN::ReLULayerConfig{};

  CNN::CNNLayerConfig flattenLayer;
  flattenLayer.type = CNN::LayerType::FLATTEN;
  flattenLayer.config = CNN::FlattenLayerConfig{};

  config.layersConfig.cnnLayers = {convLayer, reluLayer, flattenLayer};
  config.layersConfig.denseLayers = {{1, ANN::ActvFuncType::SIGMOID}};

  CNN::ConvParameters<float> initConv;
  initConv.numFilters = 1;
  initConv.inputC = 1;
  initConv.filterH = 2;
  initConv.filterW = 2;
  initConv.filters = {0.1f, -0.2f, 0.3f, -0.1f};
  initConv.biases = {0.0f};
  config.parameters.convParams = {initConv};

  ANN::Parameters<float> denseParams;
  denseParams.weights.resize(2);
  denseParams.biases.resize(2);
  denseParams.weights[0] = {};
  denseParams.biases[0] = {};
  denseParams.weights[1] = {{0.1f, -0.2f, 0.3f, -0.1f}};
  denseParams.biases[1] = {0.0f};
  config.parameters.denseParams = denseParams;

  config.costFunctionConfig.type = Common::CostFunctionType::SQUARED_DIFFERENCE;
  config.trainConfig.numEpochs = 1;
  config.trainConfig.learningRate = 1.0f;
  config.trainConfig.shuffleSamples = false;
  config.progressReports = 0;

  CNN::Samples<float> samples(1);
  samples[0].input = CNN::Tensor3D<float>({1, 3, 3});
  samples[0].input.data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f};
  samples[0].output = {1.0f};

  auto core = CNN::Core<float>::makeCore(config);
  core->train(samples.size(), CNN::makeSampleProvider(samples));

  const CNN::Parameters<float>& p = core->getParameters();

  // Conv filter - every value verified against GPU float output (tolerance 1e-6)
  CHECK_NEAR(p.convParams[0].filters[0], 0.1099648774f, 1e-6, "GPU SD conv filt[0]");
  CHECK_NEAR(p.convParams[0].filters[1], -0.1875438988f, 1e-6, "GPU SD conv filt[1]");
  CHECK_NEAR(p.convParams[0].filters[2], 0.3174385428f, 1e-6, "GPU SD conv filt[2]");
  CHECK_NEAR(p.convParams[0].filters[3], -0.08007024229f, 1e-6, "GPU SD conv filt[3]");
  CHECK_NEAR(p.convParams[0].biases[0], 0.02491219342f, 1e-6, "GPU SD conv bias");

  // Dense weights - every value verified
  CHECK_NEAR(p.denseParams.weights[1][0][0], 0.1099648774f, 1e-6, "GPU SD dw[0][0]");
  CHECK_NEAR(p.denseParams.weights[1][0][1], -0.1875438988f, 1e-6, "GPU SD dw[0][1]");
  CHECK_NEAR(p.denseParams.weights[1][0][2], 0.3174385428f, 1e-6, "GPU SD dw[0][2]");
  CHECK_NEAR(p.denseParams.weights[1][0][3], -0.08007024229f, 1e-6, "GPU SD dw[0][3]");
  CHECK_NEAR(p.denseParams.biases[1][0], 0.2491219491f, 1e-6, "GPU SD db[0]");
}

//===================================================================================================================//

static void testGPUExactForwardBackwardWeightedCrossEntropy()
{
  TestScope _t("testGPUExactForwardBackwardWeightedCrossEntropy");

  // 1x3x3 → Conv(1 filter 2x2, stride=1, valid) → ReLU → Flatten(4) → Dense(2, softmax)
  // Weighted cross-entropy cost [3.0, 0.5], SGD lr=1.0, 1 epoch, 1 sample
  CNN::CoreConfig<float> config;
  config.modeType = Common::ModeType::TRAIN;
  config.deviceType = Common::DeviceType::GPU;
  config.inputShape = {1, 3, 3};
  config.logLevel = Common::LogLevel::ERROR;
  config.numThreads = 1;
  config.numGPUs = 1;

  CNN::CNNLayerConfig convLayer;
  convLayer.type = CNN::LayerType::CONV;
  convLayer.config = CNN::ConvLayerConfig{1, 2, 2, 1, 1, CNN::SlidingStrategyType::VALID};

  CNN::CNNLayerConfig reluLayer;
  reluLayer.type = CNN::LayerType::RELU;
  reluLayer.config = CNN::ReLULayerConfig{};

  CNN::CNNLayerConfig flattenLayer;
  flattenLayer.type = CNN::LayerType::FLATTEN;
  flattenLayer.config = CNN::FlattenLayerConfig{};

  config.layersConfig.cnnLayers = {convLayer, reluLayer, flattenLayer};
  config.layersConfig.denseLayers = {{2, ANN::ActvFuncType::SOFTMAX}};

  CNN::ConvParameters<float> initConv;
  initConv.numFilters = 1;
  initConv.inputC = 1;
  initConv.filterH = 2;
  initConv.filterW = 2;
  initConv.filters = {0.1f, -0.2f, 0.3f, -0.1f};
  initConv.biases = {0.0f};
  config.parameters.convParams = {initConv};

  ANN::Parameters<float> denseParams;
  denseParams.weights.resize(2);
  denseParams.biases.resize(2);
  denseParams.weights[0] = {};
  denseParams.biases[0] = {};
  denseParams.weights[1] = {{0.1f, -0.2f, 0.3f, -0.1f}, {0.2f, 0.1f, -0.3f, 0.2f}};
  denseParams.biases[1] = {0.0f, 0.0f};
  config.parameters.denseParams = denseParams;

  config.costFunctionConfig.type = Common::CostFunctionType::CROSS_ENTROPY;
  config.costFunctionConfig.weights = {3.0f, 0.5f};
  config.trainConfig.numEpochs = 1;
  config.trainConfig.learningRate = 1.0f;
  config.trainConfig.shuffleSamples = false;
  config.progressReports = 0;

  CNN::Samples<float> samples(1);
  samples[0].input = CNN::Tensor3D<float>({1, 3, 3});
  samples[0].input.data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f};
  samples[0].output = {1.0f, 0.0f};

  auto core = CNN::Core<float>::makeCore(config);
  core->train(samples.size(), CNN::makeSampleProvider(samples));

  const CNN::Parameters<float>& p = core->getParameters();

  // Conv filter - every value verified against GPU float output (tolerance 1e-6)
  CHECK_NEAR(p.convParams[0].filters[0], 0.1300149858f, 1e-6, "GPU WCE conv filt[0]");
  CHECK_NEAR(p.convParams[0].filters[1], -0.1849925071f, 1e-6, "GPU WCE conv filt[1]");
  CHECK_NEAR(p.convParams[0].filters[2], 0.2849925756f, 1e-6, "GPU WCE conv filt[2]");
  CHECK_NEAR(p.convParams[0].filters[3], -0.1300149858f, 1e-6, "GPU WCE conv filt[3]");
  CHECK_NEAR(p.convParams[0].biases[0], -0.1500749588f, 1e-6, "GPU WCE conv bias");

  // Dense weights - every value verified
  CHECK_NEAR(p.denseParams.weights[1][0][0], 0.1600300074f, 1e-6, "GPU WCE dw[0][0]");
  CHECK_NEAR(p.denseParams.weights[1][0][1], -0.1249625087f, 1e-6, "GPU WCE dw[0][1]");
  CHECK_NEAR(p.denseParams.weights[1][0][2], 0.4050525129f, 1e-6, "GPU WCE dw[0][2]");
  CHECK_NEAR(p.denseParams.weights[1][0][3], 0.0200600028f, 1e-6, "GPU WCE dw[0][3]");
  CHECK_NEAR(p.denseParams.biases[1][0], 1.500749946f, 1e-6, "GPU WCE db[0]");

  CHECK_NEAR(p.denseParams.weights[1][1][0], 0.1399700046f, 1e-6, "GPU WCE dw[1][0]");
  CHECK_NEAR(p.denseParams.weights[1][1][1], 0.02496250719f, 1e-6, "GPU WCE dw[1][1]");
  CHECK_NEAR(p.denseParams.weights[1][1][2], -0.4050525129f, 1e-6, "GPU WCE dw[1][2]");
  CHECK_NEAR(p.denseParams.weights[1][1][3], 0.07993999869f, 1e-6, "GPU WCE dw[1][3]");
  CHECK_NEAR(p.denseParams.biases[1][1], -1.500749946f, 1e-6, "GPU WCE db[1]");
}

//===================================================================================================================//

static void testGPULargeKDFiltersParityVsCPU()
{
  // Exercises gemm_dFilters_kpar (per-output cooperative K-reduction), the kernel
  // used when the tiled gemm_dFilters grid would underfill the device
  // (tiledGroups < computeUnits * GROUPS_PER_SM). Here M=2, N=9 -> 1 tiled group,
  // which underfills any GPU, forcing the kpar path regardless of K.
  // Tiled GEMM cannot fill the GPU for a small-output (M*N) x huge-K shape, so a
  // dedicated K-parallel kernel is used. Conv output is 100x100 = K=10000.
  // Hand-computing golden values at this K is impractical, so verify GPU vs CPU.
  // Network: 1x100x100 -> Conv(2,3x3,stride1,SAME) -> ReLU -> GAP -> Dense(2,softmax)
  TestScope _t("testGPULargeKDFiltersParityVsCPU (K-parallel dFilters, small tiled grid)");

  auto buildConfig = [](Common::DeviceType dev) {
    CNN::CoreConfig<float> config;
    config.modeType = Common::ModeType::TRAIN;
    config.deviceType = dev;
    config.inputShape = {1, 100, 100};
    config.logLevel = Common::LogLevel::ERROR;
    config.numThreads = 1;
    config.numGPUs = 1;

    CNN::CNNLayerConfig convLayer;
    convLayer.type = CNN::LayerType::CONV;
    convLayer.config = CNN::ConvLayerConfig{2, 3, 3, 1, 1, CNN::SlidingStrategyType::SAME};

    CNN::CNNLayerConfig reluLayer;
    reluLayer.type = CNN::LayerType::RELU;
    reluLayer.config = CNN::ReLULayerConfig{};

    CNN::CNNLayerConfig gapLayer;
    gapLayer.type = CNN::LayerType::GLOBALAVGPOOL;
    gapLayer.config = CNN::GlobalAvgPoolLayerConfig{};

    CNN::CNNLayerConfig flattenLayer;
    flattenLayer.type = CNN::LayerType::FLATTEN;
    flattenLayer.config = CNN::FlattenLayerConfig{};

    config.layersConfig.cnnLayers = {convLayer, reluLayer, gapLayer, flattenLayer};
    config.layersConfig.denseLayers = {{2, ANN::ActvFuncType::SOFTMAX}};

    CNN::ConvParameters<float> initConv;
    initConv.numFilters = 2;
    initConv.inputC = 1;
    initConv.filterH = 3;
    initConv.filterW = 3;
    initConv.filters = {0.1f, -0.2f, 0.3f,  -0.1f, 0.05f, 0.2f,  -0.15f, 0.1f,  -0.05f,
                        0.2f, 0.1f,  -0.3f, 0.15f, -0.1f, 0.25f, 0.05f,  -0.2f, 0.1f};
    initConv.biases = {0.0f, 0.0f};
    config.parameters.convParams = {initConv};

    ANN::Parameters<float> denseParams;
    denseParams.weights.resize(2);
    denseParams.biases.resize(2);
    denseParams.weights[0] = {};
    denseParams.biases[0] = {};
    denseParams.weights[1] = {{0.1f, -0.2f}, {0.2f, 0.1f}};
    denseParams.biases[1] = {0.0f, 0.0f};
    config.parameters.denseParams = denseParams;

    config.costFunctionConfig.type = Common::CostFunctionType::CROSS_ENTROPY;
    config.trainConfig.numEpochs = 1;
    config.trainConfig.learningRate = 0.1f;
    config.trainConfig.shuffleSamples = false;
    config.progressReports = 0;

    return config;
  };

  CNN::Samples<float> samples(1);
  samples[0].input = makeGradientInput<float>({1, 100, 100});
  samples[0].output = {1.0f, 0.0f};

  auto gpuCore = CNN::Core<float>::makeCore(buildConfig(Common::DeviceType::GPU));
  gpuCore->train(samples.size(), CNN::makeSampleProvider(samples));

  auto cpuCore = CNN::Core<float>::makeCore(buildConfig(Common::DeviceType::CPU));
  cpuCore->train(samples.size(), CNN::makeSampleProvider(samples));

  const CNN::Parameters<float>& gp = gpuCore->getParameters();
  const CNN::Parameters<float>& cp = cpuCore->getParameters();

  std::cout << "  GPU conv filt[0]=" << gp.convParams[0].filters[0]
            << "  CPU conv filt[0]=" << cp.convParams[0].filters[0] << std::endl;

  // Conv filters after one SGD step: new = init - lr*dFilter. GPU uses the
  // K-parallel tree-reduction kernel (K=10000); CPU uses a serial reference.
  // Tolerance 1e-3 absorbs the reduction-order difference and float round-off.
  for (size_t i = 0; i < gp.convParams[0].filters.size(); ++i) {
    CHECK_NEAR(gp.convParams[0].filters[i], cp.convParams[0].filters[i], 1e-3f, "GPU/CPU large-K conv filt");
  }

  for (size_t i = 0; i < gp.convParams[0].biases.size(); ++i) {
    CHECK_NEAR(gp.convParams[0].biases[i], cp.convParams[0].biases[i], 1e-3f, "GPU/CPU large-K conv bias");
  }
}

//===================================================================================================================//

void runGPUExactTests()
{
  if (!gpuAvailable()) {
    std::cout << "  (no GPU device available — skipping GPU exact tests)" << std::endl;
    return;
  }

  testGPUExactForwardBackwardCrossEntropy();
  testGPUExactForwardBackwardSquaredDifference();
  testGPUExactForwardBackwardWeightedCrossEntropy();
  testGPULargeKDFiltersParityVsCPU();
}
