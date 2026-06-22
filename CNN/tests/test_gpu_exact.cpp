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

static void testGPUResidualProjectionTrainsVsCPU()
{
  // Regression test for the residual-projection training bug: the 1x1
  // projection weights (residualParams) were never updated on GPU (no
  // accumulate/update/merge/syncFromGPU wiring). Trains a network with a
  // channel-changing residual block on GPU and CPU from identical preset
  // parameters and verifies the projection weights (a) move from init and
  // (b) match CPU. Network: 2x8x8 -> resStart -> Conv(4,SAME) -> ReLU ->
  // resEnd -> GAP -> Dense(2,softmax). The skip path 2->4 needs a projection,
  // so residualParams[0] (inC=2,outC=4). No normalization -> InstanceNorm fast
  // (per-sample) path. 4 samples, 1 batch -> multi-sample accumulate + /N update.
  TestScope _t("testGPUResidualProjectionTrainsVsCPU (residual 1x1 projection trains on GPU)");

  auto buildConfig = [](Common::DeviceType dev) {
    CNN::CoreConfig<float> config;
    config.modeType = Common::ModeType::TRAIN;
    config.deviceType = dev;
    config.inputShape = {2, 8, 8};
    config.logLevel = Common::LogLevel::ERROR;
    config.numThreads = 1;
    config.numGPUs = 1;

    CNN::CNNLayerConfig resStart;
    resStart.type = CNN::LayerType::RESIDUAL_START;
    resStart.config = CNN::ResidualStartConfig{};

    CNN::CNNLayerConfig convLayer;
    convLayer.type = CNN::LayerType::CONV;
    convLayer.config = CNN::ConvLayerConfig{4, 3, 3, 1, 1, CNN::SlidingStrategyType::SAME};

    CNN::CNNLayerConfig reluLayer;
    reluLayer.type = CNN::LayerType::RELU;
    reluLayer.config = CNN::ReLULayerConfig{};

    CNN::CNNLayerConfig resEnd;
    resEnd.type = CNN::LayerType::RESIDUAL_END;
    resEnd.config = CNN::ResidualEndConfig{};

    CNN::CNNLayerConfig gapLayer;
    gapLayer.type = CNN::LayerType::GLOBALAVGPOOL;
    gapLayer.config = CNN::GlobalAvgPoolLayerConfig{};

    CNN::CNNLayerConfig flattenLayer;
    flattenLayer.type = CNN::LayerType::FLATTEN;
    flattenLayer.config = CNN::FlattenLayerConfig{};

    config.layersConfig.cnnLayers = {resStart, convLayer, reluLayer, resEnd, gapLayer, flattenLayer};
    config.layersConfig.denseLayers = {{2, ANN::ActvFuncType::SOFTMAX}};

    // Conv 2->4 (3x3): 4*2*3*3 = 72 filters, 4 biases. Deterministic pattern.
    CNN::ConvParameters<float> initConv;
    initConv.numFilters = 4;
    initConv.inputC = 2;
    initConv.filterH = 3;
    initConv.filterW = 3;
    initConv.filters.resize(72);

    for (int i = 0; i < 72; ++i)
      initConv.filters[i] = static_cast<float>((i % 5) - 2) * 0.1f;
    initConv.biases = {0.0f, 0.0f, 0.0f, 0.0f};
    config.parameters.convParams = {initConv};

    // Residual projection 2->4 (1x1): outC*inC = 8 weights, 4 biases.
    CNN::ResidualParameters<float> initRes;
    initRes.inC = 2;
    initRes.outC = 4;
    initRes.weights = {0.1f, -0.1f, 0.05f, -0.05f, 0.2f, -0.2f, 0.1f, -0.1f};
    initRes.biases = {0.0f, 0.0f, 0.0f, 0.0f};
    config.parameters.residualParams = {initRes};

    // Dense 4->2.
    ANN::Parameters<float> denseParams;
    denseParams.weights.resize(2);
    denseParams.biases.resize(2);
    denseParams.weights[0] = {};
    denseParams.biases[0] = {};
    denseParams.weights[1] = {{0.1f, -0.2f, 0.1f, -0.1f}, {0.2f, 0.1f, -0.1f, 0.05f}};
    denseParams.biases[1] = {0.0f, 0.0f};
    config.parameters.denseParams = denseParams;

    config.costFunctionConfig.type = Common::CostFunctionType::CROSS_ENTROPY;
    config.trainConfig.numEpochs = 1;
    config.trainConfig.learningRate = 0.5f;
    config.trainConfig.batchSize = 4;
    config.trainConfig.shuffleSamples = false;
    config.trainConfig.optimizer.type = Common::OptimizerType::ADAM;
    config.progressReports = 0;

    return config;
  };

  const std::vector<float> initResW = {0.1f, -0.1f, 0.05f, -0.05f, 0.2f, -0.2f, 0.1f, -0.1f};

  CNN::Samples<float> samples(4);
  samples[0].input = makeGradientInput<float>({2, 8, 8});
  samples[0].output = {1.0f, 0.0f};
  samples[1].input = CNN::Tensor3D<float>({2, 8, 8}, 0.0f);
  samples[1].output = {0.0f, 1.0f};
  samples[2].input = makeGradientInput<float>({2, 8, 8});
  samples[2].output = {1.0f, 0.0f};
  samples[3].input = CNN::Tensor3D<float>({2, 8, 8}, 0.0f);
  samples[3].output = {0.0f, 1.0f};

  // Diagnostic: forward prediction parity (init params, before training). If the GPU
  // forward differs from CPU here, a projection-forward bug drives the gradient mismatch.
  {
    auto gpuPredCore = CNN::Core<float>::makeCore(buildConfig(Common::DeviceType::GPU));
    auto cpuPredCore = CNN::Core<float>::makeCore(buildConfig(Common::DeviceType::CPU));
    CNN::Output<float> gpuPred = gpuPredCore->predict(samples[0].input).output;
    CNN::Output<float> cpuPred = cpuPredCore->predict(samples[0].input).output;

    std::cout << "  FWD GPU pred={";

    for (size_t i = 0; i < gpuPred.size(); ++i)
      std::cout << gpuPred[i] << " ";

    std::cout << "} CPU pred={";

    for (size_t i = 0; i < cpuPred.size(); ++i)
      std::cout << cpuPred[i] << " ";

    std::cout << "}" << std::endl;

    for (size_t i = 0; i < gpuPred.size(); ++i)
      CHECK_NEAR(gpuPred[i], cpuPred[i], 1e-3f, "GPU/CPU forward prediction parity (init params)");
  }

  auto gpuCore = CNN::Core<float>::makeCore(buildConfig(Common::DeviceType::GPU));
  gpuCore->train(samples.size(), CNN::makeSampleProvider(samples));

  auto cpuCore = CNN::Core<float>::makeCore(buildConfig(Common::DeviceType::CPU));
  cpuCore->train(samples.size(), CNN::makeSampleProvider(samples));

  const CNN::Parameters<float>& gp = gpuCore->getParameters();
  const CNN::Parameters<float>& cp = cpuCore->getParameters();

  CHECK(!gp.residualParams.empty(), "GPU residual projection params populated");
  CHECK(!cp.residualParams.empty(), "CPU residual projection params populated");

  std::cout << "  GPU res proj w[0]=" << gp.residualParams[0].weights[0]
            << "  CPU res proj w[0]=" << cp.residualParams[0].weights[0] << "  init w[0]=" << initResW[0] << std::endl;

  // Diagnostic: conv filter parity after training. If conv matches CPU but residual
  // does not, the bug is residual-specific (not the shared backward path).
  if (!gp.convParams.empty() && !cp.convParams.empty()) {
    std::cout << "  GPU conv filt[0]=" << gp.convParams[0].filters[0]
              << "  CPU conv filt[0]=" << cp.convParams[0].filters[0] << std::endl;

    for (size_t i = 0; i < gp.convParams[0].filters.size(); ++i)
      CHECK_NEAR(gp.convParams[0].filters[i], cp.convParams[0].filters[i], 1e-3f, "GPU/CPU conv filter");
  }

  // (a) Projections must have trained: at least one weight moved from init.
  float gpuMaxDrift = 0.0f;
  float cpuMaxDrift = 0.0f;

  for (size_t i = 0; i < initResW.size(); ++i) {
    gpuMaxDrift = std::max(gpuMaxDrift, std::fabs(gp.residualParams[0].weights[i] - initResW[i]));
    cpuMaxDrift = std::max(cpuMaxDrift, std::fabs(cp.residualParams[0].weights[i] - initResW[i]));
  }

  CHECK(gpuMaxDrift > 1e-4f, "GPU residual projection trained (weights moved from init)");
  CHECK(cpuMaxDrift > 1e-4f, "CPU residual projection trained (weights moved from init)");

  // (b) GPU matches CPU within tolerance. 1e-3 absorbs reduction-order and Adam
  // round-off differences between the GPU kernel and CPU reference.
  for (size_t i = 0; i < gp.residualParams[0].weights.size(); ++i) {
    ulong oc = i / gp.residualParams[0].inC;
    ulong ic = i % gp.residualParams[0].inC;
    std::cout << "    w[" << i << "] (oc=" << oc << ",ic=" << ic << ") GPU=" << gp.residualParams[0].weights[i]
              << " CPU=" << cp.residualParams[0].weights[i] << " init=" << initResW[i] << std::endl;
    CHECK_NEAR(gp.residualParams[0].weights[i], cp.residualParams[0].weights[i], 1e-3f,
               "GPU/CPU residual projection weight");
  }

  for (size_t i = 0; i < gp.residualParams[0].biases.size(); ++i) {
    std::cout << "    b[" << i << "] GPU=" << gp.residualParams[0].biases[i]
              << " CPU=" << cp.residualParams[0].biases[i] << std::endl;
    CHECK_NEAR(gp.residualParams[0].biases[i], cp.residualParams[0].biases[i], 1e-3f,
               "GPU/CPU residual projection bias");
  }
}

//===================================================================================================================//

static void testGPUSetLearningRateParityVsCPU()
{
  // GPU parity test for setLearningRate: change the learning rate mid-training
  // (between two 1-epoch train() calls) and verify GPU matches CPU. If
  // setLearningRate failed to propagate to the GPU kernels, the GPU would keep
  // using the old LR and the parameters would diverge from CPU.
  // Network: 1x3x3 -> Conv(1,2x2,valid) -> ReLU -> Flatten(4) -> Dense(2,softmax)
  // Cross-entropy, SGD, 1 sample. LR 1.0 -> 0.25 between epochs.
  TestScope _t("testGPUSetLearningRateParityVsCPU (LR change mid-training)");

  auto buildConfig = [](Common::DeviceType dev) {
    CNN::CoreConfig<float> config;
    config.modeType = Common::ModeType::TRAIN;
    config.deviceType = dev;
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
    config.trainConfig.numEpochs = 1;
    config.trainConfig.learningRate = 1.0f;
    config.trainConfig.optimizer.type = Common::OptimizerType::SGD;
    config.trainConfig.shuffleSamples = false;
    config.progressReports = 0;

    return config;
  };

  CNN::Samples<float> samples(1);
  samples[0].input = CNN::Tensor3D<float>({1, 3, 3});
  samples[0].input.data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f};
  samples[0].output = {1.0f, 0.0f};

  auto gpuCore = CNN::Core<float>::makeCore(buildConfig(Common::DeviceType::GPU));
  auto cpuCore = CNN::Core<float>::makeCore(buildConfig(Common::DeviceType::CPU));

  // Epoch 0 at LR=1.0
  gpuCore->train(samples.size(), CNN::makeSampleProvider(samples));
  cpuCore->train(samples.size(), CNN::makeSampleProvider(samples));

  // Snapshot conv filter[0] after epoch 0 for the "params moved" sanity check.
  const float gpuEpoch0Filt0 = gpuCore->getParameters().convParams[0].filters[0];
  const float cpuEpoch0Filt0 = cpuCore->getParameters().convParams[0].filters[0];

  // Change LR mid-training (mirrors what the LR scheduler does at epoch boundaries).
  gpuCore->setLearningRate(0.25f);
  cpuCore->setLearningRate(0.25f);

  // Epoch 1 at LR=0.25
  gpuCore->train(samples.size(), CNN::makeSampleProvider(samples));
  cpuCore->train(samples.size(), CNN::makeSampleProvider(samples));

  const CNN::Parameters<float>& gp = gpuCore->getParameters();
  const CNN::Parameters<float>& cp = cpuCore->getParameters();

  std::cout << "  post-LR-change GPU conv filt[0]=" << gp.convParams[0].filters[0]
            << "  CPU conv filt[0]=" << cp.convParams[0].filters[0] << std::endl;

  // Conv: GPU must match CPU after the LR change (1e-4 tolerance for float GPU round-off).
  for (size_t i = 0; i < gp.convParams[0].filters.size(); ++i)
    CHECK_NEAR(gp.convParams[0].filters[i], cp.convParams[0].filters[i], 1e-4f, "GPU/CPU conv filt after LR change");

  for (size_t i = 0; i < gp.convParams[0].biases.size(); ++i)
    CHECK_NEAR(gp.convParams[0].biases[i], cp.convParams[0].biases[i], 1e-4f, "GPU/CPU conv bias after LR change");

  // Dense: GPU must match CPU.
  for (size_t i = 0; i < gp.denseParams.weights[1].size(); ++i)

    for (size_t j = 0; j < gp.denseParams.weights[1][i].size(); ++j)
      CHECK_NEAR(gp.denseParams.weights[1][i][j], cp.denseParams.weights[1][i][j], 1e-4f,
                 "GPU/CPU dense weight after LR change");

  for (size_t i = 0; i < gp.denseParams.biases[1].size(); ++i)
    CHECK_NEAR(gp.denseParams.biases[1][i], cp.denseParams.biases[1][i], 1e-4f, "GPU/CPU dense bias after LR change");

  // Sanity: params moved between epoch 0 and epoch 1 (training continued at the new LR).
  CHECK(std::fabs(gp.convParams[0].filters[0] - gpuEpoch0Filt0) > 1e-5f, "GPU conv filt moved after LR change");
  CHECK(std::fabs(cp.convParams[0].filters[0] - cpuEpoch0Filt0) > 1e-5f, "CPU conv filt moved after LR change");
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
  testGPUResidualProjectionTrainsVsCPU();
  testGPUSetLearningRateParityVsCPU();
}
