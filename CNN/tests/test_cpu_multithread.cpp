#include "test_helpers.hpp"

#include "CNN_Core.hpp"
#include "CNN_CoreConfig.hpp"
#include "CNN_LayersConfig.hpp"
#include "CNN_SlidingStrategy.hpp"
#include "ANN_ActvFunc.hpp"

#include <cmath>

//===================================================================================================================//

//
// Multi-threaded training correctness test.
//
// Verifies that training with N>1 threads produces the same dense-layer
// parameters as training with 1 thread. Before the Adam fix (ccfc53c),
// multi-threaded Adam produced silently different results because each
// worker ran Adam independently and parameters were averaged (nonlinear
// operation). Now gradients are merged first.
//
// Note: CNN parameters (conv/norm/residual) were always correctly merged
// globally. Only dense layers had the per-worker-averaging bug.
//

static CNN::CoreConfig<double> makeMultiThreadTestConfig(int numThreads, bool useAdam)
{
  CNN::CoreConfig<double> config;
  config.modeType = Common::ModeType::TRAIN;
  config.deviceType = Common::DeviceType::CPU;
  config.inputShape = {1, 5, 5};
  config.logLevel = Common::LogLevel::ERROR;
  config.numThreads = numThreads;

  CNN::CNNLayerConfig convLayer;
  convLayer.type = CNN::LayerType::CONV;
  convLayer.config = CNN::ConvLayerConfig{2, 3, 3, 1, 1, CNN::SlidingStrategyType::VALID};

  CNN::CNNLayerConfig reluLayer;
  reluLayer.type = CNN::LayerType::RELU;
  reluLayer.config = CNN::ReLULayerConfig{};

  CNN::CNNLayerConfig flattenLayer;
  flattenLayer.type = CNN::LayerType::FLATTEN;
  flattenLayer.config = CNN::FlattenLayerConfig{};

  config.layersConfig.cnnLayers = {convLayer, reluLayer, flattenLayer};
  config.layersConfig.denseLayers = {{3, ANN::ActvFuncType::RELU}, {2, ANN::ActvFuncType::SOFTMAX}};

  // Preset conv params
  CNN::ConvParameters<double> initConv;
  initConv.numFilters = 2;
  initConv.inputC = 1;
  initConv.filterH = 3;
  initConv.filterW = 3;
  initConv.filters.resize(18);

  for (int i = 0; i < 18; ++i)
    initConv.filters[i] = static_cast<double>((i % 5) - 2) * 0.1;

  initConv.biases = {0.0, 0.0};
  config.parameters.convParams = {initConv};

  // Preset dense params: flatten = 2*3*3 = 18 → 3 → 2
  ANN::Parameters<double> denseParams;
  denseParams.weights.resize(3);
  denseParams.biases.resize(3);
  denseParams.weights[0] = {};
  denseParams.biases[0] = {};

  denseParams.weights[1].resize(3);
  denseParams.biases[1].resize(3);

  for (int j = 0; j < 3; ++j) {
    denseParams.weights[1][j].resize(18);

    for (int k = 0; k < 18; ++k)
      denseParams.weights[1][j][k] = static_cast<double>((j * 18 + k) % 7) * 0.05;

    denseParams.biases[1][j] = 0.0;
  }

  denseParams.weights[2].resize(2);
  denseParams.biases[2].resize(2);

  for (int j = 0; j < 2; ++j) {
    denseParams.weights[2][j].resize(3);

    for (int k = 0; k < 3; ++k)
      denseParams.weights[2][j][k] = static_cast<double>((j * 3 + k) % 5) * 0.1;

    denseParams.biases[2][j] = 0.0;
  }

  config.parameters.denseParams = denseParams;

  config.costFunctionConfig.type = Common::CostFunctionType::CROSS_ENTROPY;
  config.trainConfig.numEpochs = 3;
  config.trainConfig.learningRate = 0.1;
  config.trainConfig.batchSize = 4;
  config.trainConfig.shuffleSamples = false;

  if (useAdam)
    config.trainConfig.optimizer.type = Common::OptimizerType::ADAM;

  config.progressReports = 0;

  return config;
}

static CNN::Samples<double> makeMultiThreadTestSamples()
{
  CNN::Samples<double> samples(4);

  for (int s = 0; s < 4; ++s) {
    samples[s].input = CNN::Tensor3D<double>({1, 5, 5});

    for (ulong i = 0; i < 25; ++i)
      samples[s].input.data[i] = static_cast<double>((i + s * 7) % 10) * 0.1;

    samples[s].output = {(s % 2 == 0) ? 1.0 : 0.0, (s % 2 == 0) ? 0.0 : 1.0};
  }

  return samples;
}

//
// Train with 1 thread vs 4 threads using SGD, verify identical dense params.
//
static void testMultiThreadSGDConsistency()
{
  TestScope _t("testMultiThreadSGDConsistency");

  auto samples = makeMultiThreadTestSamples();

  auto core1 = CNN::Core<double>::makeCore(makeMultiThreadTestConfig(1, false));
  auto core4 = CNN::Core<double>::makeCore(makeMultiThreadTestConfig(4, false));

  core1->train(samples.size(), CNN::makeSampleProvider(samples));
  core4->train(samples.size(), CNN::makeSampleProvider(samples));

  const auto& p1 = core1->getParameters();
  const auto& p4 = core4->getParameters();

  // Conv params should match (always correctly merged)
  for (size_t i = 0; i < p1.convParams[0].filters.size(); ++i)
    CHECK_NEAR(p1.convParams[0].filters[i], p4.convParams[0].filters[i], 1e-6,
               "SGD conv filt[" + std::to_string(i) + "]");

  // Dense params should match
  for (size_t l = 1; l < p1.denseParams.weights.size(); ++l)

    for (size_t j = 0; j < p1.denseParams.weights[l].size(); ++j)

      for (size_t k = 0; k < p1.denseParams.weights[l][j].size(); ++k)
        CHECK_NEAR(p1.denseParams.weights[l][j][k], p4.denseParams.weights[l][j][k], 1e-6,
                   "SGD dense w[" + std::to_string(l) + "][" + std::to_string(j) + "][" + std::to_string(k) + "]");
}

//
// Train with 1 thread vs 4 threads using Adam, verify identical dense params.
// This is the test that would FAIL without the Adam gradient-merging fix.
//
static void testMultiThreadAdamConsistency()
{
  TestScope _t("testMultiThreadAdamConsistency");

  auto samples = makeMultiThreadTestSamples();

  auto core1 = CNN::Core<double>::makeCore(makeMultiThreadTestConfig(1, true));
  auto core4 = CNN::Core<double>::makeCore(makeMultiThreadTestConfig(4, true));

  core1->train(samples.size(), CNN::makeSampleProvider(samples));
  core4->train(samples.size(), CNN::makeSampleProvider(samples));

  const auto& p1 = core1->getParameters();
  const auto& p4 = core4->getParameters();

  // Conv params should match (always correctly merged)
  for (size_t i = 0; i < p1.convParams[0].filters.size(); ++i)
    CHECK_NEAR(p1.convParams[0].filters[i], p4.convParams[0].filters[i], 1e-10,
               "Adam conv filt[" + std::to_string(i) + "]");

  // Dense params should match — THIS IS THE KEY CHECK
  // Without the Adam fix, each worker ran Adam independently and params were
  // averaged (nonlinear), producing different results than single-thread.
  for (size_t l = 1; l < p1.denseParams.weights.size(); ++l) {
    for (size_t j = 0; j < p1.denseParams.weights[l].size(); ++j) {
      for (size_t k = 0; k < p1.denseParams.weights[l][j].size(); ++k) {
        CHECK_NEAR(p1.denseParams.weights[l][j][k], p4.denseParams.weights[l][j][k], 1e-6,
                   "Adam dense w[" + std::to_string(l) + "][" + std::to_string(j) + "][" + std::to_string(k) + "]");
      }

      CHECK_NEAR(p1.denseParams.biases[l][j], p4.denseParams.biases[l][j], 1e-6,
                 "Adam dense b[" + std::to_string(l) + "][" + std::to_string(j) + "]");
    }
  }
}

//===================================================================================================================//

void runMultiThreadTests()
{
  testMultiThreadSGDConsistency();
  testMultiThreadAdamConsistency();
}
