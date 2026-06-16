#include "test_helpers.hpp"

//===================================================================================================================//

static void runPredictStopRequested()
{
  TestScope _t("runPredictStopRequested (abort mid-flight)");

  // 1x5x5 → Conv(1 filter 3x3 valid) → ReLU → Flatten(9) → Dense(1, sigmoid)
  CNN::CoreConfig<double> config;
  config.modeType = Common::ModeType::PREDICT;
  config.deviceType = Common::DeviceType::CPU;
  config.inputShape = {1, 5, 5};
  config.logLevel = Common::LogLevel::ERROR;

  CNN::CNNLayerConfig convLayer;
  convLayer.type = CNN::LayerType::CONV;
  convLayer.config = CNN::ConvLayerConfig{1, 3, 3, 1, 1, CNN::SlidingStrategyType::VALID};

  CNN::CNNLayerConfig reluLayer;
  reluLayer.type = CNN::LayerType::RELU;
  reluLayer.config = CNN::ReLULayerConfig{};

  CNN::CNNLayerConfig flattenLayer;
  flattenLayer.type = CNN::LayerType::FLATTEN;
  flattenLayer.config = CNN::FlattenLayerConfig{};

  config.layersConfig.cnnLayers = {convLayer, reluLayer, flattenLayer};
  config.layersConfig.denseLayers = {{1, ANN::ActvFuncType::SIGMOID}};

  // Pre-init conv params to avoid dead ReLU from random init.
  CNN::ConvParameters<double> initConv;
  initConv.numFilters = 1;
  initConv.inputC = 1;
  initConv.filterH = 3;
  initConv.filterW = 3;
  initConv.filters.assign(9, 0.1);
  initConv.biases.assign(1, 0.0);
  config.parameters.convParams = {initConv};

  config.trainConfig.numEpochs = 0; // No training — just inference
  config.trainConfig.learningRate = 0.5f;
  config.trainConfig.shuffleSeed = 42;
  config.progressReports = 0;

  // Build a large input set (200 samples → 200 batches with batchSize=1).
  const ulong total = 200;
  CNN::Inputs<double> inputs(total);

  for (ulong i = 0; i < total; i++)
    inputs[i] = makeGradientInput<double>({1, 5, 5});

  auto core = CNN::Core<double>::makeCore(config);

  // Slice provider yielding 1 input per batch.
  auto sliceProvider = [&inputs](ulong batchSize, ulong batchIndex) {
    ulong start = batchIndex * batchSize;
    ulong end = std::min(start + batchSize, static_cast<ulong>(inputs.size()));

    if (start >= end)
      return CNN::Inputs<double>{};
    return CNN::Inputs<double>(inputs.begin() + start, inputs.begin() + end);
  };

  // Progress callback that requests stop after the first batch completes.
  core->setProgressCallback([&core](ulong current, ulong) {
    if (current >= 1)
      core->requestStop();
  });

  Common::PredictResults<double> results = core->predict(total, sliceProvider);

  // At least one batch must have completed before the stop took effect.
  CHECK(results.size() >= 1, "at least one batch completed before abort");
  // But not all 200 — the stop abort should have produced partial results.
  CHECK(results.size() < total, "abort produced partial results (not all 200)");
}

//===================================================================================================================//

static void testTrainStopRequested()
{
  TestScope _t("testTrainStopRequested (abort mid-training)");

  // 1x5x5 → Conv(1 filter 3x3 valid) → ReLU → Flatten(9) → Dense(1, sigmoid)
  CNN::CoreConfig<double> config;
  config.modeType = Common::ModeType::TRAIN;
  config.deviceType = Common::DeviceType::CPU;
  config.inputShape = {1, 5, 5};
  config.logLevel = Common::LogLevel::ERROR;

  CNN::CNNLayerConfig convLayer;
  convLayer.type = CNN::LayerType::CONV;
  convLayer.config = CNN::ConvLayerConfig{1, 3, 3, 1, 1, CNN::SlidingStrategyType::VALID};

  CNN::CNNLayerConfig reluLayer;
  reluLayer.type = CNN::LayerType::RELU;
  reluLayer.config = CNN::ReLULayerConfig{};

  CNN::CNNLayerConfig flattenLayer;
  flattenLayer.type = CNN::LayerType::FLATTEN;
  flattenLayer.config = CNN::FlattenLayerConfig{};

  config.layersConfig.cnnLayers = {convLayer, reluLayer, flattenLayer};
  config.layersConfig.denseLayers = {{1, ANN::ActvFuncType::SIGMOID}};

  // Pre-init conv params to avoid dead ReLU from random init.
  CNN::ConvParameters<double> initConv;
  initConv.numFilters = 1;
  initConv.inputC = 1;
  initConv.filterH = 3;
  initConv.filterW = 3;
  initConv.filters.assign(9, 0.1);
  initConv.biases.assign(1, 0.0);
  config.parameters.convParams = {initConv};

  config.trainConfig.numEpochs = 100;
  config.trainConfig.learningRate = 0.5f;
  config.trainConfig.shuffleSeed = 42;
  config.progressReports = 0;

  // "bright" (gradient-fill) → 1, "dark" (all 0s) → 0
  CNN::Samples<double> samples(2);
  samples[0].input = makeGradientInput<double>({1, 5, 5});
  samples[0].output = {1.0};
  samples[1].input = CNN::Tensor3D<double>({1, 5, 5}, 0.0);
  samples[1].output = {0.0};

  auto core = CNN::Core<double>::makeCore(config);

  // Train callback that requests stop after the first epoch's first sample fires.
  core->setTrainCallback([&core](const Common::TrainProgressEvent<double>& p) {
    if (p.currentEpoch >= 1)
      core->requestStop();
  });

  core->train(samples.size(), CNN::makeSampleProvider(samples));

  const auto& history = core->getTrainMetadata().epochHistory;
  CHECK(history.size() >= 1, "at least one epoch completed before abort");
  CHECK(history.size() < 100, "abort produced partial training (not all 100 epochs)");
}

//===================================================================================================================//

void runCPUPredictStopTests()
{
  runPredictStopRequested();
}

//===================================================================================================================//

void runCPUTrainStopTests()
{
  testTrainStopRequested();
}
