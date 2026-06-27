#include "test_helpers.hpp"

#include <QObject>

//===================================================================================================================//

static void testMakeCoreCPU()
{
  TestScope _t("testMakeCoreCPU");

  ANN::CoreConfig<double> config;
  config.modeType = Common::ModeType::PREDICT;
  config.deviceType = Common::DeviceType::CPU;
  config.layersConfig = makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {1, ANN::ActvFuncType::SIGMOID}});

  auto core = ANN::Core<double>::makeCore(config);
  CHECK(core != nullptr, "makeCore returns non-null");
  CHECK(core->getDeviceType() == Common::DeviceType::CPU, "device is CPU");
  CHECK(core->getModeType() == Common::ModeType::PREDICT, "mode is PREDICT");
  CHECK(core->getLayersConfig().size() == 2, "2 layers");
  CHECK(core->getLayersConfig().getTotalNumNeurons() == 3, "total neurons = 3");
}

//===================================================================================================================//

static void testPredictSimple()
{
  TestScope _t("testPredictSimple");

  // 2 inputs → 1 output with known weights
  ANN::CoreConfig<double> config;
  config.modeType = Common::ModeType::PREDICT;
  config.deviceType = Common::DeviceType::CPU;
  config.layersConfig = makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {1, ANN::ActvFuncType::SIGMOID}});

  // Set known weights: w = [0.5, 0.5], bias = 0.0
  config.parameters.weights.resize(2);
  config.parameters.weights[1] = {{0.5, 0.5}};
  config.parameters.biases.resize(2);
  config.parameters.biases[1] = {0.0};

  auto core = ANN::Core<double>::makeCore(config);
  ANN::Output<double> out = core->predict({1.0, 1.0}).output;

  CHECK(out.size() == 1, "output size = 1");
  // z = 0.5*1 + 0.5*1 + 0 = 1.0, sigmoid(1.0) ≈ 0.7311
  double expected = 1.0 / (1.0 + std::exp(-1.0));
  CHECK_NEAR(out[0], expected, 1e-5, "sigmoid(1.0) ≈ 0.7311");
}

//===================================================================================================================//

static void testTrainXOR()
{
  TestScope _t("testTrainXOR");

  ANN::Samples<double> samples = {{{0.0, 0.0}, {0.0}}, {{0.0, 1.0}, {1.0}}, {{1.0, 0.0}, {1.0}}, {{1.0, 1.0}, {0.0}}};

  ANN::CoreConfig<double> config;
  config.modeType = Common::ModeType::TRAIN;
  config.deviceType = Common::DeviceType::CPU;
  // Bigger hidden layer (8 RELUs) + Adam — combination known to converge XOR
  // robustly. SGD on a 4-unit hidden layer with deterministic init was prone
  // to saddles; the test was concealing it with a 5-attempt retry loop.
  config.layersConfig =
    makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {8, ANN::ActvFuncType::RELU}, {1, ANN::ActvFuncType::SIGMOID}});

  config.trainConfig.numEpochs = 3000;
  config.trainConfig.learningRate = 0.05;
  config.trainConfig.optimizer.type = Common::OptimizerType::ADAM;
  config.trainConfig.shuffleSeed = 42; // Fully deterministic.
  config.numThreads = 1; // Single-threaded — parallel batch reduction order is FP-non-deterministic.
  config.progressReports = 0;
  config.logLevel = Common::LogLevel::ERROR;

  auto core = ANN::Core<double>::makeCore(config);
  core->train(samples.size(), ANN::makeSampleProvider(samples));

  ANN::Output<double> p00 = core->predict({0.0, 0.0}).output;
  ANN::Output<double> p01 = core->predict({0.0, 1.0}).output;
  ANN::Output<double> p10 = core->predict({1.0, 0.0}).output;
  ANN::Output<double> p11 = core->predict({1.0, 1.0}).output;

  std::cout << "  XOR: [0,0]=" << p00[0] << " [0,1]=" << p01[0] << " [1,0]=" << p10[0] << " [1,1]=" << p11[0]
            << std::endl;
  CHECK(p00[0] < 0.3, "XOR(0,0) ≈ 0");
  CHECK(p01[0] > 0.7, "XOR(0,1) ≈ 1");
  CHECK(p10[0] > 0.7, "XOR(1,0) ≈ 1");
  CHECK(p11[0] < 0.3, "XOR(1,1) ≈ 0");
}

//===================================================================================================================//

static void testTestMethod()
{
  TestScope _t("testTestMethod");

  // Train a simple network first, then test
  ANN::CoreConfig<double> config;
  config.modeType = Common::ModeType::TRAIN;
  config.deviceType = Common::DeviceType::CPU;
  config.layersConfig =
    makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {4, ANN::ActvFuncType::SIGMOID}, {1, ANN::ActvFuncType::SIGMOID}});

  config.trainConfig.numEpochs = 500;
  config.trainConfig.learningRate = 0.5;
  config.progressReports = 0;

  ANN::Samples<double> samples = {{{0.0, 0.0}, {0.0}}, {{1.0, 1.0}, {1.0}}};

  auto core = ANN::Core<double>::makeCore(config);
  core->train(samples.size(), ANN::makeSampleProvider(samples));

  Common::TestResult<double> result = core->test(samples.size(), ANN::makeSampleProvider(samples));
  CHECK(result.numSamples == 2, "test numSamples = 2");
  CHECK(result.averageLoss >= 0.0, "test averageLoss >= 0");
  CHECK(result.totalLoss >= 0.0, "test totalLoss >= 0");
  CHECK_NEAR(result.totalLoss, result.averageLoss * 2.0, 1e-10, "totalLoss = avgLoss * numSamples");
  CHECK(result.numCorrect <= result.numSamples, "numCorrect <= numSamples");
  CHECK(result.accuracy >= 0.0 && result.accuracy <= 100.0, "accuracy in [0, 100]");
  std::cout << "  test avgLoss=" << result.averageLoss << " accuracy=" << result.accuracy << "%" << std::endl;
}

//===================================================================================================================//

static void testTrainMetadata()
{
  TestScope _t("testTrainMetadata");

  ANN::CoreConfig<double> config;
  config.modeType = Common::ModeType::TRAIN;
  config.deviceType = Common::DeviceType::CPU;
  config.layersConfig = makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {1, ANN::ActvFuncType::SIGMOID}});
  config.trainConfig.numEpochs = 10;
  config.trainConfig.learningRate = 0.1;
  config.progressReports = 0;

  ANN::Samples<double> samples = {{{1.0, 0.0}, {1.0}}};
  auto core = ANN::Core<double>::makeCore(config);
  core->train(samples.size(), ANN::makeSampleProvider(samples));

  const auto& meta = core->getTrainMetadata();
  CHECK(!meta.startTime.empty(), "startTime non-empty");
  CHECK(!meta.endTime.empty(), "endTime non-empty");
  CHECK(meta.durationSeconds >= 0.0, "durationSeconds >= 0");
  CHECK(!meta.durationFormatted.empty(), "durationFormatted non-empty");
  CHECK(meta.numSamples == 1, "numSamples = 1");
}

//===================================================================================================================//

static void testPredictMetadata()
{
  TestScope _t("testPredictMetadata");

  ANN::CoreConfig<double> config;
  config.modeType = Common::ModeType::PREDICT;
  config.deviceType = Common::DeviceType::CPU;
  config.layersConfig = makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {1, ANN::ActvFuncType::SIGMOID}});

  auto core = ANN::Core<double>::makeCore(config);
  core->predict({1.0, 0.0});

  const auto& meta = core->getPredictMetadata();
  CHECK(!meta.startTime.empty(), "predict startTime non-empty");
  CHECK(!meta.endTime.empty(), "predict endTime non-empty");
  CHECK(meta.durationSeconds >= 0.0, "predict durationSeconds >= 0");
}

//===================================================================================================================//

static void testTrainCallback()
{
  TestScope _t("testTrainCallback");

  ANN::CoreConfig<double> config;
  config.modeType = Common::ModeType::TRAIN;
  config.deviceType = Common::DeviceType::CPU;
  config.layersConfig = makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {1, ANN::ActvFuncType::SIGMOID}});
  config.trainConfig.numEpochs = 5;
  config.trainConfig.learningRate = 0.1;
  config.progressReports = 1;

  ANN::Samples<double> samples = {{{1.0, 0.0}, {1.0}}, {{0.0, 1.0}, {0.0}}};

  int callbackCount = 0;
  auto core = ANN::Core<double>::makeCore(config);
  QObject::connect(
    &core->getCoreSignals(), &ANN::CoreSignals::trainProgress, &core->getCoreSignals(),
    [&callbackCount](ulong, ulong, ulong, ulong, double, double, bool, bool, int, int) { callbackCount++; },
    Qt::DirectConnection);

  core->train(samples.size(), ANN::makeSampleProvider(samples));

  std::cout << "  callback called " << callbackCount << " times" << std::endl;
  CHECK(callbackCount > 0, "training callback was called");
  CHECK(callbackCount >= 5, "callback called at least once per epoch");
}

//===================================================================================================================//

static void testParameterRoundTrip()
{
  TestScope _t("testParameterRoundTrip");

  // Train a network, get parameters, create new predict-mode core with those params
  ANN::CoreConfig<double> trainConfig;
  trainConfig.modeType = Common::ModeType::TRAIN;
  trainConfig.deviceType = Common::DeviceType::CPU;
  trainConfig.layersConfig =
    makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {3, ANN::ActvFuncType::SIGMOID}, {1, ANN::ActvFuncType::SIGMOID}});

  trainConfig.trainConfig.numEpochs = 200;
  trainConfig.trainConfig.learningRate = 0.5;
  trainConfig.progressReports = 0;

  ANN::Samples<double> samples = {{{1.0, 1.0}, {1.0}}, {{0.0, 0.0}, {0.0}}};

  auto trainCore = ANN::Core<double>::makeCore(trainConfig);
  trainCore->train(samples.size(), ANN::makeSampleProvider(samples));

  ANN::Output<double> originalPred = trainCore->predict({1.0, 1.0}).output;
  ANN::Parameters<double> params = trainCore->getParameters();

  // Check parameters are non-empty
  CHECK(!params.weights.empty(), "weights non-empty");
  CHECK(!params.biases.empty(), "biases non-empty");

  // Create predict-only core with same params
  ANN::CoreConfig<double> predConfig;
  predConfig.modeType = Common::ModeType::PREDICT;
  predConfig.deviceType = Common::DeviceType::CPU;
  predConfig.layersConfig = trainConfig.layersConfig;
  predConfig.parameters = params;

  auto predCore = ANN::Core<double>::makeCore(predConfig);
  ANN::Output<double> loadedPred = predCore->predict({1.0, 1.0}).output;

  std::cout << "  original=" << originalPred[0] << "  from_params=" << loadedPred[0] << std::endl;
  CHECK_NEAR(originalPred[0], loadedPred[0], 1e-10, "parameter round-trip exact match");
}

//===================================================================================================================//

static void testParametersDuringTrain()
{
  TestScope _t("testParametersDuringTrain");

  // Train a network and verify that getParameters() returns non-empty data
  // during training (in the epoch-completion callback), not just after training ends.
  ANN::CoreConfig<double> config;
  config.modeType = Common::ModeType::TRAIN;
  config.deviceType = Common::DeviceType::CPU;
  config.layersConfig =
    makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {4, ANN::ActvFuncType::RELU}, {1, ANN::ActvFuncType::SIGMOID}});

  config.trainConfig.numEpochs = 10;
  config.trainConfig.learningRate = 0.5;
  config.progressReports = 0;

  ANN::Samples<double> samples = {{{1.0, 1.0}, {1.0}}, {{0.0, 0.0}, {0.0}}};

  auto core = ANN::Core<double>::makeCore(config);

  bool paramsChecked = false;
  bool weightsNonEmpty = false;
  bool biasesNonEmpty = false;

  QObject::connect(
    &core->getCoreSignals(), &ANN::CoreSignals::trainProgress, &core->getCoreSignals(),
    [&](ulong, ulong, ulong, ulong, double epochLoss, double sampleLoss, bool, bool, int, int) {
      if (epochLoss > 0 && sampleLoss == 0 && !paramsChecked) {
        const ANN::Parameters<double>& params = core->getParameters();
        weightsNonEmpty = !params.weights.empty();
        biasesNonEmpty = !params.biases.empty();
        paramsChecked = true;
      }
    },

    Qt::DirectConnection);

  core->train(samples.size(), ANN::makeSampleProvider(samples));

  CHECK(paramsChecked, "epoch-completion callback fired");
  CHECK(weightsNonEmpty, "parameters.weights non-empty during training");
  CHECK(biasesNonEmpty, "parameters.biases non-empty during training");
}

//===================================================================================================================//

//===================================================================================================================//

static void testBatchPredict()
{
  TestScope _t("testBatchPredict");

  // 2 inputs → 1 output with known weights
  ANN::CoreConfig<double> config;
  config.modeType = Common::ModeType::PREDICT;
  config.deviceType = Common::DeviceType::CPU;
  config.layersConfig = makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {1, ANN::ActvFuncType::SIGMOID}});

  config.parameters.weights.resize(2);
  config.parameters.weights[1] = {{0.5, 0.5}};
  config.parameters.biases.resize(2);
  config.parameters.biases[1] = {0.0};

  auto core = ANN::Core<double>::makeCore(config);

  // Batch predict with multiple inputs at once. The streaming API takes a
  // provider that yields one batch at a time; for in-memory data we slice
  // the original Inputs<T> by (batchSize, batchIndex).
  ANN::Inputs<double> inputs = {{1.0, 1.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
  auto sliceProvider = [&inputs](ulong batchSize, ulong batchIndex) {
    ulong start = batchIndex * batchSize;
    ulong end = std::min(start + batchSize, static_cast<ulong>(inputs.size()));

    if (start >= end)
      return ANN::InputsView<double>{};
    return ANN::InputsView<double>(inputs.data() + start, end - start);
  };

  Common::PredictResults<double> results = core->predict(inputs.size(), sliceProvider);

  CHECK(results.size() == 4, "batch predict returns 4 outputs");

  // Each output should have 1 element
  for (size_t i = 0; i < results.size(); i++)
    CHECK(results[i].output.size() == 1, "output[" + std::to_string(i) + "] size = 1");

  // Verify known values
  double exp_11 = 1.0 / (1.0 + std::exp(-1.0)); // sigmoid(0.5 + 0.5) = sigmoid(1.0)
  double exp_00 = 1.0 / (1.0 + std::exp(0.0)); // sigmoid(0) = 0.5
  double exp_10 = 1.0 / (1.0 + std::exp(-0.5)); // sigmoid(0.5)
  double exp_01 = 1.0 / (1.0 + std::exp(-0.5)); // sigmoid(0.5)

  CHECK_NEAR(results[0].output[0], exp_11, 1e-5, "batch predict {1,1}");
  CHECK_NEAR(results[1].output[0], exp_00, 1e-5, "batch predict {0,0}");
  CHECK_NEAR(results[2].output[0], exp_10, 1e-5, "batch predict {1,0}");
  CHECK_NEAR(results[3].output[0], exp_01, 1e-5, "batch predict {0,1}");

  // Logits sanity: each batch result should also carry a same-sized logits vector
  for (size_t i = 0; i < results.size(); i++)
    CHECK(results[i].logits.size() == results[i].output.size(),
          "logits[" + std::to_string(i) + "] size matches output");

  // Verify batch predict matches single predict
  for (size_t i = 0; i < inputs.size(); i++) {
    ANN::Output<double> single = core->predict(inputs[i]).output;
    CHECK_NEAR(results[i].output[0], single[0], 1e-10, "batch[" + std::to_string(i) + "] matches single predict");
  }
}

//===================================================================================================================//

static void testBatchPredictAfterTrain()
{
  TestScope _t("testBatchPredictAfterTrain");

  // Train XOR then batch predict all 4 patterns
  ANN::Samples<double> samples = {{{0.0, 0.0}, {0.0}}, {{0.0, 1.0}, {1.0}}, {{1.0, 0.0}, {1.0}}, {{1.0, 1.0}, {0.0}}};

  ANN::CoreConfig<double> config;
  config.modeType = Common::ModeType::TRAIN;
  config.deviceType = Common::DeviceType::CPU;
  // Same robust config as testTrainXOR: 8-unit hidden + Adam + fixed shuffle.
  config.layersConfig =
    makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {8, ANN::ActvFuncType::RELU}, {1, ANN::ActvFuncType::SIGMOID}});

  config.trainConfig.numEpochs = 3000;
  config.trainConfig.learningRate = 0.05;
  config.trainConfig.optimizer.type = Common::OptimizerType::ADAM;
  config.trainConfig.shuffleSeed = 42;
  config.numThreads = 1; // Single-threaded — parallel batch reduction order is FP-non-deterministic.
  config.progressReports = 0;
  config.logLevel = Common::LogLevel::ERROR;

  auto core = ANN::Core<double>::makeCore(config);
  core->train(samples.size(), ANN::makeSampleProvider(samples));

  ANN::Inputs<double> inputs = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
  auto sliceProvider = [&inputs](ulong batchSize, ulong batchIndex) {
    ulong start = batchIndex * batchSize;
    ulong end = std::min(start + batchSize, static_cast<ulong>(inputs.size()));

    if (start >= end)
      return ANN::InputsView<double>{};
    return ANN::InputsView<double>(inputs.data() + start, end - start);
  };

  Common::PredictResults<double> results = core->predict(inputs.size(), sliceProvider);

  CHECK(results.size() == 4, "batch predict returns 4 outputs");
  CHECK(results[0].output[0] < 0.5, "XOR(0,0) ≈ 0");
  CHECK(results[1].output[0] > 0.5, "XOR(0,1) ≈ 1");
  CHECK(results[2].output[0] > 0.5, "XOR(1,0) ≈ 1");
  CHECK(results[3].output[0] < 0.5, "XOR(1,1) ≈ 0");
}

//===================================================================================================================//

static void testBatchPredictAbort()
{
  TestScope _t("testBatchPredictAbort");

  // 2 inputs → 1 output, simple MLP
  ANN::CoreConfig<double> config;
  config.modeType = Common::ModeType::PREDICT;
  config.deviceType = Common::DeviceType::CPU;
  config.layersConfig = makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {1, ANN::ActvFuncType::SIGMOID}});

  auto core = ANN::Core<double>::makeCore(config);

  // 200 inputs — 200 batches with batchSize=1
  ANN::Inputs<double> inputs(200, ANN::Input<double>(2, 0.0));

  auto sliceProvider = [&inputs](ulong batchSize, ulong batchIndex) {
    ulong start = batchIndex * batchSize;
    ulong end = std::min(start + batchSize, static_cast<ulong>(inputs.size()));

    if (start >= end)
      return ANN::InputsView<double>{};
    return ANN::InputsView<double>(inputs.data() + start, end - start);
  };

  // Progress callback that aborts after the first batch completes.
  // The callback receives (current, total) where `current` is the number
  // of samples already processed. After the first batch (batchSize=1)
  // `current` will be 1, so we request stop at that point.
  QObject::connect(
    &core->getCoreSignals(), &ANN::CoreSignals::predictProgress, &core->getCoreSignals(),
    [&core](ulong current, ulong) {
      if (current >= 1)
        core->requestStop();
    },

    Qt::DirectConnection);

  Common::PredictResults<double> results = core->predict(inputs.size(), sliceProvider);

  CHECK(results.size() < inputs.size(), "abort produces partial results (size < 200)");
  CHECK(results.size() >= 1, "at least one batch completed");
}

//===================================================================================================================//

static void testTrainAbort()
{
  TestScope _t("testTrainAbort");

  // 2 inputs → 1 output, simple MLP
  ANN::CoreConfig<double> config;
  config.modeType = Common::ModeType::TRAIN;
  config.deviceType = Common::DeviceType::CPU;
  config.layersConfig = makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {1, ANN::ActvFuncType::SIGMOID}});
  config.trainConfig.numEpochs = 50;
  config.trainConfig.learningRate = 0.5;
  config.trainConfig.batchSize = 1;
  config.numThreads = 0;
  config.progressReports = 0;
  config.logLevel = Common::LogLevel::ERROR;

  ANN::Samples<double> samples = {{{0.0, 0.0}, {0.0}}, {{1.0, 1.0}, {1.0}}};

  auto core = ANN::Core<double>::makeCore(config);

  // Train callback that aborts after epoch 1 starts.
  // The callback fires per-sample during training; after the first
  // sample of epoch 1 (currentEpoch == 1) we request stop.
  QObject::connect(
    &core->getCoreSignals(), &ANN::CoreSignals::trainProgress, &core->getCoreSignals(),
    [&core](ulong currentEpoch, ulong, ulong, ulong, double, double, bool, bool, int, int) {
      if (currentEpoch >= 1)
        core->requestStop();
    },

    Qt::DirectConnection);

  core->train(samples.size(), ANN::makeSampleProvider(samples));

  const auto& meta = core->getTrainMetadata();
  CHECK(meta.epochHistory.size() >= 1, "at least one epoch completed");
  CHECK(meta.epochHistory.size() < 50, "aborted before all 50 epochs");
}

//===================================================================================================================//

void runCPUBasicTests()
{
  testMakeCoreCPU();
  testPredictSimple();
  testBatchPredict();
  testTrainXOR();
  testBatchPredictAfterTrain();
  testTestMethod();
  testTrainMetadata();
  testPredictMetadata();
  testTrainCallback();
  testParameterRoundTrip();
  testParametersDuringTrain();
  testBatchPredictAbort();
  testTrainAbort();
}
