#include "test_helpers.hpp"

//===================================================================================================================//

static void testMakeCoreCPU()
{
  std::cout << "--- testMakeCoreCPU ---" << std::endl;

  ANN::CoreConfig<double> config;
  config.modeType = ANN::ModeType::PREDICT;
  config.deviceType = ANN::DeviceType::CPU;
  config.layersConfig = makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {1, ANN::ActvFuncType::SIGMOID}});

  auto core = ANN::Core<double>::makeCore(config);
  CHECK(core != nullptr, "makeCore returns non-null");
  CHECK(core->getDeviceType() == ANN::DeviceType::CPU, "device is CPU");
  CHECK(core->getModeType() == ANN::ModeType::PREDICT, "mode is PREDICT");
  CHECK(core->getLayersConfig().size() == 2, "2 layers");
  CHECK(core->getLayersConfig().getTotalNumNeurons() == 3, "total neurons = 3");
}

//===================================================================================================================//

static void testPredictSimple()
{
  std::cout << "--- testPredictSimple ---" << std::endl;

  // 2 inputs → 1 output with known weights
  ANN::CoreConfig<double> config;
  config.modeType = ANN::ModeType::PREDICT;
  config.deviceType = ANN::DeviceType::CPU;
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
  std::cout << "--- testTrainXOR ---" << std::endl;

  ANN::Samples<double> samples = {{{0.0, 0.0}, {0.0}}, {{0.0, 1.0}, {1.0}}, {{1.0, 0.0}, {1.0}}, {{1.0, 1.0}, {0.0}}};

  bool converged = false;
  ANN::Output<double> p00, p01, p10, p11;

  for (int attempt = 0; attempt < 5 && !converged; ++attempt) {
    if (attempt > 0)
      std::cout << "  retry #" << attempt << std::endl;

    ANN::CoreConfig<double> config;
    config.modeType = ANN::ModeType::TRAIN;
    config.deviceType = ANN::DeviceType::CPU;
    config.layersConfig =
      makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {4, ANN::ActvFuncType::RELU}, {1, ANN::ActvFuncType::SIGMOID}});

    config.trainingConfig.numEpochs = 2000;
    config.trainingConfig.learningRate = 0.1;
    config.progressReports = 0;
    config.logLevel = ANN::LogLevel::ERROR;

    auto core = ANN::Core<double>::makeCore(config);
    core->train(samples.size(), ANN::makeSampleProvider(samples));

    p00 = core->predict({0.0, 0.0}).output;
    p01 = core->predict({0.0, 1.0}).output;
    p10 = core->predict({1.0, 0.0}).output;
    p11 = core->predict({1.0, 1.0}).output;

    if (p00[0] < 0.3 && p01[0] > 0.7 && p10[0] > 0.7 && p11[0] < 0.3) {
      converged = true;
    }
  }

  std::cout << "  XOR: [0,0]=" << p00[0] << " [0,1]=" << p01[0] << " [1,0]=" << p10[0] << " [1,1]=" << p11[0]
            << std::endl;
  CHECK(converged, "XOR converged (5 attempts)");
}

//===================================================================================================================//

static void testTestMethod()
{
  std::cout << "--- testTestMethod ---" << std::endl;

  // Train a simple network first, then test
  ANN::CoreConfig<double> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::CPU;
  config.layersConfig =
    makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {4, ANN::ActvFuncType::SIGMOID}, {1, ANN::ActvFuncType::SIGMOID}});

  config.trainingConfig.numEpochs = 500;
  config.trainingConfig.learningRate = 0.5;
  config.progressReports = 0;

  ANN::Samples<double> samples = {{{0.0, 0.0}, {0.0}}, {{1.0, 1.0}, {1.0}}};

  auto core = ANN::Core<double>::makeCore(config);
  core->train(samples.size(), ANN::makeSampleProvider(samples));

  ANN::TestResult<double> result = core->test(samples.size(), ANN::makeSampleProvider(samples));
  CHECK(result.numSamples == 2, "test numSamples = 2");
  CHECK(result.averageLoss >= 0.0, "test averageLoss >= 0");
  CHECK(result.totalLoss >= 0.0, "test totalLoss >= 0");
  CHECK_NEAR(result.totalLoss, result.averageLoss * 2.0, 1e-10, "totalLoss = avgLoss * numSamples");
  CHECK(result.numCorrect <= result.numSamples, "numCorrect <= numSamples");
  CHECK(result.accuracy >= 0.0 && result.accuracy <= 100.0, "accuracy in [0, 100]");
  std::cout << "  test avgLoss=" << result.averageLoss << " accuracy=" << result.accuracy << "%" << std::endl;
}

//===================================================================================================================//

static void testTrainingMetadata()
{
  std::cout << "--- testTrainingMetadata ---" << std::endl;

  ANN::CoreConfig<double> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::CPU;
  config.layersConfig = makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {1, ANN::ActvFuncType::SIGMOID}});
  config.trainingConfig.numEpochs = 10;
  config.trainingConfig.learningRate = 0.1;
  config.progressReports = 0;

  ANN::Samples<double> samples = {{{1.0, 0.0}, {1.0}}};
  auto core = ANN::Core<double>::makeCore(config);
  core->train(samples.size(), ANN::makeSampleProvider(samples));

  const auto& meta = core->getTrainingMetadata();
  CHECK(!meta.startTime.empty(), "startTime non-empty");
  CHECK(!meta.endTime.empty(), "endTime non-empty");
  CHECK(meta.durationSeconds >= 0.0, "durationSeconds >= 0");
  CHECK(!meta.durationFormatted.empty(), "durationFormatted non-empty");
  CHECK(meta.numSamples == 1, "numSamples = 1");
}

//===================================================================================================================//

static void testPredictMetadata()
{
  std::cout << "--- testPredictMetadata ---" << std::endl;

  ANN::CoreConfig<double> config;
  config.modeType = ANN::ModeType::PREDICT;
  config.deviceType = ANN::DeviceType::CPU;
  config.layersConfig = makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {1, ANN::ActvFuncType::SIGMOID}});

  auto core = ANN::Core<double>::makeCore(config);
  core->predict({1.0, 0.0});

  const auto& meta = core->getPredictMetadata();
  CHECK(!meta.startTime.empty(), "predict startTime non-empty");
  CHECK(!meta.endTime.empty(), "predict endTime non-empty");
  CHECK(meta.durationSeconds >= 0.0, "predict durationSeconds >= 0");
}

//===================================================================================================================//

static void testTrainingCallback()
{
  std::cout << "--- testTrainingCallback ---" << std::endl;

  ANN::CoreConfig<double> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::CPU;
  config.layersConfig = makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {1, ANN::ActvFuncType::SIGMOID}});
  config.trainingConfig.numEpochs = 5;
  config.trainingConfig.learningRate = 0.1;
  config.progressReports = 1;

  ANN::Samples<double> samples = {{{1.0, 0.0}, {1.0}}, {{0.0, 1.0}, {0.0}}};

  int callbackCount = 0;
  auto core = ANN::Core<double>::makeCore(config);
  core->setTrainingCallback([&callbackCount](const ANN::TrainingProgress<double>& progress) { callbackCount++; });

  core->train(samples.size(), ANN::makeSampleProvider(samples));

  std::cout << "  callback called " << callbackCount << " times" << std::endl;
  CHECK(callbackCount > 0, "training callback was called");
  CHECK(callbackCount >= 5, "callback called at least once per epoch");
}

//===================================================================================================================//

static void testParameterRoundTrip()
{
  std::cout << "--- testParameterRoundTrip ---" << std::endl;

  // Train a network, get parameters, create new predict-mode core with those params
  ANN::CoreConfig<double> trainConfig;
  trainConfig.modeType = ANN::ModeType::TRAIN;
  trainConfig.deviceType = ANN::DeviceType::CPU;
  trainConfig.layersConfig =
    makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {3, ANN::ActvFuncType::SIGMOID}, {1, ANN::ActvFuncType::SIGMOID}});

  trainConfig.trainingConfig.numEpochs = 200;
  trainConfig.trainingConfig.learningRate = 0.5;
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
  predConfig.modeType = ANN::ModeType::PREDICT;
  predConfig.deviceType = ANN::DeviceType::CPU;
  predConfig.layersConfig = trainConfig.layersConfig;
  predConfig.parameters = params;

  auto predCore = ANN::Core<double>::makeCore(predConfig);
  ANN::Output<double> loadedPred = predCore->predict({1.0, 1.0}).output;

  std::cout << "  original=" << originalPred[0] << "  from_params=" << loadedPred[0] << std::endl;
  CHECK_NEAR(originalPred[0], loadedPred[0], 1e-10, "parameter round-trip exact match");
}

//===================================================================================================================//

static void testParametersDuringTraining()
{
  std::cout << "--- testParametersDuringTraining ---" << std::endl;

  // Train a network and verify that getParameters() returns non-empty data
  // during training (in the epoch-completion callback), not just after training ends.
  ANN::CoreConfig<double> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::CPU;
  config.layersConfig =
    makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {4, ANN::ActvFuncType::RELU}, {1, ANN::ActvFuncType::SIGMOID}});

  config.trainingConfig.numEpochs = 10;
  config.trainingConfig.learningRate = 0.5;
  config.progressReports = 0;

  ANN::Samples<double> samples = {{{1.0, 1.0}, {1.0}}, {{0.0, 0.0}, {0.0}}};

  auto core = ANN::Core<double>::makeCore(config);

  bool paramsChecked = false;
  bool weightsNonEmpty = false;
  bool biasesNonEmpty = false;

  core->setTrainingCallback([&](const ANN::TrainingProgress<double>& progress) {
    // Detect epoch-completion callback: epochLoss > 0 and sampleLoss == 0
    if (progress.epochLoss > 0 && progress.sampleLoss == 0 && !paramsChecked) {
      const ANN::Parameters<double>& params = core->getParameters();
      weightsNonEmpty = !params.weights.empty();
      biasesNonEmpty = !params.biases.empty();
      paramsChecked = true;
    }
  });

  core->train(samples.size(), ANN::makeSampleProvider(samples));

  CHECK(paramsChecked, "epoch-completion callback fired");
  CHECK(weightsNonEmpty, "parameters.weights non-empty during training");
  CHECK(biasesNonEmpty, "parameters.biases non-empty during training");
}

//===================================================================================================================//

//===================================================================================================================//

static void testBatchPredict()
{
  std::cout << "--- testBatchPredict ---" << std::endl;

  // 2 inputs → 1 output with known weights
  ANN::CoreConfig<double> config;
  config.modeType = ANN::ModeType::PREDICT;
  config.deviceType = ANN::DeviceType::CPU;
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
      return ANN::Inputs<double>{};
    return ANN::Inputs<double>(inputs.begin() + start, inputs.begin() + end);
  };

  ANN::PredictResults<double> results = core->predict(inputs.size(), sliceProvider);

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

static void testBatchPredictAfterTraining()
{
  std::cout << "--- testBatchPredictAfterTraining ---" << std::endl;

  // Train XOR then batch predict all 4 patterns
  ANN::Samples<double> samples = {{{0.0, 0.0}, {0.0}}, {{0.0, 1.0}, {1.0}}, {{1.0, 0.0}, {1.0}}, {{1.0, 1.0}, {0.0}}};

  ANN::CoreConfig<double> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::CPU;
  config.layersConfig =
    makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {4, ANN::ActvFuncType::RELU}, {1, ANN::ActvFuncType::SIGMOID}});

  config.trainingConfig.numEpochs = 2000;
  config.trainingConfig.learningRate = 0.1;
  config.progressReports = 0;
  config.logLevel = ANN::LogLevel::ERROR;

  bool converged = false;
  ANN::PredictResults<double> results;

  for (int attempt = 0; attempt < 5 && !converged; ++attempt) {
    if (attempt > 0)
      std::cout << "  retry #" << attempt << std::endl;

    auto core = ANN::Core<double>::makeCore(config);
    core->train(samples.size(), ANN::makeSampleProvider(samples));

    // Batch predict all 4 XOR inputs via a slice-and-yield provider.
    ANN::Inputs<double> inputs = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
    auto sliceProvider = [&inputs](ulong batchSize, ulong batchIndex) {
      ulong start = batchIndex * batchSize;
      ulong end = std::min(start + batchSize, static_cast<ulong>(inputs.size()));

      if (start >= end)
        return ANN::Inputs<double>{};
      return ANN::Inputs<double>(inputs.begin() + start, inputs.begin() + end);
    };

    results = core->predict(inputs.size(), sliceProvider);

    if (results[0].output[0] < 0.5 && results[1].output[0] > 0.5 && results[2].output[0] > 0.5 &&
        results[3].output[0] < 0.5)
      converged = true;
  }

  CHECK(results.size() == 4, "batch predict returns 4 outputs");
  CHECK(converged, "XOR batch predict converged (5 attempts)");
}

//===================================================================================================================//

void runCoreBasicTests()
{
  testMakeCoreCPU();
  testPredictSimple();
  testBatchPredict();
  testTrainXOR();
  testBatchPredictAfterTraining();
  testTestMethod();
  testTrainingMetadata();
  testPredictMetadata();
  testTrainingCallback();
  testParameterRoundTrip();
  testParametersDuringTraining();
}
