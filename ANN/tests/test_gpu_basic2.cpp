#include "test_helpers.hpp"

static void testGPUCrossEntropyTraining()
{
  std::cout << "--- testGPUCrossEntropyTraining ---" << std::endl;

  // Classification: 2 inputs → 3 classes with softmax + cross-entropy on GPU
  ANN::Samples<float> samples = {
    {{1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}}, {{0.0f, 1.0f}, {0.0f, 1.0f, 0.0f}}, {{1.0f, 1.0f}, {0.0f, 0.0f, 1.0f}}};

  bool converged = false;

  for (int attempt = 0; attempt < 5 && !converged; ++attempt) {
    if (attempt > 0)
      std::cout << "  retry #" << attempt << std::endl;

    ANN::CoreConfig<float> config;
    config.modeType = ANN::ModeType::TRAIN;
    config.deviceType = ANN::DeviceType::GPU;
    config.layersConfig =
      makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {4, ANN::ActvFuncType::RELU}, {3, ANN::ActvFuncType::SOFTMAX}});

    config.costFunctionConfig.type = ANN::CostFunctionType::CROSS_ENTROPY;
    config.trainingConfig.numEpochs = 500;
    config.trainingConfig.learningRate = 0.1f;
    config.progressReports = 0;
    config.numGPUs = 1;
    config.logLevel = ANN::LogLevel::ERROR;

    auto core = ANN::Core<float>::makeCore(config);
    core->train(samples.size(), ANN::makeSampleProvider(samples));

    auto out0 = core->predict({1.0f, 0.0f});
    auto out1 = core->predict({0.0f, 1.0f});
    auto out2 = core->predict({1.0f, 1.0f});

    // Softmax outputs should sum to 1
    bool sumsOk = std::fabs(out0[0] + out0[1] + out0[2] - 1.0f) < 0.01f &&
                  std::fabs(out1[0] + out1[1] + out1[2] - 1.0f) < 0.01f &&
                  std::fabs(out2[0] + out2[1] + out2[2] - 1.0f) < 0.01f;

    // Correct class should have highest probability
    bool classOk = out0[0] > out0[1] && out0[0] > out0[2] && out1[1] > out1[0] && out1[1] > out1[2] &&
                   out2[2] > out2[0] && out2[2] > out2[1];

    if (sumsOk && classOk)
      converged = true;
  }

  CHECK(converged, "GPU cross-entropy + softmax converged (5 attempts)");
}

//===================================================================================================================//

static void testGPUCrossEntropyCPUParity()
{
  std::cout << "--- testGPUCrossEntropyCPUParity ---" << std::endl;

  // Train on CPU with cross-entropy, then compare predict on CPU vs GPU
  ANN::CoreConfig<float> trainConfig;
  trainConfig.modeType = ANN::ModeType::TRAIN;
  trainConfig.deviceType = ANN::DeviceType::CPU;
  trainConfig.layersConfig =
    makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {4, ANN::ActvFuncType::RELU}, {2, ANN::ActvFuncType::SOFTMAX}});

  trainConfig.costFunctionConfig.type = ANN::CostFunctionType::CROSS_ENTROPY;
  trainConfig.trainingConfig.numEpochs = 200;
  trainConfig.trainingConfig.learningRate = 0.1f;
  trainConfig.progressReports = 0;
  trainConfig.logLevel = ANN::LogLevel::ERROR;

  ANN::Samples<float> samples = {{{1.0f, 0.0f}, {1.0f, 0.0f}}, {{0.0f, 1.0f}, {0.0f, 1.0f}}};

  auto trainCore = ANN::Core<float>::makeCore(trainConfig);
  trainCore->train(samples.size(), ANN::makeSampleProvider(samples));

  ANN::Parameters<float> params = trainCore->getParameters();

  // CPU predict
  ANN::CoreConfig<float> cpuConfig;
  cpuConfig.modeType = ANN::ModeType::PREDICT;
  cpuConfig.deviceType = ANN::DeviceType::CPU;
  cpuConfig.layersConfig = trainConfig.layersConfig;
  cpuConfig.parameters = params;
  cpuConfig.logLevel = ANN::LogLevel::ERROR;

  auto cpuCore = ANN::Core<float>::makeCore(cpuConfig);
  ANN::Output<float> cpuPred = cpuCore->predict({1.0f, 0.0f});

  // GPU predict
  ANN::CoreConfig<float> gpuConfig;
  gpuConfig.modeType = ANN::ModeType::PREDICT;
  gpuConfig.deviceType = ANN::DeviceType::GPU;
  gpuConfig.layersConfig = trainConfig.layersConfig;
  gpuConfig.parameters = params;
  gpuConfig.logLevel = ANN::LogLevel::ERROR;

  auto gpuCore = ANN::Core<float>::makeCore(gpuConfig);
  ANN::Output<float> gpuPred = gpuCore->predict({1.0f, 0.0f});

  std::cout << "  CPU=[" << cpuPred[0] << "," << cpuPred[1] << "]" << "  GPU=[" << gpuPred[0] << "," << gpuPred[1]
            << "]" << std::endl;

  CHECK_NEAR(cpuPred[0], gpuPred[0], 0.01f, "GPU vs CPU cross-entropy parity [0]");
  CHECK_NEAR(cpuPred[1], gpuPred[1], 0.01f, "GPU vs CPU cross-entropy parity [1]");

  // Both should sum to 1 (softmax)
  CHECK_NEAR(gpuPred[0] + gpuPred[1], 1.0f, 0.01f, "GPU softmax sums to 1");
}

//===================================================================================================================//

static void testGPUWeightedCrossEntropyTraining()
{
  std::cout << "--- testGPUWeightedCrossEntropyTraining ---" << std::endl;

  // Cross-entropy with per-class weights on GPU
  ANN::CoreConfig<float> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::GPU;
  config.layersConfig =
    makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {4, ANN::ActvFuncType::RELU}, {2, ANN::ActvFuncType::SOFTMAX}});

  config.costFunctionConfig.type = ANN::CostFunctionType::CROSS_ENTROPY;
  config.costFunctionConfig.weights = {5.0f, 1.0f};
  config.trainingConfig.numEpochs = 200;
  config.trainingConfig.learningRate = 0.1f;
  config.progressReports = 0;
  config.numGPUs = 1;
  config.logLevel = ANN::LogLevel::ERROR;

  ANN::Samples<float> samples = {{{1.0f, 0.0f}, {1.0f, 0.0f}}, {{0.0f, 1.0f}, {0.0f, 1.0f}}};

  auto core = ANN::Core<float>::makeCore(config);
  core->train(samples.size(), ANN::makeSampleProvider(samples));

  ANN::TestResult<float> result = core->test(samples.size(), ANN::makeSampleProvider(samples));

  CHECK(result.averageLoss >= 0.0f, "GPU weighted CE: loss non-negative");
  CHECK(std::isfinite(result.averageLoss), "GPU weighted CE: loss is finite");
  CHECK(result.numSamples == 2, "GPU weighted CE: 2 samples");

  const auto& cfc = core->getCostFunctionConfig();
  CHECK(cfc.type == ANN::CostFunctionType::CROSS_ENTROPY, "GPU weighted CE: type preserved");
  CHECK(cfc.weights.size() == 2, "GPU weighted CE: weights preserved");

  std::cout << "  GPU weighted CE avgLoss=" << result.averageLoss << std::endl;
}

//===================================================================================================================//

static void testGPUTestMethod()
{
  std::cout << "--- testGPUTestMethod ---" << std::endl;

  ANN::CoreConfig<float> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::GPU;
  config.layersConfig =
    makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {4, ANN::ActvFuncType::SIGMOID}, {1, ANN::ActvFuncType::SIGMOID}});

  config.trainingConfig.numEpochs = 500;
  config.trainingConfig.learningRate = 0.5f;
  config.progressReports = 0;
  config.numGPUs = 1;
  config.logLevel = ANN::LogLevel::ERROR;

  ANN::Samples<float> samples = {{{0.0f, 0.0f}, {0.0f}}, {{1.0f, 1.0f}, {1.0f}}};

  auto core = ANN::Core<float>::makeCore(config);
  core->train(samples.size(), ANN::makeSampleProvider(samples));

  ANN::TestResult<float> result = core->test(samples.size(), ANN::makeSampleProvider(samples));
  CHECK(result.numSamples == 2, "GPU test numSamples = 2");
  CHECK(result.averageLoss >= 0.0f, "GPU test averageLoss >= 0");
  CHECK(result.totalLoss >= 0.0f, "GPU test totalLoss >= 0");
  CHECK_NEAR(result.totalLoss, result.averageLoss * 2.0f, 1e-4f, "GPU totalLoss = avgLoss * numSamples");
  CHECK(result.numCorrect <= result.numSamples, "GPU numCorrect <= numSamples");
  CHECK(result.accuracy >= 0.0f && result.accuracy <= 100.0f, "GPU accuracy in [0, 100]");
  std::cout << "  GPU test avgLoss=" << result.averageLoss << " accuracy=" << result.accuracy << "%" << std::endl;
}

//===================================================================================================================//

static void testGPUTrainingMetadata()
{
  std::cout << "--- testGPUTrainingMetadata ---" << std::endl;

  ANN::CoreConfig<float> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::GPU;
  config.layersConfig = makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {1, ANN::ActvFuncType::SIGMOID}});
  config.trainingConfig.numEpochs = 10;
  config.trainingConfig.learningRate = 0.1f;
  config.progressReports = 0;
  config.numGPUs = 1;
  config.logLevel = ANN::LogLevel::ERROR;

  ANN::Samples<float> samples = {{{1.0f, 0.0f}, {1.0f}}};
  auto core = ANN::Core<float>::makeCore(config);
  core->train(samples.size(), ANN::makeSampleProvider(samples));

  const auto& meta = core->getTrainingMetadata();
  CHECK(!meta.startTime.empty(), "GPU startTime non-empty");
  CHECK(!meta.endTime.empty(), "GPU endTime non-empty");
  CHECK(meta.durationSeconds >= 0.0, "GPU durationSeconds >= 0");
  CHECK(!meta.durationFormatted.empty(), "GPU durationFormatted non-empty");
  CHECK(meta.numSamples == 1, "GPU numSamples = 1");
}

//===================================================================================================================//

static void testGPUPredictMetadata()
{
  std::cout << "--- testGPUPredictMetadata ---" << std::endl;

  ANN::CoreConfig<float> config;
  config.modeType = ANN::ModeType::PREDICT;
  config.deviceType = ANN::DeviceType::GPU;
  config.layersConfig = makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {1, ANN::ActvFuncType::SIGMOID}});
  config.logLevel = ANN::LogLevel::ERROR;

  auto core = ANN::Core<float>::makeCore(config);
  core->predict({1.0f, 0.0f});

  const auto& meta = core->getPredictMetadata();
  CHECK(!meta.startTime.empty(), "GPU predict startTime non-empty");
  CHECK(!meta.endTime.empty(), "GPU predict endTime non-empty");
  CHECK(meta.durationSeconds >= 0.0, "GPU predict durationSeconds >= 0");
}

void runGPUBasicTests2()
{
  testGPUCrossEntropyTraining();
  testGPUCrossEntropyCPUParity();
  testGPUWeightedCrossEntropyTraining();
  testGPUTestMethod();
  testGPUTrainingMetadata();
  testGPUPredictMetadata();
}
