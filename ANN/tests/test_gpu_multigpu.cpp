#include "test_helpers.hpp"

//===================================================================================================================//

//-- Multi-GPU Tests --//
//===================================================================================================================//

static void testMultiGPUTrainSimple()
{
  std::cout << "--- testMultiGPUTrainSimple ---" << std::endl;

  ANN::CoreConfig<float> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::GPU;
  config.layersConfig =
    makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {4, ANN::ActvFuncType::SIGMOID}, {1, ANN::ActvFuncType::SIGMOID}});
  config.trainingConfig.numEpochs = 500;
  config.trainingConfig.learningRate = 0.5f;
  config.progressReports = 0;
  config.numGPUs = 2;
  config.logLevel = ANN::LogLevel::ERROR;

  ANN::Samples<float> samples = {
    {{0.0f, 0.0f}, {0.0f}}, {{1.0f, 0.0f}, {1.0f}}, {{0.0f, 1.0f}, {1.0f}}, {{1.0f, 1.0f}, {0.0f}}};

  bool converged = false;

  for (int attempt = 0; attempt < 5 && !converged; ++attempt) {
    if (attempt > 0)
      std::cout << "  retry #" << attempt << std::endl;
    auto core = ANN::Core<float>::makeCore(config);
    core->train(samples.size(), ANN::makeSampleProvider(samples));
    auto high = core->predict({1.0f, 0.0f}).output[0];
    auto low = core->predict({0.0f, 0.0f}).output[0];

    if (high > 0.5f && low < 0.5f)
      converged = true;
  }

  CHECK(converged, "multi-GPU train converged (5 attempts)");
}

//===================================================================================================================//

static void testMultiGPUTestMethod()
{
  std::cout << "--- testMultiGPUTestMethod ---" << std::endl;

  ANN::CoreConfig<float> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::GPU;
  config.layersConfig =
    makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {4, ANN::ActvFuncType::SIGMOID}, {1, ANN::ActvFuncType::SIGMOID}});
  config.trainingConfig.numEpochs = 500;
  config.trainingConfig.learningRate = 0.5f;
  config.progressReports = 0;
  config.numGPUs = 2;
  config.logLevel = ANN::LogLevel::ERROR;

  ANN::Samples<float> samples = {{{0.0f, 0.0f}, {0.0f}}, {{1.0f, 1.0f}, {1.0f}}};

  auto core = ANN::Core<float>::makeCore(config);
  core->train(samples.size(), ANN::makeSampleProvider(samples));

  ANN::TestResult<float> result = core->test(samples.size(), ANN::makeSampleProvider(samples));
  CHECK(result.numSamples == 2, "multi-GPU test numSamples = 2");
  CHECK(result.averageLoss >= 0.0f, "multi-GPU test averageLoss >= 0");
  CHECK(std::isfinite(result.averageLoss), "multi-GPU test averageLoss finite");
  std::cout << "  multi-GPU test avgLoss=" << result.averageLoss << " accuracy=" << result.accuracy << "%" << std::endl;
}

//===================================================================================================================//

static void testMultiGPUCallback()
{
  std::cout << "--- testMultiGPUCallback ---" << std::endl;

  ANN::CoreConfig<float> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::GPU;
  config.layersConfig = makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {1, ANN::ActvFuncType::SIGMOID}});
  config.trainingConfig.numEpochs = 5;
  config.trainingConfig.learningRate = 0.1f;
  config.progressReports = 1;
  config.numGPUs = 2;
  config.logLevel = ANN::LogLevel::ERROR;

  ANN::Samples<float> samples = {{{1.0f, 0.0f}, {1.0f}}, {{0.0f, 1.0f}, {0.0f}}};

  int callbackCount = 0;
  bool sawGPU0 = false;
  bool sawGPU1 = false;

  auto core = ANN::Core<float>::makeCore(config);
  core->setTrainingCallback([&](const ANN::TrainingProgress<float>& progress) {
    callbackCount++;

    if (progress.gpuIndex == 0)
      sawGPU0 = true;

    if (progress.gpuIndex == 1)
      sawGPU1 = true;
  });

  core->train(samples.size(), ANN::makeSampleProvider(samples));

  std::cout << "  multi-GPU callback called " << callbackCount << " times, gpu0=" << sawGPU0 << " gpu1=" << sawGPU1
            << std::endl;
  CHECK(callbackCount > 0, "multi-GPU callback was called");
  CHECK(sawGPU0, "multi-GPU callback saw GPU 0");
  CHECK(sawGPU1, "multi-GPU callback saw GPU 1");
}

//===================================================================================================================//

static void testMultiGPUCrossEntropyTraining()
{
  std::cout << "--- testMultiGPUCrossEntropyTraining ---" << std::endl;

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
    config.numGPUs = 2;
    config.logLevel = ANN::LogLevel::ERROR;

    auto core = ANN::Core<float>::makeCore(config);
    core->train(samples.size(), ANN::makeSampleProvider(samples));

    auto out0 = core->predict({1.0f, 0.0f}).output;
    float sum0 = out0[0] + out0[1] + out0[2];

    bool sumsOk = std::fabs(sum0 - 1.0f) < 0.01f;
    bool classOk = out0[0] > out0[1] && out0[0] > out0[2];

    if (sumsOk && classOk)
      converged = true;
  }

  CHECK(converged, "multi-GPU cross-entropy converged (5 attempts)");
}

//===================================================================================================================//

static void testMultiGPUDifferentActivations()
{
  std::cout << "--- testMultiGPUDifferentActivations ---" << std::endl;

  std::vector<ANN::ActvFuncType> activations = {ANN::ActvFuncType::SIGMOID, ANN::ActvFuncType::RELU,
                                                ANN::ActvFuncType::TANH};

  for (auto actvType : activations) {
    std::string name = ANN::ActvFunc::typeToName(actvType);
    std::cout << "  testing " << name << std::endl;

    ANN::CoreConfig<float> config;
    config.modeType = ANN::ModeType::TRAIN;
    config.deviceType = ANN::DeviceType::GPU;
    config.layersConfig = makeLayersConfig({{2, actvType}, {1, ANN::ActvFuncType::SIGMOID}});
    config.trainingConfig.numEpochs = 100;
    config.trainingConfig.learningRate = 0.1f;
    config.progressReports = 0;
    config.numGPUs = 2;
    config.logLevel = ANN::LogLevel::ERROR;

    ANN::Samples<float> samples = {{{1.0f, 0.0f}, {1.0f}}, {{0.0f, 1.0f}, {0.0f}}};

    auto core = ANN::Core<float>::makeCore(config);
    core->train(samples.size(), ANN::makeSampleProvider(samples));

    auto out = core->predict({1.0f, 0.0f}).output;
    CHECK(std::isfinite(out[0]), ("multi-GPU " + name + ": output is finite").c_str());
  }
}

//===================================================================================================================//

static void testMultiGPUMultiOutput()
{
  std::cout << "--- testMultiGPUMultiOutput ---" << std::endl;

  ANN::CoreConfig<float> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::GPU;
  config.layersConfig =
    makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {4, ANN::ActvFuncType::RELU}, {3, ANN::ActvFuncType::SIGMOID}});
  config.trainingConfig.numEpochs = 200;
  config.trainingConfig.learningRate = 0.1f;
  config.progressReports = 0;
  config.numGPUs = 2;
  config.logLevel = ANN::LogLevel::ERROR;

  ANN::Samples<float> samples = {{{1.0f, 0.0f}, {1.0f, 0.0f, 0.5f}}, {{0.0f, 1.0f}, {0.0f, 1.0f, 0.5f}}};

  auto core = ANN::Core<float>::makeCore(config);
  core->train(samples.size(), ANN::makeSampleProvider(samples));

  auto out = core->predict({1.0f, 0.0f}).output;
  CHECK(out.size() == 3, "multi-GPU multi-output: 3 outputs");

  for (ulong i = 0; i < 3; i++)
    CHECK(std::isfinite(out[i]), "multi-GPU multi-output: output is finite");
}

//===================================================================================================================//

static void testMultiGPUDropoutTraining()
{
  std::cout << "--- testMultiGPUDropoutTraining ---" << std::endl;

  ANN::CoreConfig<float> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::GPU;
  config.layersConfig =
    makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {8, ANN::ActvFuncType::RELU}, {1, ANN::ActvFuncType::SIGMOID}});
  config.trainingConfig.numEpochs = 200;
  config.trainingConfig.learningRate = 0.1f;
  config.trainingConfig.dropoutRate = 0.3f;
  config.progressReports = 0;
  config.numGPUs = 2;
  config.logLevel = ANN::LogLevel::ERROR;

  ANN::Samples<float> samples = {{{1.0f, 0.0f}, {1.0f}}, {{0.0f, 1.0f}, {0.0f}}};

  auto core = ANN::Core<float>::makeCore(config);
  core->train(samples.size(), ANN::makeSampleProvider(samples));

  auto out = core->predict({1.0f, 0.0f}).output;
  CHECK(std::isfinite(out[0]), "multi-GPU dropout: output is finite");
  CHECK(out[0] >= 0.0f && out[0] <= 1.0f, "multi-GPU dropout: output in [0,1]");
}

//===================================================================================================================//

static void testMultiGPUParametersDuringTraining()
{
  std::cout << "--- testMultiGPUParametersDuringTraining ---" << std::endl;

  ANN::CoreConfig<float> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::GPU;
  config.layersConfig = makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {1, ANN::ActvFuncType::SIGMOID}});
  config.trainingConfig.numEpochs = 50;
  config.trainingConfig.learningRate = 0.1f;
  config.progressReports = 0;
  config.numGPUs = 2;
  config.logLevel = ANN::LogLevel::ERROR;

  ANN::Samples<float> samples = {{{1.0f, 0.0f}, {1.0f}}, {{0.0f, 1.0f}, {0.0f}}};

  auto core = ANN::Core<float>::makeCore(config);
  core->train(samples.size(), ANN::makeSampleProvider(samples));
  ANN::Parameters<float> paramsAfter = core->getParameters();

  CHECK(!paramsAfter.weights.empty(), "multi-GPU params: weights non-empty after training");
  CHECK(!paramsAfter.biases.empty(), "multi-GPU params: biases non-empty after training");

  // Verify trained params work in predict mode
  ANN::CoreConfig<float> predictConfig;
  predictConfig.modeType = ANN::ModeType::PREDICT;
  predictConfig.deviceType = ANN::DeviceType::GPU;
  predictConfig.layersConfig = config.layersConfig;
  predictConfig.parameters = paramsAfter;
  predictConfig.logLevel = ANN::LogLevel::ERROR;

  auto predictCore = ANN::Core<float>::makeCore(predictConfig);
  auto out = predictCore->predict({1.0f, 0.0f}).output;
  CHECK(std::isfinite(out[0]), "multi-GPU params: predict with trained params works");
}

//===================================================================================================================//

void runGPUMultiGPUTests()
{
  testMultiGPUTrainSimple();
  testMultiGPUTestMethod();
  testMultiGPUCallback();
  testMultiGPUCrossEntropyTraining();
  testMultiGPUDifferentActivations();
  testMultiGPUMultiOutput();
  testMultiGPUDropoutTraining();
  testMultiGPUParametersDuringTraining();
}
