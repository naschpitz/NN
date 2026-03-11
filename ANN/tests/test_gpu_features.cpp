#include "test_helpers.hpp"

//===================================================================================================================//

static void testGPUTrainingCallback()
{
  std::cout << "--- testGPUTrainingCallback ---" << std::endl;

  ANN::CoreConfig<float> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::GPU;
  config.layersConfig = makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {1, ANN::ActvFuncType::SIGMOID}});
  config.trainingConfig.numEpochs = 5;
  config.trainingConfig.learningRate = 0.1f;
  config.progressReports = 1;
  config.numGPUs = 1;
  config.logLevel = ANN::LogLevel::ERROR;

  ANN::Samples<float> samples = {{{1.0f, 0.0f}, {1.0f}}, {{0.0f, 1.0f}, {0.0f}}};

  int callbackCount = 0;
  auto core = ANN::Core<float>::makeCore(config);
  core->setTrainingCallback([&callbackCount](const ANN::TrainingProgress<float>& progress) { callbackCount++; });

  core->train(samples.size(), ANN::makeSampleProvider(samples));

  std::cout << "  GPU callback called " << callbackCount << " times" << std::endl;
  CHECK(callbackCount > 0, "GPU training callback was called");
  CHECK(callbackCount >= 5, "GPU callback called at least once per epoch");
}

//===================================================================================================================//

static void testGPUParametersDuringTraining()
{
  std::cout << "--- testGPUParametersDuringTraining ---" << std::endl;

  ANN::CoreConfig<float> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::GPU;
  config.layersConfig = makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {1, ANN::ActvFuncType::SIGMOID}});
  config.trainingConfig.numEpochs = 50;
  config.trainingConfig.learningRate = 0.1f;
  config.progressReports = 0;
  config.numGPUs = 1;
  config.logLevel = ANN::LogLevel::ERROR;

  ANN::Samples<float> samples = {{{1.0f, 0.0f}, {1.0f}}, {{0.0f, 1.0f}, {0.0f}}};

  auto core = ANN::Core<float>::makeCore(config);
  core->train(samples.size(), ANN::makeSampleProvider(samples));
  ANN::Parameters<float> paramsAfter = core->getParameters();

  // After training, parameters should be populated and non-trivial
  CHECK(!paramsAfter.weights.empty(), "GPU params: weights non-empty after training");
  CHECK(!paramsAfter.biases.empty(), "GPU params: biases non-empty after training");

  // Verify we can use the trained parameters to create a predict core
  ANN::CoreConfig<float> predictConfig;
  predictConfig.modeType = ANN::ModeType::PREDICT;
  predictConfig.deviceType = ANN::DeviceType::GPU;
  predictConfig.layersConfig = config.layersConfig;
  predictConfig.parameters = paramsAfter;
  predictConfig.logLevel = ANN::LogLevel::ERROR;

  auto predictCore = ANN::Core<float>::makeCore(predictConfig);
  auto out = predictCore->predict({1.0f, 0.0f});
  CHECK(std::isfinite(out[0]), "GPU params: predict with trained params works");
}

//===================================================================================================================//

static void testGPUDifferentActivations()
{
  std::cout << "--- testGPUDifferentActivations ---" << std::endl;

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
    config.numGPUs = 1;
    config.logLevel = ANN::LogLevel::ERROR;

    ANN::Samples<float> samples = {{{1.0f, 0.0f}, {1.0f}}, {{0.0f, 1.0f}, {0.0f}}};

    auto core = ANN::Core<float>::makeCore(config);
    core->train(samples.size(), ANN::makeSampleProvider(samples));

    auto out = core->predict({1.0f, 0.0f});
    CHECK(std::isfinite(out[0]), ("GPU " + name + ": output is finite").c_str());
  }
}

//===================================================================================================================//

static void testGPUMultiLayerNetwork()
{
  std::cout << "--- testGPUMultiLayerNetwork ---" << std::endl;

  ANN::CoreConfig<float> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::GPU;
  config.layersConfig = makeLayersConfig({{3, ANN::ActvFuncType::RELU},
                                          {8, ANN::ActvFuncType::RELU},
                                          {4, ANN::ActvFuncType::RELU},
                                          {1, ANN::ActvFuncType::SIGMOID}});
  config.trainingConfig.numEpochs = 200;
  config.trainingConfig.learningRate = 0.1f;
  config.progressReports = 0;
  config.numGPUs = 1;
  config.logLevel = ANN::LogLevel::ERROR;

  ANN::Samples<float> samples = {
    {{1.0f, 0.0f, 0.0f}, {1.0f}}, {{0.0f, 1.0f, 0.0f}, {0.0f}}, {{0.0f, 0.0f, 1.0f}, {1.0f}}};

  auto core = ANN::Core<float>::makeCore(config);
  core->train(samples.size(), ANN::makeSampleProvider(samples));

  auto out = core->predict({1.0f, 0.0f, 0.0f});
  CHECK(std::isfinite(out[0]), "GPU multi-layer: output is finite");
  CHECK(out[0] >= 0.0f && out[0] <= 1.0f, "GPU multi-layer: output in [0,1]");
}

//===================================================================================================================//

static void testGPUMultiOutput()
{
  std::cout << "--- testGPUMultiOutput ---" << std::endl;

  ANN::CoreConfig<float> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::GPU;
  config.layersConfig =
    makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {4, ANN::ActvFuncType::RELU}, {3, ANN::ActvFuncType::SIGMOID}});
  config.trainingConfig.numEpochs = 200;
  config.trainingConfig.learningRate = 0.1f;
  config.progressReports = 0;
  config.numGPUs = 1;
  config.logLevel = ANN::LogLevel::ERROR;

  ANN::Samples<float> samples = {{{1.0f, 0.0f}, {1.0f, 0.0f, 0.5f}}, {{0.0f, 1.0f}, {0.0f, 1.0f, 0.5f}}};

  auto core = ANN::Core<float>::makeCore(config);
  core->train(samples.size(), ANN::makeSampleProvider(samples));

  auto out = core->predict({1.0f, 0.0f});
  CHECK(out.size() == 3, "GPU multi-output: 3 outputs");

  for (ulong i = 0; i < 3; i++)
    CHECK(std::isfinite(out[i]), "GPU multi-output: output is finite");
}

//===================================================================================================================//

static void testGPUWeightedLossAffectsTraining()
{
  std::cout << "--- testGPUWeightedLossAffectsTraining ---" << std::endl;

  auto makeConfig = [](std::vector<float> weights) {
    ANN::CoreConfig<float> config;
    config.modeType = ANN::ModeType::TRAIN;
    config.deviceType = ANN::DeviceType::GPU;
    config.layersConfig = makeLayersConfig(
      {{2, ANN::ActvFuncType::RELU}, {4, ANN::ActvFuncType::SIGMOID}, {2, ANN::ActvFuncType::SIGMOID}});
    config.costFunctionConfig.weights = weights;
    config.trainingConfig.numEpochs = 200;
    config.trainingConfig.learningRate = 0.1f;
    config.progressReports = 0;
    config.numGPUs = 1;
    config.logLevel = ANN::LogLevel::ERROR;
    return config;
  };

  ANN::Samples<float> samples = {{{1.0f, 0.0f}, {1.0f, 0.0f}}, {{0.0f, 1.0f}, {0.0f, 1.0f}}};

  auto config1 = makeConfig({10.0f, 1.0f});
  auto core1 = ANN::Core<float>::makeCore(config1);
  core1->train(samples.size(), ANN::makeSampleProvider(samples));
  auto out1 = core1->predict({1.0f, 0.0f});

  auto config2 = makeConfig({1.0f, 10.0f});
  auto core2 = ANN::Core<float>::makeCore(config2);
  core2->train(samples.size(), ANN::makeSampleProvider(samples));
  auto out2 = core2->predict({1.0f, 0.0f});

  bool different = std::fabs(out1[0] - out2[0]) > 0.01f || std::fabs(out1[1] - out2[1]) > 0.01f;
  CHECK(different, "GPU weighted loss: different weights produce different outputs");
}

//===================================================================================================================//

static void testGPUShuffleSamplesNoShuffle()
{
  std::cout << "--- testGPUShuffleSamplesNoShuffle ---" << std::endl;

  ANN::CoreConfig<float> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::GPU;
  config.layersConfig = makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {1, ANN::ActvFuncType::SIGMOID}});
  config.trainingConfig.numEpochs = 50;
  config.trainingConfig.learningRate = 0.1f;
  config.trainingConfig.shuffleSamples = false;
  config.progressReports = 0;
  config.numGPUs = 1;
  config.logLevel = ANN::LogLevel::ERROR;

  ANN::Samples<float> samples = {{{1.0f, 0.0f}, {1.0f}}, {{0.0f, 1.0f}, {0.0f}}};

  auto core = ANN::Core<float>::makeCore(config);
  core->train(samples.size(), ANN::makeSampleProvider(samples));

  auto out = core->predict({1.0f, 0.0f});
  CHECK(std::isfinite(out[0]), "GPU no-shuffle: output is finite");
}

//===================================================================================================================//

static void testGPUSoftmaxPredict()
{
  std::cout << "--- testGPUSoftmaxPredict ---" << std::endl;

  ANN::CoreConfig<float> config;
  config.modeType = ANN::ModeType::PREDICT;
  config.deviceType = ANN::DeviceType::GPU;
  config.layersConfig = makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {3, ANN::ActvFuncType::SOFTMAX}});
  config.logLevel = ANN::LogLevel::ERROR;

  auto core = ANN::Core<float>::makeCore(config);
  auto out = core->predict({1.0f, 0.5f});

  CHECK(out.size() == 3, "GPU softmax predict: 3 outputs");

  float sum = out[0] + out[1] + out[2];
  CHECK_NEAR(sum, 1.0f, 0.01f, "GPU softmax predict: sums to 1");

  for (ulong i = 0; i < 3; i++)
    CHECK(out[i] >= 0.0f, "GPU softmax predict: output >= 0");
}

//===================================================================================================================//

static void testGPUSoftmaxTrain()
{
  std::cout << "--- testGPUSoftmaxTrain ---" << std::endl;

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

    bool classOk = out0[0] > out0[1] && out0[0] > out0[2] && out1[1] > out1[0] && out1[1] > out1[2] &&
                   out2[2] > out2[0] && out2[2] > out2[1];

    if (classOk)
      converged = true;
  }

  CHECK(converged, "GPU softmax train: converged (5 attempts)");
}

//===================================================================================================================//

static void testGPUSoftmaxHiddenLayer()
{
  std::cout << "--- testGPUSoftmaxHiddenLayer ---" << std::endl;

  ANN::CoreConfig<float> config;
  config.modeType = ANN::ModeType::PREDICT;
  config.deviceType = ANN::DeviceType::GPU;
  config.layersConfig =
    makeLayersConfig({{2, ANN::ActvFuncType::SOFTMAX}, {4, ANN::ActvFuncType::RELU}, {1, ANN::ActvFuncType::SIGMOID}});
  config.logLevel = ANN::LogLevel::ERROR;

  auto core = ANN::Core<float>::makeCore(config);
  auto out = core->predict({1.0f, 0.5f});

  CHECK(out.size() == 1, "GPU softmax hidden: 1 output");
  CHECK(std::isfinite(out[0]), "GPU softmax hidden: output is finite");
}

//===================================================================================================================//

static void testGPUDropoutTraining()
{
  std::cout << "--- testGPUDropoutTraining ---" << std::endl;

  ANN::CoreConfig<float> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::GPU;
  config.layersConfig =
    makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {8, ANN::ActvFuncType::RELU}, {1, ANN::ActvFuncType::SIGMOID}});
  config.trainingConfig.numEpochs = 200;
  config.trainingConfig.learningRate = 0.1f;
  config.trainingConfig.dropoutRate = 0.3f;
  config.progressReports = 0;
  config.numGPUs = 1;
  config.logLevel = ANN::LogLevel::ERROR;

  ANN::Samples<float> samples = {{{1.0f, 0.0f}, {1.0f}}, {{0.0f, 1.0f}, {0.0f}}};

  auto core = ANN::Core<float>::makeCore(config);
  core->train(samples.size(), ANN::makeSampleProvider(samples));

  auto out = core->predict({1.0f, 0.0f});
  CHECK(std::isfinite(out[0]), "GPU dropout: output is finite");
  CHECK(out[0] >= 0.0f && out[0] <= 1.0f, "GPU dropout: output in [0,1]");
}

//===================================================================================================================//

static void testGPUCrossEntropyLossDecreases()
{
  std::cout << "--- testGPUCrossEntropyLossDecreases ---" << std::endl;

  ANN::CoreConfig<float> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::GPU;
  config.layersConfig =
    makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {4, ANN::ActvFuncType::RELU}, {2, ANN::ActvFuncType::SOFTMAX}});
  config.costFunctionConfig.type = ANN::CostFunctionType::CROSS_ENTROPY;
  config.trainingConfig.numEpochs = 50;
  config.trainingConfig.learningRate = 0.1f;
  config.progressReports = 0;
  config.numGPUs = 1;
  config.logLevel = ANN::LogLevel::ERROR;

  ANN::Samples<float> samples = {{{1.0f, 0.0f}, {1.0f, 0.0f}}, {{0.0f, 1.0f}, {0.0f, 1.0f}}};

  // Train 50 epochs, measure loss
  auto core = ANN::Core<float>::makeCore(config);
  core->train(samples.size(), ANN::makeSampleProvider(samples));
  ANN::TestResult<float> result1 = core->test(samples.size(), ANN::makeSampleProvider(samples));

  // Train 200 more epochs from same params
  config.trainingConfig.numEpochs = 200;
  config.parameters = core->getParameters();
  auto core2 = ANN::Core<float>::makeCore(config);
  core2->train(samples.size(), ANN::makeSampleProvider(samples));
  ANN::TestResult<float> result2 = core2->test(samples.size(), ANN::makeSampleProvider(samples));

  std::cout << "  GPU CE loss after 50=" << result1.averageLoss << "  after 250=" << result2.averageLoss << std::endl;
  // Loss should decrease OR already be very low (converged early)
  CHECK(result2.averageLoss <= result1.averageLoss || result2.averageLoss < 0.01f,
        "GPU CE loss decreases or already converged");
  CHECK(result2.averageLoss < 0.5f, "GPU CE loss below 0.5 after 250 epochs on trivial problem");
}

void runGPUFeaturesTests()
{
  testGPUTrainingCallback();
  testGPUParametersDuringTraining();
  testGPUDifferentActivations();
  testGPUMultiLayerNetwork();
  testGPUMultiOutput();
  testGPUWeightedLossAffectsTraining();
  testGPUShuffleSamplesNoShuffle();
  testGPUSoftmaxPredict();
  testGPUSoftmaxTrain();
  testGPUSoftmaxHiddenLayer();
  testGPUDropoutTraining();
  testGPUCrossEntropyLossDecreases();
}
