#include "test_helpers.hpp"
#include "ANN_CostFunctionConfig.hpp"

//===================================================================================================================//

static void testGPUTrainSimple()
{
  std::cout << "--- testGPUTrainSimple ---" << std::endl;

  // Simple 2→4→1 network trained on GPU with sigmoid
  ANN::Samples<float> samples = {{{1.0f, 1.0f}, {1.0f}}, {{0.0f, 0.0f}, {0.0f}}};

  bool converged = false;
  ANN::Output<float> p0, p1;

  for (int attempt = 0; attempt < 5 && !converged; ++attempt) {
    if (attempt > 0)
      std::cout << "  retry #" << attempt << std::endl;

    ANN::CoreConfig<float> config;
    config.modeType = ANN::ModeType::TRAIN;
    config.deviceType = ANN::DeviceType::GPU;
    config.layersConfig = makeLayersConfig(
      {{2, ANN::ActvFuncType::RELU}, {4, ANN::ActvFuncType::SIGMOID}, {1, ANN::ActvFuncType::SIGMOID}});

    config.trainingConfig.numEpochs = 500;
    config.trainingConfig.learningRate = 0.5f;
    config.progressReports = 0;
    config.numGPUs = 1;
    config.logLevel = ANN::LogLevel::ERROR;

    auto core = ANN::Core<float>::makeCore(config);
    core->train(samples.size(), ANN::makeSampleProvider(samples));

    p0 = core->predict({1.0f, 1.0f});
    p1 = core->predict({0.0f, 0.0f});

    if (p0[0] > 0.7f && p1[0] < 0.3f)
      converged = true;
  }

  std::cout << "  high=" << p0[0] << "  low=" << p1[0] << std::endl;
  CHECK(converged, "GPU train converged (5 attempts)");
}

//===================================================================================================================//

static void testGPUPredict()
{
  std::cout << "--- testGPUPredict ---" << std::endl;

  // Create a predict-only GPU core with known weights
  ANN::CoreConfig<float> config;
  config.modeType = ANN::ModeType::PREDICT;
  config.deviceType = ANN::DeviceType::GPU;
  config.layersConfig = makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {1, ANN::ActvFuncType::SIGMOID}});

  config.parameters.weights.resize(2);
  config.parameters.weights[1] = {{0.5f, 0.5f}};
  config.parameters.biases.resize(2);
  config.parameters.biases[1] = {0.0f};
  config.logLevel = ANN::LogLevel::ERROR;

  auto core = ANN::Core<float>::makeCore(config);
  ANN::Output<float> out = core->predict({1.0f, 1.0f});

  CHECK(out.size() == 1, "GPU predict output size = 1");
  // z = 0.5*1 + 0.5*1 = 1.0, sigmoid(1.0) ≈ 0.7311
  float expected = 1.0f / (1.0f + std::exp(-1.0f));
  CHECK_NEAR(out[0], expected, 0.01f, "GPU predict sigmoid(1.0) ≈ 0.7311");
  std::cout << "  pred=" << out[0] << "  expected=" << expected << std::endl;
}

//===================================================================================================================//

static void testGPUvsCPUParity()
{
  std::cout << "--- testGPUvsCPUParity ---" << std::endl;

  // Train on CPU, then create both CPU and GPU predict cores with same params
  ANN::CoreConfig<float> trainConfig;
  trainConfig.modeType = ANN::ModeType::TRAIN;
  trainConfig.deviceType = ANN::DeviceType::CPU;
  trainConfig.layersConfig =
    makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {4, ANN::ActvFuncType::SIGMOID}, {1, ANN::ActvFuncType::SIGMOID}});

  trainConfig.trainingConfig.numEpochs = 200;
  trainConfig.trainingConfig.learningRate = 0.5f;
  trainConfig.progressReports = 0;
  trainConfig.logLevel = ANN::LogLevel::ERROR;

  ANN::Samples<float> samples = {{{1.0f, 1.0f}, {1.0f}}, {{0.0f, 0.0f}, {0.0f}}};

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
  ANN::Output<float> cpuPred1 = cpuCore->predict({1.0f, 1.0f});
  ANN::Output<float> cpuPred2 = cpuCore->predict({0.0f, 0.0f});

  // GPU predict
  ANN::CoreConfig<float> gpuConfig;
  gpuConfig.modeType = ANN::ModeType::PREDICT;
  gpuConfig.deviceType = ANN::DeviceType::GPU;
  gpuConfig.layersConfig = trainConfig.layersConfig;
  gpuConfig.parameters = params;
  gpuConfig.logLevel = ANN::LogLevel::ERROR;

  auto gpuCore = ANN::Core<float>::makeCore(gpuConfig);
  ANN::Output<float> gpuPred1 = gpuCore->predict({1.0f, 1.0f});
  ANN::Output<float> gpuPred2 = gpuCore->predict({0.0f, 0.0f});

  std::cout << "  CPU[1,1]=" << cpuPred1[0] << "  GPU[1,1]=" << gpuPred1[0] << std::endl;
  std::cout << "  CPU[0,0]=" << cpuPred2[0] << "  GPU[0,0]=" << gpuPred2[0] << std::endl;

  CHECK_NEAR(cpuPred1[0], gpuPred1[0], 0.01f, "CPU vs GPU parity [1,1]");
  CHECK_NEAR(cpuPred2[0], gpuPred2[0], 0.01f, "CPU vs GPU parity [0,0]");
}

//===================================================================================================================//

static void testGPUShuffleSamples()
{
  std::cout << "--- testGPUShuffleSamples ---" << std::endl;

  // Verify GPU training works with both shuffle=true and shuffle=false
  ANN::Samples<float> samples = {
    {{1.0f, 1.0f}, {1.0f}}, {{0.0f, 0.0f}, {0.0f}}, {{1.0f, 0.0f}, {0.5f}}, {{0.0f, 1.0f}, {0.5f}}};

  auto makeConfig = [](bool shuffle) {
    ANN::CoreConfig<float> config;
    config.modeType = ANN::ModeType::TRAIN;
    config.deviceType = ANN::DeviceType::GPU;
    config.layersConfig = makeLayersConfig(
      {{2, ANN::ActvFuncType::RELU}, {4, ANN::ActvFuncType::SIGMOID}, {1, ANN::ActvFuncType::SIGMOID}});

    config.trainingConfig.numEpochs = 500;
    config.trainingConfig.learningRate = 0.5f;
    config.trainingConfig.shuffleSamples = shuffle;
    config.progressReports = 0;
    config.numGPUs = 1;
    config.logLevel = ANN::LogLevel::ERROR;
    return config;
  };

  bool shuffleConverged = false;

  for (int attempt = 0; attempt < 5 && !shuffleConverged; ++attempt) {
    if (attempt > 0)
      std::cout << "  retry #" << attempt << std::endl;
    auto core = ANN::Core<float>::makeCore(makeConfig(true));
    core->train(samples.size(), ANN::makeSampleProvider(samples));
    auto p0 = core->predict({1.0f, 1.0f});
    auto p1 = core->predict({0.0f, 0.0f});

    if (p0[0] > 0.7f && p1[0] < 0.3f)
      shuffleConverged = true;
  }

  CHECK(shuffleConverged, "GPU shuffle=true converged (5 attempts)");

  bool noShuffleConverged = false;

  for (int attempt = 0; attempt < 5 && !noShuffleConverged; ++attempt) {
    if (attempt > 0)
      std::cout << "  retry #" << attempt << std::endl;
    auto core = ANN::Core<float>::makeCore(makeConfig(false));
    core->train(samples.size(), ANN::makeSampleProvider(samples));
    auto p0 = core->predict({1.0f, 1.0f});
    auto p1 = core->predict({0.0f, 0.0f});

    if (p0[0] > 0.7f && p1[0] < 0.3f)
      noShuffleConverged = true;
  }

  CHECK(noShuffleConverged, "GPU shuffle=false converged (5 attempts)");

  std::cout << "  shuffle=true: " << shuffleConverged << "  shuffle=false: " << noShuffleConverged << std::endl;
}

//===================================================================================================================//

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
    auto high = core->predict({1.0f, 0.0f})[0];
    auto low = core->predict({0.0f, 0.0f})[0];

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

    auto out0 = core->predict({1.0f, 0.0f});
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

    auto out = core->predict({1.0f, 0.0f});
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

  auto out = core->predict({1.0f, 0.0f});
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

  auto out = core->predict({1.0f, 0.0f});
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
  auto out = predictCore->predict({1.0f, 0.0f});
  CHECK(std::isfinite(out[0]), "multi-GPU params: predict with trained params works");
}

//===================================================================================================================//

static void testGPUExactForwardBackwardSquaredDifference()
{
  std::cout << "--- testGPUExactForwardBackwardSquaredDifference ---" << std::endl;

  // Same hand-computed network as CPU test, but on GPU with float.
  // 2 inputs → 2 hidden (ReLU) → 1 output (sigmoid), squared-difference, SGD lr=1.0
  ANN::CoreConfig<float> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::GPU;
  config.numGPUs = 1;
  config.layersConfig =
    makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {2, ANN::ActvFuncType::RELU}, {1, ANN::ActvFuncType::SIGMOID}});
  config.trainingConfig.learningRate = 1.0f;
  config.trainingConfig.numEpochs = 1;
  config.trainingConfig.shuffleSamples = false;
  config.progressReports = 0;
  config.logLevel = ANN::LogLevel::ERROR;

  config.parameters.weights = {{}, {{0.1f, 0.2f}, {0.3f, 0.4f}}, {{0.5f, -0.3f}}};
  config.parameters.biases = {{}, {0.1f, -0.1f}, {0.0f}};

  // Train 1 epoch, 1 sample
  ANN::Samples<float> samples = {{{1.0f, 0.5f}, {1.0f}}};
  auto trainCore = ANN::Core<float>::makeCore(config);
  trainCore->train(samples.size(), ANN::makeSampleProvider(samples));

  const ANN::Parameters<float>& p = trainCore->getParameters();

  // Every weight and bias verified against GPU float output (tolerance 1e-6 for float precision)
  CHECK_NEAR(p.weights[1][0][0], 0.2230974585f, 1e-6, "GPU SD w1[0][0]");
  CHECK_NEAR(p.weights[1][0][1], 0.2615487278f, 1e-6, "GPU SD w1[0][1]");
  CHECK_NEAR(p.weights[1][1][0], 0.2261415422f, 1e-6, "GPU SD w1[1][0]");
  CHECK_NEAR(p.weights[1][1][1], 0.3630707562f, 1e-6, "GPU SD w1[1][1]");
  CHECK_NEAR(p.biases[1][0], 0.2230974585f, 1e-6, "GPU SD b1[0]");
  CHECK_NEAR(p.biases[1][1], -0.1738584787f, 1e-6, "GPU SD b1[1]");
  CHECK_NEAR(p.weights[2][0][0], 0.5738584995f, 1e-6, "GPU SD w2[0][0]");
  CHECK_NEAR(p.weights[2][0][1], -0.2015220523f, 1e-6, "GPU SD w2[0][1]");
  CHECK_NEAR(p.biases[2][0], 0.246194914f, 1e-6, "GPU SD b2[0]");
}

//===================================================================================================================//

static void testGPUExactForwardBackwardCrossEntropy()
{
  std::cout << "--- testGPUExactForwardBackwardCrossEntropy ---" << std::endl;

  // Same hand-computed network as CPU test, but on GPU with float.
  // 2 inputs → 2 hidden (ReLU) → 2 output (softmax), cross-entropy, SGD lr=1.0
  ANN::CoreConfig<float> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::GPU;
  config.numGPUs = 1;
  config.layersConfig =
    makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {2, ANN::ActvFuncType::RELU}, {2, ANN::ActvFuncType::SOFTMAX}});
  config.costFunctionConfig.type = ANN::CostFunctionType::CROSS_ENTROPY;
  config.trainingConfig.learningRate = 1.0f;
  config.trainingConfig.numEpochs = 1;
  config.trainingConfig.shuffleSamples = false;
  config.progressReports = 0;
  config.logLevel = ANN::LogLevel::ERROR;

  config.parameters.weights = {{}, {{0.1f, 0.2f}, {0.3f, 0.4f}}, {{0.5f, -0.3f}, {-0.2f, 0.6f}}};
  config.parameters.biases = {{}, {0.1f, -0.1f}, {0.0f, 0.0f}};

  // Train 1 epoch, 1 sample
  ANN::Samples<float> samples = {{{1.0f, 0.5f}, {1.0f, 0.0f}}};
  auto trainCore = ANN::Core<float>::makeCore(config);
  trainCore->train(samples.size(), ANN::makeSampleProvider(samples));

  const ANN::Parameters<float>& p = trainCore->getParameters();

  // Every weight and bias verified against GPU float output (tolerance 1e-6)
  CHECK_NEAR(p.weights[2][0][0], 0.6612289548f, 1e-6, "GPU CE w2[0][0]");
  CHECK_NEAR(p.weights[2][0][1], -0.08502806723f, 1e-6, "GPU CE w2[0][1]");
  CHECK_NEAR(p.weights[2][1][0], -0.3612289429f, 1e-6, "GPU CE w2[1][0]");
  CHECK_NEAR(p.weights[2][1][1], 0.3850280941f, 1e-6, "GPU CE w2[1][1]");
  CHECK_NEAR(p.biases[2][0], 0.5374298692f, 1e-6, "GPU CE b2[0]");
  CHECK_NEAR(p.biases[2][1], -0.5374298096f, 1e-6, "GPU CE b2[1]");

  CHECK_NEAR(p.weights[1][0][0], 0.4762008786f, 1e-6, "GPU CE w1[0][0]");
  CHECK_NEAR(p.weights[1][0][1], 0.3881004453f, 1e-6, "GPU CE w1[0][1]");
  CHECK_NEAR(p.weights[1][1][0], -0.1836868525f, 1e-6, "GPU CE w1[1][0]");
  CHECK_NEAR(p.weights[1][1][1], 0.1581565738f, 1e-6, "GPU CE w1[1][1]");
  CHECK_NEAR(p.biases[1][0], 0.4762008786f, 1e-6, "GPU CE b1[0]");
  CHECK_NEAR(p.biases[1][1], -0.5836868882f, 1e-6, "GPU CE b1[1]");
}

//===================================================================================================================//

static void testGPUExactForwardBackwardWeightedCrossEntropy()
{
  std::cout << "--- testGPUExactForwardBackwardWeightedCrossEntropy ---" << std::endl;

  // Same as CE test but with per-class weights [3.0, 0.5]
  ANN::CoreConfig<float> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::GPU;
  config.numGPUs = 1;
  config.layersConfig =
    makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {2, ANN::ActvFuncType::RELU}, {2, ANN::ActvFuncType::SOFTMAX}});
  config.costFunctionConfig.type = ANN::CostFunctionType::CROSS_ENTROPY;
  config.costFunctionConfig.weights = {3.0f, 0.5f};
  config.trainingConfig.learningRate = 1.0f;
  config.trainingConfig.numEpochs = 1;
  config.trainingConfig.shuffleSamples = false;
  config.progressReports = 0;
  config.logLevel = ANN::LogLevel::ERROR;

  config.parameters.weights = {{}, {{0.1f, 0.2f}, {0.3f, 0.4f}}, {{0.5f, -0.3f}, {-0.2f, 0.6f}}};
  config.parameters.biases = {{}, {0.1f, -0.1f}, {0.0f, 0.0f}};

  ANN::Samples<float> samples = {{{1.0f, 0.5f}, {1.0f, 0.0f}}};
  auto trainCore = ANN::Core<float>::makeCore(config);
  trainCore->train(samples.size(), ANN::makeSampleProvider(samples));

  const ANN::Parameters<float>& p = trainCore->getParameters();

  // Every weight and bias verified against GPU float output (tolerance 1e-6)
  CHECK_NEAR(p.weights[2][0][0], 0.983686924f, 1e-6, "GPU WCE w2[0][0]");
  CHECK_NEAR(p.weights[2][0][1], 0.3449158669f, 1e-6, "GPU WCE w2[0][1]");
  CHECK_NEAR(p.weights[2][1][0], -0.6836867929f, 1e-6, "GPU WCE w2[1][0]");
  CHECK_NEAR(p.weights[2][1][1], -0.04491573572f, 1e-6, "GPU WCE w2[1][1]");
  CHECK_NEAR(p.biases[2][0], 1.612289667f, 1e-6, "GPU WCE b2[0]");
  CHECK_NEAR(p.biases[2][1], -1.61228931f, 1e-6, "GPU WCE b2[1]");

  CHECK_NEAR(p.weights[1][0][0], 1.228602767f, 1e-6, "GPU WCE w1[0][0]");
  CHECK_NEAR(p.weights[1][0][1], 0.7643013597f, 1e-6, "GPU WCE w1[0][1]");
  CHECK_NEAR(p.weights[1][1][0], -1.151060581f, 1e-6, "GPU WCE w1[1][0]");
  CHECK_NEAR(p.weights[1][1][1], -0.3255302608f, 1e-6, "GPU WCE w1[1][1]");
  CHECK_NEAR(p.biases[1][0], 1.228602767f, 1e-6, "GPU WCE b1[0]");
  CHECK_NEAR(p.biases[1][1], -1.551060557f, 1e-6, "GPU WCE b1[1]");
}

//===================================================================================================================//

void runGPUTests()
{
  // Single-GPU tests
  testGPUTrainSimple();
  testGPUPredict();
  testGPUvsCPUParity();
  testGPUShuffleSamples();
  testGPUCrossEntropyTraining();
  testGPUCrossEntropyCPUParity();
  testGPUWeightedCrossEntropyTraining();
  testGPUTestMethod();
  testGPUTrainingMetadata();
  testGPUPredictMetadata();
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
  testGPUExactForwardBackwardSquaredDifference();
  testGPUExactForwardBackwardCrossEntropy();
  testGPUExactForwardBackwardWeightedCrossEntropy();

  // Multi-GPU tests
  testMultiGPUTrainSimple();
  testMultiGPUTestMethod();
  testMultiGPUCallback();
  testMultiGPUCrossEntropyTraining();
  testMultiGPUDifferentActivations();
  testMultiGPUMultiOutput();
  testMultiGPUDropoutTraining();
  testMultiGPUParametersDuringTraining();
}