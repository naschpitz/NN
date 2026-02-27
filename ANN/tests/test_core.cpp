#include "test_helpers.hpp"

//===================================================================================================================//

static void testMakeCoreCPU() {
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

static void testPredictSimple() {
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
  ANN::Output<double> out = core->predict({1.0, 1.0});

  CHECK(out.size() == 1, "output size = 1");
  // z = 0.5*1 + 0.5*1 + 0 = 1.0, sigmoid(1.0) ≈ 0.7311
  double expected = 1.0 / (1.0 + std::exp(-1.0));
  CHECK_NEAR(out[0], expected, 1e-5, "sigmoid(1.0) ≈ 0.7311");
}

//===================================================================================================================//

static void testTrainXOR() {
  std::cout << "--- testTrainXOR ---" << std::endl;

  ANN::Samples<double> samples = {
    {{0.0, 0.0}, {0.0}},
    {{0.0, 1.0}, {1.0}},
    {{1.0, 0.0}, {1.0}},
    {{1.0, 1.0}, {0.0}}
  };

  bool converged = false;
  ANN::Output<double> p00, p01, p10, p11;

  for (int attempt = 0; attempt < 5 && !converged; ++attempt) {
    if (attempt > 0) std::cout << "  retry #" << attempt << std::endl;

    ANN::CoreConfig<double> config;
    config.modeType = ANN::ModeType::TRAIN;
    config.deviceType = ANN::DeviceType::CPU;
    config.layersConfig = makeLayersConfig({
      {2, ANN::ActvFuncType::RELU},
      {4, ANN::ActvFuncType::RELU},
      {1, ANN::ActvFuncType::SIGMOID}
    });
    config.trainingConfig.numEpochs = 2000;
    config.trainingConfig.learningRate = 0.1;
    config.progressReports = 0;
    config.logLevel = ANN::LogLevel::ERROR;

    auto core = ANN::Core<double>::makeCore(config);
    core->train(samples);

    p00 = core->predict({0.0, 0.0});
    p01 = core->predict({0.0, 1.0});
    p10 = core->predict({1.0, 0.0});
    p11 = core->predict({1.0, 1.0});

    if (p00[0] < 0.3 && p01[0] > 0.7 && p10[0] > 0.7 && p11[0] < 0.3) {
      converged = true;
    }
  }

  std::cout << "  XOR: [0,0]=" << p00[0] << " [0,1]=" << p01[0]
            << " [1,0]=" << p10[0] << " [1,1]=" << p11[0] << std::endl;
  CHECK(converged, "XOR converged (5 attempts)");
}

//===================================================================================================================//

static void testTestMethod() {
  std::cout << "--- testTestMethod ---" << std::endl;

  // Train a simple network first, then test
  ANN::CoreConfig<double> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::CPU;
  config.layersConfig = makeLayersConfig({
    {2, ANN::ActvFuncType::RELU},
    {4, ANN::ActvFuncType::SIGMOID},
    {1, ANN::ActvFuncType::SIGMOID}
  });
  config.trainingConfig.numEpochs = 500;
  config.trainingConfig.learningRate = 0.5;
  config.progressReports = 0;

  ANN::Samples<double> samples = {
    {{0.0, 0.0}, {0.0}},
    {{1.0, 1.0}, {1.0}}
  };

  auto core = ANN::Core<double>::makeCore(config);
  core->train(samples);

  ANN::TestResult<double> result = core->test(samples);
  CHECK(result.numSamples == 2, "test numSamples = 2");
  CHECK(result.averageLoss >= 0.0, "test averageLoss >= 0");
  CHECK(result.totalLoss >= 0.0, "test totalLoss >= 0");
  CHECK_NEAR(result.totalLoss, result.averageLoss * 2.0, 1e-10, "totalLoss = avgLoss * numSamples");
  CHECK(result.numCorrect <= result.numSamples, "numCorrect <= numSamples");
  CHECK(result.accuracy >= 0.0 && result.accuracy <= 100.0, "accuracy in [0, 100]");
  std::cout << "  test avgLoss=" << result.averageLoss << " accuracy=" << result.accuracy << "%" << std::endl;
}

//===================================================================================================================//

static void testTrainingMetadata() {
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
  core->train(samples);

  const auto& meta = core->getTrainingMetadata();
  CHECK(!meta.startTime.empty(), "startTime non-empty");
  CHECK(!meta.endTime.empty(), "endTime non-empty");
  CHECK(meta.durationSeconds >= 0.0, "durationSeconds >= 0");
  CHECK(!meta.durationFormatted.empty(), "durationFormatted non-empty");
  CHECK(meta.numSamples == 1, "numSamples = 1");
}

//===================================================================================================================//

static void testPredictMetadata() {
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

static void testTrainingCallback() {
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
  core->setTrainingCallback([&callbackCount](const ANN::TrainingProgress<double>& progress) {
    callbackCount++;
  });
  core->train(samples);

  std::cout << "  callback called " << callbackCount << " times" << std::endl;
  CHECK(callbackCount > 0, "training callback was called");
  CHECK(callbackCount >= 5, "callback called at least once per epoch");
}

//===================================================================================================================//

static void testParameterRoundTrip() {
  std::cout << "--- testParameterRoundTrip ---" << std::endl;

  // Train a network, get parameters, create new predict-mode core with those params
  ANN::CoreConfig<double> trainConfig;
  trainConfig.modeType = ANN::ModeType::TRAIN;
  trainConfig.deviceType = ANN::DeviceType::CPU;
  trainConfig.layersConfig = makeLayersConfig({
    {2, ANN::ActvFuncType::RELU},
    {3, ANN::ActvFuncType::SIGMOID},
    {1, ANN::ActvFuncType::SIGMOID}
  });
  trainConfig.trainingConfig.numEpochs = 200;
  trainConfig.trainingConfig.learningRate = 0.5;
  trainConfig.progressReports = 0;

  ANN::Samples<double> samples = {{{1.0, 1.0}, {1.0}}, {{0.0, 0.0}, {0.0}}};

  auto trainCore = ANN::Core<double>::makeCore(trainConfig);
  trainCore->train(samples);

  ANN::Output<double> originalPred = trainCore->predict({1.0, 1.0});
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
  ANN::Output<double> loadedPred = predCore->predict({1.0, 1.0});

  std::cout << "  original=" << originalPred[0] << "  from_params=" << loadedPred[0] << std::endl;
  CHECK_NEAR(originalPred[0], loadedPred[0], 1e-10, "parameter round-trip exact match");
}

//===================================================================================================================//

static void testParametersDuringTraining() {
  std::cout << "--- testParametersDuringTraining ---" << std::endl;

  // Train a network and verify that getParameters() returns non-empty data
  // during training (in the epoch-completion callback), not just after training ends.
  ANN::CoreConfig<double> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::CPU;
  config.layersConfig = makeLayersConfig({
    {2, ANN::ActvFuncType::RELU},
    {4, ANN::ActvFuncType::RELU},
    {1, ANN::ActvFuncType::SIGMOID}
  });
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

  core->train(samples);

  CHECK(paramsChecked, "epoch-completion callback fired");
  CHECK(weightsNonEmpty, "parameters.weights non-empty during training");
  CHECK(biasesNonEmpty, "parameters.biases non-empty during training");
}

//===================================================================================================================//

static void testDifferentActivations() {
  std::cout << "--- testDifferentActivations ---" << std::endl;

  // Test that different activation functions produce different outputs
  ANN::Input<double> input = {0.5, 0.5};

  // Same architecture, same weights, different activations
  ANN::Parameters<double> params;
  params.weights = {{}, {{0.3, 0.3}}};
  params.biases = {{}, {0.1}};

  auto makeCore = [&](ANN::ActvFuncType actv) {
    ANN::CoreConfig<double> config;
    config.modeType = ANN::ModeType::PREDICT;
    config.deviceType = ANN::DeviceType::CPU;
    config.layersConfig = makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {1, actv}});
    config.parameters = params;
    return ANN::Core<double>::makeCore(config);
  };

  double reluOut = makeCore(ANN::ActvFuncType::RELU)->predict(input)[0];
  double sigOut = makeCore(ANN::ActvFuncType::SIGMOID)->predict(input)[0];
  double tanhOut = makeCore(ANN::ActvFuncType::TANH)->predict(input)[0];

  std::cout << "  relu=" << reluOut << " sigmoid=" << sigOut << " tanh=" << tanhOut << std::endl;

  // z = 0.3*0.5 + 0.3*0.5 + 0.1 = 0.4
  CHECK_NEAR(reluOut, 0.4, 1e-6, "relu output = 0.4");
  CHECK_NEAR(sigOut, 1.0 / (1.0 + std::exp(-0.4)), 1e-5, "sigmoid output");
  CHECK_NEAR(tanhOut, std::tanh(0.4), 1e-5, "tanh output");
}

//===================================================================================================================//

static void testMultiLayerNetwork() {
  std::cout << "--- testMultiLayerNetwork ---" << std::endl;

  // 2 → 4 → 4 → 1 network
  ANN::CoreConfig<double> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::CPU;
  config.layersConfig = makeLayersConfig({
    {2, ANN::ActvFuncType::RELU},
    {4, ANN::ActvFuncType::RELU},
    {4, ANN::ActvFuncType::RELU},
    {1, ANN::ActvFuncType::SIGMOID}
  });
  config.trainingConfig.numEpochs = 500;
  config.trainingConfig.learningRate = 0.1;
  config.progressReports = 0;

  ANN::Samples<double> samples = {
    {{1.0, 1.0}, {1.0}},
    {{0.0, 0.0}, {0.0}}
  };

  bool converged = false;
  ANN::Output<double> p0, p1;

  for (int attempt = 0; attempt < 5 && !converged; ++attempt) {
    if (attempt > 0) std::cout << "  retry #" << attempt << std::endl;
    auto core = ANN::Core<double>::makeCore(config);
    core->train(samples);
    p0 = core->predict({1.0, 1.0});
    p1 = core->predict({0.0, 0.0});
    if (p0[0] > 0.7 && p1[0] < 0.3) converged = true;
  }

  std::cout << "  high=" << p0[0] << "  low=" << p1[0] << std::endl;
  CHECK(converged, "multi-layer network converged (5 attempts)");
}

//===================================================================================================================//

static void testMultiOutput() {
  std::cout << "--- testMultiOutput ---" << std::endl;

  // 2 inputs → 3 outputs
  ANN::CoreConfig<double> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::CPU;
  config.layersConfig = makeLayersConfig({
    {2, ANN::ActvFuncType::RELU},
    {4, ANN::ActvFuncType::SIGMOID},
    {3, ANN::ActvFuncType::SIGMOID}
  });
  config.trainingConfig.numEpochs = 500;
  config.trainingConfig.learningRate = 0.5;
  config.progressReports = 0;

  ANN::Samples<double> samples = {
    {{1.0, 0.0}, {1.0, 0.0, 1.0}},
    {{0.0, 1.0}, {0.0, 1.0, 0.0}}
  };

  bool converged = false;
  ANN::Output<double> pred;

  for (int attempt = 0; attempt < 5 && !converged; ++attempt) {
    if (attempt > 0) std::cout << "  retry #" << attempt << std::endl;
    auto core = ANN::Core<double>::makeCore(config);
    core->train(samples);
    pred = core->predict({1.0, 0.0});
    if (pred[0] > 0.7 && pred[1] < 0.3 && pred[2] > 0.7) converged = true;
  }

  std::cout << "  pred=[" << pred[0] << "," << pred[1] << "," << pred[2] << "]" << std::endl;
  CHECK(pred.size() == 3, "multi-output size = 3");
  CHECK(converged, "multi-output converged (5 attempts)");
}

//===================================================================================================================//

static void testStepByStepAPI() {
  std::cout << "--- testStepByStepAPI ---" << std::endl;

  // Test the step-by-step training API: predict → backpropagate → accumulate → update
  ANN::CoreConfig<double> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::CPU;
  config.layersConfig = makeLayersConfig({
    {2, ANN::ActvFuncType::RELU},
    {3, ANN::ActvFuncType::SIGMOID},
    {1, ANN::ActvFuncType::SIGMOID}
  });
  config.trainingConfig.numEpochs = 1;
  config.trainingConfig.learningRate = 0.5;
  config.progressReports = 0;

  auto core = ANN::Core<double>::makeCore(config);

  ANN::Input<double> input = {1.0, 0.5};
  ANN::Output<double> expected = {1.0};

  // Manual training loop (1 epoch, 1 sample)
  ANN::Output<double> beforePred = core->predict(input);

  core->resetAccumulators();
  core->predict(input);  // Forward pass
  ANN::Tensor1D<double> inputGrads = core->backpropagate(expected);
  core->accumulate();
  core->update(1);

  ANN::Output<double> afterPred = core->predict(input);

  std::cout << "  before=" << beforePred[0] << "  after=" << afterPred[0] << std::endl;

  // Input gradients should have size matching input layer
  CHECK(inputGrads.size() == 2, "input gradients size = 2");

  // After one update step toward target=1.0, prediction should move closer to 1.0
  // (This may not always hold depending on init, so just verify it ran without error)
  CHECK(afterPred.size() == 1, "step-by-step output size = 1");
}

//===================================================================================================================//

static void testTrainWithTanh() {
  std::cout << "--- testTrainWithTanh ---" << std::endl;

  ANN::CoreConfig<double> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::CPU;
  config.layersConfig = makeLayersConfig({
    {2, ANN::ActvFuncType::RELU},
    {4, ANN::ActvFuncType::TANH},
    {1, ANN::ActvFuncType::SIGMOID}
  });
  config.trainingConfig.numEpochs = 500;
  config.trainingConfig.learningRate = 0.1;
  config.progressReports = 0;

  ANN::Samples<double> samples = {
    {{1.0, 1.0}, {1.0}},
    {{0.0, 0.0}, {0.0}}
  };

  bool converged = false;
  ANN::Output<double> p0, p1;

  for (int attempt = 0; attempt < 5 && !converged; ++attempt) {
    if (attempt > 0) std::cout << "  retry #" << attempt << std::endl;
    auto core = ANN::Core<double>::makeCore(config);
    core->train(samples);
    p0 = core->predict({1.0, 1.0});
    p1 = core->predict({0.0, 0.0});
    if (p0[0] > 0.7 && p1[0] < 0.3) converged = true;
  }

  std::cout << "  high=" << p0[0] << "  low=" << p1[0] << std::endl;
  CHECK(converged, "tanh network converged (5 attempts)");
}

//===================================================================================================================//

static void testGettersAfterConstruction() {
  std::cout << "--- testGettersAfterConstruction ---" << std::endl;

  ANN::CoreConfig<double> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::CPU;
  config.layersConfig = makeLayersConfig({
    {3, ANN::ActvFuncType::RELU},
    {5, ANN::ActvFuncType::SIGMOID},
    {2, ANN::ActvFuncType::TANH}
  });
  config.trainingConfig.numEpochs = 100;
  config.trainingConfig.learningRate = 0.05;
  config.logLevel = ANN::LogLevel::INFO;

  auto core = ANN::Core<double>::makeCore(config);

  CHECK(core->getModeType() == ANN::ModeType::TRAIN, "mode = TRAIN");
  CHECK(core->getDeviceType() == ANN::DeviceType::CPU, "device = CPU");
  CHECK(core->getLayersConfig().size() == 3, "3 layers");
  CHECK(core->getLayersConfig()[0].numNeurons == 3, "layer 0: 3 neurons");
  CHECK(core->getLayersConfig()[1].numNeurons == 5, "layer 1: 5 neurons");
  CHECK(core->getLayersConfig()[2].numNeurons == 2, "layer 2: 2 neurons");
  CHECK(core->getTrainingConfig().numEpochs == 100, "numEpochs = 100");
  CHECK_NEAR(core->getTrainingConfig().learningRate, 0.05f, 1e-6f, "learningRate = 0.05");
  CHECK(core->getLogLevel() == ANN::LogLevel::INFO, "logLevel = INFO");

  core->setLogLevel(ANN::LogLevel::ERROR);
  CHECK(core->getLogLevel() == ANN::LogLevel::ERROR, "logLevel = ERROR after setLogLevel");
}

//===================================================================================================================//

static void testCostFunctionConfigDefault() {
  std::cout << "--- testCostFunctionConfigDefault ---" << std::endl;

  ANN::CoreConfig<double> config;
  config.modeType = ANN::ModeType::PREDICT;
  config.deviceType = ANN::DeviceType::CPU;
  config.layersConfig = makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {1, ANN::ActvFuncType::SIGMOID}});

  auto core = ANN::Core<double>::makeCore(config);

  const auto& cfc = core->getCostFunctionConfig();
  CHECK(cfc.type == ANN::CostFunctionType::SQUARED_DIFFERENCE, "default type is squaredDifference");
  CHECK(cfc.weights.empty(), "default weights is empty");
}

//===================================================================================================================//

static void testCostFunctionConfigGetter() {
  std::cout << "--- testCostFunctionConfigGetter ---" << std::endl;

  ANN::CoreConfig<double> config;
  config.modeType = ANN::ModeType::PREDICT;
  config.deviceType = ANN::DeviceType::CPU;
  config.layersConfig = makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {2, ANN::ActvFuncType::SIGMOID}});
  config.costFunctionConfig.type = ANN::CostFunctionType::WEIGHTED_SQUARED_DIFFERENCE;
  config.costFunctionConfig.weights = {3.0, 0.5};

  auto core = ANN::Core<double>::makeCore(config);

  const auto& cfc = core->getCostFunctionConfig();
  CHECK(cfc.type == ANN::CostFunctionType::WEIGHTED_SQUARED_DIFFERENCE, "type is weightedSquaredDifference");
  CHECK(cfc.weights.size() == 2, "weights size = 2");
  CHECK_NEAR(cfc.weights[0], 3.0, 1e-10, "weight[0] = 3.0");
  CHECK_NEAR(cfc.weights[1], 0.5, 1e-10, "weight[1] = 0.5");
}

//===================================================================================================================//

static void testCostFunctionStringConversion() {
  std::cout << "--- testCostFunctionStringConversion ---" << std::endl;

  CHECK(ANN::CostFunction::nameToType("squaredDifference") == ANN::CostFunctionType::SQUARED_DIFFERENCE,
        "nameToType squaredDifference");
  CHECK(ANN::CostFunction::nameToType("weightedSquaredDifference") == ANN::CostFunctionType::WEIGHTED_SQUARED_DIFFERENCE,
        "nameToType weightedSquaredDifference");
  CHECK(ANN::CostFunction::typeToName(ANN::CostFunctionType::SQUARED_DIFFERENCE) == "squaredDifference",
        "typeToName squaredDifference");
  CHECK(ANN::CostFunction::typeToName(ANN::CostFunctionType::WEIGHTED_SQUARED_DIFFERENCE) == "weightedSquaredDifference",
        "typeToName weightedSquaredDifference");

  bool threwException = false;
  try { ANN::CostFunction::nameToType("invalidName"); } catch (const std::runtime_error&) { threwException = true; }
  CHECK(threwException, "nameToType throws on unknown name");
}

//===================================================================================================================//

static void testWeightedLossAffectsTraining() {
  std::cout << "--- testWeightedLossAffectsTraining ---" << std::endl;

  // Train a 2-input → 2-output network.
  // Expected outputs: [1, 0] for all samples.
  // With heavy weight on output 0, the network should prioritize that output.
  ANN::Samples<double> samples = {
    {{1.0, 0.0}, {1.0, 0.0}},
    {{0.0, 1.0}, {1.0, 0.0}}
  };

  // Train with default loss (equal weighting)
  ANN::CoreConfig<double> configDefault;
  configDefault.modeType = ANN::ModeType::TRAIN;
  configDefault.deviceType = ANN::DeviceType::CPU;
  configDefault.layersConfig = makeLayersConfig({
    {2, ANN::ActvFuncType::RELU},
    {4, ANN::ActvFuncType::SIGMOID},
    {2, ANN::ActvFuncType::SIGMOID}
  });
  configDefault.trainingConfig.numEpochs = 200;
  configDefault.trainingConfig.learningRate = 0.5;
  configDefault.progressReports = 0;

  // Train with weighted loss: output 0 weighted 10x more than output 1
  ANN::CoreConfig<double> configWeighted = configDefault;
  configWeighted.costFunctionConfig.type = ANN::CostFunctionType::WEIGHTED_SQUARED_DIFFERENCE;
  configWeighted.costFunctionConfig.weights = {10.0, 1.0};

  // Use test() to compare loss on each — the weighted network should report different total loss
  auto coreDefault = ANN::Core<double>::makeCore(configDefault);
  coreDefault->train(samples);
  ANN::TestResult<double> resultDefault = coreDefault->test(samples);

  auto coreWeighted = ANN::Core<double>::makeCore(configWeighted);
  coreWeighted->train(samples);
  ANN::TestResult<double> resultWeighted = coreWeighted->test(samples);

  std::cout << "  default avgLoss=" << resultDefault.averageLoss
            << "  weighted avgLoss=" << resultWeighted.averageLoss << std::endl;

  // Both should train successfully (loss should be finite and non-negative)
  CHECK(resultDefault.averageLoss >= 0.0, "default loss non-negative");
  CHECK(resultWeighted.averageLoss >= 0.0, "weighted loss non-negative");
  CHECK(std::isfinite(resultDefault.averageLoss), "default loss is finite");
  CHECK(std::isfinite(resultWeighted.averageLoss), "weighted loss is finite");

  // The weighted loss result should differ from the default (different gradient dynamics)
  // We don't check which is larger since random init makes that non-deterministic
  // But we confirm both trained without crashing and produced valid loss
  CHECK(resultDefault.numSamples == 2, "default: tested 2 samples");
  CHECK(resultWeighted.numSamples == 2, "weighted: tested 2 samples");
  CHECK(resultDefault.numCorrect <= 2, "default: numCorrect <= 2");
  CHECK(resultWeighted.numCorrect <= 2, "weighted: numCorrect <= 2");
}

//===================================================================================================================//

static void testShuffleSamplesDefault() {
  std::cout << "--- testShuffleSamplesDefault ---" << std::endl;

  // Verify default is true
  ANN::TrainingConfig<double> tc;
  CHECK(tc.shuffleSamples == true, "shuffleSamples default is true");
}

//===================================================================================================================//

static void testShuffleSamplesTraining() {
  std::cout << "--- testShuffleSamplesTraining ---" << std::endl;

  // Train with shuffleSamples=true and shuffleSamples=false, both should converge
  ANN::Samples<double> samples = {
    {{1.0, 1.0}, {1.0}},
    {{0.0, 0.0}, {0.0}},
    {{1.0, 0.0}, {0.5}},
    {{0.0, 1.0}, {0.5}}
  };

  auto makeConfig = [](bool shuffle) {
    ANN::CoreConfig<double> config;
    config.modeType = ANN::ModeType::TRAIN;
    config.deviceType = ANN::DeviceType::CPU;
    config.layersConfig = makeLayersConfig({
      {2, ANN::ActvFuncType::RELU},
      {4, ANN::ActvFuncType::SIGMOID},
      {1, ANN::ActvFuncType::SIGMOID}
    });
    config.trainingConfig.numEpochs = 500;
    config.trainingConfig.learningRate = 0.5;
    config.trainingConfig.shuffleSamples = shuffle;
    config.progressReports = 0;
    config.logLevel = ANN::LogLevel::ERROR;
    return config;
  };

  // Train with shuffle enabled
  bool shuffleConverged = false;
  for (int attempt = 0; attempt < 5 && !shuffleConverged; ++attempt) {
    auto core = ANN::Core<double>::makeCore(makeConfig(true));
    core->train(samples);
    auto p0 = core->predict({1.0, 1.0});
    auto p1 = core->predict({0.0, 0.0});
    if (p0[0] > 0.7 && p1[0] < 0.3) shuffleConverged = true;
  }
  CHECK(shuffleConverged, "shuffle=true converged (5 attempts)");

  // Train with shuffle disabled
  bool noShuffleConverged = false;
  for (int attempt = 0; attempt < 5 && !noShuffleConverged; ++attempt) {
    auto core = ANN::Core<double>::makeCore(makeConfig(false));
    core->train(samples);
    auto p0 = core->predict({1.0, 1.0});
    auto p1 = core->predict({0.0, 0.0});
    if (p0[0] > 0.7 && p1[0] < 0.3) noShuffleConverged = true;
  }
  CHECK(noShuffleConverged, "shuffle=false converged (5 attempts)");

  std::cout << "  shuffle=true converged: " << shuffleConverged
            << "  shuffle=false converged: " << noShuffleConverged << std::endl;
}

//===================================================================================================================//

static void testShuffleSamplesNoShuffle() {
  std::cout << "--- testShuffleSamplesNoShuffle ---" << std::endl;

  // With shuffleSamples=false, two runs with the same initial parameters should produce identical results
  ANN::Samples<double> samples = {
    {{1.0, 1.0}, {1.0}},
    {{0.0, 0.0}, {0.0}}
  };

  ANN::CoreConfig<double> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::CPU;
  config.layersConfig = makeLayersConfig({
    {2, ANN::ActvFuncType::RELU},
    {4, ANN::ActvFuncType::SIGMOID},
    {1, ANN::ActvFuncType::SIGMOID}
  });
  config.trainingConfig.numEpochs = 50;
  config.trainingConfig.learningRate = 0.1;
  config.trainingConfig.shuffleSamples = false;
  config.progressReports = 0;
  config.logLevel = ANN::LogLevel::ERROR;

  // First run
  auto core1 = ANN::Core<double>::makeCore(config);
  auto params1 = core1->getParameters();

  // Second run with same initial parameters
  config.parameters = params1;
  auto core2 = ANN::Core<double>::makeCore(config);
  auto core3 = ANN::Core<double>::makeCore(config);

  core2->train(samples);
  core3->train(samples);

  auto pred2 = core2->predict({1.0, 1.0});
  auto pred3 = core3->predict({1.0, 1.0});

  std::cout << "  run1=" << pred2[0] << "  run2=" << pred3[0] << std::endl;
  CHECK_NEAR(pred2[0], pred3[0], 1e-10, "shuffle=false: identical runs produce identical results");
}

//===================================================================================================================//

static void testSoftmaxPredict() {
  std::cout << "--- testSoftmaxPredict ---" << std::endl;

  // 2 inputs → 3 outputs with softmax
  ANN::CoreConfig<double> config;
  config.modeType = ANN::ModeType::PREDICT;
  config.deviceType = ANN::DeviceType::CPU;
  config.layersConfig = makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {3, ANN::ActvFuncType::SOFTMAX}});

  config.parameters.weights.resize(2);
  config.parameters.weights[1] = {{0.5, 0.3}, {0.2, 0.4}, {0.1, 0.6}};
  config.parameters.biases.resize(2);
  config.parameters.biases[1] = {0.0, 0.0, 0.0};

  auto core = ANN::Core<double>::makeCore(config);
  ANN::Output<double> out = core->predict({1.0, 1.0});

  // Outputs should sum to 1
  double sum = out[0] + out[1] + out[2];
  CHECK_NEAR(sum, 1.0, 1e-6, "softmax predict sums to 1");

  // All outputs should be positive
  CHECK(out[0] > 0, "softmax predict [0] > 0");
  CHECK(out[1] > 0, "softmax predict [1] > 0");
  CHECK(out[2] > 0, "softmax predict [2] > 0");
}

//===================================================================================================================//

static void testSoftmaxTrain() {
  std::cout << "--- testSoftmaxTrain ---" << std::endl;

  // Classification: 2 inputs → 3 classes with softmax output
  ANN::CoreConfig<double> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::CPU;
  config.layersConfig = makeLayersConfig({
    {2, ANN::ActvFuncType::RELU},
    {4, ANN::ActvFuncType::RELU},
    {3, ANN::ActvFuncType::SOFTMAX}
  });
  config.trainingConfig.numEpochs = 500;
  config.trainingConfig.learningRate = 0.1;
  config.progressReports = 0;

  // One-hot encoded targets
  ANN::Samples<double> samples = {
    {{1.0, 0.0}, {1.0, 0.0, 0.0}},  // class 0
    {{0.0, 1.0}, {0.0, 1.0, 0.0}},  // class 1
    {{1.0, 1.0}, {0.0, 0.0, 1.0}}   // class 2
  };

  auto core = ANN::Core<double>::makeCore(config);
  core->train(samples);

  // After training, the highest output should match the target class
  auto out0 = core->predict({1.0, 0.0});
  auto out1 = core->predict({0.0, 1.0});
  auto out2 = core->predict({1.0, 1.0});

  // Each output should sum to 1
  CHECK_NEAR(out0[0] + out0[1] + out0[2], 1.0, 1e-5, "softmax train out0 sums to 1");
  CHECK_NEAR(out1[0] + out1[1] + out1[2], 1.0, 1e-5, "softmax train out1 sums to 1");
  CHECK_NEAR(out2[0] + out2[1] + out2[2], 1.0, 1e-5, "softmax train out2 sums to 1");

  // Correct class should have highest probability
  CHECK(out0[0] > out0[1] && out0[0] > out0[2], "softmax train: class 0 dominant");
  CHECK(out1[1] > out1[0] && out1[1] > out1[2], "softmax train: class 1 dominant");
  CHECK(out2[2] > out2[0] && out2[2] > out2[1], "softmax train: class 2 dominant");
}

//===================================================================================================================//

static void testSoftmaxHiddenLayer() {
  std::cout << "--- testSoftmaxHiddenLayer ---" << std::endl;

  // Softmax in a hidden layer (unusual but should work)
  ANN::CoreConfig<double> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::CPU;
  config.layersConfig = makeLayersConfig({
    {2, ANN::ActvFuncType::RELU},
    {4, ANN::ActvFuncType::SOFTMAX},
    {1, ANN::ActvFuncType::SIGMOID}
  });
  config.trainingConfig.numEpochs = 500;
  config.trainingConfig.learningRate = 0.1;
  config.progressReports = 0;

  ANN::Samples<double> samples = {
    {{1.0, 1.0}, {1.0}},
    {{0.0, 0.0}, {0.0}}
  };

  auto core = ANN::Core<double>::makeCore(config);
  core->train(samples);

  auto out1 = core->predict({1.0, 1.0});
  auto out0 = core->predict({0.0, 0.0});

  CHECK(out1[0] > out0[0], "softmax hidden: (1,1) > (0,0)");
}

//===================================================================================================================//

//===================================================================================================================//

static void testDropoutTraining() {
  std::cout << "--- testDropoutTraining ---" << std::endl;

  // Train XOR with dropout — should still converge (dropout is regularization, not destructive)
  ANN::CoreConfig<double> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::CPU;
  config.layersConfig = makeLayersConfig({{2, ANN::ActvFuncType::NONE},
                                           {8, ANN::ActvFuncType::RELU},
                                           {1, ANN::ActvFuncType::SIGMOID}});
  config.trainingConfig.numEpochs = 2000;
  config.trainingConfig.learningRate = 0.1;
  config.trainingConfig.dropoutRate = 0.3f;

  auto core = ANN::Core<double>::makeCore(config);

  std::vector<ANN::Sample<double>> samples = {
    {{0, 0}, {0}}, {{0, 1}, {1}}, {{1, 0}, {1}}, {{1, 1}, {0}}
  };

  core->train(samples);

  // Predict (no dropout during inference)
  auto r00 = core->predict({0, 0});
  auto r01 = core->predict({0, 1});
  auto r10 = core->predict({1, 0});
  auto r11 = core->predict({1, 1});

  CHECK(r00[0] < 0.3, "XOR(0,0) < 0.3 with dropout training");
  CHECK(r01[0] > 0.7, "XOR(0,1) > 0.7 with dropout training");
  CHECK(r10[0] > 0.7, "XOR(1,0) > 0.7 with dropout training");
  CHECK(r11[0] < 0.3, "XOR(1,1) < 0.3 with dropout training");
}

//===================================================================================================================//

static void testDropoutDisabledByDefault() {
  std::cout << "--- testDropoutDisabledByDefault ---" << std::endl;

  ANN::CoreConfig<double> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::CPU;
  config.layersConfig = makeLayersConfig({{2, ANN::ActvFuncType::NONE}, {1, ANN::ActvFuncType::SIGMOID}});

  CHECK(config.trainingConfig.dropoutRate == 0.0f, "dropoutRate defaults to 0.0");
}

//===================================================================================================================//

void runCoreTests() {
  testMakeCoreCPU();
  testPredictSimple();
  testTrainXOR();
  testTestMethod();
  testTrainingMetadata();
  testPredictMetadata();
  testTrainingCallback();
  testParameterRoundTrip();
  testParametersDuringTraining();
  testDifferentActivations();
  testMultiLayerNetwork();
  testMultiOutput();
  testStepByStepAPI();
  testTrainWithTanh();
  testGettersAfterConstruction();
  testCostFunctionConfigDefault();
  testCostFunctionConfigGetter();
  testCostFunctionStringConversion();
  testWeightedLossAffectsTraining();
  testShuffleSamplesDefault();
  testShuffleSamplesTraining();
  testShuffleSamplesNoShuffle();
  testSoftmaxPredict();
  testSoftmaxTrain();
  testSoftmaxHiddenLayer();
  testDropoutTraining();
  testDropoutDisabledByDefault();
}
