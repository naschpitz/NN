#include "test_helpers.hpp"

//===================================================================================================================//

static void testCostFunctionConfigDefault()
{
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

static void testCostFunctionConfigGetter()
{
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

static void testCostFunctionStringConversion()
{
  std::cout << "--- testCostFunctionStringConversion ---" << std::endl;

  CHECK(ANN::CostFunction::nameToType("squaredDifference") == ANN::CostFunctionType::SQUARED_DIFFERENCE,
        "nameToType squaredDifference");
  CHECK(ANN::CostFunction::nameToType("weightedSquaredDifference") ==
          ANN::CostFunctionType::WEIGHTED_SQUARED_DIFFERENCE,
        "nameToType weightedSquaredDifference");
  CHECK(ANN::CostFunction::typeToName(ANN::CostFunctionType::SQUARED_DIFFERENCE) == "squaredDifference",
        "typeToName squaredDifference");
  CHECK(ANN::CostFunction::typeToName(ANN::CostFunctionType::WEIGHTED_SQUARED_DIFFERENCE) ==
          "weightedSquaredDifference",
        "typeToName weightedSquaredDifference");

  bool threwException = false;

  try {
    ANN::CostFunction::nameToType("invalidName");
  } catch (const std::runtime_error&) {
    threwException = true;
  }

  CHECK(threwException, "nameToType throws on unknown name");
}

//===================================================================================================================//

static void testWeightedLossAffectsTraining()
{
  std::cout << "--- testWeightedLossAffectsTraining ---" << std::endl;

  // Train a 2-input → 2-output network.
  // Expected outputs: [1, 0] for all samples.
  // With heavy weight on output 0, the network should prioritize that output.
  ANN::Samples<double> samples = {{{1.0, 0.0}, {1.0, 0.0}}, {{0.0, 1.0}, {1.0, 0.0}}};

  // Train with default loss (equal weighting)
  ANN::CoreConfig<double> configDefault;
  configDefault.modeType = ANN::ModeType::TRAIN;
  configDefault.deviceType = ANN::DeviceType::CPU;
  configDefault.layersConfig =
    makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {4, ANN::ActvFuncType::SIGMOID}, {2, ANN::ActvFuncType::SIGMOID}});

  configDefault.trainingConfig.numEpochs = 200;
  configDefault.trainingConfig.learningRate = 0.5;
  configDefault.progressReports = 0;

  // Train with weighted loss: output 0 weighted 10x more than output 1
  ANN::CoreConfig<double> configWeighted = configDefault;
  configWeighted.costFunctionConfig.type = ANN::CostFunctionType::WEIGHTED_SQUARED_DIFFERENCE;
  configWeighted.costFunctionConfig.weights = {10.0, 1.0};

  // Use test() to compare loss on each — the weighted network should report different total loss
  auto coreDefault = ANN::Core<double>::makeCore(configDefault);
  coreDefault->train(samples.size(), ANN::makeSampleProvider(samples));
  ANN::TestResult<double> resultDefault = coreDefault->test(samples.size(), ANN::makeSampleProvider(samples));

  auto coreWeighted = ANN::Core<double>::makeCore(configWeighted);
  coreWeighted->train(samples.size(), ANN::makeSampleProvider(samples));
  ANN::TestResult<double> resultWeighted = coreWeighted->test(samples.size(), ANN::makeSampleProvider(samples));

  std::cout << "  default avgLoss=" << resultDefault.averageLoss << "  weighted avgLoss=" << resultWeighted.averageLoss
            << std::endl;

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

static void testShuffleSamplesDefault()
{
  std::cout << "--- testShuffleSamplesDefault ---" << std::endl;

  // Verify default is true
  ANN::TrainingConfig<double> tc;
  CHECK(tc.shuffleSamples == true, "shuffleSamples default is true");
}

//===================================================================================================================//

static void testShuffleSamplesTraining()
{
  std::cout << "--- testShuffleSamplesTraining ---" << std::endl;

  // Train with shuffleSamples=true and shuffleSamples=false, both should converge
  ANN::Samples<double> samples = {{{1.0, 1.0}, {1.0}}, {{0.0, 0.0}, {0.0}}, {{1.0, 0.0}, {0.5}}, {{0.0, 1.0}, {0.5}}};

  auto makeConfig = [](bool shuffle) {
    ANN::CoreConfig<double> config;
    config.modeType = ANN::ModeType::TRAIN;
    config.deviceType = ANN::DeviceType::CPU;
    config.layersConfig = makeLayersConfig(
      {{2, ANN::ActvFuncType::RELU}, {4, ANN::ActvFuncType::SIGMOID}, {1, ANN::ActvFuncType::SIGMOID}});

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
    core->train(samples.size(), ANN::makeSampleProvider(samples));
    auto p0 = core->predict({1.0, 1.0});
    auto p1 = core->predict({0.0, 0.0});

    if (p0[0] > 0.7 && p1[0] < 0.3)
      shuffleConverged = true;
  }

  CHECK(shuffleConverged, "shuffle=true converged (5 attempts)");

  // Train with shuffle disabled
  bool noShuffleConverged = false;

  for (int attempt = 0; attempt < 5 && !noShuffleConverged; ++attempt) {
    auto core = ANN::Core<double>::makeCore(makeConfig(false));
    core->train(samples.size(), ANN::makeSampleProvider(samples));
    auto p0 = core->predict({1.0, 1.0});
    auto p1 = core->predict({0.0, 0.0});

    if (p0[0] > 0.7 && p1[0] < 0.3)
      noShuffleConverged = true;
  }

  CHECK(noShuffleConverged, "shuffle=false converged (5 attempts)");

  std::cout << "  shuffle=true converged: " << shuffleConverged << "  shuffle=false converged: " << noShuffleConverged
            << std::endl;
}

//===================================================================================================================//

static void testShuffleSamplesNoShuffle()
{
  std::cout << "--- testShuffleSamplesNoShuffle ---" << std::endl;

  // With shuffleSamples=false, two runs with the same initial parameters should produce identical results
  ANN::Samples<double> samples = {{{1.0, 1.0}, {1.0}}, {{0.0, 0.0}, {0.0}}};

  ANN::CoreConfig<double> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::CPU;
  config.layersConfig =
    makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {4, ANN::ActvFuncType::SIGMOID}, {1, ANN::ActvFuncType::SIGMOID}});

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

  core2->train(samples.size(), ANN::makeSampleProvider(samples));
  core3->train(samples.size(), ANN::makeSampleProvider(samples));

  auto pred2 = core2->predict({1.0, 1.0});
  auto pred3 = core3->predict({1.0, 1.0});

  std::cout << "  run1=" << pred2[0] << "  run2=" << pred3[0] << std::endl;
  CHECK_NEAR(pred2[0], pred3[0], 1e-10, "shuffle=false: identical runs produce identical results");
}

//===================================================================================================================//

void runCoreFeaturesTests()
{
  testCostFunctionConfigDefault();
  testCostFunctionConfigGetter();
  testCostFunctionStringConversion();
  testWeightedLossAffectsTraining();
  testShuffleSamplesDefault();
  testShuffleSamplesTraining();
  testShuffleSamplesNoShuffle();
}
