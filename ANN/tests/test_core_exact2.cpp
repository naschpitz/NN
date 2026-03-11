#include "test_helpers.hpp"

static void testCrossEntropyGradientNumerical()
{
  std::cout << "--- testCrossEntropyGradientNumerical ---" << std::endl;

  // Verify the analytical gradient formula matches numerical finite differences.
  // This tests the MATH directly, independent of the network.
  // Loss = -sum(w_i * y_i * log(a_i))
  // dL/da_j = -w_j * y_j / a_j

  std::vector<double> activations = {0.7, 0.2, 0.1};
  std::vector<double> targets = {1.0, 0.0, 0.0};
  std::vector<double> weights = {1.0, 1.0, 1.0};

  auto computeLoss = [&](const std::vector<double>& a) {
    double loss = 0;
    const double epsilon = 1e-7;

    for (ulong i = 0; i < a.size(); i++) {
      double pred = std::max(a[i], epsilon);
      loss -= weights[i] * targets[i] * std::log(pred);
    }

    return loss;
  };

  double eps = 1e-6;

  for (ulong j = 0; j < activations.size(); j++) {
    // Analytical gradient
    const double epsilon = 1e-7;
    double pred = std::max(activations[j], epsilon);
    double analyticalGrad = -weights[j] * targets[j] / pred;

    // Numerical gradient via central differences
    std::vector<double> aPlus = activations;
    aPlus[j] += eps;
    std::vector<double> aMinus = activations;
    aMinus[j] -= eps;
    double numericalGrad = (computeLoss(aPlus) - computeLoss(aMinus)) / (2.0 * eps);

    std::cout << "  j=" << j << " analytical=" << analyticalGrad << " numerical=" << numericalGrad << std::endl;
    CHECK_NEAR(analyticalGrad, numericalGrad, 1e-4, "CE gradient numerical check");
  }

  // Also test with non-uniform weights (weighted cross-entropy)
  weights = {5.0, 1.0, 2.0};
  targets = {0.0, 1.0, 0.0};
  activations = {0.1, 0.6, 0.3};

  for (ulong j = 0; j < activations.size(); j++) {
    const double epsilon = 1e-7;
    double pred = std::max(activations[j], epsilon);
    double analyticalGrad = -weights[j] * targets[j] / pred;

    std::vector<double> aPlus = activations;
    aPlus[j] += eps;
    std::vector<double> aMinus = activations;
    aMinus[j] -= eps;
    double numericalGrad = (computeLoss(aPlus) - computeLoss(aMinus)) / (2.0 * eps);

    std::cout << "  weighted j=" << j << " analytical=" << analyticalGrad << " numerical=" << numericalGrad
              << std::endl;
    CHECK_NEAR(analyticalGrad, numericalGrad, 1e-4, "weighted CE gradient numerical check");
  }
}

//===================================================================================================================//

static void testCrossEntropyTraining()
{
  std::cout << "--- testCrossEntropyTraining ---" << std::endl;

  // Classification: 2 inputs → 3 classes with softmax + cross-entropy
  ANN::CoreConfig<double> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::CPU;
  config.layersConfig =
    makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {4, ANN::ActvFuncType::RELU}, {3, ANN::ActvFuncType::SOFTMAX}});

  config.costFunctionConfig.type = ANN::CostFunctionType::CROSS_ENTROPY;
  config.trainingConfig.numEpochs = 500;
  config.trainingConfig.learningRate = 0.1;
  config.progressReports = 0;
  config.logLevel = ANN::LogLevel::ERROR;

  // One-hot encoded targets
  ANN::Samples<double> samples = {
    {{1.0, 0.0}, {1.0, 0.0, 0.0}}, // class 0
    {{0.0, 1.0}, {0.0, 1.0, 0.0}}, // class 1
    {{1.0, 1.0}, {0.0, 0.0, 1.0}} // class 2
  };

  bool converged = false;

  for (int attempt = 0; attempt < 5 && !converged; ++attempt) {
    if (attempt > 0)
      std::cout << "  retry #" << attempt << std::endl;

    auto core = ANN::Core<double>::makeCore(config);
    core->train(samples.size(), ANN::makeSampleProvider(samples));

    auto out0 = core->predict({1.0, 0.0});
    auto out1 = core->predict({0.0, 1.0});
    auto out2 = core->predict({1.0, 1.0});

    // Each output should sum to 1 (softmax)
    bool sumsOk = std::fabs(out0[0] + out0[1] + out0[2] - 1.0) < 1e-5 &&
                  std::fabs(out1[0] + out1[1] + out1[2] - 1.0) < 1e-5 &&
                  std::fabs(out2[0] + out2[1] + out2[2] - 1.0) < 1e-5;

    // Correct class should have highest probability with > 0.5 confidence
    bool classOk = out0[0] > 0.5 && out1[1] > 0.5 && out2[2] > 0.5;

    if (sumsOk && classOk)
      converged = true;
  }

  CHECK(converged, "cross-entropy + softmax converged (5 attempts)");
}

//===================================================================================================================//

static void testCrossEntropyLossDecreases()
{
  std::cout << "--- testCrossEntropyLossDecreases ---" << std::endl;

  // Train with cross-entropy and verify loss ACTUALLY DECREASES — not just "is finite".
  ANN::CoreConfig<double> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::CPU;
  config.layersConfig =
    makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {4, ANN::ActvFuncType::RELU}, {2, ANN::ActvFuncType::SOFTMAX}});

  config.costFunctionConfig.type = ANN::CostFunctionType::CROSS_ENTROPY;
  config.trainingConfig.numEpochs = 50;
  config.trainingConfig.learningRate = 0.1;
  config.progressReports = 0;
  config.logLevel = ANN::LogLevel::ERROR;

  ANN::Samples<double> samples = {{{1.0, 0.0}, {1.0, 0.0}}, {{0.0, 1.0}, {0.0, 1.0}}};

  // Train 50 epochs, measure loss, train 50 more, measure again
  auto core = ANN::Core<double>::makeCore(config);
  core->train(samples.size(), ANN::makeSampleProvider(samples));
  ANN::TestResult<double> result1 = core->test(samples.size(), ANN::makeSampleProvider(samples));

  config.trainingConfig.numEpochs = 200;
  config.parameters = core->getParameters();
  auto core2 = ANN::Core<double>::makeCore(config);
  core2->train(samples.size(), ANN::makeSampleProvider(samples));
  ANN::TestResult<double> result2 = core2->test(samples.size(), ANN::makeSampleProvider(samples));

  std::cout << "  loss after 50 epochs=" << result1.averageLoss << "  after 250 epochs=" << result2.averageLoss
            << std::endl;

  CHECK(result2.averageLoss < result1.averageLoss, "CE loss decreases with more training");
  CHECK(result2.averageLoss < 0.5, "CE loss below 0.5 after 250 epochs on trivial problem");
  CHECK(result2.accuracy >= 50.0, "CE accuracy >= 50% after 250 epochs");

  std::cout << "  accuracy after 250 epochs=" << result2.accuracy << "%" << std::endl;
}

//===================================================================================================================//

static void testWeightedCrossEntropyTraining()
{
  std::cout << "--- testWeightedCrossEntropyTraining ---" << std::endl;

  // Cross-entropy with per-class weights
  ANN::CoreConfig<double> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::CPU;
  config.layersConfig =
    makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {4, ANN::ActvFuncType::RELU}, {2, ANN::ActvFuncType::SOFTMAX}});

  config.costFunctionConfig.type = ANN::CostFunctionType::CROSS_ENTROPY;
  config.costFunctionConfig.weights = {5.0, 1.0};
  config.trainingConfig.numEpochs = 200;
  config.trainingConfig.learningRate = 0.1;
  config.progressReports = 0;
  config.logLevel = ANN::LogLevel::ERROR;

  ANN::Samples<double> samples = {{{1.0, 0.0}, {1.0, 0.0}}, {{0.0, 1.0}, {0.0, 1.0}}};

  auto core = ANN::Core<double>::makeCore(config);
  core->train(samples.size(), ANN::makeSampleProvider(samples));

  ANN::TestResult<double> result = core->test(samples.size(), ANN::makeSampleProvider(samples));

  CHECK(result.averageLoss >= 0.0, "weighted cross-entropy loss non-negative");
  CHECK(std::isfinite(result.averageLoss), "weighted cross-entropy loss is finite");
  CHECK(result.numSamples == 2, "weighted cross-entropy: 2 samples");

  // Verify config preserved
  const auto& cfc = core->getCostFunctionConfig();
  CHECK(cfc.type == ANN::CostFunctionType::CROSS_ENTROPY, "weighted CE: type preserved");
  CHECK(cfc.weights.size() == 2, "weighted CE: weights preserved");
  CHECK_NEAR(cfc.weights[0], 5.0, 1e-10, "weighted CE: weight[0] = 5.0");

  std::cout << "  weighted CE avgLoss=" << result.averageLoss << std::endl;
}

//===================================================================================================================//

void runCoreExactTests2()
{
  testCrossEntropyGradientNumerical();
  testCrossEntropyTraining();
  testCrossEntropyLossDecreases();
  testWeightedCrossEntropyTraining();
}
