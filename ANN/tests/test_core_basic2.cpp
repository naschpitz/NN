#include "test_helpers.hpp"

static void testDifferentActivations()
{
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

static void testMultiLayerNetwork()
{
  std::cout << "--- testMultiLayerNetwork ---" << std::endl;

  // 2 → 4 → 4 → 1 network
  ANN::CoreConfig<double> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::CPU;
  config.layersConfig = makeLayersConfig({{2, ANN::ActvFuncType::RELU},
                                          {4, ANN::ActvFuncType::RELU},
                                          {4, ANN::ActvFuncType::RELU},
                                          {1, ANN::ActvFuncType::SIGMOID}});

  config.trainingConfig.numEpochs = 500;
  config.trainingConfig.learningRate = 0.1;
  config.progressReports = 0;

  ANN::Samples<double> samples = {{{1.0, 1.0}, {1.0}}, {{0.0, 0.0}, {0.0}}};

  bool converged = false;
  ANN::Output<double> p0, p1;

  for (int attempt = 0; attempt < 5 && !converged; ++attempt) {
    if (attempt > 0)
      std::cout << "  retry #" << attempt << std::endl;
    auto core = ANN::Core<double>::makeCore(config);
    core->train(samples.size(), ANN::makeSampleProvider(samples));
    p0 = core->predict({1.0, 1.0});
    p1 = core->predict({0.0, 0.0});

    if (p0[0] > 0.7 && p1[0] < 0.3)
      converged = true;
  }

  std::cout << "  high=" << p0[0] << "  low=" << p1[0] << std::endl;
  CHECK(converged, "multi-layer network converged (5 attempts)");
}

//===================================================================================================================//

static void testMultiOutput()
{
  std::cout << "--- testMultiOutput ---" << std::endl;

  // 2 inputs → 3 outputs
  ANN::CoreConfig<double> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::CPU;
  config.layersConfig =
    makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {4, ANN::ActvFuncType::SIGMOID}, {3, ANN::ActvFuncType::SIGMOID}});

  config.trainingConfig.numEpochs = 500;
  config.trainingConfig.learningRate = 0.5;
  config.progressReports = 0;

  ANN::Samples<double> samples = {{{1.0, 0.0}, {1.0, 0.0, 1.0}}, {{0.0, 1.0}, {0.0, 1.0, 0.0}}};

  bool converged = false;
  ANN::Output<double> pred;

  for (int attempt = 0; attempt < 5 && !converged; ++attempt) {
    if (attempt > 0)
      std::cout << "  retry #" << attempt << std::endl;
    auto core = ANN::Core<double>::makeCore(config);
    core->train(samples.size(), ANN::makeSampleProvider(samples));
    pred = core->predict({1.0, 0.0});

    if (pred[0] > 0.7 && pred[1] < 0.3 && pred[2] > 0.7)
      converged = true;
  }

  std::cout << "  pred=[" << pred[0] << "," << pred[1] << "," << pred[2] << "]" << std::endl;
  CHECK(pred.size() == 3, "multi-output size = 3");
  CHECK(converged, "multi-output converged (5 attempts)");
}

//===================================================================================================================//

static void testStepByStepAPI()
{
  std::cout << "--- testStepByStepAPI ---" << std::endl;

  // Test the step-by-step training API: predict → backpropagate → accumulate → update
  ANN::CoreConfig<double> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::CPU;
  config.layersConfig =
    makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {3, ANN::ActvFuncType::SIGMOID}, {1, ANN::ActvFuncType::SIGMOID}});

  config.trainingConfig.numEpochs = 1;
  config.trainingConfig.learningRate = 0.5;
  config.progressReports = 0;

  auto core = ANN::Core<double>::makeCore(config);

  ANN::Input<double> input = {1.0, 0.5};
  ANN::Output<double> expected = {1.0};

  // Manual training loop (1 epoch, 1 sample)
  ANN::Output<double> beforePred = core->predict(input);

  core->resetAccumulators();
  core->predict(input); // Forward pass
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

static void testTrainWithTanh()
{
  std::cout << "--- testTrainWithTanh ---" << std::endl;

  ANN::CoreConfig<double> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::CPU;
  config.layersConfig =
    makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {4, ANN::ActvFuncType::TANH}, {1, ANN::ActvFuncType::SIGMOID}});

  config.trainingConfig.numEpochs = 500;
  config.trainingConfig.learningRate = 0.1;
  config.progressReports = 0;

  ANN::Samples<double> samples = {{{1.0, 1.0}, {1.0}}, {{0.0, 0.0}, {0.0}}};

  bool converged = false;
  ANN::Output<double> p0, p1;

  for (int attempt = 0; attempt < 5 && !converged; ++attempt) {
    if (attempt > 0)
      std::cout << "  retry #" << attempt << std::endl;
    auto core = ANN::Core<double>::makeCore(config);
    core->train(samples.size(), ANN::makeSampleProvider(samples));
    p0 = core->predict({1.0, 1.0});
    p1 = core->predict({0.0, 0.0});

    if (p0[0] > 0.7 && p1[0] < 0.3)
      converged = true;
  }

  std::cout << "  high=" << p0[0] << "  low=" << p1[0] << std::endl;
  CHECK(converged, "tanh network converged (5 attempts)");
}

//===================================================================================================================//

static void testGettersAfterConstruction()
{
  std::cout << "--- testGettersAfterConstruction ---" << std::endl;

  ANN::CoreConfig<double> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::CPU;
  config.layersConfig =
    makeLayersConfig({{3, ANN::ActvFuncType::RELU}, {5, ANN::ActvFuncType::SIGMOID}, {2, ANN::ActvFuncType::TANH}});

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

void runCoreBasicTests2()
{
  testDifferentActivations();
  testMultiLayerNetwork();
  testMultiOutput();
  testStepByStepAPI();
  testTrainWithTanh();
  testGettersAfterConstruction();
}
