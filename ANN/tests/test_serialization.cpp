#include "test_helpers.hpp"

//===================================================================================================================//

static void testParameterRoundTrip() {
  std::cout << "--- testParameterRoundTrip ---" << std::endl;

  // Train a network
  ANN::CoreConfig<double> trainConfig;
  trainConfig.modeType = ANN::ModeType::TRAIN;
  trainConfig.deviceType = ANN::DeviceType::CPU;
  trainConfig.layersConfig = makeLayersConfig({
    {2, ANN::ActvFuncType::RELU},
    {4, ANN::ActvFuncType::SIGMOID},
    {1, ANN::ActvFuncType::SIGMOID}
  });
  trainConfig.trainingConfig.numEpochs = 100;
  trainConfig.trainingConfig.learningRate = 0.5;
  trainConfig.progressReports = 0;

  ANN::Samples<double> samples = {{{1.0, 1.0}, {1.0}}, {{0.0, 0.0}, {0.0}}};

  auto trainCore = ANN::Core<double>::makeCore(trainConfig);
  trainCore->train(samples);

  ANN::Output<double> originalPred = trainCore->predict({1.0, 1.0});

  // Extract trained parameters via getters
  const ANN::Parameters<double>& trainedParams = trainCore->getParameters();

  // Create a new Core with the same architecture and extracted parameters
  ANN::CoreConfig<double> loadConfig;
  loadConfig.modeType = ANN::ModeType::PREDICT;
  loadConfig.deviceType = ANN::DeviceType::CPU;
  loadConfig.layersConfig = makeLayersConfig({
    {2, ANN::ActvFuncType::RELU},
    {4, ANN::ActvFuncType::SIGMOID},
    {1, ANN::ActvFuncType::SIGMOID}
  });
  loadConfig.parameters = trainedParams;

  auto loadedCore = ANN::Core<double>::makeCore(loadConfig);
  ANN::Output<double> loadedPred = loadedCore->predict({1.0, 1.0});

  std::cout << "  original=" << originalPred[0] << "  loaded=" << loadedPred[0] << std::endl;
  CHECK_NEAR(originalPred[0], loadedPred[0], 1e-10, "parameter round-trip exact match");

  // Check that architecture matches
  CHECK(loadedCore->getLayersConfig().size() == 3, "loaded layers = 3");
  CHECK(loadedCore->getLayersConfig()[1].numNeurons == 4, "loaded hidden = 4 neurons");
}

//===================================================================================================================//

static void testParameterRoundTripPreservesArchitecture() {
  std::cout << "--- testParameterRoundTripPreservesArchitecture ---" << std::endl;

  ANN::CoreConfig<double> config;
  config.modeType = ANN::ModeType::PREDICT;
  config.deviceType = ANN::DeviceType::CPU;
  config.layersConfig = makeLayersConfig({
    {3, ANN::ActvFuncType::RELU},
    {5, ANN::ActvFuncType::TANH},
    {2, ANN::ActvFuncType::SIGMOID}
  });

  // Set distinct parameters
  config.parameters.weights.resize(3);
  config.parameters.weights[1] = {
    {0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {0.7, 0.8, 0.9}, {1.0, 1.1, 1.2}, {1.3, 1.4, 1.5}
  };
  config.parameters.weights[2] = {{0.1, 0.2, 0.3, 0.4, 0.5}, {0.6, 0.7, 0.8, 0.9, 1.0}};
  config.parameters.biases.resize(3);
  config.parameters.biases[1] = {0.01, 0.02, 0.03, 0.04, 0.05};
  config.parameters.biases[2] = {0.06, 0.07};

  auto core = ANN::Core<double>::makeCore(config);

  // Extract parameters via getters
  const ANN::Parameters<double>& params = core->getParameters();
  const ANN::LayersConfig& layers = core->getLayersConfig();

  // Recreate from extracted state
  ANN::CoreConfig<double> newConfig;
  newConfig.modeType = ANN::ModeType::PREDICT;
  newConfig.deviceType = ANN::DeviceType::CPU;
  newConfig.layersConfig = layers;
  newConfig.parameters = params;

  auto loaded = ANN::Core<double>::makeCore(newConfig);

  // Verify architecture preserved
  CHECK(loaded->getLayersConfig().size() == 3, "loaded 3 layers");
  CHECK(loaded->getLayersConfig()[0].numNeurons == 3, "layer 0: 3 neurons");
  CHECK(loaded->getLayersConfig()[1].numNeurons == 5, "layer 1: 5 neurons");
  CHECK(loaded->getLayersConfig()[2].numNeurons == 2, "layer 2: 2 neurons");

  // Verify predictions match
  ANN::Input<double> input = {0.5, 0.3, 0.7};
  ANN::Output<double> origPred = core->predict(input);
  ANN::Output<double> loadPred = loaded->predict(input);

  CHECK(origPred.size() == 2, "output size = 2");
  CHECK_NEAR(origPred[0], loadPred[0], 1e-10, "pred[0] matches after round-trip");
  CHECK_NEAR(origPred[1], loadPred[1], 1e-10, "pred[1] matches after round-trip");
}

//===================================================================================================================//

static void testGettersReturnExpectedState() {
  std::cout << "--- testGettersReturnExpectedState ---" << std::endl;

  ANN::CoreConfig<double> config;
  config.modeType = ANN::ModeType::PREDICT;
  config.deviceType = ANN::DeviceType::CPU;
  config.layersConfig = makeLayersConfig({
    {2, ANN::ActvFuncType::RELU},
    {3, ANN::ActvFuncType::SIGMOID},
    {1, ANN::ActvFuncType::SIGMOID}
  });

  config.parameters.weights.resize(3);
  config.parameters.weights[1] = {{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}};
  config.parameters.weights[2] = {{0.7, 0.8, 0.9}};
  config.parameters.biases.resize(3);
  config.parameters.biases[1] = {0.01, 0.02, 0.03};
  config.parameters.biases[2] = {0.04};

  auto core = ANN::Core<double>::makeCore(config);

  CHECK(core->getDeviceType() == ANN::DeviceType::CPU, "device type is CPU");
  CHECK(core->getModeType() == ANN::ModeType::PREDICT, "mode type is PREDICT");
  CHECK(core->getLayersConfig().size() == 3, "3 layers");

  const auto& params = core->getParameters();
  CHECK(params.weights.size() == 3, "weights has 3 layers");
  CHECK(params.biases.size() == 3, "biases has 3 layers");
  CHECK_NEAR(params.weights[1][0][0], 0.1, 1e-10, "weight[1][0][0] = 0.1");
  CHECK_NEAR(params.biases[2][0], 0.04, 1e-10, "bias[2][0] = 0.04");
}

//===================================================================================================================//

void runSerializationTests() {
  testParameterRoundTrip();
  testParameterRoundTripPreservesArchitecture();
  testGettersReturnExpectedState();
}

