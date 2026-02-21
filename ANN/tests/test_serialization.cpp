#include "test_helpers.hpp"

#include <fstream>
#include <cstdio>

//===================================================================================================================//

static void testSaveToString() {
  std::cout << "--- testSaveToString ---" << std::endl;

  ANN::CoreConfig<double> config;
  config.modeType = ANN::ModeType::PREDICT;
  config.deviceType = ANN::DeviceType::CPU;
  config.layersConfig = makeLayersConfig({
    {2, ANN::ActvFuncType::RELU},
    {3, ANN::ActvFuncType::SIGMOID},
    {1, ANN::ActvFuncType::SIGMOID}
  });

  // Set known parameters
  config.parameters.weights.resize(3);
  config.parameters.weights[1] = {{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}};
  config.parameters.weights[2] = {{0.7, 0.8, 0.9}};
  config.parameters.biases.resize(3);
  config.parameters.biases[1] = {0.01, 0.02, 0.03};
  config.parameters.biases[2] = {0.04};

  auto core = ANN::Core<double>::makeCore(config);
  std::string json = ANN::Utils<double>::save(*core);

  CHECK(!json.empty(), "save returns non-empty JSON");
  CHECK(json.find("\"device\"") != std::string::npos, "JSON contains device");
  CHECK(json.find("\"layersConfig\"") != std::string::npos, "JSON contains layersConfig");
  CHECK(json.find("\"parameters\"") != std::string::npos, "JSON contains parameters");
  CHECK(json.find("\"cpu\"") != std::string::npos, "JSON contains cpu");
  CHECK(json.find("\"sigmoid\"") != std::string::npos, "JSON contains sigmoid");
}

//===================================================================================================================//

static void testSaveLoadFileRoundTrip() {
  std::cout << "--- testSaveLoadFileRoundTrip ---" << std::endl;

  // Train a network
  ANN::CoreConfig<double> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::CPU;
  config.layersConfig = makeLayersConfig({
    {2, ANN::ActvFuncType::RELU},
    {4, ANN::ActvFuncType::SIGMOID},
    {1, ANN::ActvFuncType::SIGMOID}
  });
  config.trainingConfig.numEpochs = 100;
  config.trainingConfig.learningRate = 0.5;
  config.trainingConfig.progressReports = 0;

  ANN::Samples<double> samples = {{{1.0, 1.0}, {1.0}}, {{0.0, 0.0}, {0.0}}};

  auto trainCore = ANN::Core<double>::makeCore(config);
  trainCore->train(samples);

  ANN::Output<double> originalPred = trainCore->predict({1.0, 1.0});

  // Save to file
  std::string tmpFile = "/tmp/test_ann_roundtrip.json";
  ANN::Utils<double>::save(*trainCore, tmpFile);

  // Load from file
  auto loadedCore = ANN::Utils<double>::load(tmpFile);
  CHECK(loadedCore != nullptr, "loaded core non-null");

  ANN::Output<double> loadedPred = loadedCore->predict({1.0, 1.0});

  std::cout << "  original=" << originalPred[0] << "  loaded=" << loadedPred[0] << std::endl;
  CHECK_NEAR(originalPred[0], loadedPred[0], 1e-10, "save/load round-trip exact match");

  // Check that loaded config matches
  CHECK(loadedCore->getLayersConfig().size() == 3, "loaded layers = 3");
  CHECK(loadedCore->getLayersConfig()[1].numNeurons == 4, "loaded hidden = 4 neurons");

  // Clean up
  std::remove(tmpFile.c_str());
}

//===================================================================================================================//

static void testSaveLoadPreservesArchitecture() {
  std::cout << "--- testSaveLoadPreservesArchitecture ---" << std::endl;

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

  std::string tmpFile = "/tmp/test_ann_arch.json";
  ANN::Utils<double>::save(*core, tmpFile);

  auto loaded = ANN::Utils<double>::load(tmpFile);

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
  CHECK_NEAR(origPred[0], loadPred[0], 1e-10, "pred[0] matches after load");
  CHECK_NEAR(origPred[1], loadPred[1], 1e-10, "pred[1] matches after load");

  std::remove(tmpFile.c_str());
}

//===================================================================================================================//

void runSerializationTests() {
  testSaveToString();
  testSaveLoadFileRoundTrip();
  testSaveLoadPreservesArchitecture();
}

