#include "test_helpers.hpp"

static void testSoftmaxPredict()
{
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

static void testSoftmaxTrain()
{
  std::cout << "--- testSoftmaxTrain ---" << std::endl;

  // Classification: 2 inputs → 3 classes with softmax output
  ANN::CoreConfig<double> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::CPU;
  config.layersConfig =
    makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {4, ANN::ActvFuncType::RELU}, {3, ANN::ActvFuncType::SOFTMAX}});

  config.trainingConfig.numEpochs = 500;
  config.trainingConfig.learningRate = 0.1;
  config.progressReports = 0;

  // One-hot encoded targets
  ANN::Samples<double> samples = {
    {{1.0, 0.0}, {1.0, 0.0, 0.0}}, // class 0
    {{0.0, 1.0}, {0.0, 1.0, 0.0}}, // class 1
    {{1.0, 1.0}, {0.0, 0.0, 1.0}} // class 2
  };

  auto core = ANN::Core<double>::makeCore(config);
  core->train(samples.size(), ANN::makeSampleProvider(samples));

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

static void testSoftmaxHiddenLayer()
{
  std::cout << "--- testSoftmaxHiddenLayer ---" << std::endl;

  // Softmax in a hidden layer (unusual but should work)
  ANN::CoreConfig<double> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::CPU;
  config.layersConfig =
    makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {4, ANN::ActvFuncType::SOFTMAX}, {1, ANN::ActvFuncType::SIGMOID}});

  config.trainingConfig.numEpochs = 500;
  config.trainingConfig.learningRate = 0.1;
  config.progressReports = 0;

  ANN::Samples<double> samples = {{{1.0, 1.0}, {1.0}}, {{0.0, 0.0}, {0.0}}};

  auto core = ANN::Core<double>::makeCore(config);
  core->train(samples.size(), ANN::makeSampleProvider(samples));

  auto out1 = core->predict({1.0, 1.0});
  auto out0 = core->predict({0.0, 0.0});

  CHECK(out1[0] > out0[0], "softmax hidden: (1,1) > (0,0)");
}

//===================================================================================================================//

//===================================================================================================================//

static void testDropoutTraining()
{
  std::cout << "--- testDropoutTraining ---" << std::endl;

  // Train XOR with dropout — should still converge (dropout is regularization, not destructive)
  ANN::CoreConfig<double> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::CPU;
  config.layersConfig =
    makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {16, ANN::ActvFuncType::RELU}, {1, ANN::ActvFuncType::SIGMOID}});
  config.trainingConfig.numEpochs = 10000;
  config.trainingConfig.learningRate = 0.1;
  config.trainingConfig.dropoutRate = 0.2f;

  auto core = ANN::Core<double>::makeCore(config);

  std::vector<ANN::Sample<double>> samples = {{{0, 0}, {0}}, {{0, 1}, {1}}, {{1, 0}, {1}}, {{1, 1}, {0}}};

  core->train(samples.size(), ANN::makeSampleProvider(samples));

  // Predict (no dropout during inference)
  auto r00 = core->predict({0, 0});
  auto r01 = core->predict({0, 1});
  auto r10 = core->predict({1, 0});
  auto r11 = core->predict({1, 1});

  CHECK(r00[0] < 0.4, "XOR(0,0) < 0.4 with dropout training");
  CHECK(r01[0] > 0.6, "XOR(0,1) > 0.6 with dropout training");
  CHECK(r10[0] > 0.6, "XOR(1,0) > 0.6 with dropout training");
  CHECK(r11[0] < 0.4, "XOR(1,1) < 0.4 with dropout training");
}

//===================================================================================================================//

static void testDropoutDisabledByDefault()
{
  std::cout << "--- testDropoutDisabledByDefault ---" << std::endl;

  ANN::CoreConfig<double> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::CPU;
  config.layersConfig = makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {1, ANN::ActvFuncType::SIGMOID}});

  CHECK(config.trainingConfig.dropoutRate == 0.0f, "dropoutRate defaults to 0.0");
}

//===================================================================================================================//

void runCoreFeaturesTests2()
{
  testSoftmaxPredict();
  testSoftmaxTrain();
  testSoftmaxHiddenLayer();
  testDropoutTraining();
  testDropoutDisabledByDefault();
}
