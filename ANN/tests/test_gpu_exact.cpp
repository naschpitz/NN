#include "test_helpers.hpp"

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

void runGPUExactTests()
{
  testGPUExactForwardBackwardSquaredDifference();
  testGPUExactForwardBackwardCrossEntropy();
  testGPUExactForwardBackwardWeightedCrossEntropy();
}
