#include "test_helpers.hpp"

static void testCrossEntropyStringConversion()
{
  std::cout << "--- testCrossEntropyStringConversion ---" << std::endl;

  CHECK(ANN::CostFunction::nameToType("crossEntropy") == ANN::CostFunctionType::CROSS_ENTROPY,
        "nameToType crossEntropy");
  CHECK(ANN::CostFunction::typeToName(ANN::CostFunctionType::CROSS_ENTROPY) == "crossEntropy",
        "typeToName crossEntropy");
}

//===================================================================================================================//

static void testExactForwardBackwardSquaredDifference()
{
  std::cout << "--- testExactForwardBackwardSquaredDifference ---" << std::endl;

  // Hand-computed test: 2 inputs → 2 hidden (ReLU) → 1 output (sigmoid)
  // Squared-difference cost, SGD lr=1.0, 1 sample, 1 step.
  // Every value below was computed by hand / Python and must match EXACTLY.
  //
  // Input:  x = [1.0, 0.5]
  // Target: y = [1.0]
  // Weights/biases: see below
  //
  // Forward:
  //   z1 = [0.1*1 + 0.2*0.5 + 0.1, 0.3*1 + 0.4*0.5 + (-0.1)] = [0.3, 0.4]
  //   a1 = relu([0.3, 0.4]) = [0.3, 0.4]
  //   z2 = [0.5*0.3 + (-0.3)*0.4 + 0.0] = [0.03]
  //   a2 = sigmoid(0.03) = 0.50749944...
  //   loss = (a2 - 1)^2 = 0.24249375...
  //
  // Backward: dL/da2 = 2*(a2 - 1) = -0.98500112...
  //   dsigmoid = a2*(1-a2) = 0.24993746...
  //   dL/dz2 = dL/da2 * dsigmoid = -0.24619303...
  //   dL/dw2 = a1 * dL/dz2 = [0.3*(-0.2462), 0.4*(-0.2462)]
  //   dL/db2 = dL/dz2
  //   dL/da1 = w2^T * dL/dz2 = [0.5*(-0.2462), (-0.3)*(-0.2462)]
  //   dL/dz1 = dL/da1 * relu'(z1) = dL/da1 (both z1 > 0)
  //   dL/dw1 = x * dL/dz1^T
  //   dL/db1 = dL/dz1
  //   SGD: new = old - lr * grad

  // Architecture: layer 0 = input (2), layer 1 = hidden ReLU (2), layer 2 = output sigmoid (1)
  // Layer 0 copies input (no weights). Layers 1 and 2 have weights.
  ANN::CoreConfig<double> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::CPU;
  config.layersConfig =
    makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {2, ANN::ActvFuncType::RELU}, {1, ANN::ActvFuncType::SIGMOID}});
  config.trainingConfig.learningRate = 1.0;
  config.logLevel = ANN::LogLevel::ERROR;

  // weights[layer][neuron][input]: layer 0 = empty (input), layer 1 = hidden, layer 2 = output
  config.parameters.weights = {{}, {{0.1, 0.2}, {0.3, 0.4}}, {{0.5, -0.3}}};
  config.parameters.biases = {{}, {0.1, -0.1}, {0.0}};

  auto core = ANN::Core<double>::makeCore(config);

  // --- Verify forward pass ---
  // Note: per-neuron activations (relu, sigmoid) are computed via float cast in C++,
  // so expected values account for float precision loss.
  ANN::Output<double> out = core->predict({1.0, 0.5});
  CHECK_NEAR(out[0], 0.50749945640563965, 1e-12, "SD forward: output exact");

  // --- Verify backward pass via train (1 epoch, 1 sample, SGD lr=1.0, no shuffle) ---
  // Must use train() because the step-by-step API only works for CNN-embedded ANN.
  config.trainingConfig.numEpochs = 1;
  config.trainingConfig.shuffleSamples = false;
  config.progressReports = 0;
  ANN::Samples<double> samples = {{{1.0, 0.5}, {1.0}}};
  auto trainCore = ANN::Core<double>::makeCore(config);
  trainCore->train(samples.size(), ANN::makeSampleProvider(samples));

  const ANN::Parameters<double>& p = trainCore->getParameters();

  // Every single weight and bias checked against hand-computed values
  CHECK_NEAR(p.weights[1][0][0], 0.22309743915421479, 1e-12, "SD w1[0][0]");
  CHECK_NEAR(p.weights[1][0][1], 0.26154871957710740, 1e-12, "SD w1[0][1]");
  CHECK_NEAR(p.weights[1][1][0], 0.22614153650747112, 1e-12, "SD w1[1][0]");
  CHECK_NEAR(p.weights[1][1][1], 0.36307076825373558, 1e-12, "SD w1[1][1]");
  CHECK_NEAR(p.biases[1][0], 0.22309743915421479, 1e-12, "SD b1[0]");
  CHECK_NEAR(p.biases[1][1], -0.17385846349252887, 1e-12, "SD b1[1]");
  CHECK_NEAR(p.weights[2][0][0], 0.57385846642740057, 1e-12, "SD w2[0][0]");
  CHECK_NEAR(p.weights[2][0][1], -0.20152204720919234, 1e-12, "SD w2[0][1]");
  CHECK_NEAR(p.biases[2][0], 0.24619487830842957, 1e-12, "SD b2[0]");
}

//===================================================================================================================//

static void testExactForwardBackwardCrossEntropy()
{
  std::cout << "--- testExactForwardBackwardCrossEntropy ---" << std::endl;

  // Hand-computed test: 2 inputs → 2 hidden (ReLU) → 2 output (softmax)
  // Cross-entropy cost, SGD lr=1.0, 1 sample, 1 step.
  // Every value was computed by hand.
  //
  // Input:  x = [1.0, 0.5]
  // Target: y = [1.0, 0.0]
  //
  // Forward:
  //   z1 = [0.3, 0.4], a1 = relu = [0.3, 0.4]
  //   z2 = [0.5*0.3 + (-0.3)*0.4, (-0.2)*0.3 + 0.6*0.4] = [0.03, 0.18]
  //   softmax: exp(0.03)=1.03045..., exp(0.18)=1.19722...
  //   a2 = [0.46259..., 0.53741...]
  //   loss = -log(0.46259...) = 0.77032...
  //
  // Backward:
  //   dL/da2 = [-1/a2[0], 0] = [-2.16165..., 0]
  //   softmax Jacobian → dL/dz2 = [a2[0]-1, a2[1]] = [-0.53741..., 0.53741...]
  //   dL/dw2, dL/db2, dL/dw1, dL/db1 all hand-computed
  //   SGD: new = old - lr * grad

  // Architecture: layer 0 = input (2), layer 1 = hidden ReLU (2), layer 2 = output softmax (2)
  ANN::CoreConfig<double> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::CPU;
  config.layersConfig =
    makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {2, ANN::ActvFuncType::RELU}, {2, ANN::ActvFuncType::SOFTMAX}});
  config.costFunctionConfig.type = ANN::CostFunctionType::CROSS_ENTROPY;
  config.trainingConfig.learningRate = 1.0;
  config.logLevel = ANN::LogLevel::ERROR;

  config.parameters.weights = {{}, {{0.1, 0.2}, {0.3, 0.4}}, {{0.5, -0.3}, {-0.2, 0.6}}};
  config.parameters.biases = {{}, {0.1, -0.1}, {0.0, 0.0}};

  auto core = ANN::Core<double>::makeCore(config);

  // --- Verify forward pass exactly ---
  // Softmax is computed in double (template T), but hidden ReLU goes through float.
  ANN::Output<double> out = core->predict({1.0, 0.5});
  CHECK_NEAR(out[0], 0.4625701553971332, 1e-12, "CE forward: a2[0] exact");
  CHECK_NEAR(out[1], 0.5374298446028668, 1e-12, "CE forward: a2[1] exact");
  CHECK_NEAR(out[0] + out[1], 1.0, 1e-14, "CE forward: softmax sums to 1");

  // --- Verify backward pass via train (1 epoch, 1 sample, SGD lr=1.0) ---
  config.trainingConfig.numEpochs = 1;
  config.trainingConfig.shuffleSamples = false;
  config.progressReports = 0;
  ANN::Samples<double> samples = {{{1.0, 0.5}, {1.0, 0.0}}};
  auto trainCore = ANN::Core<double>::makeCore(config);
  trainCore->train(samples.size(), ANN::makeSampleProvider(samples));

  const ANN::Parameters<double>& p = trainCore->getParameters();

  // Layer 2: every weight and bias
  CHECK_NEAR(p.weights[2][0][0], 0.66122895978752294, 1e-14, "CE w2[0][0]");
  CHECK_NEAR(p.weights[2][0][1], -0.085028058955521818, 1e-14, "CE w2[0][1]");
  CHECK_NEAR(p.weights[2][1][0], -0.36122895978752301, 1e-14, "CE w2[1][0]");
  CHECK_NEAR(p.weights[2][1][1], 0.38502805895552172, 1e-14, "CE w2[1][1]");
  CHECK_NEAR(p.biases[2][0], 0.53742984460286669, 1e-14, "CE b2[0]");
  CHECK_NEAR(p.biases[2][1], -0.5374298446028668, 1e-14, "CE b2[1]");

  // Layer 1: every weight and bias
  CHECK_NEAR(p.weights[1][0][0], 0.47620089122200671, 1e-14, "CE w1[0][0]");
  CHECK_NEAR(p.weights[1][0][1], 0.38810044561100338, 1e-14, "CE w1[0][1]");
  CHECK_NEAR(p.weights[1][1][0], -0.18368686014258012, 1e-14, "CE w1[1][0]");
  CHECK_NEAR(p.weights[1][1][1], 0.15815656992870997, 1e-14, "CE w1[1][1]");
  CHECK_NEAR(p.biases[1][0], 0.47620089122200671, 1e-14, "CE b1[0]");
  CHECK_NEAR(p.biases[1][1], -0.58368686014258009, 1e-14, "CE b1[1]");
}

//===================================================================================================================//

static void testExactForwardBackwardWeightedCrossEntropy()
{
  std::cout << "--- testExactForwardBackwardWeightedCrossEntropy ---" << std::endl;

  // Same network as above but with per-class weights [3.0, 0.5]
  // dL/da2[j] = -w_j * y_j / a2[j]
  // With y=[1,0]: dL/da2 = [-3/a2[0], 0]
  // The softmax Jacobian gives: dL/dz2[j] = a2[j] * (dL/da2[j] - dot)
  // dot = a2[0]*(-3/a2[0]) + a2[1]*0 = -3
  // dL/dz2[0] = a2[0]*(-3/a2[0] + 3) = a2[0]*3 - 3 = 3*(a2[0]-1)
  // dL/dz2[1] = a2[1]*(0 + 3) = 3*a2[1]

  ANN::CoreConfig<double> config;
  config.modeType = ANN::ModeType::TRAIN;
  config.deviceType = ANN::DeviceType::CPU;
  config.layersConfig =
    makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {2, ANN::ActvFuncType::RELU}, {2, ANN::ActvFuncType::SOFTMAX}});
  config.costFunctionConfig.type = ANN::CostFunctionType::CROSS_ENTROPY;
  config.costFunctionConfig.weights = {3.0, 0.5};
  config.trainingConfig.learningRate = 1.0;
  config.logLevel = ANN::LogLevel::ERROR;

  config.parameters.weights = {{}, {{0.1, 0.2}, {0.3, 0.4}}, {{0.5, -0.3}, {-0.2, 0.6}}};
  config.parameters.biases = {{}, {0.1, -0.1}, {0.0, 0.0}};

  auto core = ANN::Core<double>::makeCore(config);

  // Forward pass same as unweighted — cost function weights only affect gradients
  ANN::Output<double> out = core->predict({1.0, 0.5});
  CHECK_NEAR(out[0], 0.4625701553971332, 1e-12, "WCE forward: a2[0]");
  CHECK_NEAR(out[1], 0.5374298446028668, 1e-12, "WCE forward: a2[1]");

  // Backward with per-class weights [3.0, 0.5]
  // dL/da = [-3/a2[0], 0], dot = -3, dL/dz2 = [3*(a2[0]-1), 3*a2[1]]
  config.trainingConfig.numEpochs = 1;
  config.trainingConfig.shuffleSamples = false;
  config.progressReports = 0;
  ANN::Samples<double> samples = {{{1.0, 0.5}, {1.0, 0.0}}};
  auto trainCore = ANN::Core<double>::makeCore(config);
  trainCore->train(samples.size(), ANN::makeSampleProvider(samples));

  const ANN::Parameters<double>& p = trainCore->getParameters();

  // Layer 2: every weight and bias (gradients scaled by weight factor 3.0)
  CHECK_NEAR(p.weights[2][0][0], 0.98368687936256904, 1e-14, "WCE w2[0][0]");
  CHECK_NEAR(p.weights[2][0][1], 0.34491582313343466, 1e-14, "WCE w2[0][1]");
  CHECK_NEAR(p.weights[2][1][0], -0.68368687936256922, 1e-14, "WCE w2[1][0]");
  CHECK_NEAR(p.weights[2][1][1], -0.044915823133434674, 1e-14, "WCE w2[1][1]");
  CHECK_NEAR(p.biases[2][0], 1.6122895338086003, 1e-14, "WCE b2[0]");
  CHECK_NEAR(p.biases[2][1], -1.6122895338086005, 1e-14, "WCE b2[1]");

  // Layer 1: every weight and bias
  CHECK_NEAR(p.weights[1][0][0], 1.2286026736660203, 1e-14, "WCE w1[0][0]");
  CHECK_NEAR(p.weights[1][0][1], 0.76430133683301005, 1e-14, "WCE w1[0][1]");
  CHECK_NEAR(p.weights[1][1][0], -1.1510605804277403, 1e-14, "WCE w1[1][0]");
  CHECK_NEAR(p.weights[1][1][1], -0.32553029021387014, 1e-14, "WCE w1[1][1]");
  CHECK_NEAR(p.biases[1][0], 1.2286026736660203, 1e-14, "WCE b1[0]");
  CHECK_NEAR(p.biases[1][1], -1.5510605804277404, 1e-14, "WCE b1[1]");
}

//===================================================================================================================//

void runCoreExactTests()
{
  testCrossEntropyStringConversion();
  testExactForwardBackwardSquaredDifference();
  testExactForwardBackwardCrossEntropy();
  testExactForwardBackwardWeightedCrossEntropy();
}
