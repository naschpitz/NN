#include "test_helpers.hpp"

//===================================================================================================================//

static void testNameToType() {
  std::cout << "--- testNameToType ---" << std::endl;

  CHECK(ANN::ActvFunc::nameToType("relu") == ANN::ActvFuncType::RELU, "relu → RELU");
  CHECK(ANN::ActvFunc::nameToType("sigmoid") == ANN::ActvFuncType::SIGMOID, "sigmoid → SIGMOID");
  CHECK(ANN::ActvFunc::nameToType("tanh") == ANN::ActvFuncType::TANH, "tanh → TANH");
  CHECK(ANN::ActvFunc::nameToType("nonexistent") == ANN::ActvFuncType::UNKNOWN, "nonexistent → UNKNOWN");
  CHECK(ANN::ActvFunc::nameToType("") == ANN::ActvFuncType::UNKNOWN, "empty → UNKNOWN");
}

//===================================================================================================================//

static void testTypeToName() {
  std::cout << "--- testTypeToName ---" << std::endl;

  CHECK(ANN::ActvFunc::typeToName(ANN::ActvFuncType::RELU) == "relu", "RELU → relu");
  CHECK(ANN::ActvFunc::typeToName(ANN::ActvFuncType::SIGMOID) == "sigmoid", "SIGMOID → sigmoid");
  CHECK(ANN::ActvFunc::typeToName(ANN::ActvFuncType::TANH) == "tanh", "TANH → tanh");
  CHECK(ANN::ActvFunc::typeToName(ANN::ActvFuncType::UNKNOWN) == "unknown", "UNKNOWN → unknown");
}

//===================================================================================================================//

static void testReLU() {
  std::cout << "--- testReLU ---" << std::endl;

  // Forward
  CHECK_NEAR(ANN::ActvFunc::calculate(2.0f, ANN::ActvFuncType::RELU), 2.0f, 1e-6f, "relu(2) = 2");
  CHECK_NEAR(ANN::ActvFunc::calculate(-1.0f, ANN::ActvFuncType::RELU), 0.0f, 1e-6f, "relu(-1) = 0");
  CHECK_NEAR(ANN::ActvFunc::calculate(0.0f, ANN::ActvFuncType::RELU), 0.0f, 1e-6f, "relu(0) = 0");
  CHECK_NEAR(ANN::ActvFunc::calculate(0.5f, ANN::ActvFuncType::RELU), 0.5f, 1e-6f, "relu(0.5) = 0.5");

  // Derivative
  CHECK_NEAR(ANN::ActvFunc::calculate(2.0f, ANN::ActvFuncType::RELU, true), 1.0f, 1e-6f, "drelu(2) = 1");
  CHECK_NEAR(ANN::ActvFunc::calculate(-1.0f, ANN::ActvFuncType::RELU, true), 0.0f, 1e-6f, "drelu(-1) = 0");
  CHECK_NEAR(ANN::ActvFunc::calculate(0.5f, ANN::ActvFuncType::RELU, true), 1.0f, 1e-6f, "drelu(0.5) = 1");
}

//===================================================================================================================//

static void testSigmoid() {
  std::cout << "--- testSigmoid ---" << std::endl;

  // Forward
  CHECK_NEAR(ANN::ActvFunc::calculate(0.0f, ANN::ActvFuncType::SIGMOID), 0.5f, 1e-6f, "sigmoid(0) = 0.5");
  float sig5 = 1.0f / (1.0f + std::exp(-5.0f));
  CHECK_NEAR(ANN::ActvFunc::calculate(5.0f, ANN::ActvFuncType::SIGMOID), sig5, 1e-4f, "sigmoid(5) ≈ 0.993");
  float sigNeg5 = 1.0f / (1.0f + std::exp(5.0f));
  CHECK_NEAR(ANN::ActvFunc::calculate(-5.0f, ANN::ActvFuncType::SIGMOID), sigNeg5, 1e-4f, "sigmoid(-5) ≈ 0.007");

  // Derivative: dsigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
  CHECK_NEAR(ANN::ActvFunc::calculate(0.0f, ANN::ActvFuncType::SIGMOID, true), 0.25f, 1e-6f, "dsigmoid(0) = 0.25");
  float dsig5 = sig5 * (1.0f - sig5);
  CHECK_NEAR(ANN::ActvFunc::calculate(5.0f, ANN::ActvFuncType::SIGMOID, true), dsig5, 1e-4f, "dsigmoid(5)");
}

//===================================================================================================================//

static void testTanh() {
  std::cout << "--- testTanh ---" << std::endl;

  // Forward
  CHECK_NEAR(ANN::ActvFunc::calculate(0.0f, ANN::ActvFuncType::TANH), 0.0f, 1e-6f, "tanh(0) = 0");
  float tanh1 = std::tanh(1.0f);
  CHECK_NEAR(ANN::ActvFunc::calculate(1.0f, ANN::ActvFuncType::TANH), tanh1, 1e-5f, "tanh(1) ≈ 0.7616");
  float tanhNeg1 = std::tanh(-1.0f);
  CHECK_NEAR(ANN::ActvFunc::calculate(-1.0f, ANN::ActvFuncType::TANH), tanhNeg1, 1e-5f, "tanh(-1) ≈ -0.7616");

  // Derivative: dtanh(x) = 1 - tanh²(x)
  CHECK_NEAR(ANN::ActvFunc::calculate(0.0f, ANN::ActvFuncType::TANH, true), 1.0f, 1e-6f, "dtanh(0) = 1");
  float dtanh1 = 1.0f - tanh1 * tanh1;
  CHECK_NEAR(ANN::ActvFunc::calculate(1.0f, ANN::ActvFuncType::TANH, true), dtanh1, 1e-5f, "dtanh(1)");
}

//===================================================================================================================//

static void testCalculateUnknownThrows() {
  std::cout << "--- testCalculateUnknownThrows ---" << std::endl;

  bool threw = false;
  try {
    ANN::ActvFunc::calculate(1.0f, ANN::ActvFuncType::UNKNOWN);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw, "calculate with UNKNOWN throws");
}

//===================================================================================================================//

void runActvFuncTests() {
  testNameToType();
  testTypeToName();
  testReLU();
  testSigmoid();
  testTanh();
  testCalculateUnknownThrows();
}

