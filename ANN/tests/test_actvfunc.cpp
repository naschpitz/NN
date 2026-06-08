#include "test_helpers.hpp"

//===================================================================================================================//

static void testNameToType()
{
  std::cout << "--- testNameToType ---" << std::endl;

  CHECK(::ActvFunc::nameToType("relu") == ::ActvFuncType::RELU, "relu → RELU");
  CHECK(::ActvFunc::nameToType("sigmoid") == ::ActvFuncType::SIGMOID, "sigmoid → SIGMOID");
  CHECK(::ActvFunc::nameToType("tanh") == ::ActvFuncType::TANH, "tanh → TANH");
  CHECK(::ActvFunc::nameToType("softmax") == ::ActvFuncType::SOFTMAX, "softmax → SOFTMAX");

  CHECK_THROWS(::ActvFunc::nameToType("nonexistent"), "nonexistent throws");
  CHECK_THROWS(::ActvFunc::nameToType(""), "empty throws");
}

//===================================================================================================================//

static void testTypeToName()
{
  std::cout << "--- testTypeToName ---" << std::endl;

  CHECK(::ActvFunc::typeToName(::ActvFuncType::RELU) == "relu", "RELU → relu");
  CHECK(::ActvFunc::typeToName(::ActvFuncType::SIGMOID) == "sigmoid", "SIGMOID → sigmoid");
  CHECK(::ActvFunc::typeToName(::ActvFuncType::TANH) == "tanh", "TANH → tanh");
  CHECK(::ActvFunc::typeToName(::ActvFuncType::SOFTMAX) == "softmax", "SOFTMAX → softmax");
}

//===================================================================================================================//

static void testReLU()
{
  std::cout << "--- testReLU ---" << std::endl;

  // Forward
  CHECK_NEAR(::ActvFunc::calculate(2.0f, ::ActvFuncType::RELU), 2.0f, 1e-6f, "relu(2) = 2");
  CHECK_NEAR(::ActvFunc::calculate(-1.0f, ::ActvFuncType::RELU), 0.0f, 1e-6f, "relu(-1) = 0");
  CHECK_NEAR(::ActvFunc::calculate(0.0f, ::ActvFuncType::RELU), 0.0f, 1e-6f, "relu(0) = 0");
  CHECK_NEAR(::ActvFunc::calculate(0.5f, ::ActvFuncType::RELU), 0.5f, 1e-6f, "relu(0.5) = 0.5");

  // Derivative
  CHECK_NEAR(::ActvFunc::calculate(2.0f, ::ActvFuncType::RELU, true), 1.0f, 1e-6f, "drelu(2) = 1");
  CHECK_NEAR(::ActvFunc::calculate(-1.0f, ::ActvFuncType::RELU, true), 0.0f, 1e-6f, "drelu(-1) = 0");
  CHECK_NEAR(::ActvFunc::calculate(0.5f, ::ActvFuncType::RELU, true), 1.0f, 1e-6f, "drelu(0.5) = 1");
}

//===================================================================================================================//

static void testSigmoid()
{
  std::cout << "--- testSigmoid ---" << std::endl;

  // Forward
  CHECK_NEAR(::ActvFunc::calculate(0.0f, ::ActvFuncType::SIGMOID), 0.5f, 1e-6f, "sigmoid(0) = 0.5");
  float sig5 = 1.0f / (1.0f + std::exp(-5.0f));
  CHECK_NEAR(::ActvFunc::calculate(5.0f, ::ActvFuncType::SIGMOID), sig5, 1e-4f, "sigmoid(5) ≈ 0.993");
  float sigNeg5 = 1.0f / (1.0f + std::exp(5.0f));
  CHECK_NEAR(::ActvFunc::calculate(-5.0f, ::ActvFuncType::SIGMOID), sigNeg5, 1e-4f, "sigmoid(-5) ≈ 0.007");

  // Derivative: dsigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
  CHECK_NEAR(::ActvFunc::calculate(0.0f, ::ActvFuncType::SIGMOID, true), 0.25f, 1e-6f, "dsigmoid(0) = 0.25");
  float dsig5 = sig5 * (1.0f - sig5);
  CHECK_NEAR(::ActvFunc::calculate(5.0f, ::ActvFuncType::SIGMOID, true), dsig5, 1e-4f, "dsigmoid(5)");
}

//===================================================================================================================//

static void testTanh()
{
  std::cout << "--- testTanh ---" << std::endl;

  // Forward
  CHECK_NEAR(::ActvFunc::calculate(0.0f, ::ActvFuncType::TANH), 0.0f, 1e-6f, "tanh(0) = 0");
  float tanh1 = std::tanh(1.0f);
  CHECK_NEAR(::ActvFunc::calculate(1.0f, ::ActvFuncType::TANH), tanh1, 1e-5f, "tanh(1) ≈ 0.7616");
  float tanhNeg1 = std::tanh(-1.0f);
  CHECK_NEAR(::ActvFunc::calculate(-1.0f, ::ActvFuncType::TANH), tanhNeg1, 1e-5f, "tanh(-1) ≈ -0.7616");

  // Derivative: dtanh(x) = 1 - tanh²(x)
  CHECK_NEAR(::ActvFunc::calculate(0.0f, ::ActvFuncType::TANH, true), 1.0f, 1e-6f, "dtanh(0) = 1");
  float dtanh1 = 1.0f - tanh1 * tanh1;
  CHECK_NEAR(::ActvFunc::calculate(1.0f, ::ActvFuncType::TANH, true), dtanh1, 1e-5f, "dtanh(1)");
}

//===================================================================================================================//

static void testSoftmax()
{
  std::cout << "--- testSoftmax ---" << std::endl;

  // Forward: softmax([1, 2, 3])
  float zs[] = {1.0f, 2.0f, 3.0f};
  float actvs[3] = {};
  ::ActvFunc::calculate(zs, actvs, 3ul, ::ActvFuncType::SOFTMAX, false, static_cast<const float*>(nullptr),
                           static_cast<float*>(nullptr));

  // Verify outputs sum to 1
  float sum = actvs[0] + actvs[1] + actvs[2];
  CHECK_NEAR(sum, 1.0f, 1e-5f, "softmax sums to 1");

  // Verify relative ordering: softmax(3) > softmax(2) > softmax(1)
  CHECK(actvs[2] > actvs[1], "softmax(3) > softmax(2)");
  CHECK(actvs[1] > actvs[0], "softmax(2) > softmax(1)");

  // Verify exact values: softmax(z_i) = exp(z_i) / Σ exp(z_j)
  float e1 = std::exp(1.0f), e2 = std::exp(2.0f), e3 = std::exp(3.0f);
  float total = e1 + e2 + e3;
  CHECK_NEAR(actvs[0], e1 / total, 1e-5f, "softmax[0] exact");
  CHECK_NEAR(actvs[1], e2 / total, 1e-5f, "softmax[1] exact");
  CHECK_NEAR(actvs[2], e3 / total, 1e-5f, "softmax[2] exact");

  // Numerical stability: large values should not overflow
  float largeZs[] = {1000.0f, 1001.0f, 1002.0f};
  float largeActvs[3] = {};
  ::ActvFunc::calculate(largeZs, largeActvs, 3ul, ::ActvFuncType::SOFTMAX, false,
                           static_cast<const float*>(nullptr), static_cast<float*>(nullptr));

  float largeSum = largeActvs[0] + largeActvs[1] + largeActvs[2];
  CHECK_NEAR(largeSum, 1.0f, 1e-5f, "softmax large values sum to 1");
  CHECK(largeActvs[2] > largeActvs[1], "softmax large: [2] > [1]");
  CHECK(largeActvs[1] > largeActvs[0], "softmax large: [1] > [0]");

  // Uniform input: all equal → all outputs equal
  float uniformZs[] = {5.0f, 5.0f, 5.0f};
  float uniformActvs[3] = {};
  ::ActvFunc::calculate(uniformZs, uniformActvs, 3ul, ::ActvFuncType::SOFTMAX, false,
                           static_cast<const float*>(nullptr), static_cast<float*>(nullptr));

  CHECK_NEAR(uniformActvs[0], 1.0f / 3.0f, 1e-5f, "softmax uniform [0] = 1/3");
  CHECK_NEAR(uniformActvs[1], 1.0f / 3.0f, 1e-5f, "softmax uniform [1] = 1/3");
  CHECK_NEAR(uniformActvs[2], 1.0f / 3.0f, 1e-5f, "softmax uniform [2] = 1/3");

  // Per-element calculate should throw for SOFTMAX
  bool threw = false;

  try {
    ::ActvFunc::calculate(1.0f, ::ActvFuncType::SOFTMAX);
  } catch (const std::invalid_argument&) {
    threw = true;
  }

  CHECK(threw, "per-element calculate with SOFTMAX throws");
}

//===================================================================================================================//

static void testSoftmaxBackward()
{
  std::cout << "--- testSoftmaxBackward ---" << std::endl;

  // Forward first to get activations
  float zs[] = {1.0f, 2.0f, 3.0f};
  float actvs[3] = {};
  ::ActvFunc::calculate(zs, actvs, 3ul, ::ActvFuncType::SOFTMAX, false, static_cast<const float*>(nullptr),
                           static_cast<float*>(nullptr));

  // Backward: dCost/dActvs = [1, 0, 0] (gradient from cost function)
  float dCost_dActvs[] = {1.0f, 0.0f, 0.0f};
  float dCost_dZs[3] = {};
  ::ActvFunc::calculate(zs, actvs, 3ul, ::ActvFuncType::SOFTMAX, true, dCost_dActvs, dCost_dZs);

  // dot = s0 * 1 + s1 * 0 + s2 * 0 = s0
  // dCost/dZ_0 = s0 * (1 - s0)
  // dCost/dZ_1 = s1 * (0 - s0) = -s1 * s0
  // dCost/dZ_2 = s2 * (0 - s0) = -s2 * s0
  float s0 = actvs[0], s1 = actvs[1], s2 = actvs[2];
  CHECK_NEAR(dCost_dZs[0], s0 * (1.0f - s0), 1e-5f, "dsoftmax dZ[0]");
  CHECK_NEAR(dCost_dZs[1], -s1 * s0, 1e-5f, "dsoftmax dZ[1]");
  CHECK_NEAR(dCost_dZs[2], -s2 * s0, 1e-5f, "dsoftmax dZ[2]");

  // Gradients should sum to zero (property of softmax Jacobian)
  float gradSum = dCost_dZs[0] + dCost_dZs[1] + dCost_dZs[2];
  CHECK_NEAR(gradSum, 0.0f, 1e-5f, "dsoftmax gradients sum to 0");
}

//===================================================================================================================//

void runActvFuncTests()
{
  testNameToType();
  testTypeToName();
  testReLU();
  testSigmoid();
  testTanh();
  testSoftmax();
  testSoftmaxBackward();
}
