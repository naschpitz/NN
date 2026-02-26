#include "test_helpers.hpp"

//===================================================================================================================//

static void testDeviceNameToType() {
  std::cout << "--- testDeviceNameToType ---" << std::endl;

  CHECK(ANN::Device::nameToType("cpu") == ANN::DeviceType::CPU, "cpu → CPU");
  CHECK(ANN::Device::nameToType("gpu") == ANN::DeviceType::GPU, "gpu → GPU");

  CHECK_THROWS(ANN::Device::nameToType("nonexistent"), "nonexistent throws");
  CHECK_THROWS(ANN::Device::nameToType(""), "empty throws");
}

//===================================================================================================================//

static void testDeviceTypeToName() {
  std::cout << "--- testDeviceTypeToName ---" << std::endl;

  CHECK(ANN::Device::typeToName(ANN::DeviceType::CPU) == "cpu", "CPU → cpu");
  CHECK(ANN::Device::typeToName(ANN::DeviceType::GPU) == "gpu", "GPU → gpu");
}

//===================================================================================================================//

static void testModeNameToType() {
  std::cout << "--- testModeNameToType ---" << std::endl;

  CHECK(ANN::Mode::nameToType("train") == ANN::ModeType::TRAIN, "train → TRAIN");
  CHECK(ANN::Mode::nameToType("predict") == ANN::ModeType::PREDICT, "predict → PREDICT");
  CHECK(ANN::Mode::nameToType("test") == ANN::ModeType::TEST, "test → TEST");

  CHECK_THROWS(ANN::Mode::nameToType("nonexistent"), "nonexistent throws");
}

//===================================================================================================================//

static void testModeTypeToName() {
  std::cout << "--- testModeTypeToName ---" << std::endl;

  CHECK(ANN::Mode::typeToName(ANN::ModeType::TRAIN) == "train", "TRAIN → train");
  CHECK(ANN::Mode::typeToName(ANN::ModeType::PREDICT) == "predict", "PREDICT → predict");
  CHECK(ANN::Mode::typeToName(ANN::ModeType::TEST) == "test", "TEST → test");
}

//===================================================================================================================//

static void testLayersConfigGetTotalNumNeurons() {
  std::cout << "--- testLayersConfigGetTotalNumNeurons ---" << std::endl;

  ANN::LayersConfig config;
  config.push_back({3, ANN::ActvFuncType::RELU});
  config.push_back({5, ANN::ActvFuncType::RELU});
  config.push_back({2, ANN::ActvFuncType::SIGMOID});

  CHECK(config.getTotalNumNeurons() == 10, "getTotalNumNeurons = 3+5+2 = 10");

  ANN::LayersConfig empty;
  CHECK(empty.getTotalNumNeurons() == 0, "empty config getTotalNumNeurons = 0");
}

//===================================================================================================================//

static void testFormatDuration() {
  std::cout << "--- testFormatDuration ---" << std::endl;

  CHECK(ANN::Utils<double>::formatDuration(0.0) == "0s", "0s");
  CHECK(ANN::Utils<double>::formatDuration(-5.0) == "0s", "negative → 0s");
  CHECK(ANN::Utils<double>::formatDuration(45.0) == "45s", "45s");
  CHECK(ANN::Utils<double>::formatDuration(90.0) == "1m 30s", "1m 30s");
  CHECK(ANN::Utils<double>::formatDuration(3661.0) == "1h 1m 1s", "1h 1m 1s");
  CHECK(ANN::Utils<double>::formatDuration(86400.0) == "1d", "1d");
  CHECK(ANN::Utils<double>::formatDuration(60.0) == "1m", "1m exactly");
}

//===================================================================================================================//

static void testFlattenUnflattenTensor2D() {
  std::cout << "--- testFlattenUnflattenTensor2D ---" << std::endl;

  ANN::Tensor2D<double> original = {{1.0, 2.0, 3.0}, {4.0, 5.0}};
  ANN::Tensor1D<double> flat = ANN::Utils<double>::flatten(original);

  CHECK(flat.size() == 5, "flatten 2D size = 5");
  CHECK_NEAR(flat[0], 1.0, 1e-10, "flat[0] = 1.0");
  CHECK_NEAR(flat[4], 5.0, 1e-10, "flat[4] = 5.0");

  // Unflatten back
  ANN::Tensor2D<double> restored = {{0.0, 0.0, 0.0}, {0.0, 0.0}};
  ANN::Utils<double>::unflatten(flat, restored);

  CHECK_NEAR(restored[0][0], 1.0, 1e-10, "restored[0][0] = 1.0");
  CHECK_NEAR(restored[0][2], 3.0, 1e-10, "restored[0][2] = 3.0");
  CHECK_NEAR(restored[1][0], 4.0, 1e-10, "restored[1][0] = 4.0");
  CHECK_NEAR(restored[1][1], 5.0, 1e-10, "restored[1][1] = 5.0");
}

//===================================================================================================================//

static void testFlattenUnflattenTensor3D() {
  std::cout << "--- testFlattenUnflattenTensor3D ---" << std::endl;

  ANN::Tensor3D<double> original = {{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}};
  ANN::Tensor1D<double> flat = ANN::Utils<double>::flatten(original);

  CHECK(flat.size() == 8, "flatten 3D size = 8");
  CHECK_NEAR(flat[0], 1.0, 1e-10, "flat[0] = 1.0");
  CHECK_NEAR(flat[7], 8.0, 1e-10, "flat[7] = 8.0");

  // Unflatten back
  ANN::Tensor3D<double> restored = {{{0.0, 0.0}, {0.0, 0.0}}, {{0.0, 0.0}, {0.0, 0.0}}};
  ANN::Utils<double>::unflatten(flat, restored);

  CHECK_NEAR(restored[0][0][0], 1.0, 1e-10, "restored[0][0][0] = 1.0");
  CHECK_NEAR(restored[1][1][1], 8.0, 1e-10, "restored[1][1][1] = 8.0");
}

//===================================================================================================================//

static void testCount() {
  std::cout << "--- testCount ---" << std::endl;

  ANN::Tensor2D<double> t2d = {{1.0, 2.0}, {3.0, 4.0, 5.0}};
  CHECK(ANN::Utils<double>::count(t2d) == 5, "count 2D = 5");

  ANN::Tensor3D<double> t3d = {{{1.0, 2.0}, {3.0}}, {{4.0, 5.0, 6.0}}};
  CHECK(ANN::Utils<double>::count(t3d) == 6, "count 3D = 6");
}

//===================================================================================================================//

static void testFormatISO8601() {
  std::cout << "--- testFormatISO8601 ---" << std::endl;

  std::string iso = ANN::Utils<double>::formatISO8601();
  // Basic format: "2026-02-21T..." — just verify it's non-empty and starts with "20"
  CHECK(!iso.empty(), "formatISO8601 non-empty");
  CHECK(iso.substr(0, 2) == "20", "formatISO8601 starts with 20");
  CHECK(iso.find('T') != std::string::npos, "formatISO8601 contains T separator");
}

//===================================================================================================================//

void runUtilsTests() {
  testDeviceNameToType();
  testDeviceTypeToName();
  testModeNameToType();
  testModeTypeToName();
  testLayersConfigGetTotalNumNeurons();
  testFormatDuration();
  testFlattenUnflattenTensor2D();
  testFlattenUnflattenTensor3D();
  testCount();
  testFormatISO8601();
}

