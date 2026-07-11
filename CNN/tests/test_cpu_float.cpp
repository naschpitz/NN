#include "test_helpers.hpp"

#include "CNN_Core.hpp"
#include "CNN_CoreConfig.hpp"
#include "CNN_LayersConfig.hpp"
#include "CNN_SlidingStrategy.hpp"
#include "ANN_ActvFunc.hpp"

#include <cmath>
#include <limits>

//===================================================================================================================//

//
// Float-precision tests.
//
// All existing tests use <double>. Production models (train-app-13) use <float>.
// Float-specific risks: Adam epsilon 1e-8 near float epsilon (1.19e-7),
// reduced precision in sqrt/exp/log, accumulated rounding.
//
// These tests verify the <float> path produces sensible results: predictions
// are finite, softmax sums to ~1, and training reduces loss.
//

static void testFloatPredictProducesValidOutput()
{
  TestScope _t("testFloatPredictProducesValidOutput");

  CNN::CoreConfig<float> config;
  config.modeType = Common::ModeType::PREDICT;
  config.deviceType = Common::DeviceType::CPU;
  config.inputShape = {1, 4, 4};
  config.progressReports = 0;

  config.layersConfig.cnnLayers.push_back(
    {CNN::LayerType::CONV, CNN::ConvLayerConfig{2, 3, 3, 1, 1, CNN::SlidingStrategyType::VALID}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::INSTANCENORM, CNN::NormLayerConfig{}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::RELU, CNN::ReLULayerConfig{}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::FLATTEN, CNN::FlattenLayerConfig{}});

  config.layersConfig.denseLayers.push_back({2, ANN::ActvFuncType::SOFTMAX});

  auto core = CNN::Core<float>::makeCore(config);
  CHECK(core != nullptr, "float CNN core created");

  CNN::Input<float> input(config.inputShape);
  input.data.resize(config.inputShape.size());

  for (size_t i = 0; i < input.data.size(); ++i)
    input.data[i] = static_cast<float>(i) * 0.1f;

  auto result = core->predict(input);

  CHECK(result.output.size() == 2, "float output size = 2");

  // All outputs must be finite (no NaN, no inf)
  for (size_t i = 0; i < result.output.size(); ++i) {
    CHECK(std::isfinite(result.output[i]), "float output[" + std::to_string(i) + "] is finite");
  }

  // Softmax outputs must sum to approximately 1
  float sum = 0.0f;

  for (size_t i = 0; i < result.output.size(); ++i)
    sum += result.output[i];

  CHECK_NEAR(sum, 1.0f, 1e-4f, "float softmax sums to 1");

  // Logits must be finite
  for (size_t i = 0; i < result.logits.size(); ++i) {
    CHECK(std::isfinite(result.logits[i]), "float logits[" + std::to_string(i) + "] is finite");
  }

  std::cout << "  float output: [" << result.output[0] << ", " << result.output[1] << "]" << std::endl;
}

//===================================================================================================================//

static void testFloatTrainingReducesLoss()
{
  TestScope _t("testFloatTrainingReducesLoss");

  CNN::CoreConfig<float> config;
  config.modeType = Common::ModeType::TRAIN;
  config.deviceType = Common::DeviceType::CPU;
  config.inputShape = {1, 4, 4};
  config.logLevel = Common::LogLevel::ERROR;
  config.numThreads = 1;
  config.progressReports = 0;

  config.layersConfig.cnnLayers.push_back(
    {CNN::LayerType::CONV, CNN::ConvLayerConfig{2, 3, 3, 1, 1, CNN::SlidingStrategyType::VALID}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::RELU, CNN::ReLULayerConfig{}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::FLATTEN, CNN::FlattenLayerConfig{}});

  config.layersConfig.denseLayers.push_back({2, ANN::ActvFuncType::SOFTMAX});

  config.costFunctionConfig.type = Common::CostFunctionType::CROSS_ENTROPY;
  config.trainConfig.numEpochs = 100;
  config.trainConfig.learningRate = 0.5f;
  config.trainConfig.optimizer.type = Common::OptimizerType::ADAM;

  auto core = CNN::Core<float>::makeCore(config);
  CHECK(core != nullptr, "float CNN core created");

  // Distinct gradient-pattern inputs (distinguishable after conv)
  CNN::Samples<float> samples(2);
  samples[0].input = CNN::Tensor3D<float>({1, 4, 4});
  samples[1].input = CNN::Tensor3D<float>({1, 4, 4});

  for (ulong i = 0; i < 16; ++i) {
    samples[0].input.data[i] = static_cast<float>(i) * 0.1f;
    samples[1].input.data[i] = static_cast<float>(15 - i) * 0.1f;
  }

  samples[0].output = {1.0f, 0.0f};
  samples[1].output = {0.0f, 1.0f};

  // Measure pre-training prediction
  auto preResult = core->predict(samples[0].input);
  float preLoss = 0.0f;

  for (size_t i = 0; i < preResult.output.size(); ++i)
    preLoss -= samples[0].output[i] * std::log(std::max(preResult.output[i], 1e-7f));

  // Train
  core->train(samples.size(), CNN::makeSampleProvider(samples));

  // Measure post-training prediction
  auto postResult = core->predict(samples[0].input);
  float postLoss = 0.0f;

  for (size_t i = 0; i < postResult.output.size(); ++i)
    postLoss -= samples[0].output[i] * std::log(std::max(postResult.output[i], 1e-7f));

  // Loss must decrease (training is working)
  std::cout << "  float loss: " << preLoss << " → " << postLoss << std::endl;
  CHECK(postLoss < preLoss, "float training reduces loss");

  // Output must be finite
  for (size_t i = 0; i < postResult.output.size(); ++i) {
    CHECK(std::isfinite(postResult.output[i]), "float post-train output finite");
  }

  // Model should lean towards correct class for sample 0
  CHECK(postResult.output[0] > preResult.output[0], "float training improved class 0 confidence");
}

//===================================================================================================================//

static void testFloatSerializationRoundTrip()
{
  TestScope _t("testFloatSerializationRound Trip");

  CNN::CoreConfig<float> config;
  config.modeType = Common::ModeType::PREDICT;
  config.deviceType = Common::DeviceType::CPU;
  config.inputShape = {1, 4, 4};
  config.progressReports = 0;

  config.layersConfig.cnnLayers.push_back(
    {CNN::LayerType::CONV, CNN::ConvLayerConfig{2, 3, 3, 1, 1, CNN::SlidingStrategyType::VALID}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::INSTANCENORM, CNN::NormLayerConfig{}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::RELU, CNN::ReLULayerConfig{}});
  config.layersConfig.cnnLayers.push_back({CNN::LayerType::FLATTEN, CNN::FlattenLayerConfig{}});

  config.layersConfig.denseLayers.push_back({2, ANN::ActvFuncType::SOFTMAX});

  auto coreA = CNN::Core<float>::makeCore(config);
  CHECK(coreA != nullptr, "float coreA created");

  CNN::Input<float> input(config.inputShape);
  input.data.resize(config.inputShape.size());

  for (size_t i = 0; i < input.data.size(); ++i)
    input.data[i] = static_cast<float>(i) * 0.05f;

  auto resultA = coreA->predict(input);

  // Verify float predictions are finite and sum to ~1
  for (size_t i = 0; i < resultA.output.size(); ++i)
    CHECK(std::isfinite(resultA.output[i]), "float resultA finite");

  // Extract params and rebuild
  CNN::CoreConfig<float> loadedConfig = config;
  loadedConfig.parameters = coreA->getParameters();

  auto coreB = CNN::Core<float>::makeCore(loadedConfig);
  CHECK(coreB != nullptr, "float coreB created from loaded params");

  auto resultB = coreB->predict(input);

  // Predictions must match (same params, same input)
  for (size_t i = 0; i < resultA.output.size(); ++i) {
    CHECK_NEAR(resultA.output[i], resultB.output[i], 1e-5f, "float round-trip output[" + std::to_string(i) + "]");
  }

  for (size_t i = 0; i < resultA.logits.size(); ++i) {
    CHECK_NEAR(resultA.logits[i], resultB.logits[i], 1e-5f, "float round-trip logits[" + std::to_string(i) + "]");
  }
}

//===================================================================================================================//

void runFloatPrecisionTests()
{
  testFloatPredictProducesValidOutput();
  testFloatTrainingReducesLoss();
  testFloatSerializationRoundTrip();
}
