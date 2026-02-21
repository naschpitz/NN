#include "test_helpers.hpp"

//===================================================================================================================//

static void testGPUTrainSimple() {
  std::cout << "--- testGPUTrainSimple ---" << std::endl;

  // Simple 2→4→1 network trained on GPU with sigmoid
  ANN::Samples<float> samples = {
    {{1.0f, 1.0f}, {1.0f}},
    {{0.0f, 0.0f}, {0.0f}}
  };

  bool converged = false;
  ANN::Output<float> p0, p1;

  for (int attempt = 0; attempt < 5 && !converged; ++attempt) {
    if (attempt > 0) std::cout << "  retry #" << attempt << std::endl;

    ANN::CoreConfig<float> config;
    config.modeType = ANN::ModeType::TRAIN;
    config.deviceType = ANN::DeviceType::GPU;
    config.layersConfig = makeLayersConfig({
      {2, ANN::ActvFuncType::RELU},
      {4, ANN::ActvFuncType::SIGMOID},
      {1, ANN::ActvFuncType::SIGMOID}
    });
    config.trainingConfig.numEpochs = 500;
    config.trainingConfig.learningRate = 0.5f;
    config.trainingConfig.progressReports = 0;
    config.trainingConfig.numGPUs = 1;
    config.verbose = false;

    auto core = ANN::Core<float>::makeCore(config);
    core->train(samples);

    p0 = core->predict({1.0f, 1.0f});
    p1 = core->predict({0.0f, 0.0f});

    if (p0[0] > 0.7f && p1[0] < 0.3f) converged = true;
  }

  std::cout << "  high=" << p0[0] << "  low=" << p1[0] << std::endl;
  CHECK(converged, "GPU train converged (5 attempts)");
}

//===================================================================================================================//

static void testGPUPredict() {
  std::cout << "--- testGPUPredict ---" << std::endl;

  // Create a predict-only GPU core with known weights
  ANN::CoreConfig<float> config;
  config.modeType = ANN::ModeType::PREDICT;
  config.deviceType = ANN::DeviceType::GPU;
  config.layersConfig = makeLayersConfig({{2, ANN::ActvFuncType::RELU}, {1, ANN::ActvFuncType::SIGMOID}});

  config.parameters.weights.resize(2);
  config.parameters.weights[1] = {{0.5f, 0.5f}};
  config.parameters.biases.resize(2);
  config.parameters.biases[1] = {0.0f};
  config.verbose = false;

  auto core = ANN::Core<float>::makeCore(config);
  ANN::Output<float> out = core->predict({1.0f, 1.0f});

  CHECK(out.size() == 1, "GPU predict output size = 1");
  // z = 0.5*1 + 0.5*1 = 1.0, sigmoid(1.0) ≈ 0.7311
  float expected = 1.0f / (1.0f + std::exp(-1.0f));
  CHECK_NEAR(out[0], expected, 0.01f, "GPU predict sigmoid(1.0) ≈ 0.7311");
  std::cout << "  pred=" << out[0] << "  expected=" << expected << std::endl;
}

//===================================================================================================================//

static void testGPUvsCPUParity() {
  std::cout << "--- testGPUvsCPUParity ---" << std::endl;

  // Train on CPU, then create both CPU and GPU predict cores with same params
  ANN::CoreConfig<float> trainConfig;
  trainConfig.modeType = ANN::ModeType::TRAIN;
  trainConfig.deviceType = ANN::DeviceType::CPU;
  trainConfig.layersConfig = makeLayersConfig({
    {2, ANN::ActvFuncType::RELU},
    {4, ANN::ActvFuncType::SIGMOID},
    {1, ANN::ActvFuncType::SIGMOID}
  });
  trainConfig.trainingConfig.numEpochs = 200;
  trainConfig.trainingConfig.learningRate = 0.5f;
  trainConfig.trainingConfig.progressReports = 0;
  trainConfig.verbose = false;

  ANN::Samples<float> samples = {
    {{1.0f, 1.0f}, {1.0f}},
    {{0.0f, 0.0f}, {0.0f}}
  };

  auto trainCore = ANN::Core<float>::makeCore(trainConfig);
  trainCore->train(samples);

  ANN::Parameters<float> params = trainCore->getParameters();

  // CPU predict
  ANN::CoreConfig<float> cpuConfig;
  cpuConfig.modeType = ANN::ModeType::PREDICT;
  cpuConfig.deviceType = ANN::DeviceType::CPU;
  cpuConfig.layersConfig = trainConfig.layersConfig;
  cpuConfig.parameters = params;
  cpuConfig.verbose = false;

  auto cpuCore = ANN::Core<float>::makeCore(cpuConfig);
  ANN::Output<float> cpuPred1 = cpuCore->predict({1.0f, 1.0f});
  ANN::Output<float> cpuPred2 = cpuCore->predict({0.0f, 0.0f});

  // GPU predict
  ANN::CoreConfig<float> gpuConfig;
  gpuConfig.modeType = ANN::ModeType::PREDICT;
  gpuConfig.deviceType = ANN::DeviceType::GPU;
  gpuConfig.layersConfig = trainConfig.layersConfig;
  gpuConfig.parameters = params;
  gpuConfig.verbose = false;

  auto gpuCore = ANN::Core<float>::makeCore(gpuConfig);
  ANN::Output<float> gpuPred1 = gpuCore->predict({1.0f, 1.0f});
  ANN::Output<float> gpuPred2 = gpuCore->predict({0.0f, 0.0f});

  std::cout << "  CPU[1,1]=" << cpuPred1[0] << "  GPU[1,1]=" << gpuPred1[0] << std::endl;
  std::cout << "  CPU[0,0]=" << cpuPred2[0] << "  GPU[0,0]=" << gpuPred2[0] << std::endl;

  CHECK_NEAR(cpuPred1[0], gpuPred1[0], 0.01f, "CPU vs GPU parity [1,1]");
  CHECK_NEAR(cpuPred2[0], gpuPred2[0], 0.01f, "CPU vs GPU parity [0,0]");
}

//===================================================================================================================//

void runGPUTests() {
  testGPUTrainSimple();
  testGPUPredict();
  testGPUvsCPUParity();
}

