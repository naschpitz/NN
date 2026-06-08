#include "test_helpers.hpp"
#include "NN-CLI_Types.hpp"

#include <QDir>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>

#include <json.hpp>

// CNN/ library headers for direct instantiation tests
#include <CNN_Core.hpp>
#include <CNN_CoreConfig.hpp>
#include <CNN_CoreGPU.hpp>
#include <CNN_CoreGPUWorker.hpp>
#include <CNN_GPUBufferManager.hpp>
#include <CNN_Sample.hpp>
#include <_CoreGPUWorker.hpp>

using NN_CLI::ulong;

//===================================================================================================================//
// Deep diagnostic test: compare ALL internal state between CPU and GPU predict
//===================================================================================================================//

// Helper: compare two float vectors element-by-element, report first mismatch
static bool compareVectors(const std::string& label, const std::vector<float>& cpu, const std::vector<float>& gpu,
                           float tol = 1e-4f, int maxReport = 5)
{
  if (cpu.size() != gpu.size()) {
    std::cerr << "    " << label << ": SIZE MISMATCH cpu=" << cpu.size() << " gpu=" << gpu.size() << std::endl;
    return false;
  }

  int mismatches = 0;
  float maxDiff = 0;

  for (size_t i = 0; i < cpu.size(); i++) {
    float diff = std::fabs(cpu[i] - gpu[i]);

    if (diff > maxDiff)
      maxDiff = diff;

    if (diff > tol) {
      if (mismatches < maxReport)
        std::cerr << "    " << label << "[" << i << "]: cpu=" << cpu[i] << " gpu=" << gpu[i] << " diff=" << diff
                  << std::endl;
      mismatches++;
    }
  }

  if (mismatches > 0) {
    std::cerr << "    " << label << ": " << mismatches << "/" << cpu.size() << " mismatches (maxDiff=" << maxDiff << ")"
              << std::endl;
    return false;
  }

  return true;
}

static void testCNNGPUPredictDeepDiagnostic()
{
  std::cout << "  testCNNGPUPredictDeepDiagnostic..." << std::endl;

  if (!checkGPUAvailable()) {
    std::cout << "    (skipped — no GPU)" << std::endl;
    return;
  }

  // Build config programmatically: conv(4,3x3,same) → BN → ReLU → MaxPool(2x2)
  //                                → conv(8,3x3,same) → BN → ReLU → AvgPool(2x2)
  //                                → Flatten → Dense(4,relu) → Dense(2,sigmoid)
  // Input: 1x8x8
  CNN::CoreConfig<float> trainConfig;
  trainConfig.modeType = Common::ModeType::TRAIN;
  trainConfig.deviceType = Common::DeviceType::CPU;
  trainConfig.inputShape = {1, 8, 8};
  trainConfig.progressReports = 0;
  trainConfig.logLevel = Common::LogLevel::ERROR;

  // CNN layers
  CNN::CNNLayerConfig conv1;
  conv1.type = CNN::LayerType::CONV;
  conv1.config = CNN::ConvLayerConfig{4, 3, 3, 1, 1, CNN::SlidingStrategyType::SAME};

  CNN::CNNLayerConfig bn1;
  bn1.type = CNN::LayerType::INSTANCENORM;
  bn1.config = CNN::NormLayerConfig{1e-5f, 0.1f};

  CNN::CNNLayerConfig relu1;
  relu1.type = CNN::LayerType::RELU;
  relu1.config = CNN::ReLULayerConfig{};

  CNN::CNNLayerConfig maxpool1;
  maxpool1.type = CNN::LayerType::POOL;
  maxpool1.config = CNN::PoolLayerConfig{CNN::PoolTypeEnum::MAX, 2, 2, 2, 2};

  CNN::CNNLayerConfig conv2;
  conv2.type = CNN::LayerType::CONV;
  conv2.config = CNN::ConvLayerConfig{8, 3, 3, 1, 1, CNN::SlidingStrategyType::SAME};

  CNN::CNNLayerConfig bn2;
  bn2.type = CNN::LayerType::INSTANCENORM;
  bn2.config = CNN::NormLayerConfig{1e-5f, 0.1f};

  CNN::CNNLayerConfig relu2;
  relu2.type = CNN::LayerType::RELU;
  relu2.config = CNN::ReLULayerConfig{};

  CNN::CNNLayerConfig avgpool1;
  avgpool1.type = CNN::LayerType::POOL;
  avgpool1.config = CNN::PoolLayerConfig{CNN::PoolTypeEnum::AVG, 2, 2, 2, 2};

  CNN::CNNLayerConfig flatten;
  flatten.type = CNN::LayerType::FLATTEN;
  flatten.config = CNN::FlattenLayerConfig{};

  trainConfig.layersConfig.cnnLayers = {conv1, bn1, relu1, maxpool1, conv2, bn2, relu2, avgpool1, flatten};
  trainConfig.layersConfig.denseLayers = {{4, ::ActvFuncType::RELU}, {2, ::ActvFuncType::SIGMOID}};

  trainConfig.trainingConfig.numEpochs = 50;
  trainConfig.trainingConfig.learningRate = 0.01f;
  trainConfig.trainingConfig.batchSize = 4;
  trainConfig.trainingConfig.shuffleSamples = false;

  // Create training samples: 1x8x8 input, 2-class output
  CNN::Samples<float> samples(4);

  for (int s = 0; s < 4; s++) {
    samples[s].input = CNN::Input<float>({1, 8, 8});
    int cls = s / 2;

    for (int i = 0; i < 64; i++) {
      if (cls == 0)
        samples[s].input.data[i] = static_cast<float>(i) / 63.0f + 0.01f * s;
      else
        samples[s].input.data[i] = static_cast<float>(63 - i) / 63.0f + 0.01f * (s - 2);
    }

    samples[s].output = (cls == 0) ? CNN::Output<float>{1.0f, 0.0f} : CNN::Output<float>{0.0f, 1.0f};
  }

  // Step 1: Train on CPU
  std::cout << "    Training on CPU..." << std::flush;
  auto cpuTrainCore = CNN::Core<float>::makeCore(trainConfig);
  cpuTrainCore->train(samples.size(), CNN::makeSampleProvider(samples));
  const CNN::Parameters<float>& trainedParams = cpuTrainCore->getParameters();
  std::cout << " done" << std::endl;

  // Step 2: Create CPU predict core with trained parameters
  CNN::CoreConfig<float> cpuPredConfig;
  cpuPredConfig.modeType = Common::ModeType::PREDICT;
  cpuPredConfig.deviceType = Common::DeviceType::CPU;
  cpuPredConfig.inputShape = trainConfig.inputShape;
  cpuPredConfig.layersConfig = trainConfig.layersConfig;
  cpuPredConfig.parameters = trainedParams;
  cpuPredConfig.progressReports = 0;
  cpuPredConfig.logLevel = Common::LogLevel::ERROR;

  auto cpuCore = CNN::Core<float>::makeCore(cpuPredConfig);

  // Step 3: Create GPU predict core with same trained parameters
  CNN::CoreConfig<float> gpuPredConfig = cpuPredConfig;
  gpuPredConfig.deviceType = Common::DeviceType::GPU;

  auto gpuCoreBase = CNN::Core<float>::makeCore(gpuPredConfig);
  auto* gpuCore = dynamic_cast<CNN::CoreGPU<float>*>(gpuCoreBase.get());
  CHECK(gpuCore != nullptr, "Deep diag: GPU core created");

  if (!gpuCore) {
    std::cout << std::endl;
    return;
  }

  auto* gpuWorker = gpuCore->getWorker(0);
  CHECK(gpuWorker != nullptr, "Deep diag: GPU worker accessible");

  if (!gpuWorker) {
    std::cout << std::endl;
    return;
  }

  // Step 4: Compare parameters BEFORE predict (verify they were loaded correctly)
  std::cout << "    Comparing parameters..." << std::flush;
  const auto& cpuParams = cpuCore->getParameters();
  const auto& gpuParams = gpuWorker->getParameters();
  bool paramsOk = true;

  // Conv filters
  for (size_t li = 0; li < cpuParams.convParams.size(); li++) {
    if (!compareVectors("conv[" + std::to_string(li) + "].filters", cpuParams.convParams[li].filters,
                        gpuParams.convParams[li].filters))
      paramsOk = false;

    if (!compareVectors("conv[" + std::to_string(li) + "].biases", cpuParams.convParams[li].biases,
                        gpuParams.convParams[li].biases))
      paramsOk = false;
  }

  // InstanceNorm params
  for (size_t li = 0; li < cpuParams.normParams.size(); li++) {
    if (!compareVectors("bn[" + std::to_string(li) + "].gamma", cpuParams.normParams[li].gamma,
                        gpuParams.normParams[li].gamma))
      paramsOk = false;

    if (!compareVectors("bn[" + std::to_string(li) + "].beta", cpuParams.normParams[li].beta,
                        gpuParams.normParams[li].beta))
      paramsOk = false;

    if (!compareVectors("bn[" + std::to_string(li) + "].runningMean", cpuParams.normParams[li].runningMean,
                        gpuParams.normParams[li].runningMean))
      paramsOk = false;

    if (!compareVectors("bn[" + std::to_string(li) + "].runningVar", cpuParams.normParams[li].runningVar,
                        gpuParams.normParams[li].runningVar))
      paramsOk = false;
  }

  //  weights and biases
  for (size_t li = 0; li < cpuParams.denseParams.weights.size(); li++) {
    for (size_t ni = 0; ni < cpuParams.denseParams.weights[li].size(); ni++) {
      if (!compareVectors("ann.weights[" + std::to_string(li) + "][" + std::to_string(ni) + "]",
                          cpuParams.denseParams.weights[li][ni], gpuParams.denseParams.weights[li][ni]))
        paramsOk = false;
    }

    std::vector<float> cpuBiases(cpuParams.denseParams.biases[li].begin(), cpuParams.denseParams.biases[li].end());
    std::vector<float> gpuBiases(gpuParams.denseParams.biases[li].begin(), gpuParams.denseParams.biases[li].end());

    if (!compareVectors("ann.biases[" + std::to_string(li) + "]", cpuBiases, gpuBiases))
      paramsOk = false;
  }

  CHECK(paramsOk, "Deep diag: all parameters match between CPU and GPU");
  std::cout << (paramsOk ? " OK" : " MISMATCH") << std::endl;

  // Step 5: Run predict on both and compare outputs + intermediate activations
  CNN::Input<float> testInput({1, 8, 8});

  for (int i = 0; i < 64; i++)
    testInput.data[i] = static_cast<float>(i) / 63.0f;

  std::cout << "    Running predict on CPU and GPU..." << std::flush;
  auto cpuOutput = cpuCore->predict(testInput).output;
  auto gpuOutput = gpuCore->predict(testInput).output;
  std::cout << " done" << std::endl;

  // Compare final outputs
  bool outputsMatch = compareVectors("final_output", cpuOutput, gpuOutput, 1e-4f);
  CHECK(outputsMatch, "Deep diag: CPU and GPU final outputs match");

  // Step 6: Read GPU intermediate activations and compare with CPU
  // We need to re-run CPU predict step-by-step to get intermediates
  std::cout << "    Comparing layer-by-layer activations..." << std::flush;

  // Read ALL GPU activations from the cnn_actvs buffer
  auto& bm = *gpuWorker->bufferManager;
  std::vector<float> gpuAllActvs(bm.totalActvSize);
  gpuWorker->readGPUBuffer<float>("cnn_actvs", gpuAllActvs, 0);

  // Read GPU filters from buffer (to verify they match what was uploaded)
  if (bm.totalFilterSize > 0) {
    std::vector<float> gpuFilters(bm.totalFilterSize);
    gpuWorker->readGPUBuffer<float>("cnn_filters", gpuFilters, 0);

    // Reconstruct expected flat filters from CPU params
    std::vector<float> cpuFlatFilters;

    for (size_t ci = 0; ci < bm.convInfos.size(); ci++) {
      cpuFlatFilters.insert(cpuFlatFilters.end(), cpuParams.convParams[ci].filters.begin(),
                            cpuParams.convParams[ci].filters.end());
    }

    bool filtersOnGPUMatch = compareVectors("gpu_buffer_filters", cpuFlatFilters, gpuFilters);
    CHECK(filtersOnGPUMatch, "Deep diag: GPU buffer filters match CPU params");
  }

  // Read GPU biases from buffer
  if (bm.totalBiasSize > 0) {
    std::vector<float> gpuBiases(bm.totalBiasSize);
    gpuWorker->readGPUBuffer<float>("cnn_biases", gpuBiases, 0);

    std::vector<float> cpuFlatBiases;

    for (size_t ci = 0; ci < bm.convInfos.size(); ci++) {
      cpuFlatBiases.insert(cpuFlatBiases.end(), cpuParams.convParams[ci].biases.begin(),
                           cpuParams.convParams[ci].biases.end());
    }

    bool biasesOnGPUMatch = compareVectors("gpu_buffer_biases", cpuFlatBiases, gpuBiases);
    CHECK(biasesOnGPUMatch, "Deep diag: GPU buffer biases match CPU params");
  }

  // Read GPU BN params from buffers
  if (bm.totalNormParamSize > 0) {
    std::vector<float> gpuGamma(bm.totalNormParamSize), gpuBeta(bm.totalNormParamSize);
    std::vector<float> gpuRunMean(bm.totalNormParamSize), gpuRunVar(bm.totalNormParamSize);
    gpuWorker->readGPUBuffer<float>("cnn_norm_gamma", gpuGamma, 0);
    gpuWorker->readGPUBuffer<float>("cnn_norm_beta", gpuBeta, 0);
    gpuWorker->readGPUBuffer<float>("cnn_norm_running_mean", gpuRunMean, 0);
    gpuWorker->readGPUBuffer<float>("cnn_norm_running_var", gpuRunVar, 0);

    std::vector<float> cpuFlatGamma, cpuFlatBeta, cpuFlatRunMean, cpuFlatRunVar;

    for (size_t bi = 0; bi < bm.normInfos.size(); bi++) {
      cpuFlatGamma.insert(cpuFlatGamma.end(), cpuParams.normParams[bi].gamma.begin(),
                          cpuParams.normParams[bi].gamma.end());
      cpuFlatBeta.insert(cpuFlatBeta.end(), cpuParams.normParams[bi].beta.begin(), cpuParams.normParams[bi].beta.end());
      cpuFlatRunMean.insert(cpuFlatRunMean.end(), cpuParams.normParams[bi].runningMean.begin(),
                            cpuParams.normParams[bi].runningMean.end());
      cpuFlatRunVar.insert(cpuFlatRunVar.end(), cpuParams.normParams[bi].runningVar.begin(),
                           cpuParams.normParams[bi].runningVar.end());
    }

    CHECK(compareVectors("gpu_buffer_bn_gamma", cpuFlatGamma, gpuGamma), "Deep diag: GPU BN gamma matches");
    CHECK(compareVectors("gpu_buffer_bn_beta", cpuFlatBeta, gpuBeta), "Deep diag: GPU BN beta matches");
    CHECK(compareVectors("gpu_buffer_bn_running_mean", cpuFlatRunMean, gpuRunMean),
          "Deep diag: GPU BN running mean matches");
    CHECK(compareVectors("gpu_buffer_bn_running_var", cpuFlatRunVar, gpuRunVar),
          "Deep diag: GPU BN running var matches");
  }

  // Compare per-layer activations
  // GPU stores activations in a flat buffer with offsets per layer.
  // CPU computes them step-by-step. We need to manually propagate on CPU to get intermediates.
  // Since CoreCPUWorker::propagateCNN is private, we'll use the Loader to load the model
  // and compare via the public API. For now, compare the GPU layer activations for sanity:
  bool anyLayerAllZero = false;

  for (size_t li = 0; li < bm.layerInfos.size(); li++) {
    ulong offset = bm.layerInfos[li].actvOffset;
    ulong size = bm.layerInfos[li].actvSize;

    float sum = 0, minVal = 1e30f, maxVal = -1e30f;

    for (ulong j = 0; j < size && (offset + j) < gpuAllActvs.size(); j++) {
      float v = gpuAllActvs[offset + j];
      sum += v;

      if (v < minVal)
        minVal = v;

      if (v > maxVal)
        maxVal = v;
    }

    bool allZero = (minVal == 0.0f && maxVal == 0.0f);

    if (allZero && li > 0) { // layer 0 is input, can be zero
      std::cerr << "    Layer " << li << " (offset=" << offset << ", size=" << size << "): ALL ZEROS" << std::endl;
      anyLayerAllZero = true;
    }
  }

  CHECK(!anyLayerAllZero, "Deep diag: no intermediate layer is all zeros on GPU");

  // Step 7: Read  GPU buffers and compare with CPU  output
  auto* annGPUWorker = bm.annGPUWorker.get();

  if (annGPUWorker) {
    auto& annBM = *annGPUWorker->bufferManager;

    // Read  activations from GPU
    ulong totalNeurons = 0;

    for (size_t li = 0; li < cpuParams.denseParams.weights.size(); li++)
      totalNeurons += cpuParams.denseParams.weights[li].size();

    // Add input layer neurons (= flattenSize)
    totalNeurons += bm.flattenSize;

    if (totalNeurons > 0) {
      std::vector<float> gpuActvs(totalNeurons);
      annGPUWorker->readGPUBuffer<float>("actvs", gpuActvs, 0);

      // Check  input layer (should be the flattened CNN output)
      std::vector<float> gpuInput(gpuActvs.begin(), gpuActvs.begin() + static_cast<long>(bm.flattenSize));

      // The CNN output on GPU should be at the last CNN layer's activation
      ulong lastCNNLayerIdx = bm.layerInfos.size() - 1;
      ulong lastOffset = bm.layerInfos[lastCNNLayerIdx].actvOffset;
      ulong lastSize = bm.layerInfos[lastCNNLayerIdx].actvSize;
      std::vector<float> gpuCNNOutput(gpuAllActvs.begin() + static_cast<long>(lastOffset),
                                      gpuAllActvs.begin() + static_cast<long>(lastOffset + lastSize));

      bool cnnToAnnBridgeOk = compareVectors("cnn_to_ann_bridge", gpuCNNOutput, gpuInput, 1e-6f);
      CHECK(cnnToAnnBridgeOk, "Deep diag: CNN→ bridge (flatten output ==  input)");

      // Read  weights from GPU
      ulong totalWeights = 0;

      for (size_t li = 0; li < cpuParams.denseParams.weights.size(); li++)

        for (size_t ni = 0; ni < cpuParams.denseParams.weights[li].size(); ni++)
          totalWeights += cpuParams.denseParams.weights[li][ni].size();

      if (totalWeights > 0) {
        std::vector<float> gpuWeights(totalWeights);
        annGPUWorker->readGPUBuffer<float>("weights", gpuWeights, 0);

        std::vector<float> cpuFlatWeights;

        for (size_t li = 0; li < cpuParams.denseParams.weights.size(); li++)

          for (size_t ni = 0; ni < cpuParams.denseParams.weights[li].size(); ni++)
            cpuFlatWeights.insert(cpuFlatWeights.end(), cpuParams.denseParams.weights[li][ni].begin(),
                                     cpuParams.denseParams.weights[li][ni].end());

        CHECK(compareVectors("ann_gpu_weights", cpuFlatWeights, gpuWeights),
              "Deep diag:  GPU weights match CPU");
      }

      // Read  biases from GPU
      ulong totalBiases = 0;

      for (size_t li = 0; li < cpuParams.denseParams.biases.size(); li++)
        totalBiases += cpuParams.denseParams.biases[li].size();

      if (totalBiases > 0) {
        std::vector<float> gpuBiases(totalBiases);
        annGPUWorker->readGPUBuffer<float>("biases", gpuBiases, 0);

        std::vector<float> cpuFlatBiases;

        for (size_t li = 0; li < cpuParams.denseParams.biases.size(); li++)
          cpuFlatBiases.insert(cpuFlatBiases.end(), cpuParams.denseParams.biases[li].begin(),
                                  cpuParams.denseParams.biases[li].end());

        CHECK(compareVectors("ann_gpu_biases", cpuFlatBiases, gpuBiases), "Deep diag:  GPU biases match CPU");
      }
    }
  }

  // Step 8: Run a second input and verify outputs still differ
  CNN::Input<float> testInput2({1, 8, 8});

  for (int i = 0; i < 64; i++)
    testInput2.data[i] = static_cast<float>(63 - i) / 63.0f;

  auto cpuOutput2 = cpuCore->predict(testInput2).output;
  auto gpuOutput2 = gpuCore->predict(testInput2).output;

  CHECK(compareVectors("final_output_2", cpuOutput2, gpuOutput2, 1e-4f),
        "Deep diag: CPU and GPU match on second input");

  // Verify diversity
  bool diverse = false;

  for (size_t i = 0; i < cpuOutput.size() && i < cpuOutput2.size(); i++)

    if (std::fabs(cpuOutput[i] - cpuOutput2[i]) > 1e-6f)
      diverse = true;

  CHECK(diverse, "Deep diag: different inputs produce different outputs");

  std::cout << "    All deep diagnostic checks complete." << std::endl;
  std::cout << std::endl;
}

void runCNNGPUDiagnosticTests()
{
  testCNNGPUPredictDeepDiagnostic();
}
