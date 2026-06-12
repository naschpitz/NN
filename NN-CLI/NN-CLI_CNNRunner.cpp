#include "NN-CLI_CNNRunner.hpp"

#include "NN-CLI_CNNLoader.hpp"
#include "NN-CLI_DataLoader.hpp"
#include "NN-CLI_GpuAugmenter.hpp"
#include "Common/Common_TrainingMonitor.hpp"
#include <OCLW_Core.hpp>
#include "NN-CLI_DataSplitter.hpp"
#include "NN-CLI_ImageLoader.hpp"
#include "NN-CLI_PredictSummary.hpp"
#include "NN-CLI_RunnerUtils.hpp"
#include "NN-CLI_TestSummary.hpp"
#include "NN-CLI_TrainingSummary.hpp"
#include "NN-CLI_Utils.hpp"

#include <QDir>
#include <QFile>
#include <QFileInfo>

#include <json.hpp>

#include "Common/Common_Utils.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <QMutex>
#include <numeric>

using namespace NN_CLI;

//===================================================================================================================//

CNNRunner::CNNRunner(const QCommandLineParser& parser, LogLevel logLevel, IOConfig& ioConfig,
                     AugmentationConfig& augConfig, std::unique_ptr<CNN::Core<float>>& core,
                     CNN::CoreConfig<float>& coreConfig)
  : Runner(parser, logLevel, ioConfig, augConfig, core, coreConfig)
{
}

//===================================================================================================================//
//  Accessors
//===================================================================================================================//

std::vector<std::string> CNNRunner::getTimingLines(int maxWidth) const
{
    return this->profiler.getTimingLines(maxWidth);
}

//===================================================================================================================//

ulong CNNRunner::getNumOutputClasses() const
{
  if (!this->coreConfig.layersConfig.denseLayers.empty())
    return this->coreConfig.layersConfig.denseLayers.back().numNeurons;
  return 0;
}

//===================================================================================================================//
//  Model info overrides
//===================================================================================================================//

ulong CNNRunner::getTotalParameters() const
{
  return TrainingSummary::countCNNParameters(this->coreConfig);
}

//===================================================================================================================//

std::string CNNRunner::getNetworkType() const
{
  return "CNN";
}

//===================================================================================================================//

std::string CNNRunner::getInputShapeString() const
{
  const auto& s = this->coreConfig.inputShape;
  return std::to_string(s.c) + " x " + std::to_string(s.h) + " x " + std::to_string(s.w);
}

//===================================================================================================================//

ulong CNNRunner::getNumConvLayers() const
{
  ulong count = 0;

  for (const auto& l : this->coreConfig.layersConfig.cnnLayers) {
    if (l.type == CNN::LayerType::CONV)
      count++;
  }

  return count;
}

//===================================================================================================================//

ulong CNNRunner::getNumDenseLayers() const
{
  return this->coreConfig.layersConfig.denseLayers.size();
}

//===================================================================================================================//

ulong CNNRunner::getNumResidualBlocks() const
{
  ulong count = 0;

  for (const auto& l : this->coreConfig.layersConfig.cnnLayers) {
    if (l.type == CNN::LayerType::RESIDUAL_START)
      count++;
  }

  return count;
}

//===================================================================================================================//
//  Mode methods
//===================================================================================================================//

int CNNRunner::train()
{
  if (checkSamplesIdxDataConflict(this->parser))
    return 1;

  QString inputFilePath;
  DataLoader<CNN::Sample<float>> dataLoader;
  const CNN::Shape3D& inputShape = this->coreConfig.inputShape;
  int inputC = static_cast<int>(inputShape.c);
  int inputH = static_cast<int>(inputShape.h);
  int inputW = static_cast<int>(inputShape.w);

  if (this->parser.isSet("samples")) {
    inputFilePath = this->parser.value("samples");
    dataLoader.loadManifest(inputFilePath.toStdString(), this->ioConfig, inputC, inputH, inputW,
                            static_cast<int>(this->ioConfig.outputC), static_cast<int>(this->ioConfig.outputH),
                            static_cast<int>(this->ioConfig.outputW));
  } else {
    auto [samples, success] = this->loadSamplesFromOptions("training", inputFilePath);

    if (!success)
      return 1;

    dataLoader.loadFromMemory(std::move(samples), inputC, inputH, inputW);
  }

  ulong totalOriginalSamples = dataLoader.numSamples();

  const auto& validationConfig = this->augConfig.validationConfig;
  DataSplit split;
  float validationRatio = 0.0f;
  bool validationAuto = false;

  if (validationConfig.enabled) {
    validationRatio =
      validationConfig.autoSize ? DataSplitter::computeAutoValSize(dataLoader.numSamples()) : validationConfig.size;
    validationAuto = validationConfig.autoSize;
    auto allOutputs = dataLoader.getAllOutputs();
    split = DataSplitter::stratifiedSplit(allOutputs, validationRatio);
    this->validationState.enabled = true;
    this->validationState.checkInterval = validationConfig.checkInterval;
    this->validationState.numValSamples = split.validationIndices.size();

    split.trainIndices =
      dataLoader.planAugmentation(this->augConfig.augmentationFactor, this->augConfig.balanceAugmentation,
                                  this->augConfig.fullAugmentation, split.trainIndices);
  } else {
    this->validationState.enabled = false;
    dataLoader.planAugmentation(this->augConfig.augmentationFactor, this->augConfig.balanceAugmentation,
                                this->augConfig.fullAugmentation);
  }

  if (this->augConfig.autoClassWeights && this->coreConfig.costFunctionConfig.weights.empty()) {
    auto allOutputs = dataLoader.getAllOutputs();
    std::vector<std::vector<float>> trainingOutputs;

    if (validationConfig.enabled) {
      trainingOutputs.reserve(split.trainIndices.size());

      for (ulong idx : split.trainIndices)
        trainingOutputs.push_back(allOutputs[idx]);
    } else {
      trainingOutputs = std::move(allOutputs);
    }

    std::vector<float> weights = computeClassWeightsFromOutputs(trainingOutputs);

    if (this->coreConfig.costFunctionConfig.type == Common::CostFunctionType::SQUARED_DIFFERENCE)
      this->coreConfig.costFunctionConfig.type = Common::CostFunctionType::WEIGHTED_SQUARED_DIFFERENCE;

    this->coreConfig.costFunctionConfig.weights = weights;
    this->core = CNN::Core<float>::makeCore(this->coreConfig);
  }

  ulong numValidationSamples = validationConfig.enabled ? split.validationIndices.size() : 0;
  ulong numOriginalTrainSamples = totalOriginalSamples - numValidationSamples;
  ulong numTrainSamples = validationConfig.enabled ? split.trainIndices.size() : dataLoader.numSamples();

  this->_numOriginalTrainSamples = numOriginalTrainSamples;
  this->_numTrainSamples = numTrainSamples;
  this->_numValidationSamples = numValidationSamples;

  this->notifyModelInfoUpdated("totalOriginalSamples", std::to_string(totalOriginalSamples));
  this->notifyModelInfoUpdated("numTrainSamples", std::to_string(numTrainSamples));
  this->notifyModelInfoUpdated("numValidationSamples", std::to_string(numValidationSamples));

  // When validation is enabled, NN-CLI handles monitoring with validation loss.
  std::shared_ptr<Common::TrainingMonitor<float>> trainingMonitor;

  if (validationConfig.enabled && this->coreConfig.trainingConfig.monitoringConfig.enabled) {
    trainingMonitor =
      std::make_shared<Common::TrainingMonitor<float>>(this->coreConfig.trainingConfig.monitoringConfig);
    this->coreConfig.trainingConfig.monitoringConfig.enabled = false;
    this->core = CNN::Core<float>::makeCore(this->coreConfig);
  }

  std::shared_ptr<CNN::Core<float>> validationCore;

  if (validationConfig.enabled) {
    CNN::CoreConfig<float> validationCoreConfig = this->coreConfig;
    validationCoreConfig.modeType = Common::ModeType::TEST;

    auto allOutputs = dataLoader.getAllOutputs();
    std::vector<std::vector<float>> validationOutputs;
    validationOutputs.reserve(split.validationIndices.size());

    for (ulong idx : split.validationIndices)
      validationOutputs.push_back(allOutputs[idx]);

    std::vector<float> validationWeights = computeClassWeightsFromOutputs(validationOutputs);
    validationCoreConfig.costFunctionConfig.weights = validationWeights;

    validationCore = CNN::Core<float>::makeCore(validationCoreConfig);
  }

  std::unique_ptr<GpuAugmenterPool> gpuAugPool;

  if (this->coreConfig.deviceType == Common::DeviceType::GPU && this->ioConfig.inputType == DataType::IMAGE) {
    OpenCLWrapper::Core::initialize(false);
    int totalGpus = static_cast<int>(OpenCLWrapper::Core::getNumDevices());
    int numAugGpus = (this->coreConfig.numGPUs > 0) ? std::min(totalGpus, this->coreConfig.numGPUs) : totalGpus;

    if (numAugGpus > 0) {
      std::vector<int> deviceIndices;

      for (int i = 0; i < numAugGpus; i++)
        deviceIndices.push_back(i);

      gpuAugPool =
        std::make_unique<GpuAugmenterPool>(deviceIndices, static_cast<ulong>(inputC), static_cast<ulong>(inputH),
                                           static_cast<ulong>(inputW), this->logLevel);
      dataLoader.setGpuAugmenterPool(gpuAugPool.get());
    }
  }

  this->setupTrainingCallback(inputFilePath, validationCore, trainingMonitor,
                              validationConfig.enabled ? &dataLoader : nullptr,
                              validationConfig.enabled ? &split.validationIndices : nullptr);

  if (gpuAugPool) {
    gpuAugPool->setTimingCallback([this](bool begin) {
      this->profiler.onEvent(CNN::TimingPhase::Augmentation, begin ? CNN::TimingEvent::Begin : CNN::TimingEvent::End,
                             -1);
    });
  }


  // Prepend loaded epoch history into the core before training starts, so
  // checkpoints during training serialize the full history, not just new epochs.
  if (!this->coreConfig.loadedEpochHistory.empty()) {
    this->core->prependEpochHistory(this->coreConfig.loadedEpochHistory);
    this->coreConfig.loadedEpochHistory.clear();
  }

  // Drive the "Samples" data-loading progress bar: the DataLoader fires this
  // callback (from its worker threads) as each batch is loaded/augmented.
  dataLoader.setLoadingCallback(
    [this](ulong current, ulong total, ulong batchIndex, ulong totalBatches, SampleLoadType loadType) {
      this->notifySampleLoadProgress(current, total, batchIndex, totalBatches,
                                     loadType == SampleLoadType::Validation);
    });

  if (validationConfig.enabled) {
    auto trainProvider =
      dataLoader.makeSampleProvider(split.trainIndices, this->augConfig.transforms,
                                    this->augConfig.augmentationProbability, SampleLoadType::Training);
    this->core->train(split.trainIndices.size(), trainProvider);
  } else {
    auto sampleProvider = dataLoader.makeSampleProvider(
      this->augConfig.transforms, this->augConfig.augmentationProbability, SampleLoadType::Training);
    this->core->train(dataLoader.numSamples(), sampleProvider);
  }

  return this->finishTraining(inputFilePath);
}

//===================================================================================================================//

int CNNRunner::test()
{
  if (checkSamplesIdxDataConflict(this->parser))
    return 1;

  QString inputFilePath;
  DataLoader<CNN::Sample<float>> dataLoader;
  const CNN::Shape3D& inputShape = this->coreConfig.inputShape;
  int inputC = static_cast<int>(inputShape.c);
  int inputH = static_cast<int>(inputShape.h);
  int inputW = static_cast<int>(inputShape.w);

  if (this->parser.isSet("samples")) {
    inputFilePath = this->parser.value("samples");
    dataLoader.loadManifest(inputFilePath.toStdString(), this->ioConfig, inputC, inputH, inputW,
                            static_cast<int>(this->ioConfig.outputC), static_cast<int>(this->ioConfig.outputH),
                            static_cast<int>(this->ioConfig.outputW));
  } else {
    auto [samples, success] = this->loadSamplesFromOptions("test", inputFilePath);

    if (!success)
      return 1;
    dataLoader.loadFromMemory(std::move(samples), inputC, inputH, inputW);
  }

  if (this->logLevel > LogLevel::QUIET)
    TestSummary::printCNN(this->coreConfig, dataLoader.numSamples());

  setupModeProgressCallback(*this->core, this->logLevel, this->ioConfig.progressReports, "Testing",
                            dataLoader.numSamples());

  auto sampleProvider = dataLoader.makeSampleProvider({}, 0.0f);
  Common::TestResult<float> result = this->core->test(dataLoader.numSamples(), sampleProvider);

  if (this->logLevel > LogLevel::QUIET) {
    std::cout << "\nTest Results:\n";
    std::cout << "  Samples evaluated: " << result.numSamples << "\n";
    std::cout << "  Total loss:        " << result.totalLoss << "\n";
    std::cout << "  Average loss:      " << result.averageLoss << "\n";
    std::cout << "  Correct:           " << result.numCorrect << " / " << result.numSamples << "\n";
    std::cout << "  Accuracy:          " << std::fixed << std::setprecision(2) << result.accuracy << "%\n";
    std::cout.unsetf(std::ios_base::floatfield);
  }

  return 0;
}

//===================================================================================================================//

int CNNRunner::predict()
{
  if (!this->parser.isSet("input")) {
    std::cerr << "Error: --input option is required for predict mode.\n";
    return 1;
  }

  QString inputPath = this->parser.value("input");
  QString outputPath = resolvePredictOutputPath(this->parser, this->ioConfig);

  ulong displayProgressReports = (this->logLevel > LogLevel::QUIET) ? this->ioConfig.progressReports : 0;
  std::vector<CNN::Input<float>> inputs =
    CNNLoader::loadInputs(inputPath.toStdString(), this->coreConfig.inputShape, this->ioConfig, displayProgressReports);

  if (this->logLevel > LogLevel::QUIET)
    PredictSummary::printCNN(this->coreConfig, inputs.size(), inputPath.toStdString(), outputPath.toStdString());

  auto batchStart = std::chrono::system_clock::now();
  std::string startTimeStr = Common::Utils::formatISO8601();

  setupModeProgressCallback(*this->core, this->logLevel, this->ioConfig.progressReports, "Predicting", inputs.size());

  // The streaming predict API takes a provider that yields one batch at a
  // time. The batch JSON is already loaded into `inputs`, so we just slice it.
  auto sliceProvider = [&inputs](ulong batchSize, ulong batchIndex) {
    ulong start = batchIndex * batchSize;
    ulong end = std::min(start + batchSize, static_cast<ulong>(inputs.size()));

    if (start >= end)
      return CNN::Inputs<float>{};
    return CNN::Inputs<float>(inputs.begin() + start, inputs.begin() + end);
  };

  Common::PredictResults<float> results = this->core->predict(inputs.size(), sliceProvider);

  auto batchEnd = std::chrono::system_clock::now();
  std::string endTimeStr = Common::Utils::formatISO8601();
  std::chrono::duration<double> batchElapsed = batchEnd - batchStart;
  double batchDurationSeconds = batchElapsed.count();
  std::string batchDurationFormatted = Common::Utils::formatDuration(batchDurationSeconds);

  return writePredictOutput(results, outputPath, this->ioConfig, this->logLevel, startTimeStr, endTimeStr,
                            batchDurationSeconds, batchDurationFormatted, inputs.size());
}

//===================================================================================================================//
//  Sample loading
//===================================================================================================================//

std::pair<CNN::Samples<float>, bool> CNNRunner::loadSamplesFromOptions(const std::string& modeName,
                                                                       QString& inputFilePath)
{
  const CNN::Shape3D& inputShape = this->coreConfig.inputShape;

  return loadSamplesFromOptionsCommon<CNN::Samples<float>>(
    this->parser, this->logLevel, this->ioConfig, modeName, inputFilePath,
    [this, &inputShape](const std::string& path, ulong progressReports) {
      return CNNLoader::loadSamples(path, inputShape, this->ioConfig, progressReports);
    },

    [&inputShape](const std::string& dataPath, const std::string& labelsPath, ulong progressReports) {
      return Utils<float>::loadCNNIDX(dataPath, labelsPath, inputShape, progressReports);
    });
}

//===================================================================================================================//
//  Training helpers
//===================================================================================================================//

void CNNRunner::setupTrainingCallback(const QString& inputFilePath, std::shared_ptr<CNN::Core<float>> validationCore,
                                      std::shared_ptr<Common::TrainingMonitor<float>> trainingMonitor,
                                      const DataLoader<CNN::Sample<float>>* validationDataLoader,
                                      const std::vector<ulong>* validationIndices)
{
  this->lastEpochLoss = 0.0f;

  ulong batchSize = this->coreConfig.trainingConfig.batchSize;
  int totalEpochs = static_cast<int>(this->coreConfig.trainingConfig.numEpochs);

  // Wire the per-phase timing profiler to the CNN library's timing callback.
  this->profiler.reset();
  this->core->setTimingCallback([this](CNN::TimingPhase phase, CNN::TimingEvent event, int gpuIndex) {
    this->profiler.onEvent(phase, event, gpuIndex);
  });

  if (this->parser.isSet("gpu-profile")) {
    this->core->setGpuProfileCallback([this](const std::vector<CNN::GpuPhaseProfile>& profiles, int gpuIndex) {
      this->profiler.onGpuProfile(profiles, gpuIndex);
    });
  }

  std::shared_ptr<CNN::SampleProvider<float>> validationProviderPtr;

  if (validationDataLoader && validationIndices && !validationIndices->empty()) {
    auto provider = validationDataLoader->makeSampleProvider(*validationIndices, {}, 0.0f, SampleLoadType::Validation);
    validationProviderPtr = std::make_shared<CNN::SampleProvider<float>>(std::move(provider));
  }

  // Live progress callback: fires per batch from the GPU worker threads. It
  // only drives the progress/timing display — every epoch-boundary task lives
  // in the epoch-completed callback below.
  this->core->setTrainingCallback([this, batchSize](const Common::TrainingProgressEvent<float>& progress) {
    this->handleTrainingProgress(progress, batchSize);
  });

  // Epoch-completed callback: fires once per epoch (after the epoch's record is
  // recorded) with the 0-based epoch index. The core hands us the index
  // directly, so there is no transition tracking and no off-by-one — completion.epoch
  // matches EpochRecord::epoch and the serialized bestValidationEpoch.
  this->core->setEpochCompletedCallback([this, inputFilePath, validationCore, trainingMonitor, validationProviderPtr,
                                         validationIndices,
                                         totalEpochs](const Common::EpochCompletionEvent<float>& completion) {
    QMutexLocker<QMutex> lock(&this->callbackMutex);

    const ulong epoch = completion.epoch; // 0-based index of the just-completed epoch

    // --- Checkpointing (every saveModelInterval completed epochs) ---
    // epoch + 1 is the count of completed epochs; checkpoint filenames stay
    // 1-based for human-facing numbering.
    if (this->ioConfig.saveModelInterval > 0 && (epoch + 1) % this->ioConfig.saveModelInterval == 0) {
      std::string checkpointPath =
        ModelSerializer::generateCheckpointPath(inputFilePath, epoch + 1, this->lastEpochLoss);
      ModelSerializer::saveCNNModelToPackage(checkpointPath, *this->core, this->coreConfig, this->ioConfig,
                                             this->augConfig, this->buildValidationMetadata());
    }

    // --- Validation ---
    bool isBest = false;
    bool monitorShouldStop = false;
    float valLoss = 0.0f;
    bool hasValLoss = false;

    if (this->validationState.enabled && validationCore && validationProviderPtr && validationIndices &&
        epoch % this->validationState.checkInterval == 0) {
      ulong validationTotal = validationIndices->size();

      validationCore->setParameters(this->core->getParameters());
      validationCore->syncParametersToGPU();

      // Live "Validating" bar: route validation progress through the observer
      // instead of printing to stdout (which would corrupt the ncurses TUI).
      validationCore->setProgressCallback(
        [this, validationTotal](ulong current, ulong) { this->notifyValidationProgress(current, validationTotal); });

      auto validationResult = validationCore->test(validationTotal, *validationProviderPtr);

      this->validationState.lastValLoss = validationResult.averageLoss;
      valLoss = validationResult.averageLoss;
      hasValLoss = true;

      if (validationResult.averageLoss < this->validationState.bestValLoss) {
        this->validationState.bestValLoss = validationResult.averageLoss;
        this->validationState.bestValEpoch = epoch;
      }

      if (trainingMonitor) {
        monitorShouldStop =
          trainingMonitor->checkEpoch(epoch, this->lastEpochLoss, std::optional<float>(validationResult.averageLoss));
        isBest = trainingMonitor->isNewBest();
      }
    }

    bool isBestEpoch = (isBest || completion.isNewBest);

    // --- Best model save ---
    if (isBestEpoch) {
      std::string bestPath = ModelSerializer::generateBestModelPath(inputFilePath);
      ModelSerializer::saveCNNModelToPackage(bestPath, *this->core, this->coreConfig, this->ioConfig, this->augConfig,
                                             this->buildValidationMetadata());
    }

    // --- Write the validation results into this epoch's history record ---
    // The core's internal monitor is disabled (NN-CLI monitors externally), so
    // it recorded isBest=false / hasValLoss=false. The just-completed epoch is
    // epochHistory.back() (the core appended it immediately before this call).
    auto& epochHistory = this->core->getTrainingMetadata().epochHistory;

    if (!epochHistory.empty()) {
      auto& lastRecord = epochHistory.back();
      lastRecord.isBest = isBestEpoch;
      lastRecord.hasValLoss = hasValLoss;
      lastRecord.valLoss = valLoss;
    }

    this->profiler.setEpoch(epoch + 1);

    // --- Observer notification — epoch completed ---
    std::string epochSummary = "Epoch " + std::to_string(epoch + 1) + "/" + std::to_string(totalEpochs) +
                               " | Loss: " + std::to_string(this->lastEpochLoss);

    if (hasValLoss)
      epochSummary += " | ValLoss: " + std::to_string(valLoss);

    if (isBestEpoch)
      epochSummary += " | Best*";

    this->notifyEpochCompleted(static_cast<int>(epoch), totalEpochs, this->lastEpochLoss, hasValLoss, valLoss,
                               epochSummary);

    // --- Monitor stop requests ---
    if (monitorShouldStop) {
      std::string stopMsg = "[Monitor] Training stopped: " + trainingMonitor->getStopReason();

      this->notifyLogMessage(stopMsg, false);
      this->core->requestStop();
    }

    if (completion.stoppedEarly) {
      std::string stopMsg = "[Monitor] Training stopped: " + this->core->getTrainingMetadata().stopReason;

      this->notifyLogMessage(stopMsg, false);
      this->core->requestStop();
    }
  });
}

//===================================================================================================================//
//  Model saving (override from Runner base)
//===================================================================================================================//

void CNNRunner::doSaveModel(const std::string& outputPath)
{
  ModelSerializer::saveCNNModelToPackage(outputPath, *this->core, this->coreConfig, this->ioConfig, this->augConfig,
                                         this->buildValidationMetadata());
}

//===================================================================================================================//
//  Class weight computation — delegates to shared computeClassWeightsFromOutputs() in NN-CLI_Utils.hpp
//===================================================================================================================//
