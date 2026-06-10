#include "NN-CLI_CNNRunner.hpp"

#include "NN-CLI_CNNLoader.hpp"
#include "NN-CLI_DataLoader.hpp"
#include "NN-CLI_GpuAugmenter.hpp"
#include "NN-CLI_LossReferenceTable.hpp"
#include "Common/Common_TrainingMonitor.hpp"
#include <OCLW_Core.hpp>
#include "NN-CLI_DataSplitter.hpp"
#include "NN-CLI_ImageLoader.hpp"
#include "NN-CLI_ProgressBar.hpp"
#include "NN-CLI_PredictSummary.hpp"
#include "NN-CLI_RunnerUtils.hpp"
#include "NN-CLI_TestSummary.hpp"
#include "NN-CLI_TrainingSummary.hpp"
#include "NN-CLI_Utils.hpp"

#include <QDir>
#include <QFile>
#include <QFileInfo>

#include <json.hpp>

#include <ANN_Utils.hpp>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>

using namespace NN_CLI;

//===================================================================================================================//

CNNRunner::CNNRunner(const QCommandLineParser& parser, LogLevel logLevel, IOConfig& ioConfig,
                     AugmentationConfig& augConfig, std::unique_ptr<CNN::Core<float>>& core,
                     CNN::CoreConfig<float>& coreConfig)
  : parser(parser),
    logLevel(logLevel),
    ioConfig(ioConfig),
    augConfig(augConfig),
    core(core),
    coreConfig(coreConfig)
{
}

//===================================================================================================================//
//  Mode methods
//===================================================================================================================//

int CNNRunner::train()
{
  if (checkSamplesIdxDataConflict(this->parser))
    return 1;

  // Create and init the ncurses TUI immediately so the user sees it right away.
  this->tui = std::make_shared<TerminalUI>();

  if (this->logLevel > LogLevel::QUIET)
    this->tui->init();

  this->trainingTui.attach(this->tui, [this]() { this->profiler.resetRenderState(); });

  // Show loading status in the TUI while samples are processed.
  if (this->tui->isInitialized())
    this->tui->refreshConfigPanel();

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

    if (!success) {
      this->tui->shutdown();
      return 1;
    }

    dataLoader.loadFromMemory(std::move(samples), inputC, inputH, inputW);
  }

  this->tui->pollInput();

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

  // Collect config sections for the TUI config panel.
  ulong numValidationSamples = validationConfig.enabled ? split.validationIndices.size() : 0;
  ulong numOriginalTrainSamples = totalOriginalSamples - numValidationSamples;
  ulong numTrainSamples = validationConfig.enabled ? split.trainIndices.size() : dataLoader.numSamples();

  if (this->logLevel > LogLevel::QUIET) {
    std::vector<SummaryTable::Section> sections;
    sections.push_back(
      {"Model Configuration",
       TrainingSummary::collectCNNRows(this->coreConfig, this->augConfig, numOriginalTrainSamples, numTrainSamples,
                                       numValidationSamples, validationRatio, validationAuto)});

    ulong numOutputClasses = this->coreConfig.layersConfig.denseLayers.empty()
                               ? 0
                               : this->coreConfig.layersConfig.denseLayers.back().numNeurons;
    if (numOutputClasses >= 2) {
      sections.push_back({"Loss Reference", LossReferenceTable::collectRows(numOutputClasses)});
    }

    this->tui->setConfigSections(sections);
  }

  // Render config panel
  if (this->tui->isInitialized()) {
    this->tui->refreshConfigPanel();
  }

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

  if (this->tui && this->tui->isInitialized()) {
    this->trainingTui.resolveBarGpus(this->coreConfig.deviceType == Common::DeviceType::GPU, this->coreConfig.numGPUs);
    dataLoader.setLoadingCallback(this->trainingTui.loadingCallback());
  }

  // Pre-populate the TUI epoch table with loaded history (resumed model).
  if (this->tui && this->tui->isInitialized() && !this->coreConfig.loadedEpochHistory.empty()) {
    for (const auto& record : this->coreConfig.loadedEpochHistory) {
      int epochNum = static_cast<int>(record.epoch) + 1; // Convert 0-based to 1-based for TUI display
      float lossVal = static_cast<float>(record.loss);
      float valLossVal = static_cast<float>(record.valLoss);
      std::time_t compTime = static_cast<std::time_t>(record.completionTime);

      this->tui->pushEpochRecord(epochNum, lossVal, record.hasValLoss, valLossVal, record.isBest, compTime);
    }
  }

  // Prepend loaded epoch history into the core before training starts, so
  // checkpoints during training serialize the full history, not just new epochs.
  if (!this->coreConfig.loadedEpochHistory.empty()) {
    this->core->prependEpochHistory(this->coreConfig.loadedEpochHistory);
    this->coreConfig.loadedEpochHistory.clear();
  }

  if (validationConfig.enabled) {
    auto trainProvider =
      dataLoader.makeSampleProvider(split.trainIndices, this->augConfig.transforms,
                                    this->augConfig.augmentationProbability, SampleLoadType::Training);
    this->trainingTui.markLoadingFinished();
    this->core->train(split.trainIndices.size(), trainProvider);
  } else {
    auto sampleProvider = dataLoader.makeSampleProvider(
      this->augConfig.transforms, this->augConfig.augmentationProbability, SampleLoadType::Training);
    this->trainingTui.markLoadingFinished();
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
  std::string startTimeStr = ANN::Utils<float>::formatISO8601();

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
  std::string endTimeStr = ANN::Utils<float>::formatISO8601();
  std::chrono::duration<double> batchElapsed = batchEnd - batchStart;
  double batchDurationSeconds = batchElapsed.count();
  std::string batchDurationFormatted = ANN::Utils<float>::formatDuration(batchDurationSeconds);

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

ValidationMetadata CNNRunner::buildValidationMetadata() const
{
  return {this->validationState.enabled, this->validationState.numValSamples, this->validationState.lastValLoss,
          this->validationState.bestValLoss, this->validationState.bestValEpoch};
}

//===================================================================================================================//

void CNNRunner::setupTrainingCallback(const QString& inputFilePath, std::shared_ptr<CNN::Core<float>> validationCore,
                                      std::shared_ptr<Common::TrainingMonitor<float>> trainingMonitor,
                                      const DataLoader<CNN::Sample<float>>* validationDataLoader,
                                      const std::vector<ulong>* validationIndices)
{
  this->lastEpochLoss = 0.0f;

  ulong batchSize = this->coreConfig.trainingConfig.batchSize;
  this->progressBar = std::make_unique<ProgressBar>(this->ioConfig.progressReports, 50, std::max(2UL, batchSize / 2));

  this->progressBar->setHoldEpochLine(false);

  // TUI is already created in train(); capture it for the callback lambda.
  auto tui = this->tui;

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
  this->core->setTrainingCallback([this, tui](const Common::TrainingProgressEvent<float>& progress) {
    std::lock_guard<std::mutex> lock(this->callbackMutex);

    if (progress.epochLoss > 0)
      this->lastEpochLoss = progress.epochLoss;

    if (this->logLevel > LogLevel::QUIET) {
      ProgressInfo info{progress.currentEpoch, progress.totalEpochs, progress.currentSample, progress.totalSamples,
                        progress.epochLoss,    progress.sampleLoss,  progress.gpuIndex,      progress.totalGPUs};

      if (tui && tui->isInitialized()) {
        std::lock_guard<std::recursive_mutex> tuiLock(tui->getMutex());

        tui->handleResize();

        this->progressBar->update(info, tui->progressWindow());

        auto timingLines = this->profiler.getTimingLines(tui->timingContentWidth());

        if (!timingLines.empty())
          tui->setTimingLines(timingLines);

        tui->refresh();
      }
    }
  });

  // Epoch-completed callback: fires once per epoch (after the epoch's record is
  // recorded) with the 0-based epoch index. The core hands us the index
  // directly, so there is no transition tracking and no off-by-one — completion.epoch
  // matches EpochRecord::epoch and the serialized bestValidationEpoch.
  this->core->setEpochCompletedCallback([this, inputFilePath, validationCore, trainingMonitor, validationProviderPtr,
                                         validationIndices,
                                         tui](const Common::EpochCompletionEvent<float>& completion) {
    std::lock_guard<std::mutex> lock(this->callbackMutex);

    const ulong epoch = completion.epoch; // 0-based index of the just-completed epoch

    // --- Checkpointing (every saveModelInterval completed epochs) ---
    // epoch + 1 is the count of completed epochs; checkpoint filenames stay
    // 1-based to match the TUI's human-facing epoch numbering.
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

      // Show validation progress on the progress window
      setupValidationProgressCallback(*validationCore, tui, validationTotal, this->coreConfig.numGPUs);

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

    // --- TUI history line (1-based for display) + timing reset ---
    if (tui && tui->isInitialized() && this->logLevel > LogLevel::QUIET) {
      tui->pushEpochRecord(static_cast<int>(epoch) + 1, this->lastEpochLoss, hasValLoss, valLoss, isBestEpoch);
      tui->setTimingLines({" Timing - waiting for first batch"});
    }

    this->profiler.setEpoch(epoch + 1);

    // --- Monitor stop requests ---
    if (monitorShouldStop) {
      if (tui && tui->isInitialized())
        tui->addEpochLine("[Monitor] Training stopped: " + trainingMonitor->getStopReason());

      this->core->requestStop();
    }

    if (completion.stoppedEarly) {
      if (tui && tui->isInitialized())
        tui->addEpochLine("[Monitor] Training stopped: " + this->core->getTrainingMetadata().stopReason);

      this->core->requestStop();
    }
  });
}

//===================================================================================================================//

int CNNRunner::finishTraining(const QString& inputFilePath)
{
  // Defensive: unreachable in normal flow (train() clears loadedEpochHistory after
  // prepending), kept as safety net against future refactoring.
  if (!this->coreConfig.loadedEpochHistory.empty()) {
    this->core->prependEpochHistory(this->coreConfig.loadedEpochHistory);
    this->coreConfig.loadedEpochHistory.clear();
  }

  // Every epoch — including the last — is finalized by the epoch-completed
  // callback (validation, best-model save, history record), so there is no
  // end-of-run fix-up to do here; just persist the final model.
  return finishTrainingCommon(
    this->tui, this->logLevel, this->parser, inputFilePath, *this->core, [this](const std::string& path) {
      ModelSerializer::saveCNNModelToPackage(path, *this->core, this->coreConfig, this->ioConfig, this->augConfig,
                                             this->buildValidationMetadata());
    });
}

//===================================================================================================================//
//  Class weight computation — delegates to shared computeClassWeightsFromOutputs() in NN-CLI_Utils.hpp
//===================================================================================================================//
