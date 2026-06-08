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

  this->trainingTui_.attach(this->tui, [this]() {
    this->profiler.resetRenderState();
    ulong cw = this->tui->leftWidth() > 4 ? this->tui->leftWidth() - 4 : 80;
    this->regenerateConfigLines(cw);
  });

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

  // Collect config table lines for the TUI config panel.
  ulong numValidationSamples = validationConfig.enabled ? split.validationIndices.size() : 0;
  ulong numOriginalTrainSamples = totalOriginalSamples - numValidationSamples;
  ulong numTrainSamples = validationConfig.enabled ? split.trainIndices.size() : dataLoader.numSamples();

  if (this->logLevel > LogLevel::QUIET) {
    this->cachedNumOrigTrainSamples_ = numOriginalTrainSamples;
    this->cachedNumTrainSamples_ = numTrainSamples;
    this->cachedNumValSamples_ = numValidationSamples;
    this->cachedValRatio_ = validationRatio;
    this->cachedValAuto_ = validationAuto;

    ulong numOutputClasses = this->coreConfig.layersConfig.denseLayers.empty()
                               ? 0
                               : this->coreConfig.layersConfig.denseLayers.back().numNeurons;
    this->cachedNumOutputClasses_ = numOutputClasses;
    this->configLinesLoaded_ = true;

    ulong configWidth = this->tui->leftWidth() > 4 ? this->tui->leftWidth() - 4 : 80;

    std::vector<SummaryTable::Section> sections;

    auto trainRows =
      TrainingSummary::collectCNNRows(this->coreConfig, this->augConfig, numOriginalTrainSamples, numTrainSamples,
                                      numValidationSamples, validationRatio, validationAuto);
    sections.push_back({"Model Configuration", std::move(trainRows)});

    if (numOutputClasses >= 2) {
      auto lossRows = LossReferenceTable::collectRows(numOutputClasses);
      sections.push_back({"Loss Reference", std::move(lossRows)});
    }

    auto configLines = SummaryTable::collectSections(sections, configWidth);
    this->tui->setConfigLines(configLines);
  }

  // Render config panel
  if (this->tui->isInitialized()) {
    this->tui->refreshConfigPanel();
  }

  // When validation is enabled, NN-CLI handles monitoring with validation loss.
  std::shared_ptr<Common::TrainingMonitor<float>> trainingMonitor;

  if (validationConfig.enabled && this->coreConfig.trainingConfig.monitoringConfig.enabled) {
    trainingMonitor = std::make_shared<Common::TrainingMonitor<float>>(this->coreConfig.trainingConfig.monitoringConfig);
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
    this->trainingTui_.resolveBarGpus(this->coreConfig.deviceType == Common::DeviceType::GPU, this->coreConfig.numGPUs);
    dataLoader.setLoadingCallback(this->trainingTui_.loadingCallback());
  }

  if (validationConfig.enabled) {
    auto trainProvider =
      dataLoader.makeSampleProvider(split.trainIndices, this->augConfig.transforms,
                                    this->augConfig.augmentationProbability, SampleLoadType::Training);
    this->trainingTui_.markLoadingFinished();
    this->core->train(split.trainIndices.size(), trainProvider);
  } else {
    auto sampleProvider = dataLoader.makeSampleProvider(
      this->augConfig.transforms, this->augConfig.augmentationProbability, SampleLoadType::Training);
    this->trainingTui_.markLoadingFinished();
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

void CNNRunner::regenerateConfigLines(ulong maxWidth)
{
  if (!this->configLinesLoaded_)
    return;

  std::vector<SummaryTable::Section> sections;

  auto trainRows = TrainingSummary::collectCNNRows(this->coreConfig, this->augConfig, this->cachedNumOrigTrainSamples_,
                                                   this->cachedNumTrainSamples_, this->cachedNumValSamples_,
                                                   this->cachedValRatio_, this->cachedValAuto_);
  sections.push_back({"Model Configuration", std::move(trainRows)});

  if (this->cachedNumOutputClasses_ >= 2) {
    auto lossRows = LossReferenceTable::collectRows(this->cachedNumOutputClasses_);
    sections.push_back({"Loss Reference", std::move(lossRows)});
  }

  auto lines = SummaryTable::collectSections(sections, maxWidth);
  this->tui->setConfigLines(lines);
}

//===================================================================================================================//

void CNNRunner::setupTrainingCallback(const QString& inputFilePath, std::shared_ptr<CNN::Core<float>> validationCore,
                                      std::shared_ptr<Common::TrainingMonitor<float>> trainingMonitor,
                                      const DataLoader<CNN::Sample<float>>* validationDataLoader,
                                      const std::vector<ulong>* validationIndices)
{
  this->lastCallbackEpoch_ = 0;
  this->lastEpochLoss_ = 0.0f;

  ulong batchSize = this->coreConfig.trainingConfig.batchSize;
  this->progressBar_ = std::make_unique<ProgressBar>(this->ioConfig.progressReports, 50, std::max(2UL, batchSize / 2));

  this->progressBar_->setHoldEpochLine(false);

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

  this->core->setTrainingCallback([this, inputFilePath, validationCore, trainingMonitor, validationProviderPtr,
                                   validationIndices, validationDataLoader,
                                   tui](const Common::TrainingProgress<float>& progress) {
    {
      std::lock_guard<std::mutex> lock(this->epochTransitionMutex_);

      // Epoch transition: process epoch-end tasks when a new epoch starts.
      // saveModelInterval controls checkpoint frequency only; epoch
      // transitions must always be processed for TUI, validation, and
      // monitoring logic to work.
      bool epochTransition = progress.currentEpoch > this->lastCallbackEpoch_;

      if (epochTransition) {
        const ulong finishedEpoch = lastCallbackEpoch_;

        // --- Checkpointing (controlled by saveModelInterval) ---
        std::string checkpointPath;

        if (this->ioConfig.saveModelInterval > 0 && lastCallbackEpoch_ > 0 &&
            lastCallbackEpoch_ % this->ioConfig.saveModelInterval == 0) {
          checkpointPath = ModelSerializer::generateCheckpointPath(inputFilePath, lastCallbackEpoch_, lastEpochLoss_);
          ModelSerializer::saveCNNModel(checkpointPath, *this->core, this->coreConfig, this->ioConfig, this->augConfig,
                                        this->buildValidationMetadata());
        }

        // --- Validation ---
        bool isBest = false;
        bool monitorShouldStop = false;
        float valLoss = 0.0f;
        bool hasValLoss = false;

        if (this->lastCallbackEpoch_ > 0 && this->validationState.enabled && validationCore && validationProviderPtr &&
            validationIndices && this->lastCallbackEpoch_ % this->validationState.checkInterval == 0) {
          ulong validationTotal = validationIndices->size();

          validationCore->setParameters(this->core->getParameters());
          validationCore->syncParametersToGPU();

          // Show validation progress on the progress window
          setupValidationProgressCallback(*validationCore, tui, validationTotal, progress.totalGPUs);

          auto validationResult = validationCore->test(validationTotal, *validationProviderPtr);

          this->validationState.lastValLoss = validationResult.averageLoss;
          valLoss = validationResult.averageLoss;
          hasValLoss = true;

          if (validationResult.averageLoss < this->validationState.bestValLoss) {
            this->validationState.bestValLoss = validationResult.averageLoss;
            this->validationState.bestValEpoch = this->lastCallbackEpoch_;
          }

          if (trainingMonitor) {
            monitorShouldStop = trainingMonitor->checkEpoch(this->lastCallbackEpoch_, this->lastEpochLoss_,
                                                            std::optional<float>(validationResult.averageLoss));
            isBest = trainingMonitor->isNewBest();
          }
        }

        // --- Best model save ---
        if (isBest || progress.isNewBest) {
          std::string bestPath = ModelSerializer::generateBestModelPath(inputFilePath);
          ModelSerializer::saveCNNModel(bestPath, *this->core, this->coreConfig, this->ioConfig, this->augConfig,
                                        this->buildValidationMetadata());
        }

        // --- TUI history (skip epoch 0 line) ---
        if (tui && tui->isInitialized() && this->logLevel > LogLevel::QUIET && finishedEpoch > 0) {
          bool isBestEpoch = (isBest || progress.isNewBest);
          tui->pushEpochRecord(static_cast<int>(finishedEpoch), this->lastEpochLoss_, hasValLoss, valLoss, isBestEpoch);
        }

        // --- Timing window reset ---
        if (tui && tui->isInitialized()) {
          tui->setTimingLines({" Timing - waiting for first batch"});
        }

        // --- Monitor stop requests ---
        if (monitorShouldStop) {
          if (tui && tui->isInitialized())
            tui->addEpochLine("[Monitor] Training stopped: " + trainingMonitor->stopReason());

          this->core->requestStop();
        }

        if (progress.stoppedEarly) {
          if (tui && tui->isInitialized())
            tui->addEpochLine("[Monitor] Training stopped: " + this->core->getTrainingMetadata().stopReason);

          this->core->requestStop();
        }

        this->profiler.setEpoch(progress.currentEpoch);
        this->lastCallbackEpoch_ = progress.currentEpoch;
      }

      if (this->logLevel > LogLevel::QUIET) {
        ProgressInfo info{progress.currentEpoch, progress.totalEpochs, progress.currentSample, progress.totalSamples,
                          progress.epochLoss,    progress.sampleLoss,  progress.gpuIndex,      progress.totalGPUs};

        if (tui && tui->isInitialized()) {
          std::lock_guard<std::recursive_mutex> tuiLock(tui->mutex());

          tui->handleResize();

          this->progressBar_->update(info, tui->progressWindow());

          auto timingLines = this->profiler.getTimingLines(tui->timingWidth());

          if (!timingLines.empty())
            tui->setTimingLines(timingLines);

          tui->refresh();
        }
      }

      if (progress.epochLoss > 0)
        this->lastEpochLoss_ = progress.epochLoss;
    } // lock_guard released
  });
}

//===================================================================================================================//

int CNNRunner::finishTraining(const QString& inputFilePath)
{
  return finishTrainingCommon(this->tui, this->logLevel, this->parser, inputFilePath, *this->core,
                              [this](const std::string& path) {
                                ModelSerializer::saveCNNModel(path, *this->core, this->coreConfig, this->ioConfig,
                                                              this->augConfig, this->buildValidationMetadata());
                              });
}

//===================================================================================================================//
//  Class weight computation — delegates to shared computeClassWeightsFromOutputs() in NN-CLI_Utils.hpp
//===================================================================================================================//
