#include "NN-CLI_ANNRunner.hpp"
#include "NN-CLI_ANNLoader.hpp"

#include "NN-CLI_Loader.hpp"
#include "NN-CLI_DataLoader.hpp"
#include "NN-CLI_LossReferenceTable.hpp"
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

ANNRunner::ANNRunner(const QCommandLineParser& parser, LogLevel logLevel, IOConfig& ioConfig,
                     AugmentationConfig& augConfig, std::unique_ptr<ANN::Core<float>>& core,
                     ANN::CoreConfig<float>& coreConfig)
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

int ANNRunner::train()
{
  if (checkSamplesIdxDataConflict(this->parser))
    return 1;

  this->tui = std::make_shared<TerminalUI>();

  if (this->logLevel > LogLevel::QUIET)
    this->tui->init();

  this->trainingTui.attach(this->tui);

  if (this->tui->isInitialized())
    this->tui->refreshConfigPanel();

  QString inputFilePath;
  DataLoader<ANN::Sample<float>> dataLoader;

  int inputC = this->ioConfig.hasInputShape() ? static_cast<int>(this->ioConfig.inputC) : 0;
  int inputH = this->ioConfig.hasInputShape() ? static_cast<int>(this->ioConfig.inputH) : 0;
  int inputW = this->ioConfig.hasInputShape() ? static_cast<int>(this->ioConfig.inputW) : 0;

  if (this->parser.isSet("samples")) {
    inputFilePath = this->parser.value("samples");
    int outputC = this->ioConfig.hasOutputShape() ? static_cast<int>(this->ioConfig.outputC) : 0;
    int outputH = this->ioConfig.hasOutputShape() ? static_cast<int>(this->ioConfig.outputH) : 0;
    int outputW = this->ioConfig.hasOutputShape() ? static_cast<int>(this->ioConfig.outputW) : 0;
    dataLoader.loadManifest(inputFilePath.toStdString(), this->ioConfig, inputC, inputH, inputW, outputC, outputH,
                            outputW);
  } else {
    auto [samples, success] = this->loadSamplesFromOptions("training", inputFilePath);

    if (!success) {
      this->tui->shutdown();
      return 1;
    }

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
    this->coreConfig.costFunctionConfig.type = Common::CostFunctionType::WEIGHTED_SQUARED_DIFFERENCE;
    this->coreConfig.costFunctionConfig.weights = weights;
    this->core = ANN::Core<float>::makeCore(this->coreConfig);
  }

  ulong numValidationSamples = validationConfig.enabled ? split.validationIndices.size() : 0;
  ulong numOriginalTrainSamples = totalOriginalSamples - numValidationSamples;
  ulong numTrainSamples = validationConfig.enabled ? split.trainIndices.size() : dataLoader.numSamples();

  if (this->logLevel > LogLevel::QUIET) {
    std::vector<SummaryTable::Section> sections;
    sections.push_back(
      {"Model Configuration",
       TrainingSummary::collectRows(this->coreConfig, this->augConfig, numOriginalTrainSamples, numTrainSamples,
                                    numValidationSamples, validationRatio, validationAuto)});

    ulong numOutputClasses =
      this->coreConfig.layersConfig.empty() ? 0 : this->coreConfig.layersConfig.back().numNeurons;
    if (numOutputClasses >= 2) {
      sections.push_back({"Loss Reference", LossReferenceTable::collectRows(numOutputClasses)});
    }

    this->tui->setConfigSections(sections);
  }

  // Render config panel
  if (this->tui->isInitialized()) {
    this->tui->refreshConfigPanel();
  }

  std::shared_ptr<Common::TrainingMonitor<float>> trainingMonitor;

  if (validationConfig.enabled && this->coreConfig.trainingConfig.monitoringConfig.enabled) {
    trainingMonitor =
      std::make_shared<Common::TrainingMonitor<float>>(this->coreConfig.trainingConfig.monitoringConfig);
    this->coreConfig.trainingConfig.monitoringConfig.enabled = false;
    this->core = ANN::Core<float>::makeCore(this->coreConfig);
  }

  std::shared_ptr<ANN::Core<float>> validationCore;

  if (validationConfig.enabled) {
    ANN::CoreConfig<float> validationCoreConfig = this->coreConfig;
    validationCoreConfig.modeType = Common::ModeType::TEST;

    auto allOutputs = dataLoader.getAllOutputs();
    std::vector<std::vector<float>> validationOutputs;
    validationOutputs.reserve(split.validationIndices.size());

    for (ulong idx : split.validationIndices)
      validationOutputs.push_back(allOutputs[idx]);

    std::vector<float> validationWeights = computeClassWeightsFromOutputs(validationOutputs);
    validationCoreConfig.costFunctionConfig.weights = validationWeights;

    validationCore = ANN::Core<float>::makeCore(validationCoreConfig);
  }

  // Store validation objects as members for use in finishTraining().
  this->validationCore = validationCore;
  this->trainingMonitor = trainingMonitor;

  if (validationConfig.enabled) {
    this->validationIndices = std::make_shared<std::vector<ulong>>(split.validationIndices);
  }

  this->setupTrainingCallback(inputFilePath, validationCore, trainingMonitor,
                              validationConfig.enabled ? &dataLoader : nullptr,
                              validationConfig.enabled ? &split.validationIndices : nullptr);

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

int ANNRunner::test()
{
  if (checkSamplesIdxDataConflict(this->parser))
    return 1;

  QString inputFilePath;
  DataLoader<ANN::Sample<float>> dataLoader;

  int inputC = this->ioConfig.hasInputShape() ? static_cast<int>(this->ioConfig.inputC) : 0;
  int inputH = this->ioConfig.hasInputShape() ? static_cast<int>(this->ioConfig.inputH) : 0;
  int inputW = this->ioConfig.hasInputShape() ? static_cast<int>(this->ioConfig.inputW) : 0;

  if (this->parser.isSet("samples")) {
    inputFilePath = this->parser.value("samples");
    int outputC = this->ioConfig.hasOutputShape() ? static_cast<int>(this->ioConfig.outputC) : 0;
    int outputH = this->ioConfig.hasOutputShape() ? static_cast<int>(this->ioConfig.outputH) : 0;
    int outputW = this->ioConfig.hasOutputShape() ? static_cast<int>(this->ioConfig.outputW) : 0;
    dataLoader.loadManifest(inputFilePath.toStdString(), this->ioConfig, inputC, inputH, inputW, outputC, outputH,
                            outputW);
  } else {
    auto [samples, success] = this->loadSamplesFromOptions("test", inputFilePath);

    if (!success)
      return 1;
    dataLoader.loadFromMemory(std::move(samples), inputC, inputH, inputW);
  }

  if (this->logLevel > LogLevel::QUIET)
    TestSummary::print(this->coreConfig, dataLoader.numSamples());

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

int ANNRunner::predict()
{
  if (!this->parser.isSet("input")) {
    std::cerr << "Error: --input option is required for predict mode.\n";
    return 1;
  }

  QString inputPath = this->parser.value("input");
  QString outputPath = resolvePredictOutputPath(this->parser, this->ioConfig);

  ulong displayProgressReports = (this->logLevel > LogLevel::QUIET) ? this->ioConfig.progressReports : 0;
  std::vector<ANN::Input<float>> inputs =
    ANNLoader::loadInputs(inputPath.toStdString(), this->ioConfig, displayProgressReports);

  if (this->logLevel > LogLevel::QUIET)
    PredictSummary::printANN(this->coreConfig, inputs.size(), inputPath.toStdString(), outputPath.toStdString());

  auto batchStart = std::chrono::system_clock::now();
  std::string startTimeStr = ANN::Utils<float>::formatISO8601();

  setupModeProgressCallback(*this->core, this->logLevel, this->ioConfig.progressReports, "Predicting", inputs.size());

  // The streaming predict API takes a provider that yields one batch at a
  // time. The batch JSON is already loaded into `inputs`, so we just slice it.
  auto sliceProvider = [&inputs](ulong batchSize, ulong batchIndex) {
    ulong start = batchIndex * batchSize;
    ulong end = std::min(start + batchSize, static_cast<ulong>(inputs.size()));

    if (start >= end)
      return ANN::Inputs<float>{};
    return ANN::Inputs<float>(inputs.begin() + start, inputs.begin() + end);
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

std::pair<ANN::Samples<float>, bool> ANNRunner::loadSamplesFromOptions(const std::string& modeName,
                                                                       QString& inputFilePath)
{
  return loadSamplesFromOptionsCommon<ANN::Samples<float>>(
    this->parser, this->logLevel, this->ioConfig, modeName, inputFilePath,
    [this](const std::string& path, ulong progressReports) {
      return ANNLoader::loadSamples(path, this->ioConfig, progressReports);
    },

    [](const std::string& dataPath, const std::string& labelsPath, ulong progressReports) {
      return Utils<float>::loadIDX(dataPath, labelsPath, progressReports);
    });
}

//===================================================================================================================//
//  Training helpers
//===================================================================================================================//

ValidationMetadata ANNRunner::buildValidationMetadata() const
{
  return {this->validationState.enabled, this->validationState.numValSamples, this->validationState.lastValLoss,
          this->validationState.bestValLoss, this->validationState.bestValEpoch};
}

//===================================================================================================================//

void ANNRunner::setupTrainingCallback(const QString& inputFilePath, std::shared_ptr<ANN::Core<float>> validationCore,
                                      std::shared_ptr<Common::TrainingMonitor<float>> trainingMonitor,
                                      const DataLoader<ANN::Sample<float>>* validationDataLoader,
                                      const std::vector<ulong>* validationIndices)
{
  this->lastCallbackEpoch = 0;
  this->lastEpochLoss = 0.0f;

  ulong batchSize = this->coreConfig.trainingConfig.batchSize;
  this->progressBar = std::make_unique<ProgressBar>(this->ioConfig.progressReports, 50, std::max(2UL, batchSize / 2));

  auto tui = this->tui;

  std::shared_ptr<ANN::SampleProvider<float>> validationProviderPtr;

  if (validationDataLoader && validationIndices && !validationIndices->empty()) {
    auto provider = validationDataLoader->makeSampleProvider(*validationIndices, {}, 0.0f, SampleLoadType::Validation);
    validationProviderPtr = std::make_shared<ANN::SampleProvider<float>>(std::move(provider));
  }

  this->validationProviderPtr = validationProviderPtr;

  this->core->setTrainingCallback([this, inputFilePath, validationCore, trainingMonitor, validationProviderPtr,
                                   validationIndices, validationDataLoader,
                                   tui](const Common::TrainingProgress<float>& progress) {
    {
      std::lock_guard<std::mutex> lock(this->epochTransitionMutex);

      // Epoch transition: process epoch-end tasks when a new epoch starts.
      // saveModelInterval controls checkpoint frequency only; epoch
      // transitions must always be processed for TUI, validation, and
      // monitoring logic to work.
      bool epochTransition = progress.currentEpoch > this->lastCallbackEpoch;

      if (epochTransition) {
        const ulong finishedEpoch = this->lastCallbackEpoch;

        // --- Checkpointing (controlled by saveModelInterval) ---
        if (this->ioConfig.saveModelInterval > 0 && finishedEpoch > 0 &&
            finishedEpoch % this->ioConfig.saveModelInterval == 0) {
          std::string checkpointPath =
            ModelSerializer::generateCheckpointPath(inputFilePath, finishedEpoch, this->lastEpochLoss);
          ModelSerializer::saveANNModelToPackage(checkpointPath, *this->core, this->coreConfig, this->ioConfig,
                                                 this->augConfig, this->buildValidationMetadata());
        }

        // --- Validation ---
        bool isBest = false;
        bool monitorShouldStop = false;
        float valLoss = 0.0f;
        bool hasValLoss = false;

        if (finishedEpoch > 0 && this->validationState.enabled && validationCore && validationProviderPtr &&
            validationIndices && finishedEpoch % this->validationState.checkInterval == 0) {
          ulong validationTotal = validationIndices->size();

          validationCore->setParameters(this->core->getParameters());

          setupValidationProgressCallback(*validationCore, tui, validationTotal, progress.totalGPUs);

          // Mute the loading bar while validation streams its own samples through the shared
          // loader, so the "Loading samples" bar isn't hijacked by validation's batch counts
          // (validation has its own "Validating" bar on the progress window).
          auto validationResult = validationCore->test(validationTotal, *validationProviderPtr);

          this->validationState.lastValLoss = validationResult.averageLoss;
          valLoss = validationResult.averageLoss;
          hasValLoss = true;

          if (validationResult.averageLoss < this->validationState.bestValLoss) {
            this->validationState.bestValLoss = validationResult.averageLoss;
            this->validationState.bestValEpoch = finishedEpoch;
          }

          if (trainingMonitor) {
            monitorShouldStop = trainingMonitor->checkEpoch(finishedEpoch, this->lastEpochLoss,
                                                            std::optional<float>(validationResult.averageLoss));
            isBest = trainingMonitor->isNewBest();
          }
        }

        // --- Best model save ---
        if (isBest || progress.isNewBest) {
          std::string bestPath = ModelSerializer::generateBestModelPath(inputFilePath);
          ModelSerializer::saveANNModelToPackage(bestPath, *this->core, this->coreConfig, this->ioConfig,
                                                 this->augConfig, this->buildValidationMetadata());
        }

        // --- TUI history (skip epoch 0 line) ---
        bool isBestEpoch = (isBest || progress.isNewBest);

        if (tui && tui->isInitialized() && this->logLevel > LogLevel::QUIET && finishedEpoch > 0) {
          tui->pushEpochRecord(static_cast<int>(finishedEpoch), this->lastEpochLoss, hasValLoss, valLoss, isBestEpoch);
        }

        // --- Update core epochHistory with correct isBest/hasValLoss/valLoss ---
        // The core's internal monitor is disabled (NN-CLI manages monitoring
        // externally), so the core records isBest=false and hasValLoss=false.
        // Overwrite with the actual values computed above so that any
        // subsequent model save serializes the correct epoch metadata.
        // Guard with finishedEpoch > 0: on resume, the first transition fires
        // with finishedEpoch = 0 and would overwrite the last prepended
        // historical record with stale defaults.
        if (finishedEpoch > 0) {
          auto& epochHistory = this->core->getTrainingMetadata().epochHistory;

          if (!epochHistory.empty()) {
            auto& lastRecord = epochHistory.back();
            lastRecord.isBest = isBestEpoch;
            lastRecord.hasValLoss = hasValLoss;
            lastRecord.valLoss = valLoss;
          }

          // Cache for the final-epoch fixup in finishTraining().
          this->lastIsBest = isBestEpoch;
          this->lastHadValLoss = hasValLoss;
          this->lastValLoss = valLoss;
          this->cacheIsSet = true;
        }

        // --- Monitor stop requests ---
        if (monitorShouldStop) {
          if (tui && tui->isInitialized())
            tui->addEpochLine("[Monitor] Training stopped: " + trainingMonitor->getStopReason());

          this->core->requestStop();
        }

        if (progress.stoppedEarly) {
          if (tui && tui->isInitialized())
            tui->addEpochLine("[Monitor] Training stopped: " + this->core->getTrainingMetadata().stopReason);

          this->core->requestStop();
        }

        this->lastCallbackEpoch = progress.currentEpoch;
      }

      if (this->logLevel > LogLevel::QUIET) {
        ProgressInfo info{progress.currentEpoch, progress.totalEpochs, progress.currentSample, progress.totalSamples,
                          progress.epochLoss,    progress.sampleLoss,  progress.gpuIndex,      progress.totalGPUs};

        if (tui && tui->isInitialized()) {
          std::lock_guard<std::recursive_mutex> tuiLock(tui->getMutex());
          tui->handleResize();
          this->progressBar->update(info, tui->progressWindow());
          tui->refresh();
        } else {
          this->progressBar->update(info);
        }
      }

      if (progress.epochLoss > 0)
        this->lastEpochLoss = progress.epochLoss;
    } // lock_guard released
  });
}

//===================================================================================================================//

int ANNRunner::finishTraining(const QString& inputFilePath)
{
  // Defensive: unreachable in normal flow (train() clears loadedEpochHistory after
  // prepending), kept as safety net against future refactoring.
  if (!this->coreConfig.loadedEpochHistory.empty()) {
    this->core->prependEpochHistory(this->coreConfig.loadedEpochHistory);
    this->coreConfig.loadedEpochHistory.clear();
  }

  auto& trainingMetadata = this->core->getTrainingMetadata();
  auto& epochHistory = trainingMetadata.epochHistory;

  // The epoch transition callback updates epochHistory.back() at the START of
  // epoch N+1 (triggered by the first per-sample callback of the new epoch).
  // The final epoch has no subsequent epoch, so no transition fires and its
  // record keeps the core's default isBest=false / hasValLoss=false / valLoss=0.
  //
  // Run a fresh validation pass for the last epoch and fix up its record.
  //
  // cacheIsSet means at least one transition with finishedEpoch>0 fired
  // (>=2 epochs trained).  lastEpoch>0 covers the numEpochs=1 edge case
  // where no transition populates the cache but one epoch was still trained.
  bool needsFixup = (this->cacheIsSet || trainingMetadata.lastEpoch > 0) && !epochHistory.empty();

  if (needsFixup) {
    const ulong lastEpoch = trainingMetadata.lastEpoch;

    //-- Run validation for the last epoch (fresh values, not stale cache) --//
    bool isBest = false;
    bool hasValLoss = false;
    float valLoss = 0.0f;

    if (this->validationState.enabled && this->validationCore && this->validationProviderPtr &&
        this->validationIndices && lastEpoch % this->validationState.checkInterval == 0) {
      ulong validationTotal = this->validationIndices->size();

      this->validationCore->setParameters(this->core->getParameters());

      setupValidationProgressCallback(*this->validationCore, this->tui, validationTotal, this->coreConfig.numGPUs);

      auto validationResult = this->validationCore->test(validationTotal, *this->validationProviderPtr);

      this->validationState.lastValLoss = validationResult.averageLoss;
      valLoss = validationResult.averageLoss;
      hasValLoss = true;

      if (validationResult.averageLoss < this->validationState.bestValLoss) {
        this->validationState.bestValLoss = validationResult.averageLoss;
        this->validationState.bestValEpoch = lastEpoch;
      }

      if (this->trainingMonitor) {
        this->trainingMonitor->checkEpoch(lastEpoch, this->lastEpochLoss,
                                          std::optional<float>(validationResult.averageLoss));
        isBest = this->trainingMonitor->isNewBest();
      }
    }

    bool isBestEpoch = isBest;

    //-- Update the last epoch history record --//
    auto& lastRecord = epochHistory.back();
    if (hasValLoss) {
      lastRecord.isBest = isBestEpoch;
    }
    // else: preserve the core's isBest (set by internal monitor when validation is disabled)
    lastRecord.hasValLoss = hasValLoss;
    lastRecord.valLoss = valLoss;

    //-- Save best model if the last epoch is a new best --//
    if (isBestEpoch) {
      std::string bestPath = ModelSerializer::generateBestModelPath(inputFilePath);
      ModelSerializer::saveANNModelToPackage(bestPath, *this->core, this->coreConfig, this->ioConfig, this->augConfig,
                                             this->buildValidationMetadata());
    }

    //-- Push the last epoch to TUI (no transition fires after the last epoch) --//
    if (this->tui && this->tui->isInitialized() && this->logLevel > LogLevel::QUIET) {
      this->tui->pushEpochRecord(static_cast<int>(lastEpoch), this->lastEpochLoss, hasValLoss, valLoss, isBestEpoch);
    }
  }

  return finishTrainingCommon(
    this->tui, this->logLevel, this->parser, inputFilePath, *this->core, [this](const std::string& path) {
      ModelSerializer::saveANNModelToPackage(path, *this->core, this->coreConfig, this->ioConfig, this->augConfig,
                                             this->buildValidationMetadata());
    });
}

//===================================================================================================================//
//  Class weight computation — delegates to shared computeClassWeightsFromOutputs() in NN-CLI_Utils.hpp
//===================================================================================================================//
