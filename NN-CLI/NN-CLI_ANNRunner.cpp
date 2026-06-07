#include "NN-CLI_ANNRunner.hpp"

#include "NN-CLI_ANNLoader.hpp"
#include "NN-CLI_LossReferenceTable.hpp"
#include "NN-CLI_DataSplitter.hpp"
#include "NN-CLI_ImageLoader.hpp"
#include "NN-CLI_ProgressBar.hpp"
#include "NN-CLI_PredictSummary.hpp"
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
  if (this->parser.isSet("samples") && this->parser.isSet("idx-data")) {
    std::cerr << "Error: Cannot use both --samples and --idx-data. Choose one format.\n";
    return 1;
  }

  this->tui = std::make_shared<TerminalUI>();

  if (this->logLevel > LogLevel::QUIET)
    this->tui->init();

  this->trainingTui_.attach(this->tui, [this]() {
    ulong cw = this->tui->leftWidth() > 4 ? this->tui->leftWidth() - 4 : 80;
    this->regenerateConfigLines(cw);
  });

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
    this->coreConfig.costFunctionConfig.type = ANN::CostFunctionType::WEIGHTED_SQUARED_DIFFERENCE;
    this->coreConfig.costFunctionConfig.weights = weights;
    this->core = ANN::Core<float>::makeCore(this->coreConfig);
  }

  ulong numValidationSamples = validationConfig.enabled ? split.validationIndices.size() : 0;
  ulong numOriginalTrainSamples = totalOriginalSamples - numValidationSamples;
  ulong numTrainSamples = validationConfig.enabled ? split.trainIndices.size() : dataLoader.numSamples();

  if (this->logLevel > LogLevel::QUIET) {
    this->cachedNumOrigTrainSamples_ = numOriginalTrainSamples;
    this->cachedNumTrainSamples_ = numTrainSamples;
    this->cachedNumValSamples_ = numValidationSamples;
    this->cachedValRatio_ = validationRatio;
    this->cachedValAuto_ = validationAuto;
    this->cachedNumOutputClasses_ =
      this->coreConfig.layersConfig.empty() ? 0 : this->coreConfig.layersConfig.back().numNeurons;
    this->configLinesLoaded_ = true;

    // Fit the config tables to the left panel when the TUI is active; otherwise let them size to
    // the full terminal (maxWidth 0).
    ulong cw = this->tui->leftWidth() > 4 ? this->tui->leftWidth() - 4 : 0;
    this->regenerateConfigLines(cw);
  }

  if (this->tui->isInitialized()) {
    this->tui->refreshConfigPanel();
  }

  std::shared_ptr<ANN::TrainingMonitor<float>> trainingMonitor;

  if (validationConfig.enabled && this->coreConfig.trainingConfig.monitoringConfig.enabled) {
    trainingMonitor = std::make_shared<ANN::TrainingMonitor<float>>(this->coreConfig.trainingConfig.monitoringConfig);
    this->coreConfig.trainingConfig.monitoringConfig.enabled = false;
    this->core = ANN::Core<float>::makeCore(this->coreConfig);
  }

  std::shared_ptr<ANN::Core<float>> validationCore;

  if (validationConfig.enabled) {
    ANN::CoreConfig<float> validationCoreConfig = this->coreConfig;
    validationCoreConfig.modeType = ANN::ModeType::TEST;

    auto allOutputs = dataLoader.getAllOutputs();
    std::vector<std::vector<float>> validationOutputs;
    validationOutputs.reserve(split.validationIndices.size());

    for (ulong idx : split.validationIndices)
      validationOutputs.push_back(allOutputs[idx]);

    std::vector<float> validationWeights = computeClassWeightsFromOutputs(validationOutputs);
    validationCoreConfig.costFunctionConfig.weights = validationWeights;

    validationCore = std::shared_ptr<ANN::Core<float>>(ANN::Core<float>::makeCore(validationCoreConfig).release());
  }

  this->setupTrainingCallback(inputFilePath, validationCore, trainingMonitor,
                              validationConfig.enabled ? &dataLoader : nullptr,
                              validationConfig.enabled ? &split.validationIndices : nullptr);

  if (this->tui && this->tui->isInitialized()) {
    this->trainingTui_.resolveBarGpus(this->coreConfig.deviceType == ANN::DeviceType::GPU, this->coreConfig.numGPUs);
    dataLoader.setLoadingCallback(this->trainingTui_.loadingCallback());
  }

  if (validationConfig.enabled) {
    auto trainProvider = dataLoader.makeSampleProvider(split.trainIndices, this->augConfig.transforms,
                                                       this->augConfig.augmentationProbability);
    this->trainingTui_.markLoadingFinished();
    this->core->train(split.trainIndices.size(), trainProvider);
  } else {
    auto sampleProvider =
      dataLoader.makeSampleProvider(this->augConfig.transforms, this->augConfig.augmentationProbability);
    this->trainingTui_.markLoadingFinished();
    this->core->train(dataLoader.numSamples(), sampleProvider);
  }

  return this->finishTraining(inputFilePath);
}

//===================================================================================================================//

int ANNRunner::test()
{
  if (this->parser.isSet("samples") && this->parser.isSet("idx-data")) {
    std::cerr << "Error: Cannot use both --samples and --idx-data. Choose one format.\n";
    return 1;
  }

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
    TestSummary::printANN(this->coreConfig, dataLoader.numSamples());

  if (this->logLevel > LogLevel::QUIET) {
    ulong progressReports = this->ioConfig.progressReports;
    ProgressBar::printLoadingProgress("Testing", 0, dataLoader.numSamples(), progressReports);
    this->core->setProgressCallback([progressReports](ulong current, ulong total) {
      ProgressBar::printLoadingProgress("Testing", current, total, progressReports);
    });
  }

  auto sampleProvider = dataLoader.makeSampleProvider({}, 0.0f);
  ANN::TestResult<float> result = this->core->test(dataLoader.numSamples(), sampleProvider);

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
  QString outputPath;

  if (this->parser.isSet("output")) {
    outputPath = this->parser.value("output");
  } else {
    QFileInfo inputInfo(inputPath);
    QDir inputDir = inputInfo.absoluteDir();
    QDir outputDir(inputDir.filePath("output"));

    if (!outputDir.exists())
      inputDir.mkdir("output");

    if (this->ioConfig.outputType == DataType::IMAGE) {
      outputPath = outputDir.filePath("predict_" + inputInfo.completeBaseName());
    } else {
      outputPath = outputDir.filePath("predict_" + inputInfo.completeBaseName() + ".json");
    }
  }

  ulong displayProgressReports = (this->logLevel > LogLevel::QUIET) ? this->ioConfig.progressReports : 0;
  std::vector<ANN::Input<float>> inputs =
    ANNLoader::loadInputs(inputPath.toStdString(), this->ioConfig, displayProgressReports);

  if (this->logLevel > LogLevel::QUIET)
    PredictSummary::printANN(this->coreConfig, inputs.size(), inputPath.toStdString(), outputPath.toStdString());

  auto batchStart = std::chrono::system_clock::now();
  std::string startTimeStr = ANN::Utils<float>::formatISO8601();

  if (this->logLevel > LogLevel::QUIET) {
    ulong progressReports = this->ioConfig.progressReports;
    ProgressBar::printLoadingProgress("Predicting", 0, inputs.size(), progressReports);
    this->core->setProgressCallback([progressReports](ulong current, ulong total) {
      ProgressBar::printLoadingProgress("Predicting", current, total, progressReports);
    });
  }

  // The streaming predict API takes a provider that yields one batch at a
  // time. The batch JSON is already loaded into `inputs`, so we just slice it.
  auto sliceProvider = [&inputs](ulong batchSize, ulong batchIndex) {
    ulong start = batchIndex * batchSize;
    ulong end = std::min(start + batchSize, static_cast<ulong>(inputs.size()));

    if (start >= end)
      return ANN::Inputs<float>{};
    return ANN::Inputs<float>(inputs.begin() + start, inputs.begin() + end);
  };

  ANN::PredictResults<float> results = this->core->predict(inputs.size(), sliceProvider);

  auto batchEnd = std::chrono::system_clock::now();
  std::string endTimeStr = ANN::Utils<float>::formatISO8601();
  std::chrono::duration<double> batchElapsed = batchEnd - batchStart;
  double batchDurationSeconds = batchElapsed.count();
  std::string batchDurationFormatted = ANN::Utils<float>::formatDuration(batchDurationSeconds);

  // When outputType is IMAGE, save images to a folder
  if (this->ioConfig.outputType == DataType::IMAGE) {
    if (!this->ioConfig.hasOutputShape()) {
      std::cerr << "Error: outputType is 'image' but no outputShape provided in config.\n";
      return 1;
    }

    QDir outDir(outputPath);

    if (!outDir.exists())
      QDir().mkpath(outputPath);

    for (size_t i = 0; i < results.size(); ++i) {
      QString imgName = QString::number(i) + ".png";
      std::string imgPath = outDir.filePath(imgName).toStdString();
      ImageLoader::saveImage(imgPath, results[i].output, static_cast<int>(this->ioConfig.outputC),
                             static_cast<int>(this->ioConfig.outputH), static_cast<int>(this->ioConfig.outputW));
    }

    if (this->logLevel > LogLevel::QUIET) {
      std::cout << "Predict images saved to: " << outputPath.toStdString() << "\n";
      std::cout << "  Images: " << results.size() << "\n";
      std::cout << "  Shape: " << this->ioConfig.outputC << "x" << this->ioConfig.outputH << "x"
                << this->ioConfig.outputW << "\n";
      std::cout << "  Duration: " << batchDurationFormatted << "\n";
    }

    return 0;
  }

  // Standard vector output: save as JSON.
  // For each input we emit both the post-activation `output` and the pre-activation
  // `logits` of the last layer so callers can compute calibration / OOD scores
  // (max-logit, logit-norm, free-energy) that softmax discards.
  std::vector<ANN::Output<float>> outputs;
  std::vector<ANN::Logits<float>> logits;
  outputs.reserve(results.size());
  logits.reserve(results.size());

  for (const auto& r : results) {
    outputs.push_back(r.output);
    logits.push_back(r.logits);
  }

  nlohmann::ordered_json resultJson;
  nlohmann::ordered_json predictMetadataJson;
  predictMetadataJson["startTime"] = startTimeStr;
  predictMetadataJson["endTime"] = endTimeStr;
  predictMetadataJson["durationSeconds"] = batchDurationSeconds;
  predictMetadataJson["durationFormatted"] = batchDurationFormatted;
  predictMetadataJson["numInputs"] = inputs.size();
  resultJson["predictMetadata"] = predictMetadataJson;
  resultJson["outputs"] = outputs;
  resultJson["logits"] = logits;

  QFile outputFile(outputPath);

  if (!outputFile.open(QIODevice::WriteOnly | QIODevice::Text)) {
    std::cerr << "Error: Failed to open output file: " << outputPath.toStdString() << "\n";
    return 1;
  }

  std::string jsonStr = resultJson.dump(2);
  outputFile.write(jsonStr.c_str(), jsonStr.size());
  outputFile.close();

  if (this->logLevel > LogLevel::QUIET)
    std::cout << "Predict result saved to: " << outputPath.toStdString() << "\n";
  return 0;
}

//===================================================================================================================//
//  Sample loading
//===================================================================================================================//

std::pair<ANN::Samples<float>, bool> ANNRunner::loadSamplesFromOptions(const std::string& modeName,
                                                                       QString& inputFilePath)
{
  ANN::Samples<float> samples;

  bool hasJsonSamples = this->parser.isSet("samples");
  bool hasIdxData = this->parser.isSet("idx-data");
  bool hasIdxLabels = this->parser.isSet("idx-labels");

  if (hasJsonSamples && hasIdxData) {
    std::cerr << "Error: Cannot use both --samples and --idx-data. Choose one format.\n";
    return {samples, false};
  }

  ulong displayProgressReports = (this->logLevel > LogLevel::QUIET) ? this->ioConfig.progressReports : 0;

  if (hasJsonSamples) {
    QString samplesPath = this->parser.value("samples");
    inputFilePath = samplesPath;

    if (this->logLevel >= LogLevel::INFO)
      std::cout << "Loading " << modeName << " samples from JSON: " << samplesPath.toStdString() << "\n";
    samples = ANNLoader::loadSamples(samplesPath.toStdString(), this->ioConfig, displayProgressReports);
  } else if (hasIdxData) {
    if (!hasIdxLabels) {
      std::cerr << "Error: --idx-labels is required when using --idx-data.\n";
      return {samples, false};
    }

    QString idxDataPath = this->parser.value("idx-data");
    QString idxLabelsPath = this->parser.value("idx-labels");
    inputFilePath = idxDataPath;

    if (this->logLevel >= LogLevel::INFO) {
      std::cout << "Loading " << modeName << " samples from IDX:\n";
      std::cout << "  Data:   " << idxDataPath.toStdString() << "\n";
      std::cout << "  Labels: " << idxLabelsPath.toStdString() << "\n";
    }

    samples = Utils<float>::loadANNIDX(idxDataPath.toStdString(), idxLabelsPath.toStdString(), displayProgressReports);
  } else {
    std::cerr << "Error: " << modeName << " requires either --samples (JSON) or --idx-data and --idx-labels (IDX).\n";
    return {samples, false};
  }

  if (this->logLevel >= LogLevel::INFO)
    std::cout << "Loaded " << samples.size() << " " << modeName << " samples.\n";

  return {samples, true};
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
                                      std::shared_ptr<ANN::TrainingMonitor<float>> trainingMonitor,
                                      const DataLoader<ANN::Sample<float>>* validationDataLoader,
                                      const std::vector<ulong>* validationIndices)
{
  static ulong lastCallbackEpoch = 0;
  static float lastEpochLoss = 0.0f;
  static std::mutex epochTransitionMutex;
  lastCallbackEpoch = 0;
  lastEpochLoss = 0.0f;

  // NOTE: static ProgressBar persists across training runs.
  // Window size is fixed at first invocation; changing batchSize between runs has no effect.
  ulong batchSize = this->coreConfig.trainingConfig.batchSize;
  static ProgressBar progressBar(this->ioConfig.progressReports, 50, std::max(2UL, batchSize / 2));

  auto tui = this->tui;

  std::shared_ptr<ANN::SampleProvider<float>> validationProviderPtr;

  if (validationDataLoader && validationIndices && !validationIndices->empty()) {
    auto provider = validationDataLoader->makeSampleProvider(*validationIndices, {}, 0.0f);
    validationProviderPtr = std::make_shared<ANN::SampleProvider<float>>(std::move(provider));
  }

  this->core->setTrainingCallback([this, inputFilePath, validationCore, trainingMonitor, validationProviderPtr,
                                   validationIndices, validationDataLoader,
                                   tui](const ANN::TrainingProgress<float>& progress) {
    {
      std::lock_guard<std::mutex> lock(epochTransitionMutex);

      if (this->ioConfig.saveModelInterval > 0 && progress.currentEpoch > lastCallbackEpoch) {
        std::string checkpointPath;

        if (lastCallbackEpoch > 0 && lastCallbackEpoch % this->ioConfig.saveModelInterval == 0) {
          checkpointPath = ModelSerializer::generateCheckpointPath(inputFilePath, lastCallbackEpoch, lastEpochLoss);
          ModelSerializer::saveANNModel(checkpointPath, *this->core, this->coreConfig, this->ioConfig, this->augConfig,
                                        this->buildValidationMetadata());
        }

        bool isBest = false;
        bool monitorShouldStop = false;
        float valLoss = 0.0f;
        bool hasValLoss = false;

        if (lastCallbackEpoch > 0 && this->validationState.enabled && validationCore && validationProviderPtr &&
            validationIndices && lastCallbackEpoch % this->validationState.checkInterval == 0) {
          ulong validationTotal = validationIndices->size();

          validationCore->setParameters(this->core->getParameters());

          if (tui && tui->isInitialized()) {
            int validationGpus = std::max(1, progress.totalGPUs);
            validationCore->setProgressCallback([tui, validationTotal, validationGpus](ulong current, ulong) {
              float pct = static_cast<float>(current) / validationTotal * 100.0f;
              std::lock_guard<std::recursive_mutex> tuiLock(tui->mutex());
              tui->handleResize();
              ProgressBar::renderValidationBar(tui->progressWindow(), pct, validationGpus);
            });
          }

          // Mute the loading bar while validation streams its own samples through the shared
          // loader, so the "Loading samples" bar isn't hijacked by validation's batch counts
          // (validation has its own "Validating" bar on the progress window).
          validationDataLoader->setLoadingEnabled(false);
          auto validationResult = validationCore->test(validationTotal, *validationProviderPtr);
          validationDataLoader->setLoadingEnabled(true);

          // Loading callbacks may have been suppressed during validation,
          // causing the training bar to freeze. Catch it up now.
          if (this->logLevel > LogLevel::QUIET)
            this->trainingTui_.markCurrentLoadComplete();

          this->validationState.lastValLoss = validationResult.averageLoss;
          valLoss = validationResult.averageLoss;
          hasValLoss = true;

          if (validationResult.averageLoss < this->validationState.bestValLoss) {
            this->validationState.bestValLoss = validationResult.averageLoss;
            this->validationState.bestValEpoch = lastCallbackEpoch;
          }

          if (trainingMonitor) {
            monitorShouldStop = trainingMonitor->checkEpoch(lastCallbackEpoch, lastEpochLoss,
                                                            std::optional<float>(validationResult.averageLoss));
            isBest = trainingMonitor->isNewBest();
          }
        }

        if (isBest || progress.isNewBest) {
          std::string bestPath = ModelSerializer::generateBestModelPath(inputFilePath);
          ModelSerializer::saveANNModel(bestPath, *this->core, this->coreConfig, this->ioConfig, this->augConfig,
                                        this->buildValidationMetadata());
        }

        if (tui && tui->isInitialized() && this->logLevel > LogLevel::QUIET && lastCallbackEpoch > 0) {
          bool isBestEpoch = (isBest || progress.isNewBest);
          tui->pushEpochRecord(static_cast<int>(lastCallbackEpoch), lastEpochLoss, hasValLoss, valLoss, isBestEpoch);
        }

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

        lastCallbackEpoch = progress.currentEpoch;
      }

      if (this->logLevel > LogLevel::QUIET) {
        ProgressInfo info{progress.currentEpoch, progress.totalEpochs, progress.currentSample, progress.totalSamples,
                          progress.epochLoss,    progress.sampleLoss,  progress.gpuIndex,      progress.totalGPUs};

        if (tui && tui->isInitialized()) {
          std::lock_guard<std::recursive_mutex> tuiLock(tui->mutex());
          tui->handleResize();
          progressBar.update(info, tui->progressWindow());
          tui->refresh();
        } else {
          progressBar.update(info);
        }
      }
    } // lock_guard released

    if (progress.epochLoss > 0)
      lastEpochLoss = progress.epochLoss;
  });
}

//===================================================================================================================//

void ANNRunner::regenerateConfigLines(ulong maxWidth)
{
  if (!this->configLinesLoaded_)
    return;

  std::vector<std::string> configLines;

  auto trainLines = TrainingSummary::collectANN(this->coreConfig, this->augConfig, this->cachedNumOrigTrainSamples_,
                                                this->cachedNumTrainSamples_, this->cachedNumValSamples_,
                                                this->cachedValRatio_, this->cachedValAuto_, maxWidth);
  configLines.insert(configLines.end(), trainLines.begin(), trainLines.end());

  if (this->cachedNumOutputClasses_ >= 2) {
    auto lossLines = LossReferenceTable::collect(this->cachedNumOutputClasses_, maxWidth);
    configLines.insert(configLines.end(), lossLines.begin(), lossLines.end());
  }

  this->tui->setConfigLines(configLines);
}

//===================================================================================================================//

int ANNRunner::finishTraining(const QString& inputFilePath)
{
  if (this->tui) {
    this->tui->shutdown();
    this->tui.reset();
  }

  if (this->logLevel > LogLevel::QUIET)
    std::cout << "\nTraining completed.\n";

  const auto& trainingConfig = this->core->getTrainingConfig();
  const auto& trainingMetadata = this->core->getTrainingMetadata();
  ulong actualEpochs = trainingMetadata.lastEpoch > 0 ? trainingMetadata.lastEpoch : trainingConfig.numEpochs;

  std::string outputPathStr;

  if (this->parser.isSet("output")) {
    outputPathStr = this->parser.value("output").toStdString();
  } else {
    outputPathStr = ModelSerializer::generateDefaultOutputPath(inputFilePath, actualEpochs, trainingMetadata.numSamples,
                                                               trainingMetadata.finalLoss);
  }

  ModelSerializer::saveANNModel(outputPathStr, *this->core, this->coreConfig, this->ioConfig, this->augConfig,
                                this->buildValidationMetadata());

  if (this->logLevel > LogLevel::QUIET)
    std::cout << "Model saved to: " << outputPathStr << "\n";
  return 0;
}

//===================================================================================================================//
//  Class weight computation
//===================================================================================================================//

std::vector<float> ANNRunner::computeClassWeightsFromOutputs(const std::vector<std::vector<float>>& outputs)
{
  if (outputs.empty())
    return {};

  ulong numClasses = outputs[0].size();
  std::vector<ulong> classCounts(numClasses, 0);

  for (const auto& output : outputs) {
    ulong cls = static_cast<ulong>(std::distance(output.begin(), std::max_element(output.begin(), output.end())));

    if (cls < numClasses)
      classCounts[cls]++;
  }

  ulong totalSamples = outputs.size();
  std::vector<float> weights(numClasses, 1.0f);

  for (ulong c = 0; c < numClasses; c++) {
    if (classCounts[c] > 0) {
      weights[c] =
        static_cast<float>(totalSamples) / (static_cast<float>(numClasses) * static_cast<float>(classCounts[c]));
    }
  }

  return weights;
}
