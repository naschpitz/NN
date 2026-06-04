#include "NN-CLI_CNNRunner.hpp"

#include "NN-CLI_CNNLoader.hpp"
#include "NN-CLI_GpuAugmenter.hpp"
#include "NN-CLI_LossReferenceTable.hpp"
#include <CNN_TrainingMonitor.hpp>
#include <OCLW_Core.hpp>
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
  if (this->parser.isSet("samples") && this->parser.isSet("idx-data")) {
    std::cerr << "Error: Cannot use both --samples and --idx-data. Choose one format.\n";
    return 1;
  }

  // Create and init the ncurses TUI immediately so the user sees it right away.
  this->tui = std::make_shared<TerminalUI>();

  if (this->logLevel > LogLevel::QUIET)
    this->tui->init();

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

  if (this->tui->isInitialized())
    ProgressBar::writeStatus(this->tui->progressWindow(),
                             "Loaded " + std::to_string(dataLoader.numSamples()) + " training samples");

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

    if (this->coreConfig.costFunctionConfig.type == CNN::CostFunctionType::SQUARED_DIFFERENCE)
      this->coreConfig.costFunctionConfig.type = CNN::CostFunctionType::WEIGHTED_SQUARED_DIFFERENCE;

    this->coreConfig.costFunctionConfig.weights = weights;
    this->core = CNN::Core<float>::makeCore(this->coreConfig);
  }

  // Collect config table lines for the TUI config panel.
  ulong numValidationSamples = validationConfig.enabled ? split.validationIndices.size() : 0;
  ulong numOriginalTrainSamples = totalOriginalSamples - numValidationSamples;
  ulong numTrainSamples = validationConfig.enabled ? split.trainIndices.size() : dataLoader.numSamples();

  if (this->logLevel > LogLevel::QUIET) {
    std::vector<std::string> configLines;

    auto trainLines =
      TrainingSummary::collectCNN(this->coreConfig, this->augConfig, numOriginalTrainSamples, numTrainSamples,
                                  numValidationSamples, validationRatio, validationAuto);
    configLines.insert(configLines.end(), trainLines.begin(), trainLines.end());

    ulong numOutputClasses = this->coreConfig.layersConfig.denseLayers.empty()
                               ? 0
                               : this->coreConfig.layersConfig.denseLayers.back().numNeurons;

    if (numOutputClasses >= 2) {
      auto lossLines = LossReferenceTable::collect(numOutputClasses);
      configLines.insert(configLines.end(), lossLines.begin(), lossLines.end());
    }

    this->tui->setConfigLines(configLines);
  }

  // Clear loading status, render config panel
  if (this->tui->isInitialized()) {
    ProgressBar::clearStatus(this->tui->progressWindow());
    this->tui->refreshConfigPanel();
  }

  // When validation is enabled, NN-CLI handles monitoring with validation loss.
  std::shared_ptr<CNN::TrainingMonitor<float>> trainingMonitor;

  if (validationConfig.enabled && this->coreConfig.trainingConfig.monitoringConfig.enabled) {
    trainingMonitor = std::make_shared<CNN::TrainingMonitor<float>>(this->coreConfig.trainingConfig.monitoringConfig);
    this->coreConfig.trainingConfig.monitoringConfig.enabled = false;
    this->core = CNN::Core<float>::makeCore(this->coreConfig);
  }

  std::shared_ptr<CNN::Core<float>> validationCore;

  if (validationConfig.enabled) {
    CNN::CoreConfig<float> validationCoreConfig = this->coreConfig;
    validationCoreConfig.modeType = CNN::ModeType::TEST;

    auto allOutputs = dataLoader.getAllOutputs();
    std::vector<std::vector<float>> validationOutputs;
    validationOutputs.reserve(split.validationIndices.size());

    for (ulong idx : split.validationIndices)
      validationOutputs.push_back(allOutputs[idx]);

    std::vector<float> validationWeights = computeClassWeightsFromOutputs(validationOutputs);
    validationCoreConfig.costFunctionConfig.weights = validationWeights;

    validationCore = std::shared_ptr<CNN::Core<float>>(CNN::Core<float>::makeCore(validationCoreConfig).release());
  }

  std::unique_ptr<GpuAugmenterPool> gpuAugPool;

  if (this->coreConfig.deviceType == CNN::DeviceType::GPU && this->ioConfig.inputType == DataType::IMAGE) {
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

  if (this->tui && this->tui->isInitialized()) {
    auto loadingWin = this->tui->loadingWindow();
    auto& tuiMutex = this->tui->mutex();

    // The loading bar must reserve the same per-GPU suffix width as the epoch bar so the two
    // line up. The prefetch loader runs ahead of the first training update, so we resolve the
    // GPU count up front (same logic the GPU core uses) rather than discovering it dynamically.
    int loadingBarGpus = 1;

    if (this->coreConfig.deviceType == CNN::DeviceType::GPU) {
      OpenCLWrapper::Core::initialize(false);
      int availableGpus = static_cast<int>(OpenCLWrapper::Core::getNumDevices());
      loadingBarGpus =
        (this->coreConfig.numGPUs > 0) ? std::min(availableGpus, this->coreConfig.numGPUs) : availableGpus;
      loadingBarGpus = std::max(1, loadingBarGpus);
    }

    dataLoader.setLoadingCallback(
      [loadingWin, &tuiMutex, loadingBarGpus](ulong current, ulong total, ulong batchNum, ulong totalBatches) {
        std::lock_guard<std::recursive_mutex> lock(tuiMutex);
        ProgressBar::renderLoadingBar(loadingWin, current, total, batchNum, totalBatches, loadingBarGpus);
      });
  }

  if (validationConfig.enabled) {
    auto trainProvider = dataLoader.makeSampleProvider(split.trainIndices, this->augConfig.transforms,
                                                       this->augConfig.augmentationProbability);
    this->core->train(split.trainIndices.size(), trainProvider);
  } else {
    auto sampleProvider =
      dataLoader.makeSampleProvider(this->augConfig.transforms, this->augConfig.augmentationProbability);
    this->core->train(dataLoader.numSamples(), sampleProvider);
  }

  return this->finishTraining(inputFilePath);
}

//===================================================================================================================//

int CNNRunner::test()
{
  if (this->parser.isSet("samples") && this->parser.isSet("idx-data")) {
    std::cerr << "Error: Cannot use both --samples and --idx-data. Choose one format.\n";
    return 1;
  }

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

  if (this->logLevel > LogLevel::QUIET) {
    ulong progressReports = this->ioConfig.progressReports;
    ProgressBar::printLoadingProgress("Testing", 0, dataLoader.numSamples(), progressReports);
    this->core->setProgressCallback([progressReports](ulong current, ulong total) {
      ProgressBar::printLoadingProgress("Testing", current, total, progressReports);
    });
  }

  auto sampleProvider = dataLoader.makeSampleProvider({}, 0.0f);
  CNN::TestResult<float> result = this->core->test(dataLoader.numSamples(), sampleProvider);

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
  std::vector<CNN::Input<float>> inputs =
    CNNLoader::loadInputs(inputPath.toStdString(), this->coreConfig.inputShape, this->ioConfig, displayProgressReports);

  if (this->logLevel > LogLevel::QUIET)
    PredictSummary::printCNN(this->coreConfig, inputs.size(), inputPath.toStdString(), outputPath.toStdString());

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
      return CNN::Inputs<float>{};
    return CNN::Inputs<float>(inputs.begin() + start, inputs.begin() + end);
  };

  CNN::PredictResults<float> results = this->core->predict(inputs.size(), sliceProvider);

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
  // `logits` of the dense head so callers can compute calibration / OOD scores
  // (max-logit, logit-norm, free-energy) that softmax discards.
  std::vector<CNN::Output<float>> outputs;
  std::vector<CNN::Logits<float>> logits;
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

std::pair<CNN::Samples<float>, bool> CNNRunner::loadSamplesFromOptions(const std::string& modeName,
                                                                       QString& inputFilePath)
{
  CNN::Samples<float> samples;

  bool hasJsonSamples = this->parser.isSet("samples");
  bool hasIdxData = this->parser.isSet("idx-data");
  bool hasIdxLabels = this->parser.isSet("idx-labels");

  if (hasJsonSamples && hasIdxData) {
    std::cerr << "Error: Cannot use both --samples and --idx-data. Choose one format.\n";
    return {samples, false};
  }

  const CNN::Shape3D& inputShape = this->coreConfig.inputShape;

  ulong displayProgressReports = (this->logLevel > LogLevel::QUIET) ? this->ioConfig.progressReports : 0;

  if (hasJsonSamples) {
    QString samplesPath = this->parser.value("samples");
    inputFilePath = samplesPath;

    if (this->logLevel >= LogLevel::INFO)
      std::cout << "Loading " << modeName << " samples from JSON: " << samplesPath.toStdString() << "\n";
    samples = CNNLoader::loadSamples(samplesPath.toStdString(), inputShape, this->ioConfig, displayProgressReports);
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

    samples = Utils<float>::loadCNNIDX(idxDataPath.toStdString(), idxLabelsPath.toStdString(), inputShape,
                                       displayProgressReports);
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

ValidationMetadata CNNRunner::buildValidationMetadata() const
{
  return {this->validationState.enabled, this->validationState.numValSamples, this->validationState.lastValLoss,
          this->validationState.bestValLoss, this->validationState.bestValEpoch};
}

//===================================================================================================================//

void CNNRunner::setupTrainingCallback(const QString& inputFilePath, std::shared_ptr<CNN::Core<float>> validationCore,
                                      std::shared_ptr<CNN::TrainingMonitor<float>> trainingMonitor,
                                      const DataLoader<CNN::Sample<float>>* validationDataLoader,
                                      const std::vector<ulong>* validationIndices)
{
  static ulong lastCallbackEpoch = 0;
  static float lastEpochLoss = 0.0f;
  static std::mutex epochTransitionMutex;
  lastCallbackEpoch = 0;
  lastEpochLoss = 0.0f;

  static ProgressBar progressBar(this->ioConfig.progressReports);

  progressBar.setHoldEpochLine(false);

  // TUI is already created in train(); capture it for the callback lambda.
  auto tui = this->tui;

  // Wire the per-phase timing profiler to the CNN library's timing callback.
  this->profiler.reset();
  this->core->setTimingCallback([this](CNN::TimingPhase phase, CNN::TimingEvent event, int gpuIndex) {
    this->profiler.onEvent(phase, event, gpuIndex);
  });

  std::shared_ptr<CNN::SampleProvider<float>> validationProviderPtr;

  if (validationDataLoader && validationIndices && !validationIndices->empty()) {
    auto provider = validationDataLoader->makeSampleProvider(*validationIndices, {}, 0.0f);
    validationProviderPtr = std::make_shared<CNN::SampleProvider<float>>(std::move(provider));
  }

  this->core->setTrainingCallback([this, inputFilePath, validationCore, trainingMonitor, validationProviderPtr,
                                   validationIndices, validationDataLoader,
                                   tui](const CNN::TrainingProgress<float>& progress) {
    {
      std::lock_guard<std::mutex> lock(epochTransitionMutex);

      // Epoch transition: process epoch-end tasks when a new epoch starts.
      // saveModelInterval controls checkpoint frequency only; epoch
      // transitions must always be processed for TUI, validation, and
      // monitoring logic to work.
      bool epochTransition = progress.currentEpoch > lastCallbackEpoch;

      if (epochTransition) {
        const ulong finishedEpoch = lastCallbackEpoch;

        // --- Checkpointing (controlled by saveModelInterval) ---
        std::string checkpointPath;

        if (this->ioConfig.saveModelInterval > 0 && lastCallbackEpoch > 0 &&
            lastCallbackEpoch % this->ioConfig.saveModelInterval == 0) {
          checkpointPath = ModelSerializer::generateCheckpointPath(inputFilePath, lastCallbackEpoch, lastEpochLoss);
          ModelSerializer::saveCNNModel(checkpointPath, *this->core, this->coreConfig, this->ioConfig, this->augConfig,
                                        this->buildValidationMetadata());
        }

        // --- Validation ---
        bool isBest = false;
        bool monitorShouldStop = false;
        float valLoss = 0.0f;
        bool hasValLoss = false;

        if (lastCallbackEpoch > 0 && this->validationState.enabled && validationCore && validationProviderPtr &&
            validationIndices && lastCallbackEpoch % this->validationState.checkInterval == 0) {
          ulong validationTotal = validationIndices->size();

          validationCore->setParameters(this->core->getParameters());
          validationCore->syncParametersToGPU();

          // Show validation progress on the progress window line 2
          if (tui && tui->isInitialized()) {
            validationCore->setProgressCallback([tui, validationTotal](ulong current, ulong) {
              float pct = static_cast<float>(current) / validationTotal * 100.0f;
              std::lock_guard<std::recursive_mutex> tuiLock(tui->mutex());
              ProgressBar::renderValidationBar(tui->progressWindow(), pct);
            });
          }

          // Mute the loading bar while validation streams its own samples through the shared
          // loader, so the "Loading samples" bar isn't hijacked by validation's batch counts
          // (validation has its own "Validating" bar on the progress window).
          validationDataLoader->setLoadingEnabled(false);
          auto validationResult = validationCore->test(validationTotal, *validationProviderPtr);
          validationDataLoader->setLoadingEnabled(true);
          this->validationState.lastValLoss = validationResult.averageLoss;
          valLoss = validationResult.averageLoss;
          hasValLoss = true;

          // Clear the validation status line
          if (tui && tui->isInitialized()) {
            std::lock_guard<std::recursive_mutex> tuiLock(tui->mutex());
            ProgressBar::clearStatus(tui->progressWindow());
          }

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

        // --- Best model save ---
        if (isBest || progress.isNewBest) {
          std::string bestPath = ModelSerializer::generateBestModelPath(inputFilePath);
          ModelSerializer::saveCNNModel(bestPath, *this->core, this->coreConfig, this->ioConfig, this->augConfig,
                                        this->buildValidationMetadata());
        }

        // --- TUI history (skip epoch 0 line) ---
        if (tui && tui->isInitialized() && this->logLevel > LogLevel::QUIET && finishedEpoch > 0) {
          char histLine[256];
          int written = snprintf(histLine, sizeof(histLine), "Epoch %lu: Loss %.6f",
                                 static_cast<unsigned long>(finishedEpoch), static_cast<double>(lastEpochLoss));

          if (hasValLoss)
            snprintf(histLine + written, sizeof(histLine) - static_cast<size_t>(written), "  Val: %.6f",
                     static_cast<double>(valLoss));

          if (isBest || progress.isNewBest)
            snprintf(histLine + strlen(histLine), sizeof(histLine) - strlen(histLine), " [best]");

          if (!checkpointPath.empty())
            snprintf(histLine + strlen(histLine), sizeof(histLine) - strlen(histLine), " (checkpoint)");

          tui->addEpochLine(histLine);
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
        lastCallbackEpoch = progress.currentEpoch;
      }

      if (this->logLevel > LogLevel::QUIET) {
        ProgressInfo info{progress.currentEpoch, progress.totalEpochs, progress.currentSample, progress.totalSamples,
                          progress.epochLoss,    progress.sampleLoss,  progress.gpuIndex,      progress.totalGPUs};

        if (tui && tui->isInitialized()) {
          std::lock_guard<std::recursive_mutex> tuiLock(tui->mutex());

          progressBar.update(info, tui->progressWindow());

          if (progress.epochLoss == 0) {
            auto timingLines = this->profiler.getTimingLines();

            if (!timingLines.empty())
              tui->setTimingLines(timingLines);
          } else {
            tui->setTimingLines({" Timing - no data yet"});
          }

          tui->refresh();
        } else {
          if (progress.epochLoss > 0)
            this->profiler.clearLiveTable(std::cout);

          progressBar.update(info);

          if (progress.epochLoss == 0)
            this->profiler.renderLiveTable(std::cout);
        }
      }
    } // lock_guard released

    if (progress.epochLoss > 0)
      lastEpochLoss = progress.epochLoss;
  });
}

//===================================================================================================================//

int CNNRunner::finishTraining(const QString& inputFilePath)
{
  // Exit ncurses mode before printing final summary to stdout
  if (this->tui) {
    this->tui->shutdown();
    this->tui.reset();
  }

  if (this->logLevel > LogLevel::QUIET) {
    std::cout << "\nTraining completed.\n";
    this->profiler.renderFinalSummary(std::cout);
  }

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

  ModelSerializer::saveCNNModel(outputPathStr, *this->core, this->coreConfig, this->ioConfig, this->augConfig,
                                this->buildValidationMetadata());

  if (this->logLevel > LogLevel::QUIET)
    std::cout << "Model saved to: " << outputPathStr << "\n";
  return 0;
}

//===================================================================================================================//
//  Class weight computation
//===================================================================================================================//

std::vector<float> CNNRunner::computeClassWeightsFromOutputs(const std::vector<std::vector<float>>& outputs)
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
