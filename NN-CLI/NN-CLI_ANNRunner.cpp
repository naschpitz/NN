#include "NN-CLI_ANNRunner.hpp"
#include "NN-CLI_ANNLoader.hpp"

#include "NN-CLI_CalibrateUtils.hpp"
#include "NN-CLI_Loader.hpp"
#include "NN-CLI_DataLoader.hpp"
#include "NN-CLI_DataSplitter.hpp"
#include "NN-CLI_ImageLoader.hpp"
#include "NN-CLI_PredictSummary.hpp"
#include "NN-CLI_RunnerUtils.hpp"
#include "NN-CLI_TestSummary.hpp"
#include "NN-CLI_Utils.hpp"

#include <QDir>
#include <QFile>
#include <QFileInfo>

#include <json.hpp>

#include "Common/Common_Utils.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <QMutex>
#include <numeric>

namespace fs = std::filesystem;

using namespace NN_CLI;

//===================================================================================================================//

ANNRunner::ANNRunner(const QCommandLineParser& parser, LogLevel logLevel, IOConfig& ioConfig,
                     AugmentationConfig& augConfig, std::unique_ptr<ANN::Core<float>>& core,
                     ANN::CoreConfig<float>& coreConfig)
  : Runner(parser, logLevel, ioConfig, augConfig, core, coreConfig)
{
}

//===================================================================================================================//
//  Accessors
//===================================================================================================================//

std::vector<std::string> ANNRunner::getTimingLines(int maxWidth) const
{
    (void)maxWidth;
    return {" Timing not available for ANN"};
}

//===================================================================================================================//

ulong ANNRunner::getNumOutputClasses() const
{
  if (!this->coreConfig.layersConfig.empty())
    return this->coreConfig.layersConfig.back().numNeurons;
  return 0;
}


//===================================================================================================================//
//  Model info overrides
//===================================================================================================================//

ulong ANNRunner::getTotalParameters() const
{
  ulong total = 0;
  const auto& layers = this->coreConfig.layersConfig;

  // Layer 0 is the input layer; weights start at layer 1.
  for (size_t l = 1; l < layers.size(); l++) {
    ulong numNeurons = layers[l].numNeurons;
    ulong prevNumNeurons = layers[l - 1].numNeurons;
    total += numNeurons * prevNumNeurons; // weights
    total += numNeurons;                  // biases
  }

  return total;
}

//===================================================================================================================//

std::string ANNRunner::getNetworkType() const
{
  return "ANN";
}

//===================================================================================================================//

std::string ANNRunner::getInputShapeString() const
{
  // ANN has no explicit input shape in the config; input size is implicit
  // from the first layer's dimensions at training time.
  return "";
}

//===================================================================================================================//

ulong ANNRunner::getNumDenseLayers() const
{
  return this->coreConfig.layersConfig.size();
}
//===================================================================================================================//
//  Mode methods
//===================================================================================================================//

int ANNRunner::train()
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
    auto [samples, success] = this->loadSamplesFromOptions("train", inputFilePath);

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
    this->coreConfig.costFunctionConfig.type = Common::CostFunctionType::WEIGHTED_SQUARED_DIFFERENCE;
    this->coreConfig.costFunctionConfig.weights = weights;
    this->core = ANN::Core<float>::makeCore(this->coreConfig);
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

  std::shared_ptr<Common::TrainingMonitor<float>> trainingMonitor;

  if (validationConfig.enabled && this->coreConfig.trainConfig.monitoringConfig.enabled) {
    trainingMonitor =
      std::make_shared<Common::TrainingMonitor<float>>(this->coreConfig.trainConfig.monitoringConfig);
    this->coreConfig.trainConfig.monitoringConfig.enabled = false;
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

  this->setupTrainingCallback(inputFilePath, validationCore, trainingMonitor,
                              validationConfig.enabled ? &dataLoader : nullptr,
                              validationConfig.enabled ? &split.validationIndices : nullptr);


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
  std::string startTimeStr = Common::Utils::formatISO8601();

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
  std::string endTimeStr = Common::Utils::formatISO8601();
  std::chrono::duration<double> batchElapsed = batchEnd - batchStart;
  double batchDurationSeconds = batchElapsed.count();
  std::string batchDurationFormatted = Common::Utils::formatDuration(batchDurationSeconds);

  return writePredictOutput(results, outputPath, this->ioConfig, this->logLevel, startTimeStr, endTimeStr,
                            batchDurationSeconds, batchDurationFormatted, inputs.size());
}

//===================================================================================================================//
//  Calibration
//===================================================================================================================//

int ANNRunner::calibrate()
{
  //-- CLI-only config (not in CalibrateConfig) ----------------------------
  const std::string& idImagesDir = this->parser.value("id-images").toStdString();
  const std::string oodDir = this->parser.isSet("ood-dir")
                               ? this->parser.value("ood-dir").toStdString()
                               : (fs::current_path() / "extern-datasets" / "ood").string();
  const std::string outputPath = this->parser.isSet("output")
                                   ? this->parser.value("output").toStdString()
                                   : (fs::path(this->parser.value("config").toStdString()).parent_path() /
                                      "threshold.json").string();
  const ulong progressReports = this->ioConfig.progressReports;

  //-- Validate ID images directory ------------------------------------------
  if (!fs::exists(idImagesDir) || !fs::is_directory(idImagesDir)) {
    std::string errMsg = "Error: --id-images " + idImagesDir + " does not exist or is not a directory.";
    std::cerr << errMsg << "\n";
    this->notifyLogMessage(errMsg, true);
    return 1;
  }

  //-- Fetch OOD if needed ---------------------------------------------------
  if (this->coreConfig.calibrateConfig.fetchIfMissing && !NN_CLI::dirHasImages(oodDir)) {
    if (this->logLevel > LogLevel::QUIET) {
      std::string msg = "OOD dir is empty \u2014 fetching DTD + Places365 + synthetic.\n";
      std::cout << msg;
      this->notifyLogMessage(msg, false);
    }
    NN_CLI::ensureOODDataset(oodDir, this->logLevel, [this](const std::string& m, bool err) {
      this->notifyLogMessage(m, err);
    });
  } else if (!NN_CLI::dirHasImages(oodDir)) {
    std::string errMsg = "Error: --ood-dir " + oodDir + " has no images and --no-fetch was set.";
    std::cerr << errMsg << "\n";
    this->notifyLogMessage(errMsg, true);
    return 1;
  }

  //-- Gather + sample -------------------------------------------------------
  std::vector<std::string> idAll = NN_CLI::gatherImages(idImagesDir);
  std::vector<std::string> oodAll = NN_CLI::gatherImages(oodDir);

  if (idAll.empty()) {
    std::string errMsg = "Error: no images found under --id-images " + idImagesDir;
    std::cerr << errMsg << "\n";
    this->notifyLogMessage(errMsg, true);
    return 1;
  }

  if (oodAll.empty()) {
    std::string errMsg = "Error: no images found under --ood-dir " + oodDir;
    std::cerr << errMsg << "\n";
    this->notifyLogMessage(errMsg, true);
    return 1;
  }

  std::vector<std::string> idSample = NN_CLI::sampleImages(idAll, this->coreConfig.calibrateConfig.idSampleCount, 42);
  std::vector<std::string> oodSample = NN_CLI::sampleImages(oodAll, this->coreConfig.calibrateConfig.oodSampleCount, 42);

  if (this->logLevel > LogLevel::QUIET) {
    std::string msg = "Sampled " + std::to_string(idSample.size()) + " ID images (of " + std::to_string(idAll.size()) +
                      " available)\n"
                      "Sampled " +
                      std::to_string(oodSample.size()) + " OOD images (of " + std::to_string(oodAll.size()) +
                      " available)\n\n";
    std::cout << msg;
    this->notifyLogMessage(msg, false);
  }

  //-- Predict + free-energy -------------------------------------------------
  auto t0 = std::chrono::system_clock::now();

  int targetC = static_cast<int>(this->ioConfig.inputC);
  int targetH = static_cast<int>(this->ioConfig.inputH);
  int targetW = static_cast<int>(this->ioConfig.inputW);

  auto wrapFn = [](std::vector<float>&& flat) { return std::move(flat); };

  std::vector<std::vector<float>> idLogits =
    NN_CLI::runPredictImpl<ANN::Inputs<float>>(*this->core, idSample, "Predicting ID", targetC, targetH, targetW,
                                                progressReports, this->logLevel, wrapFn);
  std::vector<std::vector<float>> oodLogits =
    NN_CLI::runPredictImpl<ANN::Inputs<float>>(*this->core, oodSample, "Predicting OOD", targetC, targetH, targetW,
                                                progressReports, this->logLevel, wrapFn);

  std::vector<float> idEnergies, oodEnergies;
  idEnergies.reserve(idLogits.size());
  oodEnergies.reserve(oodLogits.size());

  for (const auto& l : idLogits)
    idEnergies.push_back(NN_CLI::computeFreeEnergy(l));

  for (const auto& l : oodLogits)
    oodEnergies.push_back(NN_CLI::computeFreeEnergy(l));

  std::sort(idEnergies.begin(), idEnergies.end());
  std::sort(oodEnergies.begin(), oodEnergies.end());

  //-- Write threshold.json --------------------------------------------------
  auto writeThreshold = [this](const std::string& outputPath, const std::vector<float>& idSorted,
                               const std::vector<float>& oodSorted, double idPercentile) {
    auto stats = [](const std::vector<float>& sorted, const std::vector<double>& ps) {
      nlohmann::ordered_json out;
      out["n"] = sorted.size();
      if (!sorted.empty()) {
        out["min"] = NN_CLI::roundTo(sorted.front(), 4);
        out["max"] = NN_CLI::roundTo(sorted.back(), 4);
      }
      double mean = 0.0;
      for (float v : sorted)
        mean += v;
      if (!sorted.empty())
        mean /= sorted.size();
      out["mean"] = NN_CLI::roundTo(mean, 4);
      for (double p : ps) {
        char key[16];
        std::snprintf(key, sizeof(key), "p%g", p);
        out[key] = NN_CLI::roundTo(NN_CLI::computePercentile(sorted, p), 4);
      }
      return out;
    };

    float threshold = NN_CLI::computePercentile(idSorted, idPercentile);

    std::size_t idAccepted = 0;
    for (float e : idSorted) {
      if (e <= threshold)
        idAccepted++;
    }

    std::size_t oodRejected = 0;
    for (float e : oodSorted) {
      if (e > threshold)
        oodRejected++;
    }

    nlohmann::ordered_json doc;
    doc["freeEnergyThreshold"] = NN_CLI::roundTo(threshold, 4);
    doc["idPercentileUsed"] = idPercentile;
    doc["rule"] = "predicted_ood = (free_energy > freeEnergyThreshold)";
    doc["idStats"] = stats(idSorted, {1, 5, 50, 90, 95, 99});
    doc["oodStats"] = stats(oodSorted, {1, 5, 50, 95, 99});

    nlohmann::ordered_json conf;
    conf["idAccepted"] = idAccepted;
    conf["idRejected"] = idSorted.size() - idAccepted;
    conf["oodAccepted"] = oodSorted.size() - oodRejected;
    conf["oodRejected"] = oodRejected;
    conf["idAcceptanceRate"] = NN_CLI::roundTo(static_cast<double>(idAccepted) / idSorted.size(), 4);
    conf["oodRejectionRate"] = NN_CLI::roundTo(static_cast<double>(oodRejected) / oodSorted.size(), 4);
    doc["confusion"] = conf;

    std::ofstream f(outputPath);
    if (!f)
      throw std::runtime_error("Cannot write " + outputPath);

    f << doc.dump(2) << "\n";

    if (this->logLevel > LogLevel::QUIET)
      std::cout << doc.dump(2) << "\n";
  };

  writeThreshold(outputPath, idEnergies, oodEnergies, this->coreConfig.calibrateConfig.idPercentile);

  auto t1 = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed = t1 - t0;

  if (this->logLevel > LogLevel::QUIET) {
    std::string doneMsg = "\nCalibration done in " + Common::Utils::formatDuration(elapsed.count()) +
                          "\nThreshold written to: " + outputPath + "\n";
    std::cout << doneMsg;
    this->notifyLogMessage(doneMsg, false);
  }

  std::string summary = "Calibration completed | ID: " + std::to_string(idEnergies.size()) +
                        " | OOD: " + std::to_string(oodEnergies.size()) + " | Output: " + outputPath;
  this->notifyTrainingFinished(true, summary);

  return 0;
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

void ANNRunner::setupTrainingCallback(const QString& inputFilePath, std::shared_ptr<ANN::Core<float>> validationCore,
                                      std::shared_ptr<Common::TrainingMonitor<float>> trainingMonitor,
                                      const DataLoader<ANN::Sample<float>>* validationDataLoader,
                                      const std::vector<ulong>* validationIndices)
{
  this->lastEpochLoss = 0.0f;

  ulong batchSize = this->coreConfig.trainConfig.batchSize;
  int totalEpochs = static_cast<int>(this->coreConfig.trainConfig.numEpochs);

  std::shared_ptr<ANN::SampleProvider<float>> validationProviderPtr;

  if (validationDataLoader && validationIndices && !validationIndices->empty()) {
    auto provider = validationDataLoader->makeSampleProvider(*validationIndices, {}, 0.0f, SampleLoadType::Validation);
    validationProviderPtr = std::make_shared<ANN::SampleProvider<float>>(std::move(provider));
  }

  // Live progress callback: fires per batch. It only drives the progress
  // display — every epoch-boundary task lives in the epoch-completed callback.
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
      ModelSerializer::saveANNModelToPackage(checkpointPath, *this->core, this->coreConfig, this->ioConfig,
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
      ModelSerializer::saveANNModelToPackage(bestPath, *this->core, this->coreConfig, this->ioConfig, this->augConfig,
                                             this->buildValidationMetadata());
    }

    // --- Write the validation results into this epoch's history record ---
    // The core's internal monitor is disabled (NN-CLI monitors externally), so
    // it recorded isBest=false / hasValLoss=false. The just-completed epoch is
    // epochHistory.back() (the core appended it immediately before this call).
    auto& epochHistory = this->core->getTrainMetadata().epochHistory;

    if (!epochHistory.empty()) {
      auto& lastRecord = epochHistory.back();
      lastRecord.isBest = isBestEpoch;
      lastRecord.hasValLoss = hasValLoss;
      lastRecord.valLoss = valLoss;
    }

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
      std::string stopMsg = "[Monitor] Training stopped: " + this->core->getTrainMetadata().stopReason;

      this->notifyLogMessage(stopMsg, false);
      this->core->requestStop();
    }
  });
}

//===================================================================================================================//
//  Model saving (override from Runner base)
//===================================================================================================================//

void ANNRunner::doSaveModel(const std::string& outputPath)
{
  ModelSerializer::saveANNModelToPackage(outputPath, *this->core, this->coreConfig, this->ioConfig, this->augConfig,
                                         this->buildValidationMetadata());
}

//===================================================================================================================//
//  Class weight computation — delegates to shared computeClassWeightsFromOutputs() in NN-CLI_Utils.hpp
//===================================================================================================================//
