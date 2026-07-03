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
                     ANN::CoreConfig<float>& coreConfig, const QString& configPath)
  : Runner(parser, logLevel, ioConfig, augConfig, core, coreConfig, configPath)
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
    total += numNeurons; // biases
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
  if (NN_CLI::Utils<float>::checkSamplesIdxDataConflict(this->parser))
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

    std::vector<float> weights = NN_CLI::Utils<float>::computeClassWeightsFromOutputs(trainingOutputs);
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

  emit this->modelInfoUpdated("totalOriginalSamples", std::to_string(totalOriginalSamples));
  emit this->modelInfoUpdated("numTrainSamples", std::to_string(numTrainSamples));
  emit this->modelInfoUpdated("numValidationSamples", std::to_string(numValidationSamples));

  std::shared_ptr<Common::TrainMonitor<float>> trainMonitor;

  if (validationConfig.enabled && this->coreConfig.trainConfig.monitoringConfig.enabled) {
    trainMonitor = std::make_shared<Common::TrainMonitor<float>>(this->coreConfig.trainConfig.monitoringConfig);
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

    std::vector<float> validationWeights = NN_CLI::Utils<float>::computeClassWeightsFromOutputs(validationOutputs);
    validationCoreConfig.costFunctionConfig.weights = validationWeights;

    validationCore = ANN::Core<float>::makeCore(validationCoreConfig);
  }

  this->setupTrainCallback(inputFilePath, validationCore, trainMonitor,
                           validationConfig.enabled ? &dataLoader : nullptr,
                           validationConfig.enabled ? &split.validationIndices : nullptr);

  // Prepend loaded epoch history into the core before training starts, so
  // checkpoints during training serialize the full history, not just new epochs.
  if (!this->coreConfig.loadedTrainMetadata.epochHistory.empty()) {
    this->core->prependEpochHistory(this->coreConfig.loadedTrainMetadata.epochHistory);
    this->coreConfig.loadedTrainMetadata.epochHistory.clear();
  }

  // Drive the "Samples" data-loading progress bar: the DataLoader emits this
  // signal (from its worker threads) as each batch is loaded/augmented.
  QObject::connect(
    &dataLoader.getDataLoaderSignals(), &DataLoaderSignals::loadingProgress, this,
    [this](ulong current, ulong total, ulong batchIndex, ulong totalBatches, SampleLoadType loadType) {
      emit this->sampleLoadProgress(current, total, batchIndex, totalBatches, loadType == SampleLoadType::Validation);
    },

    Qt::DirectConnection);

  if (validationConfig.enabled) {
    auto trainProvider = dataLoader.makeSampleProvider(
      this->augConfig.transforms, this->augConfig.augmentationProbability, SampleLoadType::Train, split.trainIndices);
    this->core->train(split.trainIndices.size(), trainProvider);
  } else {
    auto sampleProvider = dataLoader.makeSampleProvider(this->augConfig.transforms,
                                                        this->augConfig.augmentationProbability, SampleLoadType::Train);
    this->core->train(dataLoader.numSamples(), sampleProvider);
  }

  return this->finishTrain(inputFilePath);
}

//===================================================================================================================//

int ANNRunner::test()
{
  if (NN_CLI::Utils<float>::checkSamplesIdxDataConflict(this->parser))
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

  NN_CLI::Utils<float>::setupModeProgressCallback(*this->core, this->logLevel, this->ioConfig.progressReports,
                                                  "Testing", dataLoader.numSamples());

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

    if (!result.confusionMatrix.empty()) {
      const Common::ConfusionMatrix<float>& cm = result.confusionMatrix;
      std::cout << "\nConfusion Matrix (rows = actual, columns = predicted):\n";

      for (ulong c = 0; c < cm.numClasses; c++) {
        std::cout << "  Class " << c << ": TP=" << cm.truePositive[c] << " FP=" << cm.falsePositive[c]
                  << " FN=" << cm.falseNegative[c] << " TN=" << cm.trueNegative[c] << std::fixed << std::setprecision(4)
                  << " | P=" << cm.precision[c] << " R=" << cm.recall[c] << " F1=" << cm.f1Score[c]
                  << " support=" << cm.support[c] << "\n";
      }

      std::cout.unsetf(std::ios_base::floatfield);
      std::cout << std::fixed << std::setprecision(4);
      std::cout << "  Macro:     P=" << cm.macroPrecision << " R=" << cm.macroRecall << " F1=" << cm.macroF1 << "\n";
      std::cout << "  Micro:     P=" << cm.microPrecision << " R=" << cm.microRecall << " F1=" << cm.microF1 << "\n";
      std::cout << "  Weighted:  P=" << cm.weightedPrecision << " R=" << cm.weightedRecall << " F1=" << cm.weightedF1
                << "\n";
      std::cout.unsetf(std::ios_base::floatfield);
    }
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
  QString outputPath = NN_CLI::RunnerUtils::resolvePredictOutputPath(this->parser, this->ioConfig);

  ulong displayProgressReports = (this->logLevel > LogLevel::QUIET) ? this->ioConfig.progressReports : 0;
  std::vector<ANN::Input<float>> inputs =
    ANNLoader::loadInputs(inputPath.toStdString(), this->ioConfig, displayProgressReports);

  if (this->logLevel > LogLevel::QUIET)
    PredictSummary::printANN(this->coreConfig, inputs.size(), inputPath.toStdString(), outputPath.toStdString());

  auto batchStart = std::chrono::system_clock::now();
  std::string startTimeStr = Common::Utils::formatISO8601();

  this->setupPredictProgressCallback(inputs.size());

  // The streaming predict API takes a provider that yields one batch at a
  // time. The inputs are already loaded into `inputs`, so the provider returns
  // a non-owning view over each batch slice (no per-input copy).
  auto sliceProvider = [&inputs](ulong batchSize, ulong batchIndex) {
    ulong start = batchIndex * batchSize;
    ulong end = std::min(start + batchSize, static_cast<ulong>(inputs.size()));

    if (start >= end)
      return ANN::InputsView<float>{};
    return ANN::InputsView<float>(inputs.data() + start, end - start);
  };

  Common::PredictResults<float> results = this->core->predict(inputs.size(), sliceProvider);

  auto batchEnd = std::chrono::system_clock::now();
  std::string endTimeStr = Common::Utils::formatISO8601();
  std::chrono::duration<double> batchElapsed = batchEnd - batchStart;
  double batchDurationSeconds = batchElapsed.count();
  std::string batchDurationFormatted = Common::Utils::formatDuration(batchDurationSeconds);

  emit this->predictFinished(results, inputs.size(), batchDurationSeconds, batchDurationFormatted,
                             outputPath.toStdString());

  return NN_CLI::RunnerUtils::writePredictOutput(results, outputPath, this->ioConfig, this->logLevel, startTimeStr,
                                                 endTimeStr, batchDurationSeconds, batchDurationFormatted,
                                                 inputs.size());
}

//===================================================================================================================//
//  Calibration
//===================================================================================================================//

int ANNRunner::calibrate()
{
  //-- CLI-only config (not in CalibrateConfig) ----------------------------
  const std::string& idImagesDir = this->parser.value("id-images").toStdString();
  const std::string oodDir = this->parser.isSet("ood-dir") ? this->parser.value("ood-dir").toStdString()
                                                           : (fs::current_path() / "extern-datasets" / "ood").string();
  QString configDir = QFileInfo(this->configPath).absoluteDir().absolutePath();
  QString outputDir = NN_CLI::RunnerUtils::resolveOutputDir(this->parser, configDir);
  const std::string outputPath = QDir(outputDir).filePath("threshold.json").toStdString();
  const ulong progressReports = this->ioConfig.progressReports;

  //-- Validate ID images directory ------------------------------------------
  if (!fs::exists(idImagesDir) || !fs::is_directory(idImagesDir)) {
    std::string errMsg = "Error: --id-images " + idImagesDir + " does not exist or is not a directory.";
    std::cerr << errMsg << "\n";
    emit this->logMessage(errMsg, true);
    return 1;
  }

  //-- Fetch OOD if needed ---------------------------------------------------
  if (this->coreConfig.calibrateConfig.fetchIfMissing && !NN_CLI::dirHasImages(oodDir)) {
    if (this->logLevel > LogLevel::QUIET) {
      std::string msg = "OOD dir is empty \u2014 fetching DTD + Places365 + synthetic.\n";
      std::cout << msg;
      emit this->logMessage(msg, false);
    }

    NN_CLI::ensureOODDataset(oodDir, this->logLevel,
                             [this](const std::string& m, bool err) { emit this->logMessage(m, err); });
  } else if (!NN_CLI::dirHasImages(oodDir)) {
    std::string errMsg = "Error: --ood-dir " + oodDir + " has no images and --no-fetch was set.";
    std::cerr << errMsg << "\n";
    emit this->logMessage(errMsg, true);
    return 1;
  }

  //-- Gather + sample -------------------------------------------------------
  std::vector<std::string> idAll = NN_CLI::gatherImages(idImagesDir);
  std::vector<std::string> oodAll = NN_CLI::gatherImages(oodDir);

  if (idAll.empty()) {
    std::string errMsg = "Error: no images found under --id-images " + idImagesDir;
    std::cerr << errMsg << "\n";
    emit this->logMessage(errMsg, true);
    return 1;
  }

  if (oodAll.empty()) {
    std::string errMsg = "Error: no images found under --ood-dir " + oodDir;
    std::cerr << errMsg << "\n";
    emit this->logMessage(errMsg, true);
    return 1;
  }

  std::vector<std::string> idSample = NN_CLI::sampleImages(idAll, this->coreConfig.calibrateConfig.idSampleCount, 42);
  std::vector<std::string> oodSample =
    NN_CLI::sampleImages(oodAll, this->coreConfig.calibrateConfig.oodSampleCount, 42);

  if (this->logLevel > LogLevel::QUIET) {
    std::string msg = "Sampled " + std::to_string(idSample.size()) + " ID images (of " + std::to_string(idAll.size()) +
                      " available)\n"
                      "Sampled " +
                      std::to_string(oodSample.size()) + " OOD images (of " + std::to_string(oodAll.size()) +
                      " available)\n\n";
    std::cout << msg;
    emit this->logMessage(msg, false);
  }

  //-- Predict + free-energy -------------------------------------------------
  auto t0 = std::chrono::system_clock::now();

  int targetC = static_cast<int>(this->ioConfig.inputC);
  int targetH = static_cast<int>(this->ioConfig.inputH);
  int targetW = static_cast<int>(this->ioConfig.inputW);

  auto wrapFn = [](std::vector<float>&& flat) {
    return std::move(flat);
  };

  std::vector<std::vector<float>> idLogits = NN_CLI::runPredictImpl<ANN::Inputs<float>>(
    *this->core, idSample, "Predicting ID", targetC, targetH, targetW, progressReports, this->logLevel, wrapFn);
  std::vector<std::vector<float>> oodLogits = NN_CLI::runPredictImpl<ANN::Inputs<float>>(
    *this->core, oodSample, "Predicting OOD", targetC, targetH, targetW, progressReports, this->logLevel, wrapFn);

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
    emit this->logMessage(doneMsg, false);
  }

  std::string summary = "Calibration completed | ID: " + std::to_string(idEnergies.size()) +
                        " | OOD: " + std::to_string(oodEnergies.size()) + " | Output: " + outputPath;
  emit this->trainFinished(true, summary);

  return 0;
}

//===================================================================================================================//
//  Sample loading
//===================================================================================================================//

std::pair<ANN::Samples<float>, bool> ANNRunner::loadSamplesFromOptions(const std::string& modeName,
                                                                       QString& inputFilePath)
{
  return NN_CLI::RunnerUtils::loadSamplesFromOptionsCommon<ANN::Samples<float>>(
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

void ANNRunner::setupTrainCallback(const QString& inputFilePath, std::shared_ptr<ANN::Core<float>> validationCore,
                                   std::shared_ptr<Common::TrainMonitor<float>> trainMonitor,
                                   const DataLoader<ANN::Sample<float>>* validationDataLoader,
                                   const std::vector<ulong>* validationIndices)
{
  this->lastEpochLoss = 0.0f;

  ulong batchSize = this->coreConfig.trainConfig.batchSize;
  int totalEpochs = static_cast<int>(this->coreConfig.trainConfig.numEpochs);

  QString outputDir =
    NN_CLI::RunnerUtils::resolveOutputDir(this->parser, QFileInfo(inputFilePath).absoluteDir().filePath("output"));

  std::shared_ptr<ANN::SampleProvider<float>> validationProviderPtr;

  if (validationDataLoader && validationIndices && !validationIndices->empty()) {
    auto provider = validationDataLoader->makeSampleProvider({}, 0.0f, SampleLoadType::Validation, *validationIndices);
    validationProviderPtr = std::make_shared<ANN::SampleProvider<float>>(std::move(provider));
  }

  // Live progress: fires per batch from Core worker threads (DirectConnection
  // = synchronous on the emitting thread). Only drives the progress display —
  // every epoch-boundary task lives in the epoch-completed connection below.
  QObject::connect(
    &this->core->getCoreSignals(), &ANN::CoreSignals::trainProgress, this,
    [this, batchSize](ulong currentEpoch, ulong totalEpochs, ulong currentSample, ulong totalSamples, double epochLoss,
                      double sampleLoss, bool isNewBest, bool stoppedEarly, int gpuIndex, int totalGPUs) {
      Common::TrainProgressEvent<float> progress;
      progress.currentEpoch = currentEpoch;
      progress.totalEpochs = totalEpochs;
      progress.currentSample = currentSample;
      progress.totalSamples = totalSamples;
      progress.epochLoss = static_cast<float>(epochLoss);
      progress.sampleLoss = static_cast<float>(sampleLoss);
      progress.isNewBest = isNewBest;
      progress.stoppedEarly = stoppedEarly;
      progress.gpuIndex = gpuIndex;
      progress.totalGPUs = totalGPUs;
      this->handleTrainProgress(progress, batchSize);
    },

    Qt::DirectConnection);

  // Epoch-completed: fires once per epoch (after the epoch's record is
  // recorded) with the 0-based epoch index. DirectConnection = synchronous on
  // the Core worker thread, so validation/checkpointing/LR-scheduling complete
  // before the Core starts the next epoch.
  QObject::connect(
    &this->core->getCoreSignals(), &ANN::CoreSignals::epochCompleted, this,
    [this, outputDir, validationCore, trainMonitor, validationProviderPtr, validationIndices, totalEpochs](
      ulong epoch, ulong /*signalTotalEpochs*/, double /*signalEpochLoss*/, bool isNewBest, bool stoppedEarly) {
      QMutexLocker<QMutex> lock(&this->callbackMutex);

      // --- Checkpointing (every saveModelInterval completed epochs) ---
      // epoch + 1 is the count of completed epochs; checkpoint filenames stay
      // 1-based for human-facing numbering.
      if (this->ioConfig.saveModelInterval > 0 && (epoch + 1) % this->ioConfig.saveModelInterval == 0) {
        std::string checkpointPath = ModelSerializer::generateCheckpointPath(outputDir, epoch + 1, this->lastEpochLoss);
        ModelSerializer::saveANNModelToPackage(checkpointPath, *this->core, this->coreConfig, this->ioConfig,
                                               this->augConfig, this->buildValidationMetadata());
      }

      // --- Validation ---
      bool isBest = false;
      bool monitorShouldStop = false;
      float valLoss = 0.0f;
      bool hasValLoss = false;
      Common::ConfusionMatrix<float> valConfusionMatrix;

      if (this->validationState.enabled && validationCore && validationProviderPtr && validationIndices &&
          epoch % this->validationState.checkInterval == 0) {
        ulong validationTotal = validationIndices->size();

        validationCore->setParameters(this->core->getParameters());

        // Live "Validating" bar: route validation progress through the observer
        // instead of printing to stdout (which would corrupt the ncurses TUI).
        QObject::connect(
          &validationCore->getCoreSignals(), &ANN::CoreSignals::predictProgress, this,
          [this, validationTotal](ulong current, ulong) { emit this->validationProgress(current, validationTotal); },

          Qt::DirectConnection);

        auto validationResult = validationCore->test(validationTotal, *validationProviderPtr);

        this->validationState.lastValLoss = validationResult.averageLoss;
        valLoss = validationResult.averageLoss;
        hasValLoss = true;
        valConfusionMatrix = validationResult.confusionMatrix;

        if (validationResult.averageLoss < this->validationState.bestValidationLoss) {
          this->validationState.bestValidationLoss = validationResult.averageLoss;
          this->validationState.bestValEpoch = epoch;
        }

        if (trainMonitor) {
          monitorShouldStop =
            trainMonitor->checkEpoch(epoch, this->lastEpochLoss, std::optional<float>(validationResult.averageLoss));
          isBest = trainMonitor->isNewBest();
        }
      }

      bool isBestEpoch = (isBest || isNewBest);

      // --- Write the validation results into this epoch's history record ---
      // The core's internal monitor is disabled (NN-CLI monitors externally), so
      // it recorded isBest=false / hasValLoss=false. The just-completed epoch is
      // epochHistory.back() (the core appended it immediately before this call).
      // This MUST happen before the best-model save below, otherwise the saved
      // best model captures this epoch's record with placeholder defaults
      // (hasValLoss=false / isBest=false) even though validation just ran.
      auto& epochHistory = this->core->getTrainMetadata().epochHistory;
      float epochLearningRate = 0.0f;

      if (!epochHistory.empty()) {
        auto& lastRecord = epochHistory.back();
        lastRecord.isBest = isBestEpoch;
        lastRecord.hasValLoss = hasValLoss;
        lastRecord.valLoss = valLoss;
        lastRecord.valConfusionMatrix = valConfusionMatrix;
        lastRecord.hasValConfusionMatrix = hasValLoss;
        epochLearningRate = lastRecord.learningRate;
      }

      // --- Best model save ---
      if (isBestEpoch) {
        std::string bestPath = ModelSerializer::generateBestModelPath(outputDir);
        ModelSerializer::saveANNModelToPackage(bestPath, *this->core, this->coreConfig, this->ioConfig, this->augConfig,
                                               this->buildValidationMetadata());
      }

      // --- LR scheduler step (publishes the new LR for the next epoch) ---
      this->applyLearningRateScheduler(epoch, totalEpochs, hasValLoss, valLoss);

      // --- Observer notification — epoch completed ---
      std::string epochSummary = "Epoch " + std::to_string(epoch + 1) + "/" + std::to_string(totalEpochs) +
                                 " | Loss: " + std::to_string(this->lastEpochLoss);

      if (hasValLoss)
        epochSummary += " | ValLoss: " + std::to_string(valLoss);

      if (isBestEpoch)
        epochSummary += " | Best*";

      emit this->epochCompleted(static_cast<int>(epoch), totalEpochs, this->lastEpochLoss, hasValLoss, valLoss,
                                epochLearningRate, epochSummary);

      // --- Monitor stop requests ---
      if (monitorShouldStop) {
        std::string stopMsg = "[Monitor] Training stopped: " + trainMonitor->getStopReason();

        emit this->logMessage(stopMsg, false);
        this->core->requestStop();
      }

      if (stoppedEarly) {
        std::string stopMsg = "[Monitor] Training stopped: " + this->core->getTrainMetadata().stopReason;

        emit this->logMessage(stopMsg, false);
        this->core->requestStop();
      }
    },

    Qt::DirectConnection);
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
