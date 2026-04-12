#include "NN-CLI_ANNRunner.hpp"

#include "NN-CLI_ANNLoader.hpp"
#include "NN-CLI_DataSplitter.hpp"
#include "NN-CLI_ImageLoader.hpp"
#include "NN-CLI_Loader.hpp"
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
  // Reject conflicting input formats
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
    auto [samples, success] = this->loadSamplesFromOptions("training", inputFilePath);

    if (!success)
      return 1;
    dataLoader.loadFromMemory(std::move(samples), inputC, inputH, inputW);
  }

  dataLoader.planAugmentation(this->augConfig.augmentationFactor, this->augConfig.balanceAugmentation);

  // Auto-compute class weights
  if (this->augConfig.autoClassWeights && this->coreConfig.costFunctionConfig.weights.empty()) {
    auto allOutputs = dataLoader.getAllOutputs();
    std::vector<float> weights = computeClassWeightsFromOutputs(allOutputs);
    this->coreConfig.costFunctionConfig.type = ANN::CostFunctionType::WEIGHTED_SQUARED_DIFFERENCE;
    this->coreConfig.costFunctionConfig.weights = weights;
    this->core = ANN::Core<float>::makeCore(this->coreConfig);

    if (this->logLevel >= LogLevel::INFO) {
      std::cout << "Auto class weights: [";

      for (ulong i = 0; i < weights.size(); i++) {
        if (i > 0)
          std::cout << ", ";
        std::cout << std::fixed << std::setprecision(4) << weights[i];
      }

      std::cout << "]\n";
    }
  }

  // Validation split
  const auto& validationConfig = this->augConfig.validationDataset;
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
  } else {
    this->validationState.enabled = false;
  }

  // Print training summary
  ulong trainSamples = validationConfig.enabled ? split.trainIndices.size() : dataLoader.numSamples();
  ulong validationSamples = validationConfig.enabled ? split.validationIndices.size() : 0;

  if (this->logLevel > LogLevel::QUIET)
    TrainingSummary::printANN(this->coreConfig, this->augConfig, trainSamples, validationSamples, validationRatio,
                              validationAuto);

  // Create a separate core for validation to avoid re-entering the training core's GPU buffers
  std::shared_ptr<ANN::Core<float>> validationCore;

  if (validationConfig.enabled) {
    ANN::CoreConfig<float> validationCoreConfig = this->coreConfig;
    validationCoreConfig.modeType = ANN::ModeType::TEST;
    validationCore = std::shared_ptr<ANN::Core<float>>(ANN::Core<float>::makeCore(validationCoreConfig).release());
  }

  this->setupTrainingCallback(inputFilePath, validationCore, validationConfig.enabled ? &dataLoader : nullptr,
                              validationConfig.enabled ? &split.validationIndices : nullptr);

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

  std::vector<ANN::Output<float>> outputs = this->core->predict(inputs);

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

    for (size_t i = 0; i < outputs.size(); ++i) {
      QString imgName = QString::number(i) + ".png";
      std::string imgPath = outDir.filePath(imgName).toStdString();
      ImageLoader::saveImage(imgPath, outputs[i], static_cast<int>(this->ioConfig.outputC),
                             static_cast<int>(this->ioConfig.outputH), static_cast<int>(this->ioConfig.outputW));
    }

    if (this->logLevel > LogLevel::QUIET) {
      std::cout << "Predict images saved to: " << outputPath.toStdString() << "\n";
      std::cout << "  Images: " << outputs.size() << "\n";
      std::cout << "  Shape: " << this->ioConfig.outputC << "x" << this->ioConfig.outputH << "x"
                << this->ioConfig.outputW << "\n";
      std::cout << "  Duration: " << batchDurationFormatted << "\n";
    }

    return 0;
  }

  // Standard vector output: save as JSON
  nlohmann::ordered_json resultJson;
  nlohmann::ordered_json predictMetadataJson;
  predictMetadataJson["startTime"] = startTimeStr;
  predictMetadataJson["endTime"] = endTimeStr;
  predictMetadataJson["durationSeconds"] = batchDurationSeconds;
  predictMetadataJson["durationFormatted"] = batchDurationFormatted;
  predictMetadataJson["numInputs"] = inputs.size();
  resultJson["predictMetadata"] = predictMetadataJson;
  resultJson["outputs"] = outputs;

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
                                      const DataLoader<ANN::Sample<float>>* validationDataLoader,
                                      const std::vector<ulong>* validationIndices)
{
  static ulong lastCallbackEpoch = 0;
  static float lastEpochLoss = 0.0f;
  lastCallbackEpoch = 0;
  lastEpochLoss = 0.0f;

  static ProgressBar progressBar(this->ioConfig.progressReports);

  std::shared_ptr<ANN::SampleProvider<float>> validationProviderPtr;

  if (validationDataLoader && validationIndices && !validationIndices->empty()) {
    auto provider = validationDataLoader->makeSampleProvider(*validationIndices, {}, 0.0f);
    validationProviderPtr = std::make_shared<ANN::SampleProvider<float>>(std::move(provider));
  }

  this->core->setTrainingCallback([this, inputFilePath, validationCore, validationProviderPtr,
                                   validationIndices](const ANN::TrainingProgress<float>& progress) {
    if (this->logLevel > LogLevel::QUIET) {
      ProgressInfo info{progress.currentEpoch, progress.totalEpochs, progress.currentSample, progress.totalSamples,
                        progress.epochLoss,    progress.sampleLoss,  progress.gpuIndex,      progress.totalGPUs};
      progressBar.update(info);
    }

    if (this->ioConfig.saveModelInterval > 0 && progress.currentEpoch > lastCallbackEpoch) {
      if (lastCallbackEpoch > 0 && lastCallbackEpoch % this->ioConfig.saveModelInterval == 0) {
        std::string checkpointPath =
          ModelSerializer::generateCheckpointPath(inputFilePath, lastCallbackEpoch, lastEpochLoss);
        ModelSerializer::saveANNModel(checkpointPath, *this->core, this->coreConfig, this->ioConfig, this->augConfig,
                                      this->buildValidationMetadata());

        if (this->logLevel > LogLevel::QUIET)
          std::cout << "\nCheckpoint saved to: " << checkpointPath << "\n";
      }

      // Run validation at check intervals using separate core (skip epoch 0 — no training yet)
      if (lastCallbackEpoch > 0 && this->validationState.enabled && validationCore && validationProviderPtr &&
          validationIndices && lastCallbackEpoch % this->validationState.checkInterval == 0) {
        validationCore->setParameters(this->core->getParameters());
        auto validationResult = validationCore->test(validationIndices->size(), *validationProviderPtr);
        this->validationState.lastValLoss = validationResult.averageLoss;

        if (validationResult.averageLoss < this->validationState.bestValLoss) {
          this->validationState.bestValLoss = validationResult.averageLoss;
          this->validationState.bestValEpoch = lastCallbackEpoch;
        }

        if (this->logLevel > LogLevel::QUIET) {
          std::cout << " - Validation Loss: " << std::fixed << std::setprecision(6) << validationResult.averageLoss;
          std::cout.unsetf(std::ios_base::floatfield);
        }
      }

      lastCallbackEpoch = progress.currentEpoch;
    }

    if (progress.epochLoss > 0)
      lastEpochLoss = progress.epochLoss;
  });
}

//===================================================================================================================//

int ANNRunner::finishTraining(const QString& inputFilePath)
{
  if (this->logLevel > LogLevel::QUIET)
    std::cout << "\nTraining completed.\n";

  const auto& trainingConfig = this->core->getTrainingConfig();
  const auto& trainingMetadata = this->core->getTrainingMetadata();

  std::string outputPathStr;

  if (this->parser.isSet("output")) {
    outputPathStr = this->parser.value("output").toStdString();
  } else {
    outputPathStr = ModelSerializer::generateDefaultOutputPath(inputFilePath, trainingConfig.numEpochs,
                                                               trainingMetadata.numSamples, trainingMetadata.finalLoss);
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
