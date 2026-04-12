#include "NN-CLI_CNNRunner.hpp"

#include "NN-CLI_CNNLoader.hpp"
#include "NN-CLI_DataSplitter.hpp"
#include "NN-CLI_ImageLoader.hpp"
#include "NN-CLI_Loader.hpp"
#include "NN-CLI_ProgressBar.hpp"
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
  // Reject conflicting input formats
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
    // Keep the configured cost function type (e.g. crossEntropy) — weights work with all types.
    // Only override to WEIGHTED_SQUARED_DIFFERENCE if the user chose plain squaredDifference.
    if (this->coreConfig.costFunctionConfig.type == CNN::CostFunctionType::SQUARED_DIFFERENCE) {
      this->coreConfig.costFunctionConfig.type = CNN::CostFunctionType::WEIGHTED_SQUARED_DIFFERENCE;
    }

    this->coreConfig.costFunctionConfig.weights = weights;
    this->core = CNN::Core<float>::makeCore(this->coreConfig);

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
  const auto& valConfig = this->augConfig.validationDataset;
  DataSplit split;
  float valRatio = 0.0f;
  bool valAuto = false;

  if (valConfig.enabled) {
    valRatio = valConfig.autoSize ? DataSplitter::computeAutoValSize(dataLoader.numSamples()) : valConfig.size;
    valAuto = valConfig.autoSize;
    auto allOutputs = dataLoader.getAllOutputs();
    split = DataSplitter::stratifiedSplit(allOutputs, valRatio);
    this->valState.enabled = true;
    this->valState.checkInterval = valConfig.checkInterval;
    this->valState.numValSamples = split.valIndices.size();
  } else {
    this->valState.enabled = false;
  }

  // Print training summary
  ulong trainSamples = valConfig.enabled ? split.trainIndices.size() : dataLoader.numSamples();
  ulong valSamples = valConfig.enabled ? split.valIndices.size() : 0;

  if (this->logLevel >= LogLevel::INFO)
    TrainingSummary::printCNN(this->coreConfig, this->augConfig, trainSamples, valSamples, valRatio, valAuto);

  // Create a separate core for validation to avoid re-entering the training core's GPU buffers
  std::shared_ptr<CNN::Core<float>> valCore;

  if (valConfig.enabled) {
    CNN::CoreConfig<float> valCoreConfig = this->coreConfig;
    valCoreConfig.modeType = CNN::ModeType::TEST;
    valCoreConfig.numGPUs = 1; // Single GPU is sufficient for the small validation set
    valCoreConfig.numThreads = 1;
    valCore = std::shared_ptr<CNN::Core<float>>(CNN::Core<float>::makeCore(valCoreConfig).release());
  }

  this->setupTrainingCallback(inputFilePath, valCore, valConfig.enabled ? &dataLoader : nullptr,
                              valConfig.enabled ? &split.valIndices : nullptr);

  if (valConfig.enabled) {
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

  if (this->logLevel >= LogLevel::INFO)
    std::cout << "Running CNN evaluation...\n";

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

  if (this->logLevel >= LogLevel::INFO)
    std::cout << "Loading inputs from: " << inputPath.toStdString() << "\n";

  ulong displayProgressReports = (this->logLevel > LogLevel::QUIET) ? this->ioConfig.progressReports : 0;
  std::vector<CNN::Input<float>> inputs =
    CNNLoader::loadInputs(inputPath.toStdString(), this->coreConfig.inputShape, this->ioConfig, displayProgressReports);

  if (this->logLevel >= LogLevel::INFO) {
    std::cout << "Loaded " << inputs.size() << " input(s), each with " << inputs[0].data.size() << " values\n";
  }

  auto batchStart = std::chrono::system_clock::now();
  std::string startTimeStr = ANN::Utils<float>::formatISO8601();

  if (this->logLevel > LogLevel::QUIET) {
    ulong progressReports = this->ioConfig.progressReports;
    ProgressBar::printLoadingProgress("Predicting", 0, inputs.size(), progressReports);
    this->core->setProgressCallback([progressReports](ulong current, ulong total) {
      ProgressBar::printLoadingProgress("Predicting", current, total, progressReports);
    });
  }

  std::vector<CNN::Output<float>> outputs = this->core->predict(inputs);

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
  return {this->valState.enabled, this->valState.numValSamples, this->valState.lastValLoss, this->valState.bestValLoss,
          this->valState.bestValEpoch};
}

//===================================================================================================================//

void CNNRunner::setupTrainingCallback(const QString& inputFilePath, std::shared_ptr<CNN::Core<float>> valCore,
                                      const DataLoader<CNN::Sample<float>>* valDataLoader,
                                      const std::vector<ulong>* valIndices)
{
  static ulong lastCallbackEpoch = 0;
  static float lastEpochLoss = 0.0f;
  lastCallbackEpoch = 0;
  lastEpochLoss = 0.0f;

  static ProgressBar progressBar(this->ioConfig.progressReports);

  std::shared_ptr<CNN::SampleProvider<float>> valProviderPtr;

  if (valDataLoader && valIndices && !valIndices->empty()) {
    auto provider = valDataLoader->makeSampleProvider(*valIndices, {}, 0.0f);
    valProviderPtr = std::make_shared<CNN::SampleProvider<float>>(std::move(provider));
  }

  this->core->setTrainingCallback(
    [this, inputFilePath, valCore, valProviderPtr, valIndices](const CNN::TrainingProgress<float>& progress) {
      if (this->logLevel > LogLevel::QUIET) {
        ProgressInfo info{progress.currentEpoch, progress.totalEpochs, progress.currentSample, progress.totalSamples,
                          progress.epochLoss,    progress.sampleLoss,  progress.gpuIndex,      progress.totalGPUs};
        progressBar.update(info);
      }

      if (this->ioConfig.saveModelInterval > 0 && progress.currentEpoch > lastCallbackEpoch) {
        if (lastCallbackEpoch > 0 && lastCallbackEpoch % this->ioConfig.saveModelInterval == 0) {
          std::string checkpointPath =
            ModelSerializer::generateCheckpointPath(inputFilePath, lastCallbackEpoch, lastEpochLoss);
          ModelSerializer::saveCNNModel(checkpointPath, *this->core, this->coreConfig, this->ioConfig, this->augConfig,
                                        this->buildValidationMetadata());

          if (this->logLevel > LogLevel::QUIET)
            std::cout << "\nCheckpoint saved to: " << checkpointPath << "\n";
        }

        // Run validation at check intervals using separate core
        if (this->valState.enabled && valCore && valProviderPtr && valIndices &&
            lastCallbackEpoch % this->valState.checkInterval == 0) {
          valCore->setParameters(this->core->getParameters());
          auto valResult = valCore->test(valIndices->size(), *valProviderPtr);
          this->valState.lastValLoss = valResult.averageLoss;

          if (valResult.averageLoss < this->valState.bestValLoss) {
            this->valState.bestValLoss = valResult.averageLoss;
            this->valState.bestValEpoch = lastCallbackEpoch;
          }

          if (this->logLevel > LogLevel::QUIET) {
            std::cout << " - Validation Loss: " << std::fixed << std::setprecision(6) << valResult.averageLoss;
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

int CNNRunner::finishTraining(const QString& inputFilePath)
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
