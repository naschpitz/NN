#include "NN-CLI_Runner.hpp"

#include "NN-CLI_ImageLoader.hpp"
#include "NN-CLI_Loader.hpp"
#include "NN-CLI_ProgressBar.hpp"
#include "NN-CLI_Utils.hpp"

#include <QDir>
#include <QFile>
#include <QFileInfo>

#include <ANN_Utils.hpp>

#include <json.hpp>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>

using namespace NN_CLI;

//===================================================================================================================//

Runner::Runner(const QCommandLineParser& parser, LogLevel logLevel)
    : parser(parser), logLevel(logLevel) {
  QString configPath = this->parser.value("config");

  // Detect network type from config file
  this->networkType = Loader::detectNetworkType(configPath.toStdString());

  // Build optional mode/device overrides as strings
  std::optional<std::string> modeOverride;
  if (this->parser.isSet("mode")) {
    modeOverride = this->parser.value("mode").toLower().toStdString();
  }

  std::optional<std::string> deviceOverride;
  if (this->parser.isSet("device")) {
    deviceOverride = this->parser.value("device").toLower().toStdString();
  }

  // Extract shuffle-samples override
  std::optional<bool> shuffleSamplesOverride;
  if (this->parser.isSet("shuffle-samples")) {
    shuffleSamplesOverride = (this->parser.value("shuffle-samples").toLower() == "true");
  }

  // Load I/O config (inputType, outputType, shapes) with optional CLI overrides
  std::optional<std::string> inputTypeOverride;
  if (this->parser.isSet("input-type")) {
    inputTypeOverride = this->parser.value("input-type").toLower().toStdString();
  }

  std::optional<std::string> outputTypeOverride;
  if (this->parser.isSet("output-type")) {
    outputTypeOverride = this->parser.value("output-type").toLower().toStdString();
  }

  this->ioConfig = Loader::loadIOConfig(configPath.toStdString(), inputTypeOverride, outputTypeOverride);

  // Display info (verbose level >= 1)
  std::string networkTypeStr = (this->networkType == NetworkType::CNN) ? "CNN" : "ANN";
  std::string modeDisplay = modeOverride.has_value() ? (modeOverride.value() + " (CLI)") : "from config file";
  std::string deviceDisplay = deviceOverride.has_value() ? (deviceOverride.value() + " (CLI)") : "from config file";

  if (this->logLevel >= LogLevel::INFO) {
    std::cout << "Network type: " << networkTypeStr << "\n";
    std::cout << "Loading configuration from: " << configPath.toStdString() << "\n";
    std::cout << "Mode: " << modeDisplay << ", Device: " << deviceDisplay << "\n";
    std::cout << "Input type: " << dataTypeToString(this->ioConfig.inputType)
              << ", Output type: " << dataTypeToString(this->ioConfig.outputType) << "\n";
  }

  // Load NN-CLI-level settings from config root
  this->progressReports = Loader::loadProgressReports(configPath.toStdString());
  this->saveModelInterval = Loader::loadSaveModelInterval(configPath.toStdString());

  // Load data augmentation config
  auto augConfig = Loader::loadAugmentationConfig(configPath.toStdString());
  this->augmentationFactor = augConfig.augmentationFactor;
  this->balanceAugmentation = augConfig.balanceAugmentation;
  this->autoClassWeights = augConfig.autoClassWeights;
  this->augmentationProbability = augConfig.augmentationProbability;
  this->augTransforms = augConfig.transforms;

  if (this->logLevel >= LogLevel::INFO && this->saveModelInterval > 0) {
    std::cout << "Save model interval: every " << this->saveModelInterval << " epoch(s)\n";
  }

  if (this->networkType == NetworkType::ANN) {
    // Convert string overrides to ANN enum overrides
    std::optional<ANN::ModeType> annModeOverride;
    if (modeOverride.has_value()) annModeOverride = ANN::Mode::nameToType(modeOverride.value());

    std::optional<ANN::DeviceType> annDeviceOverride;
    if (deviceOverride.has_value()) annDeviceOverride = ANN::Device::nameToType(deviceOverride.value());

    this->annCoreConfig = Loader::loadANNConfig(configPath.toStdString(), annModeOverride, annDeviceOverride);
    this->annCoreConfig.logLevel = static_cast<ANN::LogLevel>(this->logLevel);
    if (shuffleSamplesOverride.has_value()) this->annCoreConfig.trainingConfig.shuffleSamples = shuffleSamplesOverride.value();
    this->mode = ANN::Mode::typeToName(this->annCoreConfig.modeType);
    this->annCore = ANN::Core<float>::makeCore(this->annCoreConfig);
  } else {
    this->cnnCoreConfig = Loader::loadCNNConfig(configPath.toStdString(), modeOverride, deviceOverride);
    this->cnnCoreConfig.logLevel = static_cast<CNN::LogLevel>(this->logLevel);
    if (shuffleSamplesOverride.has_value()) this->cnnCoreConfig.trainingConfig.shuffleSamples = shuffleSamplesOverride.value();
    this->mode = CNN::Mode::typeToName(this->cnnCoreConfig.modeType);
    this->cnnCore = CNN::Core<float>::makeCore(this->cnnCoreConfig);
  }
}

//===================================================================================================================//

int Runner::run() {
  if (this->networkType == NetworkType::ANN) {
    if (this->mode == "train")   return this->runANNTrain();
    if (this->mode == "test")    return this->runANNTest();
    return this->runANNPredict();
  } else {
    if (this->mode == "train")   return this->runCNNTrain();
    if (this->mode == "test")    return this->runCNNTest();
    return this->runCNNPredict();
  }
}

//===================================================================================================================//

int Runner::runANNTrain() {
  QString inputFilePath;
  auto [samples, success] = this->loadANNSamplesFromOptions("training", inputFilePath);
  if (!success) return 1;

  this->augmentSamples(samples);

  // Auto-compute class weights if enabled and no manual weights specified
  if (this->autoClassWeights && this->annCoreConfig.costFunctionConfig.weights.empty()) {
    auto weights = this->computeClassWeights(samples);
    this->annCoreConfig.costFunctionConfig.type = ANN::CostFunctionType::WEIGHTED_SQUARED_DIFFERENCE;
    this->annCoreConfig.costFunctionConfig.weights = weights;
    this->annCore = ANN::Core<float>::makeCore(this->annCoreConfig);
    if (this->logLevel >= LogLevel::INFO) {
      std::cout << "Auto class weights: [";
      for (ulong i = 0; i < weights.size(); i++) {
        if (i > 0) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(4) << weights[i];
      }
      std::cout << "]\n";
    }
  }

  if (this->logLevel >= LogLevel::INFO) std::cout << "Starting ANN training...\n";

  ProgressBar progressBar(this->progressReports);

  ulong lastCallbackEpoch = 0;
  float lastEpochLoss = 0.0f;

  this->annCore->setTrainingCallback([&](const ANN::TrainingProgress<float>& progress) {
    if (this->logLevel > LogLevel::QUIET) {
      ProgressInfo info{progress.currentEpoch, progress.totalEpochs,
                        progress.currentSample, progress.totalSamples,
                        progress.epochLoss, progress.sampleLoss,
                        progress.gpuIndex, progress.totalGPUs};
      progressBar.update(info);
    }

    // Checkpoint saving: detect epoch transition
    if (this->saveModelInterval > 0 && progress.currentEpoch > lastCallbackEpoch) {
      // An epoch boundary was crossed — lastCallbackEpoch is the completed epoch
      if (lastCallbackEpoch > 0 && lastCallbackEpoch % this->saveModelInterval == 0) {
        std::string checkpointPath = generateCheckpointPath(inputFilePath, lastCallbackEpoch, lastEpochLoss);
        saveANNModel(*this->annCore, checkpointPath, this->ioConfig, this->progressReports, this->saveModelInterval);
        if (this->logLevel > LogLevel::QUIET) std::cout << "\nCheckpoint saved to: " << checkpointPath << "\n";
      }
      lastCallbackEpoch = progress.currentEpoch;
    }

    // Track epoch loss for checkpoint filename
    if (progress.epochLoss > 0) lastEpochLoss = progress.epochLoss;
  });

  this->annCore->train(samples);
  if (this->logLevel > LogLevel::QUIET) std::cout << "\nTraining completed.\n";

  const auto& trainingConfig = this->annCore->getTrainingConfig();
  const auto& trainingMetadata = this->annCore->getTrainingMetadata();

  std::string outputPathStr;
  if (this->parser.isSet("output")) {
    outputPathStr = this->parser.value("output").toStdString();
  } else {
    outputPathStr = generateDefaultOutputPath(
      inputFilePath, trainingConfig.numEpochs,
      trainingMetadata.numSamples, trainingMetadata.finalLoss);
  }

  saveANNModel(*this->annCore, outputPathStr, this->ioConfig, this->progressReports, this->saveModelInterval);
  if (this->logLevel > LogLevel::QUIET) std::cout << "Model saved to: " << outputPathStr << "\n";

  return 0;
}

//===================================================================================================================//

int Runner::runANNTest() {
  QString inputFilePath;
  auto [samples, success] = this->loadANNSamplesFromOptions("test", inputFilePath);
  if (!success) return 1;

  if (this->logLevel >= LogLevel::INFO) std::cout << "Running ANN evaluation...\n";

  ANN::TestResult<float> result = this->annCore->test(samples);

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

int Runner::runANNPredict() {
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
    if (!outputDir.exists()) inputDir.mkdir("output");

    if (this->ioConfig.outputType == DataType::IMAGE) {
      outputPath = outputDir.filePath("predict_" + inputInfo.completeBaseName());
    } else {
      outputPath = outputDir.filePath("predict_" + inputInfo.completeBaseName() + ".json");
    }
  }

  if (this->logLevel >= LogLevel::INFO) std::cout << "Loading inputs from: " << inputPath.toStdString() << "\n";

  ulong displayProgressReports = (this->logLevel > LogLevel::QUIET) ? this->progressReports : 0;
  std::vector<ANN::Input<float>> inputs = Loader::loadANNInputs(inputPath.toStdString(), this->ioConfig, displayProgressReports);

  if (this->logLevel >= LogLevel::INFO) {
    std::cout << "Loaded " << inputs.size() << " input(s), each with " << inputs[0].size() << " values\n";
  }

  // Track overall batch timing
  auto batchStart = std::chrono::system_clock::now();
  std::string startTimeStr = ANN::Utils<float>::formatISO8601();

  std::vector<ANN::Output<float>> outputs;
  outputs.reserve(inputs.size());

  for (size_t i = 0; i < inputs.size(); ++i) {
    ANN::Output<float> output = this->annCore->predict(inputs[i]);
    outputs.push_back(std::move(output));
    if (this->logLevel >= LogLevel::INFO && inputs.size() > 1) {
      std::cout << "  Predicted input " << (i + 1) << "/" << inputs.size() << "\n";
    }
  }

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

    // outputPath is a directory for batch image output
    QDir outDir(outputPath);
    if (!outDir.exists()) QDir().mkpath(outputPath);

    for (size_t i = 0; i < outputs.size(); ++i) {
      QString imgName = QString::number(i) + ".png";
      std::string imgPath = outDir.filePath(imgName).toStdString();
      ImageLoader::saveImage(imgPath, outputs[i],
          static_cast<int>(this->ioConfig.outputC),
          static_cast<int>(this->ioConfig.outputH),
          static_cast<int>(this->ioConfig.outputW));
    }

    if (this->logLevel > LogLevel::QUIET) {
      std::cout << "Predict images saved to: " << outputPath.toStdString() << "\n";
      std::cout << "  Images: " << outputs.size() << "\n";
      std::cout << "  Shape: " << this->ioConfig.outputC << "x" << this->ioConfig.outputH << "x" << this->ioConfig.outputW << "\n";
      std::cout << "  Duration: " << batchDurationFormatted << "\n";
    }
    return 0;
  }

  // Standard vector output: save as JSON with "outputs" array
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

  if (this->logLevel > LogLevel::QUIET) std::cout << "Predict result saved to: " << outputPath.toStdString() << "\n";
  return 0;
}

//===================================================================================================================//
//  CNN mode methods
//===================================================================================================================//

int Runner::runCNNTrain() {
  QString inputFilePath;
  auto [samples, success] = this->loadCNNSamplesFromOptions("training", inputFilePath);
  if (!success) return 1;

  this->augmentSamples(samples);

  // Auto-compute class weights if enabled and no manual weights specified
  if (this->autoClassWeights && this->cnnCoreConfig.costFunctionConfig.weights.empty()) {
    auto weights = this->computeClassWeights(samples);
    this->cnnCoreConfig.costFunctionConfig.type = CNN::CostFunctionType::WEIGHTED_SQUARED_DIFFERENCE;
    this->cnnCoreConfig.costFunctionConfig.weights = weights;
    this->cnnCore = CNN::Core<float>::makeCore(this->cnnCoreConfig);
    if (this->logLevel >= LogLevel::INFO) {
      std::cout << "Auto class weights: [";
      for (ulong i = 0; i < weights.size(); i++) {
        if (i > 0) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(4) << weights[i];
      }
      std::cout << "]\n";
    }
  }

  if (this->logLevel >= LogLevel::INFO) std::cout << "Starting CNN training...\n";

  ProgressBar progressBar(this->progressReports);

  ulong lastCallbackEpoch = 0;
  float lastEpochLoss = 0.0f;

  this->cnnCore->setTrainingCallback([&](const CNN::TrainingProgress<float>& progress) {
    if (this->logLevel > LogLevel::QUIET) {
      ProgressInfo info{progress.currentEpoch, progress.totalEpochs,
                        progress.currentSample, progress.totalSamples,
                        progress.epochLoss, progress.sampleLoss,
                        progress.gpuIndex, progress.totalGPUs};
      progressBar.update(info);
    }

    // Checkpoint saving: detect epoch transition
    if (this->saveModelInterval > 0 && progress.currentEpoch > lastCallbackEpoch) {
      // An epoch boundary was crossed — lastCallbackEpoch is the completed epoch
      if (lastCallbackEpoch > 0 && lastCallbackEpoch % this->saveModelInterval == 0) {
        std::string checkpointPath = generateCheckpointPath(inputFilePath, lastCallbackEpoch, lastEpochLoss);
        saveCNNModel(*this->cnnCore, checkpointPath, this->ioConfig, this->progressReports, this->saveModelInterval);
        if (this->logLevel > LogLevel::QUIET) std::cout << "\nCheckpoint saved to: " << checkpointPath << "\n";
      }
      lastCallbackEpoch = progress.currentEpoch;
    }

    // Track epoch loss for checkpoint filename
    if (progress.epochLoss > 0) lastEpochLoss = progress.epochLoss;
  });

  this->cnnCore->train(samples);
  if (this->logLevel > LogLevel::QUIET) std::cout << "\nTraining completed.\n";

  const auto& trainingConfig = this->cnnCore->getTrainingConfig();
  const auto& trainingMetadata = this->cnnCore->getTrainingMetadata();

  std::string outputPathStr;
  if (this->parser.isSet("output")) {
    outputPathStr = this->parser.value("output").toStdString();
  } else {
    outputPathStr = generateDefaultOutputPath(
      inputFilePath, trainingConfig.numEpochs,
      trainingMetadata.numSamples, trainingMetadata.finalLoss);
  }

  saveCNNModel(*this->cnnCore, outputPathStr, this->ioConfig, this->progressReports, this->saveModelInterval);
  if (this->logLevel > LogLevel::QUIET) std::cout << "Model saved to: " << outputPathStr << "\n";

  return 0;
}

//===================================================================================================================//

int Runner::runCNNTest() {
  QString inputFilePath;
  auto [samples, success] = this->loadCNNSamplesFromOptions("test", inputFilePath);
  if (!success) return 1;

  if (this->logLevel >= LogLevel::INFO) std::cout << "Running CNN evaluation...\n";

  CNN::TestResult<float> result = this->cnnCore->test(samples);

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

int Runner::runCNNPredict() {
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
    if (!outputDir.exists()) inputDir.mkdir("output");

    if (this->ioConfig.outputType == DataType::IMAGE) {
      outputPath = outputDir.filePath("predict_" + inputInfo.completeBaseName());
    } else {
      outputPath = outputDir.filePath("predict_" + inputInfo.completeBaseName() + ".json");
    }
  }

  if (this->logLevel >= LogLevel::INFO) std::cout << "Loading inputs from: " << inputPath.toStdString() << "\n";

  ulong displayProgressReports = (this->logLevel > LogLevel::QUIET) ? this->progressReports : 0;
  std::vector<CNN::Input<float>> inputs = Loader::loadCNNInputs(
      inputPath.toStdString(), this->cnnCoreConfig.inputShape, this->ioConfig, displayProgressReports);

  if (this->logLevel >= LogLevel::INFO) {
    std::cout << "Loaded " << inputs.size() << " input(s), each with " << inputs[0].data.size() << " values\n";
  }

  // Track overall batch timing
  auto batchStart = std::chrono::system_clock::now();
  std::string startTimeStr = ANN::Utils<float>::formatISO8601();

  std::vector<CNN::Output<float>> outputs;
  outputs.reserve(inputs.size());

  for (size_t i = 0; i < inputs.size(); ++i) {
    CNN::Output<float> output = this->cnnCore->predict(inputs[i]);
    outputs.push_back(std::move(output));
    if (this->logLevel >= LogLevel::INFO && inputs.size() > 1) {
      std::cout << "  Predicted input " << (i + 1) << "/" << inputs.size() << "\n";
    }
  }

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

    // outputPath is a directory for batch image output
    QDir outDir(outputPath);
    if (!outDir.exists()) QDir().mkpath(outputPath);

    for (size_t i = 0; i < outputs.size(); ++i) {
      QString imgName = QString::number(i) + ".png";
      std::string imgPath = outDir.filePath(imgName).toStdString();
      ImageLoader::saveImage(imgPath, outputs[i],
          static_cast<int>(this->ioConfig.outputC),
          static_cast<int>(this->ioConfig.outputH),
          static_cast<int>(this->ioConfig.outputW));
    }

    if (this->logLevel > LogLevel::QUIET) {
      std::cout << "Predict images saved to: " << outputPath.toStdString() << "\n";
      std::cout << "  Images: " << outputs.size() << "\n";
      std::cout << "  Shape: " << this->ioConfig.outputC << "x" << this->ioConfig.outputH << "x" << this->ioConfig.outputW << "\n";
      std::cout << "  Duration: " << batchDurationFormatted << "\n";
    }
    return 0;
  }

  // Standard vector output: save as JSON with "outputs" array
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

  if (this->logLevel > LogLevel::QUIET) std::cout << "Predict result saved to: " << outputPath.toStdString() << "\n";
  return 0;
}

//===================================================================================================================//
//  Sample loading helpers
//===================================================================================================================//

std::pair<ANN::Samples<float>, bool> Runner::loadANNSamplesFromOptions(
    const std::string& modeName,
    QString& inputFilePath) {
  ANN::Samples<float> samples;

  bool hasJsonSamples = this->parser.isSet("samples");
  bool hasIdxData = this->parser.isSet("idx-data");
  bool hasIdxLabels = this->parser.isSet("idx-labels");

  if (hasJsonSamples && hasIdxData) {
    std::cerr << "Error: Cannot use both --samples and --idx-data. Choose one format.\n";
    return {samples, false};
  }

  ulong displayProgressReports = (this->logLevel > LogLevel::QUIET) ? this->progressReports : 0;

  if (hasJsonSamples) {
    QString samplesPath = this->parser.value("samples");
    inputFilePath = samplesPath;
    if (this->logLevel >= LogLevel::INFO) std::cout << "Loading " << modeName << " samples from JSON: " << samplesPath.toStdString() << "\n";
    samples = Loader::loadANNSamples(samplesPath.toStdString(), this->ioConfig, displayProgressReports);
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

  if (this->logLevel >= LogLevel::INFO) std::cout << "Loaded " << samples.size() << " " << modeName << " samples.\n";

  return {samples, true};
}

//===================================================================================================================//

std::pair<CNN::Samples<float>, bool> Runner::loadCNNSamplesFromOptions(
    const std::string& modeName,
    QString& inputFilePath) {
  CNN::Samples<float> samples;

  bool hasJsonSamples = this->parser.isSet("samples");
  bool hasIdxData = this->parser.isSet("idx-data");
  bool hasIdxLabels = this->parser.isSet("idx-labels");

  if (hasJsonSamples && hasIdxData) {
    std::cerr << "Error: Cannot use both --samples and --idx-data. Choose one format.\n";
    return {samples, false};
  }

  const CNN::Shape3D& inputShape = this->cnnCoreConfig.inputShape;

  ulong displayProgressReports = (this->logLevel > LogLevel::QUIET) ? this->progressReports : 0;

  if (hasJsonSamples) {
    QString samplesPath = this->parser.value("samples");
    inputFilePath = samplesPath;
    if (this->logLevel >= LogLevel::INFO) std::cout << "Loading " << modeName << " samples from JSON: " << samplesPath.toStdString() << "\n";
    samples = Loader::loadCNNSamples(samplesPath.toStdString(), inputShape, this->ioConfig, displayProgressReports);
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

    samples = Utils<float>::loadCNNIDX(idxDataPath.toStdString(), idxLabelsPath.toStdString(), inputShape, displayProgressReports);
  } else {
    std::cerr << "Error: " << modeName << " requires either --samples (JSON) or --idx-data and --idx-labels (IDX).\n";
    return {samples, false};
  }

  if (this->logLevel >= LogLevel::INFO) std::cout << "Loaded " << samples.size() << " " << modeName << " samples.\n";

  return {samples, true};
}

//===================================================================================================================//
//  Model saving
//===================================================================================================================//

void Runner::saveANNModel(const ANN::Core<float>& core, const std::string& filePath,
                           const IOConfig& ioConfig, ulong progressReports, ulong saveModelInterval) {
  nlohmann::ordered_json json;

  json["mode"] = ANN::Mode::typeToName(core.getModeType());
  json["device"] = ANN::Device::typeToName(core.getDeviceType());
  json["numThreads"] = core.getNumThreads();
  json["numGPUs"] = core.getNumGPUs();

  // NN-CLI settings
  json["progressReports"] = progressReports;
  json["saveModelInterval"] = saveModelInterval;

  // I/O types (NN-CLI concept, persisted so predict/test can reload them)
  json["inputType"] = dataTypeToString(ioConfig.inputType);
  json["outputType"] = dataTypeToString(ioConfig.outputType);

  if (ioConfig.hasInputShape()) {
    nlohmann::ordered_json isJson;
    isJson["c"] = ioConfig.inputC;
    isJson["h"] = ioConfig.inputH;
    isJson["w"] = ioConfig.inputW;
    json["inputShape"] = isJson;
  }

  if (ioConfig.hasOutputShape()) {
    nlohmann::ordered_json osJson;
    osJson["c"] = ioConfig.outputC;
    osJson["h"] = ioConfig.outputH;
    osJson["w"] = ioConfig.outputW;
    json["outputShape"] = osJson;
  }

  // Layers config
  nlohmann::ordered_json layersArr = nlohmann::ordered_json::array();
  for (const auto& layer : core.getLayersConfig()) {
    nlohmann::ordered_json layerJson;
    layerJson["numNeurons"] = layer.numNeurons;
    layerJson["actvFunc"] = ANN::ActvFunc::typeToName(layer.actvFuncType);
    layersArr.push_back(layerJson);
  }
  json["layersConfig"] = layersArr;

  // Cost function config
  nlohmann::ordered_json cfcJson;
  cfcJson["type"] = ANN::CostFunction::typeToName(core.getCostFunctionConfig().type);
  if (!core.getCostFunctionConfig().weights.empty()) {
    cfcJson["weights"] = core.getCostFunctionConfig().weights;
  }
  json["costFunctionConfig"] = cfcJson;

  // Training config
  nlohmann::ordered_json tcJson;
  tcJson["numEpochs"] = core.getTrainingConfig().numEpochs;
  tcJson["learningRate"] = core.getTrainingConfig().learningRate;
  tcJson["batchSize"] = core.getTrainingConfig().batchSize;
  tcJson["shuffleSamples"] = core.getTrainingConfig().shuffleSamples;
  if (core.getTrainingConfig().dropoutRate > 0.0f)
    tcJson["dropoutRate"] = core.getTrainingConfig().dropoutRate;
  json["trainingConfig"] = tcJson;

  // Training metadata
  const auto& md = core.getTrainingMetadata();
  nlohmann::ordered_json mdJson;
  mdJson["startTime"] = md.startTime;
  mdJson["endTime"] = md.endTime;
  mdJson["durationSeconds"] = md.durationSeconds;
  mdJson["durationFormatted"] = md.durationFormatted;
  mdJson["numSamples"] = md.numSamples;
  mdJson["finalLoss"] = md.finalLoss;
  json["trainingMetadata"] = mdJson;

  // Parameters
  nlohmann::ordered_json paramsJson;
  paramsJson["weights"] = core.getParameters().weights;
  paramsJson["biases"] = core.getParameters().biases;
  json["parameters"] = paramsJson;

  // Write to file
  QFile file(QString::fromStdString(filePath));
  if (!file.open(QIODevice::WriteOnly)) {
    throw std::runtime_error("Failed to open file for writing: " + filePath);
  }
  std::string jsonStr = json.dump(4);
  file.write(jsonStr.c_str());
  file.close();
}

//===================================================================================================================//

void Runner::saveCNNModel(const CNN::Core<float>& core, const std::string& filePath,
                           const IOConfig& ioConfig, ulong progressReports, ulong saveModelInterval) {
  nlohmann::ordered_json json;

  json["mode"] = CNN::Mode::typeToName(core.getModeType());
  json["device"] = CNN::Device::typeToName(core.getDeviceType());
  json["numThreads"] = core.getNumThreads();
  json["numGPUs"] = core.getNumGPUs();

  // NN-CLI settings
  json["progressReports"] = progressReports;
  json["saveModelInterval"] = saveModelInterval;

  // I/O types (NN-CLI concept, persisted so predict/test can reload them)
  json["inputType"] = dataTypeToString(ioConfig.inputType);
  json["outputType"] = dataTypeToString(ioConfig.outputType);

  // Input shape (CNN network shape, always present)
  const auto& shape = core.getInputShape();
  nlohmann::ordered_json shapeJson;
  shapeJson["c"] = shape.c;
  shapeJson["h"] = shape.h;
  shapeJson["w"] = shape.w;
  json["inputShape"] = shapeJson;

  // Output shape (for image output reconstruction)
  if (ioConfig.hasOutputShape()) {
    nlohmann::ordered_json osJson;
    osJson["c"] = ioConfig.outputC;
    osJson["h"] = ioConfig.outputH;
    osJson["w"] = ioConfig.outputW;
    json["outputShape"] = osJson;
  }

  // CNN layers config
  nlohmann::ordered_json cnnLayersArr = nlohmann::ordered_json::array();
  for (const auto& layer : core.getLayersConfig().cnnLayers) {
    nlohmann::ordered_json layerJson;
    switch (layer.type) {
      case CNN::LayerType::CONV: {
        const auto& conv = std::get<CNN::ConvLayerConfig>(layer.config);
        layerJson["type"] = "conv";
        layerJson["numFilters"] = conv.numFilters;
        layerJson["filterH"] = conv.filterH;
        layerJson["filterW"] = conv.filterW;
        layerJson["strideY"] = conv.strideY;
        layerJson["strideX"] = conv.strideX;
        layerJson["slidingStrategy"] = CNN::SlidingStrategy::typeToName(conv.slidingStrategy);
        break;
      }
      case CNN::LayerType::RELU:
        layerJson["type"] = "relu";
        break;
      case CNN::LayerType::POOL: {
        const auto& pool = std::get<CNN::PoolLayerConfig>(layer.config);
        layerJson["type"] = "pool";
        layerJson["poolType"] = CNN::PoolType::typeToName(pool.poolType);
        layerJson["poolH"] = pool.poolH;
        layerJson["poolW"] = pool.poolW;
        layerJson["strideY"] = pool.strideY;
        layerJson["strideX"] = pool.strideX;
        break;
      }
      case CNN::LayerType::FLATTEN:
        layerJson["type"] = "flatten";
        break;
    }
    cnnLayersArr.push_back(layerJson);
  }
  json["convolutionalLayersConfig"] = cnnLayersArr;

  // Dense layers config
  nlohmann::ordered_json denseLayersArr = nlohmann::ordered_json::array();
  for (const auto& layer : core.getLayersConfig().denseLayers) {
    nlohmann::ordered_json layerJson;
    layerJson["numNeurons"] = layer.numNeurons;
    layerJson["actvFunc"] = ANN::ActvFunc::typeToName(layer.actvFuncType);
    denseLayersArr.push_back(layerJson);
  }
  json["denseLayersConfig"] = denseLayersArr;

  // Cost function config
  nlohmann::ordered_json cfcJson;
  cfcJson["type"] = CNN::CostFunction::typeToName(core.getCostFunctionConfig().type);
  if (!core.getCostFunctionConfig().weights.empty()) {
    cfcJson["weights"] = core.getCostFunctionConfig().weights;
  }
  json["costFunctionConfig"] = cfcJson;

  // Training config
  nlohmann::ordered_json tcJson;
  tcJson["numEpochs"] = core.getTrainingConfig().numEpochs;
  tcJson["learningRate"] = core.getTrainingConfig().learningRate;
  tcJson["batchSize"] = core.getTrainingConfig().batchSize;
  tcJson["shuffleSamples"] = core.getTrainingConfig().shuffleSamples;
  if (core.getTrainingConfig().dropoutRate > 0.0f)
    tcJson["dropoutRate"] = core.getTrainingConfig().dropoutRate;
  json["trainingConfig"] = tcJson;

  // Training metadata
  const auto& md = core.getTrainingMetadata();
  nlohmann::ordered_json mdJson;
  mdJson["startTime"] = md.startTime;
  mdJson["endTime"] = md.endTime;
  mdJson["durationSeconds"] = md.durationSeconds;
  mdJson["durationFormatted"] = md.durationFormatted;
  mdJson["numSamples"] = md.numSamples;
  mdJson["finalLoss"] = md.finalLoss;
  json["trainingMetadata"] = mdJson;

  // Parameters
  nlohmann::ordered_json paramsJson;

  // Conv parameters
  nlohmann::ordered_json convArr = nlohmann::ordered_json::array();
  for (const auto& cp : core.getParameters().convParams) {
    nlohmann::ordered_json cpJson;
    cpJson["numFilters"] = cp.numFilters;
    cpJson["inputC"] = cp.inputC;
    cpJson["filterH"] = cp.filterH;
    cpJson["filterW"] = cp.filterW;
    cpJson["filters"] = cp.filters;
    cpJson["biases"] = cp.biases;
    convArr.push_back(cpJson);
  }
  paramsJson["convolutional"] = convArr;

  // Dense parameters
  nlohmann::ordered_json denseParamsJson;
  denseParamsJson["weights"] = core.getParameters().denseParams.weights;
  denseParamsJson["biases"] = core.getParameters().denseParams.biases;
  paramsJson["dense"] = denseParamsJson;

  json["parameters"] = paramsJson;

  // Write to file
  QFile file(QString::fromStdString(filePath));
  if (!file.open(QIODevice::WriteOnly)) {
    throw std::runtime_error("Failed to open file for writing: " + filePath);
  }
  std::string jsonStr = json.dump(4);
  file.write(jsonStr.c_str());
  file.close();
}

//===================================================================================================================//
//  Output path helpers
//===================================================================================================================//

std::string Runner::generateTrainingFilename(ulong epochs, ulong samples, float loss) {
  std::ostringstream oss;
  oss << "trained_E-" << epochs
      << "_S-" << samples
      << "_L-" << std::fixed << std::setprecision(6) << loss
      << ".json";
  return oss.str();
}

//===================================================================================================================//

std::string Runner::generateDefaultOutputPath(
    const QString& inputFilePath,
    ulong epochs,
    ulong samples,
    float loss) {
  QFileInfo inputInfo(inputFilePath);
  QDir inputDir = inputInfo.absoluteDir();
  QDir outputDir(inputDir.filePath("output"));

  if (!outputDir.exists()) {
    inputDir.mkdir("output");
  }

  QString outputPath = outputDir.filePath(QString::fromStdString(generateTrainingFilename(epochs, samples, loss)));
  return outputPath.toStdString();
}

//===================================================================================================================//

std::string Runner::generateCheckpointPath(
    const QString& inputFilePath,
    ulong epoch,
    float loss) {
  QFileInfo inputInfo(inputFilePath);
  QDir inputDir = inputInfo.absoluteDir();
  QDir outputDir(inputDir.filePath("output"));

  if (!outputDir.exists()) {
    inputDir.mkdir("output");
  }

  std::ostringstream oss;
  oss << "checkpoint_E-" << epoch
      << "_L-" << std::fixed << std::setprecision(6) << loss
      << ".json";

  QString outputPath = outputDir.filePath(QString::fromStdString(oss.str()));
  return outputPath.toStdString();
}

//===================================================================================================================//

//===================================================================================================================//
//-- Data augmentation helpers --//
//===================================================================================================================//

// Helper: get class index from one-hot output vector
static ulong getClassIndex(const std::vector<float>& output) {
  return static_cast<ulong>(std::distance(output.begin(),
      std::max_element(output.begin(), output.end())));
}

// Helper: get data pointer and shape from a sample's input (works for both ANN and CNN)
static std::vector<float>& getInputData(ANN::Sample<float>& sample) { return sample.input; }
static std::vector<float>& getInputData(CNN::Sample<float>& sample) { return sample.input.data; }

static const std::vector<float>& getOutputVec(const ANN::Sample<float>& sample) { return sample.output; }
static const std::vector<float>& getOutputVec(const CNN::Sample<float>& sample) { return sample.output; }

// Helper: get image shape from IOConfig or CNN shape
struct ImageShape { int c, h, w; };

template <typename SampleT>
void Runner::augmentSamples(std::vector<SampleT>& samples) {
  if (this->augmentationFactor == 0 && !this->balanceAugmentation) return;
  if (samples.empty()) return;

  // Determine image shape
  ImageShape shape{0, 0, 0};
  if (this->networkType == NetworkType::CNN) {
    shape.c = static_cast<int>(this->cnnCoreConfig.inputShape.c);
    shape.h = static_cast<int>(this->cnnCoreConfig.inputShape.h);
    shape.w = static_cast<int>(this->cnnCoreConfig.inputShape.w);
  } else if (this->ioConfig.hasInputShape()) {
    shape.c = static_cast<int>(this->ioConfig.inputC);
    shape.h = static_cast<int>(this->ioConfig.inputH);
    shape.w = static_cast<int>(this->ioConfig.inputW);
  }

  bool hasImageShape = (shape.c > 0 && shape.h > 0 && shape.w > 0);

  // Count samples per class
  std::map<ulong, std::vector<ulong>> classIndices;
  for (ulong i = 0; i < samples.size(); i++) {
    ulong cls = getClassIndex(getOutputVec(samples[i]));
    classIndices[cls].push_back(i);
  }

  ulong maxClassCount = 0;
  for (const auto& [cls, indices] : classIndices) {
    maxClassCount = std::max(maxClassCount, static_cast<ulong>(indices.size()));
  }

  std::mt19937 rng(42);
  std::vector<SampleT> augmented;

  for (const auto& [cls, indices] : classIndices) {
    ulong currentCount = indices.size();
    ulong targetCount = currentCount;

    if (this->augmentationFactor > 0) {
      targetCount = currentCount * this->augmentationFactor;
    }

    if (this->balanceAugmentation) {
      ulong balancedTarget = maxClassCount;
      if (this->augmentationFactor > 0)
        balancedTarget = maxClassCount * this->augmentationFactor;
      targetCount = std::max(targetCount, balancedTarget);
    }

    // Generate augmented samples to reach targetCount (original samples are kept)
    ulong toGenerate = (targetCount > currentCount) ? (targetCount - currentCount) : 0;

    for (ulong i = 0; i < toGenerate; i++) {
      // Pick a random original sample from this class
      std::uniform_int_distribution<ulong> dist(0, currentCount - 1);
      ulong srcIdx = indices[dist(rng)];

      SampleT newSample = samples[srcIdx]; // copy

      if (hasImageShape) {
        ImageLoader::applyRandomTransforms(getInputData(newSample), shape.c, shape.h, shape.w, rng,
                                              this->augTransforms, this->augmentationProbability);
      } else if (this->augTransforms.gaussianNoise > 0.0f) {
        // For non-image data, add Gaussian noise with configured stddev
        ImageLoader::addGaussianNoise(getInputData(newSample), this->augTransforms.gaussianNoise, rng);
      }

      augmented.push_back(std::move(newSample));
    }
  }

  if (this->logLevel >= LogLevel::INFO && !augmented.empty()) {
    std::cout << "Data augmentation: " << samples.size() << " original + "
              << augmented.size() << " augmented = "
              << (samples.size() + augmented.size()) << " total samples\n";
  }

  samples.insert(samples.end(),
      std::make_move_iterator(augmented.begin()),
      std::make_move_iterator(augmented.end()));
}

// Explicit template instantiations
template void Runner::augmentSamples<ANN::Sample<float>>(std::vector<ANN::Sample<float>>&);
template void Runner::augmentSamples<CNN::Sample<float>>(std::vector<CNN::Sample<float>>&);

//===================================================================================================================//

template <typename SampleT>
std::vector<float> Runner::computeClassWeights(const std::vector<SampleT>& samples) {
  if (samples.empty()) return {};

  ulong numClasses = getOutputVec(samples[0]).size();

  // Count samples per class
  std::vector<ulong> classCounts(numClasses, 0);
  for (const auto& sample : samples) {
    ulong cls = getClassIndex(getOutputVec(sample));
    if (cls < numClasses) classCounts[cls]++;
  }

  ulong totalSamples = samples.size();

  // Inverse-frequency weighting: weight_c = totalSamples / (numClasses * count_c)
  std::vector<float> weights(numClasses, 1.0f);
  for (ulong c = 0; c < numClasses; c++) {
    if (classCounts[c] > 0) {
      weights[c] = static_cast<float>(totalSamples) /
                   (static_cast<float>(numClasses) * static_cast<float>(classCounts[c]));
    }
  }

  return weights;
}

template std::vector<float> Runner::computeClassWeights<ANN::Sample<float>>(const std::vector<ANN::Sample<float>>&);
template std::vector<float> Runner::computeClassWeights<CNN::Sample<float>>(const std::vector<CNN::Sample<float>>&);