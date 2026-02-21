#include "NN-CLI_Runner.hpp"

#include "NN-CLI_Loader.hpp"
#include "NN-CLI_ProgressBar.hpp"
#include "NN-CLI_Utils.hpp"

#include <ANN_Utils.hpp>
#include <CNN_Utils.hpp>

#include <QDir>
#include <QFile>
#include <QFileInfo>

#include <json.hpp>

#include <iomanip>
#include <iostream>
#include <sstream>

using namespace NN_CLI;

//===================================================================================================================//

Runner::Runner(const QCommandLineParser& parser, bool verbose)
    : parser_(parser), verbose_(verbose) {
  QString configPath = parser_.value("config");

  // Detect network type from config file
  networkType_ = Loader::detectNetworkType(configPath.toStdString());

  // Build optional mode/device overrides as strings
  std::optional<std::string> modeOverride;
  if (parser_.isSet("mode")) {
    modeOverride = parser_.value("mode").toLower().toStdString();
  }

  std::optional<std::string> deviceOverride;
  if (parser_.isSet("device")) {
    deviceOverride = parser_.value("device").toLower().toStdString();
  }

  // Display info
  std::string networkTypeStr = (networkType_ == NetworkType::CNN) ? "CNN" : "ANN";
  std::string modeDisplay = modeOverride.has_value() ? (modeOverride.value() + " (CLI)") : "from config file";
  std::string deviceDisplay = deviceOverride.has_value() ? (deviceOverride.value() + " (CLI)") : "from config file";

  if (verbose_) {
    std::cout << "Network type: " << networkTypeStr << "\n";
    std::cout << "Loading configuration from: " << configPath.toStdString() << "\n";
    std::cout << "Mode: " << modeDisplay << ", Device: " << deviceDisplay << "\n";
  }

  if (networkType_ == NetworkType::ANN) {
    // Convert string overrides to ANN enum overrides
    std::optional<ANN::ModeType> annModeOverride;
    if (modeOverride.has_value()) annModeOverride = ANN::Mode::nameToType(modeOverride.value());

    std::optional<ANN::DeviceType> annDeviceOverride;
    if (deviceOverride.has_value()) annDeviceOverride = ANN::Device::nameToType(deviceOverride.value());

    annCoreConfig_ = Loader::loadANNConfig(configPath.toStdString(), annModeOverride, annDeviceOverride);
    annCoreConfig_.verbose = verbose_;
    mode_ = ANN::Mode::typeToName(annCoreConfig_.modeType);
    annCore_ = ANN::Core<float>::makeCore(annCoreConfig_);
  } else {
    cnnCoreConfig_ = Loader::loadCNNConfig(configPath.toStdString(), modeOverride, deviceOverride);
    cnnCoreConfig_.verbose = verbose_;
    mode_ = CNN::Mode::typeToName(cnnCoreConfig_.modeType);
    cnnCore_ = CNN::Core<float>::makeCore(cnnCoreConfig_);
  }
}

//===================================================================================================================//

int Runner::run() {
  if (networkType_ == NetworkType::ANN) {
    if (mode_ == "train")   return runANNTrain();
    if (mode_ == "test")    return runANNTest();
    return runANNPredict();
  } else {
    if (mode_ == "train")   return runCNNTrain();
    if (mode_ == "test")    return runCNNTest();
    return runCNNPredict();
  }
}

//===================================================================================================================//

int Runner::runANNTrain() {
  QString inputFilePath;
  auto [samples, success] = loadANNSamplesFromOptions("training", inputFilePath);
  if (!success) return 1;

  if (verbose_) std::cout << "Starting ANN training...\n";

  ProgressBar progressBar;
  annCore_->setTrainingCallback([&progressBar](const ANN::TrainingProgress<float>& progress) {
    ProgressInfo info{progress.currentEpoch, progress.totalEpochs,
                      progress.currentSample, progress.totalSamples,
                      progress.epochLoss, progress.sampleLoss,
                      progress.gpuIndex, progress.totalGPUs};
    progressBar.update(info);
  });

  annCore_->train(samples);
  std::cout << "\nTraining completed.\n";

  const auto& trainingConfig = annCore_->getTrainingConfig();
  const auto& trainingMetadata = annCore_->getTrainingMetadata();

  std::string outputPathStr;
  if (parser_.isSet("output")) {
    outputPathStr = parser_.value("output").toStdString();
  } else {
    outputPathStr = generateDefaultOutputPath(
      inputFilePath, trainingConfig.numEpochs,
      trainingMetadata.numSamples, trainingMetadata.finalLoss);
  }

  ANN::Utils<float>::save(*annCore_, outputPathStr);
  std::cout << "Model saved to: " << outputPathStr << "\n";

  return 0;
}

//===================================================================================================================//

int Runner::runANNTest() {
  QString inputFilePath;
  auto [samples, success] = loadANNSamplesFromOptions("test", inputFilePath);
  if (!success) return 1;

  if (verbose_) std::cout << "Running ANN evaluation...\n";

  ANN::TestResult<float> result = annCore_->test(samples);

  std::cout << "\nTest Results:\n";
  std::cout << "  Samples evaluated: " << result.numSamples << "\n";
  std::cout << "  Total loss:        " << result.totalLoss << "\n";
  std::cout << "  Average loss:      " << result.averageLoss << "\n";

  return 0;
}

//===================================================================================================================//

int Runner::runANNPredict() {
  if (!parser_.isSet("input")) {
    std::cerr << "Error: --input option is required for predict mode.\n";
    return 1;
  }

  QString inputPath = parser_.value("input");
  QString outputPath;

  if (parser_.isSet("output")) {
    outputPath = parser_.value("output");
  } else {
    QFileInfo inputInfo(inputPath);
    QDir inputDir = inputInfo.absoluteDir();
    QDir outputDir(inputDir.filePath("output"));
    if (!outputDir.exists()) inputDir.mkdir("output");
    outputPath = outputDir.filePath("predict_" + inputInfo.completeBaseName() + ".json");
  }

  if (verbose_) std::cout << "Loading input from: " << inputPath.toStdString() << "\n";

  ANN::Input<float> input = Loader::loadANNInput(inputPath.toStdString());

  if (verbose_) {
    std::cout << "Running ANN with input: ";
    for (size_t i = 0; i < input.size(); ++i) {
      std::cout << input[i];
      if (i < input.size() - 1) std::cout << ", ";
    }
    std::cout << "\n";
  }

  ANN::Output<float> output = annCore_->predict(input);
  const auto& predictMetadata = annCore_->getPredictMetadata();

  nlohmann::ordered_json resultJson;
  nlohmann::ordered_json predictMetadataJson;
  predictMetadataJson["startTime"] = predictMetadata.startTime;
  predictMetadataJson["endTime"] = predictMetadata.endTime;
  predictMetadataJson["durationSeconds"] = predictMetadata.durationSeconds;
  predictMetadataJson["durationFormatted"] = predictMetadata.durationFormatted;
  resultJson["predictMetadata"] = predictMetadataJson;
  resultJson["output"] = output;

  QFile outputFile(outputPath);
  if (!outputFile.open(QIODevice::WriteOnly | QIODevice::Text)) {
    std::cerr << "Error: Failed to open output file: " << outputPath.toStdString() << "\n";
    return 1;
  }

  std::string jsonStr = resultJson.dump(2);
  outputFile.write(jsonStr.c_str(), jsonStr.size());
  outputFile.close();

  std::cout << "Predict result saved to: " << outputPath.toStdString() << "\n";
  return 0;
}

//===================================================================================================================//
//  CNN mode methods
//===================================================================================================================//

int Runner::runCNNTrain() {
  QString inputFilePath;
  auto [samples, success] = loadCNNSamplesFromOptions("training", inputFilePath);
  if (!success) return 1;

  if (verbose_) std::cout << "Starting CNN training...\n";

  ProgressBar progressBar;
  cnnCore_->setTrainingCallback([&progressBar](const CNN::TrainingProgress<float>& progress) {
    ProgressInfo info{progress.currentEpoch, progress.totalEpochs,
                      progress.currentSample, progress.totalSamples,
                      progress.epochLoss, progress.sampleLoss,
                      progress.gpuIndex, progress.totalGPUs};
    progressBar.update(info);
  });

  cnnCore_->train(samples);
  std::cout << "\nTraining completed.\n";

  const auto& trainingConfig = cnnCore_->getTrainingConfig();
  const auto& trainingMetadata = cnnCore_->getTrainingMetadata();

  std::string outputPathStr;
  if (parser_.isSet("output")) {
    outputPathStr = parser_.value("output").toStdString();
  } else {
    outputPathStr = generateDefaultOutputPath(
      inputFilePath, trainingConfig.numEpochs,
      trainingMetadata.numSamples, trainingMetadata.finalLoss);
  }

  CNN::Utils<float>::save(*cnnCore_, outputPathStr);
  std::cout << "Model saved to: " << outputPathStr << "\n";

  return 0;
}

//===================================================================================================================//

int Runner::runCNNTest() {
  QString inputFilePath;
  auto [samples, success] = loadCNNSamplesFromOptions("test", inputFilePath);
  if (!success) return 1;

  if (verbose_) std::cout << "Running CNN evaluation...\n";

  CNN::TestResult<float> result = cnnCore_->test(samples);

  std::cout << "\nTest Results:\n";
  std::cout << "  Samples evaluated: " << result.numSamples << "\n";
  std::cout << "  Total loss:        " << result.totalLoss << "\n";
  std::cout << "  Average loss:      " << result.averageLoss << "\n";

  return 0;
}

//===================================================================================================================//

int Runner::runCNNPredict() {
  if (!parser_.isSet("input")) {
    std::cerr << "Error: --input option is required for predict mode.\n";
    return 1;
  }

  QString inputPath = parser_.value("input");
  QString outputPath;

  if (parser_.isSet("output")) {
    outputPath = parser_.value("output");
  } else {
    QFileInfo inputInfo(inputPath);
    QDir inputDir = inputInfo.absoluteDir();
    QDir outputDir(inputDir.filePath("output"));
    if (!outputDir.exists()) inputDir.mkdir("output");
    outputPath = outputDir.filePath("predict_" + inputInfo.completeBaseName() + ".json");
  }

  if (verbose_) std::cout << "Loading input from: " << inputPath.toStdString() << "\n";

  CNN::Input<float> input = Loader::loadCNNInput(inputPath.toStdString(), cnnCoreConfig_.inputShape);

  CNN::Output<float> output = cnnCore_->predict(input);
  const auto& predictMetadata = cnnCore_->getPredictMetadata();

  nlohmann::ordered_json resultJson;
  nlohmann::ordered_json predictMetadataJson;
  predictMetadataJson["startTime"] = predictMetadata.startTime;
  predictMetadataJson["endTime"] = predictMetadata.endTime;
  predictMetadataJson["durationSeconds"] = predictMetadata.durationSeconds;
  predictMetadataJson["durationFormatted"] = predictMetadata.durationFormatted;
  resultJson["predictMetadata"] = predictMetadataJson;
  resultJson["output"] = output;

  QFile outputFile(outputPath);
  if (!outputFile.open(QIODevice::WriteOnly | QIODevice::Text)) {
    std::cerr << "Error: Failed to open output file: " << outputPath.toStdString() << "\n";
    return 1;
  }

  std::string jsonStr = resultJson.dump(2);
  outputFile.write(jsonStr.c_str(), jsonStr.size());
  outputFile.close();

  std::cout << "Predict result saved to: " << outputPath.toStdString() << "\n";
  return 0;
}

//===================================================================================================================//
//  Sample loading helpers
//===================================================================================================================//

std::pair<ANN::Samples<float>, bool> Runner::loadANNSamplesFromOptions(
    const std::string& modeName,
    QString& inputFilePath) {
  ANN::Samples<float> samples;

  bool hasJsonSamples = parser_.isSet("samples");
  bool hasIdxData = parser_.isSet("idx-data");
  bool hasIdxLabels = parser_.isSet("idx-labels");

  if (hasJsonSamples && hasIdxData) {
    std::cerr << "Error: Cannot use both --samples and --idx-data. Choose one format.\n";
    return {samples, false};
  }

  if (hasJsonSamples) {
    QString samplesPath = parser_.value("samples");
    inputFilePath = samplesPath;
    if (verbose_) std::cout << "Loading " << modeName << " samples from JSON: " << samplesPath.toStdString() << "\n";
    samples = Loader::loadANNSamples(samplesPath.toStdString());
  } else if (hasIdxData) {
    if (!hasIdxLabels) {
      std::cerr << "Error: --idx-labels is required when using --idx-data.\n";
      return {samples, false};
    }

    QString idxDataPath = parser_.value("idx-data");
    QString idxLabelsPath = parser_.value("idx-labels");
    inputFilePath = idxDataPath;

    if (verbose_) {
      std::cout << "Loading " << modeName << " samples from IDX:\n";
      std::cout << "  Data:   " << idxDataPath.toStdString() << "\n";
      std::cout << "  Labels: " << idxLabelsPath.toStdString() << "\n";
    }

    samples = Utils<float>::loadANNIDX(idxDataPath.toStdString(), idxLabelsPath.toStdString());
  } else {
    std::cerr << "Error: " << modeName << " requires either --samples (JSON) or --idx-data and --idx-labels (IDX).\n";
    return {samples, false};
  }

  if (verbose_) std::cout << "Loaded " << samples.size() << " " << modeName << " samples.\n";

  return {samples, true};
}

//===================================================================================================================//

std::pair<CNN::Samples<float>, bool> Runner::loadCNNSamplesFromOptions(
    const std::string& modeName,
    QString& inputFilePath) {
  CNN::Samples<float> samples;

  bool hasJsonSamples = parser_.isSet("samples");
  bool hasIdxData = parser_.isSet("idx-data");
  bool hasIdxLabels = parser_.isSet("idx-labels");

  if (hasJsonSamples && hasIdxData) {
    std::cerr << "Error: Cannot use both --samples and --idx-data. Choose one format.\n";
    return {samples, false};
  }

  const CNN::Shape3D& inputShape = cnnCoreConfig_.inputShape;

  if (hasJsonSamples) {
    QString samplesPath = parser_.value("samples");
    inputFilePath = samplesPath;
    if (verbose_) std::cout << "Loading " << modeName << " samples from JSON: " << samplesPath.toStdString() << "\n";
    samples = Loader::loadCNNSamples(samplesPath.toStdString(), inputShape);
  } else if (hasIdxData) {
    if (!hasIdxLabels) {
      std::cerr << "Error: --idx-labels is required when using --idx-data.\n";
      return {samples, false};
    }

    QString idxDataPath = parser_.value("idx-data");
    QString idxLabelsPath = parser_.value("idx-labels");
    inputFilePath = idxDataPath;

    if (verbose_) {
      std::cout << "Loading " << modeName << " samples from IDX:\n";
      std::cout << "  Data:   " << idxDataPath.toStdString() << "\n";
      std::cout << "  Labels: " << idxLabelsPath.toStdString() << "\n";
    }

    samples = Utils<float>::loadCNNIDX(idxDataPath.toStdString(), idxLabelsPath.toStdString(), inputShape);
  } else {
    std::cerr << "Error: " << modeName << " requires either --samples (JSON) or --idx-data and --idx-labels (IDX).\n";
    return {samples, false};
  }

  if (verbose_) std::cout << "Loaded " << samples.size() << " " << modeName << " samples.\n";

  return {samples, true};
}

//===================================================================================================================//
//  Utility methods
//===================================================================================================================//

std::string Runner::generateTrainingFilename(ulong epochs, ulong samples, float loss) {
  std::ostringstream oss;
  oss << "trained_model_"
      << epochs << "_"
      << samples << "_"
      << std::fixed << std::setprecision(6) << loss
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