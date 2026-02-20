#include "ANN-CLI_Runner.hpp"

#include "ANN-CLI_Loader.hpp"
#include "ANN-CLI_ProgressBar.hpp"
#include "ANN-CLI_Utils.hpp"

#include <ANN_Utils.hpp>

#include <QDir>
#include <QFile>
#include <QFileInfo>

#include <json.hpp>

#include <iomanip>
#include <iostream>
#include <sstream>

using namespace ANN_CLI;

//===================================================================================================================//

Runner::Runner(const QCommandLineParser& parser, bool verbose)
    : parser_(parser), verbose_(verbose) {
  QString configPath = parser_.value("config");

  // Build optional mode override
  std::optional<ANN::ModeType> modeOverride;

  if (parser_.isSet("mode")) {
    QString modeStr = parser_.value("mode").toLower();
    if (modeStr == "train") {
      modeOverride = ANN::ModeType::TRAIN;
    } else if (modeStr == "test") {
      modeOverride = ANN::ModeType::TEST;
    } else if (modeStr == "predict") {
      modeOverride = ANN::ModeType::PREDICT;
    }
  }

  // Build optional device override
  std::optional<ANN::DeviceType> deviceOverride;

  if (parser_.isSet("device")) {
    QString deviceStr = parser_.value("device").toLower();
    deviceOverride = (deviceStr == "gpu") ? ANN::DeviceType::GPU : ANN::DeviceType::CPU;
  }

  // Display info
  std::string deviceDisplay = deviceOverride.has_value()
      ? (deviceOverride.value() == ANN::DeviceType::GPU ? "gpu (CLI)" : "cpu (CLI)")
      : "from config file";

  std::string modeDisplay;

  if (modeOverride.has_value()) {
    switch (modeOverride.value()) {
      case ANN::ModeType::TRAIN:     modeDisplay = "train (CLI)";     break;
      case ANN::ModeType::TEST:      modeDisplay = "test (CLI)";      break;
      case ANN::ModeType::PREDICT:   modeDisplay = "predict (CLI)";   break;
      default:                       modeDisplay = "unknown (CLI)";   break;
    }
  } else {
    modeDisplay = "from config file";
  }

  if (verbose_) {
    std::cout << "Loading configuration from: " << configPath.toStdString() << "\n";
    std::cout << "Mode: " << modeDisplay << ", Device: " << deviceDisplay << "\n";
  }

  coreConfig_ = Loader::loadConfig(configPath.toStdString(), modeOverride, deviceOverride);
  coreConfig_.verbose = verbose_;
  core_ = ANN::Core<float>::makeCore(coreConfig_);
}

//===================================================================================================================//

int Runner::run() {
  if (coreConfig_.modeType == ANN::ModeType::TRAIN) {
    return runTrain();
  } else if (coreConfig_.modeType == ANN::ModeType::TEST) {
    return runTest();
  } else {
    return runPredict();
  }
}

//===================================================================================================================//

int Runner::runTrain() {
  QString inputFilePath;
  auto [samples, success] = loadSamplesFromOptions("training", inputFilePath);
  if (!success) {
    return 1;
  }

  if (verbose_) std::cout << "Starting training...\n";

  // Set up progress callback
  ProgressBar progressBar;
  core_->setTrainingCallback([&progressBar](const ANN::TrainingProgress<float>& progress) {
    progressBar.update(progress);
  });

  core_->train(samples);
  std::cout << "\nTraining completed.\n";

  // Get training info for filename generation
  const auto& trainingConfig = core_->getTrainingConfig();
  const auto& trainingMetadata = core_->getTrainingMetadata();

  // Save the trained model
  std::string outputPathStr;

  if (parser_.isSet("output")) {
    outputPathStr = parser_.value("output").toStdString();
  } else {
    outputPathStr = generateDefaultOutputPath(
      inputFilePath,
      trainingConfig.numEpochs,
      trainingMetadata.numSamples,
      trainingMetadata.finalLoss
    );
  }

  ANN::Utils<float>::save(*core_, outputPathStr);
  std::cout << "Model saved to: " << outputPathStr << "\n";

  return 0;
}

//===================================================================================================================//

int Runner::runTest() {
  QString inputFilePath;
  auto [samples, success] = loadSamplesFromOptions("test", inputFilePath);

  if (!success) {
    return 1;
  }

  if (verbose_) std::cout << "Running evaluation...\n";

  ANN::TestResult<float> result = core_->test(samples);

  std::cout << "\nTest Results:\n";
  std::cout << "  Samples evaluated: " << result.numSamples << "\n";
  std::cout << "  Total loss:        " << result.totalLoss << "\n";
  std::cout << "  Average loss:      " << result.averageLoss << "\n";

  return 0;
}

//===================================================================================================================//

int Runner::runPredict() {
  if (!parser_.isSet("input")) {
    std::cerr << "Error: --input option is required for predict mode.\n";
    return 1;
  }

  QString inputPath = parser_.value("input");

  // Generate default output path if not specified
  QString outputPath;

  if (parser_.isSet("output")) {
    outputPath = parser_.value("output");
  } else {
    QFileInfo inputInfo(inputPath);
    QString baseName = inputInfo.completeBaseName();
    QDir inputDir = inputInfo.absoluteDir();
    QDir outputDir(inputDir.filePath("output"));

    if (!outputDir.exists()) {
      inputDir.mkdir("output");
    }

    outputPath = outputDir.filePath("predict_" + baseName + ".json");
  }

  if (verbose_) std::cout << "Loading input from: " << inputPath.toStdString() << "\n";

  ANN::Input<float> input = Loader::loadInput(inputPath.toStdString());

  if (verbose_) {
    std::cout << "Running ANN with input: ";
    for (size_t i = 0; i < input.size(); ++i) {
      std::cout << input[i];
      if (i < input.size() - 1) std::cout << ", ";
    }
    std::cout << "\n";
  }

  ANN::Output<float> output = core_->predict(input);

  // Get predict metadata from core
  const auto& predictMetadata = core_->getPredictMetadata();

  nlohmann::ordered_json resultJson;

  nlohmann::ordered_json predictMetadataJson;
  predictMetadataJson["startTime"] = predictMetadata.startTime;
  predictMetadataJson["endTime"] = predictMetadata.endTime;
  predictMetadataJson["durationSeconds"] = predictMetadata.durationSeconds;
  predictMetadataJson["durationFormatted"] = predictMetadata.durationFormatted;
  resultJson["predictMetadata"] = predictMetadataJson;

  resultJson["output"] = output;

  // Write to file
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

std::pair<ANN::Samples<float>, bool> Runner::loadSamplesFromOptions(
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
    samples = Loader::loadSamples(samplesPath.toStdString());
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

    samples = Utils<float>::loadIDX(idxDataPath.toStdString(), idxLabelsPath.toStdString());
  } else {
    std::cerr << "Error: " << modeName << " requires either --samples (JSON) or --idx-data and --idx-labels (IDX).\n";
    return {samples, false};
  }

  if (verbose_) std::cout << "Loaded " << samples.size() << " " << modeName << " samples.\n";

  return {samples, true};
}

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