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

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace NN_CLI;

//===================================================================================================================//

Runner::Runner(const QCommandLineParser& parser, bool verbose)
    : parser(parser), verbose(verbose) {
  QString configPath = this->parser.value("config");

  // Detect network type from config file
  networkType = Loader::detectNetworkType(configPath.toStdString());

  // Build optional mode/device overrides as strings
  std::optional<std::string> modeOverride;
  if (this->parser.isSet("mode")) {
    modeOverride = this->parser.value("mode").toLower().toStdString();
  }

  std::optional<std::string> deviceOverride;
  if (this->parser.isSet("device")) {
    deviceOverride = this->parser.value("device").toLower().toStdString();
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

  ioConfig = Loader::loadIOConfig(configPath.toStdString(), inputTypeOverride, outputTypeOverride);

  // Display info
  std::string networkTypeStr = (networkType == NetworkType::CNN) ? "CNN" : "ANN";
  std::string modeDisplay = modeOverride.has_value() ? (modeOverride.value() + " (CLI)") : "from config file";
  std::string deviceDisplay = deviceOverride.has_value() ? (deviceOverride.value() + " (CLI)") : "from config file";

  if (this->verbose) {
    std::cout << "Network type: " << networkTypeStr << "\n";
    std::cout << "Loading configuration from: " << configPath.toStdString() << "\n";
    std::cout << "Mode: " << modeDisplay << ", Device: " << deviceDisplay << "\n";
    std::cout << "Input type: " << dataTypeToString(ioConfig.inputType)
              << ", Output type: " << dataTypeToString(ioConfig.outputType) << "\n";
  }

  if (networkType == NetworkType::ANN) {
    // Convert string overrides to ANN enum overrides
    std::optional<ANN::ModeType> annModeOverride;
    if (modeOverride.has_value()) annModeOverride = ANN::Mode::nameToType(modeOverride.value());

    std::optional<ANN::DeviceType> annDeviceOverride;
    if (deviceOverride.has_value()) annDeviceOverride = ANN::Device::nameToType(deviceOverride.value());

    annCoreConfig = Loader::loadANNConfig(configPath.toStdString(), annModeOverride, annDeviceOverride);
    annCoreConfig.verbose = this->verbose;
    mode = ANN::Mode::typeToName(annCoreConfig.modeType);
    annCore = ANN::Core<float>::makeCore(annCoreConfig);
  } else {
    cnnCoreConfig = Loader::loadCNNConfig(configPath.toStdString(), modeOverride, deviceOverride);
    cnnCoreConfig.verbose = this->verbose;
    mode = CNN::Mode::typeToName(cnnCoreConfig.modeType);
    cnnCore = CNN::Core<float>::makeCore(cnnCoreConfig);
  }
}

//===================================================================================================================//

int Runner::run() {
  if (networkType == NetworkType::ANN) {
    if (mode == "train")   return runANNTrain();
    if (mode == "test")    return runANNTest();
    return runANNPredict();
  } else {
    if (mode == "train")   return runCNNTrain();
    if (mode == "test")    return runCNNTest();
    return runCNNPredict();
  }
}

//===================================================================================================================//

int Runner::runANNTrain() {
  QString inputFilePath;
  auto [samples, success] = loadANNSamplesFromOptions("training", inputFilePath);
  if (!success) return 1;

  if (verbose) std::cout << "Starting ANN training...\n";

  ProgressBar progressBar;
  annCore->setTrainingCallback([&progressBar](const ANN::TrainingProgress<float>& progress) {
    ProgressInfo info{progress.currentEpoch, progress.totalEpochs,
                      progress.currentSample, progress.totalSamples,
                      progress.epochLoss, progress.sampleLoss,
                      progress.gpuIndex, progress.totalGPUs};
    progressBar.update(info);
  });

  annCore->train(samples);
  std::cout << "\nTraining completed.\n";

  const auto& trainingConfig = annCore->getTrainingConfig();
  const auto& trainingMetadata = annCore->getTrainingMetadata();

  std::string outputPathStr;
  if (parser.isSet("output")) {
    outputPathStr = parser.value("output").toStdString();
  } else {
    outputPathStr = generateDefaultOutputPath(
      inputFilePath, trainingConfig.numEpochs,
      trainingMetadata.numSamples, trainingMetadata.finalLoss);
  }

  saveANNModel(*annCore, outputPathStr, ioConfig);
  std::cout << "Model saved to: " << outputPathStr << "\n";

  return 0;
}

//===================================================================================================================//

int Runner::runANNTest() {
  QString inputFilePath;
  auto [samples, success] = loadANNSamplesFromOptions("test", inputFilePath);
  if (!success) return 1;

  if (verbose) std::cout << "Running ANN evaluation...\n";

  ANN::TestResult<float> result = annCore->test(samples);

  std::cout << "\nTest Results:\n";
  std::cout << "  Samples evaluated: " << result.numSamples << "\n";
  std::cout << "  Total loss:        " << result.totalLoss << "\n";
  std::cout << "  Average loss:      " << result.averageLoss << "\n";

  return 0;
}

//===================================================================================================================//

int Runner::runANNPredict() {
  if (!parser.isSet("input")) {
    std::cerr << "Error: --input option is required for predict mode.\n";
    return 1;
  }

  QString inputPath = parser.value("input");
  QString outputPath;

  if (parser.isSet("output")) {
    outputPath = parser.value("output");
  } else {
    QFileInfo inputInfo(inputPath);
    QDir inputDir = inputInfo.absoluteDir();
    QDir outputDir(inputDir.filePath("output"));
    if (!outputDir.exists()) inputDir.mkdir("output");

    if (ioConfig.outputType == DataType::IMAGE) {
      outputPath = outputDir.filePath("predict_" + inputInfo.completeBaseName());
    } else {
      outputPath = outputDir.filePath("predict_" + inputInfo.completeBaseName() + ".json");
    }
  }

  if (verbose) std::cout << "Loading inputs from: " << inputPath.toStdString() << "\n";

  std::vector<ANN::Input<float>> inputs = Loader::loadANNInputs(inputPath.toStdString(), ioConfig, annCoreConfig.progressReports);

  if (verbose) {
    std::cout << "Loaded " << inputs.size() << " input(s), each with " << inputs[0].size() << " values\n";
  }

  // Track overall batch timing
  auto batchStart = std::chrono::system_clock::now();
  std::string startTimeStr = ANN::Utils<float>::formatISO8601();

  std::vector<ANN::Output<float>> outputs;
  outputs.reserve(inputs.size());

  for (size_t i = 0; i < inputs.size(); ++i) {
    ANN::Output<float> output = annCore->predict(inputs[i]);
    outputs.push_back(std::move(output));
    if (verbose && inputs.size() > 1) {
      std::cout << "  Predicted input " << (i + 1) << "/" << inputs.size() << "\n";
    }
  }

  auto batchEnd = std::chrono::system_clock::now();
  std::string endTimeStr = ANN::Utils<float>::formatISO8601();
  std::chrono::duration<double> batchElapsed = batchEnd - batchStart;
  double batchDurationSeconds = batchElapsed.count();
  std::string batchDurationFormatted = ANN::Utils<float>::formatDuration(batchDurationSeconds);

  // When outputType is IMAGE, save images to a folder
  if (ioConfig.outputType == DataType::IMAGE) {
    if (!ioConfig.hasOutputShape()) {
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
          static_cast<int>(ioConfig.outputC),
          static_cast<int>(ioConfig.outputH),
          static_cast<int>(ioConfig.outputW));
    }

    std::cout << "Predict images saved to: " << outputPath.toStdString() << "\n";
    std::cout << "  Images: " << outputs.size() << "\n";
    std::cout << "  Shape: " << ioConfig.outputC << "x" << ioConfig.outputH << "x" << ioConfig.outputW << "\n";
    std::cout << "  Duration: " << batchDurationFormatted << "\n";
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

  if (verbose) std::cout << "Starting CNN training...\n";

  ProgressBar progressBar;
  cnnCore->setTrainingCallback([&progressBar](const CNN::TrainingProgress<float>& progress) {
    ProgressInfo info{progress.currentEpoch, progress.totalEpochs,
                      progress.currentSample, progress.totalSamples,
                      progress.epochLoss, progress.sampleLoss,
                      progress.gpuIndex, progress.totalGPUs};
    progressBar.update(info);
  });

  cnnCore->train(samples);
  std::cout << "\nTraining completed.\n";

  const auto& trainingConfig = cnnCore->getTrainingConfig();
  const auto& trainingMetadata = cnnCore->getTrainingMetadata();

  std::string outputPathStr;
  if (parser.isSet("output")) {
    outputPathStr = parser.value("output").toStdString();
  } else {
    outputPathStr = generateDefaultOutputPath(
      inputFilePath, trainingConfig.numEpochs,
      trainingMetadata.numSamples, trainingMetadata.finalLoss);
  }

  saveCNNModel(*cnnCore, outputPathStr, ioConfig);
  std::cout << "Model saved to: " << outputPathStr << "\n";

  return 0;
}

//===================================================================================================================//

int Runner::runCNNTest() {
  QString inputFilePath;
  auto [samples, success] = loadCNNSamplesFromOptions("test", inputFilePath);
  if (!success) return 1;

  if (verbose) std::cout << "Running CNN evaluation...\n";

  CNN::TestResult<float> result = cnnCore->test(samples);

  std::cout << "\nTest Results:\n";
  std::cout << "  Samples evaluated: " << result.numSamples << "\n";
  std::cout << "  Total loss:        " << result.totalLoss << "\n";
  std::cout << "  Average loss:      " << result.averageLoss << "\n";

  return 0;
}

//===================================================================================================================//

int Runner::runCNNPredict() {
  if (!parser.isSet("input")) {
    std::cerr << "Error: --input option is required for predict mode.\n";
    return 1;
  }

  QString inputPath = parser.value("input");
  QString outputPath;

  if (parser.isSet("output")) {
    outputPath = parser.value("output");
  } else {
    QFileInfo inputInfo(inputPath);
    QDir inputDir = inputInfo.absoluteDir();
    QDir outputDir(inputDir.filePath("output"));
    if (!outputDir.exists()) inputDir.mkdir("output");

    if (ioConfig.outputType == DataType::IMAGE) {
      outputPath = outputDir.filePath("predict_" + inputInfo.completeBaseName());
    } else {
      outputPath = outputDir.filePath("predict_" + inputInfo.completeBaseName() + ".json");
    }
  }

  if (verbose) std::cout << "Loading inputs from: " << inputPath.toStdString() << "\n";

  std::vector<CNN::Input<float>> inputs = Loader::loadCNNInputs(
      inputPath.toStdString(), cnnCoreConfig.inputShape, ioConfig, cnnCoreConfig.progressReports);

  if (verbose) {
    std::cout << "Loaded " << inputs.size() << " input(s), each with " << inputs[0].data.size() << " values\n";
  }

  // Track overall batch timing
  auto batchStart = std::chrono::system_clock::now();
  std::string startTimeStr = ANN::Utils<float>::formatISO8601();

  std::vector<CNN::Output<float>> outputs;
  outputs.reserve(inputs.size());

  for (size_t i = 0; i < inputs.size(); ++i) {
    CNN::Output<float> output = cnnCore->predict(inputs[i]);
    outputs.push_back(std::move(output));
    if (verbose && inputs.size() > 1) {
      std::cout << "  Predicted input " << (i + 1) << "/" << inputs.size() << "\n";
    }
  }

  auto batchEnd = std::chrono::system_clock::now();
  std::string endTimeStr = ANN::Utils<float>::formatISO8601();
  std::chrono::duration<double> batchElapsed = batchEnd - batchStart;
  double batchDurationSeconds = batchElapsed.count();
  std::string batchDurationFormatted = ANN::Utils<float>::formatDuration(batchDurationSeconds);

  // When outputType is IMAGE, save images to a folder
  if (ioConfig.outputType == DataType::IMAGE) {
    if (!ioConfig.hasOutputShape()) {
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
          static_cast<int>(ioConfig.outputC),
          static_cast<int>(ioConfig.outputH),
          static_cast<int>(ioConfig.outputW));
    }

    std::cout << "Predict images saved to: " << outputPath.toStdString() << "\n";
    std::cout << "  Images: " << outputs.size() << "\n";
    std::cout << "  Shape: " << ioConfig.outputC << "x" << ioConfig.outputH << "x" << ioConfig.outputW << "\n";
    std::cout << "  Duration: " << batchDurationFormatted << "\n";
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

  bool hasJsonSamples = parser.isSet("samples");
  bool hasIdxData = parser.isSet("idx-data");
  bool hasIdxLabels = parser.isSet("idx-labels");

  if (hasJsonSamples && hasIdxData) {
    std::cerr << "Error: Cannot use both --samples and --idx-data. Choose one format.\n";
    return {samples, false};
  }

  if (hasJsonSamples) {
    QString samplesPath = parser.value("samples");
    inputFilePath = samplesPath;
    if (verbose) std::cout << "Loading " << modeName << " samples from JSON: " << samplesPath.toStdString() << "\n";
    samples = Loader::loadANNSamples(samplesPath.toStdString(), ioConfig, annCoreConfig.progressReports);
  } else if (hasIdxData) {
    if (!hasIdxLabels) {
      std::cerr << "Error: --idx-labels is required when using --idx-data.\n";
      return {samples, false};
    }

    QString idxDataPath = parser.value("idx-data");
    QString idxLabelsPath = parser.value("idx-labels");
    inputFilePath = idxDataPath;

    if (verbose) {
      std::cout << "Loading " << modeName << " samples from IDX:\n";
      std::cout << "  Data:   " << idxDataPath.toStdString() << "\n";
      std::cout << "  Labels: " << idxLabelsPath.toStdString() << "\n";
    }

    samples = Utils<float>::loadANNIDX(idxDataPath.toStdString(), idxLabelsPath.toStdString(), annCoreConfig.progressReports);
  } else {
    std::cerr << "Error: " << modeName << " requires either --samples (JSON) or --idx-data and --idx-labels (IDX).\n";
    return {samples, false};
  }

  if (verbose) std::cout << "Loaded " << samples.size() << " " << modeName << " samples.\n";

  return {samples, true};
}

//===================================================================================================================//

std::pair<CNN::Samples<float>, bool> Runner::loadCNNSamplesFromOptions(
    const std::string& modeName,
    QString& inputFilePath) {
  CNN::Samples<float> samples;

  bool hasJsonSamples = parser.isSet("samples");
  bool hasIdxData = parser.isSet("idx-data");
  bool hasIdxLabels = parser.isSet("idx-labels");

  if (hasJsonSamples && hasIdxData) {
    std::cerr << "Error: Cannot use both --samples and --idx-data. Choose one format.\n";
    return {samples, false};
  }

  const CNN::Shape3D& inputShape = cnnCoreConfig.inputShape;

  if (hasJsonSamples) {
    QString samplesPath = parser.value("samples");
    inputFilePath = samplesPath;
    if (verbose) std::cout << "Loading " << modeName << " samples from JSON: " << samplesPath.toStdString() << "\n";
    samples = Loader::loadCNNSamples(samplesPath.toStdString(), inputShape, ioConfig, cnnCoreConfig.progressReports);
  } else if (hasIdxData) {
    if (!hasIdxLabels) {
      std::cerr << "Error: --idx-labels is required when using --idx-data.\n";
      return {samples, false};
    }

    QString idxDataPath = parser.value("idx-data");
    QString idxLabelsPath = parser.value("idx-labels");
    inputFilePath = idxDataPath;

    if (verbose) {
      std::cout << "Loading " << modeName << " samples from IDX:\n";
      std::cout << "  Data:   " << idxDataPath.toStdString() << "\n";
      std::cout << "  Labels: " << idxLabelsPath.toStdString() << "\n";
    }

    samples = Utils<float>::loadCNNIDX(idxDataPath.toStdString(), idxLabelsPath.toStdString(), inputShape, cnnCoreConfig.progressReports);
  } else {
    std::cerr << "Error: " << modeName << " requires either --samples (JSON) or --idx-data and --idx-labels (IDX).\n";
    return {samples, false};
  }

  if (verbose) std::cout << "Loaded " << samples.size() << " " << modeName << " samples.\n";

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
//  Model save helpers
//===================================================================================================================//

void Runner::saveANNModel(const ANN::Core<float>& core, const std::string& filePath,
                           const IOConfig& ioConfig) {
  nlohmann::ordered_json json;

  json["device"] = ANN::Device::typeToName(core.getDeviceType());
  json["mode"] = ANN::Mode::typeToName(core.getModeType());

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

  // Training config
  nlohmann::ordered_json tcJson;
  tcJson["numEpochs"] = core.getTrainingConfig().numEpochs;
  tcJson["learningRate"] = core.getTrainingConfig().learningRate;
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
                           const IOConfig& ioConfig) {
  nlohmann::ordered_json json;

  json["device"] = CNN::Device::typeToName(core.getDeviceType());
  json["mode"] = CNN::Mode::typeToName(core.getModeType());

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
  json["cnnLayersConfig"] = cnnLayersArr;

  // Dense layers config
  nlohmann::ordered_json denseLayersArr = nlohmann::ordered_json::array();
  for (const auto& layer : core.getLayersConfig().denseLayers) {
    nlohmann::ordered_json layerJson;
    layerJson["numNeurons"] = layer.numNeurons;
    layerJson["actvFunc"] = ANN::ActvFunc::typeToName(layer.actvFuncType);
    denseLayersArr.push_back(layerJson);
  }
  json["denseLayersConfig"] = denseLayersArr;

  // Training config
  nlohmann::ordered_json tcJson;
  tcJson["numEpochs"] = core.getTrainingConfig().numEpochs;
  tcJson["learningRate"] = core.getTrainingConfig().learningRate;
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
  paramsJson["conv"] = convArr;

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