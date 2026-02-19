#include "ANN_Utils.hpp"
#include "ANN_Core.hpp"

#include <QDebug>
#include <QFile>

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

using namespace ANN;

//===================================================================================================================//

template <typename T>
std::unique_ptr<Core<T>> Utils<T>::load(const std::string& configFilePath) {
  QFile file(QString::fromStdString(configFilePath));

  bool result = file.open(QIODevice::ReadOnly);

  if (!result) {
    throw std::runtime_error("Failed to open config file: " + configFilePath);
  }

  QByteArray fileData = file.readAll();
  std::string jsonString = fileData.toStdString();

  CoreConfig<T> coreConfig;

  try {
    nlohmann::json json = nlohmann::json::parse(jsonString);

    // Load core config (device, mode) - these have defaults if not present
    Utils::loadCoreConfig(json, coreConfig);

    coreConfig.layersConfig = Utils::loadLayersConfig(json);
    coreConfig.trainingConfig = Utils::loadTrainingConfig(json);
    coreConfig.parameters = Utils::loadParameters(json);

  } catch (const nlohmann::json::parse_error& e){
    qCritical() << "Config file JSON parse error: " << e.what();
  }

  return Core<T>::makeCore(coreConfig);
}

//===================================================================================================================//

template <typename T>
void Utils<T>::save(const Core<T>& core, const std::string& filePath) {
  QFile file(QString::fromStdString(filePath));

  if (!file.open(QIODevice::WriteOnly)) {
    throw std::runtime_error("Failed to open file for writing: " + filePath);
  }

  // Convert the core object to a JSON string
  std::string jsonString = save(core);

  // Write to file
  file.write(jsonString.c_str());
  file.close();
}

//===================================================================================================================//

template <typename T>
std::string Utils<T>::save(const Core<T>& core) {
  nlohmann::ordered_json json;

  // Save CoreConfig (device, mode) at the top level
  json["device"] = Device::typeToName(core.getDeviceType());
  json["mode"] = Mode::typeToName(core.getModeType());

  // Save LayersConfig
  json["layersConfig"] = getLayersConfigJson(core.getLayersConfig());

  // Save TrainingConfig
  json["trainingConfig"] = getTrainingConfigJson(core.getTrainingConfig());

  // Save TrainingMetadata (before parameters for readability)
  json["trainingMetadata"] = getTrainingMetadataJson(core.getTrainingMetadata());

  // Save Parameters
  json["parameters"] = getParametersJson(core.getParameters());

  return json.dump(4);  // Pretty-print with 4 spaces indentation
}

//===================================================================================================================//

template <typename T>
std::string Utils<T>::formatISO8601() {
  auto now = std::chrono::system_clock::now();
  std::time_t time = std::chrono::system_clock::to_time_t(now);
  std::tm* localTime = std::localtime(&time);

  std::ostringstream oss;
  oss << std::put_time(localTime, "%Y-%m-%dT%H:%M:%S");

  // Add UTC offset in ISO 8601 format (e.g., +01:00)
  // %z gives +0100, we need to insert the colon
  char tzOffset[8];
  std::strftime(tzOffset, sizeof(tzOffset), "%z", localTime);

  // Insert colon: "+0100" -> "+01:00"
  std::string offset(tzOffset);

  if (offset.length() >= 5) {
    offset.insert(3, ":");
  }

  oss << offset;

  return oss.str();
}

//===================================================================================================================//

template <typename T>
std::string Utils<T>::formatDuration(double totalSeconds) {
  // Handle negative or zero durations
  if (totalSeconds <= 0) {
    return "0s";
  }

  // Constants for time calculations
  constexpr int SECONDS_PER_MINUTE = 60;
  constexpr int SECONDS_PER_HOUR = 3600;
  constexpr int SECONDS_PER_DAY = 86400;
  constexpr int DAYS_PER_MONTH = 30;  // Approximate
  constexpr int DAYS_PER_YEAR = 365;  // Approximate

  // Calculate each unit
  long long totalSecs = static_cast<long long>(totalSeconds);

  long long years = totalSecs / (DAYS_PER_YEAR * SECONDS_PER_DAY);
  totalSecs %= (DAYS_PER_YEAR * SECONDS_PER_DAY);

  long long months = totalSecs / (DAYS_PER_MONTH * SECONDS_PER_DAY);
  totalSecs %= (DAYS_PER_MONTH * SECONDS_PER_DAY);

  long long days = totalSecs / SECONDS_PER_DAY;
  totalSecs %= SECONDS_PER_DAY;

  long long hours = totalSecs / SECONDS_PER_HOUR;
  totalSecs %= SECONDS_PER_HOUR;

  long long minutes = totalSecs / SECONDS_PER_MINUTE;
  long long seconds = totalSecs % SECONDS_PER_MINUTE;

  // Build the formatted string, only including non-zero units
  std::ostringstream oss;

  if (years > 0) {
    oss << years << "y ";
  }

  if (months > 0) {
    oss << months << "mo ";
  }

  if (days > 0) {
    oss << days << "d ";
  }

  if (hours > 0) {
    oss << hours << "h ";
  }

  if (minutes > 0) {
    oss << minutes << "m ";
  }

  if (seconds > 0 || oss.str().empty()) {
    oss << seconds << "s";
  }

  std::string result = oss.str();

  // Trim trailing space if present
  if (!result.empty() && result.back() == ' ') {
    result.pop_back();
  }

  return result;
}

//===================================================================================================================//

template <typename T>
void Utils<T>::loadCoreConfig(const nlohmann::json& json, CoreConfig<T>& coreConfig) {
  // Load device type (optional, defaults to CPU)
  if (json.contains("device")) {
    std::string deviceName = json.at("device").get<std::string>();
    coreConfig.deviceType = Device::nameToType(deviceName);
  } else {
    coreConfig.deviceType = DeviceType::CPU;
  }

  // Load mode type (optional, defaults to RUN)
  if (json.contains("mode")) {
    std::string modeName = json.at("mode").get<std::string>();
    coreConfig.modeType = Mode::nameToType(modeName);
  } else {
    coreConfig.modeType = ModeType::RUN;
  }
}

//===================================================================================================================//

template <typename T>
LayersConfig Utils<T>::loadLayersConfig(const nlohmann::json& json) {
  const nlohmann::json& layersConfigJsonArray = json.at("layersConfig");

  LayersConfig layersConfig;

  foreach (nlohmann::json layerJson, layersConfigJsonArray) {
    Layer layer;

    layer.numNeurons = layerJson.at("numNeurons").get<ulong>();

    std::string actvFuncName = layerJson.at("actvFunc").get<std::string>();
    layer.actvFuncType = ActvFunc::nameToType(actvFuncName);

    layersConfig.push_back(layer);
  }

  return layersConfig;
}

//===================================================================================================================//

template <typename T>
TrainingConfig<T> Utils<T>::loadTrainingConfig(const nlohmann::json& json) {
  TrainingConfig<T> trainingConfig;

  if (!json.contains("trainingConfig")) {
    return trainingConfig;
  }

  const nlohmann::json& trainingConfigJsonObject = json.at("trainingConfig");

  trainingConfig.numEpochs = trainingConfigJsonObject.at("numEpochs").get<ulong>();
  trainingConfig.learningRate = trainingConfigJsonObject.at("learningRate").get<float>();

  // numThreads is optional, defaults to 0 (use all available cores)
  if (trainingConfigJsonObject.contains("numThreads")) {
    trainingConfig.numThreads = trainingConfigJsonObject.at("numThreads").get<int>();
  }

  // progressReports is optional, defaults to 1000 reports per epoch
  if (trainingConfigJsonObject.contains("progressReports")) {
    trainingConfig.progressReports = trainingConfigJsonObject.at("progressReports").get<ulong>();
  }

  return trainingConfig;
}

//===================================================================================================================//

template <typename T>
Parameters<T> Utils<T>::loadParameters(const nlohmann::json& json) {
  Parameters<T> parameters;

  if (!json.contains("parameters")) {
    return parameters;
  }

  const nlohmann::json& parametersJsonObject = json.at("parameters");

  parameters.weights = parametersJsonObject.at("weights").get<Tensor3D<T>>();
  parameters.biases = parametersJsonObject.at("biases").get<Tensor2D<T>>();

  return parameters;
}

//===================================================================================================================//

template <typename T>
nlohmann::ordered_json Utils<T>::getLayersConfigJson(const LayersConfig& layersConfig) {
  nlohmann::ordered_json layerConfigJsonArray = nlohmann::ordered_json::array();

  for (const Layer& layer : layersConfig) {
    ulong numNeurons = layer.numNeurons;
    std::string actvFuncName = ActvFunc::typeToName(layer.actvFuncType);

    nlohmann::ordered_json layerJson;
    layerJson["numNeurons"] = numNeurons;
    layerJson["actvFunc"] = actvFuncName;
    layerConfigJsonArray.push_back(layerJson);
  }

  return layerConfigJsonArray;
}

//===================================================================================================================//

template <typename T>
nlohmann::ordered_json Utils<T>::getTrainingConfigJson(const TrainingConfig<T>& trainingConfig) {
  nlohmann::ordered_json trainingConfigJsonObject;
  trainingConfigJsonObject["numEpochs"] = trainingConfig.numEpochs;
  trainingConfigJsonObject["learningRate"] = trainingConfig.learningRate;

  return trainingConfigJsonObject;
}

//===================================================================================================================//

template <typename T>
nlohmann::ordered_json Utils<T>::getTrainingMetadataJson(const TrainingMetadata<T>& metadata) {
  nlohmann::ordered_json json;
  json["startTime"] = metadata.startTime;
  json["endTime"] = metadata.endTime;
  json["durationSeconds"] = metadata.durationSeconds;
  json["durationFormatted"] = metadata.durationFormatted;
  json["numSamples"] = metadata.numSamples;
  json["finalLoss"] = metadata.finalLoss;

  return json;
}

//===================================================================================================================//

template <typename T>
nlohmann::ordered_json Utils<T>::getParametersJson(const Parameters<T>& parameters) {
  nlohmann::ordered_json json;
  json["weights"] = parameters.weights;
  json["biases"] = parameters.biases;

  return json;
}

//===================================================================================================================//

// Explicit template instantiations.
template class ANN::Utils<int>;
template class ANN::Utils<double>;
template class ANN::Utils<float>;
