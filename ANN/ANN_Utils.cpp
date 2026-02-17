#include "ANN_Utils.hpp"
#include "ANN_Core.hpp"

#include <QDebug>
#include <QFile>

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
void Utils<T>::save(const Core<T>& core, const std::string& configFilePath) {
  QFile file(QString::fromStdString(configFilePath));

  if (!file.open(QIODevice::WriteOnly)) {
    throw std::runtime_error("Failed to open config file for writing: " + configFilePath);
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
  nlohmann::json json;

  // Save LayersConfig
  json["layersConfig"] = getLayersConfigJson(core.getLayersConfig());

  // Save TrainingConfig
  json["trainingConfig"] = getTrainingConfigJson(core.getTrainingConfig());

  // Save Parameters
  json["parameters"] = getParametersJson(core.getParameters());

  return json.dump(4);  // Pretty-print with 4 spaces indentation
}

//===================================================================================================================//

template <typename T>
nlohmann::json Utils<T>::getLayersConfigJson(const LayersConfig& layersConfig) {
  nlohmann::json layerConfigJsonArray = nlohmann::json::array();

  for (const Layer& layer : layersConfig) {
    ulong numNeurons = layer.numNeurons;
    std::string actvFuncName = ActvFunc::typeToName(layer.actvFuncType);

    layerConfigJsonArray.push_back({
        {"numNeurons", numNeurons},
        {"actvFunc", actvFuncName}
    });
  }

  return layerConfigJsonArray;
}

//===================================================================================================================//

template <typename T>
nlohmann::json Utils<T>::getTrainingConfigJson(const TrainingConfig<T>& trainingConfig) {
  nlohmann::json trainingConfigJsonObject;
  trainingConfigJsonObject["numEpochs"] = trainingConfig.numEpochs;
  trainingConfigJsonObject["learningRate"] = trainingConfig.learningRate;

  return trainingConfigJsonObject;
}

//===================================================================================================================//

template <typename T>
nlohmann::json Utils<T>::getParametersJson(const Parameters<T>& parameters) {
  return {
    {"weights", parameters.weights},
    {"biases", parameters.biases}
  };
}

//===================================================================================================================//

// Explicit template instantiations.
template class ANN::Utils<int>;
template class ANN::Utils<double>;
template class ANN::Utils<float>;
