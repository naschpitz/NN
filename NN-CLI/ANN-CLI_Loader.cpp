#include "ANN-CLI_Loader.hpp"

#include <QFile>
#include <json.hpp>

#include <stdexcept>

namespace ANN_CLI {

ANN::CoreConfig<float> Loader::loadConfig(const std::string& configFilePath,
                                           std::optional<ANN::ModeType> modeType,
                                           std::optional<ANN::DeviceType> deviceType) {
    QFile file(QString::fromStdString(configFilePath));

    if (!file.open(QIODevice::ReadOnly)) {
        throw std::runtime_error("Failed to open config file: " + configFilePath);
    }

    QByteArray fileData = file.readAll();
    std::string jsonString = fileData.toStdString();

    nlohmann::json json = nlohmann::json::parse(jsonString);

    ANN::CoreConfig<float> coreConfig;

    // Load device type from JSON (optional, defaults to CPU)
    if (json.contains("device")) {
        std::string deviceName = json.at("device").get<std::string>();
        coreConfig.deviceType = ANN::Device::nameToType(deviceName);
    } else {
        coreConfig.deviceType = ANN::DeviceType::CPU;
    }

    // Load mode type from JSON (optional, defaults to RUN)
    if (json.contains("mode")) {
        std::string modeName = json.at("mode").get<std::string>();
        coreConfig.modeType = ANN::Mode::nameToType(modeName);
    } else {
        coreConfig.modeType = ANN::ModeType::RUN;
    }

    // Override with CLI arguments if explicitly provided
    if (modeType.has_value()) {
        coreConfig.modeType = modeType.value();
    }
    if (deviceType.has_value()) {
        coreConfig.deviceType = deviceType.value();
    }

    // Load layers config (required for all modes)
    if (!json.contains("layersConfig")) {
        throw std::runtime_error("Config file missing 'layersConfig': " + configFilePath);
    }

    const nlohmann::json& layersConfigJson = json.at("layersConfig");

    for (const auto& layerJson : layersConfigJson) {
        ANN::Layer layer;
        layer.numNeurons = layerJson.at("numNeurons").get<ulong>();
        std::string actvFuncName = layerJson.at("actvFunc").get<std::string>();
        layer.actvFuncType = ANN::ActvFunc::nameToType(actvFuncName);
        coreConfig.layersConfig.push_back(layer);
    }

    // Load training config (optional - only relevant for TRAIN mode)
    if (json.contains("trainingConfig")) {
        const nlohmann::json& trainingConfigJson = json.at("trainingConfig");
        coreConfig.trainingConfig.numEpochs = trainingConfigJson.at("numEpochs").get<ulong>();
        coreConfig.trainingConfig.learningRate = trainingConfigJson.at("learningRate").get<float>();

        // numThreads is optional, defaults to 0 (use all available cores)
        if (trainingConfigJson.contains("numThreads")) {
            coreConfig.trainingConfig.numThreads = trainingConfigJson.at("numThreads").get<int>();
        }

        // progressReports is optional, defaults to 1000 reports per epoch
        if (trainingConfigJson.contains("progressReports")) {
            coreConfig.trainingConfig.progressReports = trainingConfigJson.at("progressReports").get<ulong>();
        }
    }

    // Load parameters (optional for TRAIN mode to allow resuming, required for RUN/TEST modes)
    if (json.contains("parameters")) {
        const nlohmann::json& parametersJson = json.at("parameters");
        coreConfig.parameters.weights = parametersJson.at("weights").get<ANN::Tensor3D<float>>();
        coreConfig.parameters.biases = parametersJson.at("biases").get<ANN::Tensor2D<float>>();
    }

    // Validate: RUN and TEST modes require parameters (trained weights/biases)
    bool isRunOrTestMode = (coreConfig.modeType == ANN::ModeType::RUN ||
                            coreConfig.modeType == ANN::ModeType::TEST);

    if (isRunOrTestMode && !json.contains("parameters")) {
        throw std::runtime_error("Config file missing 'parameters' (weights/biases) required for run/test modes: " + configFilePath);
    }

    return coreConfig;
}

ANN::Samples<float> Loader::loadSamples(const std::string& samplesFilePath) {
    QFile file(QString::fromStdString(samplesFilePath));

    if (!file.open(QIODevice::ReadOnly)) {
        throw std::runtime_error("Failed to open samples file: " + samplesFilePath);
    }

    QByteArray fileData = file.readAll();
    std::string jsonString = fileData.toStdString();

    nlohmann::json json = nlohmann::json::parse(jsonString);

    ANN::Samples<float> samples;

    for (const auto& sampleJson : json.at("samples")) {
        ANN::Sample<float> sample;
        sample.input = sampleJson.at("input").get<std::vector<float>>();
        sample.output = sampleJson.at("output").get<std::vector<float>>();
        samples.push_back(sample);
    }

    return samples;
}

ANN::Input<float> Loader::loadInput(const std::string& inputFilePath) {
    QFile file(QString::fromStdString(inputFilePath));

    if (!file.open(QIODevice::ReadOnly)) {
        throw std::runtime_error("Failed to open input file: " + inputFilePath);
    }

    QByteArray fileData = file.readAll();
    std::string jsonString = fileData.toStdString();

    nlohmann::json json = nlohmann::json::parse(jsonString);

    ANN::Input<float> input = json.at("input").get<std::vector<float>>();

    return input;
}

} // namespace ANN_CLI

