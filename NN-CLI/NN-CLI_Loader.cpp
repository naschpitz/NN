#include "NN-CLI_Loader.hpp"

#include <CNN_Utils.hpp>

#include <QFile>
#include <json.hpp>

#include <stdexcept>

namespace NN_CLI {

//===================================================================================================================//
// Network type detection
//===================================================================================================================//

NetworkType Loader::detectNetworkType(const std::string& configFilePath) {
    QFile file(QString::fromStdString(configFilePath));

    if (!file.open(QIODevice::ReadOnly)) {
        throw std::runtime_error("Failed to open config file: " + configFilePath);
    }

    QByteArray fileData = file.readAll();
    nlohmann::json json = nlohmann::json::parse(fileData.toStdString());

    // CNN configs have "inputShape" and/or "cnnLayersConfig"
    if (json.contains("inputShape") || json.contains("cnnLayersConfig")) {
        return NetworkType::CNN;
    }

    return NetworkType::ANN;
}

//===================================================================================================================//
// ANN config loading (unchanged logic, renamed)
//===================================================================================================================//

ANN::CoreConfig<float> Loader::loadANNConfig(const std::string& configFilePath,
                                               std::optional<ANN::ModeType> modeType,
                                               std::optional<ANN::DeviceType> deviceType) {
    QFile file(QString::fromStdString(configFilePath));

    if (!file.open(QIODevice::ReadOnly)) {
        throw std::runtime_error("Failed to open config file: " + configFilePath);
    }

    QByteArray fileData = file.readAll();
    nlohmann::json json = nlohmann::json::parse(fileData.toStdString());

    ANN::CoreConfig<float> coreConfig;

    if (json.contains("device")) {
        coreConfig.deviceType = ANN::Device::nameToType(json.at("device").get<std::string>());
    } else {
        coreConfig.deviceType = ANN::DeviceType::CPU;
    }

    if (json.contains("mode")) {
        coreConfig.modeType = ANN::Mode::nameToType(json.at("mode").get<std::string>());
    } else {
        coreConfig.modeType = ANN::ModeType::PREDICT;
    }

    if (modeType.has_value()) coreConfig.modeType = modeType.value();
    if (deviceType.has_value()) coreConfig.deviceType = deviceType.value();

    if (!json.contains("layersConfig")) {
        throw std::runtime_error("Config file missing 'layersConfig': " + configFilePath);
    }

    for (const auto& layerJson : json.at("layersConfig")) {
        ANN::Layer layer;
        layer.numNeurons = layerJson.at("numNeurons").get<ulong>();
        layer.actvFuncType = ANN::ActvFunc::nameToType(layerJson.at("actvFunc").get<std::string>());
        coreConfig.layersConfig.push_back(layer);
    }

    if (json.contains("trainingConfig")) {
        const auto& tc = json.at("trainingConfig");
        coreConfig.trainingConfig.numEpochs = tc.at("numEpochs").get<ulong>();
        coreConfig.trainingConfig.learningRate = tc.at("learningRate").get<float>();
        if (tc.contains("numThreads")) coreConfig.trainingConfig.numThreads = tc.at("numThreads").get<int>();
        if (tc.contains("progressReports")) coreConfig.trainingConfig.progressReports = tc.at("progressReports").get<ulong>();
    }

    if (json.contains("parameters")) {
        const auto& p = json.at("parameters");
        coreConfig.parameters.weights = p.at("weights").get<ANN::Tensor3D<float>>();
        coreConfig.parameters.biases = p.at("biases").get<ANN::Tensor2D<float>>();
    }

    bool isPredictOrTest = (coreConfig.modeType == ANN::ModeType::PREDICT || coreConfig.modeType == ANN::ModeType::TEST);
    if (isPredictOrTest && !json.contains("parameters")) {
        throw std::runtime_error("Config file missing 'parameters' required for predict/test modes: " + configFilePath);
    }

    return coreConfig;
}

//===================================================================================================================//
// CNN config loading
//===================================================================================================================//

CNN::CoreConfig<float> Loader::loadCNNConfig(const std::string& configFilePath,
                                               std::optional<std::string> modeOverride,
                                               std::optional<std::string> deviceOverride) {
    QFile file(QString::fromStdString(configFilePath));

    if (!file.open(QIODevice::ReadOnly)) {
        throw std::runtime_error("Failed to open config file: " + configFilePath);
    }

    QByteArray fileData = file.readAll();
    nlohmann::json json = nlohmann::json::parse(fileData.toStdString());

    CNN::CoreConfig<float> coreConfig;

    // Device
    if (deviceOverride.has_value()) {
        coreConfig.deviceType = CNN::Device::nameToType(deviceOverride.value());
    } else if (json.contains("device")) {
        coreConfig.deviceType = CNN::Device::nameToType(json.at("device").get<std::string>());
    } else {
        coreConfig.deviceType = CNN::DeviceType::CPU;
    }

    // Mode
    if (modeOverride.has_value()) {
        coreConfig.modeType = CNN::Mode::nameToType(modeOverride.value());
    } else if (json.contains("mode")) {
        coreConfig.modeType = CNN::Mode::nameToType(json.at("mode").get<std::string>());
    } else {
        coreConfig.modeType = CNN::ModeType::PREDICT;
    }

    // Input shape (required for CNN)
    if (!json.contains("inputShape")) {
        throw std::runtime_error("CNN config file missing 'inputShape': " + configFilePath);
    }
    const auto& shapeJson = json.at("inputShape");
    coreConfig.inputShape.c = shapeJson.at("c").get<ulong>();
    coreConfig.inputShape.h = shapeJson.at("h").get<ulong>();
    coreConfig.inputShape.w = shapeJson.at("w").get<ulong>();

    // CNN layers
    if (json.contains("cnnLayersConfig")) {
        for (const auto& layerJson : json.at("cnnLayersConfig")) {
            std::string type = layerJson.at("type").get<std::string>();
            CNN::CNNLayerConfig layerConfig;

            if (type == "conv") {
                layerConfig.type = CNN::LayerType::CONV;
                CNN::ConvLayerConfig conv;
                conv.numFilters = layerJson.at("numFilters").get<ulong>();
                conv.filterH = layerJson.at("filterH").get<ulong>();
                conv.filterW = layerJson.at("filterW").get<ulong>();
                conv.strideY = layerJson.at("strideY").get<ulong>();
                conv.strideX = layerJson.at("strideX").get<ulong>();
                conv.slidingStrategy = CNN::SlidingStrategy::nameToType(layerJson.at("slidingStrategy").get<std::string>());
                layerConfig.config = conv;
            } else if (type == "relu") {
                layerConfig.type = CNN::LayerType::RELU;
                layerConfig.config = CNN::ReLULayerConfig{};
            } else if (type == "pool") {
                layerConfig.type = CNN::LayerType::POOL;
                CNN::PoolLayerConfig pool;
                pool.poolType = CNN::PoolType::nameToType(layerJson.at("poolType").get<std::string>());
                pool.poolH = layerJson.at("poolH").get<ulong>();
                pool.poolW = layerJson.at("poolW").get<ulong>();
                pool.strideY = layerJson.at("strideY").get<ulong>();
                pool.strideX = layerJson.at("strideX").get<ulong>();
                layerConfig.config = pool;
            } else if (type == "flatten") {
                layerConfig.type = CNN::LayerType::FLATTEN;
                layerConfig.config = CNN::FlattenLayerConfig{};
            } else {
                throw std::runtime_error("Unknown CNN layer type: " + type);
            }

            coreConfig.layersConfig.cnnLayers.push_back(layerConfig);
        }
    }

    // Dense layers
    if (json.contains("denseLayersConfig")) {
        for (const auto& layerJson : json.at("denseLayersConfig")) {
            CNN::DenseLayerConfig dense;
            dense.numNeurons = layerJson.at("numNeurons").get<ulong>();
            dense.actvFuncType = ANN::ActvFunc::nameToType(layerJson.at("actvFunc").get<std::string>());
            coreConfig.layersConfig.denseLayers.push_back(dense);
        }
    }

    // Training config
    if (json.contains("trainingConfig")) {
        const auto& tc = json.at("trainingConfig");
        coreConfig.trainingConfig.numEpochs = tc.at("numEpochs").get<ulong>();
        coreConfig.trainingConfig.learningRate = tc.at("learningRate").get<float>();
        if (tc.contains("numThreads")) coreConfig.trainingConfig.numThreads = tc.at("numThreads").get<int>();
        if (tc.contains("progressReports")) coreConfig.trainingConfig.progressReports = tc.at("progressReports").get<ulong>();
    }

    // Parameters (for predict/test modes or resuming training)
    if (json.contains("parameters")) {
        const auto& paramsJson = json.at("parameters");

        if (paramsJson.contains("conv")) {
            for (const auto& convJson : paramsJson.at("conv")) {
                CNN::ConvParameters<float> cp;
                cp.numFilters = convJson.at("numFilters").get<ulong>();
                cp.inputC = convJson.at("inputC").get<ulong>();
                cp.filterH = convJson.at("filterH").get<ulong>();
                cp.filterW = convJson.at("filterW").get<ulong>();
                cp.filters = convJson.at("filters").get<std::vector<float>>();
                cp.biases = convJson.at("biases").get<std::vector<float>>();
                coreConfig.parameters.convParams.push_back(std::move(cp));
            }
        }

        if (paramsJson.contains("dense")) {
            const auto& denseJson = paramsJson.at("dense");
            coreConfig.parameters.denseParams.weights = denseJson.at("weights").get<ANN::Tensor3D<float>>();
            coreConfig.parameters.denseParams.biases = denseJson.at("biases").get<ANN::Tensor2D<float>>();
        }
    }

    bool isPredictOrTest = (coreConfig.modeType == CNN::ModeType::PREDICT || coreConfig.modeType == CNN::ModeType::TEST);
    if (isPredictOrTest && !json.contains("parameters")) {
        throw std::runtime_error("CNN config file missing 'parameters' required for predict/test modes: " + configFilePath);
    }

    return coreConfig;
}

//===================================================================================================================//
// Sample and input loading
//===================================================================================================================//

ANN::Samples<float> Loader::loadANNSamples(const std::string& samplesFilePath) {
    QFile file(QString::fromStdString(samplesFilePath));

    if (!file.open(QIODevice::ReadOnly)) {
        throw std::runtime_error("Failed to open samples file: " + samplesFilePath);
    }

    QByteArray fileData = file.readAll();
    nlohmann::json json = nlohmann::json::parse(fileData.toStdString());

    ANN::Samples<float> samples;
    for (const auto& sampleJson : json.at("samples")) {
        ANN::Sample<float> sample;
        sample.input = sampleJson.at("input").get<std::vector<float>>();
        sample.output = sampleJson.at("output").get<std::vector<float>>();
        samples.push_back(sample);
    }
    return samples;
}

CNN::Samples<float> Loader::loadCNNSamples(const std::string& samplesFilePath, const CNN::Shape3D& inputShape) {
    return CNN::Utils<float>::loadSamples(samplesFilePath, inputShape);
}

ANN::Input<float> Loader::loadANNInput(const std::string& inputFilePath) {
    QFile file(QString::fromStdString(inputFilePath));

    if (!file.open(QIODevice::ReadOnly)) {
        throw std::runtime_error("Failed to open input file: " + inputFilePath);
    }

    QByteArray fileData = file.readAll();
    nlohmann::json json = nlohmann::json::parse(fileData.toStdString());
    return json.at("input").get<std::vector<float>>();
}

CNN::Input<float> Loader::loadCNNInput(const std::string& inputFilePath, const CNN::Shape3D& inputShape) {
    QFile file(QString::fromStdString(inputFilePath));

    if (!file.open(QIODevice::ReadOnly)) {
        throw std::runtime_error("Failed to open input file: " + inputFilePath);
    }

    QByteArray fileData = file.readAll();
    nlohmann::json json = nlohmann::json::parse(fileData.toStdString());

    std::vector<float> flatInput = json.at("input").get<std::vector<float>>();

    if (flatInput.size() != inputShape.size()) {
        throw std::runtime_error("Input size (" + std::to_string(flatInput.size()) +
          ") does not match expected input shape size (" + std::to_string(inputShape.size()) + ")");
    }

    CNN::Input<float> input(inputShape);
    input.data = std::move(flatInput);
    return input;
}

} // namespace NN_CLI

