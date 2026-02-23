#include "NN-CLI_Loader.hpp"
#include "NN-CLI_ImageLoader.hpp"
#include "NN-CLI_ProgressBar.hpp"

#include <QFile>
#include <QFileInfo>
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
// I/O config loading
//===================================================================================================================//

IOConfig Loader::loadIOConfig(const std::string& configFilePath,
                               std::optional<std::string> inputTypeOverride,
                               std::optional<std::string> outputTypeOverride) {
    QFile file(QString::fromStdString(configFilePath));

    if (!file.open(QIODevice::ReadOnly)) {
        throw std::runtime_error("Failed to open config file: " + configFilePath);
    }

    QByteArray fileData = file.readAll();
    nlohmann::json json = nlohmann::json::parse(fileData.toStdString());

    IOConfig ioConfig;

    // Read inputType / outputType (default to "vector")
    if (json.contains("inputType")) {
        ioConfig.inputType = dataTypeFromString(json.at("inputType").get<std::string>());
    }
    if (json.contains("outputType")) {
        ioConfig.outputType = dataTypeFromString(json.at("outputType").get<std::string>());
    }

    // CLI overrides
    if (inputTypeOverride.has_value()) {
        ioConfig.inputType = dataTypeFromString(inputTypeOverride.value());
    }
    if (outputTypeOverride.has_value()) {
        ioConfig.outputType = dataTypeFromString(outputTypeOverride.value());
    }

    // Input shape (for ANN image input — CNN uses CoreConfig.inputShape)
    if (json.contains("inputShape")) {
        const auto& s = json.at("inputShape");
        ioConfig.inputC = s.at("c").get<ulong>();
        ioConfig.inputH = s.at("h").get<ulong>();
        ioConfig.inputW = s.at("w").get<ulong>();
    }

    // Output shape (for image output reconstruction)
    if (json.contains("outputShape")) {
        const auto& s = json.at("outputShape");
        ioConfig.outputC = s.at("c").get<ulong>();
        ioConfig.outputH = s.at("h").get<ulong>();
        ioConfig.outputW = s.at("w").get<ulong>();
    }

    return ioConfig;
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

ANN::Samples<float> Loader::loadANNSamples(const std::string& samplesFilePath,
                                             const IOConfig& ioConfig,
                                             ulong progressReports) {
    QFile file(QString::fromStdString(samplesFilePath));

    if (!file.open(QIODevice::ReadOnly)) {
        throw std::runtime_error("Failed to open samples file: " + samplesFilePath);
    }

    QByteArray fileData = file.readAll();
    nlohmann::json json = nlohmann::json::parse(fileData.toStdString());

    // Resolve base directory for relative image paths
    std::string baseDir = QFileInfo(QString::fromStdString(samplesFilePath)).absolutePath().toStdString();

    const auto& samplesArray = json.at("samples");
    size_t totalSamples = samplesArray.size();

    ANN::Samples<float> samples;
    samples.reserve(totalSamples);
    size_t idx = 0;

    for (const auto& sampleJson : samplesArray) {
        ANN::Sample<float> sample;

        // Input
        if (ioConfig.inputType == DataType::IMAGE) {
            if (!ioConfig.hasInputShape()) {
                throw std::runtime_error("inputType is 'image' but no inputShape provided in config.");
            }
            std::string imgPath = ImageLoader::resolvePath(
                sampleJson.at("input").get<std::string>(), baseDir);
            sample.input = ImageLoader::loadImage(imgPath,
                static_cast<int>(ioConfig.inputC),
                static_cast<int>(ioConfig.inputH),
                static_cast<int>(ioConfig.inputW));
        } else {
            sample.input = sampleJson.at("input").get<std::vector<float>>();
        }

        // Output
        if (ioConfig.outputType == DataType::IMAGE) {
            if (!ioConfig.hasOutputShape()) {
                throw std::runtime_error("outputType is 'image' but no outputShape provided in config.");
            }
            std::string imgPath = ImageLoader::resolvePath(
                sampleJson.at("output").get<std::string>(), baseDir);
            sample.output = ImageLoader::loadImage(imgPath,
                static_cast<int>(ioConfig.outputC),
                static_cast<int>(ioConfig.outputH),
                static_cast<int>(ioConfig.outputW));
        } else {
            sample.output = sampleJson.at("output").get<std::vector<float>>();
        }

        samples.push_back(std::move(sample));
        ProgressBar::printLoadingProgress("Loading samples:", ++idx, totalSamples, progressReports);
    }
    return samples;
}

//===================================================================================================================//

CNN::Samples<float> Loader::loadCNNSamples(const std::string& samplesFilePath,
                                             const CNN::Shape3D& inputShape,
                                             const IOConfig& ioConfig,
                                             ulong progressReports) {
    QFile file(QString::fromStdString(samplesFilePath));

    if (!file.open(QIODevice::ReadOnly)) {
        throw std::runtime_error("Failed to open samples file: " + samplesFilePath);
    }

    QByteArray fileData = file.readAll();
    nlohmann::json json = nlohmann::json::parse(fileData.toStdString());

    std::string baseDir = QFileInfo(QString::fromStdString(samplesFilePath)).absolutePath().toStdString();

    const nlohmann::json& samplesArray = json.at("samples");
    size_t totalSamples = samplesArray.size();

    CNN::Samples<float> samples;
    samples.reserve(totalSamples);
    size_t idx = 0;

    for (const auto& sampleJson : samplesArray) {
        CNN::Sample<float> sample;

        // Input
        if (ioConfig.inputType == DataType::IMAGE) {
            std::string imgPath = ImageLoader::resolvePath(
                sampleJson.at("input").get<std::string>(), baseDir);
            std::vector<float> flatInput = ImageLoader::loadImage(imgPath,
                static_cast<int>(inputShape.c),
                static_cast<int>(inputShape.h),
                static_cast<int>(inputShape.w));
            sample.input = CNN::Input<float>(inputShape);
            sample.input.data = std::move(flatInput);
        } else {
            std::vector<float> flatInput = sampleJson.at("input").get<std::vector<float>>();
            if (flatInput.size() != inputShape.size()) {
                throw std::runtime_error("Sample input size (" + std::to_string(flatInput.size()) +
                  ") does not match expected input shape size (" + std::to_string(inputShape.size()) + ")");
            }
            sample.input = CNN::Input<float>(inputShape);
            sample.input.data = std::move(flatInput);
        }

        // Output
        if (ioConfig.outputType == DataType::IMAGE) {
            if (!ioConfig.hasOutputShape()) {
                throw std::runtime_error("outputType is 'image' but no outputShape provided in config.");
            }
            std::string imgPath = ImageLoader::resolvePath(
                sampleJson.at("output").get<std::string>(), baseDir);
            sample.output = ImageLoader::loadImage(imgPath,
                static_cast<int>(ioConfig.outputC),
                static_cast<int>(ioConfig.outputH),
                static_cast<int>(ioConfig.outputW));
        } else {
            sample.output = sampleJson.at("output").get<CNN::Output<float>>();
        }

        samples.push_back(std::move(sample));
        ProgressBar::printLoadingProgress("Loading samples:", ++idx, totalSamples, progressReports);
    }

    return samples;
}

//===================================================================================================================//

std::vector<ANN::Input<float>> Loader::loadANNInputs(const std::string& inputFilePath,
                                                       const IOConfig& ioConfig,
                                                       ulong progressReports) {
    QFile file(QString::fromStdString(inputFilePath));

    if (!file.open(QIODevice::ReadOnly)) {
        throw std::runtime_error("Failed to open input file: " + inputFilePath);
    }

    QByteArray fileData = file.readAll();
    nlohmann::json json = nlohmann::json::parse(fileData.toStdString());

    const auto& inputsArray = json.at("inputs");
    if (!inputsArray.is_array() || inputsArray.empty()) {
        throw std::runtime_error("'inputs' must be a non-empty array in: " + inputFilePath);
    }

    std::string baseDir = QFileInfo(QString::fromStdString(inputFilePath)).absolutePath().toStdString();
    size_t totalInputs = inputsArray.size();
    std::vector<ANN::Input<float>> inputs;
    inputs.reserve(totalInputs);
    size_t idx = 0;

    for (const auto& entry : inputsArray) {
        if (ioConfig.inputType == DataType::IMAGE) {
            if (!ioConfig.hasInputShape()) {
                throw std::runtime_error("inputType is 'image' but no inputShape provided in config.");
            }
            std::string imgPath = ImageLoader::resolvePath(entry.get<std::string>(), baseDir);
            inputs.push_back(ImageLoader::loadImage(imgPath,
                static_cast<int>(ioConfig.inputC),
                static_cast<int>(ioConfig.inputH),
                static_cast<int>(ioConfig.inputW)));
        } else {
            inputs.push_back(entry.get<std::vector<float>>());
        }
        ProgressBar::printLoadingProgress("Loading inputs:", ++idx, totalInputs, progressReports);
    }

    return inputs;
}

//===================================================================================================================//

std::vector<CNN::Input<float>> Loader::loadCNNInputs(const std::string& inputFilePath,
                                                       const CNN::Shape3D& inputShape,
                                                       const IOConfig& ioConfig,
                                                       ulong progressReports) {
    QFile file(QString::fromStdString(inputFilePath));

    if (!file.open(QIODevice::ReadOnly)) {
        throw std::runtime_error("Failed to open input file: " + inputFilePath);
    }

    QByteArray fileData = file.readAll();
    nlohmann::json json = nlohmann::json::parse(fileData.toStdString());

    const auto& inputsArray = json.at("inputs");
    if (!inputsArray.is_array() || inputsArray.empty()) {
        throw std::runtime_error("'inputs' must be a non-empty array in: " + inputFilePath);
    }

    std::string baseDir = QFileInfo(QString::fromStdString(inputFilePath)).absolutePath().toStdString();
    size_t totalInputs = inputsArray.size();
    std::vector<CNN::Input<float>> inputs;
    inputs.reserve(totalInputs);
    size_t idx = 0;

    for (const auto& entry : inputsArray) {
        std::vector<float> flatInput;

        if (ioConfig.inputType == DataType::IMAGE) {
            std::string imgPath = ImageLoader::resolvePath(entry.get<std::string>(), baseDir);
            flatInput = ImageLoader::loadImage(imgPath,
                static_cast<int>(inputShape.c),
                static_cast<int>(inputShape.h),
                static_cast<int>(inputShape.w));
        } else {
            flatInput = entry.get<std::vector<float>>();
        }

        if (flatInput.size() != inputShape.size()) {
            throw std::runtime_error("Input size (" + std::to_string(flatInput.size()) +
              ") does not match expected input shape size (" + std::to_string(inputShape.size()) + ")");
        }

        CNN::Input<float> input(inputShape);
        input.data = std::move(flatInput);
        inputs.push_back(std::move(input));
        ProgressBar::printLoadingProgress("Loading inputs:", ++idx, totalInputs, progressReports);
    }

    return inputs;
}

//===================================================================================================================//
// progressReports loading
//===================================================================================================================//

ulong Loader::loadProgressReports(const std::string& configFilePath) {
    QFile file(QString::fromStdString(configFilePath));

    if (!file.open(QIODevice::ReadOnly)) {
        throw std::runtime_error("Failed to open config file: " + configFilePath);
    }

    QByteArray fileData = file.readAll();
    nlohmann::json json = nlohmann::json::parse(fileData.toStdString());

    if (json.contains("progressReports")) {
        return json.at("progressReports").get<ulong>();
    }

    return 1000; // default
}

//===================================================================================================================//
// saveModelInterval loading
//===================================================================================================================//

ulong Loader::loadSaveModelInterval(const std::string& configFilePath) {
    QFile file(QString::fromStdString(configFilePath));

    if (!file.open(QIODevice::ReadOnly)) {
        throw std::runtime_error("Failed to open config file: " + configFilePath);
    }

    QByteArray fileData = file.readAll();
    nlohmann::json json = nlohmann::json::parse(fileData.toStdString());

    if (json.contains("saveModelInterval")) {
        return json.at("saveModelInterval").get<ulong>();
    }

    return 0; // disabled by default
}

//===================================================================================================================//

} // namespace NN_CLI

