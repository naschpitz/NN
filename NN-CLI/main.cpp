#include <QCoreApplication>
#include <QCommandLineParser>
#include <QCommandLineOption>
#include <QFile>
#include <QDebug>

#include <ANN_Core.hpp>
#include <ANN_CoreMode.hpp>
#include <ANN_CoreType.hpp>
#include <ANN_ActvFunc.hpp>
#include <ANN_LayersConfig.hpp>
#include <ANN_Utils.hpp>

#include <json.hpp>

#include <iostream>
#include <memory>

void printUsage() {
    std::cout << "ANN-CLI - Artificial Neural Network Command Line Interface\n\n";
    std::cout << "Usage:\n";
    std::cout << "  ANN-CLI --config <file> --mode <train|run> [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --config, -c <file>    Path to JSON configuration file\n";
    std::cout << "  --mode, -m <mode>      Mode: 'train' or 'run'\n";
    std::cout << "  --type, -t <type>      Core type: 'cpu' or 'gpu' (default: cpu)\n";
    std::cout << "  --input, -i <values>   Input values for run mode (comma-separated)\n";
    std::cout << "  --samples, -s <file>   Path to JSON file with training samples\n";
    std::cout << "  --output, -o <file>    Output file for saving trained model\n";
    std::cout << "  --help, -h             Show this help message\n";
}

ANN::CoreConfig<float> loadConfig(const std::string& configFilePath,
                                   ANN::CoreModeType modeType,
                                   ANN::CoreTypeType coreType) {
    QFile file(QString::fromStdString(configFilePath));

    if (!file.open(QIODevice::ReadOnly)) {
        throw std::runtime_error("Failed to open config file: " + configFilePath);
    }

    QByteArray fileData = file.readAll();
    std::string jsonString = fileData.toStdString();

    nlohmann::json json = nlohmann::json::parse(jsonString);

    ANN::CoreConfig<float> coreConfig;
    coreConfig.coreModeType = modeType;
    coreConfig.coreTypeType = coreType;

    // Load layers config
    const nlohmann::json& layersConfigJson = json.at("layersConfig");
    for (const auto& layerJson : layersConfigJson) {
        ANN::Layer layer;
        layer.numNeurons = layerJson.at("numNeurons").get<ulong>();
        std::string actvFuncName = layerJson.at("actvFunc").get<std::string>();
        layer.actvFuncType = ANN::ActvFunc::nameToType(actvFuncName);
        coreConfig.layersConfig.push_back(layer);
    }

    // Load training config (optional)
    if (json.contains("trainingConfig")) {
        const nlohmann::json& trainingConfigJson = json.at("trainingConfig");
        coreConfig.trainingConfig.numEpochs = trainingConfigJson.at("numEpochs").get<ulong>();
        coreConfig.trainingConfig.learningRate = trainingConfigJson.at("learningRate").get<float>();
    }

    // Load parameters (optional - for pre-trained models)
    if (json.contains("parameters")) {
        const nlohmann::json& parametersJson = json.at("parameters");
        coreConfig.parameters.weights = parametersJson.at("weights").get<ANN::Tensor3D<float>>();
        coreConfig.parameters.biases = parametersJson.at("biases").get<ANN::Tensor2D<float>>();
    }

    return coreConfig;
}

ANN::Samples<float> loadSamples(const std::string& samplesFilePath) {
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

ANN::Input<float> parseInput(const QString& inputStr) {
    ANN::Input<float> input;
    QStringList values = inputStr.split(',');

    for (const QString& val : values) {
        input.push_back(val.toFloat());
    }

    return input;
}

int main(int argc, char *argv[])
{
    QCoreApplication app(argc, argv);
    QCoreApplication::setApplicationName("ANN-CLI");
    QCoreApplication::setApplicationVersion("1.0");

    QCommandLineParser parser;
    parser.setApplicationDescription("Artificial Neural Network CLI");
    parser.addHelpOption();
    parser.addVersionOption();

    // Config file option
    QCommandLineOption configOption(
        QStringList() << "c" << "config",
        "Path to JSON configuration file.",
        "file"
    );
    parser.addOption(configOption);

    // Mode option (train or run)
    QCommandLineOption modeOption(
        QStringList() << "m" << "mode",
        "Mode: 'train' or 'run'.",
        "mode"
    );
    parser.addOption(modeOption);

    // Core type option (cpu or gpu)
    QCommandLineOption typeOption(
        QStringList() << "t" << "type",
        "Core type: 'cpu' or 'gpu' (default: cpu).",
        "type",
        "cpu"
    );
    parser.addOption(typeOption);

    // Input values for run mode
    QCommandLineOption inputOption(
        QStringList() << "i" << "input",
        "Input values for run mode (comma-separated).",
        "values"
    );
    parser.addOption(inputOption);

    // Samples file for training
    QCommandLineOption samplesOption(
        QStringList() << "s" << "samples",
        "Path to JSON file with training samples.",
        "file"
    );
    parser.addOption(samplesOption);

    // Output file for saving trained model
    QCommandLineOption outputOption(
        QStringList() << "o" << "output",
        "Output file for saving trained model.",
        "file"
    );
    parser.addOption(outputOption);

    parser.process(app);

    // Validate required options
    if (!parser.isSet(configOption)) {
        std::cerr << "Error: --config option is required.\n\n";
        printUsage();
        return 1;
    }

    if (!parser.isSet(modeOption)) {
        std::cerr << "Error: --mode option is required.\n\n";
        printUsage();
        return 1;
    }

    QString configPath = parser.value(configOption);
    QString mode = parser.value(modeOption).toLower();
    QString coreTypeStr = parser.value(typeOption).toLower();

    if (mode != "train" && mode != "run") {
        std::cerr << "Error: Mode must be 'train' or 'run'.\n";
        return 1;
    }

    if (coreTypeStr != "cpu" && coreTypeStr != "gpu") {
        std::cerr << "Error: Type must be 'cpu' or 'gpu'.\n";
        return 1;
    }

    // Convert mode string to enum
    ANN::CoreModeType modeType = (mode == "train") ? ANN::CoreModeType::TRAIN : ANN::CoreModeType::RUN;

    // Convert core type string to enum
    ANN::CoreTypeType coreType = (coreTypeStr == "gpu") ? ANN::CoreTypeType::GPU : ANN::CoreTypeType::CPU;

    try {
        // Load the ANN configuration from JSON file
        std::cout << "Loading configuration from: " << configPath.toStdString() << "\n";
        std::cout << "Mode: " << mode.toStdString() << ", Core type: " << coreTypeStr.toStdString() << "\n";

        ANN::CoreConfig<float> coreConfig = loadConfig(configPath.toStdString(), modeType, coreType);
        auto core = ANN::Core<float>::makeCore(coreConfig);

        if (mode == "train") {
            // Training mode
            if (!parser.isSet(samplesOption)) {
                std::cerr << "Error: --samples option is required for training mode.\n";
                return 1;
            }

            QString samplesPath = parser.value(samplesOption);
            std::cout << "Loading training samples from: " << samplesPath.toStdString() << "\n";

            ANN::Samples<float> samples = loadSamples(samplesPath.toStdString());
            std::cout << "Loaded " << samples.size() << " training samples.\n";

            std::cout << "Starting training...\n";
            core->train(samples);
            std::cout << "Training completed.\n";

            // Save the trained model if output file specified
            if (parser.isSet(outputOption)) {
                QString outputPath = parser.value(outputOption);
                ANN::Utils<float>::save(*core, outputPath.toStdString());
                std::cout << "Model saved to: " << outputPath.toStdString() << "\n";
            }

        } else if (mode == "run") {
            // Run mode
            if (!parser.isSet(inputOption)) {
                std::cerr << "Error: --input option is required for run mode.\n";
                return 1;
            }

            ANN::Input<float> input = parseInput(parser.value(inputOption));
            std::cout << "Running ANN with input: ";
            for (size_t i = 0; i < input.size(); ++i) {
                std::cout << input[i];
                if (i < input.size() - 1) std::cout << ", ";
            }
            std::cout << "\n";

            ANN::Output<float> output = core->run(input);

            std::cout << "Output: ";
            for (size_t i = 0; i < output.size(); ++i) {
                std::cout << output[i];
                if (i < output.size() - 1) std::cout << ", ";
            }
            std::cout << "\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
