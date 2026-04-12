#include "NN-CLI_TrainingSummary.hpp"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <variant>

namespace NN_CLI
{

  //===================================================================================================================//

  std::string TrainingSummary::formatWithCommas(ulong value)
  {
    std::string str = std::to_string(value);
    int insertPos = static_cast<int>(str.length()) - 3;

    while (insertPos > 0) {
      str.insert(insertPos, ",");
      insertPos -= 3;
    }

    return str;
  }

  //===================================================================================================================//

  ulong TrainingSummary::countCNNParameters(const CNN::CoreConfig<float>& config)
  {
    ulong total = 0;
    const auto& layers = config.layersConfig.cnnLayers;
    const auto& denseLayers = config.layersConfig.denseLayers;
    const CNN::Shape3D& inputShape = config.inputShape;

    ulong currentChannels = inputShape.c;
    ulong currentH = inputShape.h;
    ulong currentW = inputShape.w;
    ulong residualStartChannels = 0;

    for (const auto& layer : layers) {
      switch (layer.type) {
      case CNN::LayerType::CONV: {
        const auto& c = std::get<CNN::ConvLayerConfig>(layer.config);
        total += c.numFilters * (currentChannels * c.filterH * c.filterW) + c.numFilters;
        currentChannels = c.numFilters;

        if (c.slidingStrategy == CNN::SlidingStrategyType::SAME) {
          currentH = (currentH + c.strideY - 1) / c.strideY;
          currentW = (currentW + c.strideX - 1) / c.strideX;
        } else {
          currentH = (currentH - c.filterH) / c.strideY + 1;
          currentW = (currentW - c.filterW) / c.strideX + 1;
        }

        break;
      }

      case CNN::LayerType::INSTANCENORM:
      case CNN::LayerType::BATCHNORM:
        total += 2 * currentChannels; // gamma + beta
        break;
      case CNN::LayerType::POOL: {
        const auto& p = std::get<CNN::PoolLayerConfig>(layer.config);
        currentH = (currentH - p.poolH) / p.strideY + 1;
        currentW = (currentW - p.poolW) / p.strideX + 1;
        break;
      }

      case CNN::LayerType::GLOBALAVGPOOL:
        currentH = 1;
        currentW = 1;
        break;
      case CNN::LayerType::GLOBALDUALPOOL:
        currentH = 1;
        currentW = 1;
        currentChannels *= 2; // avg + max concatenated
        break;
      case CNN::LayerType::RESIDUAL_START:
        residualStartChannels = currentChannels;
        break;
      case CNN::LayerType::RESIDUAL_END:

        if (currentChannels != residualStartChannels) {
          total += currentChannels * residualStartChannels; // 1x1 projection
        }

        break;
      default:
        break;
      }
    }

    // Dense layers
    ulong annInputSize = currentChannels * currentH * currentW;

    for (const auto& dl : denseLayers) {
      total += dl.numNeurons * annInputSize + dl.numNeurons;
      annInputSize = dl.numNeurons;
    }

    return total;
  }

  //===================================================================================================================//

  void TrainingSummary::printCNN(const CNN::CoreConfig<float>& cnnConfig, const AugmentationConfig& augConfig,
                                 ulong trainSamples, ulong validationSamples, float validationRatio,
                                 bool validationAuto)
  {
    const auto& tc = cnnConfig.trainingConfig;
    const auto& layers = cnnConfig.layersConfig;
    const auto& inputShape = cnnConfig.inputShape;
    const auto& costConfig = cnnConfig.costFunctionConfig;

    // Count layer types
    ulong convCount = 0;
    ulong residualCount = 0;

    for (const auto& l : layers.cnnLayers) {
      if (l.type == CNN::LayerType::CONV)
        convCount++;
      else if (l.type == CNN::LayerType::RESIDUAL_START)
        residualCount++;
    }

    ulong denseCount = layers.denseLayers.size();
    ulong totalParams = countCNNParameters(cnnConfig);

    // Device string
    std::string deviceStr;

    if (cnnConfig.deviceType == CNN::DeviceType::GPU) {
      int gpus = cnnConfig.numGPUs;
      deviceStr = "GPU" + std::string(gpus > 0 ? " (" + std::to_string(gpus) + "x)" : "");
    } else {
      int threads = cnnConfig.numThreads;
      deviceStr = "CPU" + std::string(threads > 0 ? " (" + std::to_string(threads) + " threads)" : "");
    }

    // Cost function string
    std::string costStr;

    switch (costConfig.type) {
    case CNN::CostFunctionType::CROSS_ENTROPY:
      costStr = "Cross-entropy";
      break;
    case CNN::CostFunctionType::SQUARED_DIFFERENCE:
      costStr = "Squared difference";
      break;
    case CNN::CostFunctionType::WEIGHTED_SQUARED_DIFFERENCE:
      costStr = "Weighted squared difference";
      break;
    }

    // Optimizer string
    std::string optStr = CNN::Optimizer<float>::typeToName(tc.optimizer.type);
    optStr[0] = toupper(optStr[0]);

    // Augmentation string
    std::string augStr;

    if (augConfig.augmentationFactor > 0 || augConfig.balanceAugmentation || augConfig.augmentationProbability > 0) {
      std::vector<std::string> parts;

      if (augConfig.transforms.horizontalFlip)
        parts.push_back("flip");

      if (augConfig.transforms.rotation > 0)
        parts.push_back("rot " + std::to_string(static_cast<int>(augConfig.transforms.rotation)) + "\xC2\xB0");

      if (augConfig.transforms.translation > 0)
        parts.push_back("trans " + std::to_string(static_cast<int>(augConfig.transforms.translation * 100)) + "%");

      if (augConfig.transforms.brightness > 0)
        parts.push_back("bright " + std::to_string(static_cast<int>(augConfig.transforms.brightness * 100)) + "%");

      if (augConfig.transforms.contrast > 0)
        parts.push_back("contrast " + std::to_string(static_cast<int>(augConfig.transforms.contrast * 100)) + "%");

      if (augConfig.transforms.gaussianNoise > 0) {
        std::ostringstream oss;
        oss << "noise " << augConfig.transforms.gaussianNoise;
        parts.push_back(oss.str());
      }

      if (parts.empty()) {
        augStr = "None";
      } else {
        for (ulong i = 0; i < parts.size(); i++) {
          if (i > 0)
            augStr += ", ";
          augStr += parts[i];
        }
      }
    } else {
      augStr = "None";
    }

    // Class weights string
    std::string weightsStr;

    if (costConfig.weights.empty()) {
      weightsStr = "Uniform";
    } else {
      std::ostringstream oss;

      if (augConfig.autoClassWeights)
        oss << "Auto ";

      oss << "[";

      for (ulong i = 0; i < costConfig.weights.size(); i++) {
        if (i > 0)
          oss << ", ";
        oss << std::fixed << std::setprecision(2) << costConfig.weights[i];
      }

      oss << "]";
      weightsStr = oss.str();
    }

    // Validation string
    std::string validationStr;

    if (validationSamples > 0) {
      std::ostringstream oss;
      oss << formatWithCommas(validationSamples) << " (" << static_cast<int>(validationRatio * 100) << "%"
          << (validationAuto ? ", auto" : "") << ")";
      validationStr = oss.str();
    } else {
      validationStr = "Disabled";
    }

    // Print table
    const int keyW = 21;
    const int valW = 35;
    std::string sep = "+" + std::string(keyW + 2, '-') + "+" + std::string(valW + 2, '-') + "+";
    std::string titleLine(keyW + valW + 5, '-');

    std::cout << "\n";
    std::cout << "+" << titleLine << "+\n";
    std::cout << "|" << std::string((titleLine.size() - 22) / 2, ' ') << "Training Configuration"
              << std::string((titleLine.size() - 22 + 1) / 2, ' ') << "|\n";
    std::cout << sep << "\n";

    auto row = [&](const std::string& key, const std::string& val) {
      std::cout << "| " << std::left << std::setw(keyW) << key << " | " << std::setw(valW) << val << " |\n";
    };

    row("Device", deviceStr);
    row("Input shape",
        std::to_string(inputShape.c) + " x " + std::to_string(inputShape.h) + " x " + std::to_string(inputShape.w));
    row("Network type", "CNN");
    std::cout << sep << "\n";
    row("Conv layers", std::to_string(convCount));
    row("Dense layers", std::to_string(denseCount));
    row("Residual blocks", std::to_string(residualCount));
    row("Total parameters", formatWithCommas(totalParams));
    std::cout << sep << "\n";
    row("Training samples", formatWithCommas(trainSamples));
    row("Validation samples", validationStr);
    row("Augmentation", augStr);
    row("Class weights", weightsStr);
    std::cout << sep << "\n";
    row("Epochs", std::to_string(tc.numEpochs));
    row("Batch size", std::to_string(tc.batchSize));

    {
      std::ostringstream oss;
      oss << tc.learningRate;
      row("Learning rate", oss.str());
    }

    row("Optimizer", optStr);

    if (tc.dropoutRate > 0)
      row("Dropout", std::to_string(static_cast<int>(tc.dropoutRate * 100)) + "%");

    row("Cost function", costStr);
    row("Shuffle", tc.shuffleSamples ? "Yes" : "No");
    std::cout << sep << "\n";
    std::cout << "\n";
    std::cout.unsetf(std::ios_base::floatfield);
  }

  //===================================================================================================================//

  void TrainingSummary::printANN(const ANN::CoreConfig<float>& annConfig, const AugmentationConfig& augConfig,
                                 ulong trainSamples, ulong validationSamples, float validationRatio,
                                 bool validationAuto)
  {
    const auto& tc = annConfig.trainingConfig;
    const auto& costConfig = annConfig.costFunctionConfig;

    ulong denseCount = annConfig.layersConfig.size();

    // Device string
    std::string deviceStr;

    if (annConfig.deviceType == ANN::DeviceType::GPU) {
      int gpus = annConfig.numGPUs;
      deviceStr = "GPU" + std::string(gpus > 0 ? " (" + std::to_string(gpus) + "x)" : "");
    } else {
      int threads = annConfig.numThreads;
      deviceStr = "CPU" + std::string(threads > 0 ? " (" + std::to_string(threads) + " threads)" : "");
    }

    // Cost function string
    std::string costStr;

    switch (costConfig.type) {
    case ANN::CostFunctionType::CROSS_ENTROPY:
      costStr = "Cross-entropy";
      break;
    case ANN::CostFunctionType::SQUARED_DIFFERENCE:
      costStr = "Squared difference";
      break;
    case ANN::CostFunctionType::WEIGHTED_SQUARED_DIFFERENCE:
      costStr = "Weighted squared difference";
      break;
    }

    // Optimizer string
    std::string optStr = ANN::Optimizer<float>::typeToName(tc.optimizer.type);
    optStr[0] = toupper(optStr[0]);

    // Validation string
    std::string validationStr;

    if (validationSamples > 0) {
      std::ostringstream oss;
      oss << formatWithCommas(validationSamples) << " (" << static_cast<int>(validationRatio * 100) << "%"
          << (validationAuto ? ", auto" : "") << ")";
      validationStr = oss.str();
    } else {
      validationStr = "Disabled";
    }

    const int keyW = 21;
    const int valW = 35;
    std::string sep = "+" + std::string(keyW + 2, '-') + "+" + std::string(valW + 2, '-') + "+";
    std::string titleLine(keyW + valW + 5, '-');

    std::cout << "\n";
    std::cout << "+" << titleLine << "+\n";
    std::cout << "|" << std::string((titleLine.size() - 22) / 2, ' ') << "Training Configuration"
              << std::string((titleLine.size() - 22 + 1) / 2, ' ') << "|\n";
    std::cout << sep << "\n";

    auto row = [&](const std::string& key, const std::string& val) {
      std::cout << "| " << std::left << std::setw(keyW) << key << " | " << std::setw(valW) << val << " |\n";
    };

    row("Device", deviceStr);
    row("Network type", "ANN");
    std::cout << sep << "\n";
    row("Dense layers", std::to_string(denseCount));
    std::cout << sep << "\n";
    row("Training samples", formatWithCommas(trainSamples));
    row("Validation samples", validationStr);
    std::cout << sep << "\n";
    row("Epochs", std::to_string(tc.numEpochs));
    row("Batch size", std::to_string(tc.batchSize));

    {
      std::ostringstream oss;
      oss << tc.learningRate;
      row("Learning rate", oss.str());
    }

    row("Optimizer", optStr);

    if (tc.dropoutRate > 0)
      row("Dropout", std::to_string(static_cast<int>(tc.dropoutRate * 100)) + "%");

    row("Cost function", costStr);
    row("Shuffle", tc.shuffleSamples ? "Yes" : "No");
    std::cout << sep << "\n";
    std::cout << "\n";
    std::cout.unsetf(std::ios_base::floatfield);
  }

} // namespace NN_CLI
