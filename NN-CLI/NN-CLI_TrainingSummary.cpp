#include "NN-CLI_TrainingSummary.hpp"
#include "NN-CLI_SummaryTable.hpp"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <variant>

namespace NN_CLI
{

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
                                 ulong numOriginalTrainSamples, ulong numTrainSamples, ulong numValidationSamples,
                                 float validationRatio, bool validationAuto)
  {
    auto lines = collectCNN(cnnConfig, augConfig, numOriginalTrainSamples, numTrainSamples, numValidationSamples,
                            validationRatio, validationAuto);

    for (const auto& l : lines) {
      if (!l.empty())
        std::cout << l << "\n";
      else
        std::cout << "\n";
    }
  }

  //===================================================================================================================//

  std::vector<std::string> TrainingSummary::collectCNN(const CNN::CoreConfig<float>& cnnConfig,
                                                       const AugmentationConfig& augConfig,
                                                       ulong numOriginalTrainSamples, ulong numTrainSamples,
                                                       ulong numValidationSamples, float validationRatio,
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

    if (augConfig.augmentationFactor > 0 || augConfig.balanceAugmentation || augConfig.fullAugmentation) {
      std::vector<std::string> parts;

      if (augConfig.fullAugmentation)
        parts.push_back("all-images");

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

      if (augConfig.transforms.randomErasing > 0)
        parts.push_back("erase " + std::to_string(static_cast<int>(augConfig.transforms.randomErasing * 100)) + "%");

      if (augConfig.transforms.hueShift > 0)
        parts.push_back("hue " + std::to_string(static_cast<int>(augConfig.transforms.hueShift * 100)) + "%");

      if (augConfig.transforms.scaling > 0)
        parts.push_back("scale " + std::to_string(static_cast<int>(augConfig.transforms.scaling * 100)) + "%");

      if (augConfig.transforms.elasticDeformation.alpha > 0)
        parts.push_back("elastic");

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

    if (numValidationSamples > 0) {
      std::ostringstream oss;
      oss << SummaryTable::formatWithCommas(numValidationSamples) << " (" << std::fixed << std::setprecision(2)
          << (validationRatio * 100) << "%" << (validationAuto ? ", auto" : "") << ")";
      validationStr = oss.str();
    } else {
      validationStr = "Disabled";
    }

    // Build rows
    std::string inputShapeStr =
      std::to_string(inputShape.c) + " x " + std::to_string(inputShape.h) + " x " + std::to_string(inputShape.w);

    std::ostringstream lrOss;
    lrOss << tc.learningRate;

    std::vector<SummaryRow> rows;
    rows.push_back({"Device", deviceStr});
    rows.push_back({"Input shape", inputShapeStr});
    rows.push_back({"Network type", "CNN"});
    rows.push_back({"", ""}); // separator
    rows.push_back({"Conv layers", std::to_string(convCount)});
    rows.push_back({"Dense layers", std::to_string(denseCount)});
    rows.push_back({"Residual blocks", std::to_string(residualCount)});
    rows.push_back({"Total parameters", SummaryTable::formatWithCommas(totalParams)});
    rows.push_back({"", ""}); // separator
    rows.push_back({"Total samples", SummaryTable::formatWithCommas(numOriginalTrainSamples + numValidationSamples)});

    if (numTrainSamples != numOriginalTrainSamples) {
      ulong numAugmented = numTrainSamples - numOriginalTrainSamples;
      rows.push_back({"Training samples", SummaryTable::formatWithCommas(numOriginalTrainSamples) + " + " +
                                            SummaryTable::formatWithCommas(numAugmented) +
                                            " augmented = " + SummaryTable::formatWithCommas(numTrainSamples)});
    } else {
      rows.push_back({"Training samples", SummaryTable::formatWithCommas(numTrainSamples)});
    }

    rows.push_back({"Validation samples", validationStr});

    if (augStr != "None")
      rows.push_back({"Augmentation", augStr});

    rows.push_back({"Class weights", weightsStr});
    rows.push_back({"", ""}); // separator
    rows.push_back({"Epochs", std::to_string(tc.numEpochs)});
    rows.push_back({"Batch size", std::to_string(tc.batchSize)});
    rows.push_back({"Learning rate", lrOss.str()});
    rows.push_back({"Optimizer", optStr});

    if (tc.dropoutRate > 0)
      rows.push_back({"Dropout", std::to_string(static_cast<int>(tc.dropoutRate * 100)) + "%"});

    rows.push_back({"Cost function", costStr});
    rows.push_back({"Shuffle", tc.shuffleSamples ? "Yes" : "No"});

    return SummaryTable::collect("Training Configuration", rows);
  }

  //===================================================================================================================//

  void TrainingSummary::printANN(const ANN::CoreConfig<float>& annConfig, const AugmentationConfig& augConfig,
                                 ulong numOriginalTrainSamples, ulong numTrainSamples, ulong numValidationSamples,
                                 float validationRatio, bool validationAuto)
  {
    auto lines = collectANN(annConfig, augConfig, numOriginalTrainSamples, numTrainSamples, numValidationSamples,
                            validationRatio, validationAuto);

    for (const auto& l : lines) {
      if (!l.empty())
        std::cout << l << "\n";
      else
        std::cout << "\n";
    }
  }

  //===================================================================================================================//

  std::vector<std::string> TrainingSummary::collectANN(const ANN::CoreConfig<float>& annConfig,
                                                       const AugmentationConfig& augConfig,
                                                       ulong numOriginalTrainSamples, ulong numTrainSamples,
                                                       ulong numValidationSamples, float validationRatio,
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

    if (numValidationSamples > 0) {
      std::ostringstream oss;
      oss << SummaryTable::formatWithCommas(numValidationSamples) << " (" << std::fixed << std::setprecision(2)
          << (validationRatio * 100) << "%" << (validationAuto ? ", auto" : "") << ")";
      validationStr = oss.str();
    } else {
      validationStr = "Disabled";
    }

    std::ostringstream lrOss;
    lrOss << tc.learningRate;

    std::vector<SummaryRow> rows;
    rows.push_back({"Device", deviceStr});
    rows.push_back({"Network type", "ANN"});
    rows.push_back({"", ""}); // separator
    rows.push_back({"Dense layers", std::to_string(denseCount)});
    rows.push_back({"", ""}); // separator
    rows.push_back({"Total samples", SummaryTable::formatWithCommas(numOriginalTrainSamples + numValidationSamples)});

    if (numTrainSamples != numOriginalTrainSamples) {
      ulong numAugmented = numTrainSamples - numOriginalTrainSamples;
      rows.push_back({"Training samples", SummaryTable::formatWithCommas(numOriginalTrainSamples) + " + " +
                                            SummaryTable::formatWithCommas(numAugmented) +
                                            " augmented = " + SummaryTable::formatWithCommas(numTrainSamples)});
    } else {
      rows.push_back({"Training samples", SummaryTable::formatWithCommas(numTrainSamples)});
    }

    rows.push_back({"Validation samples", validationStr});
    rows.push_back({"", ""}); // separator
    rows.push_back({"Epochs", std::to_string(tc.numEpochs)});
    rows.push_back({"Batch size", std::to_string(tc.batchSize)});
    rows.push_back({"Learning rate", lrOss.str()});
    rows.push_back({"Optimizer", optStr});

    if (tc.dropoutRate > 0)
      rows.push_back({"Dropout", std::to_string(static_cast<int>(tc.dropoutRate * 100)) + "%"});

    rows.push_back({"Cost function", costStr});
    rows.push_back({"Shuffle", tc.shuffleSamples ? "Yes" : "No"});

    return SummaryTable::collect("Training Configuration", rows);
  }

} // namespace NN_CLI
