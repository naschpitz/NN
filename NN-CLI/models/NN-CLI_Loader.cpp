#include "NN-CLI_Loader.hpp"

#include <QFile>
#include <json.hpp>

#include <cstdint>
#include <stdexcept>

namespace NN_CLI
{

  //===================================================================================================================//

  nlohmann::json Loader::parseConfigFile(const std::string& configFilePath)
  {
    QFile file(QString::fromStdString(configFilePath));

    if (!file.open(QIODevice::ReadOnly)) {
      throw std::runtime_error("Failed to open config file: " + configFilePath);
    }

    QByteArray fileData = file.readAll();
    return nlohmann::json::parse(fileData.toStdString());
  }

  //===================================================================================================================//

  NetworkType Loader::detectNetworkType(const std::string& configFilePath)
  {
    return detectNetworkType(parseConfigFile(configFilePath));
  }

  NetworkType Loader::detectNetworkType(const nlohmann::json& json)
  {
    //  configs use "layers" for their dense layers.
    // CNN configs use "convolutionalLayers" and/or "denseLayers".
    // "inputShape" is NOT used for detection — both types can have it (e.g.  image input).
    if (json.contains("layers")) {
      return NetworkType::ANN;
    }

    return NetworkType::CNN;
  }

  //===================================================================================================================//

  IOConfig Loader::loadIOConfig(const std::string& configFilePath, std::optional<std::string> inputTypeOverride,
                                std::optional<std::string> outputTypeOverride)
  {
    return loadIOConfig(parseConfigFile(configFilePath), inputTypeOverride, outputTypeOverride);
  }

  IOConfig Loader::loadIOConfig(const nlohmann::json& json, std::optional<std::string> inputTypeOverride,
                                std::optional<std::string> outputTypeOverride)
  {
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

    // Input shape (for  image input — CNN uses CoreConfig.inputShape)
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

  ulong Loader::loadProgressReports(const std::string& configFilePath)
  {
    return loadProgressReports(parseConfigFile(configFilePath));
  }

  ulong Loader::loadProgressReports(const nlohmann::json& json)
  {
    if (json.contains("progressReports")) {
      return json.at("progressReports").get<ulong>();
    }

    return 1000; // default
  }

  //===================================================================================================================//

  ulong Loader::loadSaveModelInterval(const std::string& configFilePath)
  {
    return loadSaveModelInterval(parseConfigFile(configFilePath));
  }

  ulong Loader::loadSaveModelInterval(const nlohmann::json& json)
  {
    if (json.contains("saveModelInterval")) {
      return json.at("saveModelInterval").get<ulong>();
    }

    return 10; // default: save every 10 epochs
  }

  //===================================================================================================================//

  AugmentationConfig Loader::loadAugmentationConfig(const std::string& configFilePath)
  {
    return loadAugmentationConfig(parseConfigFile(configFilePath));
  }

  AugmentationConfig Loader::loadAugmentationConfig(const nlohmann::json& json)
  {
    AugmentationConfig config;

    if (json.contains("train")) {
      const auto& tc = json.at("train");

      if (tc.contains("augmentationFactor"))
        config.augmentationFactor = tc.at("augmentationFactor").get<ulong>();

      if (tc.contains("balanceAugmentation"))
        config.balanceAugmentation = tc.at("balanceAugmentation").get<bool>();

      if (tc.contains("fullAugmentation"))
        config.fullAugmentation = tc.at("fullAugmentation").get<bool>();

      if (tc.contains("autoClassWeights"))
        config.autoClassWeights = tc.at("autoClassWeights").get<bool>();

      if (tc.contains("augmentationProbability"))
        config.augmentationProbability = tc.at("augmentationProbability").get<float>();

      if (tc.contains("augmentationTransforms")) {
        const auto& at = tc.at("augmentationTransforms");
        auto& t = config.transforms;

        if (at.contains("horizontalFlip"))
          t.horizontalFlip = at.at("horizontalFlip").get<bool>();

        if (at.contains("rotation"))
          t.rotation = at.at("rotation").get<float>();

        if (at.contains("translation"))
          t.translation = at.at("translation").get<float>();

        if (at.contains("brightness"))
          t.brightness = at.at("brightness").get<float>();

        if (at.contains("contrast"))
          t.contrast = at.at("contrast").get<float>();

        if (at.contains("gaussianNoise"))
          t.gaussianNoise = at.at("gaussianNoise").get<float>();

        if (at.contains("randomErasing"))
          t.randomErasing = at.at("randomErasing").get<float>();

        if (at.contains("hueShift"))
          t.hueShift = at.at("hueShift").get<float>();

        if (at.contains("scaling"))
          t.scaling = at.at("scaling").get<float>();

        if (at.contains("elasticDeformation")) {
          const auto& ed = at.at("elasticDeformation");

          if (ed.contains("alpha"))
            t.elasticDeformation.alpha = ed.at("alpha").get<float>();

          if (ed.contains("sigma"))
            t.elasticDeformation.sigma = ed.at("sigma").get<float>();
        }
      }

      if (tc.contains("validation")) {
        const auto& vd = tc.at("validation");
        auto& vc = config.validationConfig;

        if (vd.contains("enabled"))
          vc.enabled = vd.at("enabled").get<bool>();

        if (vd.contains("autoSize"))
          vc.autoSize = vd.at("autoSize").get<bool>();

        if (vd.contains("size"))
          vc.size = vd.at("size").get<float>();

        if (vd.contains("checkInterval"))
          vc.checkInterval = vd.at("checkInterval").get<ulong>();
      }
    }

    return config;
  }

  //===================================================================================================================//

  Common::TrainConfig<float> Loader::loadTrainConfig(const nlohmann::json& json)
  {
    Common::TrainConfig<float> tc;

    if (json.contains("train")) {
      const auto& train = json.at("train");

      tc.numEpochs = train.at("numEpochs").get<ulong>();
      tc.learningRate = train.at("learningRate").get<float>();

      if (train.contains("batchSize"))
        tc.batchSize = train.at("batchSize").get<ulong>();

      if (train.contains("fetchSize"))
        tc.fetchSize = train.at("fetchSize").get<ulong>();

      if (train.contains("shuffleSamples"))
        tc.shuffleSamples = train.at("shuffleSamples").get<bool>();

      if (train.contains("shuffleSeed"))
        tc.shuffleSeed = train.at("shuffleSeed").get<uint32_t>();

      if (train.contains("dropoutRate"))
        tc.dropoutRate = train.at("dropoutRate").get<float>();

      if (train.contains("optimizer")) {
        const auto& opt = train.at("optimizer");

        if (opt.contains("type"))
          tc.optimizer.type = Common::optimizerNameToType(opt.at("type").get<std::string>());

        if (opt.contains("beta1"))
          tc.optimizer.beta1 = opt.at("beta1").get<float>();

        if (opt.contains("beta2"))
          tc.optimizer.beta2 = opt.at("beta2").get<float>();

        if (opt.contains("epsilon"))
          tc.optimizer.epsilon = opt.at("epsilon").get<float>();
      }

      if (train.contains("learningRateScheduler")) {
        const auto& sched = train.at("learningRateScheduler");

        tc.learningRateScheduler.type =
          Common::LearningRateSchedulerConfig::nameToType(sched.at("type").get<std::string>());
        tc.learningRateScheduler.gamma = sched.value("gamma", tc.learningRateScheduler.gamma);
        tc.learningRateScheduler.stepSize = sched.value("stepSize", tc.learningRateScheduler.stepSize);
        tc.learningRateScheduler.minLearningRate =
          sched.value("minLearningRate", tc.learningRateScheduler.minLearningRate);
        tc.learningRateScheduler.patience = sched.value("patience", tc.learningRateScheduler.patience);
        tc.learningRateScheduler.minDelta = sched.value("minDelta", tc.learningRateScheduler.minDelta);
      }

      if (train.contains("monitoring")) {
        const auto& mon = train.at("monitoring");
        auto& mc = tc.monitoringConfig;

        if (mon.contains("enabled"))
          mc.enabled = mon.at("enabled").get<bool>();

        if (mon.contains("checkInterval"))
          mc.checkInterval = mon.at("checkInterval").get<ulong>();

        if (mon.contains("patience"))
          mc.patience = mon.at("patience").get<ulong>();

        if (mon.contains("metrics")) {
          const auto& metrics = mon.at("metrics");

          if (metrics.contains("lossStagnation")) {
            const auto& ls = metrics.at("lossStagnation");

            if (ls.contains("enabled"))
              mc.metrics.lossStagnation.enabled = ls.at("enabled").get<bool>();

            if (ls.contains("minDelta"))
              mc.metrics.lossStagnation.minDelta = ls.at("minDelta").get<float>();
          }

          if (metrics.contains("lossExplosion")) {
            const auto& le = metrics.at("lossExplosion");

            if (le.contains("enabled"))
              mc.metrics.lossExplosion.enabled = le.at("enabled").get<bool>();

            if (le.contains("threshold"))
              mc.metrics.lossExplosion.threshold = le.at("threshold").get<float>();
          }
        }
      }
    }

    return tc;
  }

  //===================================================================================================================//

  Common::TestConfig Loader::loadTestConfig(const nlohmann::json& json)
  {
    Common::TestConfig config;

    if (json.contains("test")) {
      const auto& test = json.at("test");

      if (test.contains("fetchSize"))
        config.fetchSize = test.at("fetchSize").get<ulong>();
    }

    return config;
  }

  //===================================================================================================================//

  Common::CalibrateConfig Loader::loadCalibrateConfig(const nlohmann::json& json)
  {
    Common::CalibrateConfig config;

    if (json.contains("calibrate")) {
      const auto& cal = json.at("calibrate");

      if (cal.contains("idSampleCount"))
        config.idSampleCount = cal.at("idSampleCount").get<std::size_t>();

      if (cal.contains("oodSampleCount"))
        config.oodSampleCount = cal.at("oodSampleCount").get<std::size_t>();

      if (cal.contains("idPercentile"))
        config.idPercentile = cal.at("idPercentile").get<double>();

      if (cal.contains("fetchIfMissing"))
        config.fetchIfMissing = cal.at("fetchIfMissing").get<bool>();
    }

    return config;
  }

  //===================================================================================================================//

  Common::TrainMetadata<float> Loader::loadTrainMetadata(const nlohmann::json& json)
  {
    Common::TrainMetadata<float> md;

    if (json.contains("trainMetadata")) {
      const auto& meta = json.at("trainMetadata");

      md.startTime = meta.value("startTime", "");
      md.endTime = meta.value("endTime", "");
      md.durationSeconds = meta.value("durationSeconds", 0.0);
      md.durationFormatted = meta.value("durationFormatted", "");
      md.numSamples = meta.value("numSamples", 0UL);
      md.finalLoss = static_cast<float>(meta.value("finalLoss", 0.0));
      md.lastEpoch = meta.value("lastEpoch", 0UL);
      md.stopReason = meta.value("stopReason", "");
      md.bestEpoch = meta.value("bestEpoch", 0UL);
      md.bestLoss = static_cast<float>(meta.value("bestLoss", 0.0));

      if (meta.contains("learningRateSchedulerState")) {
        const auto& ss = meta.at("learningRateSchedulerState");

        md.learningRateSchedulerState.currentLearningRate = ss.value("currentLearningRate", 0.0f);
        md.learningRateSchedulerState.baseLearningRate = ss.value("baseLearningRate", 0.0f);
        md.learningRateSchedulerState.epochsSinceImprovement = ss.value("epochsSinceImprovement", 0UL);
        md.learningRateSchedulerState.bestValidationLoss = ss.value("bestValidationLoss", 0.0f);
        md.learningRateSchedulerState.initialized = ss.value("initialized", false);
      }

      if (meta.contains("epochs") && meta.at("epochs").is_array()) {
        const auto& epochsArr = meta.at("epochs");

        md.epochHistory.reserve(epochsArr.size());

        for (const auto& recordJson : epochsArr) {
          Common::EpochRecord<float> record;

          record.epoch = recordJson.at("epoch").get<ulong>();
          record.loss = recordJson.at("loss").get<float>();
          record.learningRate = recordJson.value("learningRate", 0.0f);

          if (recordJson.contains("valLoss") && recordJson.value("hasValLoss", false)) {
            record.valLoss = recordJson.at("valLoss").get<float>();
            record.hasValLoss = true;
          }

          record.isBest = recordJson.value("isBest", false);
          record.completionTime = recordJson.value("completionTime", 0UL);

          md.epochHistory.push_back(record);
        }
      }
    }

    return md;
  }

  //===================================================================================================================//

} // namespace NN_CLI
