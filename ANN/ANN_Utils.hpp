#ifndef ANN_UTILS_HPP
#define ANN_UTILS_HPP

#include <json.hpp>
#include "ANN_Core.hpp"

namespace ANN {
  template <typename T>
  class Utils
  {
    public:
      static Core<T> load(const std::string& configFilePath);
      static void save(const Core<T>& core, const std::string& configFilePath);

      static std::string save(const Core<T>& core);

    private:
      static LayersConfig loadLayersConfig(const nlohmann::json& json);
      static TrainingConfig<T> loadTrainingConfig(const nlohmann::json& json);
      static Parameters<T> loadParameters(const nlohmann::json& json);

      static nlohmann::json getLayersConfigJson(const LayersConfig& layersConfig);
      static nlohmann::json getTrainingConfigJson(const TrainingConfig<T>& trainingConfig);
      static nlohmann::json getParametersJson(const Parameters<T>& parameters);
  };
}

#endif // ANN_UTILS_HPP
