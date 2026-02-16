#ifndef ANN_UTILS_HPP
#define ANN_UTILS_HPP

#include <json.hpp>
#include "ANN_Core.hpp"

#include <vector>
#include <type_traits>

//===================================================================================================================//

namespace ANN {
  template <typename T>
  class Utils
  {
    public:
      static std::unique_ptr<Core<T>> load(const std::string& configFilePath);
      static void save(const Core<T>& core, const std::string& configFilePath);

      static std::string save(const Core<T>& core);

      template <typename V>
      static ulong count(const V& nestedVec) {
        ulong result = 0;

        Utils<T>::countHelper(result, nestedVec);

        return result;
      }

      template <typename V>
      static Tensor1D<T> flatten(const V& nestedVec) {
        Tensor1D<T> result;

        Utils<T>::flattenHelper(result, nestedVec);

        return result;
      }

    private:
      static LayersConfig loadLayersConfig(const nlohmann::json& json);
      static TrainingConfig<T> loadTrainingConfig(const nlohmann::json& json);
      static Parameters<T> loadParameters(const nlohmann::json& json);

      static nlohmann::json getLayersConfigJson(const LayersConfig& layersConfig);
      static nlohmann::json getTrainingConfigJson(const TrainingConfig<T>& trainingConfig);
      static nlohmann::json getParametersJson(const Parameters<T>& parameters);

      template <typename V>
      static void flattenHelper(Tensor1D<T>& result, const V& nestedVec) {
        for (auto nestedVecItem : nestedVec) {
          if constexpr (std::is_same_v<std::decay_t<decltype(nestedVecItem)>, std::vector<T>>) {
            Utils<T>::flattenHelper(result, nestedVecItem);
          } else {
            result.push_back(nestedVecItem);
          }
        }
      }

      template <typename V>
      static void countHelper(ulong& result, const V& nestedVec) {
        for (auto nestedVecItem : nestedVec) {
          if constexpr (std::is_same_v<std::decay_t<decltype(nestedVecItem)>, std::vector<T>>) {
            Utils<T>::countHelper(result, nestedVecItem);
          } else {
            result++;
          }
        }
      }
  };
}

#endif // ANN_UTILS_HPP
