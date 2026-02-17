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

      // Unflatten a 1D vector into a 2D tensor (for biases) using the shape template
      static void unflatten(const Tensor1D<T>& flat, Tensor2D<T>& target) {
        ulong idx = 0;

        for (auto& row : target) {
          for (auto& elem : row) {
            elem = flat[idx++];
          }
        }
      }

      // Unflatten a 1D vector into a 3D tensor (for weights) using the shape template
      static void unflatten(const Tensor1D<T>& flat, Tensor3D<T>& target) {
        ulong idx = 0;
        
        for (auto& layer : target) {
          for (auto& row : layer) {
            for (auto& elem : row) {
              elem = flat[idx++];
            }
          }
        }
      }

    private:
      static LayersConfig loadLayersConfig(const nlohmann::json& json);
      static TrainingConfig<T> loadTrainingConfig(const nlohmann::json& json);
      static Parameters<T> loadParameters(const nlohmann::json& json);

      static nlohmann::json getLayersConfigJson(const LayersConfig& layersConfig);
      static nlohmann::json getTrainingConfigJson(const TrainingConfig<T>& trainingConfig);
      static nlohmann::json getParametersJson(const Parameters<T>& parameters);

      // Helper to detect if a type is a std::vector
      template <typename U>
      struct is_vector : std::false_type {};

      template <typename U, typename A>
      struct is_vector<std::vector<U, A>> : std::true_type {};

      template <typename V>
      static void flattenHelper(Tensor1D<T>& result, const V& nestedVec) {
        for (const auto& nestedVecItem : nestedVec) {
          if constexpr (is_vector<std::decay_t<decltype(nestedVecItem)>>::value) {
            Utils<T>::flattenHelper(result, nestedVecItem);
          } else {
            result.push_back(nestedVecItem);
          }
        }
      }

      template <typename V>
      static void countHelper(ulong& result, const V& nestedVec) {
        for (const auto& nestedVecItem : nestedVec) {
          if constexpr (is_vector<std::decay_t<decltype(nestedVecItem)>>::value) {
            Utils<T>::countHelper(result, nestedVecItem);
          } else {
            result++;
          }
        }
      }
  };
}

#endif // ANN_UTILS_HPP
