#ifndef ANN_ACTVFUNC_H
#define ANN_ACTVFUNC_H

#include <cmath>
#include <string>
#include <unordered_map>

namespace ANN {
  enum class ActvFuncType {
    RELU,
    SIGMOID,
    TANH,
    SOFTMAX
  };

  const std::unordered_map<std::string, ActvFuncType> actvMap = {
    {"relu", ActvFuncType::RELU},
    {"sigmoid", ActvFuncType::SIGMOID},
    {"tanh", ActvFuncType::TANH},
    {"softmax", ActvFuncType::SOFTMAX}
  };

  class ActvFunc {
    public:
      static ActvFuncType nameToType(const std::string& name);
      static std::string typeToName(const ActvFuncType& actvFuncType);

      // Per-neuron (relu, sigmoid, tanh)
      static float calculate(float x, ActvFuncType type, bool derivative = false);

      // Layer-wide (all types including softmax)
      // Forward  (derivative=false): reads zs, writes actvs. dCost_dActvs and dCost_dZs unused.
      // Backward (derivative=true) : reads zs, actvs, dCost_dActvs, writes dCost_dZs.
      template <typename T>
      static void calculate(const T* zs, T* actvs, unsigned long numNeurons, ActvFuncType type,
                            bool derivative, const T* dCost_dActvs, T* dCost_dZs);

    private:
      static float relu(float x);
      static float sigmoid(float x);
      static float tanh(float x);

      static float drelu(float x);
      static float dsigmoid(float x);
      static float dtanh(float x);
  };

  //===================================================================================================================//
  // Template implementation (must be in header)
  //===================================================================================================================//

  template <typename T>
  void ActvFunc::calculate(const T* zs, T* actvs, unsigned long numNeurons, ActvFuncType type,
                            bool derivative, const T* dCost_dActvs, T* dCost_dZs) {
    if (!derivative) {
      // === Forward pass ===
      if (type == ActvFuncType::SOFTMAX) {
        // Softmax: layer-wide activation with numerical stability
        T maxZ = zs[0];
        for (unsigned long j = 1; j < numNeurons; j++) {
          if (zs[j] > maxZ) maxZ = zs[j];
        }

        T sumExp = 0;
        for (unsigned long j = 0; j < numNeurons; j++) {
          actvs[j] = std::exp(zs[j] - maxZ);
          sumExp += actvs[j];
        }

        for (unsigned long j = 0; j < numNeurons; j++) {
          actvs[j] /= sumExp;
        }
      } else {
        // Element-wise activation (relu, sigmoid, tanh)
        for (unsigned long j = 0; j < numNeurons; j++) {
          actvs[j] = static_cast<T>(ActvFunc::calculate(static_cast<float>(zs[j]), type));
        }
      }
    } else {
      // === Backward pass: compute dCost/dZ ===
      if (type == ActvFuncType::SOFTMAX) {
        // Softmax Jacobian: dCost/dZ_j = s_j * (dCost/dActv_j - dot)
        // where dot = Σ_i s_i * dCost/dActv_i and s = softmax output
        T dot = 0;
        for (unsigned long j = 0; j < numNeurons; j++) {
          dot += actvs[j] * dCost_dActvs[j];
        }

        for (unsigned long j = 0; j < numNeurons; j++) {
          dCost_dZs[j] = actvs[j] * (dCost_dActvs[j] - dot);
        }
      } else {
        // Element-wise: dCost/dZ_j = dActvFunc(z_j) * dCost/dActv_j
        for (unsigned long j = 0; j < numNeurons; j++) {
          T dActv_dZ = static_cast<T>(ActvFunc::calculate(static_cast<float>(zs[j]), type, true));
          dCost_dZs[j] = dActv_dZ * dCost_dActvs[j];
        }
      }
    }
  }
}

#endif // ANN_ACTVFUNC_H
