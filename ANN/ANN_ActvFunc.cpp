#include "ANN_ActvFunc.hpp"

#include <cmath>
#include <stdexcept>

using namespace ANN;

//===================================================================================================================//

ActvFuncType ActvFunc::nameToType(const std::string& name) {
  auto it = actvMap.find(name);

  if (it == actvMap.end()) {
    return ActvFuncType::UNKNOWN;
  } else {
    return it->second;
  }
}

//===================================================================================================================//

std::string ActvFunc::typeToName(const ActvFuncType& actvFuncType) {
  for (const auto& pair : actvMap) {
    if (pair.second == actvFuncType) {
      return pair.first;
    }
  }

  return "unknown"; // Default return value for unknown types
}

//===================================================================================================================//

float ActvFunc::calculate(float x, ActvFuncType type, bool isDerivative) {
  switch(type) {
    case ActvFuncType::RELU:
      return !isDerivative ? ActvFunc::relu(x) : ActvFunc::drelu(x);
    case ActvFuncType::SIGMOID:
      return !isDerivative ? ActvFunc::sigmoid(x) : ActvFunc::dsigmoid(x);
    case ActvFuncType::TANH:
      return !isDerivative ? ActvFunc::tanh(x) : ActvFunc::dtanh(x);
    case ActvFuncType::SOFTMAX:
      throw std::invalid_argument("Softmax is a layer-wide function; use the layer-wide calculate() overload instead");
    default:
      throw std::invalid_argument("Unknown actv function");
  }
}

//===================================================================================================================//

float ActvFunc::relu(float x) {
  return (x > 0) ? x : 0;
}

//===================================================================================================================//

// Function to calculate Sigmoid
float ActvFunc::sigmoid(float x) {
  return 1.0 / (1.0 + std::exp(-x));
}

//===================================================================================================================//

// Function to calculate Tanh
float ActvFunc::tanh(float x) {
  return std::tanh(x);  // std::tanh() is the mathematical tanh function
}

//===================================================================================================================//

float ActvFunc::drelu(float x) {
  return (x > 0) ? 1.0 : 0.0;
}

//===================================================================================================================//

// Derivative of Sigmoid
float ActvFunc::dsigmoid(float x) {
  float sig = ActvFunc::sigmoid(x);
  return sig * (1.0 - sig);
}

//===================================================================================================================//

// Derivative of Tanh
float ActvFunc::dtanh(float x) {
  float tanh_x = std::tanh(x);
  return 1.0 - (tanh_x * tanh_x);
}

//===================================================================================================================//
