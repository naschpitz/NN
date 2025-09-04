#ifndef ACTVFUNC_CPP_CL
#define ACTVFUNC_CPP_CL

#include "../Defines.hpp.cl"

//===================================================================================================================//

TYPE actvFunc_calculate(TYPE z, ActvFuncType type, bool isDerivative) {
  switch(type) {
    case ActvFuncType::RELU:
      return !isDerivative ? actvFunc_relu(x) : actvFunc_drelu(x);
    case ActvFuncType::SIGMOID:
      return !isDerivative ? actvFunc_sigmoid(x) : actvFunc_dsigmoid(x);
    case ActvFuncType::TANH:
      return !isDerivative ? actvFunc_tanh(x) : actvFunc_dtanh(x);
  }
}

//===================================================================================================================//

TYPE actvFunc_relu(TYPE x) {
  return (x > 0) ? x : 0;
}

//===================================================================================================================//

// Function to calculate Sigmoid
TYPE actvFunc_sigmoid(TYPE x) {
  return 1.0 / (1.0 + exp(-x));
}

//===================================================================================================================//

// Function to calculate Tanh
TYPE actvFunc_tanh(TYPE x) {
  return tanh(x);  // std::tanh() is the mathematical tanh function
}

//===================================================================================================================//

TYPE actvFunc_drelu(TYPE x) {
  return (x > 0) ? 1.0 : 0.0;
}

//===================================================================================================================//

// Derivative of Sigmoid
TYPE actvFunc_dsigmoid(TYPE x) {
  TYPE sig = actvFunc_sigmoid(x);
  return sig * (1.0 - sig);
}

//===================================================================================================================//

// Derivative of Tanh
TYPE actvFunc_dtanh(TYPE x) {
  TYPE tanh_x = tanh(x);
  return 1.0 - (tanh_x * tanh_x);
}

//===================================================================================================================//
