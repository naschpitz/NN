#ifndef ACTVFUNC_CPP_CL
#define ACTVFUNC_CPP_CL

// Note: Defines.hpp.cl must be loaded before this file

//===================================================================================================================//

TYPE actvFunc_relu(TYPE x) {
  return (x > 0) ? x : 0;
}

//===================================================================================================================//

TYPE actvFunc_sigmoid(TYPE x) {
  return 1.0f / (1.0f + exp(-x));
}

//===================================================================================================================//

TYPE actvFunc_tanh_impl(TYPE x) {
  return tanh(x);
}

//===================================================================================================================//

TYPE actvFunc_drelu(TYPE x) {
  return (x > 0) ? 1.0f : 0.0f;
}

//===================================================================================================================//

TYPE actvFunc_dsigmoid(TYPE x) {
  TYPE sig = actvFunc_sigmoid(x);
  return sig * (1.0f - sig);
}

//===================================================================================================================//

TYPE actvFunc_dtanh(TYPE x) {
  TYPE t = tanh(x);
  return 1.0f - (t * t);
}

//===================================================================================================================//

TYPE actvFunc_calculate(TYPE z, ActvFuncType type) {
  switch(type) {
    case ACTV_RELU:
      return actvFunc_relu(z);
    case ACTV_SIGMOID:
      return actvFunc_sigmoid(z);
    case ACTV_TANH:
      return actvFunc_tanh_impl(z);
    default:
      return z;
  }
}

//===================================================================================================================//

TYPE actvFunc_derivative(TYPE z, ActvFuncType type) {
  switch(type) {
    case ACTV_RELU:
      return actvFunc_drelu(z);
    case ACTV_SIGMOID:
      return actvFunc_dsigmoid(z);
    case ACTV_TANH:
      return actvFunc_dtanh(z);
    default:
      return 1.0f;
  }
}

//===================================================================================================================//

#endif // ACTVFUNC_CPP_CL
