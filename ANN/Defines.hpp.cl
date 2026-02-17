#ifndef DEFINES_HPP_CL
#define DEFINES_HPP_CL

#define TYPE float

//===================================================================================================================//
// ActvFuncType enum (must match C++ ANN::ActvFuncType)
//===================================================================================================================//

typedef enum {
  ACTV_RELU = 0,
  ACTV_SIGMOID = 1,
  ACTV_TANH = 2,
  ACTV_UNKNOWN = 3
} ActvFuncType;

//===================================================================================================================//
// Layer struct (must match C++ ANN::Layer)
//===================================================================================================================//

typedef struct {
  ulong numNeurons;
  ActvFuncType actvFuncType;
} Layer;

#endif // DEFINES_HPP_CL
