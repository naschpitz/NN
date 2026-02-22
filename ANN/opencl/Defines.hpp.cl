#ifndef DEFINES_HPP_CL
#define DEFINES_HPP_CL

// Guard TYPE to avoid redefinition when loaded after another library's defines (e.g., CNN)
#ifndef TYPE
#define TYPE float
#endif

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
