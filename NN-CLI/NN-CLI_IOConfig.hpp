#ifndef NN_CLI_IOCONFIG_HPP
#define NN_CLI_IOCONFIG_HPP

#include "NN-CLI_DataType.hpp"

#include <sys/types.h>

namespace NN_CLI {

// I/O configuration: how input and output data should be interpreted.
// This is an NN-CLI concept only — the underlying libraries (ANN, CNN) never see it.
struct IOConfig {
  DataType inputType  = DataType::VECTOR;
  DataType outputType = DataType::VECTOR;

  // Shape of input images (for ANN with image input; CNN uses CoreConfig.inputShape)
  ulong inputC = 0, inputH = 0, inputW = 0;

  // Shape of output images (required when outputType == IMAGE)
  ulong outputC = 0, outputH = 0, outputW = 0;

  bool hasInputShape()  const { return inputC > 0 && inputH > 0 && inputW > 0; }
  bool hasOutputShape() const { return outputC > 0 && outputH > 0 && outputW > 0; }
};

} // namespace NN_CLI

#endif // NN_CLI_IOCONFIG_HPP

