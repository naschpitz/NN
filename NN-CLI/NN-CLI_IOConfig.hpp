#ifndef NN_CLI_IOCONFIG_HPP
#define NN_CLI_IOCONFIG_HPP

#include "NN-CLI_DataType.hpp"

#include <sys/types.h>

namespace NN_CLI
{

  // I/O configuration: how input and output data should be interpreted.
  // This is an NN-CLI concept only — the underlying libraries (, CNN) never see it.
  struct IOConfig {
      //-- Methods --//
      bool hasInputShape() const
      {
        return this->inputC > 0 && this->inputH > 0 && this->inputW > 0;
      }

      bool hasOutputShape() const
      {
        return this->outputC > 0 && this->outputH > 0 && this->outputW > 0;
      }

      //-- Members --//
      DataType inputType = DataType::VECTOR;
      DataType outputType = DataType::VECTOR;

      // Shape of input images (for  with image input; CNN uses CoreConfig.inputShape)
      ulong inputC = 0, inputH = 0, inputW = 0;

      // Shape of output images (required when outputType == IMAGE)
      ulong outputC = 0, outputH = 0, outputW = 0;

      // Progress/checkpoint settings (NN-CLI display & persistence)
      ulong progressReports = 1000;
      ulong saveModelInterval = 10; // 0 = disabled
  };

} // namespace NN_CLI

#endif // NN_CLI_IOCONFIG_HPP
