#ifndef NN_CLI_MODELSERIALIZER_HPP
#define NN_CLI_MODELSERIALIZER_HPP

#include "NN-CLI_AugmentationConfig.hpp"
#include "NN-CLI_IOConfig.hpp"

#include <ANN_Core.hpp>
#include <CNN_Core.hpp>

#include <QString>

#include <string>

namespace NN_CLI
{

  using ulong = unsigned long;

  // Validation state passed to the serializer for inclusion in training metadata.
  struct ValidationMetadata {
      bool enabled = false;
      ulong numValSamples = 0;
      float lastValLoss = 0.0f;
      float bestValLoss = 0.0f;
      ulong bestValEpoch = 0;
  };

  class ModelSerializer
  {
    public:
      //-- Model saving --//
      static void saveANNModel(const std::string& filePath, const ANN::Core<float>& core,
                               const ANN::CoreConfig<float>& coreConfig, const IOConfig& ioConfig,
                               const AugmentationConfig& augConfig, const ValidationMetadata& valMeta);

      static void saveCNNModel(const std::string& filePath, const CNN::Core<float>& core,
                               const CNN::CoreConfig<float>& coreConfig, const IOConfig& ioConfig,
                               const AugmentationConfig& augConfig, const ValidationMetadata& valMeta);

      //-- Output path helpers --//
      static std::string generateTrainingFilename(ulong epochs, ulong samples, float loss);
      static std::string generateDefaultOutputPath(const QString& inputFilePath, ulong epochs, ulong samples,
                                                   float loss);
      static std::string generateCheckpointPath(const QString& inputFilePath, ulong epoch, float loss);
  };

} // namespace NN_CLI

#endif // NN_CLI_MODELSERIALIZER_HPP
