#ifndef NN_CLI_MODELSERIALIZER_HPP
#define NN_CLI_MODELSERIALIZER_HPP

#include "NN-CLI_AugmentationConfig.hpp"
#include "NN-CLI_IOConfig.hpp"
#include "NN-CLI_Types.hpp"

#include <ANN_Core.hpp>
#include <CNN_Core.hpp>

#include <json.hpp>

#include <QString>

#include <string>
#include <vector>

namespace NN_CLI
{

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
      //-- Binary parameter I/O --//
      static void saveANNParametersBinary(const std::string& binPath,
                                          const ANN::Core<float>& core);
      static void saveCNNParametersBinary(const std::string& binPath,
                                          const CNN::Core<float>& core);

      static void loadANNParametersBinary(const std::vector<char>& data,
                                          ANN::CoreConfig<float>& config,
                                          const ANN::LayersConfig& layersConfig);
      static void loadCNNParametersBinary(const std::vector<char>& data,
                                          CNN::CoreConfig<float>& config,
                                          const CNN::LayersConfig& layersConfig);

      //-- Package-aware save (single call produces .nnmodel) --//
      static void saveANNModelToPackage(const std::string& packagePath,
                                        const ANN::Core<float>& core,
                                        const ANN::CoreConfig<float>& coreConfig,
                                        const IOConfig& ioConfig,
                                        const AugmentationConfig& augConfig,
                                        const ValidationMetadata& validationMeta);
      static void saveCNNModelToPackage(const std::string& packagePath,
                                        const CNN::Core<float>& core,
                                        const CNN::CoreConfig<float>& coreConfig,
                                        const IOConfig& ioConfig,
                                        const AugmentationConfig& augConfig,
                                        const ValidationMetadata& validationMeta);

      //-- Output path helpers --//
      static std::string generateTrainingFilename(ulong epochs, ulong samples, float loss);
      static std::string generateDefaultOutputPath(const QString& inputFilePath, ulong epochs, ulong samples,
                                                   float loss);
      static std::string generateCheckpointPath(const QString& inputFilePath, ulong epoch, float loss);
      static std::string generateBestModelPath(const QString& inputFilePath);

    private:
      //-- JSON-building helpers --//
      static nlohmann::ordered_json buildANNModelJson(const ANN::Core<float>& core,
                                                      const ANN::CoreConfig<float>& coreConfig,
                                                      const IOConfig& ioConfig,
                                                      const AugmentationConfig& augConfig,
                                                      const ValidationMetadata& validationMeta);

      static nlohmann::ordered_json buildCNNModelJson(const CNN::Core<float>& core,
                                                      const CNN::CoreConfig<float>& coreConfig,
                                                      const IOConfig& ioConfig,
                                                      const AugmentationConfig& augConfig,
                                                      const ValidationMetadata& validationMeta);
  };

} // namespace NN_CLI

#endif // NN_CLI_MODELSERIALIZER_HPP
