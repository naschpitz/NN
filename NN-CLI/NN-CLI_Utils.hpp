#ifndef NN_CLI_UTILS_HPP
#define NN_CLI_UTILS_HPP

#include <ANN_Core.hpp>
#include <CNN_Types.hpp>
#include <CNN_Sample.hpp>

#include <fstream>
#include <string>
#include <vector>

//===================================================================================================================//

namespace NN_CLI {

  template <typename T>
  class Utils {
    public:
      /// Load IDX dataset as ANN samples (flat input vectors)
      static ANN::Samples<T> loadANNIDX(const std::string& dataPath, const std::string& labelsPath);

      /// Load IDX dataset as CNN samples (3D tensor inputs with given shape)
      static CNN::Samples<T> loadCNNIDX(const std::string& dataPath, const std::string& labelsPath,
                                         const CNN::Shape3D& inputShape);

    private:
      static uint32_t readBigEndianUInt32(std::ifstream& stream);
      static std::vector<std::vector<unsigned char>> loadIDXData(const std::string& path);
      static std::vector<unsigned char> loadIDXLabels(const std::string& path);
  };

} // namespace NN_CLI

#endif // NN_CLI_UTILS_HPP

