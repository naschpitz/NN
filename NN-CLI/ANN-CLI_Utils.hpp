#ifndef ANN_CLI_UTILS_HPP
#define ANN_CLI_UTILS_HPP

#include <ANN_Core.hpp>

#include <fstream>
#include <string>
#include <vector>

//===================================================================================================================//

namespace ANN_CLI {

  template <typename T>
  class Utils {
    public:
      /**
       * Load MNIST dataset from idx1 (labels) and idx3 (images) files.
       * @param imagesPath Path to the idx3-ubyte images file
       * @param labelsPath Path to the idx1-ubyte labels file
       * @return Samples with normalized pixel values (0-1) and one-hot encoded labels
       */
      static ANN::Samples<T> loadMNIST(const std::string& imagesPath, const std::string& labelsPath);

    private:
      static uint32_t readBigEndianUInt32(std::ifstream& stream);
      static std::vector<std::vector<unsigned char>> loadMNISTImages(const std::string& path);
      static std::vector<unsigned char> loadMNISTLabels(const std::string& path);
  };

} // namespace ANN_CLI

#endif // ANN_CLI_UTILS_HPP

