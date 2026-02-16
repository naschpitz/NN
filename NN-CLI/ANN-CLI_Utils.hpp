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
       * Load dataset from IDX format files (idx1 for labels, idx3 for data).
       * @param dataPath Path to the idx3-ubyte data file (e.g., images, feature vectors)
       * @param labelsPath Path to the idx1-ubyte labels file
       * @return Samples with normalized values (0-1) and one-hot encoded labels
       */
      static ANN::Samples<T> loadIDX(const std::string& dataPath, const std::string& labelsPath);

    private:
      static uint32_t readBigEndianUInt32(std::ifstream& stream);
      static std::vector<std::vector<unsigned char>> loadIDXData(const std::string& path);
      static std::vector<unsigned char> loadIDXLabels(const std::string& path);
  };

} // namespace ANN_CLI

#endif // ANN_CLI_UTILS_HPP

