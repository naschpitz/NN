#include "NN-CLI_Utils.hpp"
#include "NN-CLI_ProgressBar.hpp"

#include <stdexcept>

using namespace NN_CLI;

//===================================================================================================================//
// ANN and CNN use fundamentally different input representations:
//   - ANN expects a flat std::vector<T> per sample (e.g. 784 values for a 28×28 image).
//   - CNN expects a Tensor3D<T> per sample with explicit (C, H, W) shape (e.g. 1×28×28).
// IDX files store raw flat byte arrays, so we need two loaders: one that keeps the data flat
// for ANN, and one that reshapes it into the 3D tensor layout that CNN requires.
//===================================================================================================================//

template <typename T>
ANN::Samples<T> Utils<T>::loadANNIDX(const std::string& dataPath, const std::string& labelsPath,
                                      ulong progressReports) {
  std::vector<std::vector<unsigned char>> data = loadIDXData(dataPath);
  std::vector<unsigned char> labels = loadIDXLabels(labelsPath);

  if (data.size() != labels.size()) {
    throw std::runtime_error("IDX data and labels count mismatch");
  }

  // Determine the number of unique labels for one-hot encoding
  unsigned char maxLabel = 0;
  for (unsigned char label : labels) {
    if (label > maxLabel) {
      maxLabel = label;
    }
  }
  size_t numClasses = static_cast<size_t>(maxLabel) + 1;

  ANN::Samples<T> samples;
  samples.reserve(data.size());
  size_t totalSamples = data.size();

  for (size_t i = 0; i < totalSamples; ++i) {
    ANN::Sample<T> sample;

    // Convert data to normalized input (0-1 range)
    sample.input.reserve(data[i].size());
    for (unsigned char value : data[i]) {
      sample.input.push_back(static_cast<T>(value) / static_cast<T>(255));
    }

    // Convert label to one-hot encoded output
    sample.output.resize(numClasses, static_cast<T>(0));
    sample.output[labels[i]] = static_cast<T>(1);

    samples.push_back(std::move(sample));
    ProgressBar::printLoadingProgress("Loading samples:", i + 1, totalSamples, progressReports);
  }

  return samples;
}

//===================================================================================================================//

template <typename T>
uint32_t Utils<T>::readBigEndianUInt32(std::ifstream& stream) {
  unsigned char bytes[4];
  stream.read(reinterpret_cast<char*>(bytes), 4);
  
  return (static_cast<uint32_t>(bytes[0]) << 24) |
         (static_cast<uint32_t>(bytes[1]) << 16) |
         (static_cast<uint32_t>(bytes[2]) << 8) |
         (static_cast<uint32_t>(bytes[3]));
}

//===================================================================================================================//

template <typename T>
std::vector<std::vector<unsigned char>> Utils<T>::loadIDXData(const std::string& path) {
  std::ifstream file(path, std::ios::binary);

  if (!file.is_open()) {
    throw std::runtime_error("Failed to open IDX data file: " + path);
  }

  uint32_t magic = readBigEndianUInt32(file);

  if (magic != 0x00000803) {
    throw std::runtime_error("Invalid IDX3 data file magic number");
  }

  uint32_t numItems = readBigEndianUInt32(file);
  uint32_t numRows = readBigEndianUInt32(file);
  uint32_t numCols = readBigEndianUInt32(file);
  uint32_t itemSize = numRows * numCols;

  std::vector<std::vector<unsigned char>> data;
  data.reserve(numItems);

  for (uint32_t i = 0; i < numItems; ++i) {
    std::vector<unsigned char> item(itemSize);
    file.read(reinterpret_cast<char*>(item.data()), itemSize);
    data.push_back(std::move(item));
  }

  return data;
}

//===================================================================================================================//

template <typename T>
std::vector<unsigned char> Utils<T>::loadIDXLabels(const std::string& path) {
  std::ifstream file(path, std::ios::binary);

  if (!file.is_open()) {
    throw std::runtime_error("Failed to open IDX labels file: " + path);
  }

  uint32_t magic = readBigEndianUInt32(file);

  if (magic != 0x00000801) {
    throw std::runtime_error("Invalid IDX1 labels file magic number");
  }

  uint32_t numLabels = readBigEndianUInt32(file);

  std::vector<unsigned char> labels(numLabels);
  file.read(reinterpret_cast<char*>(labels.data()), numLabels);

  return labels;
}

//===================================================================================================================//

template <typename T>
CNN::Samples<T> Utils<T>::loadCNNIDX(const std::string& dataPath, const std::string& labelsPath,
                                      const CNN::Shape3D& inputShape, ulong progressReports) {
  std::vector<std::vector<unsigned char>> data = loadIDXData(dataPath);
  std::vector<unsigned char> labels = loadIDXLabels(labelsPath);

  if (data.size() != labels.size()) {
    throw std::runtime_error("IDX data and labels count mismatch");
  }

  // Determine the number of unique labels for one-hot encoding
  unsigned char maxLabel = 0;
  for (unsigned char label : labels) {
    if (label > maxLabel) {
      maxLabel = label;
    }
  }
  size_t numClasses = static_cast<size_t>(maxLabel) + 1;

  CNN::Samples<T> samples;
  samples.reserve(data.size());
  size_t totalSamples = data.size();

  for (size_t i = 0; i < totalSamples; ++i) {
    CNN::Sample<T> sample;

    // Validate data size matches input shape
    if (data[i].size() != inputShape.size()) {
      throw std::runtime_error("IDX data item size (" + std::to_string(data[i].size()) +
        ") does not match expected input shape size (" + std::to_string(inputShape.size()) + ")");
    }

    // Reshape flat data into Tensor3D with given shape
    sample.input = CNN::Tensor3D<T>(inputShape);
    for (size_t j = 0; j < data[i].size(); ++j) {
      sample.input.data[j] = static_cast<T>(data[i][j]) / static_cast<T>(255);
    }

    // Convert label to one-hot encoded output
    sample.output.resize(numClasses, static_cast<T>(0));
    sample.output[labels[i]] = static_cast<T>(1);

    samples.push_back(std::move(sample));
    ProgressBar::printLoadingProgress("Loading samples:", i + 1, totalSamples, progressReports);
  }

  return samples;
}

//===================================================================================================================//

// Explicit template instantiations
template class NN_CLI::Utils<int>;
template class NN_CLI::Utils<float>;
template class NN_CLI::Utils<double>;

