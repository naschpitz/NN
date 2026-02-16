#include "ANN-CLI_Utils.hpp"

#include <stdexcept>

using namespace ANN_CLI;

//===================================================================================================================//

template <typename T>
ANN::Samples<T> Utils<T>::loadIDX(const std::string& dataPath, const std::string& labelsPath) {
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

  for (size_t i = 0; i < data.size(); ++i) {
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

// Explicit template instantiations
template class ANN_CLI::Utils<int>;
template class ANN_CLI::Utils<float>;
template class ANN_CLI::Utils<double>;

