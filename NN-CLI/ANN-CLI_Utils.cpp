#include "ANN-CLI_Utils.hpp"

#include <stdexcept>

using namespace ANN_CLI;

//===================================================================================================================//

template <typename T>
ANN::Samples<T> Utils<T>::loadMNIST(const std::string& imagesPath, const std::string& labelsPath) {
  std::vector<std::vector<unsigned char>> images = loadMNISTImages(imagesPath);
  std::vector<unsigned char> labels = loadMNISTLabels(labelsPath);

  if (images.size() != labels.size()) {
    throw std::runtime_error("MNIST images and labels count mismatch");
  }

  ANN::Samples<T> samples;
  samples.reserve(images.size());

  for (size_t i = 0; i < images.size(); ++i) {
    ANN::Sample<T> sample;

    // Convert image pixels to normalized input (0-1 range)
    sample.input.reserve(images[i].size());
    for (unsigned char pixel : images[i]) {
      sample.input.push_back(static_cast<T>(pixel) / static_cast<T>(255));
    }

    // Convert label to one-hot encoded output (10 classes for digits 0-9)
    sample.output.resize(10, static_cast<T>(0));
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
std::vector<std::vector<unsigned char>> Utils<T>::loadMNISTImages(const std::string& path) {
  std::ifstream file(path, std::ios::binary);

  if (!file.is_open()) {
    throw std::runtime_error("Failed to open MNIST images file: " + path);
  }

  uint32_t magic = readBigEndianUInt32(file);

  if (magic != 0x00000803) {
    throw std::runtime_error("Invalid MNIST images file magic number");
  }

  uint32_t numImages = readBigEndianUInt32(file);
  uint32_t numRows = readBigEndianUInt32(file);
  uint32_t numCols = readBigEndianUInt32(file);
  uint32_t imageSize = numRows * numCols;

  std::vector<std::vector<unsigned char>> images;
  images.reserve(numImages);

  for (uint32_t i = 0; i < numImages; ++i) {
    std::vector<unsigned char> image(imageSize);
    file.read(reinterpret_cast<char*>(image.data()), imageSize);
    images.push_back(std::move(image));
  }

  return images;
}

//===================================================================================================================//

template <typename T>
std::vector<unsigned char> Utils<T>::loadMNISTLabels(const std::string& path) {
  std::ifstream file(path, std::ios::binary);

  if (!file.is_open()) {
    throw std::runtime_error("Failed to open MNIST labels file: " + path);
  }

  uint32_t magic = readBigEndianUInt32(file);

  if (magic != 0x00000801) {
    throw std::runtime_error("Invalid MNIST labels file magic number");
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

