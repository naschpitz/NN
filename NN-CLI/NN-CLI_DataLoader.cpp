#include "NN-CLI_DataLoader.hpp"

#include <QFile>
#include <QFileInfo>

#include <json.hpp>

#include <QThreadPool>
#include <QtConcurrent>

#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace NN_CLI {

//===================================================================================================================//
//-- loadManifest --//
//===================================================================================================================//

template <typename SampleT>
void DataLoader<SampleT>::loadManifest(const std::string& samplesFilePath, const IOConfig& ioConfig,
                                        int inputC, int inputH, int inputW,
                                        int outputC, int outputH, int outputW) {
  this->ioConfig = ioConfig;
  this->inputC = inputC;
  this->inputH = inputH;
  this->inputW = inputW;
  this->outputC = outputC;
  this->outputH = outputH;
  this->outputW = outputW;
  this->baseDir = QFileInfo(QString::fromStdString(samplesFilePath)).absolutePath().toStdString();

  QFile file(QString::fromStdString(samplesFilePath));
  if (!file.open(QIODevice::ReadOnly))
    throw std::runtime_error("Failed to open samples file: " + samplesFilePath);

  QByteArray fileData = file.readAll();
  nlohmann::json json = nlohmann::json::parse(fileData.toStdString());
  const nlohmann::json& samplesArray = json.at("samples");

  this->manifest.clear();
  this->manifest.reserve(samplesArray.size());

  for (const auto& sampleJson : samplesArray) {
    SampleManifest entry;

    // Store input reference (path or raw data — but do NOT load images)
    if (ioConfig.inputType == DataType::IMAGE) {
      entry.inputPath = sampleJson.at("input").get<std::string>();
      entry.inputIsImage = true;
    } else {
      entry.inputData = sampleJson.at("input").get<std::vector<float>>();
      entry.inputIsImage = false;
    }

    // Store output reference
    if (ioConfig.outputType == DataType::IMAGE) {
      entry.outputPath = sampleJson.at("output").get<std::string>();
      entry.outputIsImage = true;
    } else {
      entry.output = sampleJson.at("output").get<std::vector<float>>();
      entry.outputIsImage = false;
    }

    this->manifest.push_back(std::move(entry));
  }

  // Initialize entries as 1:1 mapping to manifest (no augmentation yet)
  this->fromMemory = false;
  this->memorySamples.clear();
  this->entries.clear();
  this->entries.reserve(this->manifest.size());
  for (ulong i = 0; i < this->manifest.size(); i++) {
    this->entries.push_back({i, false});
  }
}

//===================================================================================================================//
//-- loadFromMemory --//
//===================================================================================================================//

template <typename SampleT>
void DataLoader<SampleT>::loadFromMemory(std::vector<SampleT>&& samples,
                                          int inputC, int inputH, int inputW) {
  this->inputC = inputC;
  this->inputH = inputH;
  this->inputW = inputW;
  this->fromMemory = true;
  this->manifest.clear();
  this->memorySamples = std::move(samples);

  this->entries.clear();
  this->entries.reserve(this->memorySamples.size());
  for (ulong i = 0; i < this->memorySamples.size(); i++) {
    this->entries.push_back({i, false});
  }
}

//===================================================================================================================//
//-- planAugmentation --//
//===================================================================================================================//

// Helper to get the output vector from a sample (works for both ANN and CNN).
static const std::vector<float>& sampleOutput(const ANN::Sample<float>& s) { return s.output; }
static const std::vector<float>& sampleOutput(const CNN::Sample<float>& s) { return s.output; }

template <typename SampleT>
void DataLoader<SampleT>::planAugmentation(ulong augmentationFactor, bool balanceAugmentation) {
  ulong originalCount = this->fromMemory ? this->memorySamples.size() : this->manifest.size();
  if (augmentationFactor == 0 && !balanceAugmentation) return;
  if (originalCount == 0) return;

  // Count samples per class
  auto getClassIndex = [](const std::vector<float>& output) -> ulong {
    return static_cast<ulong>(std::distance(output.begin(),
        std::max_element(output.begin(), output.end())));
  };

  std::map<ulong, std::vector<ulong>> classIndices;
  for (ulong i = 0; i < originalCount; i++) {
    const std::vector<float>& output = this->fromMemory
        ? sampleOutput(this->memorySamples[i])
        : this->manifest[i].output;
    ulong cls = getClassIndex(output);
    classIndices[cls].push_back(i);
  }

  ulong maxClassCount = 0;
  for (const auto& [cls, indices] : classIndices)
    maxClassCount = std::max(maxClassCount, static_cast<ulong>(indices.size()));

  std::mt19937 rng(42); // deterministic augmentation plan

  for (const auto& [cls, indices] : classIndices) {
    ulong currentCount = indices.size();
    ulong targetCount = currentCount;

    if (augmentationFactor > 0)
      targetCount = currentCount * augmentationFactor;

    if (balanceAugmentation) {
      ulong balancedTarget = maxClassCount;
      if (augmentationFactor > 0)
        balancedTarget = maxClassCount * augmentationFactor;
      targetCount = std::max(targetCount, balancedTarget);
    }

    ulong toGenerate = (targetCount > currentCount) ? (targetCount - currentCount) : 0;

    for (ulong i = 0; i < toGenerate; i++) {
      std::uniform_int_distribution<ulong> dist(0, currentCount - 1);
      ulong srcIdx = indices[dist(rng)];
      this->entries.push_back({srcIdx, true});
    }
  }

  std::cout << "Data augmentation: " << originalCount << " original + "
            << (this->entries.size() - originalCount) << " augmented = "
            << this->entries.size() << " total samples\n";
}

//===================================================================================================================//
//-- getAllOutputs --//
//===================================================================================================================//

template <typename SampleT>
std::vector<std::vector<float>> DataLoader<SampleT>::getAllOutputs() const {
  std::vector<std::vector<float>> outputs;
  outputs.reserve(this->entries.size());
  for (const auto& entry : this->entries) {
    if (this->fromMemory)
      outputs.push_back(sampleOutput(this->memorySamples[entry.sourceIndex]));
    else
      outputs.push_back(this->manifest[entry.sourceIndex].output);
  }
  return outputs;
}

//===================================================================================================================//
//-- makeSampleProvider --//
//===================================================================================================================//

template <typename SampleT>
std::vector<SampleT>
DataLoader<SampleT>::loadBatch(const std::vector<ulong>& entryIndices,
                               const Loader::AugmentationTransforms& transforms,
                               float augmentationProbability) const {
  ulong count = entryIndices.size();
  std::vector<SampleT> batch(count);

  // Load all images in parallel using the global QThreadPool.
  int numThreads = std::min(QThreadPool::globalInstance()->maxThreadCount(),
                            static_cast<int>(count));

  ulong chunkSize = count / numThreads;
  ulong remainder = count % numThreads;

  QVector<QFuture<void>> futures;
  futures.reserve(numThreads);

  ulong offset = 0;
  for (int t = 0; t < numThreads; t++) {
    ulong thisChunk = chunkSize + (static_cast<ulong>(t) < remainder ? 1 : 0);
    ulong chunkStart = offset;
    ulong chunkEnd = offset + thisChunk;
    offset = chunkEnd;

    futures.append(QtConcurrent::run([this, &entryIndices, &batch, &transforms,
                                      augmentationProbability, chunkStart, chunkEnd]() {
      std::mt19937 rng(std::random_device{}());

      for (ulong i = chunkStart; i < chunkEnd; i++) {
        batch[i] = this->loadSample(entryIndices[i], rng, transforms, augmentationProbability);
      }
    }));
  }

  for (auto& f : futures) f.waitForFinished();

  return batch;
}

template <typename SampleT>
typename DataLoader<SampleT>::ProviderT
DataLoader<SampleT>::makeSampleProvider(const Loader::AugmentationTransforms& transforms,
                                       float augmentationProbability) const {
  // Shared state for prefetching the next batch in the background.
  auto prefetch = std::make_shared<QFuture<std::vector<SampleT>>>();
  auto hasPrefetch = std::make_shared<bool>(false);

  return [this, prefetch, hasPrefetch, transforms, augmentationProbability](
      const std::vector<ulong>& sampleIndices, ulong batchSize, ulong batchIndex) -> std::vector<SampleT> {
    ulong numSamples = sampleIndices.size();
    ulong start = batchIndex * batchSize;
    ulong end = std::min(start + batchSize, numSamples);

    // If the previous call prefetched this batch, retrieve it; otherwise load now.
    std::vector<SampleT> batch;
    if (*hasPrefetch) {
      prefetch->waitForFinished();
      batch = prefetch->result();
      *hasPrefetch = false;
    } else {
      std::vector<ulong> indices(sampleIndices.begin() + start, sampleIndices.begin() + end);
      batch = this->loadBatch(indices, transforms, augmentationProbability);
    }

    // Prefetch the next batch in the background (parallel image loading inside loadBatch).
    ulong nextStart = end;
    if (nextStart < numSamples) {
      ulong nextEnd = std::min(nextStart + batchSize, numSamples);
      std::vector<ulong> nextIndices(sampleIndices.begin() + nextStart,
                                     sampleIndices.begin() + nextEnd);

      *prefetch = QtConcurrent::run(
          [this, indices = std::move(nextIndices), transforms, augmentationProbability]() {
            return this->loadBatch(indices, transforms, augmentationProbability);
          });
      *hasPrefetch = true;
    }

    return batch;
  };
}

//===================================================================================================================//
//-- loadSample specializations --//
//===================================================================================================================//

template <>
ANN::Sample<float> DataLoader<ANN::Sample<float>>::loadSample(
    ulong entryIndex, std::mt19937& rng,
    const Loader::AugmentationTransforms& transforms,
    float augmentationProbability) const {
  const AugmentedEntry& entry = this->entries[entryIndex];

  ANN::Sample<float> sample;

  if (this->fromMemory) {
    sample = this->memorySamples[entry.sourceIndex]; // copy
  } else {
    const SampleManifest& m = this->manifest[entry.sourceIndex];
    if (m.inputIsImage) {
      std::string fullPath = ImageLoader::resolvePath(m.inputPath, this->baseDir);
      sample.input = ImageLoader::loadImage(fullPath, this->inputC, this->inputH, this->inputW);
    } else {
      sample.input = m.inputData;
    }
    if (m.outputIsImage) {
      std::string fullPath = ImageLoader::resolvePath(m.outputPath, this->baseDir);
      sample.output = ImageLoader::loadImage(fullPath, this->outputC, this->outputH, this->outputW);
    } else {
      sample.output = m.output;
    }
  }

  // Apply augmentation if this is an augmented entry
  if (entry.augmented) {
    bool hasImageShape = (this->inputC > 0 && this->inputH > 0 && this->inputW > 0);
    if (hasImageShape) {
      ImageLoader::applyRandomTransforms(sample.input, this->inputC, this->inputH, this->inputW,
                                          rng, transforms, augmentationProbability);
    } else if (transforms.gaussianNoise > 0.0f) {
      ImageLoader::addGaussianNoise(sample.input, transforms.gaussianNoise, rng);
    }
  }

  return sample;
}

//===================================================================================================================//

template <>
CNN::Sample<float> DataLoader<CNN::Sample<float>>::loadSample(
    ulong entryIndex, std::mt19937& rng,
    const Loader::AugmentationTransforms& transforms,
    float augmentationProbability) const {
  const AugmentedEntry& entry = this->entries[entryIndex];

  CNN::Sample<float> sample;

  if (this->fromMemory) {
    sample = this->memorySamples[entry.sourceIndex]; // copy
  } else {
    const SampleManifest& m = this->manifest[entry.sourceIndex];
    if (m.inputIsImage) {
      std::string fullPath = ImageLoader::resolvePath(m.inputPath, this->baseDir);
      std::vector<float> flatInput = ImageLoader::loadImage(fullPath, this->inputC, this->inputH, this->inputW);
      CNN::Shape3D shape{static_cast<ulong>(this->inputC),
                         static_cast<ulong>(this->inputH),
                         static_cast<ulong>(this->inputW)};
      sample.input = CNN::Input<float>(shape);
      sample.input.data = std::move(flatInput);
    } else {
      CNN::Shape3D shape{static_cast<ulong>(this->inputC),
                         static_cast<ulong>(this->inputH),
                         static_cast<ulong>(this->inputW)};
      sample.input = CNN::Input<float>(shape);
      sample.input.data = m.inputData;
    }
    if (m.outputIsImage) {
      std::string fullPath = ImageLoader::resolvePath(m.outputPath, this->baseDir);
      sample.output = ImageLoader::loadImage(fullPath, this->outputC, this->outputH, this->outputW);
    } else {
      sample.output = m.output;
    }
  }

  // Apply augmentation if this is an augmented entry
  if (entry.augmented) {
    ImageLoader::applyRandomTransforms(sample.input.data, this->inputC, this->inputH, this->inputW,
                                        rng, transforms, augmentationProbability);
  }

  return sample;
}

//===================================================================================================================//
//-- Explicit template instantiations --//
//===================================================================================================================//

template class DataLoader<ANN::Sample<float>>;
template class DataLoader<CNN::Sample<float>>;

} // namespace NN_CLI