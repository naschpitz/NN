#include "NN-CLI_DataLoader.hpp"

#include <QFile>
#include <QFileInfo>

#include <json.hpp>

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
  std::mt19937 rng(std::random_device{}());

  std::vector<SampleT> batch;
  batch.reserve(entryIndices.size());

  for (ulong idx : entryIndices) {
    batch.push_back(this->loadSample(idx, rng, transforms, augmentationProbability));
  }

  return batch;
}

template <typename SampleT>
typename DataLoader<SampleT>::ProviderT
DataLoader<SampleT>::makeSampleProvider(const Loader::AugmentationTransforms& transforms,
                                       float augmentationProbability) const {
  // Shared prefetch state between the provider callback and its worker thread.
  auto state = std::make_shared<PrefetchState<SampleT>>();

  // Persistent worker thread — waits for prefetch requests, loads batches in background.
  // Uses raw pointer to avoid circular reference (state → thread → state).
  // Safe because PrefetchState::~PrefetchState() joins the thread before destruction.
  PrefetchState<SampleT>* statePtr = state.get();

  state->worker = std::thread([this, statePtr, transforms, augmentationProbability]() {
    while (true) {
      std::vector<ulong> indices;

      {
        std::unique_lock<std::mutex> lock(statePtr->mutex);
        statePtr->workerCV.wait(lock, [statePtr]() { return statePtr->hasRequest || statePtr->shutdown; });

        if (statePtr->shutdown) return;

        indices = std::move(statePtr->requestIndices);
        statePtr->hasRequest = false;
      }

      // Load the batch outside the lock — this is the expensive I/O work.
      auto batch = this->loadBatch(indices, transforms, augmentationProbability);

      {
        std::lock_guard<std::mutex> lock(statePtr->mutex);
        statePtr->result = std::move(batch);
        statePtr->hasResult = true;
      }

      statePtr->callerCV.notify_one();
    }
  });

  // The provider callback. Captures the shared state.
  // When the last copy of this lambda is destroyed, PrefetchState's destructor
  // shuts down and joins the worker thread.
  return [this, state, transforms, augmentationProbability](
      const std::vector<ulong>& sampleIndices, ulong batchSize, ulong batchIndex) -> std::vector<SampleT> {
    ulong numSamples = sampleIndices.size();
    ulong start = batchIndex * batchSize;
    ulong end = std::min(start + batchSize, numSamples);

    std::vector<SampleT> batch;

    // Check if this batch was prefetched
    {
      std::unique_lock<std::mutex> lock(state->mutex);

      if (state->hasResult) {
        // Prefetched batch ready — take it
        batch = std::move(state->result);
        state->hasResult = false;
      } else if (state->hasRequest) {
        // Prefetch is still in progress — wait for it
        state->callerCV.wait(lock, [&state]() { return state->hasResult; });
        batch = std::move(state->result);
        state->hasResult = false;
      }
    }

    // If no prefetch was available (first call, or epoch boundary), load synchronously
    if (batch.empty() && start < numSamples) {
      std::vector<ulong> indices(sampleIndices.begin() + start, sampleIndices.begin() + end);
      batch = this->loadBatch(indices, transforms, augmentationProbability);
    }

    // Submit prefetch request for the next batch
    ulong nextStart = end;
    if (nextStart < numSamples) {
      ulong nextEnd = std::min(nextStart + batchSize, numSamples);

      {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->requestIndices.assign(sampleIndices.begin() + nextStart,
                                     sampleIndices.begin() + nextEnd);
        state->hasRequest = true;
      }

      state->workerCV.notify_one();
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