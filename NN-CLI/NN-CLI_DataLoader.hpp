#ifndef NN_CLI_DATALOADER_HPP
#define NN_CLI_DATALOADER_HPP

#include "NN-CLI_ImageLoader.hpp"
#include "NN-CLI_Loader.hpp"

#include <ANN_Sample.hpp>
#include <CNN_Sample.hpp>

#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace NN_CLI {

using ulong = unsigned long;

// Lightweight manifest entry — stores paths/labels, NOT pixel data.
struct SampleManifest {
  std::string inputPath;                // File path for image inputs
  std::vector<float> inputData;         // Raw numeric input (non-image)
  std::string outputPath;               // File path for image outputs
  std::vector<float> output;            // Expected output vector
  bool inputIsImage = true;             // Whether input is an image path
  bool outputIsImage = false;           // Whether output is an image path
};

// Entry in the expanded (augmented) sample list.
// For original samples: sourceIndex == own index in the original list, augmented == false.
// For augmented samples: sourceIndex == original sample index, augmented == true.
struct AugmentedEntry {
  ulong sourceIndex;  // Index into the original sample list (manifest or memorySamples)
  bool augmented;     // Whether to apply random transforms when loading
};

// Trait to map Sample type to the corresponding SampleProvider type.
template <typename SampleT> struct SampleProviderFor;
template <> struct SampleProviderFor<ANN::Sample<float>> { using type = ANN::SampleProvider<float>; };
template <> struct SampleProviderFor<CNN::Sample<float>> { using type = CNN::SampleProvider<float>; };

template <typename SampleT>
class DataLoader {
  public:
    using ProviderT = typename SampleProviderFor<SampleT>::type;

    // Parse samples JSON and store lightweight manifest (no pixel data loaded).
    void loadManifest(const std::string& samplesFilePath, const IOConfig& ioConfig,
                      int inputC, int inputH, int inputW,
                      int outputC = 0, int outputH = 0, int outputW = 0);

    // Load from pre-loaded samples (e.g. IDX format). Stores samples in memory.
    void loadFromMemory(std::vector<SampleT>&& samples, int inputC, int inputH, int inputW);

    // Compute augmentation plan (expand entries without loading data).
    void planAugmentation(ulong augmentationFactor, bool balanceAugmentation);

    // Total number of samples (original + augmented).
    ulong numSamples() const { return this->entries.size(); }

    // Get all output vectors (for class weight computation without loading images).
    std::vector<std::vector<float>> getAllOutputs() const;

    // Build a SampleProvider with async prefetching for use with train().
    // The provider receives the full shuffled index array, batch size, and current batch index.
    // It returns the current batch's samples and prefetches the next batch in the background.
    ProviderT makeSampleProvider(const Loader::AugmentationTransforms& transforms = {},
                                 float augmentationProbability = 0.5f) const;

  private:
    std::vector<SampleManifest> manifest;   // Original samples — paths + labels (JSON path)
    std::vector<SampleT> memorySamples;     // Original samples — fully loaded (memory path)
    bool fromMemory = false;                // Which source to use
    std::vector<AugmentedEntry> entries;    // Expanded list (original + augmented)
    std::string baseDir;                    // Base directory for resolving relative paths
    int inputC = 0, inputH = 0, inputW = 0;
    int outputC = 0, outputH = 0, outputW = 0;
    IOConfig ioConfig;

    // Retrieve a single sample by entry index, optionally applying augmentation.
    SampleT loadSample(ulong entryIndex, std::mt19937& rng,
                       const Loader::AugmentationTransforms& transforms,
                       float augmentationProbability) const;
};

} // namespace NN_CLI

#endif // NN_CLI_DATALOADER_HPP

