#include "test_helpers.hpp"
#include "../NN-CLI_DataLoader.hpp"

#include <ANN_Sample.hpp>
#include <CNN_Sample.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <numeric>
#include <thread>
#include <vector>

using namespace NN_CLI;

//===================================================================================================================//

// Create N ANN samples with input = {idx} and output = one-hot class assignment.
static ANN::Samples<float> makeANNSamples(ulong count, ulong numClasses = 3) {
  ANN::Samples<float> samples(count);
  for (ulong i = 0; i < count; i++) {
    samples[i].input  = {static_cast<float>(i)};
    samples[i].output.assign(numClasses, 0.0f);
    samples[i].output[i % numClasses] = 1.0f;
  }
  return samples;
}

//===================================================================================================================//

static void testProviderReturnsCorrectBatches() {
  std::cout << "  testProviderReturnsCorrectBatches... ";

  auto samples = makeANNSamples(10);
  DataLoader<ANN::Sample<float>> loader;
  loader.loadFromMemory(std::move(samples), 1, 1, 1);

  auto provider = loader.makeSampleProvider();

  // Shuffled indices: 0..9
  std::vector<ulong> indices(10);
  std::iota(indices.begin(), indices.end(), 0);

  ulong batchSize = 3;

  // Batch 0: indices 0,1,2
  auto batch0 = provider(indices, batchSize, 0);
  CHECK(batch0.size() == 3, "batch 0 has 3 samples");
  CHECK(batch0[0].input[0] == 0.0f, "batch 0 sample 0 correct");
  CHECK(batch0[2].input[0] == 2.0f, "batch 0 sample 2 correct");

  // Batch 1: indices 3,4,5
  auto batch1 = provider(indices, batchSize, 1);
  CHECK(batch1.size() == 3, "batch 1 has 3 samples");
  CHECK(batch1[0].input[0] == 3.0f, "batch 1 sample 0 correct");

  // Batch 2: indices 6,7,8
  auto batch2 = provider(indices, batchSize, 2);
  CHECK(batch2.size() == 3, "batch 2 has 3 samples");
  CHECK(batch2[0].input[0] == 6.0f, "batch 2 sample 0 correct");

  // Batch 3 (partial): indices 9
  auto batch3 = provider(indices, batchSize, 3);
  CHECK(batch3.size() == 1, "last batch has 1 sample");
  CHECK(batch3[0].input[0] == 9.0f, "last batch sample correct");

  std::cout << std::endl;
}

//===================================================================================================================//

static void testProviderRespectsShuffledIndices() {
  std::cout << "  testProviderRespectsShuffledIndices... ";

  auto samples = makeANNSamples(6);
  DataLoader<ANN::Sample<float>> loader;
  loader.loadFromMemory(std::move(samples), 1, 1, 1);

  auto provider = loader.makeSampleProvider();

  // Reversed indices: 5,4,3,2,1,0
  std::vector<ulong> indices = {5, 4, 3, 2, 1, 0};
  ulong batchSize = 3;

  auto batch0 = provider(indices, batchSize, 0);
  CHECK(batch0.size() == 3, "batch has 3 samples");
  CHECK(batch0[0].input[0] == 5.0f, "shuffled batch 0 sample 0 = original[5]");
  CHECK(batch0[1].input[0] == 4.0f, "shuffled batch 0 sample 1 = original[4]");
  CHECK(batch0[2].input[0] == 3.0f, "shuffled batch 0 sample 2 = original[3]");

  auto batch1 = provider(indices, batchSize, 1);
  CHECK(batch1[0].input[0] == 2.0f, "shuffled batch 1 sample 0 = original[2]");
  CHECK(batch1[1].input[0] == 1.0f, "shuffled batch 1 sample 1 = original[1]");
  CHECK(batch1[2].input[0] == 0.0f, "shuffled batch 1 sample 2 = original[0]");

  std::cout << std::endl;
}

//===================================================================================================================//

static void testPrefetchOverlapsWithProcessing() {
  std::cout << "  testPrefetchOverlapsWithProcessing... ";

  // With prefetching, batch 1 should be faster than batch 0 because it was
  // loaded in the background while we "processed" batch 0.
  // For in-memory samples the effect is small, so we just verify it doesn't
  // error out and returns correct data when called sequentially with a sleep
  // between calls (simulating training work).

  auto samples = makeANNSamples(20);
  DataLoader<ANN::Sample<float>> loader;
  loader.loadFromMemory(std::move(samples), 1, 1, 1);

  auto provider = loader.makeSampleProvider();

  std::vector<ulong> indices(20);
  std::iota(indices.begin(), indices.end(), 0);

  ulong batchSize = 5;

  // Simulate the training loop pattern: get batch, sleep (train), get next
  for (ulong b = 0; b < 4; b++) {
    auto batch = provider(indices, batchSize, b);
    CHECK(batch.size() == 5, "batch " + std::to_string(b) + " has 5 samples");
    CHECK(batch[0].input[0] == static_cast<float>(b * 5),
          "batch " + std::to_string(b) + " first sample correct");

    // Simulate training time — prefetch of next batch happens during this sleep
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  std::cout << std::endl;
}

//===================================================================================================================//

static void testNewEpochResetsPrefetch() {
  std::cout << "  testNewEpochResetsPrefetch... ";

  auto samples = makeANNSamples(6);
  DataLoader<ANN::Sample<float>> loader;
  loader.loadFromMemory(std::move(samples), 1, 1, 1);

  auto provider = loader.makeSampleProvider();
  ulong batchSize = 3;

  // Epoch 1: indices 0..5
  std::vector<ulong> epoch1 = {0, 1, 2, 3, 4, 5};
  auto b0 = provider(epoch1, batchSize, 0);
  auto b1 = provider(epoch1, batchSize, 1);
  CHECK(b0[0].input[0] == 0.0f, "epoch 1 batch 0 correct");
  CHECK(b1[0].input[0] == 3.0f, "epoch 1 batch 1 correct");

  // Epoch 2: different shuffle — batchIndex resets to 0
  // The prefetched batch from epoch 1 (if any) should NOT be used
  std::vector<ulong> epoch2 = {5, 4, 3, 2, 1, 0};
  auto e2b0 = provider(epoch2, batchSize, 0);
  CHECK(e2b0[0].input[0] == 5.0f, "epoch 2 batch 0 uses new indices");
  CHECK(e2b0[1].input[0] == 4.0f, "epoch 2 batch 0 sample 1 correct");

  auto e2b1 = provider(epoch2, batchSize, 1);
  CHECK(e2b1[0].input[0] == 2.0f, "epoch 2 batch 1 correct");

  std::cout << std::endl;
}

//===================================================================================================================//

void runDataLoaderTests() {
  testProviderReturnsCorrectBatches();
  testProviderRespectsShuffledIndices();
  testPrefetchOverlapsWithProcessing();
  testNewEpochResetsPrefetch();
}

