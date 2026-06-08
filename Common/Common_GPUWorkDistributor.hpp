#ifndef COMMON_GPUWORKDISTRIBUTOR_HPP
#define COMMON_GPUWORKDISTRIBUTOR_HPP

#include "Common_TestResult.hpp"
#include "Common_ProgressCallback.hpp"

#include <QVector>
#include <QtConcurrent>

#include <algorithm>
#include <utility>
#include <vector>

//===================================================================================================================//

namespace Common
{
  //===================================================================================================================//
  //-- GPU work distribution --//
  //===================================================================================================================//

  // Describes a slice of work assigned to a single GPU.
  struct GPUWorkItem {
      size_t gpuIdx;
      ulong startIdx;
      ulong endIdx;
  };

  //===================================================================================================================//

  // Split batchLen items across numGPUs, returning one GPUWorkItem per active
  // GPU. GPUs that would receive zero items are omitted.
  inline QVector<GPUWorkItem> distributeBatchAcrossGPUs(ulong batchLen, size_t numGPUs)
  {
    ulong perGPU = batchLen / numGPUs;
    ulong remainder = batchLen % numGPUs;

    QVector<GPUWorkItem> workItems;

    for (size_t gpuIdx = 0; gpuIdx < numGPUs; gpuIdx++) {
      ulong startIdx = gpuIdx * perGPU + std::min(gpuIdx, remainder);
      ulong endIdx = startIdx + perGPU + (gpuIdx < remainder ? 1 : 0);

      if (startIdx < endIdx)
        workItems.append({gpuIdx, startIdx, endIdx});
    }

    return workItems;
  }

  //===================================================================================================================//

  // Execute test() across multiple GPUs with batch-wise data loading.
  //
  // Template parameters:
  //   T                - numeric type (float, double, int)
  //   SampleProviderFn - callable: (sampleIndices, batchSize, batchIndex) -> BatchT
  //   TestSubsetFn     - callable: (gpuIdx, batch, startIdx, endIdx) -> std::pair<T, ulong>
  //                      Returns {loss, numCorrect} for the given slice.
  //
  // The function handles:
  //   - Sequential sample index creation (no shuffling for test)
  //   - Batch iteration with bounded memory
  //   - GPU work distribution via distributeBatchAcrossGPUs()
  //   - Parallel dispatch via QtConcurrent::blockingMap
  //   - Loss and accuracy aggregation
  //   - Progress reporting
  template <typename T, typename SampleProviderFn, typename TestSubsetFn>
  TestResult<T> distributeTestAcrossGPUs(ulong numSamples, SampleProviderFn sampleProvider, size_t numGPUs,
                                         ulong batchSize, const ProgressCallback& progressCallback,
                                         TestSubsetFn&& testSubsetFn)
  {
    // Sequential index array (no shuffling for test)
    std::vector<ulong> sampleIndices(numSamples);

    for (ulong i = 0; i < numSamples; i++) {
      sampleIndices[i] = i;
    }

    ulong numBatches = (numSamples + batchSize - 1) / batchSize;

    T totalLoss = static_cast<T>(0);
    ulong totalCorrect = 0;

    for (ulong b = 0; b < numBatches; b++) {
      auto batch = sampleProvider(sampleIndices, batchSize, b);

      // Distribute batch across GPUs
      ulong batchLen = batch.size();
      QVector<GPUWorkItem> workItems = distributeBatchAcrossGPUs(batchLen, numGPUs);

      std::vector<std::pair<T, ulong>> gpuResults(numGPUs, {0, 0});

      QtConcurrent::blockingMap(workItems, [&batch, &gpuResults, &testSubsetFn](const GPUWorkItem& item) {
        gpuResults[item.gpuIdx] = testSubsetFn(item.gpuIdx, batch, item.startIdx, item.endIdx);
      });

      for (size_t i = 0; i < numGPUs; i++) {
        totalLoss += gpuResults[i].first;
        totalCorrect += gpuResults[i].second;
      }

      if (progressCallback) {
        ulong samplesProcessed = std::min((b + 1) * batchSize, numSamples);
        progressCallback(samplesProcessed, numSamples);
      }
    }

    TestResult<T> result;
    result.numSamples = numSamples;
    result.totalLoss = totalLoss;
    result.numCorrect = totalCorrect;
    result.averageLoss = (numSamples > 0) ? totalLoss / static_cast<T>(numSamples) : static_cast<T>(0);
    result.accuracy = (numSamples > 0) ? static_cast<T>(totalCorrect) / static_cast<T>(numSamples) * static_cast<T>(100)
                                       : static_cast<T>(0);

    return result;
  }

} // namespace Common

//===================================================================================================================//

#endif // COMMON_GPUWORKDISTRIBUTOR_HPP
