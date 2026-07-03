#ifndef COMMON_CONFUSIONMATRIX_HPP
#define COMMON_CONFUSIONMATRIX_HPP

#include <sys/types.h>

#include <vector>

//===================================================================================================================//

namespace Common
{
  // Confusion matrix for classification evaluation.
  //
  // Raw counts are the source of truth: `matrix` is numClasses*numClasses,
  // row-major with `matrix[actual * numClasses + predicted]`. Derived metrics
  // (per-class TP/FP/FN/TN, precision/recall/F1, accuracy, macro/micro/weighted
  // averages) are filled by computeMetrics() and are only meaningful afterwards.
  //
  // Convention: any 0/0 division yields 0. `accuracy` is a 0-1 fraction (trace /
  // totalSamples), distinct from TestResult::accuracy which is 0-100.
  template <typename T>
  class ConfusionMatrix
  {
    public:
      //-- Methods --//

      // Initialize the raw matrix for `classes` classes the first time it is
      // called (no-op once numClasses is already set). Call before incrementing.
      void ensureSized(ulong classes);

      // True when no matrix has been recorded.
      bool empty() const;

      // Compute all derived metrics from the raw counts. Refills the derived
      // vectors. Convention: 0/0 division yields 0.
      void computeMetrics();

      //-- Members --//

      // Raw counts (source of truth)
      ulong numClasses = 0; // 0 means "uninitialized / empty"
      std::vector<ulong> matrix; // numClasses², row-major [actual * numClasses + predicted]
      ulong totalSamples = 0; // sum of all counts

      // Derived per-class counts (sized numClasses by computeMetrics)
      std::vector<ulong> truePositive;
      std::vector<ulong> falsePositive;
      std::vector<ulong> falseNegative;
      std::vector<ulong> trueNegative;
      std::vector<ulong> support; // actual sample count per class (TP + FN)

      // Derived per-class metrics
      std::vector<T> precision;
      std::vector<T> recall;
      std::vector<T> f1Score;

      // Derived overall metrics
      T accuracy = 0; // trace / totalSamples (0-1)
      T macroPrecision = 0;
      T macroRecall = 0;
      T macroF1 = 0;
      T microPrecision = 0;
      T microRecall = 0;
      T microF1 = 0;
      T weightedPrecision = 0;
      T weightedRecall = 0;
      T weightedF1 = 0;
  };

  //===================================================================================================================//

  // Add `src` raw counts into `dst` (raw-only; does not recompute metrics).
  // Used by GPU/CPU orchestrators to merge per-worker / per-GPU subsets before
  // a single computeMetrics() call on the aggregate.
  template <typename T>
  void mergeConfusionMatrix(ConfusionMatrix<T>& dst, const ConfusionMatrix<T>& src);
}

//===================================================================================================================//

#endif // COMMON_CONFUSIONMATRIX_HPP
