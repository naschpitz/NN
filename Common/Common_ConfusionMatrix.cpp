#include "Common_ConfusionMatrix.hpp"

//===================================================================================================================//

namespace Common
{
  //===================================================================================================================//
  //-- ConfusionMatrix methods --//
  //===================================================================================================================//

  // Initialize the raw matrix for `classes` classes the first time it is called
  // (no-op once numClasses is already set).
  template <typename T>
  void ConfusionMatrix<T>::ensureSized(ulong classes)
  {
    if (this->numClasses == 0 && classes > 0) {
      this->numClasses = classes;
      this->matrix.assign(classes * classes, 0);
    }
  }

  //===================================================================================================================//

  // True when no matrix has been recorded.
  template <typename T>
  bool ConfusionMatrix<T>::empty() const
  {
    return this->numClasses == 0;
  }

  //===================================================================================================================//

  // Compute all derived metrics from the raw counts. Convention: 0/0 division
  // yields 0.
  template <typename T>
  void ConfusionMatrix<T>::computeMetrics()
  {
    ulong n = this->numClasses;

    this->truePositive.assign(n, 0);
    this->falsePositive.assign(n, 0);
    this->falseNegative.assign(n, 0);
    this->trueNegative.assign(n, 0);
    this->support.assign(n, 0);
    this->precision.assign(n, static_cast<T>(0));
    this->recall.assign(n, static_cast<T>(0));
    this->f1Score.assign(n, static_cast<T>(0));

    // Safe unsigned division: a / b, or 0 when b == 0.
    auto safeDiv = [](ulong a, ulong b) -> T {
      return (b > 0) ? static_cast<T>(a) / static_cast<T>(b) : static_cast<T>(0);
    };

    ulong sumTP = 0;
    ulong sumFP = 0;
    ulong sumFN = 0;

    for (ulong c = 0; c < n; c++) {
      ulong tp = this->matrix[c * n + c];

      // Column c: all samples predicted as c (TP + FP)
      ulong colSum = 0;

      for (ulong r = 0; r < n; r++) {
        colSum += this->matrix[r * n + c];
      }

      // Row c: all samples whose actual class is c (TP + FN)
      ulong rowSum = 0;

      for (ulong p = 0; p < n; p++) {
        rowSum += this->matrix[c * n + p];
      }

      ulong fp = colSum - tp;
      ulong fn = rowSum - tp;
      ulong tn = this->totalSamples - tp - fp - fn;

      this->truePositive[c] = tp;
      this->falsePositive[c] = fp;
      this->falseNegative[c] = fn;
      this->trueNegative[c] = tn;
      this->support[c] = rowSum;

      T prec = safeDiv(tp, tp + fp);
      T rec = safeDiv(tp, tp + fn);
      T f1 = (prec + rec > static_cast<T>(0)) ? static_cast<T>(2) * prec * rec / (prec + rec) : static_cast<T>(0);

      this->precision[c] = prec;
      this->recall[c] = rec;
      this->f1Score[c] = f1;

      sumTP += tp;
      sumFP += fp;
      sumFN += fn;
    }

    this->accuracy = safeDiv(sumTP, this->totalSamples);
    this->microPrecision = safeDiv(sumTP, sumTP + sumFP);
    this->microRecall = safeDiv(sumTP, sumTP + sumFN);
    this->microF1 =
      (this->microPrecision + this->microRecall > static_cast<T>(0))
        ? static_cast<T>(2) * this->microPrecision * this->microRecall / (this->microPrecision + this->microRecall)
        : static_cast<T>(0);

    // Macro: unweighted mean over classes. Weighted: support-weighted mean.
    T macroP = 0;
    T macroR = 0;
    T macroF = 0;
    T weightedP = 0;
    T weightedR = 0;
    T weightedF = 0;
    ulong totalSupport = 0;

    for (ulong c = 0; c < n; c++) {
      macroP += this->precision[c];
      macroR += this->recall[c];
      macroF += this->f1Score[c];

      ulong sup = this->support[c];
      weightedP += this->precision[c] * static_cast<T>(sup);
      weightedR += this->recall[c] * static_cast<T>(sup);
      weightedF += this->f1Score[c] * static_cast<T>(sup);
      totalSupport += sup;
    }

    if (n > 0) {
      this->macroPrecision = macroP / static_cast<T>(n);
      this->macroRecall = macroR / static_cast<T>(n);
      this->macroF1 = macroF / static_cast<T>(n);
    }

    this->weightedPrecision = (totalSupport > 0) ? weightedP / static_cast<T>(totalSupport) : static_cast<T>(0);
    this->weightedRecall = (totalSupport > 0) ? weightedR / static_cast<T>(totalSupport) : static_cast<T>(0);
    this->weightedF1 = (totalSupport > 0) ? weightedF / static_cast<T>(totalSupport) : static_cast<T>(0);
  }

  //===================================================================================================================//
  //-- Free function: merge --//
  //===================================================================================================================//

  template <typename T>
  void mergeConfusionMatrix(ConfusionMatrix<T>& dst, const ConfusionMatrix<T>& src)
  {
    if (src.empty()) {
      return;
    }

    dst.ensureSized(src.numClasses);

    for (size_t i = 0; i < src.matrix.size(); i++) {
      dst.matrix[i] += src.matrix[i];
    }

    dst.totalSamples += src.totalSamples;
  }
}

//===================================================================================================================//
//-- Explicit instantiations (match the T types the Core classes instantiate) --//
//===================================================================================================================//

template class Common::ConfusionMatrix<int>;
template class Common::ConfusionMatrix<double>;
template class Common::ConfusionMatrix<float>;

template void Common::mergeConfusionMatrix<int>(Common::ConfusionMatrix<int>&, const Common::ConfusionMatrix<int>&);
template void Common::mergeConfusionMatrix<double>(Common::ConfusionMatrix<double>&,
                                                   const Common::ConfusionMatrix<double>&);
template void Common::mergeConfusionMatrix<float>(Common::ConfusionMatrix<float>&,
                                                  const Common::ConfusionMatrix<float>&);
