#ifndef CNN_CORESIGNALS_HPP
#define CNN_CORESIGNALS_HPP

#include "CNN_TimingCallback.hpp"

#include <QObject>
#include <sys/types.h>
#include <vector>

//===================================================================================================================//

namespace CNN
{
  // Forward declaration for friendship (Core<T> emits these signals directly).
  template <typename T>
  class Core;

  // Non-template QObject "signals hub" owned by Core<T>. Core<T> is a class
  // template (T = float/double) and moc cannot process Q_OBJECT in a template,
  // so the signal surface lives in this non-template class. Core<T> holds a
  // CoreSignals member and emits through it.
  //
  // Loss values use concrete double (lossless widening for both float and double
  // instantiations). Count/epoch/sample parameters use ulong to match the rest of
  // the library. timing/gpuProfile mirror the CNN-only TimingCallback and
  // GpuProfileCallback surfaces.
  //
  // NOTE(Phase 2): when connections flip to Qt::QueuedConnection, register
  // qRegisterMetaType<ulong> ("ulong"), qRegisterMetaType<CNN::TimingPhase> /
  // TimingEvent, and qRegisterMetaType<std::vector<GpuPhaseProfile>> once at
  // startup so queued delivery works.
  class CoreSignals : public QObject
  {
      Q_OBJECT

    public:
      //-- Constructors --//
      explicit CoreSignals(QObject* parent = nullptr) : QObject(parent) {}

    signals:
      //-- Training progress (per-sample) --//
      void trainProgress(ulong currentEpoch, ulong totalEpochs, ulong currentSample, ulong totalSamples,
                         double epochLoss, double sampleLoss, bool isNewBest, bool stoppedEarly, int gpuIndex,
                         int totalGPUs);

      //-- Epoch completion (once per finished epoch) --//
      void epochCompleted(ulong epoch, ulong totalEpochs, double epochLoss, bool isNewBest, bool stoppedEarly);

      //-- Test/predict progress --//
      void predictProgress(ulong current, ulong total);

      //-- Host-side timing phase boundaries (orchestrator + worker scope) --//
      void timing(TimingPhase phase, TimingEvent event, int gpuIndex);

      //-- GPU-measured per-sub-phase kernel profile (worker scope) --//
      void gpuProfile(const std::vector<GpuPhaseProfile>& profile, int gpuIndex);

    private:
      //-- Core<T> emits the signals directly (signals are protected, hence friendship) --//
      template <typename T>
      friend class Core;
  };
}

//===================================================================================================================//

#endif // CNN_CORESIGNALS_HPP
