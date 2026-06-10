#ifndef COMMON_EPOCHCOMPLETIONEVENT_HPP
#define COMMON_EPOCHCOMPLETIONEVENT_HPP

#include <functional>
#include <sys/types.h>

//===================================================================================================================//

namespace Common
{
  // Epoch-completion information passed to the epoch-completed callback.
  // Reports the just-finished epoch with a 0-based index, matching
  // EpochRecord::epoch and the serialized epochs[] history — unlike
  // TrainingProgressEvent, whose high-frequency currentEpoch is 1-based.
  template <typename T>
  struct EpochCompletionEvent {
      ulong epoch; // 0-based index of the completed epoch
      ulong totalEpochs; // total epochs the run will perform
      T epochLoss; // average training loss for this epoch
      bool isNewBest = false; // a core-internal monitor flagged a new best
      bool stoppedEarly = false; // a core-internal monitor requested an early stop
  };

  // Invoked exactly once per completed epoch, after the epoch's record is
  // appended to the training history. Distinct from the per-sample
  // TrainingCallback that drives live progress display: this is the hook for
  // epoch-boundary work (validation, checkpointing, monitoring).
  template <typename T>
  using EpochCompletedCallback = std::function<void(const EpochCompletionEvent<T>&)>;
}

//===================================================================================================================//

#endif // COMMON_EPOCHCOMPLETIONEVENT_HPP
