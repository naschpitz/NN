#ifndef CNN_INPUTPROVIDER_HPP
#define CNN_INPUTPROVIDER_HPP

#include "CNN_Types.hpp"

#include <functional>
#include <sys/types.h>

//===================================================================================================================//

namespace CNN
{
  // Lazy supplier used by the streaming predict(): given a batch size and a
  // 0-based batch index, returns a NON-OWNING view over the corresponding
  // chunk of inputs. The last batch may be shorter than batchSize. The caller
  // must keep the backing storage alive for the view's lifetime (e.g. an
  // in-memory Inputs<T> that outlives the whole predict() call). Mirrors
  // SampleProvider but without the expected-output side, since predict has no
  // labels to compare against.
  template <typename T>
  using InputProvider = std::function<InputsView<T>(ulong batchSize, ulong batchIndex)>;
}

//===================================================================================================================//

#endif // CNN_INPUTPROVIDER_HPP
