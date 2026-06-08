#ifndef _WORKER_H
#define _WORKER_H

#include "_Types.hpp"
#include "Common/Common_CostFunctionConfig.hpp"

//===================================================================================================================//

namespace ANN
{
  using namespace Common;
  template <typename T>
  class Worker
  {
    public:
      virtual ~Worker() = default;

    protected:
      CostFunctionConfig<T> costFunctionConfig;

      //-- Loss calculation (shared by CPU and GPU workers) --//
      T calculateLoss(const Output<T>& predicted, const Output<T>& expected);
  };
}

#endif // _WORKER_H
