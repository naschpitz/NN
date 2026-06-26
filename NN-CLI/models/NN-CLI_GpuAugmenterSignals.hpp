#ifndef NN_CLI_GPUAUGMENTERSIGNALS_HPP
#define NN_CLI_GPUAUGMENTERSIGNALS_HPP

#include <QObject>

namespace NN_CLI
{
  //===================================================================================================================//

  // Non-template QObject signals hub owned by GpuAugmenterPool via composition.
  class GpuAugmenterPoolSignals : public QObject
  {
      Q_OBJECT

    public:
      //-- Constructors --//
      explicit GpuAugmenterPoolSignals(QObject* parent = nullptr) : QObject(parent) {}

    signals:
      void timingUpdate(bool active);

    private:
      friend class GpuAugmenterPool;
  };

} // namespace NN_CLI

#endif // NN_CLI_GPUAUGMENTERSIGNALS_HPP
