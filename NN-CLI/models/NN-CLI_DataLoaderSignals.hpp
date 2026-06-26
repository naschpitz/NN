#ifndef NN_CLI_DATALOADERSIGNALS_HPP
#define NN_CLI_DATALOADERSIGNALS_HPP

#include <QObject>

namespace NN_CLI
{
  //-- SampleLoadType moved here so the signal can reference it without a circular include. --//
  enum class SampleLoadType { Train, Validation };

  //===================================================================================================================//

  // Non-template QObject signals hub owned by DataLoader<T> via composition.
  // moc cannot host Q_OBJECT in a class template, so the hub is a separate non-template class.
  class DataLoaderSignals : public QObject
  {
      Q_OBJECT

    public:
      //-- Constructors --//
      explicit DataLoaderSignals(QObject* parent = nullptr) : QObject(parent) {}

    signals:
      void loadingProgress(ulong current, ulong count, ulong batchIndex, ulong totalBatches, SampleLoadType loadType);

    private:
      template <typename SampleT>
      friend class DataLoader;
  };

} // namespace NN_CLI

#endif // NN_CLI_DATALOADERSIGNALS_HPP
