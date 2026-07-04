#ifndef NN_CLI_CALIBRATERESULT_HPP
#define NN_CLI_CALIBRATERESULT_HPP

#include <cstddef>
#include <string>

//===================================================================================================================//

namespace NN_CLI
{

  //===================================================================================================================//

  // Aggregate result of a calibration run, emitted via
  // RunnerBase::calibrateFinished so the CalibrateController can render the
  // threshold and ID/OOD acceptance/rejection figures in the TUI.  Computed
  // by the typed Runner from the free-energy distributions of the ID and OOD
  // image samples; not a Core-level concept, so it lives here rather than in
  // Common.
  struct CalibrateResult {
      float freeEnergyThreshold = 0.0f;
      double idPercentileUsed = 0.0;
      std::size_t idCount = 0;
      std::size_t oodCount = 0;
      std::size_t idAccepted = 0;
      std::size_t oodRejected = 0;
      double idAcceptanceRate = 0.0;
      double oodRejectionRate = 0.0;
      double durationSeconds = 0.0;
      std::string durationFormatted;
      std::string outputPath;
  };

  //===================================================================================================================//

} // namespace NN_CLI

//===================================================================================================================//

#endif // NN_CLI_CALIBRATERESULT_HPP
