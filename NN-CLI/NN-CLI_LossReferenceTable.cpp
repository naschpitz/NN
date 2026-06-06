#include "NN-CLI_LossReferenceTable.hpp"
#include "NN-CLI_SummaryTable.hpp"

#include <cmath>
#include <iomanip>
#include <sstream>
#include <vector>

namespace NN_CLI
{

  std::vector<std::string> LossReferenceTable::collect(ulong numClasses, ulong maxWidth)
  {
    auto rows = collectRows(numClasses);

    if (rows.empty())
      return {};

    std::string title = "Loss Reference (" + std::to_string(numClasses) + " classes)";
    return SummaryTable::collect(title, rows, maxWidth);
  }

  //===================================================================================================================//

  std::vector<SummaryRow> LossReferenceTable::collectRows(ulong numClasses)
  {
    if (numClasses < 2)
      return {};

    float randomConfidence = 1.0f / static_cast<float>(numClasses);

    struct RefPoint {
        float confidence;
        std::string interpretation;
        bool isRandom;
    };

    std::vector<RefPoint> points = {
      {1.00f, "Perfect", false},
      {0.90f, "Very confident", false},
      {0.70f, "Fairly confident", false},
      {0.50f, "50% confidence", false},
      {randomConfidence, "Random (" + std::to_string(numClasses) + "-class)", true},
      {0.10f, "Confident, wrong", false},
      {0.05f, "Very wrong", false},
      {0.01f, "Extremely wrong", false},
    };

    std::vector<RefPoint> filtered;

    for (const auto& p : points) {
      bool duplicate = false;

      if (!p.isRandom) {
        if (std::fabs(p.confidence - randomConfidence) < 0.01f)
          duplicate = true;
      }

      if (!duplicate)
        filtered.push_back(p);
    }

    std::vector<SummaryRow> rows;

    for (const auto& p : filtered) {
      float loss = (p.confidence >= 1.0f) ? 0.0f : -std::log(p.confidence);

      std::ostringstream confStr;
      confStr << std::fixed << std::setprecision(1) << (p.confidence * 100.0f) << "%";

      std::ostringstream lossStr;
      lossStr << std::fixed << std::setprecision(6) << loss;

      rows.push_back({confStr.str(), lossStr.str() + "    " + p.interpretation});
    }

    return rows;
  }

} // namespace NN_CLI
