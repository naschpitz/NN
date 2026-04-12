#include "NN-CLI_SummaryTable.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>

namespace NN_CLI
{

  //===================================================================================================================//

  std::string SummaryTable::formatWithCommas(ulong value)
  {
    std::string str = std::to_string(value);
    int insertPos = static_cast<int>(str.length()) - 3;

    while (insertPos > 0) {
      str.insert(insertPos, ",");
      insertPos -= 3;
    }

    return str;
  }

  //===================================================================================================================//

  void SummaryTable::print(const std::string& title, const std::vector<SummaryRow>& rows)
  {
    ulong keyW = title.size();
    ulong valueW = 0;

    for (const auto& r : rows) {
      if (r.key.empty())
        continue;

      keyW = std::max(keyW, r.key.size());
      valueW = std::max(valueW, r.value.size());
    }

    ulong totalInner = keyW + 3 + valueW;

    if (totalInner < title.size()) {
      valueW += title.size() - totalInner;
    }

    std::string sep = "+" + std::string(keyW + 2, '-') + "+" + std::string(valueW + 2, '-') + "+";
    ulong titleSpace = keyW + valueW + 5;

    std::cout << "\n";
    std::cout << sep << "\n";
    ulong pad = titleSpace - title.size();
    ulong padLeft = pad / 2;
    ulong padRight = pad - padLeft;
    std::cout << "|" << std::string(padLeft, ' ') << title << std::string(padRight, ' ') << "|\n";
    std::cout << sep << "\n";

    for (const auto& r : rows) {
      if (r.key.empty()) {
        std::cout << sep << "\n";
        continue;
      }

      std::cout << "| " << std::left << std::setw(static_cast<int>(keyW)) << r.key << " | "
                << std::setw(static_cast<int>(valueW)) << r.value << " |\n";
    }

    std::cout << sep << "\n";
    std::cout << "\n";
  }

} // namespace NN_CLI
