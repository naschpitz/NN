#include "NN-CLI_SummaryTable.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/ioctl.h>
#include <unistd.h>
#include <vector>

namespace NN_CLI
{

  namespace
  {
    ulong detectTerminalWidth()
    {
      struct winsize ws;

      if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0)
        return static_cast<ulong>(ws.ws_col);

      if (ioctl(STDIN_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0)
        return static_cast<ulong>(ws.ws_col);

      return 120;
    }
  }

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
    auto lines = collect(title, rows);

    for (const auto& l : lines)
      std::cout << l << "\n";
  }

  //===================================================================================================================//

  std::vector<std::string> SummaryTable::collect(const std::string& title, const std::vector<SummaryRow>& rows)
  {
    std::vector<std::string> lines;

    ulong keyW = 0;
    ulong contentValueW = 0;

    for (const auto& r : rows) {
      if (!r.key.empty())
        keyW = std::max(keyW, r.key.size());

      if (!r.value.empty())
        contentValueW = std::max(contentValueW, r.value.size());
    }

    keyW = std::max(keyW, title.size());
    keyW = std::max(keyW, 6UL);
    contentValueW = std::max(contentValueW, 6UL);

    ulong termWidth = detectTerminalWidth();
    ulong containerWidth = termWidth > 5 ? termWidth - 5 : termWidth;
    ulong tableOverhead = keyW + 7;
    ulong valueW = containerWidth > tableOverhead ? containerWidth - tableOverhead : contentValueW;
    valueW = std::max(valueW, contentValueW);

    ulong totalInner = keyW + 3 + valueW;

    if (totalInner < title.size())
      totalInner = title.size();

    std::string sep = "+" + std::string(keyW + 2, '-') + "+" + std::string(valueW + 2, '-') + "+";

    lines.push_back("");
    lines.push_back(sep);

    ulong titleSpace = totalInner + 2;
    ulong pad = titleSpace - title.size();
    ulong padLeft = pad / 2;
    ulong padRight = pad - padLeft;
    lines.push_back("|" + std::string(padLeft, ' ') + title + std::string(padRight, ' ') + "|");

    lines.push_back(sep);

    for (const auto& r : rows) {
      if (r.key.empty()) {
        lines.push_back(sep);
        continue;
      }

      std::string truncatedValue = r.value;

      if (truncatedValue.size() > valueW)
        truncatedValue = truncatedValue.substr(0, valueW);

      std::ostringstream oss;
      oss << "| " << std::left << std::setw(static_cast<int>(keyW)) << r.key << " | " << std::left
          << std::setw(static_cast<int>(valueW)) << truncatedValue << " |";
      lines.push_back(oss.str());
    }

    lines.push_back(sep);

    return lines;
  }

} // namespace NN_CLI
