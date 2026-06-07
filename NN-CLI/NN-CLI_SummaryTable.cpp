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

    // Approximate display width accounting for UTF-8 multi-byte sequences.
    // Each UTF-8 continuation byte (10xxxxxx) is 0 display columns,
    // so display width = string bytes - continuation bytes.
    ulong displayWidth(const std::string& s)
    {
      ulong w = 0;

      for (unsigned char c : s) {
        if ((c & 0xC0) != 0x80) // Not a UTF-8 continuation byte
          ++w;
      }

      return w;
    }

    // Split text into lines that each fit within maxWidth, breaking at word boundaries.
    // Words longer than maxWidth are force-broken at maxWidth.
    std::vector<std::string> wordWrap(const std::string& text, ulong maxWidth)
    {
      std::vector<std::string> lines;

      if (text.empty()) {
        lines.emplace_back();
        return lines;
      }

      if (maxWidth == 0) {
        lines.push_back(text);
        return lines;
      }

      std::istringstream stream(text);
      std::string word;
      std::string currentLine;

      while (stream >> word) {
        // If the word itself is longer than maxWidth, force-break it
        if (displayWidth(word) > maxWidth) {
          if (!currentLine.empty()) {
            lines.push_back(currentLine);
            currentLine.clear();
          }

          for (ulong i = 0; i < word.size(); i += maxWidth)
            lines.push_back(word.substr(i, maxWidth));
          continue;
        }

        if (currentLine.empty()) {
          currentLine = word;
        } else if (displayWidth(currentLine) + 1 + displayWidth(word) <= maxWidth) {
          currentLine += " " + word;
        } else {
          lines.push_back(currentLine);
          currentLine = word;
        }
      }

      if (!currentLine.empty())
        lines.push_back(currentLine);

      if (lines.empty())
        lines.emplace_back();

      return lines;
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

  std::vector<std::string> SummaryTable::collect(const std::string& title, const std::vector<SummaryRow>& rows,
                                                 ulong maxWidth)
  {
    ulong keyW = 0;

    for (const auto& r : rows) {
      if (!r.key.empty())
        keyW = std::max(keyW, r.key.size());
    }

    keyW = std::max(keyW, title.size());
    keyW = std::max(keyW, 6UL);

    ulong termWidth = (maxWidth > 0) ? maxWidth : detectTerminalWidth();
    ulong containerWidth = termWidth > 5 ? termWidth - 5 : termWidth;
    ulong tableOverhead = keyW + 7;
    ulong valueW = containerWidth > tableOverhead ? containerWidth - tableOverhead : 6UL;
    valueW = std::max(valueW, 6UL);

    return collect(title, rows, maxWidth, keyW, valueW);
  }

  //===================================================================================================================//

  std::vector<std::string> SummaryTable::collect(const std::string& title, const std::vector<SummaryRow>& rows,
                                                 ulong maxWidth, ulong keyW, ulong valueW)
  {
    std::vector<std::string> lines;
    ulong termWidth = (maxWidth > 0) ? maxWidth : detectTerminalWidth();
    ulong containerWidth = termWidth > 5 ? termWidth - 5 : termWidth;
    ulong tableOverhead = keyW + 7;
    ulong maxValueW = containerWidth > tableOverhead ? containerWidth - tableOverhead : 6UL;
    valueW = std::min(valueW, maxValueW);
    valueW = std::max(valueW, 6UL);
    keyW = std::max(keyW, title.size());
    keyW = std::max(keyW, 6UL);

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

      // Word-wrap the value into chunks so long values don't force an overly wide table
      auto wrappedLines = wordWrap(r.value, valueW);

      for (ulong wi = 0; wi < wrappedLines.size(); wi++) {
        const auto& val = wrappedLines[wi];
        ulong valDW = displayWidth(val);
        ulong pad = (valDW < valueW) ? valueW - valDW : 0;

        std::ostringstream oss;

        if (wi == 0) {
          // First line: show the key
          oss << "| " << std::left << std::setw(static_cast<int>(keyW)) << r.key << " | " << val
              << std::string(pad, ' ') << " |";
        } else {
          // Continuation lines: blank key column, just the wrapped value
          oss << "| " << std::left << std::setw(static_cast<int>(keyW)) << ""
              << " | " << val << std::string(pad, ' ') << " |";
        }

        lines.push_back(oss.str());
      }
    }

    lines.push_back(sep);

    return lines;
  }

  //===================================================================================================================//

  std::vector<std::string> SummaryTable::collectSections(const std::vector<Section>& sections, ulong maxWidth)
  {
    ulong keyW = 0;

    for (const auto& sec : sections) {
      keyW = std::max(keyW, sec.title.size());

      for (const auto& r : sec.rows) {
        if (!r.key.empty())
          keyW = std::max(keyW, r.key.size());
      }
    }

    keyW = std::max(keyW, 6UL);

    ulong termWidth = (maxWidth > 0) ? maxWidth : detectTerminalWidth();
    ulong containerWidth = termWidth > 5 ? termWidth - 5 : termWidth;
    ulong tableOverhead = keyW + 7;
    ulong valueW = containerWidth > tableOverhead ? containerWidth - tableOverhead : 6UL;
    valueW = std::max(valueW, 6UL);

    std::vector<std::string> allLines;

    for (const auto& sec : sections) {
      auto secLines = collect(sec.title, sec.rows, maxWidth, keyW, valueW);
      allLines.insert(allLines.end(), secLines.begin(), secLines.end());
    }

    return allLines;
  }

} // namespace NN_CLI
