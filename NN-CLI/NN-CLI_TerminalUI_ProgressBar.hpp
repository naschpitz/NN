#ifndef NN_CLI_TERMINALUI_PROGRESSBAR_HPP
#define NN_CLI_TERMINALUI_PROGRESSBAR_HPP

#include "NN-CLI_Types.hpp"

#include <string>
#include <vector>

// Forward-declare ncurses WINDOW to avoid pulling in <curses.h> (which defines
// a `timeout` macro that conflicts with Qt's QTimer::timeout).
struct _win_st;
typedef struct _win_st WINDOW;

namespace NN_CLI
{

  // Generic terminal progress-bar rendering component.  Provides ncurses and
  // stdout rendering primitives for single and segmented (multi-segment)
  // progress bars.  Has NO knowledge of training-specific concepts like
  // ProgressInfo, epochs, or GPU indices — those concerns live in ProgressBar.
  //
  // Layout constants (label width, bracket width, right padding, etc.) are
  // shared across all bars so they line up vertically in the Training panel.

  class TerminalUI_ProgressBar
  {
    public:
      //-- Ctors --//

      TerminalUI_ProgressBar();

      //-- ncurses Rendering --//

      // Render a single progress bar on line 0 of `win`:
      //   "label             [████████░░░░░░░░░]  XX.XXX%"
      // The label is left-padded to a fixed width so bars align vertically.
      // `fraction` is clamped to [0.0, 1.0].
      void renderSingleBar(WINDOW* win, const std::string& label, float fraction);

      // Render a segmented (multi-segment) progress bar on line 0 of `win`:
      //   "label             [████░░│████████░░]  XX.XXX% (0: XX% | 1: XX%)"
      // Each segment is an equal-width sub-bar separated by "│".  When there
      // is only one fraction the layout is identical to renderSingleBar
      // (no per-segment suffix is emitted).  All fractions are clamped.
      void renderMultiBar(WINDOW* win, const std::string& label, const std::vector<float>& fractions);

      // Render `text` on line 1 of `win` (the sub-line below the bar).
      // If `colorPair` > 0 the text is drawn in that ncurses color pair.
      // The rest of the line is cleared to avoid stale content.
      void renderSubLine(WINDOW* win, const std::string& text, int colorPair = 0);

      // Clear line 1 of `win` (the sub-line below the bar).
      void clearSubLine(WINDOW* win);

      //-- stdout Rendering --//

      // Simple loading progress bar printed to std::cout (no ncurses).
      // Throttled by `progressReports` — only updates every total/progressReports
      // steps.  Prints a final newline when current == total.
      static void printLoadingProgress(const std::string& label, size_t current, size_t total,
                                       ulong progressReports = 1000, int barWidth = 40);
  };

} // namespace NN_CLI

#endif // NN_CLI_TERMINALUI_PROGRESSBAR_HPP
