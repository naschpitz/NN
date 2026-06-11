#ifndef NN_CLI_TERMINALUI_PROGRESSBAR_HPP
#define NN_CLI_TERMINALUI_PROGRESSBAR_HPP

#include "NN-CLI_TerminalUI_Widget.hpp"
#include "NN-CLI_Types.hpp"

#include <string>
#include <vector>

// Forward-declare ncurses WINDOW to avoid pulling in <curses.h> (which defines
// a `timeout` macro that conflicts with Qt's QTimer::timeout).
struct _win_st;
typedef struct _win_st WINDOW;

namespace NN_CLI
{

  // Generic terminal progress-bar rendering widget.  Provides ncurses and
  // stdout rendering primitives for single and segmented (multi-segment)
  // progress bars.  Has NO knowledge of training-specific concepts like
  // epochs or GPU indices — the Controller maps those onto label/fraction
  // updates.
  //
  // Inherits from TerminalUI_Widget so it can be placed inside a Panel or
  // Window as part of the MVC view hierarchy.  The unified draw() override
  // renders the bar at the widget's (x, y) position on stdscr using the
  // stored label and fractions.
  //
  // The legacy methods that accept an explicit WINDOW* parameter remain
  // available for the stdout/console rendering paths still used by the
  // Loaders and Calibrate mode.

  class TerminalUI_ProgressBar : public TerminalUI_Widget
  {
    public:
      //-- Ctors --//

      TerminalUI_ProgressBar();

      //-- Widget data (for draw() mode) --//

      // Set the label and single-segment fraction for Widget-mode rendering.
      void setBarData(const std::string& label, float fraction);

      // Set the label and multi-segment fractions for Widget-mode rendering.
      void setBarData(const std::string& label, const std::vector<float>& fractions);

      // Set the sub-line text (rendered on row y+1).
      void setSubLineText(const std::string& text, int colorPair = 0);

      // Clear the sub-line text.
      void clearSubLineText();

      // Show or hide the bar.  A hidden bar occupies its layout slot but draws
      // nothing (used for the loading bar before any samples are loaded).
      void setVisible(bool visible);

      //-- Widget overrides --//

      // Render the progress bar at (x, y) on stdscr using stored bar data.
      void draw() override;

      // Update position and size.  Width controls the bar width; height is
      // typically 2 (bar row + sub-line row).
      void resize(int width, int height, int x, int y) override;

      //-- ncurses Rendering (legacy — explicit WINDOW*) --//

      // Render a single progress bar on line 0 of `win`:
      //   "label             [████████░░░░░░░░░]  XX.XXX%"
      // The label is left-padded to a fixed width so bars align vertically.
      // `fraction` is clamped to [0.0, 1.0].
      void renderSingleBar(WINDOW* win, const std::string& label, float fraction);

      // Render a segmented (multi-segment) progress bar on line 0 of `win`:
      //   "label             [████░░│████████░░]  XX.XXX% (0: XX% | 1: XX%)"
      // Each segment is an equal-width sub-bar separated by "|".  When there
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

    private:
      //-- Members --//

      std::string barLabel;
      std::vector<float> barFractions;
      std::string subLineText;
      int subLineColorPair = 0;
      bool visible = true;
  };

} // namespace NN_CLI

#endif // NN_CLI_TERMINALUI_PROGRESSBAR_HPP
