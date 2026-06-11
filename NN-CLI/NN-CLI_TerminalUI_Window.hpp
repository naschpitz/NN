#ifndef NN_CLI_TERMINALUI_WINDOW_HPP
#define NN_CLI_TERMINALUI_WINDOW_HPP

#include "NN-CLI_TerminalUI_Widget.hpp"

#include <atomic>
#include <memory>
#include <vector>

// Forward-declare ncurses WINDOW to avoid pulling in <curses.h> (which defines
// a `timeout` macro that conflicts with Qt's QTimer::timeout).
struct _win_st;
typedef struct _win_st WINDOW;

namespace NN_CLI
{

  // Top-level terminal window that manages the ncurses screen lifecycle and
  // owns a collection of child widgets.  Handles terminal resize events and
  // propagates geometry changes down the widget tree.
  //
  // Lifecycle: call init() to start ncurses, shutdown() to tear it down.
  // Children are added via addChild() and owned via unique_ptr.
  //
  // The default layoutChildren() gives every child the full window area.
  // Override it in a subclass to implement custom layout strategies (e.g.
  // the Training/Epochs/Config/Timing four-panel grid).

  class TerminalUI_Window : public TerminalUI_Widget
  {
    public:
      //-- Ctors / Dtors --//

      TerminalUI_Window();

      ~TerminalUI_Window() override;

      //-- Lifecycle --//

      // Initialize ncurses (cbreak, noecho, keypad, colors).  Returns false on
      // failure (e.g. no TTY attached).
      bool init();

      // Tear down ncurses and release all children.
      void shutdown();

      bool isInitialized() const
      {
        return this->initialized;
      }

      // Called from the SIGWINCH handler to schedule a deferred resize.
      // Safe to call from a signal handler (sets an atomic flag only).
      void requestResize();

      //-- Terminal dimensions --//

      int getRows() const
      {
        return this->rows;
      }

      int getCols() const
      {
        return this->cols;
      }

      //-- Child management --//

      // Take ownership of a child widget.  The child will be drawn and resized
      // together with the window.
      void addChild(std::unique_ptr<TerminalUI_Widget> child);

      // Release and return the child at the given index, or nullptr if the
      // index is out of range.
      std::unique_ptr<TerminalUI_Widget> removeChild(int index);

      int childCount() const
      {
        return static_cast<int>(this->children.size());
      }

      TerminalUI_Widget* getChild(int index) const;

      //-- Layout --//

      // Reposition and resize all children according to the current window
      // geometry.  Override in subclasses to implement custom layout strategies.
      virtual void layoutChildren();

      //-- Widget overrides --//

      void draw() override;
      void resize(int width, int height, int x, int y) override;
      bool handleEvent(int ch) override;

    protected:
      //-- Members --//

      bool initialized = false;
      int rows = 0;
      int cols = 0;
      std::atomic<bool> resizeRequested{false};
      std::vector<std::unique_ptr<TerminalUI_Widget>> children;
  };

} // namespace NN_CLI

#endif // NN_CLI_TERMINALUI_WINDOW_HPP
