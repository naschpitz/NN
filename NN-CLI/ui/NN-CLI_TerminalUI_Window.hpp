#ifndef NN_CLI_TERMINALUI_WINDOW_HPP
#define NN_CLI_TERMINALUI_WINDOW_HPP

#include "NN-CLI_TerminalUI_Widget.hpp"

#include <atomic>
#include <memory>
#include <string>
#include <vector>

// Forward-declare ncurses WINDOW to avoid pulling in <curses.h> (which defines
// a `timeout` macro that conflicts with Qt's QTimer::timeout).
struct _win_st;
typedef struct _win_st WINDOW;

class QTimer;

namespace NN_CLI
{

  // Top-level terminal window that manages the ncurses screen lifecycle and
  // owns a collection of child widgets.  Handles terminal resize events and
  // propagates geometry changes down the widget tree.
  //
  // Lifecycle: call init() to start ncurses, shutdown() to tear it down.
  // Children are added via addChild() and owned via unique_ptr.
  //
  // Threading model: after startUiTimer(), the main-thread QTimer (driven by
  // the QCoreApplication event loop) owns all ncurses work — rendering, input,
  // and resize handling.  It fires ~every 10 ms, drains buffered input
  // (non-blocking), and repaints only when the widget tree is dirty.  Every
  // mutating setter on a widget raises that widget's dirty flag, and
  // isDirtyTree() propagates it up, so worker threads simply emit signals —
  // queued delivery brings the update to the main thread, the timer sees the
  // dirty flag on its next tick and renders.  No mutex is needed because all
  // view mutations arrive on the main thread via the event loop.
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

      // Start the UI timer on the main thread (fires ~every 10 ms).  Call
      // after a successful init(); no-op when the TUI failed to initialize or
      // the timer already runs.  Requires a running QCoreApplication event
      // loop (the timer's timeout signal is delivered through it).
      void startUiTimer();

      // Stop the UI timer.  Called automatically by shutdown().
      void stopUiTimer();

      // Called from the SIGWINCH handler to schedule a deferred resize.
      // Safe to call from a signal handler (sets an atomic flag only); the UI
      // thread picks it up via needsRepaint() on its next tick.
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
      // Storage and implementation live in TerminalUI_Widget; re-exposed here so
      // window callers can build the child list exactly as before.

      using TerminalUI_Widget::addChild;
      using TerminalUI_Widget::childCount;
      using TerminalUI_Widget::getChild;
      using TerminalUI_Widget::removeChild;

      //-- Shortcut bar --//

      void setShortcutBar(const std::string& text);
      const std::string& getShortcutBar() const;

      //-- Title bar --//

      void setTitleBar(const std::string& text, int colorPair);
      const std::string& getTitleBar() const;
      int getTitleBarColor() const;

      //-- Layout --//

      // Reposition and resize all children according to the current window
      // geometry.  Override in subclasses to implement custom layout strategies.
      virtual void layoutChildren();

      //-- Widget overrides --//

      void draw() override;
      void resize(int width, int height, int x, int y) override;
      bool handleEvent(int ch) override;

    protected:
      //-- Hooks --//

      // Called before each render() in draw().  Subclasses override to
      // update visual state (e.g. panel highlight colors) that depends
      // on input-driven state changes (Tab cycling, scroll).  Called
      // before the first render and again before the re-render that
      // follows any consumed input events.
      virtual void preRender() {}

      //-- Shortcut bar --//

      int shortcutBarHeight() const;

      //-- Title bar --//

      int titleBarHeight() const;

      //-- Members --//

      bool initialized = false;
      int rows = 0;
      int cols = 0;
      std::atomic<bool> resizeRequested{false};

    private:
      //-- Methods --//

      // Render all children to stdscr and flush.
      void render();

      // Non-blocking input poll: read one key from getch() and dispatch it
      // to handleEvent().  Returns true if an event was consumed (caller
      // should re-render).
      bool pollAndDispatchInput();

      // True when a repaint is warranted this tick: the widget tree is dirty
      // or a terminal resize was requested by the SIGWINCH handler.
      bool needsRepaint() const;

      //-- Shortcut bar --//

      void drawShortcutBar() const;

      //-- Title bar --//

      void drawTitleBar() const;

      //-- Members --//

      std::unique_ptr<QTimer> uiTimer;

      std::string shortcutBar;
      std::string titleBar;
      int titleBarColor = 0;
  };

} // namespace NN_CLI

#endif // NN_CLI_TERMINALUI_WINDOW_HPP
