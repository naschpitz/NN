#ifndef NN_CLI_TERMINALUI_WINDOW_HPP
#define NN_CLI_TERMINALUI_WINDOW_HPP

#include "NN-CLI_TerminalUI_Widget.hpp"

#include <QMutex>

#include <atomic>
#include <memory>
#include <string>
#include <vector>

// Forward-declare ncurses WINDOW to avoid pulling in <curses.h> (which defines
// a `timeout` macro that conflicts with Qt's QTimer::timeout).
struct _win_st;
typedef struct _win_st WINDOW;

class QThread;

namespace NN_CLI
{

  // Top-level terminal window that manages the ncurses screen lifecycle and
  // owns a collection of child widgets.  Handles terminal resize events and
  // propagates geometry changes down the widget tree.
  //
  // Lifecycle: call init() to start ncurses, shutdown() to tear it down.
  // Children are added via addChild() and owned via unique_ptr.
  //
  // Threading model: after startUiThread(), a single UI thread owns all
  // ncurses work — rendering, input, and resize handling.  It sleeps ~10 ms
  // between ticks and repaints only when something changed: an observer
  // raised the redraw flag via requestRedraw() after mutating widget data,
  // a keystroke arrived, or SIGWINCH fired.  Worker threads never call
  // draw(); they update widget data under getMutex() and then
  // requestRedraw(), and the UI thread renders the change on the next tick.
  // A flag check plus a non-blocking getch at ~100 Hz is negligible CPU (the
  // old hog was the full repaint, not the wake), so an idle terminal — and
  // the post-completion dismiss wait — stays cheap instead of repainting at
  // a fixed frame rate.
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

      // Start the dedicated UI thread.  Call after a successful init();
      // no-op when the TUI failed to initialize or the thread already runs.
      void startUiThread();

      // Stop and join the UI thread.  Called automatically by shutdown();
      // must not be called while holding getMutex() (the UI thread takes it
      // each frame and the join would deadlock).
      void stopUiThread();

      // Serializes the widget tree (and ncurses) between the UI thread and
      // the data-updating threads.  Lock it around any mutation of window or
      // panel content.
      QRecursiveMutex& getMutex()
      {
        return this->uiMutex;
      }

      // Called from the SIGWINCH handler to schedule a deferred resize.
      // Safe to call from a signal handler (sets an atomic flag only).
      void requestResize();

      // Set the redraw flag so the UI thread renders on its next tick.  Used
      // by the content-commit methods after they mutate widget data under
      // getMutex().  Setting an atomic bool is both thread- and async-signal-
      // safe, so requestResize() also calls this from its signal-handler path.
      void requestRedraw();

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
      std::vector<std::unique_ptr<TerminalUI_Widget>> children;

    private:
      //-- Methods --//

      // Render all children to stdscr and flush.
      void render();

      // Non-blocking input poll: read one key from getch() and dispatch it
      // to handleEvent().  Returns true if an event was consumed (caller
      // should re-render).
      bool pollAndDispatchInput();

      // UI thread body: sleep ~10 ms, drain buffered input, and run draw()
      // (render + input + resize) under the mutex only when the redraw flag
      // was set or input was consumed.  Loops until stopUiThread() clears the
      // flag (the thread notices on its next tick).
      void uiThreadLoop();

      //-- Shortcut bar --//

      void drawShortcutBar() const;

      //-- Title bar --//

      void drawTitleBar() const;

      //-- Members --//

      std::unique_ptr<QThread> uiThread;
      std::atomic<bool> uiThreadRunning{false};
      QRecursiveMutex uiMutex;

      // Set by requestRedraw() (from worker threads or the SIGWINCH handler)
      // and cleared by the UI thread on the tick it renders.  At rest it
      // stays false, so the UI thread renders nothing and spends ~0 CPU.
      std::atomic<bool> redrawRequested{false};
      std::string shortcutBar;
      std::string titleBar;
      int titleBarColor = 0;
  };

} // namespace NN_CLI

#endif // NN_CLI_TERMINALUI_WINDOW_HPP
