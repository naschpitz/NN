#ifndef NN_CLI_TERMINALUI_HPP
#define NN_CLI_TERMINALUI_HPP

#include <functional>
#include <mutex>
#include <string>
#include <vector>

struct _win_st;
typedef struct _win_st WINDOW;

namespace NN_CLI
{

  class TerminalUI
  {
    public:
      TerminalUI();
      ~TerminalUI();

      bool init();
      void shutdown();

      bool isInitialized() const
      {
        return this->initialized_;
      }

      int rows() const
      {
        return this->rows_;
      }

      int cols() const
      {
        return this->cols_;
      }

      int timingWidth() const
      {
        return this->timingWidth_;
      }

      void setConfigLines(const std::vector<std::string>& lines);
      void refreshConfigPanel();

      WINDOW* progressWindow() const
      {
        return this->progressWin_;
      }

      WINDOW* loadingWindow() const
      {
        return this->loadingWin_;
      }

      void setTimingLines(const std::vector<std::string>& lines);
      void addEpochLine(const std::string& line);
      void redraw();
      void refresh();
      void pollInput();

      void setResizeCallback(std::function<void()> callback)
      {
        this->resizeCallback_ = std::move(callback);
      }

      std::recursive_mutex& mutex()
      {
        return this->mutex_;
      }

    private:
      void layout();
      void drawPanelFrame(int y, int h, const char* title, int titleColor = 2);
      void drawPanelFrame(int y, int h, int x, int w, const char* title, int titleColor);
      void drawAllPanels();
      bool handleScrollInput(int ch);

      WINDOW* progressWin_ = nullptr;
      WINDOW* loadingWin_ = nullptr;
      WINDOW* timingWin_ = nullptr;

      int rows_ = 0;
      int cols_ = 0;
      bool initialized_ = false;

      int leftWidth_ = 0;
      int timingWidth_ = 0;

      int configY_ = 0;
      int configH_ = 0;
      int trainingY_ = 0;
      int trainingH_ = 0;
      int epochsY_ = 0;
      int epochsH_ = 0;
      int helpY_ = 0;

      int configScroll_ = 0;
      int timingScroll_ = 0;

      int activePanel_ = 0; // 0=Config, 1=Epochs, 2=Timing
      int epochScroll_ = 0;
      bool epochsAutoScroll_ = true;

      std::recursive_mutex mutex_;
      std::function<void()> resizeCallback_;
      bool resized_ = false;

      std::vector<std::string> configLines_;
      std::vector<std::string> timingLines_;
      std::vector<std::string> epochLines_;
  };

} // namespace NN_CLI

#endif // NN_CLI_TERMINALUI_HPP
