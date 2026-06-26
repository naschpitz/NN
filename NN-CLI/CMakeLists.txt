cmake_minimum_required(VERSION 3.14)

project(NN-CLI LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt6 REQUIRED COMPONENTS Core Concurrent)

# Add CNN as subdirectory — it brings ANN and OpenCLWrapper transitively via PUBLIC
if(NOT TARGET CNN)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../CNN ${CMAKE_CURRENT_BINARY_DIR}/CNN)
endif()

set(NNCLI_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ui
    ${CMAKE_CURRENT_SOURCE_DIR}/controllers
    ${CMAKE_CURRENT_SOURCE_DIR}/models
)

add_executable(NN-CLI
  main.cpp
  models/NN-CLI_ANNRunner.cpp
  models/NN-CLI_CalibrateRunner.cpp
  models/NN-CLI_CNNLoader.cpp
  models/NN-CLI_CNNRunner.cpp
  models/NN-CLI_Runner.cpp
  models/NN-CLI_RunnerSignals.hpp
  models/NN-CLI_LossReferenceTable.cpp
  models/NN-CLI_ModelSerializer.cpp
  models/NN-CLI_PredictSummary.cpp
  controllers/NN-CLI_App.cpp
  ui/NN-CLI_TerminalUI_Panel.cpp
  ui/NN-CLI_TerminalUI_Table.cpp
  ui/NN-CLI_TerminalUI_Window.cpp
  ui/NN-CLI_TerminalUI_TrainWindow.cpp
  ui/NN-CLI_TerminalUI_PredictWindow.cpp
  models/NN-CLI_TestSummary.cpp
  models/NN-CLI_TrainProfiler.cpp
  controllers/NN-CLI_TrainController.cpp
  controllers/NN-CLI_PredictController.cpp
  controllers/NN-CLI_TestController.cpp
  controllers/NN-CLI_CalibrateController.cpp
  models/NN-CLI_Utils.cpp
)

# ncurses for the training terminal UI
set(CURSES_NEED_WIDE TRUE)
set(CURSES_NEED_NCURSES TRUE)
find_package(Curses REQUIRED)
if(NOT CURSES_LIBRARY MATCHES "ncursesw")
  find_library(NCURSESW_LIBRARY NAMES ncursesw)
  if(NCURSESW_LIBRARY)
    set(CURSES_LIBRARIES ${NCURSESW_LIBRARY})
    message(STATUS "Forcing ncursesw: ${NCURSESW_LIBRARY}")
  endif()
endif()

# Object library for shared sources — compiled once and linked into both NN-CLI and test_nncli.
# Without this, 10 .cpp files are compiled twice (once per target), wasting build time.
add_library(nncli_common OBJECT
    models/NN-CLI_ANNLoader.cpp
    models/NN-CLI_CalibrateUtils.cpp
    models/NN-CLI_DataLoader.cpp
    models/NN-CLI_GpuAugmenter.cpp
    models/NN-CLI_DataSplitter.cpp
    models/NN-CLI_DataType.cpp
    models/NN-CLI_ImageLoader.cpp
    models/NN-CLI_Loader.cpp
    models/NN-CLI_ModelPackage.cpp
    ui/NN-CLI_TerminalUI_ProgressBar.cpp
    ui/NN-CLI_TerminalUI_Widget.cpp
    models/NN-CLI_SummaryTable.cpp
    models/NN-CLI_TrainSummary.cpp
)
target_include_directories(nncli_common PRIVATE
    ${NNCLI_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/nlohmann
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/stb
    ${CURSES_INCLUDE_DIR}
)
target_link_libraries(nncli_common PRIVATE
    CNN
    Qt6::Core
    Qt6::Concurrent
    ${CURSES_LIBRARIES}
)

# nlohmann JSON library (header-only, used directly by NN-CLI for config serialisation)
# stb image library (header-only, used for image loading/saving)
target_include_directories(NN-CLI PRIVATE
    ${NNCLI_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/nlohmann
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/stb
    ${CURSES_INCLUDE_DIR}
)

# Link to CNN — ANN, OpenCLWrapper, Qt::Core, OpenCL are propagated automatically via PUBLIC
target_link_libraries(NN-CLI
    PRIVATE nncli_common
    PRIVATE CNN
    PRIVATE Qt6::Concurrent
    PRIVATE ${CURSES_LIBRARIES}
)

# tcmalloc replaces glibc's malloc/free across the whole process. Glibc's per-thread
# arena strategy fragments badly under the workload here (many ioPool workers,
# thousands of mid-size std::vector<float> allocations per batch from augmentation
# transforms and uint8->float conversion), and never returns arena memory to the OS.
# tcmalloc/jemalloc handle this pattern dramatically better. tcmalloc was already on
# the system; jemalloc would also work if preferred (-ljemalloc).
#
# --no-as-needed forces the linker to keep the dependency even though the NN-CLI
# code never references tcmalloc symbols directly — without it, modern linkers
# strip the link as unused and glibc malloc stays in effect.
find_library(TCMALLOC_LIB NAMES tcmalloc_minimal tcmalloc)

# Fallback: -dev package (unversioned .so symlink) often isn't installed; locate
# the versioned runtime library directly.
if(NOT TCMALLOC_LIB)
    find_file(TCMALLOC_LIB
        NAMES libtcmalloc_minimal.so.4 libtcmalloc.so.4
        PATHS /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu /usr/lib /lib
        NO_DEFAULT_PATH
    )
endif()

if(TCMALLOC_LIB)
    message(STATUS "Linking tcmalloc: ${TCMALLOC_LIB}")
    target_link_options(NN-CLI PRIVATE "LINKER:--no-as-needed")
    target_link_libraries(NN-CLI PRIVATE ${TCMALLOC_LIB})
else()
    message(WARNING "tcmalloc not found — falling back to glibc malloc (expect higher RSS under heavy augmentation)")
endif()

include(GNUInstallDirs)
install(TARGETS NN-CLI
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Test executable — integration tests use QProcess; unit tests link against libraries directly
add_executable(test_nncli
  tests/test_main.cpp
  tests/test_ann_cpu_basic.cpp
  tests/test_ann_cpu_mnist.cpp
  tests/test_ann_cpu_features.cpp
  tests/test_ann_cpu_saveload.cpp
  tests/test_ann_gpu_mnist.cpp
  tests/test_cnn_cpu_basic.cpp
  tests/test_cnn_cpu_features.cpp
  tests/test_cnn_cpu_saveload.cpp
  tests/test_cnn_cpu_predict.cpp
  tests/test_cnn_gpu_layers.cpp
  tests/test_cnn_gpu_diagnostic.cpp
  tests/test_cnn_gpu_isic.cpp
  tests/test_cnn_gpu_mnist.cpp
  tests/test_cnn_gpu_saveload.cpp
  tests/test_errors.cpp
  tests/test_dataloader.cpp
  tests/test_validation.cpp
  tests/test_monitoring.cpp
  tests/test_gpu_augment.cpp
  tests/test_terminal_ui.cpp
  tests/test_calibrate_controller.cpp
  tests/test_lr_scheduler.cpp
  # TerminalUI sources needed for unit tests (panel, table, window, training window).
  # Compiled separately from nncli_common to keep the shared object library lean.
  ui/NN-CLI_TerminalUI_Panel.cpp
  ui/NN-CLI_TerminalUI_Table.cpp
  ui/NN-CLI_TerminalUI_Window.cpp
  ui/NN-CLI_TerminalUI_TrainWindow.cpp
  ui/NN-CLI_TerminalUI_PredictWindow.cpp
)
target_include_directories(test_nncli PRIVATE
    ${NNCLI_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/nlohmann
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/stb
    ${CURSES_INCLUDE_DIR}
)
# Resolve test fixtures/examples from the source tree, not the (possibly nested) build dir.
target_compile_definitions(test_nncli PRIVATE NN_CLI_PROJECT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(test_nncli PRIVATE
    nncli_common
    Qt6::Core
    Qt6::Concurrent
    CNN
    ${CURSES_LIBRARIES}
)
