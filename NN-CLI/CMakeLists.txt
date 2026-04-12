cmake_minimum_required(VERSION 3.14)

project(NN-CLI LANGUAGES CXX)

# Auto-update submodules recursively at configure time
# Disable with: cmake .. -DAUTO_UPDATE_SUBMODULES=OFF (persists in CMake cache)
option(AUTO_UPDATE_SUBMODULES "Automatically update submodules at configure time" ON)
if(AUTO_UPDATE_SUBMODULES)
  find_package(Git QUIET)
  if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    execute_process(
      COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
      RESULT_VARIABLE GIT_SUBMOD_RESULT
    )
    if(NOT GIT_SUBMOD_RESULT EQUAL "0")
      message(WARNING "git submodule update --init --recursive failed: ${GIT_SUBMOD_RESULT}")
    endif()
  endif()
endif()

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Core Concurrent)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Core Concurrent)

# Add CNN as subdirectory — it brings ANN and OpenCLWrapper transitively via PUBLIC
if(NOT TARGET CNN)
  add_subdirectory(extern/CNN)
endif()

add_executable(NN-CLI
  main.cpp
  NN-CLI_ANNLoader.cpp
  NN-CLI_ANNRunner.cpp
  NN-CLI_CNNLoader.cpp
  NN-CLI_CNNRunner.cpp
  NN-CLI_DataLoader.cpp
  NN-CLI_DataSplitter.cpp
  NN-CLI_DataType.cpp
  NN-CLI_ImageLoader.cpp
  NN-CLI_Loader.cpp
  NN-CLI_ModelSerializer.cpp
  NN-CLI_ProgressBar.cpp
  NN-CLI_Runner.cpp
  NN-CLI_TrainingSummary.cpp
  NN-CLI_Utils.cpp
)

# nlohmann JSON library (header-only, used directly by NN-CLI for config serialisation)
# stb image library (header-only, used for image loading/saving)
target_include_directories(NN-CLI PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/nlohmann
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/stb
)

# Link to CNN — ANN, OpenCLWrapper, Qt::Core, OpenCL are propagated automatically via PUBLIC
target_link_libraries(NN-CLI
    PRIVATE CNN
    PRIVATE Qt${QT_VERSION_MAJOR}::Concurrent
)

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
  tests/test_ann_gpu_mnist.cpp
  tests/test_cnn_cpu_basic.cpp
  tests/test_cnn_cpu_features.cpp
  tests/test_cnn_cpu_saveload.cpp
  tests/test_cnn_cpu_predict.cpp
  tests/test_cnn_cpu_shuffle.cpp
  tests/test_cnn_gpu_layers.cpp
  tests/test_cnn_gpu_diagnostic.cpp
  tests/test_cnn_gpu_isic.cpp
  tests/test_cnn_gpu_mnist.cpp
  tests/test_cnn_gpu_saveload.cpp
  tests/test_errors.cpp
  tests/test_dataloader.cpp
  tests/test_validation.cpp
  NN-CLI_DataLoader.cpp
  NN-CLI_DataSplitter.cpp
  NN-CLI_DataType.cpp
  NN-CLI_ImageLoader.cpp
  NN-CLI_Loader.cpp
  NN-CLI_ProgressBar.cpp
  NN-CLI_TrainingSummary.cpp
)
target_include_directories(test_nncli PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/nlohmann
    ${CMAKE_CURRENT_SOURCE_DIR}/libs/stb
)
target_link_libraries(test_nncli PRIVATE
    Qt${QT_VERSION_MAJOR}::Core
    Qt${QT_VERSION_MAJOR}::Concurrent
    CNN
)
