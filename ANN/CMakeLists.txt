cmake_minimum_required(VERSION 3.14)

project(ANN LANGUAGES CXX)

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

# Include OpenCLWrapper submodule as a subdirectory if not already included
# This allows ANN to be built standalone or as part of a larger project
if(NOT TARGET OpenCLWrapper)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extern/OpenCLWrapper ${CMAKE_CURRENT_BINARY_DIR}/OpenCLWrapper)
endif()

add_library(ANN STATIC
  ANN_Core.hpp ANN_Core.cpp
  ANN_ActvFunc.hpp ANN_ActvFunc.cpp
  ANN_Utils.hpp ANN_Utils.cpp
  ANN_Mode.hpp ANN_Mode.cpp
  ANN_Worker.hpp ANN_Worker.cpp
  ANN_CoreCPU.hpp ANN_CoreCPU.cpp
  ANN_CoreCPUWorker.hpp ANN_CoreCPUWorker.cpp
  ANN_CoreGPU.hpp ANN_CoreGPU.cpp
  ANN_CoreGPUWorker.hpp ANN_CoreGPUWorker.cpp
  ANN_GPUBufferManager.hpp ANN_GPUBufferManager.cpp
  ANN_GPUKernelBuilder.hpp ANN_GPUKernelBuilder.cpp
  ANN_Device.hpp ANN_Device.cpp
  ANN_LayersConfig.hpp ANN_LayersConfig.cpp
  ANN_Types.hpp
  ANN_Optimizer.hpp ANN_Optimizer.cpp
  ANN_TrainingConfig.hpp
  ANN_Parameters.hpp
  ANN_Sample.hpp
  ANN_ProgressCallback.hpp
  ANN_TrainingProgress.hpp
  ANN_TrainingMetadata.hpp
  ANN_TestResult.hpp
  ANN_CoreConfig.hpp
  opencl/ANN_Defines.hpp.cl
  opencl/ANN_IdxHelper.cpp.cl
  opencl/ANN_ActvFunc.hpp.cl opencl/ANN_ActvFunc.cpp.cl
  opencl/ANN_Propagate.cpp.cl
  opencl/ANN_Backpropagate.cpp.cl
  opencl/ANN_Update.cpp.cl
  opencl/ANN_Loss.cpp.cl
  LICENSE.md
)

# PUBLIC: Consumers of ANN need access to ANN headers
target_include_directories(ANN PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# PUBLIC: Consumers of ANN will also need OpenCLWrapper (it's part of our public API)
# PRIVATE: Qt Concurrent is only used internally
target_link_libraries(ANN
  PUBLIC OpenCLWrapper
  PRIVATE Qt${QT_VERSION_MAJOR}::Concurrent
)

target_compile_definitions(ANN PRIVATE ANN_LIBRARY)




# --- Test executable ---
add_executable(test_ann
  tests/test_main.cpp
  tests/test_actvfunc.cpp
  tests/test_utils.cpp
  tests/test_core_basic.cpp
  tests/test_core_basic2.cpp
  tests/test_core_features.cpp
  tests/test_core_features2.cpp
  tests/test_core_exact.cpp
  tests/test_core_exact2.cpp
  tests/test_serialization.cpp
  tests/test_gpu_basic.cpp
  tests/test_gpu_basic2.cpp
  tests/test_gpu_features.cpp
  tests/test_gpu_multigpu.cpp
  tests/test_gpu_exact.cpp
)

target_link_libraries(test_ann PRIVATE ANN Qt${QT_VERSION_MAJOR}::Core Qt${QT_VERSION_MAJOR}::Concurrent)
