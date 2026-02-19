cmake_minimum_required(VERSION 3.14)

project(ANN LANGUAGES CXX)

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
  ANN_RunMode.hpp ANN_RunMode.cpp
  ANN_CoreCPU.hpp ANN_CoreCPU.cpp
  ANN_CoreGPU.hpp ANN_CoreGPU.cpp
  ANN_CoreGPUWorker.hpp ANN_CoreGPUWorker.cpp
  ANN_Device.hpp ANN_Device.cpp
  ANN_LayersConfig.hpp ANN_LayersConfig.cpp
  Kernels.cpp.cl
  IdxHelper.cpp.cl
  ActvFunc.hpp.cl ActvFunc.cpp.cl
  Defines.hpp.cl
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

# nlohmann JSON library (header-only)
set(LIB_NLOHMANN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libs/nlohmann")
target_include_directories(ANN PUBLIC ${LIB_NLOHMANN_DIR})
