cmake_minimum_required(VERSION 3.14)
project(NN LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Allow building only specific targets: 
#   cmake -DBUILD_ANN=ON -DBUILD_CNN=ON -DBUILD_NN_CLI=OFF -DBUILD_NN_SERVER=OFF ..
option(BUILD_ANN "Build ANN library" ON)
option(BUILD_CNN "Build CNN library" ON)
option(BUILD_NN_CLI "Build NN-CLI executable" ON)
option(BUILD_NN_SERVER "Build NN-Server executable" ON)

message(STATUS "NN Monorepo")
message(STATUS "  BUILD_ANN:      ${BUILD_ANN}")
message(STATUS "  BUILD_CNN:      ${BUILD_CNN}")
message(STATUS "  BUILD_NN_CLI:   ${BUILD_NN_CLI}")
message(STATUS "  BUILD_NN_SERVER:${BUILD_NN_SERVER}")

# Validation: dependencies must be enabled
if(BUILD_CNN AND NOT BUILD_ANN)
  message(FATAL_ERROR "CNN requires ANN. Set -DBUILD_ANN=ON or build from within CNN/ directly.")
endif()

# Make Common headers available — include dir is NN/ so #include "Common/Xxx.hpp" resolves
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/Common)
  add_library(NN_Common INTERFACE)
  target_include_directories(NN_Common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
endif()

# Dependencies are PUBLIC-linked, so order matters
if(BUILD_ANN)
  add_subdirectory(ANN)
  if(TARGET NN_Common)
    target_link_libraries(ANN PUBLIC NN_Common)
  endif()
endif()

if(BUILD_CNN)
  if(NOT BUILD_ANN)
    # CNN will pull in ANN via add_subdirectory
  endif()
  add_subdirectory(CNN)
  if(TARGET NN_Common)
    target_link_libraries(CNN PUBLIC NN_Common)
  endif()
endif()

if(BUILD_NN_CLI)
  add_subdirectory(NN-CLI)
endif()

if(BUILD_NN_SERVER)
  add_subdirectory(NN-Server)
endif()
