# Common config/result structs + compiled sources shared across the monorepo.
# The include root is the repo root (NN/) so #include "Common/Xxx.hpp" resolves.
add_library(NN_Common STATIC
    Common_LRSchedulerConfig.cpp
    Common_LRScheduler.cpp
)
target_include_directories(NN_Common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
