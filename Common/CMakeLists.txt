# Common config/result structs + compiled sources shared across the monorepo.
# The include root is the repo root (NN/) so #include "Common/Xxx.hpp" resolves.
add_library(NN_Common STATIC
    Common_ConfusionMatrix.cpp
    Common_CostFunctionConfig.cpp
    Common_Device.cpp
    Common_LearningRateScheduler.cpp
    Common_LearningRateSchedulerConfig.cpp
    Common_Mode.cpp
    Common_Optimizer.cpp
    Common_Utils.cpp
)
target_include_directories(NN_Common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
