#ifndef CL_HPP_MINIMUM_OPENCL_VERSION
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#endif

#ifndef CL_HPP_TARGET_OPENCL_VERSION
#define CL_HPP_TARGET_OPENCL_VERSION 300
#endif

#ifndef OCLW_KERNEL_H
#define OCLW_KERNEL_H

#include <CL/opencl.hpp>

namespace OpenCLWrapper
{
  struct Kernel
  {
    std::string name;
    cl_uint nElements;
    cl::Kernel kernel;
    uint argsCount = 0;
  };
}

#endif // OCLW_KERNEL_H
