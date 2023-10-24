//
// Created by andiw on 24/10/2023.
//

#ifndef TRAYRACING_OPTIX_LIB_H
#define TRAYRACING_OPTIX_LIB_H

#include <string>

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>


#include <spdlog/spdlog.h>

void say_hello();

#define OP_CHECK( call )                                                      \
  {                                                                           \
    const OptixResult res = call;                                             \
    if( res != OPTIX_SUCCESS )                                                \
      {                                                                       \
        spdlog::error("Optix call {} failed (line {})\n",                     \
            std::string{#call}, __LINE__                                      \
            );                                                                \
        spdlog::error("{}: {}",                                               \
            std::string{optixGetErrorName(res)},                              \
            std::string{optixGetErrorString(res)}                             \
           );                                                                 \
        exit( 2 );                                                            \
      }                                                                       \
  }

#endif //TRAYRACING_OPTIX_LIB_H
