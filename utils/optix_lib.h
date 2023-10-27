//
// Created by andiw on 24/10/2023.
//

#ifndef TRAYRACING_OPTIX_LIB_H
#define TRAYRACING_OPTIX_LIB_H

#include <string>
#include <stdexcept>
#include <format>

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
        spdlog::error("{} failed | {} | {}:{}",                               \
            std::string{#call},                                               \
            std::string{optixGetErrorString(res)},                            \
            __FILE__,                                                         \
            __LINE__                                                          \
            );                                                                \
      }                                                                       \
  }

#define OP_CHECK_FATAL( call )                                                \
  {                                                                           \
    const OptixResult res = call;                                             \
    if( res != OPTIX_SUCCESS )                                                \
      {                                                                       \
        throw std::runtime_error(                                             \
            std::format("{} failed | {} | {}:{}",                             \
                        std::string{#call},                                   \
                        std::string{optixGetErrorString(res)},                \
                        __FILE__,                                             \
                        __LINE__                                              \
            )                                                                 \
        );                                                                    \
      }                                                                       \
  }
#define CU_CHECK( call ) \
{                         \
    auto const err = cudaSetDevice(call); \
    if (err != cudaError::cudaSuccess) { \
    spdlog::error("{}: {} | {}:{}", cudaGetErrorName(err), cudaGetErrorString(err), __FILE__, __LINE__);\
    }\
}

#define CU_CHECK_FATAL( call )                                   \
{                                                                \
    auto const err = cudaSetDevice(call);                        \
    if (err != cudaError::cudaSuccess) {                         \
        throw std::runtime_error(                                \
        std::format("{} failed | {} | {}:{}",                    \
                    std::string{#call},                          \
                    cudaGetErrorString(err), __FILE__, __LINE__) \
        );                                                       \
    }                                                            \
}

#endif //TRAYRACING_OPTIX_LIB_H
