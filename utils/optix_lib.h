//
// Created by andiw on 24/10/2023.
//

#ifndef TRAYRACING_OPTIX_LIB_H
#define TRAYRACING_OPTIX_LIB_H

#include <string>
#include <vector>
#include <stdexcept>
#include <format>

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <nvrtc.h>


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
#define CUDA_SYNC_CHECK()                                               \
  {                                                                     \
    cudaDeviceSynchronize();                                            \
    cudaError_t error = cudaGetLastError();                             \
    if( error != cudaSuccess )                                          \
      {                                                                 \
        fprintf( stderr, "error (%s: line %d): %s\n", __FILE__, __LINE__, cudaGetErrorString( error ) ); \
        exit( 2 );                                                      \
      }                                                                 \
  }
/// Convenience class for NVRTC,
/// holds a context for log and ptx string as long as it's alive
///
class Nvrt {
    public:
        /// Compile a PTX source
        /// @param name Name of the program (kernel)
        /// @param source PTX source
        /// @param options Compiler options
        std::string const& compile_source(
                std::string const& name,
                std::string const& source,
                std::vector<std::string> const& options
                ) {

            spdlog::info("compiling the followin source:\n{}", source);
            spdlog::info("and options");

            for (auto const& opt : options) {
                spdlog::info("{}", opt);
            }
            nvrtcProgram prog;
            if (nvrtcCreateProgram(&prog, source.c_str(),name.c_str(), 0, NULL, NULL) != NVRTC_SUCCESS) {
                throw std::runtime_error("nvrtcCreateProgram failed");
            }

            const char ** opts = new const char*[options.size()];

            for (size_t i = 0; i < options.size(); ++i) {
                opts[i] = options[i].c_str(); 
            }


            if (nvrtcCompileProgram(prog, static_cast<int>(options.size()), opts) != NVRTC_SUCCESS) {
                size_t logSize;
                nvrtcGetProgramLogSize(prog, &logSize);
                char* log = new char[logSize];
                nvrtcGetProgramLog(prog, log);

                auto message = std::format("nvrtcCompileProgram failed: {}", log);

                delete[] opts;
                delete[] log;
                throw std::runtime_error(message);
            }

            delete[] opts;

            spdlog::info("{} compiled", name);

            size_t ptxsize;
            nvrtcGetPTXSize(prog, &ptxsize);
            ptxBuffer.resize(ptxsize);
            nvrtcGetPTX(prog, ptxBuffer.data());

            nvrtcDestroyProgram(&prog);

            return ptxBuffer;
        }

        /// Loads the program into a CUmodule
        void load_module(CUmodule& module) {
            CU_CHECK_FATAL(cuModuleLoadDataEx(&module, ptxBuffer.data(), 0, nullptr, nullptr));
        }

    private:
        std::string logBuffer;
        std::string ptxBuffer;
};


#endif //TRAYRACING_OPTIX_LIB_H
