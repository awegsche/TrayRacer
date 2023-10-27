//
// Created by andiw on 24/10/2023.
//

#include "SampleRenderer.h"

#include <cuda_runtime.h>
#include <optix_lib.h>
#include <spdlog/spdlog.h>

SampleRenderer::SampleRenderer() {
    initOptix();

    spdlog::info("creating optix context");
    createContext();

    spdlog::info("creating module");
    createModule();

    spdlog::info("creating raygen programs");
    createRaygenPrograms();
}

void SampleRenderer::initOptix() {

    spdlog::info("initialising optix");

    cudaFree(nullptr);
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    if (num_devices == 0)
        throw std::runtime_error("no CUDA devices found");
    spdlog::info("found {} CUDA devices", num_devices);

    OP_CHECK(optixInit());
    spdlog::info("successfully initialised optix");
}

static void context_log_callback(unsigned int level,
                                 const char *tag,
                                 const char *message,
                                 void *) {
    switch(level) {
        case 0:
            spdlog::info("{} {:12} {}", level, tag, message);
            break;
        default:
            spdlog::info("{} {:12} {}", level, tag, message);
            break;
    }
}
void SampleRenderer::createContext() {
    const int deviceID = 0;

    CU_CHECK_FATAL(cudaSetDevice(deviceID));
    CU_CHECK_FATAL(cudaStreamCreate(&stream));

    cudaGetDeviceProperties(&deviceProps, deviceID);

    spdlog::info("device {}", deviceProps.name);

    CU_CHECK(cuCtxGetCurrent(&cudaContext));
    OP_CHECK_FATAL(optixDeviceContextCreate(cudaContext, 0, &optixContext));
    OP_CHECK_FATAL(optixDeviceContextSetLogCallback(optixContext, context_log_callback, nullptr, 4));
}

void SampleRenderer::createModule() {
    moduleCompileOptions.maxRegisterCount = 50;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    pipelineCompileOptions = {};
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.numPayloadValues = 2;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

    pipelineLinkOptions.maxTraceDepth = 2;

    const std::string ptxCode = embedded_ptx_code;

    size_t LOGSIZE = 2048;
    std::string log;
    log.resize(LOGSIZE);

    OP_CHECK_FATAL(optixModuleCreate(
            optixContext,
            &moduleCompileOptions,
            &pipelineCompileOptions,
            ptxCode.c_str(),
            ptxCode.size(),
            log.data(),
            &LOGSIZE,
            &module
            ));

    if (LOGSIZE > 1) spdlog::info("{}", log);
}

void SampleRenderer::createRaygenPrograms() {

}

void SampleRenderer::createMissPrograms() {

}

void SampleRenderer::createHitgoupPrograms() {

}

void SampleRenderer::createPipeline() {

}
