//
// Created by andiw on 24/10/2023.
//

#ifndef TRAYRACING_SAMPLERENDERER_H
#define TRAYRACING_SAMPLERENDERER_H

#include <vector>

#include <cuda_runtime.h>
#include <optix.h>

#include "device_buffer.h"
#include "launch_params.h"

class SampleRenderer {
public:
    SampleRenderer();

    void render();

    void resize(int width, int height);

    void downloadPixels(std::vector<unsigned int>& pixels);

protected:
    void initOptix();

    void createContext();

    void createModule();

    void createRaygenPrograms();

    void createMissPrograms();

    void createHitgoupPrograms();

    void createPipeline();

    void buildSBT();

protected:
    CUcontext cudaContext;
    CUstream stream;
    cudaDeviceProp deviceProps;

    OptixDeviceContext optixContext;

    OptixPipeline pipeline;
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    OptixPipelineLinkOptions pipelineLinkOptions = {};

    OptixModule module;
    OptixModuleCompileOptions moduleCompileOptions = {};

    std::vector<OptixProgramGroup> raygenPGs;
    DeviceBuffer raygenRecordsBuffer;
    std::vector<OptixProgramGroup> missPGs;
    DeviceBuffer missRecordsBuffer;
    std::vector<OptixProgramGroup> hitgroupPGs;
    DeviceBuffer hitgroupRecordsBuffer;
    OptixShaderBindingTable sbt;

    LaunchParams launchParams;
    DeviceBuffer launchParamsBuffer;

    DeviceBuffer colorBuffer;

    std::string logBuffer;
};


#endif //TRAYRACING_SAMPLERENDERER_H
