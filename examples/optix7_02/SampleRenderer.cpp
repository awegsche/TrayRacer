//
// Created by andiw on 24/10/2023.
//

#include "SampleRenderer.h"
#include "launch_params.h"
#include "optix_host.h"
#include "optix_types.h"

#include <fstream>

#include <cuda_runtime.h>
#include <optix_lib.h>
#include <device_buffer.h>
#include <spdlog/spdlog.h>
#include <sstream>

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    void* data;
};

struct __align__ (OPTIX_SBT_RECORD_ALIGNMENT) MissRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];

    void *data;
};

struct __align__ (OPTIX_SBT_RECORD_ALIGNMENT) HitGroupRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];

    int objectID;
};

SampleRenderer::SampleRenderer() {
    logBuffer.resize(2048);
    initOptix();

    spdlog::info("creating optix context");
    createContext();

    spdlog::info("creating module");
    createModule();

    spdlog::info("creating raygen programs");
    createRaygenPrograms();

    spdlog::info("creating miss programs");
    createMissPrograms();

    spdlog::info("creating hitgroup programs");
    createHitgoupPrograms();

    spdlog::info("create pipeline");
    createPipeline();

    spdlog::info("build SBT");
    buildSBT();

    spdlog::info("alloc launch params buffer");
    launchParamsBuffer.alloc<LaunchParams>(1);
}

void SampleRenderer::initOptix() {

    spdlog::info("initialising optix");
    CU_CHECK_FATAL(cuInit(0));

    CU_CHECK_FATAL(cudaFree(nullptr));
    int num_devices;
    CU_CHECK_FATAL(cudaGetDeviceCount(&num_devices));
    if (num_devices == 0)
        throw std::runtime_error("no CUDA devices found");
    spdlog::info("found {} CUDA devices", num_devices);


    OP_CHECK_FATAL(optixInit());
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

    CU_CHECK_FATAL(cuCtxGetCurrent(&cudaContext));
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

    Nvrt nvrt;

    std::ifstream cudaCode("cuDevicePrograms.cu");

    std::ostringstream code;
    code << cudaCode.rdbuf();

    spdlog::info("cuda incl: {}", CUDA_INCLUDE_DIRS);

    const std::string &ptxCode = nvrt.compile_source(
            "cuDeviceCode",
            code.str(),
            {
            "--gpu-architecture=compute_86",
            "--relocatable-device-code=true",
            std::format("--include-path={}", CUDA_INCLUDE_DIRS).c_str(),
            std::format("--include-path={}", OPTIX_INCLUDE).c_str(),
            std::format("--include-path={}", CURRENT_SOURCE_DIR).c_str(),
            });

    //spdlog::info("ptx:\n{}", ptxCode);

    size_t logsize = 2048;
    logBuffer.clear();


    OP_CHECK_FATAL(optixModuleCreate(
            optixContext,
            &moduleCompileOptions,
            &pipelineCompileOptions,
            ptxCode.c_str(),
            ptxCode.size(),
            logBuffer.data(),
            &logsize,
            &module
            ));

    if (logsize > 1) spdlog::info("{}", logBuffer);
}

void SampleRenderer::createRaygenPrograms() {
    raygenPGs.resize(1);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module = module;
    pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";

    logBuffer.clear();
    size_t logsize=2048;

    OP_CHECK_FATAL(optixProgramGroupCreate(
                optixContext,
                &pgDesc,
                1,
                &pgOptions,
                logBuffer.data(),
                &logsize,
                &raygenPGs[0]
                ));

    if (logsize > 1) spdlog::info("{}", logBuffer);
        
}

void SampleRenderer::createMissPrograms() {
    missPGs.resize(1);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.miss.module = module;
    pgDesc.miss.entryFunctionName = "__miss__radiance";

    size_t logsize = 2048;

    OP_CHECK_FATAL(optixProgramGroupCreate(
            optixContext,
            &pgDesc,
            1,
            &pgOptions,
            logBuffer.data(),
            &logsize,
            &missPGs[0]
    ));

    if (logsize > 1) spdlog::info("{}", logBuffer);
}

void SampleRenderer::createHitgoupPrograms() {
    hitgroupPGs.resize(1);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.hitgroup.moduleCH = module;
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    pgDesc.hitgroup.moduleAH = module;
    pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

    size_t logsize = 2048;
    logBuffer.clear();

    OP_CHECK_FATAL(optixProgramGroupCreate(
            optixContext,
            &pgDesc,
            1,
            &pgOptions,
            logBuffer.data(),
            &logsize,
            &hitgroupPGs[0]
    ));

    if (logsize > 1) spdlog::info("{}", logBuffer);
}

void SampleRenderer::createPipeline() {
    std::vector<OptixProgramGroup> programGroups;

    for(auto pg: raygenPGs) programGroups.push_back(pg);
    for(auto pg: missPGs) programGroups.push_back(pg);
    for(auto pg: hitgroupPGs) programGroups.push_back(pg);

    size_t logsize = 2048;
    logBuffer.clear();

    OP_CHECK_FATAL(optixPipelineCreate(
            optixContext,
            &pipelineCompileOptions,
            &pipelineLinkOptions,
            programGroups.data(),
            static_cast<int>(programGroups.size()),
            logBuffer.data(),
            &logsize,
            &pipeline
    ));

    if (logsize > 1) spdlog::info("{}", logBuffer);

    OP_CHECK_FATAL(optixPipelineSetStackSize(
            pipeline,
            2*1024,
            2*1024,
            2*1024,
            1
    ));

    if (logsize > 1) spdlog::info("{}", logBuffer);

}

void SampleRenderer::buildSBT() {
    std::vector<RaygenRecord> raygenRecords;

    sbt = {};

    for(int i = 0; i < raygenPGs.size(); ++i) {
        RaygenRecord record;

        OP_CHECK_FATAL(optixSbtRecordPackHeader( raygenPGs[i],&record));

        record.data = nullptr;
        raygenRecords.push_back(record);
    }

    spdlog::info("alloc and upload raygen records");
    raygenRecordsBuffer.alloc_and_upload(raygenRecords);
    sbt.raygenRecord = raygenRecordsBuffer.get_CUdevice_ptr();

    std::vector<MissRecord> missRecords;
    for(int i = 0; i < missPGs.size(); ++i) {
        MissRecord record;
        OP_CHECK_FATAL(optixSbtRecordPackHeader( missPGs[i],&record));
        record.data = nullptr;
        missRecords.push_back(record);
    }
    spdlog::info("alloc and upload miss records");
    missRecordsBuffer.alloc_and_upload(missRecords);
    spdlog::info("worked");
    sbt.missRecordBase = missRecordsBuffer.get_CUdevice_ptr();
    sbt.missRecordStrideInBytes = sizeof(MissRecord);
    sbt.missRecordCount = static_cast<int>(missRecords.size());

    int numObjects = 1;
    std::vector<HitGroupRecord> hitgroupRecords;
    for(int i = 0; i < numObjects; ++i) {
        int objectType = i;
        HitGroupRecord record;
        spdlog::info("object type: {}", objectType);
        spdlog::info("hitgroup_size: {}", hitgroupPGs.size());
        OP_CHECK_FATAL(optixSbtRecordPackHeader( hitgroupPGs[objectType],&record));
        record.objectID = i;
        hitgroupRecords.push_back(record);
    }
    spdlog::info("alloc and upload hit group records");
    hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
    sbt.hitgroupRecordBase = hitgroupRecordsBuffer.get_CUdevice_ptr();
    sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
    sbt.hitgroupRecordCount = static_cast<int>(hitgroupRecords.size());
}


void SampleRenderer::render() {
    if (launchParams.fbWidth == 0 || launchParams.fbHeight == 0) return;

    launchParamsBuffer.data.to_device(&launchParams,1);
    launchParams.frameID++;

    OP_CHECK(optixLaunch(
            pipeline,
            stream,
            launchParamsBuffer.get_CUdevice_ptr(),
            sizeof(LaunchParams),
            &sbt,
            launchParams.fbWidth,
            launchParams.fbHeight,
            1
            ));
    spdlog::info("rendered");

}


void SampleRenderer::resize(int width, int height) {
    colorBuffer.data.resize<uint32_t>(width*height);

    launchParams.fbWidth = width;
    launchParams.fbHeight = height;
    launchParams.colorBuffer = colorBuffer.get_device_ptr<unsigned int>();
}

void SampleRenderer::downloadPixels(std::vector<unsigned int>& pixels) {
    colorBuffer.download(pixels);
}
