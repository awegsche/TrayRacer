//
// Created by andiw on 24/10/2023.
//
#include <cuda_runtime.h>
#include <optix_lib.h>
#include <spdlog/spdlog.h>

void initOptiX()
{
    cudaFree(nullptr);

    int numDevices;
    cudaGetDeviceCount(&numDevices);

    if (numDevices == 0) {
        spdlog::error("no CUDA capable devices found");
        return;
    }

    spdlog::info("found {} CUDA devices", numDevices);


    OP_CHECK(optixInit());
}

int main()
{

    spdlog::info("hello world");

    say_hello();

    initOptiX();

    return 0;
}
