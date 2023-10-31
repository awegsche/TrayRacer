//
// Created by andiw on 24/10/2023.
//

#include "SampleRenderer.h"

#include <format>
#include <stdexcept>
#include <fstream>

#include <spdlog/spdlog.h>
#include <nvrtc.h>

#include "optix_lib.h"

const char *saxpy = "                                          \n\
extern \"C\" __global__                                        \n\
void saxpy(float a, float *x, float *y, float *out, size_t n)  \n\
{                                                              \n\
   size_t tid = blockIdx.x * blockDim.x + threadIdx.x;         \n\
   if (tid < n) {                                              \n\
      out[tid] = a * x[tid] + y[tid];                          \n\
   }                                                           \n\
}      \n";

const char *opts[] = {
    "--gpu-architecture=compute_86",
    "--fmad=true",
};

int main() {

    spdlog::info("first, let's try running a simple CUDA program");

    try {
        SampleRenderer renderer;

        spdlog::info("resized");
        renderer.resize(300, 200);
        spdlog::info("rendering");
        renderer.render();
        CUDA_SYNC_CHECK();
        spdlog::info("downloading image");
        std::vector<uint32_t> colorBuffer(300 * 200);
        renderer.downloadPixels(colorBuffer);

        std::ofstream file("image.ppm");

        file << "P3\n";
        file << "300 200\n";
        file << "255\n";

        for (int y = 0; y < 200; ++y) {
            for (int x = 0; x < 300; ++x) {
                file << (colorBuffer[y * 300 + x] & 0xFF) << " "
                    << ((colorBuffer[y * 300 + x] >> 8) & 0xFF) << " "
                    << ((colorBuffer[y * 300 + x] >> 16) & 0xFF) << " ";
            }
            file << "\n";
        }
        spdlog::info("done");
        return 0;
    }
    catch (std::exception const& e) {
        spdlog::error("Exception occured:\n{}", e.what());
        return 1;
    }
}
