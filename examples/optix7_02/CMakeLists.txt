include_directories(${OptiX_INCLUDE_DIR})

#cuda_compile_and_embed(embedded_ptx_code devicePrograms.cu)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

message("CUDAToolkit_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
message("CUDAToolkit_INCLUDE: ${CUDA_INCLUDE}")
include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${OptiX_INCLUDE}
        ${CUDA_INCLUDE_DIRS}
)

add_executable(ex02
        main.cpp
        SampleRenderer.cpp
)

target_link_libraries(ex02
        ${optix_LIBRARY}
        ${CUDA_LIBRARIES}
        ${CUDA_CUDA_LIBRARY}
        spdlog::spdlog
        utils
)
