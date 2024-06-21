include_directories(${OptiX_INCLUDE_DIR})

#cuda_compile_and_embed(embedded_ptx_code devicePrograms.cu)
#cuda_compile_ptx(embedded_ptx_code devicePrograms.cu)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

message("CUDAToolkit_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
message("CUDAToolkit_INCLUDE: ${CUDA_INCLUDE}")

message("CUDA_CUDA_LIBRARY: ${CUDA_CUDA_LIBRARY}")
message("CUDART LIBRARY: ${CUDA_CUDART_LIBRARY}")
message("CUDA LIBRARIES: ${CUDA_LIBRARIES}")
include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${OptiX_INCLUDE}
        ${CUDA_INCLUDE_DIRS}
)

add_compile_definitions(CUDA_INCLUDE_DIRS="${CUDA_INCLUDE_DIRS}")
add_compile_definitions(OPTIX_INCLUDE="${OptiX_INCLUDE}")
add_compile_definitions(CURRENT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")



add_executable(raylib_output
        main.cpp
        SampleRenderer.cpp
        devicePrograms.cu
)

target_link_libraries(raylib_output
        ${optix_LIBRARY}
        ${CUDA_LIBRARIES}
        ${CUDA_CUDA_LIBRARY}
        ${CUDA_CUDART_LIBRARY}
        ${CUDA_nvrtc_LIBRARY}
        #spdlog::spdlog
        utils_no_spdlog
        raylib
        raylib_cpp
)

target_compile_definitions(raylib_output PRIVATE NO_SPDLOG=1)

add_custom_command(
    TARGET raylib_output POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/devicePrograms.cu
        ${CMAKE_CURRENT_BINARY_DIR}/cuDevicePrograms.cu
        )

add_custom_command(
    TARGET raylib_output POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_SOURCE_DIR}/../../resources/fira_code/ttf/FiraCode-SemiBold.ttf
    ${CMAKE_CURRENT_BINARY_DIR}/FiraCode-SemiBold.ttf
)
