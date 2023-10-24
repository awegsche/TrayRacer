include_directories(${OptiX_INCLUDE_DIR})

message("CUDAToolkit_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
message("CUDAToolkit_INCLUDE: ${CUDA_INCLUDE}")
include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${OptiX_INCLUDE}
        ${CUDA_INCLUDE_DIRS}
)

add_executable(ex01
        main.cpp
)

target_link_libraries(ex01
        ${optix_LIBRARY}
        ${CUDA_LIBRARIES}
        ${CUDA_CUDA_LIBRARY}
        spdlog::spdlog
        utils
)