project(utils LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${OptiX_INCLUDE}
        ${CUDA_INCLUDE_DIRS}
)

add_library(utils
    utils.cpp
)

target_link_libraries(utils
        spdlog::spdlog
)