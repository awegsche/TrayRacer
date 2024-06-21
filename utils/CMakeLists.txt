project(utils LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${OptiX_INCLUDE}
        ${CUDA_INCLUDE_DIRS}
)

# ---- the default using spdlog --------------------------------------------------------------------

add_library(utils
    utils.cpp
)

target_link_libraries(utils
        spdlog::spdlog
)

# ---- for raylib we skip spdlog -------------------------------------------------------------------

add_library(utils_no_spdlog
    utils.cpp
    logging.cpp
)

target_compile_definitions(utils_no_spdlog PRIVATE NO_SPDLOG=1)
