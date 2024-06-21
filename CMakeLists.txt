cmake_minimum_required(VERSION 3.25)
project(TrayRacing)

set(CMAKE_CXX_STANDARD 23)

if(NOT DEFINED ENV{OptiX_PATH})
    message(FATAL_ERROR
        "Env variable `OptiX_PATH` not set."
        "Please make sure that OptiX SDK is installed and set the env variable manually if needed.")
endif()

set(CMAKE_MODULE_PATH "$ENV{OptiX_PATH}/SDK/CMake")


message("toolchain file: ${CMAKE_TOOLCHAIN_FILE}")

message("\n--------------------------------------------------------------------------------")
message("---- installing external libs --------------------------------------------------")
message("--------------------------------------------------------------------------------")

message("---- OptiX ---------------------------------------------------------------------")
set(OptiX_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/OptiX_SDK" CACHE PATH "Path to OptiX installed location.")
find_package(OptiX REQUIRED)

message("---- vcpkg installed -----------------------------------------------------------")
find_package(glfw3 CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
#find_package(imgui CONFIG REQUIRED)

include(FetchContent)

message("---- raylib --------------------------------------------------------------------")
find_package(raylib QUIET)
if (NOT raylib_FOUND)
    FetchContent_Declare(
        raylib
        GIT_REPOSITORY https://github.com/raysan5/raylib.git
        GIT_TAG 5.0
        GIT_SHALLOW 1
    )
    FetchContent_MakeAvailable(raylib)
endif()

message("---- raylib C++ ----------------------------------------------------------------")
find_package(raylib_cpp QUIET)
if (NOT raylib_cpp_FOUND)
    if (NOT DEFINED RAYLIB_CPP_VERSION)
        set(RAYLIB_CPP_VERSION v5.0.1)
    endif()
    include(FetchContent)
    FetchContent_Declare(
        raylib_cpp
        GIT_REPOSITORY https://github.com/RobLoach/raylib-cpp.git
        GIT_TAG ${RAYLIB_CPP_VERSION}
    )
    FetchContent_MakeAvailable(raylib_cpp)
endif()

message("---- find CUDA -----------------------------------------------------------------")
find_package(CUDA REQUIRED)


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/utils)
add_subdirectory(utils)


message("\n--------------------------------------------------------------------------------")
message("---- EXAMPLES ------------------------------------------------------------------")
message("--------------------------------------------------------------------------------")
add_subdirectory(examples/optix7_01)
add_subdirectory(examples/optix7_02)
add_subdirectory(examples/raylib_output)

