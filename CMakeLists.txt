cmake_minimum_required(VERSION 3.25)
project(TrayRacing)

set(CMAKE_CXX_STANDARD 23)

set(CMAKE_MODULE_PATH "$ENV{OptiX_PATH}/SDK/CMake")

message("toolchain file: ${CMAKE_TOOLCHAIN_FILE}")

find_package(glfw3 CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

set(OptiX_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/OptiX_SDK" CACHE PATH "Path to OptiX installed location.")
find_package(OptiX REQUIRED)

find_package(CUDA REQUIRED)

#find_package(imgui CONFIG REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/utils)
add_subdirectory(utils)


add_subdirectory(examples/optix7_01)
add_subdirectory(examples/optix7_02)

