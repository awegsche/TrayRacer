//
// Created by andiw on 24/10/2023.
//

#include <spdlog/spdlog.h>

#include "optix_lib.h"
#include "device_buffer.h"

#include <optix_function_table_definition.h>

void say_hello() {
    spdlog::info("hello from utils");
}