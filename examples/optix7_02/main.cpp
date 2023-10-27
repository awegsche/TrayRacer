//
// Created by andiw on 24/10/2023.
//

#include "SampleRenderer.h"

#include <spdlog/spdlog.h>

int main() {

    try {
        SampleRenderer renderer;
        return 0;
    }
    catch (std::exception const& e) {
        spdlog::error("Exception occured:\n{}", e.what());
        return 1;
    }
}