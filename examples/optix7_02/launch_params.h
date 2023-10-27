//
// Created by andiw on 24/10/2023.
//

#ifndef TRAYRACING_LAUNCH_PARAMS_H
#define TRAYRACING_LAUNCH_PARAMS_H

#include <stdint.h>

struct LaunchParams{
    int frameID { 0 };
    uint32_t* colorBuffer;
    int fbWidth;
    int fbHeight;
};

#endif //TRAYRACING_LAUNCH_PARAMS_H
