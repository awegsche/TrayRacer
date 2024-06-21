//
// Created by andiw on 24/10/2023.
//

#if defined(_WIN32)
#define NOGDI
#define NOUSER
#endif

#include "SampleRenderer.h"

#include <format>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <nvrtc.h>

#include "logging.h"
#include "optix_lib.h"

#if defined(_WIN32)
#undef near
#undef far
#endif

#include <raylib.h>

enum class Section { Init, Idle, End };

constexpr auto WIDTH = 800;
constexpr auto HEIGHT = 1200;
constexpr auto FONTSIZE = 14;
constexpr auto LINEHEIGHT = FONTSIZE + 2;

std::vector<std::string> logmessages = {};

int main()
{
    InitWindow(WIDTH, HEIGHT, "rl out");
    SetTargetFPS(30);
    Font font = LoadFontEx("FiraCode-SemiBold.ttf", FONTSIZE * 4, 0, 0);
    GenTextureMipmaps(&font.texture);
    SetTextureFilter(font.texture, TEXTURE_FILTER_BILINEAR);


    // state
    int offset = 0;
    bool show_log = true;
    bool show_log_down = false;

    Section section = Section::Init;

    try {
        SampleRenderer renderer;

        while (!WindowShouldClose()) {
            BeginDrawing();
            ClearBackground(RAYWHITE);
            switch (section) {
            case Section::Init:

                // spdlog::info("resized");
                renderer.resize(300, 200);
                section = Section::Idle;
                break;
            case Section::Idle:
                // spdlog::info("rendering");
                renderer.render();
                CUDA_SYNC_CHECK();
                break;
            case Section::End:
                break;
            }
            // always render messages
            if (IsKeyPressed(KEY_L)) {
                show_log = !show_log;
                logging::info("pressed L");
                std::cout << "l\n";
            }
            if (IsKeyPressed(KEY_Q)) {
                goto close;
            }
            if (show_log) {
                const size_t n_messages = logging::log_msg.size();
                if (GetMouseWheelMove() > 0.0 && offset < n_messages) { offset += 1; }
                if (GetMouseWheelMove() < 0.0 && offset > 0) { offset -= 1; }
                float pos = 10.0;
                const int max_lines = HEIGHT / LINEHEIGHT + 1;
                const size_t start_print = max_lines < n_messages - offset ? n_messages - max_lines - offset : 0;
                const size_t end_print = n_messages - offset;

                for (size_t i = start_print; i < end_print; ++i) {
                    DrawTextEx(font, logging::log_msg[i].msg.c_str(), Vector2{ 10.0f, pos }, FONTSIZE, 0, BLACK);
                    // DrawText(logging::log_msg[i].msg.c_str(), 10.0f, pos, FONTSIZE, BLACK);
                    pos += static_cast<float>(FONTSIZE);
                }
            }
            EndDrawing();
        }
    close:
        CloseWindow();
        return 0;
    } catch (std::exception const &e) {
        //       spdlog::error("Exception occured:\n{}", e.what());
        CloseWindow();
        return 1;
    }
}
