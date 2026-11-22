#pragma once

// The display the recompiled game is told it runs on (Phase 5 R5b widescreen, 2026-11-23).
//
// Port infrastructure, not a reconstruction: no retail layout, no citations.
//
// Warband sizes its window and back buffer from the desktop when `rgl_config.txt` has
// `display_width = 0` (the default): `GetSystemMetrics(SM_CXSCREEN)` in the user32 shim and
// `GetAdapterDisplayMode` / `EnumAdapterModes` in the D3D9 shim. Both read this size, so a phone
// sets it to its screen (e.g. 2412 x 1080, 20:9) before the shims are created and the game draws
// at the screen's own shape: no black bars, nothing stretched. The default is 1920 x 1080.

namespace wse::recomp::shim {

struct ScreenSize {
    int width = 1920;
    int height = 1080;
};

// Read when a shim is constructed; set it before the game starts.
inline ScreenSize& screen_size() noexcept {
    static ScreenSize size{};
    return size;
}

// Sets the screen from a window's size: landscape (wider than tall), at least 640 x 480.
inline void set_screen_size(int width, int height) noexcept {
    if (width < height) {
        const int t = width;
        width = height;
        height = t;
    }
    if (width < 640 || height < 480) return;
    screen_size() = {width, height};
}

}  // namespace wse::recomp::shim
