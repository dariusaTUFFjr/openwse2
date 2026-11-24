// Unified R6 Input Hub, Adaptive v4.3-Style Touch Controller & Procedural Overlay Builder
// (Phase 5 R6, `docs/RECOMP_PLAN.md` "R6 — Audio, Input, Touch Controls & Desktop Mode").
//
// Single thread-safe source of truth feeding:
//   - DirectInput 8 (`shim_d3d9.h`: keyboard state/events, mouse relative deltas/buttons)
//   - USER32 (`shim_user32.h`: cursor position, WM_MOUSE*, WM_KEY*, WM_CHAR, ShowCursor)
//   - XInput 1.3 (`shim_system.h`: controller state + rumble callback)
//   - Vulkan `Present()` touch overlay (`platform/vulkan/d3d9_vulkan.cpp`: full-res GeometryFrame
//     with simplistic procedural vector icons, zero per-frame heap allocations, and menu pinch-zoom)
//
// Clean-room, header-only, standalone-compilable.

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "wse/platform/input.h"
#include "wse/platform/mod_controls_detector.h"
#include "wse/platform/rhi_geometry.h"
#include "wse/platform/touch_icons.h"

namespace wse::recomp::shim {

// Standard DirectInput 8 scan codes (subset used by Warband & mod controls).
namespace dik {
inline constexpr std::uint8_t kEscape = 0x01U;
inline constexpr std::uint8_t k1 = 0x02U;
inline constexpr std::uint8_t k2 = 0x03U;
inline constexpr std::uint8_t k3 = 0x04U;
inline constexpr std::uint8_t k4 = 0x05U;
inline constexpr std::uint8_t kBackspace = 0x0EU;
inline constexpr std::uint8_t kTab = 0x0FU;
inline constexpr std::uint8_t kQ = 0x10U;
inline constexpr std::uint8_t kW = 0x11U;
inline constexpr std::uint8_t kE = 0x12U;
inline constexpr std::uint8_t kR = 0x13U;
inline constexpr std::uint8_t kT = 0x14U;
inline constexpr std::uint8_t kU = 0x16U;
inline constexpr std::uint8_t kI = 0x17U;
inline constexpr std::uint8_t kO = 0x18U;
inline constexpr std::uint8_t kP = 0x19U;
inline constexpr std::uint8_t kReturn = 0x1CU;
inline constexpr std::uint8_t kLCtrl = 0x1DU;
inline constexpr std::uint8_t kA = 0x1EU;
inline constexpr std::uint8_t kS = 0x1FU;
inline constexpr std::uint8_t kD = 0x20U;
inline constexpr std::uint8_t kF = 0x21U;
inline constexpr std::uint8_t kG = 0x22U;
inline constexpr std::uint8_t kH = 0x23U;
inline constexpr std::uint8_t kJ = 0x24U;
inline constexpr std::uint8_t kK = 0x25U;
inline constexpr std::uint8_t kL = 0x26U;
inline constexpr std::uint8_t kLShift = 0x2AU;
inline constexpr std::uint8_t kGrave = 0x29U;
inline constexpr std::uint8_t kZ = 0x2CU;
inline constexpr std::uint8_t kX = 0x2DU;
inline constexpr std::uint8_t kC = 0x2EU;
inline constexpr std::uint8_t kV = 0x2FU;
inline constexpr std::uint8_t kB = 0x30U;
inline constexpr std::uint8_t kN = 0x31U;
inline constexpr std::uint8_t kM = 0x32U;
inline constexpr std::uint8_t kLAlt = 0x38U;
inline constexpr std::uint8_t kSpace = 0x39U;
inline constexpr std::uint8_t kF1 = 0x3BU;
inline constexpr std::uint8_t kF2 = 0x3CU;
inline constexpr std::uint8_t kF3 = 0x3DU;
inline constexpr std::uint8_t kF4 = 0x3EU;
inline constexpr std::uint8_t kF5 = 0x3FU;
inline constexpr std::uint8_t kF9 = 0x43U;
inline constexpr std::uint8_t kF12 = 0x58U;
inline constexpr std::uint8_t kUp = 0xC8U;
inline constexpr std::uint8_t kLeft = 0xCBU;
inline constexpr std::uint8_t kRight = 0xCDU;
inline constexpr std::uint8_t kDown = 0xD0U;
inline constexpr std::uint8_t kHome = 0xC7U;
}  // namespace dik

enum class TouchScene : std::uint8_t {
    main_menu = 0,
    battle = 1,
    orders = 2,
    campaign_map = 3,
    party = 4,
    trade = 5,
    dialog = 6,
    screens = 7,
};

enum class ActiveInputMode : std::uint8_t {
    touch = 0,
    gamepad = 1,
    keyboard_mouse = 2,
};

// Within-battle context flags (read via PDB symbols when available; any bit absent from
// `known_mask` falls back to visible so the player is never locked out of an action).
struct BattleContextFlags {
    static constexpr std::uint32_t kInteract = 1U << 0;
    static constexpr std::uint32_t kMount = 1U << 1;
    static constexpr std::uint32_t kDismount = 1U << 2;
    static constexpr std::uint32_t kOnFoot = 1U << 3;
    static constexpr std::uint32_t kRiding = 1U << 4;
    static constexpr std::uint32_t kArmed = 1U << 5;
    static constexpr std::uint32_t kHasWeapons = 1U << 6;
    static constexpr std::uint32_t kHasShield = 1U << 7;
    static constexpr std::uint32_t kHasAltMode = 1U << 8;
    static constexpr std::uint32_t kCrouchAllowed = 1U << 9;
    static constexpr std::uint32_t kHasTroops = 1U << 10;
    static constexpr std::uint32_t kMultiplayer = 1U << 11;
    static constexpr std::uint32_t kSpectating = 1U << 12;
    static constexpr std::uint32_t kDialogChoices = 1U << 13;
    static constexpr std::uint32_t kPartyMovable = 1U << 14;

    std::uint32_t value_mask = kOnFoot | kArmed | kHasWeapons | kHasShield | kHasTroops | kPartyMovable;
    // Bits set in `known_mask` are authoritative; unset bits fall back gracefully to visible.
    std::uint32_t known_mask = kCrouchAllowed | kMultiplayer;

    [[nodiscard]] bool test_or_default(std::uint32_t bit, bool fallback_if_unknown) const noexcept {
        if ((known_mask & bit) != 0U) {
            return (value_mask & bit) != 0U;
        }
        return fallback_if_unknown;
    }
};

struct TouchControlConfig {
    float opacity = 0.55F;           // Idle button visibility (0.2..1.0)
    float button_scale = 1.0F;       // Overall button sizing multiplier
    float look_sensitivity = 1.0F;   // Right drag-to-look multiplier
    float aim_sensitivity = 0.55F;   // Multiplier applied while zooming/aiming
    bool aim_hold = true;            // Hold Zoom to zoom/aim instead of toggle
    bool gyro_enabled = false;       // Opt-in gyro aiming (default OFF)
    float gyro_sensitivity = 1.0F;
    bool gyro_invert_y = false;
    bool show_touch_controls = true;
    bool cheats_enabled = false;     // Auto-enabled when `cheat_mode = 1` in rgl_config.txt
};

// Menu pinch-to-zoom viewport transform applied by Vulkan `Present()` in cursor scenes.
struct MenuZoomView {
    float zoom = 1.0F;   // 1.0 = full frame, up to 2.25
    float pan_x = 0.5F;  // normalized center [0..1]
    float pan_y = 0.5F;
};

struct BufferedInputEvent {
    std::uint32_t ofs = 0U;
    std::uint32_t data = 0U;
    std::uint32_t timestamp_ms = 0U;
    std::uint32_t sequence = 0U;
};

struct XInputPadState {
    bool connected = false;
    std::uint32_t packet_number = 0U;
    std::uint16_t buttons = 0U;
    std::uint8_t left_trigger = 0U;
    std::uint8_t right_trigger = 0U;
    std::int16_t thumb_lx = 0;
    std::int16_t thumb_ly = 0;
    std::int16_t thumb_rx = 0;
    std::int16_t thumb_ry = 0;
};

enum class OverlayActionKind : std::uint8_t {
    none = 0,
    key_hold,           // Hold `dik_code` (+ optional Ctrl/Shift)
    key_toggle,         // Sticky toggle `dik_code` (+ optional Ctrl/Shift)
    mouse_button,       // Hold mouse button 0=LMB, 1=RMB, 2=MMB
    mouse_wheel_step,   // Pulse wheel delta (+120 / -120)
    slide_attack,       // Slide-to-aim melee direction + LMB attack
    easy_dismount,      // Brief downward look pitch + F interact
    zoom_hold,          // Hold-to-zoom (aim slowdown + opt-in gyro)
    orders_open,        // Switch to Orders 4-button sub-pad
    orders_close,       // Return from Orders sub-pad to Battle
    map_look_toggle,    // Toggle Campaign Map pan/zoom mode
    map_camera_action,  // Follow party / Home recenter
    map_stop_rest,      // Pause/resume map time (Space)
    trade_take_all,     // Bulk transfer shortcut
    trade_give_all,
    scene_cycle,        // Manual corner scene override pill
    pause_drawer,       // Open in-game pause/quick settings
    soft_keyboard,      // Request Android soft keyboard
    cheat_menu_toggle,  // Expand / collapse context-aware Cheats strip
    cheat_chord_pulse,  // Send 1-frame Ctrl(+Shift)(+Alt)+key cheat chord
};

struct TouchButtonSpec {
    std::string id;
    std::string label;
    TouchScene scene = TouchScene::battle;
    wse::platform::TouchIconId icon = wse::platform::TouchIconId::none;
    OverlayActionKind action = OverlayActionKind::none;
    std::uint8_t dik_code = 0U;
    std::uint8_t mouse_btn = 0U;
    bool need_ctrl = false;
    bool need_shift = false;
    float nx = 0.5F;        // Normalized screen center X [0..1]
    float ny = 0.5F;        // Normalized screen center Y [0..1]
    float size_short = 0.11F;  // Diameter as fraction of screen short side
    bool visible = true;
    bool enabled = true;
    bool latched = false;   // Visual gold highlight for sticky toggles
    bool pressed = false;
    bool is_extra = false;  // Assignable extra or auto-detected mod button
    bool need_alt = false;  // Alt modifier (e.g. Ctrl+Alt+F4)
    bool is_cheat = false;  // Shown only when `cfg_.cheats_enabled` and cheat strip is open
    bool any_scene = false; // Shown in every scene (e.g. Cheats toggle / Console)
};

class RecompInputHub {
public:
    using PostWindowMsgFn =
        std::function<void(std::uint32_t message, std::uint64_t wparam, std::uint64_t lparam)>;
    using UiActionFn = std::function<void(std::string_view action)>;
    using RumbleFn = std::function<void(std::uint16_t left_motor, std::uint16_t right_motor)>;

    // Global hook instance used by `r4_jni.cpp` and shims without modifying `r4_startup.cpp`.
    static RecompInputHub& shared() {
        static RecompInputHub instance;
        return instance;
    }

    RecompInputHub() : start_(std::chrono::steady_clock::now()) {
        init_default_buttons();
    }

    // ---- Embedder callbacks ----
    void set_window_msg_sink(PostWindowMsgFn fn) {
        const std::lock_guard<std::mutex> lock(m_);
        post_msg_ = std::move(fn);
    }
    void set_ui_action_sink(UiActionFn fn) {
        const std::lock_guard<std::mutex> lock(m_);
        on_ui_action_ = std::move(fn);
    }
    void set_rumble_sink(RumbleFn fn) {
        const std::lock_guard<std::mutex> lock(m_);
        on_rumble_ = std::move(fn);
    }

    // ---- Configuration & Mod Auto-Detection ----
    void set_config(const TouchControlConfig& cfg) {
        const std::lock_guard<std::mutex> lock(m_);
        cfg_ = cfg;
        cfg_.opacity = std::clamp(cfg_.opacity, 0.15F, 1.0F);
        cfg_.button_scale = std::clamp(cfg_.button_scale, 0.7F, 1.45F);
        cfg_.look_sensitivity = std::clamp(cfg_.look_sensitivity, 0.25F, 3.0F);
        cfg_.aim_sensitivity = std::clamp(cfg_.aim_sensitivity, 0.15F, 1.5F);
    }
    [[nodiscard]] TouchControlConfig config() const {
        const std::lock_guard<std::mutex> lock(m_);
        return cfg_;
    }

    void set_screen_size(int width, int height) {
        const std::lock_guard<std::mutex> lock(m_);
        screen_w_ = std::max(1, width);
        screen_h_ = std::max(1, height);
        if (surface_w_ <= 0 || surface_h_ <= 0) {
            surface_w_ = screen_w_;
            surface_h_ = screen_h_;
        }
        cursor_x_ = std::clamp(cursor_x_, 0, screen_w_ - 1);
        cursor_y_ = std::clamp(cursor_y_, 0, screen_h_ - 1);
    }

    // Tracks the live Android SurfaceView / DeX / external display pixel size so touch and mouse
    // coordinates are inverse-mapped through the same letterbox rect used by Vulkan Present.
    void set_surface_size(int width, int height) {
        const std::lock_guard<std::mutex> lock(m_);
        surface_w_ = std::max(1, width);
        surface_h_ = std::max(1, height);
    }

    // Releases all held touches, stick keys, mouse buttons, and modifiers (called on app pause).
    void release_all() {
        PostWindowMsgFn sink;
        int cx = 0;
        int cy = 0;
        bool release_lmb = false;
        bool release_rmb = false;
        {
            const std::lock_guard<std::mutex> lock(m_);
            sink = post_msg_;
            cx = cursor_x_;
            cy = cursor_y_;
            touches_.clear();
            stick_pointer_ = -1;
            update_stick_keys_locked(0.0F, 0.0F);
            attack_slide_active_ = false;
            zoom_active_ = false;
            dismount_frames_ = 0;
            if (pulse_release_dik_ != 0U) {
                set_key_locked(pulse_release_dik_, false);
                pulse_release_dik_ = 0U;
            }
            pulse_release_ctrl_ = false;
            pulse_release_shift_ = false;
            pulse_release_alt_ = false;
            phys_ctrl_ = false;
            phys_shift_ = false;
            phys_alt_ = false;
            pad_ctrl_ = false;
            pad_shift_ = false;
            for (auto& b : buttons_) {
                if (b.pressed) {
                    b.pressed = false;
                    if (b.dik_code != 0U) set_key_locked(b.dik_code, false);
                }
            }
            if (mouse_buttons_[0] != 0U) {
                set_mouse_button_locked(0, false);
                release_lmb = true;
            }
            if (mouse_buttons_[1] != 0U) {
                set_mouse_button_locked(1, false);
                release_rmb = true;
            }
            tap_phase_ = TapPhase::idle;
            tap_release_queued_ = false;
            recompute_modifiers_locked();
        }
        if (sink) {
            if (release_lmb) sink(0x0202U, 0U, pack_lparam(cx, cy));
            if (release_rmb) sink(0x0205U, 0U, pack_lparam(cx, cy));
        }
    }

    void set_safe_area(const wse::platform::SafeAreaInsets& insets) {
        const std::lock_guard<std::mutex> lock(m_);
        safe_area_ = insets;
    }

    // Applies automatically detected mod controls (from `detect_mod_controls`) into the
    // assignable extra slots without overwriting user-customized buttons.
    void apply_detected_mod_controls(const wse::platform::DetectedModControls& detected) {
        const std::lock_guard<std::mutex> lock(m_);
        if (detected.crouch_supported) {
            battle_ctx_.known_mask |= BattleContextFlags::kCrouchAllowed;
            battle_ctx_.value_mask |= BattleContextFlags::kCrouchAllowed;
        }
        // Populate extra slots 0..9 along the upper-right / mid-left edge cleanly.
        std::size_t slot = 0U;
        for (const auto& mod_btn : detected.buttons) {
            if (slot >= 10U) break;
            const std::size_t idx = extra_start_index_ + slot;
            if (idx >= buttons_.size()) break;
            TouchButtonSpec& dst = buttons_[idx];
            dst.id = mod_btn.id;
            dst.label = mod_btn.label;
            dst.icon = mod_btn.icon;
            dst.dik_code = mod_btn.dik_code;
            dst.need_ctrl = mod_btn.need_ctrl;
            dst.need_shift = mod_btn.need_shift;
            dst.action = mod_btn.toggle_mode ? OverlayActionKind::key_toggle : OverlayActionKind::key_hold;
            dst.visible = true;
            dst.enabled = true;
            ++slot;
        }
        // Hide unused extra slots unless manually enabled.
        for (; slot < 10U; ++slot) {
            const std::size_t idx = extra_start_index_ + slot;
            if (idx < buttons_.size()) {
                buttons_[idx].visible = false;
            }
        }
    }

    void set_cheats_enabled(bool enabled, bool auto_expand = false) {
        const std::lock_guard<std::mutex> lock(m_);
        cfg_.cheats_enabled = enabled;
        if (!enabled) {
            cheats_expanded_ = false;
        } else if (auto_expand) {
            cheats_expanded_ = true;
        }
        for (auto& b : buttons_) {
            if (b.id == "cheat_menu") b.latched = cheats_expanded_;
        }
    }
    [[nodiscard]] bool cheats_enabled() const {
        const std::lock_guard<std::mutex> lock(m_);
        return cfg_.cheats_enabled;
    }

    // ---- Scene & Cursor State ----
    void on_show_cursor(bool visible) {
        const std::lock_guard<std::mutex> lock(m_);
        cursor_visible_ = visible;
        if (!manual_scene_override_) {
            const TouchScene prev = scene_;
            if (!visible) {
                if (scene_ != TouchScene::orders) {
                    scene_ = TouchScene::battle;
                }
                reset_menu_zoom_locked();
            } else if (scene_ == TouchScene::battle || scene_ == TouchScene::orders) {
                scene_ = inferred_ui_scene_;
            }
            if (scene_ != prev) {
                on_scene_changed_locked(prev, scene_);
            }
        }
    }

    [[nodiscard]] bool cursor_visible() const {
        const std::lock_guard<std::mutex> lock(m_);
        return cursor_visible_;
    }

    void set_inferred_scene(TouchScene s) {
        const std::lock_guard<std::mutex> lock(m_);
        if (s != TouchScene::battle && s != TouchScene::orders) {
            inferred_ui_scene_ = s;
        }
        if (!manual_scene_override_) {
            const TouchScene prev = scene_;
            scene_ = s;
            if (s == TouchScene::battle || s == TouchScene::orders) {
                reset_menu_zoom_locked();
            }
            if (scene_ != prev) {
                on_scene_changed_locked(prev, scene_);
            }
        }
    }

    void set_battle_context(const BattleContextFlags& ctx) {
        const std::lock_guard<std::mutex> lock(m_);
        battle_ctx_ = ctx;
    }

    [[nodiscard]] TouchScene current_scene() const {
        const std::lock_guard<std::mutex> lock(m_);
        return scene_;
    }

    [[nodiscard]] ActiveInputMode active_input_mode() const {
        const std::lock_guard<std::mutex> lock(m_);
        return input_mode_;
    }

    [[nodiscard]] MenuZoomView menu_zoom_view() const {
        const std::lock_guard<std::mutex> lock(m_);
        return menu_zoom_;
    }

    // ---- Physical Keyboard & Mouse (Desktop Mode / DeX / Host) ----
    void inject_key(std::uint8_t dik_code, bool down, std::uint32_t vk_code = 0U) {
        PostWindowMsgFn sink;
        {
            const std::lock_guard<std::mutex> lock(m_);
            input_mode_ = ActiveInputMode::keyboard_mouse;
            if (dik_code == dik::kLCtrl) {
                phys_ctrl_ = down;
                recompute_modifiers_locked();
            } else if (dik_code == dik::kLShift) {
                phys_shift_ = down;
                recompute_modifiers_locked();
            } else if (dik_code == dik::kLAlt) {
                phys_alt_ = down;
                recompute_modifiers_locked();
            } else {
                set_key_locked(dik_code, down);
            }
            sink = post_msg_;
        }
        if (sink && vk_code != 0U) {
            sink(down ? 0x0100U : 0x0101U, vk_code, 0U);  // WM_KEYDOWN / WM_KEYUP
        }
    }

    // Injects UTF-16 / ASCII text from Android's soft keyboard or physical keyboard into USER32 `WM_CHAR`.
    void inject_text_utf8(std::string_view utf8) {
        PostWindowMsgFn sink;
        {
            const std::lock_guard<std::mutex> lock(m_);
            sink = post_msg_;
        }
        if (!sink) return;
        std::size_t i = 0U;
        while (i < utf8.size()) {
            std::uint32_t cp = 0U;
            const auto b0 = static_cast<std::uint8_t>(utf8[i]);
            if (b0 < 0x80U) {
                cp = b0;
                i += 1U;
            } else if ((b0 & 0xE0U) == 0xC0U && i + 1U < utf8.size()) {
                cp = ((b0 & 0x1FU) << 6U) | (static_cast<std::uint8_t>(utf8[i + 1U]) & 0x3FU);
                i += 2U;
            } else if ((b0 & 0xF0U) == 0xE0U && i + 2U < utf8.size()) {
                cp = ((b0 & 0x0FU) << 12U) | ((static_cast<std::uint8_t>(utf8[i + 1U]) & 0x3FU) << 6U) |
                     (static_cast<std::uint8_t>(utf8[i + 2U]) & 0x3FU);
                i += 3U;
            } else {
                ++i;
                continue;
            }
            if (cp == '\n') {
                sink(0x0100U, 0x0DU, 0U);  // VK_RETURN
                sink(0x0102U, 0x0DU, 0U);
                sink(0x0101U, 0x0DU, 0U);
            } else if (cp == '\b') {
                sink(0x0100U, 0x08U, 0U);  // VK_BACK
                sink(0x0102U, 0x08U, 0U);
                sink(0x0101U, 0x08U, 0U);
            } else if (cp <= 0xFFFFU) {
                sink(0x0102U, cp, 1U);  // WM_CHAR
            }
        }
    }

    void inject_mouse_move_abs(int x, int y) {
        PostWindowMsgFn sink;
        int cx = 0;
        int cy = 0;
        {
            const std::lock_guard<std::mutex> lock(m_);
            input_mode_ = ActiveInputMode::keyboard_mouse;
            float bbx = static_cast<float>(x);
            float bby = static_cast<float>(y);
            map_surface_to_backbuffer_locked(bbx, bby);
            const int nx = std::clamp(static_cast<int>(std::lround(bbx)), 0, screen_w_ - 1);
            const int ny = std::clamp(static_cast<int>(std::lround(bby)), 0, screen_h_ - 1);
            add_mouse_delta_float_locked(static_cast<float>(nx - cursor_x_), static_cast<float>(ny - cursor_y_));
            cursor_x_ = nx;
            cursor_y_ = ny;
            cx = cursor_x_;
            cy = cursor_y_;
            sink = post_msg_;
        }
        if (sink) {
            sink(0x0200U, 0U, pack_lparam(cx, cy));  // WM_MOUSEMOVE
        }
    }

    void inject_mouse_delta(int dx, int dy) {
        const std::lock_guard<std::mutex> lock(m_);
        input_mode_ = ActiveInputMode::keyboard_mouse;
        mouse_dx_ += dx;
        mouse_dy_ += dy;
        push_mouse_event_locked(0U, static_cast<std::uint32_t>(dx));
        push_mouse_event_locked(4U, static_cast<std::uint32_t>(dy));
    }

    void inject_mouse_button(int button, bool down) {
        PostWindowMsgFn sink;
        int cx = 0;
        int cy = 0;
        {
            const std::lock_guard<std::mutex> lock(m_);
            input_mode_ = ActiveInputMode::keyboard_mouse;
            set_mouse_button_locked(button, down);
            cx = cursor_x_;
            cy = cursor_y_;
            sink = post_msg_;
        }
        if (sink) {
            std::uint32_t msg = 0U;
            if (button == 0) msg = down ? 0x0201U : 0x0202U;       // WM_LBUTTONDOWN / UP
            else if (button == 1) msg = down ? 0x0204U : 0x0205U;  // WM_RBUTTONDOWN / UP
            else if (button == 2) msg = down ? 0x0207U : 0x0208U;  // WM_MBUTTONDOWN / UP
            if (msg != 0U) {
                sink(msg, down ? 1U : 0U, pack_lparam(cx, cy));
            }
        }
    }

    void inject_mouse_wheel(int wheel_delta) {
        PostWindowMsgFn sink;
        int cx = 0;
        int cy = 0;
        {
            const std::lock_guard<std::mutex> lock(m_);
            mouse_dz_ += wheel_delta;
            push_mouse_event_locked(8U, static_cast<std::uint32_t>(wheel_delta));
            cx = cursor_x_;
            cy = cursor_y_;
            sink = post_msg_;
        }
        if (sink) {
            const auto wp = static_cast<std::uint64_t>(static_cast<std::uint16_t>(wheel_delta)) << 16U;
            sink(0x020AU, wp, pack_lparam(cx, cy));  // WM_MOUSEWHEEL
        }
    }

    // ---- Gamepad (XInput + v4.3-style L1/L2/R2 map & party shortcuts) ----
    void update_gamepad(std::uint32_t user_index, const XInputPadState& state) {
        if (user_index >= 4U) return;
        const std::lock_guard<std::mutex> lock(m_);
        const XInputPadState prev = pads_[user_index];
        pads_[user_index] = state;
        pads_[user_index].packet_number = prev.packet_number + 1U;
        if (state.connected &&
            (state.buttons != 0U || state.left_trigger > 20U || state.right_trigger > 20U ||
             std::abs(state.thumb_lx) > 6000 || std::abs(state.thumb_ly) > 6000 ||
             std::abs(state.thumb_rx) > 6000 || std::abs(state.thumb_ry) > 6000)) {
            input_mode_ = ActiveInputMode::gamepad;
        }

        // Player 0 gamepad -> v4.3-style Warband QoL shortcuts when connected.
        if (user_index == 0U) {
            if (!state.connected) {
                if (pad_ctrl_ || pad_shift_) {
                    pad_ctrl_ = false;
                    pad_shift_ = false;
                    recompute_modifiers_locked();
                }
            } else {
                constexpr std::uint16_t kBtnLb = 0x0100U;
                bool want_ctrl = false;
                bool want_shift = false;
                // L1 (LB) on Campaign Map -> Fast Travel (Ctrl+Space)
                if (scene_ == TouchScene::campaign_map) {
                    const bool lb_now = (state.buttons & kBtnLb) != 0U;
                    const bool lb_prev = (prev.buttons & kBtnLb) != 0U;
                    want_ctrl = lb_now;
                    if (lb_now != lb_prev) {
                        set_key_locked(dik::kSpace, lb_now);
                    }
                }
                // L2 / R2 on Party screen -> Move Stack (Ctrl) / Move 10 (Shift)
                if (scene_ == TouchScene::party) {
                    want_ctrl = state.left_trigger > 80U;
                    want_shift = state.right_trigger > 80U;
                }
                if (want_ctrl != pad_ctrl_ || want_shift != pad_shift_) {
                    pad_ctrl_ = want_ctrl;
                    pad_shift_ = want_shift;
                    recompute_modifiers_locked();
                }
            }
        }
    }

    // ---- Opt-in Gyro Aiming (active only while zooming/aiming when enabled) ----
    // Android TYPE_GYROSCOPE in landscape (ROTATION_90): values[0] (device X, vertical) = yaw,
    // values[1] (device Y, horizontal) = pitch. Under ROTATION_270 both axes flip sign.
    void on_gyro_sample(float rad_per_sec_x, float rad_per_sec_y, float dt_sec, bool rotated_270 = false) {
        const std::lock_guard<std::mutex> lock(m_);
        if (!cfg_.gyro_enabled || !zoom_active_ || scene_ != TouchScene::battle) {
            return;
        }
        if (!std::isfinite(rad_per_sec_x) || !std::isfinite(rad_per_sec_y) || dt_sec <= 0.0F || dt_sec > 0.25F) {
            return;
        }
        // Low-pass filter (alpha = 0.35) + small deadzone to suppress hand tremor.
        gyro_lp_x_ = gyro_lp_x_ * 0.65F + rad_per_sec_x * 0.35F;
        gyro_lp_y_ = gyro_lp_y_ * 0.65F + rad_per_sec_y * 0.35F;
        const float yaw_rate = (std::fabs(gyro_lp_x_) < 0.025F) ? 0.0F : gyro_lp_x_;
        const float pitch_rate = (std::fabs(gyro_lp_y_) < 0.025F) ? 0.0F : gyro_lp_y_;
        const float scale = 320.0F * cfg_.gyro_sensitivity * cfg_.aim_sensitivity * dt_sec;
        const float rot_sign = rotated_270 ? -1.0F : 1.0F;
        const float sy = cfg_.gyro_invert_y ? -1.0F : 1.0F;
        add_mouse_delta_float_locked(-yaw_rate * scale * rot_sign, -pitch_rate * scale * rot_sign * sy);
    }

    // ---- Android Multi-Touch Input (`action`: 0=DOWN, 1=UP, 2=MOVE, 3=CANCEL) ----
    void on_touch_event(int pointer_id, int action, float x_px, float y_px) {
        PostWindowMsgFn msg_sink;
        UiActionFn ui_sink;
        std::string triggered_ui_action;
        int move_cx = -1;
        int move_cy = -1;

        {
            const std::lock_guard<std::mutex> lock(m_);
            input_mode_ = ActiveInputMode::touch;
            float bx = x_px;
            float by = y_px;
            map_surface_to_backbuffer_locked(bx, by);
            const float nx = std::clamp(bx / static_cast<float>(screen_w_), 0.0F, 1.0F);
            const float ny = std::clamp(by / static_cast<float>(screen_h_), 0.0F, 1.0F);

            if (action == 0) {  // DOWN
                handle_pointer_down_locked(pointer_id, bx, by, nx, ny, triggered_ui_action, move_cx, move_cy);
            } else if (action == 2) {  // MOVE
                handle_pointer_move_locked(pointer_id, bx, by, nx, ny, move_cx, move_cy);
            } else {  // UP or CANCEL
                handle_pointer_up_locked(pointer_id, bx, by, move_cx, move_cy);
            }
            msg_sink = post_msg_;
            ui_sink = on_ui_action_;
        }

        if (msg_sink && move_cx >= 0 && move_cy >= 0) {
            msg_sink(0x0200U, 0U, pack_lparam(move_cx, move_cy));
        }
        if (ui_sink && !triggered_ui_action.empty()) {
            ui_sink(triggered_ui_action);
        }
    }

    // Advances the two-phase cursor tap/drop state machine and easy-dismount micro-sequence
    // once per presented frame (called on the game thread at Present/poll).
    void tick_frame() {
        PostWindowMsgFn sink;
        std::vector<std::pair<std::uint32_t, std::pair<std::uint64_t, std::uint64_t>>> queued_msgs;
        {
            const std::lock_guard<std::mutex> lock(m_);
            sink = post_msg_;

            // Two-phase menu tap: frame N moves cursor, frame N+1 clicks, frame N+2 releases.
            if (tap_phase_ == TapPhase::hover_then_down) {
                set_mouse_button_locked(0, true);
                queued_msgs.push_back({0x0201U, {1U, pack_lparam(cursor_x_, cursor_y_)}});
                tap_phase_ = tap_release_queued_ ? TapPhase::hold_then_up : TapPhase::holding;
                tap_release_queued_ = false;
            } else if (tap_phase_ == TapPhase::hold_then_up) {
                set_mouse_button_locked(0, false);
                queued_msgs.push_back({0x0202U, {0U, pack_lparam(cursor_x_, cursor_y_)}});
                tap_phase_ = TapPhase::idle;
            }

            // Easy-dismount sequence: pitch down briefly, press F, release F.
            if (dismount_frames_ > 0) {
                if (dismount_frames_ > 2) {
                    add_mouse_delta_float_locked(0.0F, 110.0F);
                } else if (dismount_frames_ == 2) {
                    set_key_locked(dik::kF, true);
                } else if (dismount_frames_ == 1) {
                    set_key_locked(dik::kF, false);
                }
                --dismount_frames_;
            }

            // One-frame pulse release for discrete UI & cheat chord actions.
            if (pulse_release_dik_ != 0U) {
                set_key_locked(pulse_release_dik_, false);
                pulse_release_dik_ = 0U;
                pulse_release_ctrl_ = false;
                pulse_release_shift_ = false;
                pulse_release_alt_ = false;
                recompute_modifiers_locked();
            }
        }
        if (sink) {
            for (const auto& ev : queued_msgs) {
                sink(ev.first, ev.second.first, ev.second.second);
            }
        }
    }

    // ---- Consumer APIs for Shims (DirectInput 8, USER32, XInput 1.3) ----
    void get_cursor_pos(int& out_x, int& out_y) const {
        const std::lock_guard<std::mutex> lock(m_);
        out_x = cursor_x_;
        out_y = cursor_y_;
    }

    void set_cursor_pos(int x, int y) {
        const std::lock_guard<std::mutex> lock(m_);
        cursor_x_ = std::clamp(x, 0, screen_w_ - 1);
        cursor_y_ = std::clamp(y, 0, screen_h_ - 1);
    }

    void fill_dinput_keyboard_state(std::uint8_t* out_256, std::size_t size) const {
        if (out_256 == nullptr || size == 0U) return;
        const std::lock_guard<std::mutex> lock(m_);
        const std::size_t n = std::min<std::size_t>(size, keys_.size());
        std::memcpy(out_256, keys_.data(), n);
        if (size > n) std::memset(out_256 + n, 0, size - n);
    }

    void fill_dinput_mouse_state(void* out_ptr, std::size_t size) {
        if (out_ptr == nullptr || size < 16U) return;
        const std::lock_guard<std::mutex> lock(m_);
        auto* dst = static_cast<std::uint8_t*>(out_ptr);
        std::memset(dst, 0, size);
        const std::int32_t lx = mouse_dx_;
        const std::int32_t ly = mouse_dy_;
        const std::int32_t lz = mouse_dz_;
        mouse_dx_ = 0;
        mouse_dy_ = 0;
        mouse_dz_ = 0;
        std::memcpy(dst + 0U, &lx, 4U);
        std::memcpy(dst + 4U, &ly, 4U);
        std::memcpy(dst + 8U, &lz, 4U);
        const std::size_t btn_count = std::min<std::size_t>(size - 12U, mouse_buttons_.size());
        std::memcpy(dst + 12U, mouse_buttons_.data(), btn_count);
    }

    std::uint32_t read_dinput_events(bool is_mouse,
                                     std::uint32_t stride,
                                     void* out_buf,
                                     std::uint32_t max_count,
                                     bool peek) {
        const std::lock_guard<std::mutex> lock(m_);
        auto& q = is_mouse ? mouse_events_ : key_events_;
        if (out_buf == nullptr) {
            const auto avail = static_cast<std::uint32_t>(q.size());
            if (!peek && max_count == 0xFFFFFFFFU) {
                q.clear();
            }
            return std::min(avail, max_count);
        }
        const std::uint32_t count = std::min<std::uint32_t>(max_count, static_cast<std::uint32_t>(q.size()));
        const std::uint32_t eff_stride = std::max<std::uint32_t>(stride, 16U);
        auto* dst = static_cast<std::uint8_t*>(out_buf);
        for (std::uint32_t i = 0U; i < count; ++i) {
            const BufferedInputEvent& ev = q[i];
            std::uint8_t* slot = dst + static_cast<std::size_t>(i) * eff_stride;
            std::memset(slot, 0, eff_stride);
            std::memcpy(slot + 0U, &ev.ofs, 4U);
            std::memcpy(slot + 4U, &ev.data, 4U);
            std::memcpy(slot + 8U, &ev.timestamp_ms, 4U);
            std::memcpy(slot + 12U, &ev.sequence, 4U);
        }
        if (!peek) {
            for (std::uint32_t i = 0U; i < count; ++i) {
                q.pop_front();
            }
        }
        return count;
    }

    bool get_xinput_state(std::uint32_t user_index, XInputPadState& out) const {
        if (user_index >= 4U) return false;
        const std::lock_guard<std::mutex> lock(m_);
        if (!pads_[user_index].connected) {
            return false;
        }
        if (!xinput_enabled_) {
            // Per XInput 1.3 spec, XInputEnable(FALSE) returns ERROR_SUCCESS with neutral state.
            out = XInputPadState{};
            out.connected = true;
            out.packet_number = pads_[user_index].packet_number;
            return true;
        }
        out = pads_[user_index];
        return true;
    }

    bool set_xinput_vibration(std::uint32_t user_index, std::uint16_t left_motor, std::uint16_t right_motor) {
        RumbleFn cb;
        {
            const std::lock_guard<std::mutex> lock(m_);
            if (user_index >= 4U || !pads_[user_index].connected) return false;
            cb = on_rumble_;
        }
        if (cb) cb(left_motor, right_motor);
        return true;
    }

    void set_xinput_enabled(bool enabled) {
        const std::lock_guard<std::mutex> lock(m_);
        xinput_enabled_ = enabled;
    }

    // ---- Full-Resolution Procedural Touch Overlay Builder (Zero Per-Frame Allocations) ----
    // Fills `out_frame` (reusing its capacity) with iron-and-gold touch controls and simplistic
    // vector icons for the active scene. Returns false when touch overlay is hidden (e.g. gamepad
    // or keyboard/mouse active).
    bool build_overlay_geometry(wse::platform::GeometryFrame& out_frame,
                                std::uint32_t viewport_w,
                                std::uint32_t viewport_h) const {
        out_frame.vertices.clear();
        out_frame.indices.clear();
        out_frame.batches.clear();
        out_frame.width = viewport_w;
        out_frame.height = viewport_h;
        if (viewport_w == 0U || viewport_h == 0U) {
            return false;
        }

        const std::lock_guard<std::mutex> lock(m_);
        if (!cfg_.show_touch_controls || input_mode_ != ActiveInputMode::touch) {
            return false;
        }

        const float w = static_cast<float>(viewport_w);
        const float h = static_cast<float>(viewport_h);
        const float short_side = std::min(w, h);
        const std::size_t base_vtx = out_frame.vertices.size();
        const std::size_t base_idx = out_frame.indices.size();

        // P13 Iron-and-Gold translucent palette.
        const float idle_a = cfg_.opacity;
        const float press_a = std::min(0.92F, cfg_.opacity + 0.30F);

        // 1. Left virtual movement joystick (Battle & Orders scenes).
        if (scene_ == TouchScene::battle || scene_ == TouchScene::orders) {
            const float sx = stick_center_px_x(w);
            const float sy = stick_center_px_y(h);
            const float sr = short_side * 0.135F * cfg_.button_scale;
            const bool active = (stick_pointer_ >= 0);
            const float a = active ? press_a : idle_a;
            emit_disc(out_frame, base_vtx, sx, sy, sr, 0.10F, 0.11F, 0.13F, a * 0.55F, 24);
            emit_ring(out_frame, base_vtx, sx, sy, sr, 2.4F,
                      active ? 0.85F : 0.72F, active ? 0.68F : 0.68F, active ? 0.30F : 0.62F, a * 0.90F, 24);
            const float kx = sx + stick_dx_ * sr * 0.72F;
            const float ky = sy + stick_dy_ * sr * 0.72F;
            emit_disc(out_frame, base_vtx, kx, ky, sr * 0.36F,
                      active ? 0.88F : 0.78F, active ? 0.72F : 0.75F, active ? 0.35F : 0.68F, a * 0.88F, 18);
        }

        // 2. Scene-filtered & context-gated touch buttons + simplistic vector icons.
        for (const auto& btn : buttons_) {
            if (!is_button_visible_locked(btn)) {
                continue;
            }
            float bx = 0.0F;
            float by = 0.0F;
            float br = 0.0F;
            button_pixel_circle_locked(btn, w, h, bx, by, br);

            const bool hi = btn.pressed || btn.latched;
            const bool gold_cheat = btn.is_cheat || btn.id == "cheat_menu";
            const bool dialog_dimmed =
                (scene_ == TouchScene::dialog && (btn.id == "dialog_up" || btn.id == "dialog_down") &&
                 !battle_ctx_.test_or_default(BattleContextFlags::kDialogChoices, true));
            const bool eff_enabled = btn.enabled && !dialog_dimmed;
            const float alpha = !eff_enabled ? (idle_a * 0.32F) : (hi ? press_a : idle_a);
            const float fill_r = hi ? 0.24F : (gold_cheat ? 0.19F : 0.10F);
            const float fill_g = hi ? 0.19F : (gold_cheat ? 0.14F : 0.11F);
            const float fill_b = hi ? 0.11F : (gold_cheat ? 0.08F : 0.13F);
            const float rim_r = (hi || gold_cheat) ? 0.92F : 0.74F;
            const float rim_g = (hi || gold_cheat) ? 0.74F : 0.70F;
            const float rim_b = (hi || gold_cheat) ? 0.30F : 0.64F;

            emit_disc(out_frame, base_vtx, bx, by, br, fill_r, fill_g, fill_b, alpha * 0.70F, 20);
            emit_ring(out_frame, base_vtx, bx, by, br, hi ? 2.8F : 2.0F, rim_r, rim_g, rim_b, alpha * 0.95F, 20);

            // Slide-to-attack visual swing direction indicator arc on the Attack button.
            if (btn.action == OverlayActionKind::slide_attack && attack_slide_active_) {
                const float ax = bx + std::cos(attack_slide_angle_) * (br * 1.18F);
                const float ay = by + std::sin(attack_slide_angle_) * (br * 1.18F);
                emit_disc(out_frame, base_vtx, ax, ay, br * 0.22F, 0.95F, 0.78F, 0.32F, 0.95F, 12);
            }

            const wse::platform::TouchIconId icon = resolved_icon_locked(btn);
            const float icon_r = br * 0.56F;
            wse::platform::emit_touch_icon(out_frame, base_vtx, icon, bx, by, icon_r,
                                           hi ? 0.98F : 0.92F, hi ? 0.88F : 0.90F, hi ? 0.58F : 0.84F,
                                           btn.enabled ? std::min(1.0F, alpha + 0.25F) : alpha);
        }

        if (out_frame.indices.size() > base_idx) {
            wse::platform::GeometryBatch batch{};
            batch.texture = 0U;
            batch.blend = wse::platform::BlendMode::alpha;
            batch.first_vertex = static_cast<std::uint32_t>(base_vtx);
            batch.vertex_count = static_cast<std::uint32_t>(out_frame.vertices.size() - base_vtx);
            batch.first_index = static_cast<std::uint32_t>(base_idx);
            batch.index_count = static_cast<std::uint32_t>(out_frame.indices.size() - base_idx);
            out_frame.batches.push_back(batch);
            return true;
        }
        return false;
    }

    // Test/inspection helper returning visible buttons for the current scene & context.
    [[nodiscard]] std::vector<TouchButtonSpec> visible_buttons_snapshot() const {
        const std::lock_guard<std::mutex> lock(m_);
        std::vector<TouchButtonSpec> out;
        for (const auto& b : buttons_) {
            if (is_button_visible_locked(b)) {
                TouchButtonSpec copy = b;
                copy.icon = resolved_icon_locked(b);
                out.push_back(std::move(copy));
            }
        }
        return out;
    }

private:
    enum class TapPhase : std::uint8_t { idle = 0, hover_then_down = 1, holding = 2, hold_then_up = 3 };

    struct ActiveTouch {
        int pointer_id = -1;
        float start_x = 0.0F;
        float start_y = 0.0F;
        float last_x = 0.0F;
        float last_y = 0.0F;
        int button_index = -1;  // index into `buttons_` if bound to a button
        bool is_stick = false;
        bool is_look = false;
        bool is_cursor = false;
        bool defer_tap = false;
        bool dragged = false;
        float scroll_accum = 0.0F;
    };

    static std::uint64_t pack_lparam(int x, int y) noexcept {
        return (static_cast<std::uint64_t>(static_cast<std::uint16_t>(y)) << 16U) |
               static_cast<std::uint64_t>(static_cast<std::uint16_t>(x));
    }

    std::uint32_t now_ms_locked() const noexcept {
        return static_cast<std::uint32_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_).count());
    }

    void reset_menu_zoom_locked() noexcept {
        menu_zoom_ = MenuZoomView{1.0F, 0.5F, 0.5F};
    }

    void set_key_locked(std::uint8_t dik_code, bool down) {
        if (dik_code == 0U) return;
        const std::uint8_t val = down ? 0x80U : 0x00U;
        if (keys_[dik_code] == val) return;
        keys_[dik_code] = val;
        if (key_events_.size() >= 256U) key_events_.pop_front();
        key_events_.push_back(BufferedInputEvent{dik_code, val, now_ms_locked(), ++event_seq_});
    }

    void set_mouse_button_locked(int button, bool down) {
        if (button < 0 || button >= 8) return;
        const std::uint8_t val = down ? 0x80U : 0x00U;
        if (mouse_buttons_[static_cast<std::size_t>(button)] == val) return;
        mouse_buttons_[static_cast<std::size_t>(button)] = val;
        push_mouse_event_locked(static_cast<std::uint32_t>(12 + button), val);
    }

    void push_mouse_event_locked(std::uint32_t ofs, std::uint32_t data) {
        if (mouse_events_.size() >= 256U) mouse_events_.pop_front();
        mouse_events_.push_back(BufferedInputEvent{ofs, data, now_ms_locked(), ++event_seq_});
    }

    float stick_center_px_x(float w) const noexcept {
        const float left = safe_area_.left * w;
        const float usable_w = w * (1.0F - safe_area_.left - safe_area_.right);
        return left + usable_w * 0.135F;
    }

    float stick_center_px_y(float h) const noexcept {
        const float top = safe_area_.top * h;
        const float usable_h = h * (1.0F - safe_area_.top - safe_area_.bottom);
        return top + usable_h * 0.72F;
    }

    void button_pixel_circle_locked(const TouchButtonSpec& btn, float w, float h,
                                    float& out_cx, float& out_cy, float& out_r) const noexcept {
        const float left = safe_area_.left * w;
        const float top = safe_area_.top * h;
        const float uw = w * (1.0F - safe_area_.left - safe_area_.right);
        const float uh = h * (1.0F - safe_area_.top - safe_area_.bottom);
        const float short_side = std::min(w, h);
        out_r = std::max(14.0F, 0.5F * short_side * btn.size_short * cfg_.button_scale);
        out_cx = std::clamp(left + btn.nx * uw, left + out_r + 4.0F, left + uw - out_r - 4.0F);
        out_cy = std::clamp(top + btn.ny * uh, top + out_r + 4.0F, top + uh - out_r - 4.0F);
    }

    // Maps raw screen touch pixel into game back-buffer coordinates, accounting for menu pinch-zoom.
    void map_screen_to_game_locked(float x_px, float y_px, int& out_gx, int& out_gy) const noexcept {
        const float u = std::clamp(x_px / static_cast<float>(screen_w_), 0.0F, 1.0F);
        const float v = std::clamp(y_px / static_cast<float>(screen_h_), 0.0F, 1.0F);
        const float z = std::max(1.0F, menu_zoom_.zoom);
        const float gu = std::clamp(menu_zoom_.pan_x + (u - 0.5F) / z, 0.0F, 1.0F);
        const float gv = std::clamp(menu_zoom_.pan_y + (v - 0.5F) / z, 0.0F, 1.0F);
        out_gx = std::clamp(static_cast<int>(std::lround(gu * static_cast<float>(screen_w_ - 1))), 0, screen_w_ - 1);
        out_gy = std::clamp(static_cast<int>(std::lround(gv * static_cast<float>(screen_h_ - 1))), 0, screen_h_ - 1);
    }

    [[nodiscard]] bool is_button_visible_locked(const TouchButtonSpec& b) const noexcept {
        if (!b.visible) return false;
        // Global corner controls (pause/drawer & manual mode override).
        if (b.id == "pause_drawer" || b.id == "scene_mode") {
            return true;
        }
        if (b.id == "cheat_menu") {
            return cfg_.cheats_enabled;
        }
        if (b.is_cheat) {
            if (!cfg_.cheats_enabled || !cheats_expanded_) return false;
            return b.any_scene || b.scene == scene_;
        }
        if (b.id == "soft_keyboard") {
            return scene_ == TouchScene::main_menu || scene_ == TouchScene::screens ||
                   scene_ == TouchScene::dialog || scene_ == TouchScene::trade || scene_ == TouchScene::party;
        }
        if (b.is_extra) {
            return scene_ == TouchScene::battle;
        }
        if (b.scene != scene_) {
            return false;
        }
        if (scene_ == TouchScene::battle) {
            if (b.id == "use") {
                return battle_ctx_.test_or_default(BattleContextFlags::kInteract, true) ||
                       battle_ctx_.test_or_default(BattleContextFlags::kMount, false) ||
                       battle_ctx_.test_or_default(BattleContextFlags::kDismount, false);
            }
            if (b.id == "kick") {
                return battle_ctx_.test_or_default(BattleContextFlags::kOnFoot, true) ||
                       battle_ctx_.test_or_default(BattleContextFlags::kRiding, false);
            }
            if (b.id == "switch_weapon") {
                return battle_ctx_.test_or_default(BattleContextFlags::kHasWeapons, true);
            }
            if (b.id == "next_shield") {
                return battle_ctx_.test_or_default(BattleContextFlags::kHasShield, true);
            }
            if (b.id == "weapon_mode") {
                return battle_ctx_.test_or_default(BattleContextFlags::kHasAltMode, true);
            }
            if (b.id == "drop") {
                return battle_ctx_.test_or_default(BattleContextFlags::kArmed, true);
            }
            if (b.id == "crouch") {
                return battle_ctx_.test_or_default(BattleContextFlags::kCrouchAllowed, false) &&
                       battle_ctx_.test_or_default(BattleContextFlags::kOnFoot, true);
            }
            if (b.id == "orders" || b.id == "orders_more") {
                return battle_ctx_.test_or_default(BattleContextFlags::kHasTroops, true);
            }
            if (b.id == "chat") {
                return battle_ctx_.test_or_default(BattleContextFlags::kMultiplayer, false);
            }
        }
        if (scene_ == TouchScene::party) {
            if (b.id == "party_stack" || b.id == "party_ten") {
                return battle_ctx_.test_or_default(BattleContextFlags::kPartyMovable, true);
            }
        }
        return true;
    }

    [[nodiscard]] wse::platform::TouchIconId resolved_icon_locked(const TouchButtonSpec& b) const noexcept {
        if (b.id == "use") {
            if (battle_ctx_.test_or_default(BattleContextFlags::kDismount, false)) {
                return wse::platform::TouchIconId::horse_dismount;
            }
            if (battle_ctx_.test_or_default(BattleContextFlags::kMount, false)) {
                return wse::platform::TouchIconId::horse_mount;
            }
            return wse::platform::TouchIconId::hand_use;
        }
        if (b.id == "kick" && battle_ctx_.test_or_default(BattleContextFlags::kRiding, false) &&
            !battle_ctx_.test_or_default(BattleContextFlags::kOnFoot, true)) {
            return wse::platform::TouchIconId::hand_use;
        }
        if (b.id == "map_camera") {
            return map_look_mode_ ? wse::platform::TouchIconId::map_home : wse::platform::TouchIconId::map_follow;
        }
        return b.icon;
    }

    void update_stick_keys_locked(float dx, float dy) {
        stick_dx_ = dx;
        stick_dy_ = dy;
        constexpr float kDead = 0.24F;
        set_key_locked(dik::kW, dy < -kDead);
        set_key_locked(dik::kS, dy > kDead);
        set_key_locked(dik::kA, dx < -kDead);
        set_key_locked(dik::kD, dx > kDead);
    }

    void handle_pointer_down_locked(int pointer_id, float x_px, float y_px, float, float,
                                    std::string& out_ui_action, int& out_move_cx, int& out_move_cy) {
        const float w = static_cast<float>(screen_w_);
        const float h = static_cast<float>(screen_h_);
        const float short_side = std::min(w, h);

        ActiveTouch t{};
        t.pointer_id = pointer_id;
        t.start_x = t.last_x = x_px;
        t.start_y = t.last_y = y_px;

        // 1. Hit-test visible & enabled touch buttons first (when touch overlay is shown).
        if (cfg_.show_touch_controls) {
            for (std::size_t i = 0U; i < buttons_.size(); ++i) {
                TouchButtonSpec& b = buttons_[i];
                if (!is_button_visible_locked(b) || !b.enabled) continue;
                float bx = 0.0F, by = 0.0F, br = 0.0F;
                button_pixel_circle_locked(b, w, h, bx, by, br);
                const float dx = x_px - bx;
                const float dy = y_px - by;
                if (dx * dx + dy * dy <= (br * 1.12F) * (br * 1.12F)) {
                    t.button_index = static_cast<int>(i);
                    b.pressed = true;
                    activate_button_down_locked(b, out_ui_action);
                    touches_.push_back(t);
                    return;
                }
            }

            // 2. Hit-test left virtual movement stick in Battle / Orders.
            if ((scene_ == TouchScene::battle || scene_ == TouchScene::orders) && stick_pointer_ < 0) {
                const float sx = stick_center_px_x(w);
                const float sy = stick_center_px_y(h);
                const float sr = short_side * 0.16F * cfg_.button_scale;
                const float dx = x_px - sx;
                const float dy = y_px - sy;
                if (dx * dx + dy * dy <= sr * sr) {
                    stick_pointer_ = pointer_id;
                    t.is_stick = true;
                    const float len = std::sqrt(dx * dx + dy * dy);
                    const float norm = std::max(sr * 0.85F, 1.0F);
                    const float scale = (len > norm) ? (norm / len) : 1.0F;
                    update_stick_keys_locked((dx * scale) / norm, (dy * scale) / norm);
                    touches_.push_back(t);
                    return;
                }
            }
        }

        // 3. In Battle / Orders: any remaining touch drives drag-to-look camera.
        if (scene_ == TouchScene::battle || scene_ == TouchScene::orders) {
            t.is_look = true;
            touches_.push_back(t);
            return;
        }

        // 4. In cursor scenes (Main Menu, Campaign Map, Party, Trade, Dialog, Screens):
        //    Two fingers engage pinch-to-zoom / map camera pan; single finger starts two-phase tap!
        t.is_cursor = true;
        touches_.push_back(t);
        if (count_cursor_touches_locked() >= 2U) {
            // Cancel any pending single-finger tap when second finger lands for pinch/pan.
            if (tap_phase_ == TapPhase::hover_then_down) {
                tap_phase_ = TapPhase::idle;
            } else if (tap_phase_ == TapPhase::holding) {
                tap_phase_ = TapPhase::hold_then_up;
            }
            begin_two_finger_gesture_locked();
            return;
        }

        // Campaign Map in `Map Look` mode: single finger pans/rotates camera via RMB drag.
        if (scene_ == TouchScene::campaign_map && map_look_mode_) {
            set_mouse_button_locked(1, true);
            return;
        }

        // Two-phase menu tap:
        // In scrollable list scenes (`dialog`, `screens`), defer LMB press until finger-up so
        // dragging to scroll never accidentally clicks and commits a dialog/list choice!
        // Also flush any pending LMB release before starting a rapid second tap.
        if (tap_phase_ == TapPhase::hold_then_up || mouse_buttons_[0] != 0U) {
            set_mouse_button_locked(0, false);
            tap_phase_ = TapPhase::idle;
            tap_release_queued_ = false;
        }
        map_screen_to_game_locked(x_px, y_px, cursor_x_, cursor_y_);
        out_move_cx = cursor_x_;
        out_move_cy = cursor_y_;
        if (scene_ == TouchScene::dialog || scene_ == TouchScene::screens) {
            touches_.back().defer_tap = true;
        } else {
            tap_phase_ = TapPhase::hover_then_down;
            tap_release_queued_ = false;
        }
    }

    void handle_pointer_move_locked(int pointer_id, float x_px, float y_px, float, float,
                                    int& out_move_cx, int& out_move_cy) {
        ActiveTouch* t = find_touch_locked(pointer_id);
        if (t == nullptr) return;
        const float dx = x_px - t->last_x;
        const float dy = y_px - t->last_y;
        t->last_x = x_px;
        t->last_y = y_px;

        const float w = static_cast<float>(screen_w_);
        const float h = static_cast<float>(screen_h_);
        const float short_side = std::min(w, h);

        if (t->is_stick) {
            const float sx = stick_center_px_x(w);
            const float sy = stick_center_px_y(h);
            const float norm = std::max(short_side * 0.135F * cfg_.button_scale, 1.0F);
            float vx = (x_px - sx) / norm;
            float vy = (y_px - sy) / norm;
            const float len = std::sqrt(vx * vx + vy * vy);
            if (len > 1.0F) {
                vx /= len;
                vy /= len;
            }
            update_stick_keys_locked(vx, vy);
            return;
        }

        if (t->button_index >= 0 && static_cast<std::size_t>(t->button_index) < buttons_.size()) {
            const TouchButtonSpec& b = buttons_[static_cast<std::size_t>(t->button_index)];
            if (b.action == OverlayActionKind::slide_attack) {
                const float total_dx = x_px - t->start_x;
                const float total_dy = y_px - t->start_y;
                if (total_dx * total_dx + total_dy * total_dy > 144.0F) {
                    attack_slide_active_ = true;
                    attack_slide_angle_ = std::atan2(total_dy, total_dx);
                }
                const float sens = 0.65F * cfg_.look_sensitivity * (zoom_active_ ? cfg_.aim_sensitivity : 1.0F);
                add_mouse_delta_float_locked(dx * sens, dy * sens);
            }
            return;
        }

        if (t->is_look) {
            const float sens = 0.85F * cfg_.look_sensitivity * (zoom_active_ ? cfg_.aim_sensitivity : 1.0F);
            add_mouse_delta_float_locked(dx * sens, dy * sens);
            return;
        }

        if (t->is_cursor) {
            if (count_cursor_touches_locked() >= 2U) {
                update_two_finger_gesture_locked();
                return;
            }
            if (scene_ == TouchScene::campaign_map && map_look_mode_) {
                add_mouse_delta_float_locked(dx * 0.8F, dy * 0.8F);
                return;
            }
            // In Dialog / Screens, vertical drag > 18 px smoothly scrolls lists via mouse wheel.
            const float dist_sq = (x_px - t->start_x) * (x_px - t->start_x) +
                                  (y_px - t->start_y) * (y_px - t->start_y);
            if (!t->dragged && dist_sq > 324.0F) {
                t->dragged = true;
            }
            if (t->dragged && (scene_ == TouchScene::dialog || scene_ == TouchScene::screens)) {
                t->scroll_accum += dy;
                while (t->scroll_accum >= 36.0F) {
                    mouse_dz_ += 120;
                    push_mouse_event_locked(8U, 120U);
                    t->scroll_accum -= 36.0F;
                }
                while (t->scroll_accum <= -36.0F) {
                    mouse_dz_ -= 120;
                    push_mouse_event_locked(8U, static_cast<std::uint32_t>(-120));
                    t->scroll_accum += 36.0F;
                }
            }
            // Keep cursor synchronized with finger so dragged items drop right where lifted.
            map_screen_to_game_locked(x_px, y_px, cursor_x_, cursor_y_);
            out_move_cx = cursor_x_;
            out_move_cy = cursor_y_;
        }
    }

    void handle_pointer_up_locked(int pointer_id, float x_px, float y_px, int& out_move_cx, int& out_move_cy) {
        for (auto it = touches_.begin(); it != touches_.end(); ++it) {
            if (it->pointer_id != pointer_id) continue;
            const ActiveTouch t = *it;
            touches_.erase(it);

            if (t.is_stick) {
                stick_pointer_ = -1;
                update_stick_keys_locked(0.0F, 0.0F);
                return;
            }
            if (t.button_index >= 0 && static_cast<std::size_t>(t.button_index) < buttons_.size()) {
                TouchButtonSpec& b = buttons_[static_cast<std::size_t>(t.button_index)];
                b.pressed = false;
                activate_button_up_locked(b);
                return;
            }
            if (t.is_cursor) {
                if (scene_ == TouchScene::campaign_map && map_look_mode_) {
                    set_mouse_button_locked(1, false);
                    return;
                }
                // Update final drop coordinates before releasing LMB so dragged items land accurately!
                map_screen_to_game_locked(x_px, y_px, cursor_x_, cursor_y_);
                out_move_cx = cursor_x_;
                out_move_cy = cursor_y_;
                if (t.defer_tap) {
                    // Only click if the user tapped without scrolling the list!
                    if (!t.dragged) {
                        tap_phase_ = TapPhase::hover_then_down;
                        tap_release_queued_ = true;
                    }
                } else if (tap_phase_ == TapPhase::hover_then_down) {
                    // Quick tap: ensure press happens on next frame and release on the following frame.
                    tap_release_queued_ = true;
                } else if (tap_phase_ == TapPhase::holding) {
                    tap_phase_ = TapPhase::hold_then_up;
                }
            }
            return;
        }
    }

    void activate_button_down_locked(TouchButtonSpec& b, std::string& out_ui_action) {
        switch (b.action) {
        case OverlayActionKind::key_hold:
            set_key_locked(b.dik_code, true);
            recompute_modifiers_locked();
            break;
        case OverlayActionKind::key_toggle:
            b.latched = !b.latched;
            if (b.dik_code != dik::kLCtrl && b.dik_code != dik::kLShift && b.dik_code != dik::kLAlt) {
                set_key_locked(b.dik_code, b.latched);
            }
            recompute_modifiers_locked();
            break;
        case OverlayActionKind::mouse_button:
            set_mouse_button_locked(b.mouse_btn, true);
            break;
        case OverlayActionKind::mouse_wheel_step: {
            const int delta = (b.mouse_btn == 0U) ? 120 : -120;
            mouse_dz_ += delta;
            push_mouse_event_locked(8U, static_cast<std::uint32_t>(delta));
            break;
        }
        case OverlayActionKind::slide_attack:
            attack_slide_active_ = false;
            set_mouse_button_locked(0, true);
            break;
        case OverlayActionKind::easy_dismount:
            if (battle_ctx_.test_or_default(BattleContextFlags::kDismount, false)) {
                dismount_frames_ = 5;
            } else {
                set_key_locked(dik::kF, true);
            }
            break;
        case OverlayActionKind::zoom_hold:
            if (cfg_.aim_hold) {
                zoom_active_ = true;
                b.latched = true;
            } else {
                zoom_active_ = !zoom_active_;
                b.latched = zoom_active_;
            }
            recompute_modifiers_locked();
            break;
        case OverlayActionKind::orders_open: {
            const TouchScene prev = scene_;
            scene_ = TouchScene::orders;
            on_scene_changed_locked(prev, scene_);
            break;
        }
        case OverlayActionKind::orders_close: {
            const TouchScene prev = scene_;
            scene_ = TouchScene::battle;
            on_scene_changed_locked(prev, scene_);
            break;
        }
        case OverlayActionKind::map_look_toggle:
            map_look_mode_ = !map_look_mode_;
            b.latched = map_look_mode_;
            break;
        case OverlayActionKind::map_camera_action:
            if (map_look_mode_) {
                map_look_mode_ = false;
                for (auto& ob : buttons_) {
                    if (ob.id == "map_look") ob.latched = false;
                }
            }
            // Home recenters/snaps camera on the player party (Space is reserved for Rest/Pause).
            set_key_locked(dik::kHome, true);
            pulse_release_dik_ = dik::kHome;
            break;
        case OverlayActionKind::map_stop_rest:
            set_key_locked(dik::kSpace, true);
            break;
        case OverlayActionKind::trade_take_all:
            pulse_release_ctrl_ = true;
            recompute_modifiers_locked();
            set_key_locked(dik::kA, true);
            pulse_release_dik_ = dik::kA;
            break;
        case OverlayActionKind::trade_give_all:
            pulse_release_ctrl_ = true;
            recompute_modifiers_locked();
            set_key_locked(dik::kD, true);
            pulse_release_dik_ = dik::kD;
            break;
        case OverlayActionKind::scene_cycle: {
            const TouchScene prev = scene_;
            if (manual_scene_override_ && scene_ == TouchScene::screens) {
                // Final slot returns to automatic PDB/cursor scene detection.
                manual_scene_override_ = false;
                b.latched = false;
                scene_ = cursor_visible_ ? inferred_ui_scene_ : TouchScene::battle;
            } else {
                manual_scene_override_ = true;
                b.latched = true;
                const auto next_val = (static_cast<std::uint8_t>(scene_) + 1U) % 8U;
                scene_ = static_cast<TouchScene>(next_val);
                if (scene_ == TouchScene::orders) scene_ = TouchScene::campaign_map;
            }
            if (scene_ != prev) {
                on_scene_changed_locked(prev, scene_);
            }
            break;
        }
        case OverlayActionKind::pause_drawer:
            out_ui_action = "pause_drawer";
            break;
        case OverlayActionKind::soft_keyboard:
            out_ui_action = "soft_keyboard";
            break;
        case OverlayActionKind::cheat_menu_toggle:
            cheats_expanded_ = !cheats_expanded_;
            b.latched = cheats_expanded_;
            break;
        case OverlayActionKind::cheat_chord_pulse:
            pulse_release_ctrl_ = b.need_ctrl;
            pulse_release_shift_ = b.need_shift;
            pulse_release_alt_ = b.need_alt;
            recompute_modifiers_locked();
            set_key_locked(b.dik_code, true);
            pulse_release_dik_ = b.dik_code;
            break;
        case OverlayActionKind::none:
        default:
            break;
        }
    }

    void activate_button_up_locked(TouchButtonSpec& b) {
        switch (b.action) {
        case OverlayActionKind::key_hold:
            set_key_locked(b.dik_code, false);
            recompute_modifiers_locked();
            break;
        case OverlayActionKind::mouse_button:
            set_mouse_button_locked(b.mouse_btn, false);
            break;
        case OverlayActionKind::slide_attack:
            attack_slide_active_ = false;
            set_mouse_button_locked(0, false);
            break;
        case OverlayActionKind::easy_dismount:
            set_key_locked(dik::kF, false);
            break;
        case OverlayActionKind::zoom_hold:
            if (cfg_.aim_hold) {
                zoom_active_ = false;
                b.latched = false;
                recompute_modifiers_locked();
            }
            break;
        case OverlayActionKind::map_stop_rest:
            set_key_locked(dik::kSpace, false);
            break;
        default:
            break;
        }
    }

    void on_scene_changed_locked(TouchScene prev, TouchScene next) {
        if (prev == next) return;
        for (auto& b : buttons_) {
            if (b.action == OverlayActionKind::key_toggle && b.latched &&
                b.scene != next && b.id != "scene_mode" && b.id != "cheat_menu") {
                b.latched = false;
                if (b.dik_code != 0U && b.dik_code != dik::kLCtrl && b.dik_code != dik::kLShift &&
                    b.dik_code != dik::kLAlt) {
                    set_key_locked(b.dik_code, false);
                }
            }
        }
        if (next != TouchScene::battle) {
            zoom_active_ = false;
            for (auto& b : buttons_) {
                if (b.action == OverlayActionKind::zoom_hold) b.latched = false;
            }
        }
        if (next != TouchScene::campaign_map) {
            map_look_mode_ = false;
            for (auto& b : buttons_) {
                if (b.action == OverlayActionKind::map_look_toggle) b.latched = false;
            }
        }
        recompute_modifiers_locked();
    }

    void recompute_modifiers_locked() {
        bool ctrl = phys_ctrl_ || pulse_release_ctrl_ || pad_ctrl_;
        bool shift = phys_shift_ || pulse_release_shift_ || pad_shift_ || zoom_active_;
        bool alt = phys_alt_ || pulse_release_alt_;
        for (const auto& b : buttons_) {
            if (b.pressed) {
                if (b.need_ctrl || b.dik_code == dik::kLCtrl) ctrl = true;
                if (b.need_shift || b.dik_code == dik::kLShift) shift = true;
                if (b.need_alt || b.dik_code == dik::kLAlt) alt = true;
            }
            if (b.action == OverlayActionKind::key_toggle && b.latched && is_button_visible_locked(b)) {
                if (b.need_ctrl || b.dik_code == dik::kLCtrl) ctrl = true;
                if (b.need_shift || b.dik_code == dik::kLShift) shift = true;
                if (b.need_alt || b.dik_code == dik::kLAlt) alt = true;
            }
        }
        set_key_locked(dik::kLCtrl, ctrl);
        set_key_locked(dik::kLShift, shift);
        set_key_locked(dik::kLAlt, alt);
    }

    void add_mouse_delta_float_locked(float fdx, float fdy) {
        if (!std::isfinite(fdx)) fdx = 0.0F;
        if (!std::isfinite(fdy)) fdy = 0.0F;
        mouse_rem_x_ += fdx;
        mouse_rem_y_ += fdy;
        const int ix = static_cast<int>(std::trunc(mouse_rem_x_));
        const int iy = static_cast<int>(std::trunc(mouse_rem_y_));
        if (ix != 0) {
            mouse_rem_x_ -= static_cast<float>(ix);
            mouse_dx_ += ix;
            push_mouse_event_locked(0U, static_cast<std::uint32_t>(ix));
        }
        if (iy != 0) {
            mouse_rem_y_ -= static_cast<float>(iy);
            mouse_dy_ += iy;
            push_mouse_event_locked(4U, static_cast<std::uint32_t>(iy));
        }
    }

    // Inverse-maps live SurfaceView/DeX pixel coordinates through Vulkan's letterbox rect into
    // D3D9 backbuffer pixels [0..screen_w_) x [0..screen_h_).
    void map_surface_to_backbuffer_locked(float& x_px, float& y_px) const noexcept {
        if (surface_w_ <= 0 || surface_h_ <= 0 || (surface_w_ == screen_w_ && surface_h_ == screen_h_)) {
            return;
        }
        const std::int64_t lhs = static_cast<std::int64_t>(surface_w_) * screen_h_;
        const std::int64_t rhs = static_cast<std::int64_t>(surface_h_) * screen_w_;
        float rx = 0.0F;
        float ry = 0.0F;
        float rw = static_cast<float>(surface_w_);
        float rh = static_cast<float>(surface_h_);
        if (lhs > rhs) {
            rh = static_cast<float>(surface_h_);
            rw = std::max(1.0F, static_cast<float>(rhs) / static_cast<float>(screen_h_));
            rx = (static_cast<float>(surface_w_) - rw) * 0.5F;
            ry = 0.0F;
        } else {
            rw = static_cast<float>(surface_w_);
            rh = std::max(1.0F, static_cast<float>(lhs) / static_cast<float>(screen_w_));
            rx = 0.0F;
            ry = (static_cast<float>(surface_h_) - rh) * 0.5F;
        }
        x_px = std::clamp((x_px - rx) / rw, 0.0F, 1.0F) * static_cast<float>(screen_w_);
        y_px = std::clamp((y_px - ry) / rh, 0.0F, 1.0F) * static_cast<float>(screen_h_);
    }

    ActiveTouch* find_touch_locked(int pointer_id) noexcept {
        for (auto& t : touches_) {
            if (t.pointer_id == pointer_id) return &t;
        }
        return nullptr;
    }

    std::size_t count_cursor_touches_locked() const noexcept {
        std::size_t n = 0U;
        for (const auto& t : touches_) {
            if (t.is_cursor) ++n;
        }
        return n;
    }

    void begin_two_finger_gesture_locked() noexcept {
        const ActiveTouch* p0 = nullptr;
        const ActiveTouch* p1 = nullptr;
        for (const auto& t : touches_) {
            if (!t.is_cursor) continue;
            if (!p0) p0 = &t;
            else {
                p1 = &t;
                break;
            }
        }
        if (!p0 || !p1) return;
        const float dx = p1->last_x - p0->last_x;
        const float dy = p1->last_y - p0->last_y;
        pinch_prev_dist_ = std::max(10.0F, std::sqrt(dx * dx + dy * dy));
        pinch_prev_mid_x_ = 0.5F * (p0->last_x + p1->last_x);
        pinch_prev_mid_y_ = 0.5F * (p0->last_y + p1->last_y);
    }

    void update_two_finger_gesture_locked() noexcept {
        const ActiveTouch* p0 = nullptr;
        const ActiveTouch* p1 = nullptr;
        for (const auto& t : touches_) {
            if (!t.is_cursor) continue;
            if (!p0) p0 = &t;
            else {
                p1 = &t;
                break;
            }
        }
        if (!p0 || !p1) return;
        const float dx = p1->last_x - p0->last_x;
        const float dy = p1->last_y - p0->last_y;
        const float dist = std::max(10.0F, std::sqrt(dx * dx + dy * dy));
        const float mid_x = 0.5F * (p0->last_x + p1->last_x);
        const float mid_y = 0.5F * (p0->last_y + p1->last_y);
        const float delta_dist = dist - pinch_prev_dist_;
        const float delta_mx = mid_x - pinch_prev_mid_x_;
        const float delta_my = mid_y - pinch_prev_mid_y_;

        if (scene_ == TouchScene::campaign_map) {
            // Campaign Map: two-finger pinch -> zoom (wheel), two-finger drag -> rotate/pan.
            if (std::fabs(delta_dist) >= 12.0F) {
                const int wheel = (delta_dist > 0.0F) ? 120 : -120;
                mouse_dz_ += wheel;
                push_mouse_event_locked(8U, static_cast<std::uint32_t>(wheel));
                pinch_prev_dist_ = dist;
            }
            mouse_dx_ += static_cast<int>(std::lround(delta_mx * 0.75F));
            mouse_dy_ += static_cast<int>(std::lround(delta_my * 0.75F));
        } else {
            // Readable Menus (P14.5): pinch zooms and pans the rendered frame cleanly in Present().
            if (pinch_prev_dist_ > 1.0F) {
                const float ratio = dist / pinch_prev_dist_;
                menu_zoom_.zoom = std::clamp(menu_zoom_.zoom * ratio, 1.0F, 2.25F);
            }
            if (menu_zoom_.zoom <= 1.02F) {
                reset_menu_zoom_locked();
            } else {
                const float z = menu_zoom_.zoom;
                const float half = 0.5F / z;
                menu_zoom_.pan_x = std::clamp(menu_zoom_.pan_x - (delta_mx / static_cast<float>(screen_w_)) / z,
                                              half, 1.0F - half);
                menu_zoom_.pan_y = std::clamp(menu_zoom_.pan_y - (delta_my / static_cast<float>(screen_h_)) / z,
                                              half, 1.0F - half);
            }
            pinch_prev_dist_ = dist;
        }
        pinch_prev_mid_x_ = mid_x;
        pinch_prev_mid_y_ = mid_y;
    }

    static void emit_disc(wse::platform::GeometryFrame& f, std::size_t base_vtx,
                          float cx, float cy, float r,
                          float cr, float cg, float cb, float ca, int segs) {
        constexpr float kTwoPi = 6.28318530717958647692F;
        auto push = [&](float x, float y) -> std::int32_t {
            wse::platform::GeometryVertex gv{};
            gv.x = x;
            gv.y = y;
            gv.r = cr;
            gv.g = cg;
            gv.b = cb;
            gv.a = ca;
            const auto idx = static_cast<std::int32_t>(f.vertices.size() - base_vtx);
            f.vertices.push_back(gv);
            return idx;
        };
        const std::int32_t c = push(cx, cy);
        const std::int32_t first = static_cast<std::int32_t>(f.vertices.size() - base_vtx);
        for (int i = 0; i < segs; ++i) {
            const float a = (kTwoPi * static_cast<float>(i)) / static_cast<float>(segs);
            (void)push(cx + std::cos(a) * r, cy + std::sin(a) * r);
        }
        for (int i = 0; i < segs; ++i) {
            f.indices.push_back(c);
            f.indices.push_back(first + i);
            f.indices.push_back(first + ((i + 1) % segs));
        }
    }

    static void emit_ring(wse::platform::GeometryFrame& f, std::size_t base_vtx,
                          float cx, float cy, float r, float thickness,
                          float cr, float cg, float cb, float ca, int segs) {
        constexpr float kTwoPi = 6.28318530717958647692F;
        const float r_in = std::max(1.0F, r - thickness * 0.5F);
        const float r_out = r + thickness * 0.5F;
        auto push = [&](float x, float y) -> std::int32_t {
            wse::platform::GeometryVertex gv{};
            gv.x = x;
            gv.y = y;
            gv.r = cr;
            gv.g = cg;
            gv.b = cb;
            gv.a = ca;
            const auto idx = static_cast<std::int32_t>(f.vertices.size() - base_vtx);
            f.vertices.push_back(gv);
            return idx;
        };
        const std::int32_t first = static_cast<std::int32_t>(f.vertices.size() - base_vtx);
        for (int i = 0; i < segs; ++i) {
            const float a = (kTwoPi * static_cast<float>(i)) / static_cast<float>(segs);
            const float ca_v = std::cos(a);
            const float sa_v = std::sin(a);
            (void)push(cx + ca_v * r_in, cy + sa_v * r_in);
            (void)push(cx + ca_v * r_out, cy + sa_v * r_out);
        }
        for (int i = 0; i < segs; ++i) {
            const int n = (i + 1) % segs;
            const std::int32_t i0 = first + i * 2;
            const std::int32_t o0 = i0 + 1;
            const std::int32_t i1 = first + n * 2;
            const std::int32_t o1 = i1 + 1;
            f.indices.push_back(i0);
            f.indices.push_back(o0);
            f.indices.push_back(o1);
            f.indices.push_back(i0);
            f.indices.push_back(o1);
            f.indices.push_back(i1);
        }
    }

    void init_default_buttons() {
        using Icon = wse::platform::TouchIconId;
        buttons_.clear();
        auto add = [&](TouchButtonSpec b) { buttons_.push_back(std::move(b)); };

        // Global corner buttons (compact, unobstructed)
        add({"pause_drawer", "Menu", TouchScene::battle, Icon::pause_bars, OverlayActionKind::pause_drawer, 0, 0, false, false, 0.045F, 0.095F, 0.095F});
        add({"scene_mode", "Mode", TouchScene::battle, Icon::compass_look, OverlayActionKind::scene_cycle, 0, 0, false, false, 0.105F, 0.095F, 0.085F});
        add({"cheat_menu", "Cheats", TouchScene::battle, Icon::cheat_menu, OverlayActionKind::cheat_menu_toggle, 0, 0, false, false, 0.045F, 0.340F, 0.092F, true, true, false, false, false, false, false, true});
        add({"soft_keyboard", "Keyboard", TouchScene::main_menu, Icon::keyboard, OverlayActionKind::soft_keyboard, 0, 0, false, false, 0.945F, 0.095F, 0.095F});

        // Context-aware Cheats strip (expanded on left margin only when cheats are enabled & toggled open)
        auto add_cheat = [&](std::string id, std::string label, TouchScene sc, Icon ic, OverlayActionKind act,
                             std::uint8_t k, bool ctrl, bool shift, bool alt, float x, float y, bool all_sc = false) {
            TouchButtonSpec cb{};
            cb.id = std::move(id);
            cb.label = std::move(label);
            cb.scene = sc;
            cb.icon = ic;
            cb.action = act;
            cb.dik_code = k;
            cb.need_ctrl = ctrl;
            cb.need_shift = shift;
            cb.need_alt = alt;
            cb.nx = x;
            cb.ny = y;
            cb.size_short = 0.088F;
            cb.visible = true;
            cb.enabled = true;
            cb.is_cheat = true;
            cb.any_scene = all_sc;
            buttons_.push_back(std::move(cb));
        };
        // Universal cheat console: beside Cheats in Battle/Orders, or stacked in left margin on UI screens
        add_cheat("cheat_console_b", "Console", TouchScene::battle, Icon::cheat_console, OverlayActionKind::cheat_chord_pulse, dik::kGrave, true, false, false, 0.105F, 0.340F);
        add_cheat("cheat_console_o", "Console", TouchScene::orders, Icon::cheat_console, OverlayActionKind::cheat_chord_pulse, dik::kGrave, true, false, false, 0.105F, 0.340F);
        add_cheat("cheat_console_m", "Console", TouchScene::campaign_map, Icon::cheat_console, OverlayActionKind::cheat_chord_pulse, dik::kGrave, true, false, false, 0.045F, 0.220F);
        add_cheat("cheat_console_t", "Console", TouchScene::trade, Icon::cheat_console, OverlayActionKind::cheat_chord_pulse, dik::kGrave, true, false, false, 0.045F, 0.580F);
        add_cheat("cheat_console_p", "Console", TouchScene::party, Icon::cheat_console, OverlayActionKind::cheat_chord_pulse, dik::kGrave, true, false, false, 0.045F, 0.580F);
        add_cheat("cheat_console_d", "Console", TouchScene::dialog, Icon::cheat_console, OverlayActionKind::cheat_chord_pulse, dik::kGrave, true, false, false, 0.045F, 0.580F);
        add_cheat("cheat_console_s", "Console", TouchScene::screens, Icon::cheat_console, OverlayActionKind::cheat_chord_pulse, dik::kGrave, true, false, false, 0.045F, 0.580F);
        add_cheat("cheat_console_mm", "Console", TouchScene::main_menu, Icon::cheat_console, OverlayActionKind::cheat_chord_pulse, dik::kGrave, true, false, false, 0.045F, 0.460F);
        // Battle cheats (neat row extending right from Cheats button, above joystick)
        add_cheat("cheat_heal", "Heal", TouchScene::battle, Icon::cheat_heal, OverlayActionKind::cheat_chord_pulse, dik::kH, true, false, false, 0.165F, 0.340F);
        add_cheat("cheat_heal_horse", "Heal Horse", TouchScene::battle, Icon::cheat_heal_horse, OverlayActionKind::cheat_chord_pulse, dik::kH, true, true, false, 0.225F, 0.340F);
        add_cheat("cheat_ko_one", "KO Enemy", TouchScene::battle, Icon::cheat_ko_one, OverlayActionKind::cheat_chord_pulse, dik::kF4, true, false, false, 0.285F, 0.340F);
        add_cheat("cheat_ko_all", "KO All", TouchScene::battle, Icon::cheat_ko_all, OverlayActionKind::cheat_chord_pulse, dik::kF4, true, false, true, 0.345F, 0.340F);
        add_cheat("cheat_slowmo", "Slow-Mo", TouchScene::battle, Icon::cheat_slowmo, OverlayActionKind::cheat_chord_pulse, dik::kF9, true, false, false, 0.405F, 0.340F);
        add_cheat("cheat_ai_ctrl", "AI Control", TouchScene::battle, Icon::cheat_ai_ctrl, OverlayActionKind::cheat_chord_pulse, dik::kF5, true, false, false, 0.465F, 0.340F);
        // Campaign Map cheats (left margin above map travel buttons)
        add_cheat("cheat_teleport", "Teleport", TouchScene::campaign_map, Icon::cheat_teleport, OverlayActionKind::key_toggle, dik::kLCtrl, false, false, false, 0.045F, 0.460F);
        add_cheat("cheat_reveal_map", "Reveal Map", TouchScene::campaign_map, Icon::cheat_reveal_map, OverlayActionKind::cheat_chord_pulse, dik::kT, true, false, false, 0.045F, 0.580F);
        // Trade / Inventory cheats (left 20:9 margin, outside merchant grid)
        add_cheat("cheat_gold", "+1000 Gold", TouchScene::trade, Icon::cheat_gold, OverlayActionKind::cheat_chord_pulse, dik::kX, true, false, false, 0.045F, 0.460F);
        // Party cheats (left 20:9 margin, outside garrison list)
        add_cheat("cheat_party_xp", "+1000 XP", TouchScene::party, Icon::cheat_xp, OverlayActionKind::cheat_chord_pulse, dik::kX, true, false, false, 0.045F, 0.460F);
        // Character / Screens / Dialog cheats (left 20:9 margin)
        add_cheat("cheat_screen_xp", "+1000 XP", TouchScene::screens, Icon::cheat_xp, OverlayActionKind::cheat_chord_pulse, dik::kX, true, false, false, 0.045F, 0.460F);
        add_cheat("cheat_dialog_xp", "+1000 XP", TouchScene::dialog, Icon::cheat_xp, OverlayActionKind::cheat_chord_pulse, dik::kX, true, false, false, 0.045F, 0.460F);

        // Battle controls (clean-room ergonomics designed for 20:9 landscape thumb reach;
        // bottom-right HP/horse/shield HUD and top-center kill-feed stay unobstructed)
        add({"attack", "Attack", TouchScene::battle, Icon::sword, OverlayActionKind::slide_attack, 0, 0, false, false, 0.895F, 0.65F, 0.175F});
        add({"defend", "Defend", TouchScene::battle, Icon::shield, OverlayActionKind::mouse_button, 0, 1, false, false, 0.775F, 0.69F, 0.135F});
        add({"jump", "Jump", TouchScene::battle, Icon::jump_arrow, OverlayActionKind::key_hold, dik::kSpace, 0, false, false, 0.905F, 0.43F, 0.112F});
        add({"kick", "Kick", TouchScene::battle, Icon::boot_kick, OverlayActionKind::key_hold, dik::kE, 0, false, false, 0.815F, 0.49F, 0.102F});
        add({"use", "Use", TouchScene::battle, Icon::hand_use, OverlayActionKind::easy_dismount, dik::kF, 0, false, false, 0.905F, 0.25F, 0.112F});
        add({"zoom", "Zoom", TouchScene::battle, Icon::crosshair, OverlayActionKind::zoom_hold, dik::kLShift, 0, false, false, 0.750F, 0.37F, 0.102F});
        add({"switch_weapon", "Weapon", TouchScene::battle, Icon::switch_weapon, OverlayActionKind::mouse_wheel_step, 0, 0, false, false, 0.680F, 0.77F, 0.098F});
        add({"next_shield", "Shield", TouchScene::battle, Icon::next_shield, OverlayActionKind::mouse_wheel_step, 0, 1, false, false, 0.765F, 0.82F, 0.092F});
        add({"weapon_mode", "Mode", TouchScene::battle, Icon::weapon_mode, OverlayActionKind::key_hold, dik::kX, 0, false, false, 0.690F, 0.59F, 0.094F});
        add({"drop", "Drop", TouchScene::battle, Icon::drop_item, OverlayActionKind::key_hold, dik::kG, 0, false, false, 0.610F, 0.80F, 0.088F});
        add({"crouch", "Crouch", TouchScene::battle, Icon::crouch, OverlayActionKind::key_toggle, dik::kZ, 0, false, false, 0.065F, 0.44F, 0.098F});
        add({"camera", "Camera", TouchScene::battle, Icon::eye_cam, OverlayActionKind::key_hold, dik::kR, 0, false, false, 0.180F, 0.095F, 0.090F});
        add({"orders", "Orders", TouchScene::battle, Icon::banner_orders, OverlayActionKind::orders_open, 0, 0, false, false, 0.255F, 0.095F, 0.095F});
        add({"orders_more", "More", TouchScene::battle, Icon::orders_more, OverlayActionKind::key_hold, dik::kF2, 0, false, false, 0.325F, 0.095F, 0.088F});
        add({"leave", "Leave", TouchScene::battle, Icon::door_exit, OverlayActionKind::key_hold, dik::kTab, 0, false, false, 0.945F, 0.095F, 0.095F});
        add({"chat", "Chat", TouchScene::battle, Icon::chat_bubble, OverlayActionKind::key_hold, dik::kT, 0, false, false, 0.400F, 0.095F, 0.088F});

        // Orders 4-button sub-pad (replaces battle action clutter when issuing troop commands)
        add({"orders_up", "Up", TouchScene::orders, Icon::arrow_up, OverlayActionKind::key_hold, dik::kF1, 0, false, false, 0.84F, 0.42F, 0.112F});
        add({"orders_choose", "Choose", TouchScene::orders, Icon::arrow_right, OverlayActionKind::key_hold, dik::kF3, 0, false, false, 0.92F, 0.58F, 0.112F});
        add({"orders_down", "Down", TouchScene::orders, Icon::arrow_down, OverlayActionKind::key_hold, dik::kF2, 0, false, false, 0.84F, 0.74F, 0.112F});
        add({"orders_back", "Back", TouchScene::orders, Icon::arrow_left, OverlayActionKind::orders_close, 0, 0, false, false, 0.76F, 0.58F, 0.112F});

        // Campaign Map controls (lifted above Warband's bottom toolbar and bottom-right Date/Gold bar)
        add({"stop_rest", "Rest", TouchScene::campaign_map, Icon::hourglass, OverlayActionKind::map_stop_rest, dik::kSpace, 0, false, false, 0.065F, 0.72F, 0.108F});
        add({"fast_travel", "Fast", TouchScene::campaign_map, Icon::fast_forward, OverlayActionKind::key_hold, dik::kSpace, 0, true, false, 0.145F, 0.72F, 0.108F});
        add({"map_camera", "Follow", TouchScene::campaign_map, Icon::map_follow, OverlayActionKind::map_camera_action, 0, 0, false, false, 0.855F, 0.72F, 0.108F});
        add({"map_look", "Look", TouchScene::campaign_map, Icon::compass_look, OverlayActionKind::map_look_toggle, 0, 0, false, false, 0.935F, 0.72F, 0.108F});

        // Party controls (vertical right-margin rail so troop lists and bottom Done/Cancel stay clear)
        add({"party_stack", "Stack", TouchScene::party, Icon::stack_all, OverlayActionKind::key_toggle, dik::kLCtrl, 0, false, false, 0.935F, 0.42F, 0.108F});
        add({"party_ten", "10", TouchScene::party, Icon::ten_badge, OverlayActionKind::key_toggle, dik::kLShift, 0, false, false, 0.935F, 0.58F, 0.108F});
        add({"party_back", "Back", TouchScene::party, Icon::cancel_x, OverlayActionKind::key_hold, dik::kEscape, 0, false, false, 0.935F, 0.74F, 0.104F});

        // Trade / Looting controls (vertical right-margin rail so merchant/loot & player grids stay clear)
        add({"trade_take_all", "Take All", TouchScene::trade, Icon::stack_all, OverlayActionKind::trade_take_all, 0, 0, false, false, 0.935F, 0.42F, 0.108F});
        add({"trade_give_all", "Give All", TouchScene::trade, Icon::give_all, OverlayActionKind::trade_give_all, 0, 0, false, false, 0.935F, 0.58F, 0.108F});
        add({"trade_cancel", "Cancel", TouchScene::trade, Icon::cancel_x, OverlayActionKind::key_hold, dik::kEscape, 0, false, false, 0.935F, 0.74F, 0.104F});

        // Dialog / Conversation controls (right margin outside the dialog box)
        add({"dialog_up", "Up", TouchScene::dialog, Icon::arrow_up, OverlayActionKind::key_hold, dik::kUp, 0, false, false, 0.935F, 0.42F, 0.104F});
        add({"dialog_select", "Select", TouchScene::dialog, Icon::check_select, OverlayActionKind::key_hold, dik::kReturn, 0, false, false, 0.935F, 0.58F, 0.112F});
        add({"dialog_down", "Down", TouchScene::dialog, Icon::arrow_down, OverlayActionKind::key_hold, dik::kDown, 0, false, false, 0.935F, 0.74F, 0.104F});

        // General UI screens (right margin)
        add({"screen_back", "Back", TouchScene::screens, Icon::cancel_x, OverlayActionKind::key_hold, dik::kEscape, 0, false, false, 0.935F, 0.74F, 0.104F});

        // 10 Assignable Extra / Auto-Detected Mod Button Slots (hidden by default until detected or enabled).
        // Placed cleanly below the top bar (left & upper-right arcs) so the top-center kill feed and HUD stay clear.
        extra_start_index_ = buttons_.size();
        for (int i = 0; i < 10; ++i) {
            const float ex = (i < 5) ? (0.045F + static_cast<float>(i) * 0.068F) : (0.560F + static_cast<float>(i - 5) * 0.068F);
            const float ey = (i < 5) ? 0.235F : 0.095F;
            TouchButtonSpec ex_btn{};
            ex_btn.id = "extra" + std::to_string(i + 1);
            ex_btn.label = std::to_string(i + 1);
            ex_btn.scene = TouchScene::battle;
            ex_btn.icon = Icon::star;
            ex_btn.action = OverlayActionKind::key_hold;
            ex_btn.nx = ex;
            ex_btn.ny = ey;
            ex_btn.size_short = 0.090F;
            ex_btn.visible = false;
            ex_btn.enabled = true;
            ex_btn.is_extra = true;
            buttons_.push_back(std::move(ex_btn));
        }
    }

    mutable std::mutex m_;
    std::chrono::steady_clock::time_point start_;
    TouchControlConfig cfg_{};
    wse::platform::SafeAreaInsets safe_area_{};
    int screen_w_ = 1920;
    int screen_h_ = 1080;
    int surface_w_ = 0;
    int surface_h_ = 0;
    int cursor_x_ = 960;
    int cursor_y_ = 540;
    float mouse_rem_x_ = 0.0F;
    float mouse_rem_y_ = 0.0F;
    bool phys_ctrl_ = false;
    bool phys_shift_ = false;
    bool phys_alt_ = false;
    bool pad_ctrl_ = false;
    bool pad_shift_ = false;
    bool cursor_visible_ = true;
    bool manual_scene_override_ = false;
    TouchScene scene_ = TouchScene::main_menu;
    TouchScene inferred_ui_scene_ = TouchScene::main_menu;
    ActiveInputMode input_mode_ = ActiveInputMode::touch;
    BattleContextFlags battle_ctx_{};
    MenuZoomView menu_zoom_{};

    std::array<std::uint8_t, 256> keys_{};
    std::array<std::uint8_t, 8> mouse_buttons_{};
    int mouse_dx_ = 0;
    int mouse_dy_ = 0;
    int mouse_dz_ = 0;
    std::deque<BufferedInputEvent> key_events_;
    std::deque<BufferedInputEvent> mouse_events_;
    std::uint32_t event_seq_ = 0U;

    std::array<XInputPadState, 4> pads_{};
    bool xinput_enabled_ = true;

    std::vector<TouchButtonSpec> buttons_;
    std::size_t extra_start_index_ = 0U;
    std::vector<ActiveTouch> touches_;
    int stick_pointer_ = -1;
    float stick_dx_ = 0.0F;
    float stick_dy_ = 0.0F;
    bool attack_slide_active_ = false;
    float attack_slide_angle_ = 0.0F;
    bool zoom_active_ = false;
    bool map_look_mode_ = false;
    float gyro_lp_x_ = 0.0F;
    float gyro_lp_y_ = 0.0F;

    TapPhase tap_phase_ = TapPhase::idle;
    bool tap_release_queued_ = false;
    int dismount_frames_ = 0;
    std::uint8_t pulse_release_dik_ = 0U;
    bool pulse_release_ctrl_ = false;
    bool pulse_release_shift_ = false;
    bool pulse_release_alt_ = false;
    bool cheats_expanded_ = false;
    float pinch_prev_dist_ = 0.0F;
    float pinch_prev_mid_x_ = 0.0F;
    float pinch_prev_mid_y_ = 0.0F;

    PostWindowMsgFn post_msg_;
    UiActionFn on_ui_action_;
    RumbleFn on_rumble_;
};

}  // namespace wse::recomp::shim
