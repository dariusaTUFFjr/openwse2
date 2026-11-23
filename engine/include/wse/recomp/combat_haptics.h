// Sound-driven combat haptics classifier for Phase 5 R6 (`docs/RECOMP_PLAN.md` "R6").
//
// Port infrastructure: the recompiled Warband exe raises no haptic events of its own, so the
// FMOD shim inspects sounds played near the player's listener position and maps combat sound
// names (shield blocks, melee/ranged hits, couched lances, heavy impacts) to calibrated
// vibration pulses. Clean-room, header-only.

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <string>
#include <string_view>

namespace wse::recomp::shim {

enum class CombatHapticKind : std::uint8_t {
    none = 0,
    shield_block = 1,
    melee_hit = 2,
    player_hit = 3,
    heavy_blow = 4,
};

struct CombatHapticsSettings {
    bool enabled = true;
    bool when_hit = true;       // "When you are hit" (default ON)
    bool when_dealing = true;   // "When you land a hit" (default ON)
    int strength_percent = 60;  // 10..100, default 60%
    float max_distance = 3.8F;  // meters from player's 3D audio listener
    std::uint32_t cooldown_ms = 45U;
};

struct CombatHapticPulse {
    CombatHapticKind kind = CombatHapticKind::none;
    std::uint32_t duration_ms = 0U;
    std::uint8_t amplitude = 0U;  // Android VibrationEffect amplitude: 1..255
};

class CombatHapticsClassifier {
public:
    explicit CombatHapticsClassifier(CombatHapticsSettings settings = {}) : settings_(sanitize(settings)) {}

    void set_settings(CombatHapticsSettings s) noexcept { settings_ = sanitize(s); }
    [[nodiscard]] const CombatHapticsSettings& settings() const noexcept { return settings_; }

    // Extracts the lowercase filename stem (without path or extension) and classifies combat audio.
    [[nodiscard]] static CombatHapticKind classify_sound_name(std::string_view path) noexcept {
        std::size_t slash = path.find_last_of("/\\");
        std::string_view base = (slash == std::string_view::npos) ? path : path.substr(slash + 1U);
        std::size_t dot = base.find_last_of('.');
        if (dot != std::string_view::npos) {
            base = base.substr(0U, dot);
        }
        if (base.empty()) {
            return CombatHapticKind::none;
        }
        std::string lower;
        lower.reserve(base.size());
        for (char c : base) {
            if (c >= 'A' && c <= 'Z') {
                lower.push_back(static_cast<char>(c - 'A' + 'a'));
            } else {
                lower.push_back(c);
            }
        }
        const auto has = [&](std::string_view needle) noexcept {
            return lower.find(needle) != std::string::npos;
        };

        // Ignore ambience, footsteps, music, menus, and distant arrows whizzing by.
        if (has("music") || has("footstep") || has("gallop") || has("trot") || has("canter") ||
            has("click") || has("button") || has("ambient") || has("rain") || has("wind") ||
            has("pass_by") || has("whiz") || has("draw") || has("release") || has("sheath")) {
            return CombatHapticKind::none;
        }

        // Shield blocks and weapon parries.
        if (has("shield_hit") || has("shield_block") || has("metal_block") || has("wood_block") ||
            has("parry") || has("blocked") || has("clank") || has("shield_broken")) {
            return CombatHapticKind::shield_block;
        }

        // Heavy impacts: couched lances, blunt knockdowns, horse collisions, siege impacts.
        if (has("couch") || has("lance_hit") || has("blunt_hit") || has("horse_hit") ||
            has("horse_fall") || has("body_fall") || has("siege_hit") || has("explosion") ||
            has("battering")) {
            return CombatHapticKind::heavy_blow;
        }

        // Player/agent hurt vocalizations or direct hits taken.
        if (has("man_hit") || has("woman_hit") || has("player_hit") || has("man_die") ||
            has("woman_die") || has("man_grunt") || has("woman_grunt")) {
            return CombatHapticKind::player_hit;
        }

        // Weapon/projectile hits dealt.
        if (has("metal_hit") || has("flesh_hit") || has("cut_hit") || has("pierce_hit") ||
            has("sword_hit") || has("weapon_hit") || has("arrow_hit") || has("bolt_hit") ||
            has("jav_hit") || has("stone_hit") || has("hit_body") || has("hit_flesh") ||
            has("hit_metal") || has("hit_wood")) {
            return CombatHapticKind::melee_hit;
        }

        return CombatHapticKind::none;
    }

    // Evaluates whether a sound playback should emit a haptic pulse.
    // `is_3d`: false for 2D sounds (treated as distance 0 if combat-named), true for spatial sounds.
    // `now_ms`: monotonic timestamp in milliseconds for cooldown throttling.
    [[nodiscard]] CombatHapticPulse evaluate(std::string_view sound_name,
                                             bool is_3d,
                                             float distance_meters,
                                             float channel_volume,
                                             std::uint64_t now_ms) noexcept {
        if (!settings_.enabled || settings_.strength_percent <= 0) {
            return {};
        }
        if (is_3d && (!std::isfinite(distance_meters) || distance_meters > settings_.max_distance)) {
            return {};
        }
        const CombatHapticKind kind = classify_sound_name(sound_name);
        if (kind == CombatHapticKind::none) {
            return {};
        }
        if (kind == CombatHapticKind::player_hit && !settings_.when_hit) {
            return {};
        }
        if (kind == CombatHapticKind::melee_hit && !settings_.when_dealing) {
            return {};
        }
        if ((kind == CombatHapticKind::shield_block || kind == CombatHapticKind::heavy_blow) &&
            !settings_.when_hit && !settings_.when_dealing) {
            return {};
        }
        if (has_last_pulse_ && now_ms >= last_pulse_ms_ &&
            (now_ms - last_pulse_ms_) < static_cast<std::uint64_t>(settings_.cooldown_ms)) {
            return {};
        }

        std::uint32_t base_ms = 24U;
        float base_gain = 0.65F;
        switch (kind) {
        case CombatHapticKind::shield_block:
            base_ms = 18U;
            base_gain = 0.58F;
            break;
        case CombatHapticKind::melee_hit:
            base_ms = 25U;
            base_gain = 0.68F;
            break;
        case CombatHapticKind::player_hit:
            base_ms = 34U;
            base_gain = 0.85F;
            break;
        case CombatHapticKind::heavy_blow:
            base_ms = 45U;
            base_gain = 1.00F;
            break;
        case CombatHapticKind::none:
        default:
            return {};
        }

        const float dist_scale =
            is_3d ? std::clamp(1.0F - (std::max(0.0F, distance_meters) / std::max(0.1F, settings_.max_distance)) * 0.65F,
                               0.35F, 1.0F)
                  : 1.0F;
        const float vol_scale = std::clamp(std::isfinite(channel_volume) ? channel_volume : 1.0F, 0.25F, 1.0F);
        const float user_scale = static_cast<float>(settings_.strength_percent) / 100.0F;
        const float total = std::clamp(base_gain * dist_scale * vol_scale * user_scale, 0.08F, 1.0F);
        const auto amp = static_cast<std::uint8_t>(std::clamp<int>(static_cast<int>(std::lround(total * 255.0F)), 1, 255));

        has_last_pulse_ = true;
        last_pulse_ms_ = now_ms;
        return CombatHapticPulse{kind, base_ms, amp};
    }

    void reset_cooldown() noexcept {
        has_last_pulse_ = false;
        last_pulse_ms_ = 0U;
    }

private:
    static CombatHapticsSettings sanitize(CombatHapticsSettings s) noexcept {
        s.strength_percent = std::clamp(s.strength_percent, 0, 100);
        if (!std::isfinite(s.max_distance) || s.max_distance < 0.5F) {
            s.max_distance = 3.8F;
        }
        return s;
    }

    CombatHapticsSettings settings_{};
    bool has_last_pulse_ = false;
    std::uint64_t last_pulse_ms_ = 0U;
};

}  // namespace wse::recomp::shim
