// Simplistic procedural 2D vector icon library for touch controls and mod hotkeys
// (Phase 5 R6, `docs/RECOMP_PLAN.md` "R6").
//
// Port infrastructure: emits crisp, minimalist geometric shapes (thick strokes, rings,
// triangles, diamonds) directly into a pre-reserved `GeometryFrame` (`texture = 0`) at
// native display resolution with zero texture assets and zero per-frame allocations.
// Covers every core Warband touch control plus 30+ mod-specific and general RPG icons
// (Phantasy Calradia, Warsword Conquest, TLD, PBOD, Viking Conquest, AWOIAF, BannerPage,
// World of Vasnar, 108 Heroes, Five Dynasties, Gekokujo, etc.).

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include "wse/platform/rhi_geometry.h"

namespace wse::platform {

enum class TouchIconId : std::uint8_t {
    none = 0,
    // Core Battle & Camera controls
    sword,           // Attack / Slide-to-aim
    shield,          // Defend / Block
    boot_kick,       // Kick (on foot)
    jump_arrow,      // Jump
    hand_use,        // Use / Interact / Talk / Pick up
    horse_mount,     // Mount horse
    horse_dismount,  // Easy dismount
    crosshair,       // Zoom / Aim
    switch_weapon,   // Next weapon
    next_shield,     // Next shield
    weapon_mode,     // Toggle weapon mode (couch / melee-thrown)
    crouch,          // Crouch
    drop_item,       // Drop weapon
    eye_cam,         // 1st / 3rd person camera toggle
    door_exit,       // Leave area
    pause_bars,      // Pause / Quick menu
    banner_orders,   // Tactical orders
    orders_more,     // Orders page 2
    arrow_up,        // Orders / Dialog Up
    arrow_down,      // Orders / Dialog Down
    arrow_left,      // Orders Back
    arrow_right,     // Orders Choose
    check_select,    // Dialog Select / Confirm
    // Campaign Map, Party, Trade & UI controls
    compass_look,    // Map Look pan/zoom toggle
    fast_forward,    // Fast travel (Ctrl+Space)
    map_home,        // Recenter camera on party
    map_follow,      // Follow party camera
    hourglass,       // Stop / Rest (Space)
    stack_all,       // Move stack / Take all
    give_all,        // Give all
    ten_badge,       // Move 10 (Shift)
    cancel_x,        // Cancel / Close / Back
    keyboard,        // Soft keyboard
    chat_bubble,     // Multiplayer chat
    // Mod-specific simplistic icons (Warband total conversions & OSP kits)
    spell_spark,     // Arcane spells (Phantasy Calradia, Warsword Conquest)
    sun_blessing,    // Divine blessings / Faith (Phantasy Calradia)
    horse_whistle,   // Whistle horse (Viking Conquest, Floris, AWOIAF)
    spear_brace,     // Brace spear / Pike wall (PBOD, Perisno, PoP, BannerPage)
    shield_bash,     // Shield bash (Viking Conquest, AWOIAF, Floris)
    war_horn,        // Warcry / Battle shout / Taunt (VC, TLD, Five Dynasties)
    formation_wall,  // Shieldwall / Formations (PBOD, TLD, VC, Floris)
    sprint_dash,     // Sprint / Dodge roll (Five Dynasties, BannerPage)
    volley_arrows,   // Volley fire / Salvo (PBOD, Gekokujo, Napoleonic)
    helm_anchor,     // Naval / Ship helm (Viking Conquest, AWOIAF)
    tactical_map,    // Battle tactical minimap (Backspace)
    heal_cross,      // Heal / Bandage / Medkit
    flask_potion,    // Drink / Cycle potion (Warsword Conquest, RPG mods)
    skull_necro,     // Necromancy / Dark magic / Vampire (World of Vasnar)
    book_grimoire,   // Spellbook / Grimoire / Magic info (Warsword, Vasnar)
    yin_yang,        // Wuxia / Qi / Hero career skill (108 Heroes, Five Dynasties)
    flame_aura,      // Rage / Focus willpower / Special aura (Warsword, 108 Heroes)
    eye_insight,     // Mind-read / Scout / Inspect (108 Heroes)
    fist_martial,    // Martial / Unarmed / Duel ability (108 Heroes)
    horse_rear,      // Rear horse / Instant stop (BannerPage)
    // General assignable extra icons
    star,
    crown,
    tower,
    torch,
    purse,
    scroll,
    tent,
    key_item,
    heart,
    bell,
    lightning,
    bag,
    target_flag,
    // Cheat menu icons (shown only when cheats are enabled)
    cheat_menu,       // Expandable Cheats menu toggle
    cheat_heal,       // Heal Self (Ctrl+H)
    cheat_heal_horse, // Heal Horse (Ctrl+Shift+H)
    cheat_ko_one,     // Knock out one enemy (Ctrl+F4)
    cheat_ko_all,     // Knock out all enemies (Ctrl+Alt+F4)
    cheat_slowmo,     // Slow motion (Ctrl+F9)
    cheat_ai_ctrl,    // AI takeover (Ctrl+F5)
    cheat_teleport,   // Map teleport (Ctrl+Click)
    cheat_reveal_map, // Reveal whole map (Ctrl+T)
    cheat_gold,       // +1000 Denars (Ctrl+X)
    cheat_xp,         // +1000 XP / Upgrade (Ctrl+X)
    cheat_console,    // Command console (Ctrl+~)
    count_
};

[[nodiscard]] inline std::string_view touch_icon_name(TouchIconId id) noexcept {
    switch (id) {
    case TouchIconId::sword: return "sword";
    case TouchIconId::shield: return "shield";
    case TouchIconId::boot_kick: return "kick";
    case TouchIconId::jump_arrow: return "jump";
    case TouchIconId::hand_use: return "use";
    case TouchIconId::horse_mount: return "mount";
    case TouchIconId::horse_dismount: return "dismount";
    case TouchIconId::crosshair: return "zoom";
    case TouchIconId::switch_weapon: return "weapon";
    case TouchIconId::next_shield: return "next_shield";
    case TouchIconId::weapon_mode: return "weapon_mode";
    case TouchIconId::crouch: return "crouch";
    case TouchIconId::drop_item: return "drop";
    case TouchIconId::eye_cam: return "camera";
    case TouchIconId::door_exit: return "leave";
    case TouchIconId::pause_bars: return "pause";
    case TouchIconId::banner_orders: return "orders";
    case TouchIconId::orders_more: return "orders_more";
    case TouchIconId::arrow_up: return "up";
    case TouchIconId::arrow_down: return "down";
    case TouchIconId::arrow_left: return "left";
    case TouchIconId::arrow_right: return "right";
    case TouchIconId::check_select: return "select";
    case TouchIconId::compass_look: return "map_look";
    case TouchIconId::fast_forward: return "fast_travel";
    case TouchIconId::map_home: return "map_home";
    case TouchIconId::map_follow: return "map_follow";
    case TouchIconId::hourglass: return "rest";
    case TouchIconId::stack_all: return "stack_all";
    case TouchIconId::give_all: return "give_all";
    case TouchIconId::ten_badge: return "ten";
    case TouchIconId::cancel_x: return "cancel";
    case TouchIconId::keyboard: return "keyboard";
    case TouchIconId::chat_bubble: return "chat";
    case TouchIconId::spell_spark: return "spell";
    case TouchIconId::sun_blessing: return "blessing";
    case TouchIconId::horse_whistle: return "whistle";
    case TouchIconId::spear_brace: return "brace";
    case TouchIconId::shield_bash: return "shield_bash";
    case TouchIconId::war_horn: return "warcry";
    case TouchIconId::formation_wall: return "formation";
    case TouchIconId::sprint_dash: return "sprint";
    case TouchIconId::volley_arrows: return "volley";
    case TouchIconId::helm_anchor: return "naval";
    case TouchIconId::tactical_map: return "tactical_map";
    case TouchIconId::heal_cross: return "heal";
    case TouchIconId::flask_potion: return "potion";
    case TouchIconId::skull_necro: return "necro";
    case TouchIconId::book_grimoire: return "grimoire";
    case TouchIconId::yin_yang: return "hero_skill";
    case TouchIconId::flame_aura: return "aura";
    case TouchIconId::eye_insight: return "insight";
    case TouchIconId::fist_martial: return "martial";
    case TouchIconId::horse_rear: return "rear_horse";
    case TouchIconId::star: return "star";
    case TouchIconId::crown: return "crown";
    case TouchIconId::tower: return "tower";
    case TouchIconId::torch: return "torch";
    case TouchIconId::purse: return "purse";
    case TouchIconId::scroll: return "scroll";
    case TouchIconId::tent: return "tent";
    case TouchIconId::key_item: return "key";
    case TouchIconId::heart: return "heart";
    case TouchIconId::bell: return "bell";
    case TouchIconId::lightning: return "lightning";
    case TouchIconId::bag: return "bag";
    case TouchIconId::target_flag: return "flag";
    case TouchIconId::cheat_menu: return "cheats";
    case TouchIconId::cheat_heal: return "heal_self";
    case TouchIconId::cheat_heal_horse: return "heal_horse";
    case TouchIconId::cheat_ko_one: return "ko_enemy";
    case TouchIconId::cheat_ko_all: return "ko_all";
    case TouchIconId::cheat_slowmo: return "slowmo";
    case TouchIconId::cheat_ai_ctrl: return "ai_ctrl";
    case TouchIconId::cheat_teleport: return "teleport";
    case TouchIconId::cheat_reveal_map: return "reveal_map";
    case TouchIconId::cheat_gold: return "gold";
    case TouchIconId::cheat_xp: return "xp";
    case TouchIconId::cheat_console: return "console";
    case TouchIconId::none:
    case TouchIconId::count_:
    default: return "none";
    }
}

namespace touch_icon_detail {

inline constexpr float kIconPi = 3.14159265358979323846F;

struct IconPen {
    GeometryFrame& frame;
    std::size_t base_vertex = 0U;
    float cx = 0.0F;
    float cy = 0.0F;
    float r = 16.0F;
    float stroke = 2.4F;
    float cr = 1.0F;
    float cg = 1.0F;
    float cb = 1.0F;
    float ca = 1.0F;

    [[nodiscard]] std::int32_t vtx(float px, float py) {
        GeometryVertex g{};
        g.x = px;
        g.y = py;
        g.r = cr;
        g.g = cg;
        g.b = cb;
        g.a = ca;
        const auto idx = static_cast<std::int32_t>(frame.vertices.size() - base_vertex);
        frame.vertices.push_back(g);
        return idx;
    }

    void tri_px(float x0, float y0, float x1, float y1, float x2, float y2) {
        const std::int32_t a = vtx(x0, y0);
        const std::int32_t b = vtx(x1, y1);
        const std::int32_t c = vtx(x2, y2);
        frame.indices.push_back(a);
        frame.indices.push_back(b);
        frame.indices.push_back(c);
    }

    // Normalized [-1, 1] coordinates (y+ downward, matching screen space).
    void tri(float x0, float y0, float x1, float y1, float x2, float y2) {
        tri_px(cx + x0 * r, cy + y0 * r, cx + x1 * r, cy + y1 * r, cx + x2 * r, cy + y2 * r);
    }

    void rect(float x0, float y0, float x1, float y1) {
        const std::int32_t i0 = vtx(cx + x0 * r, cy + y0 * r);
        const std::int32_t i1 = vtx(cx + x1 * r, cy + y0 * r);
        const std::int32_t i2 = vtx(cx + x1 * r, cy + y1 * r);
        const std::int32_t i3 = vtx(cx + x0 * r, cy + y1 * r);
        frame.indices.push_back(i0);
        frame.indices.push_back(i1);
        frame.indices.push_back(i2);
        frame.indices.push_back(i0);
        frame.indices.push_back(i2);
        frame.indices.push_back(i3);
    }

    void line(float x0, float y0, float x1, float y1, float width_scale = 1.0F) {
        const float px0 = cx + x0 * r;
        const float py0 = cy + y0 * r;
        const float px1 = cx + x1 * r;
        const float py1 = cy + y1 * r;
        const float dx = px1 - px0;
        const float dy = py1 - py0;
        const float len = std::sqrt(dx * dx + dy * dy);
        if (len <= 1e-4F) return;
        const float half = 0.5F * stroke * width_scale;
        const float nx = (-dy / len) * half;
        const float ny = (dx / len) * half;
        const std::int32_t i0 = vtx(px0 + nx, py0 + ny);
        const std::int32_t i1 = vtx(px1 + nx, py1 + ny);
        const std::int32_t i2 = vtx(px1 - nx, py1 - ny);
        const std::int32_t i3 = vtx(px0 - nx, py0 - ny);
        frame.indices.push_back(i0);
        frame.indices.push_back(i1);
        frame.indices.push_back(i2);
        frame.indices.push_back(i0);
        frame.indices.push_back(i2);
        frame.indices.push_back(i3);
    }

    void dot(float nx, float ny, float rad_norm, int segs = 12) {
        const float px = cx + nx * r;
        const float py = cy + ny * r;
        const float pr = rad_norm * r;
        const std::int32_t center = vtx(px, py);
        const std::int32_t first = static_cast<std::int32_t>(frame.vertices.size() - base_vertex);
        for (int i = 0; i < segs; ++i) {
            const float a = (2.0F * kIconPi * static_cast<float>(i)) / static_cast<float>(segs);
            (void)vtx(px + std::cos(a) * pr, py + std::sin(a) * pr);
        }
        for (int i = 0; i < segs; ++i) {
            frame.indices.push_back(center);
            frame.indices.push_back(first + i);
            frame.indices.push_back(first + ((i + 1) % segs));
        }
    }

    void ring(float nx, float ny, float rad_norm, float width_scale = 1.0F, int segs = 16) {
        const float px = cx + nx * r;
        const float py = cy + ny * r;
        const float half = 0.5F * stroke * width_scale;
        const float r_out = rad_norm * r + half;
        const float r_in = std::max(0.5F, rad_norm * r - half);
        const std::int32_t first = static_cast<std::int32_t>(frame.vertices.size() - base_vertex);
        for (int i = 0; i < segs; ++i) {
            const float a = (2.0F * kIconPi * static_cast<float>(i)) / static_cast<float>(segs);
            const float ca_v = std::cos(a);
            const float sa_v = std::sin(a);
            (void)vtx(px + ca_v * r_in, py + sa_v * r_in);
            (void)vtx(px + ca_v * r_out, py + sa_v * r_out);
        }
        for (int i = 0; i < segs; ++i) {
            const int n = (i + 1) % segs;
            const std::int32_t i0 = first + i * 2;
            const std::int32_t o0 = i0 + 1;
            const std::int32_t i1 = first + n * 2;
            const std::int32_t o1 = i1 + 1;
            frame.indices.push_back(i0);
            frame.indices.push_back(o0);
            frame.indices.push_back(o1);
            frame.indices.push_back(i0);
            frame.indices.push_back(o1);
            frame.indices.push_back(i1);
        }
    }

    void shield_outline(float scale = 1.0F) {
        const float s = scale;
        line(-0.58F * s, -0.65F * s, 0.58F * s, -0.65F * s);
        line(-0.58F * s, -0.65F * s, -0.52F * s, 0.12F * s);
        line(0.58F * s, -0.65F * s, 0.52F * s, 0.12F * s);
        line(-0.52F * s, 0.12F * s, 0.0F, 0.78F * s);
        line(0.52F * s, 0.12F * s, 0.0F, 0.78F * s);
    }

    void chevron(float cx_n, float cy_n, float dir_x, float dir_y, float size_n) {
        const float px = -dir_y;
        const float py = dir_x;
        const float tip_x = cx_n + dir_x * size_n;
        const float tip_y = cy_n + dir_y * size_n;
        const float bx = cx_n - dir_x * size_n * 0.45F;
        const float by = cy_n - dir_y * size_n * 0.45F;
        line(bx + px * size_n * 0.85F, by + py * size_n * 0.85F, tip_x, tip_y);
        line(bx - px * size_n * 0.85F, by - py * size_n * 0.85F, tip_x, tip_y);
    }
};

}  // namespace touch_icon_detail

// Emits a simplistic vector icon centered at `(cx, cy)` inside radius `icon_radius` into an
// open `GeometryFrame` batch (caller creates the `GeometryDrawCall` around vertices/indices).
inline void emit_touch_icon(GeometryFrame& frame,
                            std::size_t base_vertex,
                            TouchIconId icon,
                            float cx,
                            float cy,
                            float icon_radius,
                            float cr,
                            float cg,
                            float cb,
                            float ca) {
    if (icon == TouchIconId::none || icon_radius <= 1.0F || ca <= 0.01F) {
        return;
    }
    const float stroke = std::clamp(icon_radius * 0.14F, 1.8F, 5.2F);
    touch_icon_detail::IconPen p{frame, base_vertex, cx, cy, icon_radius, stroke, cr, cg, cb, ca};

    switch (icon) {
    case TouchIconId::sword:
        // Diagonal blade + crossguard + pommel.
        p.line(-0.55F, 0.55F, 0.62F, -0.62F, 1.25F);
        p.tri(0.48F, -0.68F, 0.74F, -0.74F, 0.68F, -0.48F);
        p.line(-0.42F, 0.08F, -0.08F, 0.42F, 1.15F);
        p.dot(-0.62F, 0.62F, 0.12F);
        break;
    case TouchIconId::shield:
        p.shield_outline(1.0F);
        p.line(0.0F, -0.45F, 0.0F, 0.50F, 0.8F);
        break;
    case TouchIconId::boot_kick:
        // Simple boot silhouette + forward strike lines.
        p.line(-0.50F, -0.55F, -0.22F, 0.25F, 1.25F);
        p.line(-0.22F, 0.25F, 0.42F, 0.32F, 1.35F);
        p.line(0.45F, -0.15F, 0.78F, -0.25F, 0.85F);
        p.line(0.50F, 0.10F, 0.82F, 0.10F, 0.85F);
        break;
    case TouchIconId::jump_arrow:
        p.line(0.0F, 0.58F, 0.0F, -0.58F, 1.15F);
        p.chevron(0.0F, -0.35F, 0.0F, -1.0F, 0.42F);
        p.line(-0.48F, 0.68F, 0.48F, 0.68F, 0.9F);
        break;
    case TouchIconId::hand_use:
        // Open palm ring + 4 short fingers (minimalist interact glyph).
        p.ring(0.0F, 0.22F, 0.36F, 1.0F, 14);
        p.line(-0.30F, -0.12F, -0.34F, -0.52F, 0.95F);
        p.line(-0.10F, -0.18F, -0.10F, -0.68F, 0.95F);
        p.line(0.12F, -0.18F, 0.14F, -0.65F, 0.95F);
        p.line(0.32F, -0.06F, 0.40F, -0.46F, 0.95F);
        break;
    case TouchIconId::horse_mount:
        // Stirrup / saddle arch + upward arrow.
        p.line(-0.55F, 0.45F, 0.55F, 0.45F, 1.1F);
        p.line(-0.45F, 0.45F, -0.35F, -0.05F, 1.0F);
        p.line(0.45F, 0.45F, 0.35F, -0.05F, 1.0F);
        p.line(0.0F, 0.15F, 0.0F, -0.62F, 1.05F);
        p.chevron(0.0F, -0.45F, 0.0F, -1.0F, 0.34F);
        break;
    case TouchIconId::horse_dismount:
        // Saddle line + downward arrow.
        p.line(-0.55F, -0.48F, 0.55F, -0.48F, 1.1F);
        p.line(0.0F, -0.22F, 0.0F, 0.58F, 1.05F);
        p.chevron(0.0F, 0.40F, 0.0F, 1.0F, 0.36F);
        break;
    case TouchIconId::crosshair:
        p.ring(0.0F, 0.0F, 0.56F, 0.95F, 18);
        p.line(0.0F, -0.82F, 0.0F, -0.34F, 0.9F);
        p.line(0.0F, 0.34F, 0.0F, 0.82F, 0.9F);
        p.line(-0.82F, 0.0F, -0.34F, 0.0F, 0.9F);
        p.line(0.34F, 0.0F, 0.82F, 0.0F, 0.9F);
        p.dot(0.0F, 0.0F, 0.10F);
        break;
    case TouchIconId::switch_weapon:
        // Two opposite curved/diagonal cycle arrows around a short blade.
        p.line(-0.45F, -0.42F, 0.45F, -0.42F, 1.0F);
        p.chevron(0.35F, -0.42F, 1.0F, 0.0F, 0.25F);
        p.line(0.45F, 0.42F, -0.45F, 0.42F, 1.0F);
        p.chevron(-0.35F, 0.42F, -1.0F, 0.0F, 0.25F);
        p.line(-0.22F, 0.18F, 0.22F, -0.18F, 0.95F);
        break;
    case TouchIconId::next_shield:
        p.shield_outline(0.75F);
        p.chevron(0.55F, 0.0F, 1.0F, 0.0F, 0.28F);
        break;
    case TouchIconId::weapon_mode:
        // Double-ended spear/toggle glyph.
        p.line(-0.60F, 0.60F, 0.60F, -0.60F, 1.1F);
        p.chevron(0.45F, -0.45F, 0.707F, -0.707F, 0.30F);
        p.chevron(-0.45F, 0.45F, -0.707F, 0.707F, 0.30F);
        break;
    case TouchIconId::crouch:
        // Downward knee/body silhouette.
        p.dot(0.0F, -0.48F, 0.18F);
        p.line(0.0F, -0.22F, -0.28F, 0.22F, 1.15F);
        p.line(-0.28F, 0.22F, 0.38F, 0.22F, 1.15F);
        p.line(0.38F, 0.22F, 0.38F, 0.62F, 1.15F);
        p.line(-0.55F, 0.68F, 0.55F, 0.68F, 0.85F);
        break;
    case TouchIconId::drop_item:
        p.line(0.0F, -0.62F, 0.0F, 0.26F, 1.05F);
        p.chevron(0.0F, 0.14F, 0.0F, 1.0F, 0.34F);
        p.line(-0.56F, 0.58F, 0.56F, 0.58F, 1.05F);
        p.line(-0.56F, 0.36F, -0.56F, 0.58F, 1.0F);
        p.line(0.56F, 0.36F, 0.56F, 0.58F, 1.0F);
        break;
    case TouchIconId::eye_cam:
        p.line(-0.72F, 0.0F, 0.0F, -0.46F, 1.0F);
        p.line(0.0F, -0.46F, 0.72F, 0.0F, 1.0F);
        p.line(-0.72F, 0.0F, 0.0F, 0.46F, 1.0F);
        p.line(0.0F, 0.46F, 0.72F, 0.0F, 1.0F);
        p.dot(0.0F, 0.0F, 0.22F);
        break;
    case TouchIconId::door_exit:
        p.line(-0.48F, -0.62F, 0.18F, -0.62F, 1.0F);
        p.line(-0.48F, -0.62F, -0.48F, 0.62F, 1.0F);
        p.line(-0.48F, 0.62F, 0.18F, 0.62F, 1.0F);
        p.line(-0.08F, 0.0F, 0.68F, 0.0F, 1.1F);
        p.chevron(0.52F, 0.0F, 1.0F, 0.0F, 0.32F);
        break;
    case TouchIconId::pause_bars:
        p.rect(-0.38F, -0.52F, -0.12F, 0.52F);
        p.rect(0.12F, -0.52F, 0.38F, 0.52F);
        break;
    case TouchIconId::banner_orders:
        p.line(-0.45F, -0.68F, -0.45F, 0.70F, 1.1F);
        p.line(-0.45F, -0.60F, 0.55F, -0.32F, 1.05F);
        p.line(0.55F, -0.32F, -0.45F, 0.05F, 1.05F);
        break;
    case TouchIconId::orders_more:
        p.dot(-0.48F, 0.0F, 0.15F);
        p.dot(0.0F, 0.0F, 0.15F);
        p.dot(0.48F, 0.0F, 0.15F);
        break;
    case TouchIconId::arrow_up:
        p.chevron(0.0F, -0.08F, 0.0F, -1.0F, 0.52F);
        break;
    case TouchIconId::arrow_down:
        p.chevron(0.0F, 0.08F, 0.0F, 1.0F, 0.52F);
        break;
    case TouchIconId::arrow_left:
        p.chevron(-0.08F, 0.0F, -1.0F, 0.0F, 0.52F);
        break;
    case TouchIconId::arrow_right:
        p.chevron(0.08F, 0.0F, 1.0F, 0.0F, 0.52F);
        break;
    case TouchIconId::check_select:
        p.line(-0.52F, 0.06F, -0.14F, 0.48F, 1.2F);
        p.line(-0.14F, 0.48F, 0.60F, -0.45F, 1.2F);
        break;
    case TouchIconId::compass_look:
        p.ring(0.0F, 0.0F, 0.62F, 0.95F, 18);
        p.tri(0.0F, -0.48F, -0.18F, 0.04F, 0.18F, 0.04F);
        p.line(-0.14F, 0.06F, 0.0F, 0.46F, 0.85F);
        p.line(0.14F, 0.06F, 0.0F, 0.46F, 0.85F);
        break;
    case TouchIconId::fast_forward:
        p.tri(-0.62F, -0.45F, 0.04F, 0.0F, -0.62F, 0.45F);
        p.tri(0.02F, -0.45F, 0.68F, 0.0F, 0.02F, 0.45F);
        break;
    case TouchIconId::map_home:
        p.line(-0.60F, 0.02F, 0.0F, -0.58F, 1.1F);
        p.line(0.0F, -0.58F, 0.60F, 0.02F, 1.1F);
        p.line(-0.42F, 0.02F, -0.42F, 0.58F, 1.0F);
        p.line(0.42F, 0.02F, 0.42F, 0.58F, 1.0F);
        p.line(-0.42F, 0.58F, 0.42F, 0.58F, 1.0F);
        break;
    case TouchIconId::map_follow:
        p.ring(0.0F, 0.0F, 0.45F, 1.0F, 16);
        p.dot(0.0F, 0.0F, 0.16F);
        p.chevron(0.0F, -0.68F, 0.0F, -1.0F, 0.22F);
        break;
    case TouchIconId::hourglass:
        p.line(-0.46F, -0.60F, 0.46F, -0.60F, 1.1F);
        p.line(-0.46F, 0.60F, 0.46F, 0.60F, 1.1F);
        p.line(-0.38F, -0.60F, 0.38F, 0.60F, 0.95F);
        p.line(0.38F, -0.60F, -0.38F, 0.60F, 0.95F);
        break;
    case TouchIconId::stack_all:
        p.line(-0.55F, -0.38F, 0.55F, -0.38F, 1.05F);
        p.line(-0.55F, 0.0F, 0.55F, 0.0F, 1.05F);
        p.line(-0.55F, 0.38F, 0.55F, 0.38F, 1.05F);
        p.chevron(0.45F, 0.0F, 1.0F, 0.0F, 0.28F);
        break;
    case TouchIconId::give_all:
        p.line(-0.55F, -0.38F, 0.55F, -0.38F, 1.05F);
        p.line(-0.55F, 0.0F, 0.55F, 0.0F, 1.05F);
        p.line(-0.55F, 0.38F, 0.55F, 0.38F, 1.05F);
        p.chevron(-0.45F, 0.0F, -1.0F, 0.0F, 0.28F);
        break;
    case TouchIconId::ten_badge:
        // Crisp minimalist "10" glyph.
        p.line(-0.42F, -0.45F, -0.42F, 0.48F, 1.15F);
        p.line(-0.56F, -0.28F, -0.42F, -0.45F, 0.95F);
        p.ring(0.24F, 0.0F, 0.38F, 1.05F, 14);
        break;
    case TouchIconId::cancel_x:
        p.line(-0.48F, -0.48F, 0.48F, 0.48F, 1.2F);
        p.line(0.48F, -0.48F, -0.48F, 0.48F, 1.2F);
        break;
    case TouchIconId::keyboard:
        p.line(-0.68F, -0.38F, 0.68F, -0.38F, 0.95F);
        p.line(-0.68F, 0.40F, 0.68F, 0.40F, 0.95F);
        p.line(-0.68F, -0.38F, -0.68F, 0.40F, 0.95F);
        p.line(0.68F, -0.38F, 0.68F, 0.40F, 0.95F);
        p.dot(-0.38F, -0.10F, 0.08F, 8);
        p.dot(0.0F, -0.10F, 0.08F, 8);
        p.dot(0.38F, -0.10F, 0.08F, 8);
        p.line(-0.34F, 0.18F, 0.34F, 0.18F, 0.9F);
        break;
    case TouchIconId::chat_bubble:
        p.line(-0.60F, -0.45F, 0.60F, -0.45F, 0.95F);
        p.line(0.60F, -0.45F, 0.60F, 0.28F, 0.95F);
        p.line(0.60F, 0.28F, -0.10F, 0.28F, 0.95F);
        p.line(-0.10F, 0.28F, -0.48F, 0.60F, 0.95F);
        p.line(-0.48F, 0.60F, -0.40F, 0.28F, 0.95F);
        p.line(-0.40F, 0.28F, -0.60F, 0.28F, 0.95F);
        p.line(-0.60F, 0.28F, -0.60F, -0.45F, 0.95F);
        break;
    // ---- Mod-specific simplistic icons ----
    case TouchIconId::spell_spark:
        // 4-point arcane star + inner diamond.
        p.tri(0.0F, -0.78F, -0.16F, 0.0F, 0.16F, 0.0F);
        p.tri(0.0F, 0.78F, -0.16F, 0.0F, 0.16F, 0.0F);
        p.tri(-0.78F, 0.0F, 0.0F, -0.16F, 0.0F, 0.16F);
        p.tri(0.78F, 0.0F, 0.0F, -0.16F, 0.0F, 0.16F);
        break;
    case TouchIconId::sun_blessing:
        // Radiant sun / faith glyph.
        p.ring(0.0F, 0.0F, 0.32F, 1.0F, 14);
        for (int i = 0; i < 8; ++i) {
            const float a = static_cast<float>(i) * (touch_icon_detail::kIconPi * 0.25F);
            p.line(std::cos(a) * 0.48F, std::sin(a) * 0.48F, std::cos(a) * 0.76F, std::sin(a) * 0.76F, 0.9F);
        }
        break;
    case TouchIconId::horse_whistle:
        // Minimalist horse-head silhouette + two sound waves.
        p.line(-0.45F, 0.55F, -0.25F, -0.35F, 1.25F);
        p.line(-0.25F, -0.35F, 0.15F, -0.10F, 1.2F);
        p.line(-0.25F, -0.35F, -0.35F, -0.62F, 1.0F);
        p.line(0.32F, -0.38F, 0.56F, -0.52F, 0.85F);
        p.line(0.38F, -0.08F, 0.68F, -0.10F, 0.85F);
        p.line(0.34F, 0.20F, 0.58F, 0.30F, 0.85F);
        break;
    case TouchIconId::spear_brace:
        // Angled braced polearm anchored on the ground.
        p.line(-0.55F, 0.58F, 0.58F, -0.55F, 1.15F);
        p.tri(0.38F, -0.62F, 0.72F, -0.72F, 0.62F, -0.38F);
        p.line(-0.70F, 0.62F, 0.18F, 0.62F, 1.0F);
        break;
    case TouchIconId::shield_bash:
        p.shield_outline(0.72F);
        p.line(0.46F, -0.30F, 0.76F, -0.45F, 0.95F);
        p.line(0.50F, 0.0F, 0.82F, 0.0F, 0.95F);
        p.line(0.46F, 0.30F, 0.76F, 0.45F, 0.95F);
        break;
    case TouchIconId::war_horn:
        // Curved war horn + shout rays.
        p.line(-0.62F, 0.32F, 0.15F, -0.15F, 1.0F);
        p.tri(0.05F, -0.18F, 0.58F, -0.58F, 0.58F, 0.22F);
        p.line(-0.62F, 0.32F, -0.40F, 0.54F, 1.0F);
        break;
    case TouchIconId::formation_wall:
        // Three interlocking shields side-by-side.
        p.rect(-0.68F, -0.38F, -0.28F, 0.42F);
        p.rect(-0.18F, -0.48F, 0.18F, 0.48F);
        p.rect(0.28F, -0.38F, 0.68F, 0.42F);
        break;
    case TouchIconId::sprint_dash:
        // Forward double-chevron + speed streaks.
        p.chevron(-0.15F, 0.0F, 1.0F, 0.0F, 0.42F);
        p.chevron(0.32F, 0.0F, 1.0F, 0.0F, 0.42F);
        p.line(-0.72F, -0.28F, -0.38F, -0.28F, 0.9F);
        p.line(-0.72F, 0.28F, -0.38F, 0.28F, 0.9F);
        break;
    case TouchIconId::volley_arrows:
        // Three parallel diagonal arrows.
        for (float off : {-0.34F, 0.0F, 0.34F}) {
            p.line(-0.48F + off * 0.5F, 0.48F + off * 0.5F, 0.48F + off * 0.5F, -0.48F + off * 0.5F, 0.85F);
            p.chevron(0.36F + off * 0.5F, -0.36F + off * 0.5F, 0.707F, -0.707F, 0.18F);
        }
        break;
    case TouchIconId::helm_anchor:
        // Ship helm / anchor cross.
        p.ring(0.0F, -0.46F, 0.16F, 0.9F, 10);
        p.line(0.0F, -0.28F, 0.0F, 0.62F, 1.1F);
        p.line(-0.40F, -0.08F, 0.40F, -0.08F, 1.0F);
        p.line(-0.52F, 0.28F, 0.0F, 0.65F, 1.05F);
        p.line(0.52F, 0.28F, 0.0F, 0.65F, 1.05F);
        break;
    case TouchIconId::tactical_map:
        // Folded 3-panel map.
        p.line(-0.62F, -0.46F, -0.20F, -0.58F, 0.95F);
        p.line(-0.20F, -0.58F, 0.22F, -0.42F, 0.95F);
        p.line(0.22F, -0.42F, 0.62F, -0.56F, 0.95F);
        p.line(-0.62F, 0.54F, -0.20F, 0.42F, 0.95F);
        p.line(-0.20F, 0.42F, 0.22F, 0.58F, 0.95F);
        p.line(0.22F, 0.58F, 0.62F, 0.44F, 0.95F);
        p.line(-0.62F, -0.46F, -0.62F, 0.54F, 0.95F);
        p.line(-0.20F, -0.58F, -0.20F, 0.42F, 0.8F);
        p.line(0.22F, -0.42F, 0.22F, 0.58F, 0.8F);
        p.line(0.62F, -0.56F, 0.62F, 0.44F, 0.95F);
        break;
    case TouchIconId::heal_cross:
        p.rect(-0.18F, -0.60F, 0.18F, 0.60F);
        p.rect(-0.60F, -0.18F, 0.60F, 0.18F);
        break;
    case TouchIconId::flask_potion:
        p.line(-0.20F, -0.64F, 0.20F, -0.64F, 1.05F);
        p.line(-0.14F, -0.64F, -0.14F, -0.26F, 0.95F);
        p.line(0.14F, -0.64F, 0.14F, -0.26F, 0.95F);
        p.line(-0.14F, -0.26F, -0.52F, 0.42F, 1.0F);
        p.line(0.14F, -0.26F, 0.52F, 0.42F, 1.0F);
        p.line(-0.52F, 0.42F, -0.38F, 0.62F, 1.0F);
        p.line(0.52F, 0.42F, 0.38F, 0.62F, 1.0F);
        p.line(-0.38F, 0.62F, 0.38F, 0.62F, 1.0F);
        break;
    case TouchIconId::skull_necro:
        p.ring(0.0F, -0.14F, 0.46F, 1.0F, 16);
        p.dot(-0.18F, -0.16F, 0.11F, 8);
        p.dot(0.18F, -0.16F, 0.11F, 8);
        p.line(-0.22F, 0.34F, -0.22F, 0.60F, 0.95F);
        p.line(0.0F, 0.34F, 0.0F, 0.60F, 0.95F);
        p.line(0.22F, 0.34F, 0.22F, 0.60F, 0.95F);
        break;
    case TouchIconId::book_grimoire:
        p.line(-0.58F, -0.52F, 0.0F, -0.38F, 1.0F);
        p.line(0.58F, -0.52F, 0.0F, -0.38F, 1.0F);
        p.line(-0.58F, 0.48F, 0.0F, 0.62F, 1.0F);
        p.line(0.58F, 0.48F, 0.0F, 0.62F, 1.0F);
        p.line(-0.58F, -0.52F, -0.58F, 0.48F, 1.0F);
        p.line(0.58F, -0.52F, 0.58F, 0.48F, 1.0F);
        p.line(0.0F, -0.38F, 0.0F, 0.62F, 0.9F);
        break;
    case TouchIconId::yin_yang:
        p.ring(0.0F, 0.0F, 0.62F, 1.0F, 18);
        p.line(0.0F, -0.60F, 0.16F, -0.18F, 0.9F);
        p.line(0.16F, -0.18F, -0.16F, 0.18F, 0.9F);
        p.line(-0.16F, 0.18F, 0.0F, 0.60F, 0.9F);
        p.dot(0.0F, -0.30F, 0.10F, 8);
        p.dot(0.0F, 0.30F, 0.10F, 8);
        break;
    case TouchIconId::flame_aura:
        p.tri(0.0F, -0.74F, -0.46F, 0.22F, 0.46F, 0.22F);
        p.tri(-0.46F, 0.22F, 0.46F, 0.22F, 0.0F, 0.68F);
        break;
    case TouchIconId::eye_insight:
        p.line(-0.70F, 0.0F, 0.0F, -0.44F, 0.95F);
        p.line(0.0F, -0.44F, 0.70F, 0.0F, 0.95F);
        p.line(-0.70F, 0.0F, 0.0F, 0.44F, 0.95F);
        p.line(0.0F, 0.44F, 0.70F, 0.0F, 0.95F);
        p.ring(0.0F, 0.0F, 0.20F, 0.9F, 12);
        p.line(0.0F, -0.74F, 0.0F, -0.54F, 0.85F);
        break;
    case TouchIconId::fist_martial:
        p.rect(-0.46F, -0.40F, 0.46F, 0.28F);
        p.rect(-0.28F, 0.28F, 0.28F, 0.62F);
        break;
    case TouchIconId::horse_rear:
        p.line(-0.45F, 0.52F, 0.15F, -0.48F, 1.2F);
        p.line(0.15F, -0.48F, 0.55F, -0.22F, 1.1F);
        p.line(-0.62F, 0.62F, 0.62F, 0.62F, 1.0F);
        break;
    // ---- General extra icons ----
    case TouchIconId::star:
        for (int i = 0; i < 5; ++i) {
            const float a0 = -touch_icon_detail::kIconPi * 0.5F + static_cast<float>(i) * (2.0F * touch_icon_detail::kIconPi / 5.0F);
            const float a1 = a0 + (2.0F * touch_icon_detail::kIconPi / 5.0F) * 2.0F;
            p.line(std::cos(a0) * 0.68F, std::sin(a0) * 0.68F, std::cos(a1) * 0.68F, std::sin(a1) * 0.68F, 0.95F);
        }
        break;
    case TouchIconId::crown:
        p.line(-0.60F, 0.44F, 0.60F, 0.44F, 1.1F);
        p.line(-0.60F, 0.44F, -0.60F, -0.34F, 1.0F);
        p.line(0.60F, 0.44F, 0.60F, -0.34F, 1.0F);
        p.line(-0.60F, -0.34F, -0.22F, 0.04F, 1.0F);
        p.line(-0.22F, 0.04F, 0.0F, -0.50F, 1.0F);
        p.line(0.0F, -0.50F, 0.22F, 0.04F, 1.0F);
        p.line(0.22F, 0.04F, 0.60F, -0.34F, 1.0F);
        break;
    case TouchIconId::tower:
        p.line(-0.38F, -0.56F, -0.38F, 0.62F, 1.05F);
        p.line(0.38F, -0.56F, 0.38F, 0.62F, 1.05F);
        p.line(-0.46F, 0.62F, 0.46F, 0.62F, 1.1F);
        p.line(-0.44F, -0.56F, 0.44F, -0.56F, 1.05F);
        p.rect(-0.14F, 0.16F, 0.14F, 0.62F);
        break;
    case TouchIconId::torch:
        p.line(0.0F, -0.10F, 0.0F, 0.68F, 1.15F);
        p.tri(0.0F, -0.72F, -0.28F, -0.14F, 0.28F, -0.14F);
        break;
    case TouchIconId::purse:
        p.ring(0.0F, 0.20F, 0.42F, 1.05F, 14);
        p.line(-0.24F, -0.26F, 0.24F, -0.26F, 1.05F);
        p.line(-0.18F, -0.26F, -0.28F, -0.54F, 0.95F);
        p.line(0.18F, -0.26F, 0.28F, -0.54F, 0.95F);
        break;
    case TouchIconId::scroll:
        p.line(-0.44F, -0.56F, 0.44F, -0.56F, 1.05F);
        p.line(-0.44F, 0.56F, 0.44F, 0.56F, 1.05F);
        p.line(-0.44F, -0.56F, -0.44F, 0.56F, 0.95F);
        p.line(0.44F, -0.56F, 0.44F, 0.56F, 0.95F);
        p.line(-0.24F, -0.20F, 0.24F, -0.20F, 0.85F);
        p.line(-0.24F, 0.14F, 0.24F, 0.14F, 0.85F);
        break;
    case TouchIconId::tent:
        p.line(-0.64F, 0.54F, 0.0F, -0.56F, 1.1F);
        p.line(0.0F, -0.56F, 0.64F, 0.54F, 1.1F);
        p.line(-0.64F, 0.54F, 0.64F, 0.54F, 1.05F);
        p.line(0.0F, -0.02F, 0.0F, 0.54F, 0.95F);
        break;
    case TouchIconId::key_item:
        p.ring(-0.34F, 0.0F, 0.28F, 1.0F, 12);
        p.line(-0.04F, 0.0F, 0.64F, 0.0F, 1.05F);
        p.line(0.38F, 0.0F, 0.38F, 0.26F, 1.0F);
        p.line(0.58F, 0.0F, 0.58F, 0.26F, 1.0F);
        break;
    case TouchIconId::heart:
        p.tri(-0.56F, -0.10F, 0.56F, -0.10F, 0.0F, 0.64F);
        p.dot(-0.28F, -0.20F, 0.30F, 12);
        p.dot(0.28F, -0.20F, 0.30F, 12);
        break;
    case TouchIconId::bell:
        p.line(-0.50F, 0.34F, 0.50F, 0.34F, 1.1F);
        p.line(-0.42F, 0.34F, -0.26F, -0.44F, 1.0F);
        p.line(0.42F, 0.34F, 0.26F, -0.44F, 1.0F);
        p.line(-0.26F, -0.44F, 0.26F, -0.44F, 1.0F);
        p.dot(0.0F, 0.56F, 0.12F, 8);
        break;
    case TouchIconId::lightning:
        p.line(0.18F, -0.68F, -0.28F, 0.04F, 1.15F);
        p.line(-0.28F, 0.04F, 0.22F, 0.04F, 1.15F);
        p.line(0.22F, 0.04F, -0.18F, 0.68F, 1.15F);
        break;
    case TouchIconId::bag:
        p.line(-0.46F, -0.24F, 0.46F, -0.24F, 1.0F);
        p.line(-0.52F, 0.56F, 0.52F, 0.56F, 1.0F);
        p.line(-0.46F, -0.24F, -0.52F, 0.56F, 1.0F);
        p.line(0.46F, -0.24F, 0.52F, 0.56F, 1.0F);
        p.line(-0.24F, -0.24F, -0.24F, -0.52F, 0.95F);
        p.line(0.24F, -0.24F, 0.24F, -0.52F, 0.95F);
        p.line(-0.24F, -0.52F, 0.24F, -0.52F, 0.95F);
        break;
    case TouchIconId::target_flag:
        p.line(-0.34F, -0.64F, -0.34F, 0.64F, 1.1F);
        p.tri(-0.34F, -0.60F, 0.56F, -0.28F, -0.34F, 0.04F);
        p.line(-0.56F, 0.64F, 0.08F, 0.64F, 0.95F);
        break;
    // ---- Cheat menu icons ----
    case TouchIconId::cheat_menu:
        p.line(-0.56F, 0.44F, 0.56F, 0.44F, 1.05F);
        p.line(-0.56F, 0.44F, -0.48F, -0.22F, 1.0F);
        p.line(0.56F, 0.44F, 0.48F, -0.22F, 1.0F);
        p.line(-0.48F, -0.22F, -0.18F, 0.08F, 1.0F);
        p.line(-0.18F, 0.08F, 0.0F, -0.44F, 1.0F);
        p.line(0.0F, -0.44F, 0.18F, 0.08F, 1.0F);
        p.line(0.18F, 0.08F, 0.48F, -0.22F, 1.0F);
        p.dot(0.0F, -0.66F, 0.10F, 8);
        break;
    case TouchIconId::cheat_heal:
        p.ring(0.0F, 0.0F, 0.62F, 0.9F, 18);
        p.rect(-0.14F, -0.38F, 0.14F, 0.38F);
        p.rect(-0.38F, -0.14F, 0.38F, 0.14F);
        break;
    case TouchIconId::cheat_heal_horse:
        p.line(-0.54F, 0.50F, -0.34F, -0.34F, 1.2F);
        p.line(-0.34F, -0.34F, 0.02F, -0.12F, 1.15F);
        p.line(-0.34F, -0.34F, -0.42F, -0.60F, 0.95F);
        p.rect(0.28F, -0.34F, 0.46F, 0.26F);
        p.rect(0.08F, -0.14F, 0.66F, 0.06F);
        break;
    case TouchIconId::cheat_ko_one:
        p.ring(0.0F, 0.0F, 0.58F, 0.9F, 16);
        p.line(0.14F, -0.42F, -0.16F, 0.04F, 1.05F);
        p.line(-0.16F, 0.04F, 0.14F, 0.04F, 1.05F);
        p.line(0.14F, 0.04F, -0.14F, 0.44F, 1.05F);
        break;
    case TouchIconId::cheat_ko_all:
        p.line(-0.14F, -0.64F, -0.46F, 0.02F, 1.05F);
        p.line(-0.46F, 0.02F, -0.14F, 0.02F, 1.05F);
        p.line(-0.14F, 0.02F, -0.44F, 0.64F, 1.05F);
        p.line(0.46F, -0.64F, 0.14F, 0.02F, 1.05F);
        p.line(0.14F, 0.02F, 0.46F, 0.02F, 1.05F);
        p.line(0.46F, 0.02F, 0.16F, 0.64F, 1.05F);
        break;
    case TouchIconId::cheat_slowmo:
        p.ring(0.0F, 0.0F, 0.62F, 0.95F, 18);
        p.line(0.0F, 0.0F, 0.0F, -0.42F, 1.05F);
        p.line(0.0F, 0.0F, 0.34F, 0.22F, 1.05F);
        p.dot(0.0F, 0.0F, 0.10F, 8);
        break;
    case TouchIconId::cheat_ai_ctrl:
        p.line(-0.48F, -0.34F, 0.48F, -0.34F, 1.0F);
        p.line(-0.48F, 0.44F, 0.48F, 0.44F, 1.0F);
        p.line(-0.48F, -0.34F, -0.48F, 0.44F, 1.0F);
        p.line(0.48F, -0.34F, 0.48F, 0.44F, 1.0F);
        p.line(0.0F, -0.34F, 0.0F, -0.62F, 0.95F);
        p.dot(0.0F, -0.66F, 0.10F, 8);
        p.dot(-0.20F, 0.04F, 0.10F, 8);
        p.dot(0.20F, 0.04F, 0.10F, 8);
        break;
    case TouchIconId::cheat_teleport:
        p.ring(0.0F, -0.22F, 0.30F, 1.0F, 14);
        p.tri(-0.26F, -0.02F, 0.26F, -0.02F, 0.0F, 0.48F);
        p.line(-0.56F, 0.60F, 0.56F, 0.60F, 0.95F);
        break;
    case TouchIconId::cheat_reveal_map:
        p.line(-0.62F, 0.0F, 0.0F, -0.36F, 0.95F);
        p.line(0.0F, -0.36F, 0.62F, 0.0F, 0.95F);
        p.line(-0.62F, 0.0F, 0.0F, 0.36F, 0.95F);
        p.line(0.0F, 0.36F, 0.62F, 0.0F, 0.95F);
        p.dot(0.0F, 0.0F, 0.16F, 10);
        p.line(0.0F, -0.72F, 0.0F, -0.50F, 0.9F);
        p.line(0.0F, 0.50F, 0.0F, 0.72F, 0.9F);
        break;
    case TouchIconId::cheat_gold:
        p.ring(-0.14F, 0.06F, 0.46F, 1.0F, 16);
        p.line(-0.14F, -0.18F, -0.14F, 0.30F, 0.95F);
        p.rect(0.42F, -0.62F, 0.56F, -0.22F);
        p.rect(0.29F, -0.49F, 0.69F, -0.35F);
        break;
    case TouchIconId::cheat_xp:
        p.chevron(-0.14F, -0.24F, 0.0F, -1.0F, 0.34F);
        p.chevron(-0.14F, 0.24F, 0.0F, -1.0F, 0.34F);
        p.rect(0.44F, -0.46F, 0.58F, -0.06F);
        p.rect(0.31F, -0.33F, 0.71F, -0.19F);
        break;
    case TouchIconId::cheat_console:
        p.chevron(-0.22F, 0.0F, 1.0F, 0.0F, 0.36F);
        p.line(0.10F, 0.34F, 0.62F, 0.34F, 1.15F);
        break;
    case TouchIconId::none:
    case TouchIconId::count_:
    default:
        break;
    }
}

}  // namespace wse::platform
