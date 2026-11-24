// Automatic Warband Mod Controls Detector (Phase 5 R6, `docs/RECOMP_PLAN.md` "R6").
//
// Port infrastructure: inspects a user-supplied Warband module folder (`module.ini`,
// `mission_templates.txt`, `scripts.txt`, `strings.txt`, `quick_strings.txt`,
// `variables.txt`, `presentations.txt`) through `wse::platform::Vfs` and automatically
// activates matching extra touch buttons (with simplistic procedural icons and plain-English
// labels) whenever a mod uses crouch, spell casting, blessings, potions, horse whistling,
// spear bracing, shield bash, warcries, formations, sprint/dodge, volley fire, naval helms,
// hero skills, or necromancy/grimoire triggers. Header-only, zero retail assets.

#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <string_view>
#include <vector>

#include "wse/platform/touch_icons.h"
#include "wse/platform/vfs.h"

namespace wse::platform {

struct DetectedModButton {
    std::string id;
    std::string label;  // Plain player words (e.g. "Spells", "Whistle", "Brace")
    TouchIconId icon = TouchIconId::star;
    std::uint8_t dik_code = 0U;  // DirectInput scan code (DIK_*)
    bool need_ctrl = false;
    bool need_shift = false;
    bool toggle_mode = false;
    std::string reason;  // Plain-English explanation shown in Launcher / Controls settings
};

struct DetectedModControls {
    std::string module_name;
    bool crouch_supported = false;
    bool advanced_formations = false;
    std::vector<DetectedModButton> buttons;

    [[nodiscard]] std::string summary_text() const {
        if (buttons.empty() && !crouch_supported) {
            return {};
        }
        std::string out;
        if (crouch_supported) {
            out += "Crouch";
        }
        for (const auto& b : buttons) {
            if (!out.empty()) out += ", ";
            out += b.label;
        }
        return out;
    }
};

namespace mod_detect_detail {

// Standard Warband / DirectInput scan codes (matching `wse/platform/input.h`).
inline constexpr std::uint8_t kDikBackspace = 0x0EU;
inline constexpr std::uint8_t kDikQ = 0x10U;
inline constexpr std::uint8_t kDikT = 0x14U;
inline constexpr std::uint8_t kDikU = 0x16U;
inline constexpr std::uint8_t kDikO = 0x18U;
inline constexpr std::uint8_t kDikH = 0x23U;
inline constexpr std::uint8_t kDikJ = 0x24U;
inline constexpr std::uint8_t kDikK = 0x25U;
inline constexpr std::uint8_t kDikL = 0x26U;
inline constexpr std::uint8_t kDikSemicolon = 0x27U;
inline constexpr std::uint8_t kDikZ = 0x2CU;
inline constexpr std::uint8_t kDikC = 0x2EU;
inline constexpr std::uint8_t kDikV = 0x2FU;
inline constexpr std::uint8_t kDikB = 0x30U;
inline constexpr std::uint8_t kDikN = 0x31U;
inline constexpr std::uint8_t kDikM = 0x32U;
inline constexpr std::uint8_t kDikF4 = 0x3EU;
inline constexpr std::uint8_t kDikEnd = 0xCFU;

inline std::string to_lower_ascii(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c >= 'A' && c <= 'Z') {
            out.push_back(static_cast<char>(c - 'A' + 'a'));
        } else {
            out.push_back(c);
        }
    }
    return out;
}

inline bool parse_ini_bool(std::string_view ini_lower, std::string_view key, bool& out_val) {
    std::size_t pos = 0U;
    while (pos < ini_lower.size()) {
        std::size_t end = ini_lower.find('\n', pos);
        if (end == std::string_view::npos) end = ini_lower.size();
        std::string_view line = ini_lower.substr(pos, end - pos);
        pos = (end < ini_lower.size()) ? end + 1U : end;

        // Strip comments (#) and leading whitespace.
        const std::size_t hash = line.find('#');
        if (hash != std::string_view::npos) line = line.substr(0U, hash);
        while (!line.empty() && (line.front() == ' ' || line.front() == '\t' || line.front() == '\r')) {
            line.remove_prefix(1U);
        }
        const std::size_t eq = line.find('=');
        if (eq == std::string_view::npos) continue;
        std::string_view lhs = line.substr(0U, eq);
        while (!lhs.empty() && (lhs.back() == ' ' || lhs.back() == '\t')) lhs.remove_suffix(1U);
        if (lhs != key) continue;
        std::string_view rhs = line.substr(eq + 1U);
        while (!rhs.empty() && (rhs.front() == ' ' || rhs.front() == '\t')) rhs.remove_prefix(1U);
        if (!rhs.empty() && (rhs.front() == '1' || rhs.rfind("true", 0U) == 0U)) {
            out_val = true;
            return true;
        }
        if (!rhs.empty() && (rhs.front() == '0' || rhs.rfind("false", 0U) == 0U)) {
            out_val = false;
            return true;
        }
    }
    return false;
}

// Scans compiled Warband `mission_templates.txt` statements for opcode 70 (`key_is_down`) and
// opcode 71 (`key_clicked`) followed by operand count `1` and an immediate DIK scan code `1..255`.
// Format in compiled txt: `71 1 48` (opcode 71 = key_clicked, 1 arg, 48 = 0x30 = DIK_B).
inline void scan_mission_template_keys(std::string_view text, std::array<bool, 256>& keys_used) {
    const std::size_t n = text.size();
    std::size_t i = 0U;
    std::int64_t prev2 = -1;
    std::int64_t prev1 = -1;
    while (i < n) {
        while (i < n && (text[i] == ' ' || text[i] == '\t' || text[i] == '\r' || text[i] == '\n')) {
            ++i;
        }
        if (i >= n) break;
        const std::size_t start = i;
        while (i < n && text[i] != ' ' && text[i] != '\t' && text[i] != '\r' && text[i] != '\n') {
            ++i;
        }
        const std::string_view tok = text.substr(start, i - start);
        bool all_digits = !tok.empty();
        std::uint64_t val = 0U;
        bool overflow = false;
        for (char c : tok) {
            if (c < '0' || c > '9') {
                all_digits = false;
                break;
            }
            if (val <= 1800000000000000000ULL) {
                val = val * 10ULL + static_cast<std::uint64_t>(c - '0');
            } else {
                overflow = true;
            }
        }
        if (!all_digits) {
            prev2 = -1;
            prev1 = -1;
            continue;
        }
        // Check for `(70|71) 1 <key_code>` or negated `(0x80000000 | (70|71)) 1 <key_code>`.
        // Ignore 64-bit global-variable / register IDs (>= 256) and unrelated opcodes.
        const std::uint64_t op_u = (prev2 >= 0) ? (static_cast<std::uint64_t>(prev2) & ~0x80000000ULL) : 0ULL;
        if (!overflow && (op_u == 70ULL || op_u == 71ULL) && prev1 == 1 && val > 0U && val < 256U) {
            keys_used[static_cast<std::size_t>(val)] = true;
        }
        prev2 = prev1;
        prev1 = overflow ? -1 : static_cast<std::int64_t>(val);
    }
}

inline bool contains_any(std::string_view haystack, std::initializer_list<std::string_view> needles) {
    for (std::string_view needle : needles) {
        if (haystack.find(needle) != std::string_view::npos) {
            return true;
        }
    }
    return false;
}

}  // namespace mod_detect_detail

// Detects mod-specific touch controls from a module's directory inside `vfs` (for example
// `module_dir = "Modules/Phantasy Calradia"`). Safe to call in the launcher or at game start.
[[nodiscard]] inline DetectedModControls detect_mod_controls(Vfs& vfs,
                                                             std::string_view module_dir,
                                                             std::string_view module_name = {}) {
    DetectedModControls out{};
    out.module_name = std::string(module_name);
    std::string base(module_dir);
    while (!base.empty() && (base.back() == '/' || base.back() == '\\')) {
        base.pop_back();
    }
    const auto join = [&](std::string_view file) {
        return base.empty() ? std::string(file) : (base + "/" + std::string(file));
    };

    std::string ini_text;
    if (vfs.read_text(join("module.ini"), ini_text)) {
        const std::string ini_low = mod_detect_detail::to_lower_ascii(ini_text);
        bool flag = false;
        if (mod_detect_detail::parse_ini_bool(ini_low, "can_crouch", flag) && flag) {
            out.crouch_supported = true;
        }
        if (mod_detect_detail::parse_ini_bool(ini_low, "use_advanced_formation", flag) && flag) {
            out.advanced_formations = true;
        }
    }

    std::array<bool, 256> mt_keys{};
    std::string mt_text;
    if (vfs.read_text(join("mission_templates.txt"), mt_text)) {
        mod_detect_detail::scan_mission_template_keys(mt_text, mt_keys);
    }

    // Build a combined lowercase corpus of mod identifiers and UI strings (scripts/strings/vars/presentations).
    std::string corpus = mod_detect_detail::to_lower_ascii(module_name);
    corpus.push_back('\n');
    for (std::string_view fname : {"strings.txt", "quick_strings.txt", "variables.txt", "presentations.txt"}) {
        std::string txt;
        if (vfs.read_text(join(fname), txt)) {
            corpus += mod_detect_detail::to_lower_ascii(txt);
            corpus.push_back('\n');
        }
    }
    // Also inspect script header names (skip body numbers to keep memory/time minimal).
    {
        std::string scripts_txt;
        if (vfs.read_text(join("scripts.txt"), scripts_txt)) {
            std::size_t pos = 0U;
            while (pos < scripts_txt.size()) {
                std::size_t end = scripts_txt.find('\n', pos);
                if (end == std::string_view::npos) end = scripts_txt.size();
                const std::string_view line = std::string_view(scripts_txt).substr(pos, end - pos);
                pos = (end < scripts_txt.size()) ? end + 1U : end;
                if (!line.empty() && std::isalpha(static_cast<unsigned char>(line.front()))) {
                    corpus += mod_detect_detail::to_lower_ascii(line);
                    corpus.push_back('\n');
                }
            }
        }
    }

    const auto add_unique = [&](DetectedModButton btn) {
        if (out.buttons.size() >= 10U) return;  // fits within the 10 assignable extra button slots
        for (const auto& existing : out.buttons) {
            if (existing.id == btn.id) return;
        }
        out.buttons.push_back(std::move(btn));
    };

    // Many popular mods (PBOD, Phantasy, Warsword, Viking Conquest) store configurable hotkeys
    // in global variables (`71 1 144115188075856xxx`), so immediate DIK literals are optional
    // when a module's scripts/strings/presentations already confirm the mechanic.
    const bool has_mt_keys = std::any_of(mt_keys.begin(), mt_keys.end(), [](bool v) { return v; });
    const auto key_active = [&](std::uint8_t dik, bool confirmed_by_corpus = true) noexcept {
        return confirmed_by_corpus || !has_mt_keys || mt_keys[dik];
    };

    // 1. Arcane Magic / Spells (Phantasy Calradia, Warsword Conquest, etc.)
    if (mod_detect_detail::contains_any(corpus, {"spell_list", "cast_spell", "spell_book", "magic_power",
                                                 "phantasy", "warsword", "arcane_magic"})) {
        const bool is_warsword = mod_detect_detail::contains_any(corpus, {"warsword", "focus_willpower", "winds_of_magic"});
        if (key_active(mod_detect_detail::kDikB)) {
            add_unique(DetectedModButton{
                "mod_spells",
                is_warsword ? "Next Spell" : "Spells",
                TouchIconId::spell_spark,
                mod_detect_detail::kDikB,
                false,
                false,
                false,
                "Magic spells detected in module",
            });
        }
        if (mod_detect_detail::contains_any(corpus, {"blessing", "divine_magic", "cleric", "faith_point", "phantasy"}) &&
            key_active(mod_detect_detail::kDikN)) {
            add_unique(DetectedModButton{
                "mod_blessings",
                "Blessings",
                TouchIconId::sun_blessing,
                mod_detect_detail::kDikN,
                false,
                false,
                false,
                "Divine blessings detected in module",
            });
        }
        if (is_warsword && key_active(mod_detect_detail::kDikM)) {
            add_unique(DetectedModButton{
                "mod_focus_will",
                "Focus Will",
                TouchIconId::flame_aura,
                mod_detect_detail::kDikM,
                false,
                false,
                false,
                "Focus Willpower detected in Warsword",
            });
        }
        if (is_warsword && key_active(mod_detect_detail::kDikN)) {
            add_unique(DetectedModButton{
                "mod_magic_info",
                "Grimoire",
                TouchIconId::book_grimoire,
                mod_detect_detail::kDikN,
                false,
                false,
                false,
                "Magic Info grimoire detected in Warsword",
            });
        }
    }

    // 2. Necromancy / Vampiric / Dark Grimoire (World of Vasnar & gothic fantasy mods)
    if (mod_detect_detail::contains_any(corpus, {"vasnar", "necromancy", "vampir", "lich_"})) {
        if (key_active(mod_detect_detail::kDikV)) {
            add_unique(DetectedModButton{
                "mod_necro",
                "Dark Art",
                TouchIconId::skull_necro,
                mod_detect_detail::kDikV,
                false,
                false,
                false,
                "Dark magic / vampiric abilities detected in module",
            });
        }
    }

    // 3. Potions & Consumables (Warsword Conquest, RPG overhauls)
    if (mod_detect_detail::contains_any(corpus, {"drink_potion", "health_potion", "healing_potion", "warsword"}) &&
        key_active(mod_detect_detail::kDikH)) {
        add_unique(DetectedModButton{
            "mod_potion",
            "Potion",
            TouchIconId::flask_potion,
            mod_detect_detail::kDikH,
            false,
            false,
            false,
            "Battle potions detected in module",
        });
    }

    // 4. Whistle Horse / Call Mount (Viking Conquest, Floris, AWOIAF, Diplomacy, BannerPage)
    if (mod_detect_detail::contains_any(corpus, {"whistle_for_horse", "horse_whistle", "call_horse",
                                                 "whistle_horse", "viking_conquest"}) &&
        key_active(mod_detect_detail::kDikH)) {
        add_unique(DetectedModButton{
            "mod_whistle",
            "Whistle",
            TouchIconId::horse_whistle,
            mod_detect_detail::kDikH,
            false,
            false,
            false,
            "Horse whistle detected in module",
        });
    }

    // 5. Spear Bracing / Pike Wall (PBOD, Perisno, Floris, AWOIAF, BannerPage, PoP)
    if (mod_detect_detail::contains_any(corpus, {"spear_bracing", "brace_spear", "pike_wall", "bracing_spear"})) {
        const bool uses_b = mt_keys[mod_detect_detail::kDikB];
        add_unique(DetectedModButton{
            "mod_brace",
            "Brace",
            TouchIconId::spear_brace,
            mod_detect_detail::kDikB,
            !uses_b,  // PBOD uses Ctrl+B or B depending on mod
            false,
            true,
            "Spear bracing detected in module",
        });
    }

    // 6. Shield Bash (Viking Conquest, Brytenwalda, Floris, AWOIAF, Perisno)
    if (mod_detect_detail::contains_any(corpus, {"shield_bash", "bash_shield"})) {
        add_unique(DetectedModButton{
            "mod_shield_bash",
            "Bash",
            TouchIconId::shield_bash,
            mod_detect_detail::kDikQ,
            false,
            false,
            false,
            "Shield bash detected in module",
        });
    }

    // 7. Warcry / Battle Shout / Taunt / Horn (VC, TLD, Perisno, Five Dynasties, 108 Heroes)
    if (mod_detect_detail::contains_any(corpus, {"warcry", "war_cry", "battle_cry", "taunt_enemy",
                                                 "battle_shout", "blow_horn"})) {
        const std::uint8_t k = mt_keys[mod_detect_detail::kDikT] ? mod_detect_detail::kDikT
                             : (mt_keys[mod_detect_detail::kDikU] ? mod_detect_detail::kDikU : mod_detect_detail::kDikV);
        add_unique(DetectedModButton{
            "mod_warcry",
            "Warcry",
            TouchIconId::war_horn,
            k,
            false,
            false,
            false,
            "Warcry / battle shout detected in module",
        });
    }

    // 8. Hero Skills / Wuxia / Qi / Special Ability (108 Heroes, Five Dynasties, RPG mods)
    if (mod_detect_detail::contains_any(corpus, {"108_heroes", "108heroes", "five_dynasties", "fivedynasties",
                                                 "hero_skill", "special_ability", "mind_read", "qi_skill"})) {
        const std::uint8_t k = mt_keys[mod_detect_detail::kDikV] ? mod_detect_detail::kDikV
                             : (mt_keys[mod_detect_detail::kDikT] ? mod_detect_detail::kDikT : mod_detect_detail::kDikQ);
        add_unique(DetectedModButton{
            "mod_hero_skill",
            "Hero Skill",
            TouchIconId::yin_yang,
            k,
            false,
            false,
            false,
            "Hero special skill detected in module",
        });
    }

    // 9. Formations (Shieldwall / Ranks / Wedge / Square — TLD, PBOD, VC, Floris, Five Dynasties)
    if (out.advanced_formations ||
        mod_detect_detail::contains_any(corpus, {"form_shieldwall", "shield_wall", "form_ranks",
                                                 "motomataru", "pbod_formations", "formation_ranks"})) {
        const bool has_k = mt_keys[mod_detect_detail::kDikK];
        const bool has_j = mt_keys[mod_detect_detail::kDikJ];
        if (has_k || !has_mt_keys) {
            add_unique(DetectedModButton{
                "mod_shieldwall",
                "Shieldwall",
                TouchIconId::formation_wall,
                has_k ? mod_detect_detail::kDikK : mod_detect_detail::kDikF4,
                false,
                false,
                false,
                "Shieldwall / formation orders detected in module",
            });
        }
        if (has_j && mt_keys[mod_detect_detail::kDikL]) {
            add_unique(DetectedModButton{
                "mod_ranks",
                "Ranks",
                TouchIconId::target_flag,
                mod_detect_detail::kDikJ,
                false,
                false,
                false,
                "Ranks formation detected in module",
            });
        }
        if (mt_keys[mod_detect_detail::kDikO]) {
            add_unique(DetectedModButton{
                "mod_weapon_order",
                "Wep Order",
                TouchIconId::switch_weapon,
                mod_detect_detail::kDikO,
                false,
                false,
                false,
                "Weapon-type formation orders detected in module",
            });
        }
    }

    // 10. Sprint / Dodge Roll (Five Dynasties, BannerPage, Stamina mods)
    if (mod_detect_detail::contains_any(corpus, {"dodge_roll", "sprint_stamina", "stamina_sprint", "combat_roll"})) {
        add_unique(DetectedModButton{
            "mod_sprint",
            "Sprint",
            TouchIconId::sprint_dash,
            mod_detect_detail::kDikC,
            false,
            false,
            false,
            "Sprint / dodge mechanic detected in module",
        });
    }

    // 11. Volley Fire (PBOD, Gekokujo, Napoleonic / Firearms mods)
    if (mod_detect_detail::contains_any(corpus, {"volley_fire", "fire_volley", "salvo_fire", "gekokujo"})) {
        add_unique(DetectedModButton{
            "mod_volley",
            "Volley",
            TouchIconId::volley_arrows,
            mod_detect_detail::kDikF4,
            false,
            false,
            false,
            "Volley fire order detected in module",
        });
    }

    // 12. Naval / Ship Battles (Viking Conquest, AWOIAF)
    if (mod_detect_detail::contains_any(corpus, {"sea_battle", "naval_battle", "ship_control", "viking_conquest"})) {
        add_unique(DetectedModButton{
            "mod_naval",
            "Helm",
            TouchIconId::helm_anchor,
            mod_detect_detail::kDikF4,
            false,
            false,
            false,
            "Naval battle controls detected in module",
        });
    }

    // 13. Rear Horse / Instant Stop (BannerPage)
    if (mod_detect_detail::contains_any(corpus, {"bannerpage", "rear_horse"})) {
        add_unique(DetectedModButton{
            "mod_rear_horse",
            "Rear",
            TouchIconId::horse_rear,
            mod_detect_detail::kDikJ,
            true,
            false,
            false,
            "Instant horse stop / rear detected in module",
        });
    }

    // 14. Crouch fallback from script corpus even if module.ini omitted `can_crouch`
    if (!out.crouch_supported &&
        mod_detect_detail::contains_any(corpus, {"crouch_toggle", "agent_crouch", "pbod_crouch"})) {
        out.crouch_supported = true;
    }

    // 15. Tactical Battle Map (`Backspace`) when a mod battle-map / PBOD presentation is detected
    // (avoid matching vanilla Native's `prsnt_battle` so Native stays clean).
    if (mod_detect_detail::contains_any(corpus, {"prsnt_pbod_battle", "tactical_map", "pbod_minimap",
                                                 "battle_minimap", "world_of_vasnar", "108_heroes"}) &&
        key_active(mod_detect_detail::kDikBackspace)) {
        add_unique(DetectedModButton{
            "mod_tactical_map",
            "Battle Map",
            TouchIconId::tactical_map,
            mod_detect_detail::kDikBackspace,
            false,
            false,
            true,
            "Tactical battle map overlay detected in module",
        });
    }

    return out;
}

}  // namespace wse::platform
