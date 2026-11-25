// Phase 5 R7 Savegame Oracle tool (`docs/RECOMP_PLAN.md` "R7 — WSL x86-64 Bring-Up & Savegame
// Oracle"): validates a binary Warband/WSE2 `.sav` file structurally via our Phase 2A reader
// (`wse/game/savegame.h`), prints header + world summary (player, day, gold, party size/pos),
// and optionally compares two saves (`--compare a.sav b.sav`).
//
// Portable C++20 (Linux / Windows). Never embeds or commits retail files.
//
//   wse_savecheck <file.sav> [--install <dir>] [--module <name>] [--dont-load-regular-inv]
//   wse_savecheck --compare <a.sav> <b.sav> [--install <dir>] [--module <name>]

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <span>
#include <string>
#include <vector>

#include "wse/game/module_bundle.h"
#include "wse/game/savegame.h"
#include "wse/io/binary_stream.h"
#include "wse/platform/vfs.h"
#include "wse/vm/block_vm.h"

namespace fs = std::filesystem;

namespace {

bool read_bytes(const fs::path& p, std::vector<std::uint8_t>& out) {
    std::ifstream in(p, std::ios::binary);
    if (!in) return false;
    out.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    return static_cast<bool>(in) || in.eof();
}

struct ParsedSave {
    bool file_ok = false;
    bool module_loaded = false;
    std::size_t file_size = 0U;
    std::uint64_t checksum = 0U;
    wse::game::SavegameLoadReport report{};
    std::string player_name;
    std::int32_t player_gold = -1;
    std::int32_t main_companions = -1;
    std::int32_t main_prisoners = -1;
    float map_x = 0.0f;
    float map_y = 0.0f;
    bool has_map_pos = false;
    std::size_t active_parties = 0U;
    std::size_t troop_count = 0U;
    std::size_t faction_count = 0U;
    std::size_t quest_count = 0U;
};

ParsedSave inspect_save(const fs::path& sav_path,
                        const std::string& install_dir,
                        const std::string& module_name,
                        bool dont_load_regular_inv) {
    ParsedSave out{};
    std::vector<std::uint8_t> raw;
    if (!read_bytes(sav_path, raw) || raw.empty()) {
        return out;
    }
    out.file_ok = true;
    out.file_size = raw.size();
    const std::span<const std::byte> bytes(reinterpret_cast<const std::byte*>(raw.data()), raw.size());
    out.checksum = wse::io::compute_savegame_checksum(bytes);

    wse::vm::VmContext vm_ctx;
    wse::game::ModuleBundle bundle;
    if (!install_dir.empty()) {
        wse::platform::HostVfs vfs(install_dir);
        wse::game::ModuleLoadOptions opts{};
        opts.require_core_files = false;
        const auto mres = wse::game::load_module(vfs, "Modules/" + module_name, vm_ctx, bundle, opts);
        out.module_loaded = mres.ok;
        if (!mres.ok) {
            std::fprintf(stderr, "wse_savecheck: warning: load_module(%s) failed: %s\n",
                         module_name.c_str(), mres.format().c_str());
        } else if (!dont_load_regular_inv) {
            const auto it = bundle.module_ini.settings.find("dont_load_regular_troop_inventories");
            if (it != bundle.module_ini.settings.end() && it->second != "0") {
                dont_load_regular_inv = true;
            }
        }
    }

    wse::game::SavegameRuntimeContext rc{};
    rc.world = &vm_ctx.world;
    rc.global_variables = &vm_ctx.globalVariables;
    rc.triggers = &bundle.game_triggers;
    rc.simple_triggers = &bundle.simple_triggers;
    rc.timers_total = vm_ctx.timersTotal;
    rc.dont_load_regular_troop_inventories = dont_load_regular_inv;

    out.report = wse::game::read_savegame_container(bytes, rc);
    out.player_name = out.report.header.player_name;
    out.troop_count = vm_ctx.world.troops.size();
    out.faction_count = vm_ctx.world.factions.size();
    out.quest_count = vm_ctx.world.quests.size();
    for (const auto& p : vm_ctx.world.parties) {
        if (p.active) ++out.active_parties;
    }
    if (vm_ctx.world.is_valid_troop(vm_ctx.world.player_troop_no)) {
        out.player_gold = vm_ctx.world.troops[static_cast<std::size_t>(vm_ctx.world.player_troop_no)].gold;
    }
    if (vm_ctx.world.is_valid_party(vm_ctx.world.main_party_no)) {
        const auto& mp = vm_ctx.world.parties[static_cast<std::size_t>(vm_ctx.world.main_party_no)];
        out.main_companions = mp.get_num_companions();
        out.main_prisoners = mp.get_num_prisoners();
        out.map_x = mp.position.o.x;
        out.map_y = mp.position.o.y;
        out.has_map_pos = true;
    }
    return out;
}

void print_save(const char* label, const fs::path& path, const ParsedSave& s) {
    const auto status_sv = wse::game::describe(s.report.status);
    std::printf(
        "[%s] %s (%zu bytes, checksum=0x%016llx)\n"
        "  status=%.*s (%s) magic=0x%08x ver=%d mod_ver=%d consumed=%zu/%zu count_exceeded=%d\n"
        "  slot=\"%s\" player=\"%s\" level=%d date_hours=%.2f\n",
        label, path.string().c_str(), s.file_size, static_cast<unsigned long long>(s.checksum),
        static_cast<int>(status_sv.size()), status_sv.data(),
        s.module_loaded ? "with module defs" : "header/standalone",
        s.report.header.magic, s.report.header.savegame_version, s.report.header.module_version,
        s.report.bytes_consumed, s.file_size, s.report.count_exceeded ? 1 : 0,
        s.report.header.name.c_str(), s.player_name.c_str(), s.report.header.player_level,
        s.report.header.game_date_hours);
    if (s.report.succeeded()) {
        std::printf(
            "  gold=%d main_party=(%d troops, %d prisoners) pos=(%.3f, %.3f) active_parties=%zu troops=%zu factions=%zu quests=%zu\n",
            s.player_gold, s.main_companions, s.main_prisoners, s.map_x, s.map_y,
            s.active_parties, s.troop_count, s.faction_count, s.quest_count);
    }
}

bool valid_structural(const ParsedSave& s, bool require_full_body) {
    if (!s.file_ok) return false;
    if (require_full_body) return s.report.succeeded();
    return s.report.header.magic == wse::game::kSavegameMagic &&
           s.report.header.savegame_version >= wse::game::kSavegameMinVersion &&
           s.report.header.savegame_version <= wse::game::kSavegameFormatVersion;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage:\n"
                     "  %s <save.sav> [--install <dir>] [--module <name>] [--dont-load-regular-inv] [--header-only]\n"
                     "  %s --compare <a.sav> <b.sav> [--install <dir>] [--module <name>] [--dont-load-regular-inv] [--header-only]\n",
                     argv[0], argv[0]);
        return 2;
    }

    bool compare_mode = false;
    fs::path path_a;
    fs::path path_b;
    std::string install_dir;
    std::string module_name = "Native";
    bool dont_load_regular_inv = false;
    bool header_only = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--compare" && i + 2 < argc) {
            compare_mode = true;
            path_a = argv[++i];
            path_b = argv[++i];
        } else if (a == "--install" && i + 1 < argc) {
            install_dir = argv[++i];
        } else if (a == "--module" && i + 1 < argc) {
            module_name = argv[++i];
        } else if (a == "--dont-load-regular-inv") {
            dont_load_regular_inv = true;
        } else if (a == "--header-only") {
            header_only = true;
        } else if (path_a.empty()) {
            path_a = a;
        } else if (compare_mode && path_b.empty()) {
            path_b = a;
        }
    }

    if (path_a.empty() || (compare_mode && path_b.empty())) {
        std::fprintf(stderr, "wse_savecheck: missing input savegame path\n");
        return 2;
    }

    const ParsedSave sa = inspect_save(path_a, install_dir, module_name, dont_load_regular_inv);
    print_save("A", path_a, sa);
    const bool ok_a = valid_structural(sa, !header_only);

    if (!compare_mode) {
        std::printf("RESULT %s\n", ok_a ? "pass" : "fail");
        return ok_a ? 0 : 1;
    }

    const ParsedSave sb = inspect_save(path_b, install_dir, module_name, dont_load_regular_inv);
    print_save("B", path_b, sb);
    const bool ok_b = valid_structural(sb, !header_only);
    bool match = ok_a && ok_b &&
                 sa.report.header.savegame_version == sb.report.header.savegame_version &&
                 sa.player_name == sb.player_name &&
                 sa.report.header.player_level == sb.report.header.player_level &&
                 std::fabs(sa.report.header.game_date_hours - sb.report.header.game_date_hours) < 0.05f;
    if (!header_only && ok_a && ok_b) {
        match = match &&
                sa.player_gold == sb.player_gold &&
                sa.main_companions == sb.main_companions &&
                sa.main_prisoners == sb.main_prisoners &&
                std::fabs(sa.map_x - sb.map_x) < 0.05f &&
                std::fabs(sa.map_y - sb.map_y) < 0.05f;
    }
    std::printf("COMPARE %s\n", match ? "match" : "mismatch");
    return match ? 0 : 1;
}
