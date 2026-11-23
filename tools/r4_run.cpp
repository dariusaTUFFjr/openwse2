// R4 bring-up run (`docs/RECOMP_PLAN.md` "R4 design", "Bring-up tool"). LOCAL ONLY: it reads the
// user's own install and runs retail code translated on this machine; never run in CI, and
// nothing it reads or writes is committed. Linux x86-64 (WSL, `tools/wsl_gate.sh` builds it).
//
//   r4_run <install dir> [--work <dir>] [--retranslate] [--batch <n>] [--opt <0-3>] [--jobs <n>]
//          [--reuse] [--quiet] [--module <name>] [--frames <n>] [--for-phone <dir>]
//          [--capture <file> [--capture-frames <n>]] [--screen <w>x<h>]
//
// The runner itself is `engine/src/recomp/r4_startup.{h,cpp}` (shared with the phone's Advanced >
// "R4 startup run"); this tool reads the exe, `lua5164.dll` and the PDB from the install, mounts the
// install and `<work>/user` as host directories, and keeps the translated batches in `<work>/so`,
// each reused while the image, the options and the translator library are unchanged. `--reuse`
// keeps every batch already built whatever its key, for when only a non-lifter file of the
// translator library changed. `--frames <n>` ends the run after n frames (exit code 0). A fault (a
// missing import names its DLL and function: the next thing to implement) ends the run with the
// fault and the last imports called.
//
// `--for-phone <dir>` (2026-11-21): translate for the phone instead (the portable AArch64 Android
// target the app also uses, `portable_android_target`) into `<dir>`, run nothing, and exit 0. The
// batches are the user's own exe translated on their own PC for their own phone: the app imports
// them ("Import prepared game files") instead of translating for hours. Wrapped for players by
// `tools/prepare_for_phone.ps1`.
//
// `--capture <file>` (R5a step 3, 2026-11-22): write every D3D9/D3DX call the run makes, with the
// data its pointers reach, to `<file>` (`wse/recomp/d3d9_capture.h`), up to `--capture-frames`
// presented frames (default 60). A Windows tool replays it on the real d3d9.dll for reference
// frames. The file holds retail data: it stays on this PC and is never committed.

#include <sys/stat.h>

#include <algorithm>
#include <span>
#include <set>
#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "../engine/src/recomp/r4_startup.h"
#include "../engine/src/recomp/recomp_llvm.h"
#include "wse/recomp/d3d9_capture.h"
#include "wse/recomp/screen.h"
#include "wse/recomp/shim_d3d9.h"
#include "wse/asset/dds.h"
#include "wse/asset/etc2.h"
#include "wse/asset/texture_cache.h"
#include "wse/platform/vfs.h"

namespace fs = std::filesystem;

namespace {

std::vector<std::uint8_t> read_file(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

// A file in `dir` whose name matches `name` case-insensitively (Windows installs mix cases).
fs::path find_ci(const fs::path& dir, const std::string& name) {
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(dir, ec)) {
        std::string a = e.path().filename().string();
        std::string b = name;
        for (char& c : a) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        for (char& c : b) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (a == b) return e.path();
    }
    return dir / name;
}

// The textures the phone converts to ETC2 the first time it loads them (its GPU has no BC),
// converted here instead: the .dds files the game loads, which are those in TexturesHD and, for a
// name with no HD version, in Textures (the install's and the module's; the phone's own cache
// showed 525 of its first 798 textures from TexturesHD, 2026-11-23),
// written as the phone's cache files (`texture_cache.h` name and format) under `out`. A texture is
// written for each size cap a phone may use (none, 2048, 1024) where the cap changes its first mip,
// since that is part of the name. Returns the number of files written.
std::size_t convert_textures_for_phone(const fs::path& install, const std::string& module, const fs::path& out,
                                       unsigned threads) {
    std::vector<fs::path> files;
    auto lower = [](std::string t) {
        for (char& c : t) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return t;
    };
    // `skip`: names (lower case, relative to the folder) that a higher-priority folder supplies.
    auto add_dir = [&](const fs::path& dir, const std::set<std::string>& skip, std::set<std::string>* names) {
        std::error_code ec;
        if (!fs::is_directory(dir, ec)) return;
        for (fs::recursive_directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
            if (lower(it->path().extension().string()) != ".dds" || !it->is_regular_file(ec)) continue;
            const std::string rel = lower(fs::relative(it->path(), dir, ec).generic_string());
            if (skip.count(rel) != 0) continue;
            if (names != nullptr) names->insert(rel);
            files.push_back(it->path());
        }
    };
    for (const fs::path& base : {install, find_ci(find_ci(install, "Modules"), module)}) {
        std::set<std::string> hd;
        add_dir(find_ci(base, "TexturesHD"), {}, &hd);
        add_dir(find_ci(base, "Textures"), hd, nullptr);
    }
    std::error_code ec;
    fs::create_directories(out, ec);
    std::atomic<std::size_t> next{0};
    std::atomic<std::size_t> written{0};
    std::atomic<std::size_t> failed{0};
    auto work = [&]() {
        for (std::size_t i = next++; i < files.size(); i = next++) {
            const std::vector<std::uint8_t> bytes = read_file(files[i]);
            wse::asset::DdsImageView view;
            const auto span = std::span<const std::byte>(reinterpret_cast<const std::byte*>(bytes.data()), bytes.size());
            if (!wse::asset::parse_dds(span, view, nullptr) || view.mip_count == 0U) {
                ++failed;
                continue;
            }
            std::set<std::uint32_t> mips;
            for (const std::uint32_t cap : {0U, 2048U, 1024U}) {
                mips.insert(wse::asset::first_mip_for_cap(view.width, view.height, view.mip_count, cap));
            }
            for (const std::uint32_t first_mip : mips) {
                const fs::path dest = out / wse::asset::cache_key(bytes, first_mip);
                std::error_code fec;
                if (fs::exists(dest, fec)) continue;  // a run that is resumed keeps what it wrote
                wse::asset::etc2::Image img;
                if (!wse::asset::etc2::transcode_dds(view, first_mip, img, nullptr, 1)) {
                    ++failed;
                    continue;
                }
                const std::vector<std::uint8_t> enc = wse::asset::encode(img);
                const fs::path tmp = fs::path(dest.string() + ".tmp");
                std::ofstream(tmp, std::ios::binary).write(reinterpret_cast<const char*>(enc.data()),
                                                           static_cast<std::streamsize>(enc.size()));
                fs::rename(tmp, dest, fec);
                if (fec) {
                    fs::remove(tmp, fec);
                    ++failed;
                } else {
                    ++written;
                }
            }
        }
    };
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < std::max(1U, threads); ++t) pool.emplace_back(work);
    for (std::thread& t : pool) t.join();
    std::printf("textures: %zu .dds files, %zu converted files written, %zu failed\n", files.size(), written.load(),
                failed.load());
    return written.load();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: %s <install dir> [--work <dir>] [--retranslate] [--batch <n>] [--opt <0-3>] [--jobs <n>] "
                     "[--reuse] [--quiet] [--module <name>] [--frames <n>] [--for-phone <dir>] [--capture <file> [--capture-frames <n>]] [--screen <w>x<h>]\n",
                     argv[0]);
        return 2;
    }
    const fs::path install = argv[1];
    fs::path work = "/root/wse/r4";
    wse::recomp::StartupOptions o;
    fs::path for_phone;
    std::string capture;
    std::uint64_t capture_frames = 60;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--work" && i + 1 < argc) work = argv[++i];
        else if (a == "--retranslate") o.retranslate = true;
        else if (a == "--batch" && i + 1 < argc) o.batch_kib = std::strtoul(argv[++i], nullptr, 10);
        else if (a == "--opt" && i + 1 < argc) o.opt = std::atoi(argv[++i]);
        else if (a == "--jobs" && i + 1 < argc) o.jobs = static_cast<unsigned>(std::max(1, std::atoi(argv[++i])));
        else if (a == "--reuse") o.reuse = true;
        else if (a == "--quiet") o.quiet = true;
        else if (a == "--module" && i + 1 < argc) o.module = argv[++i];
        else if (a == "--frames" && i + 1 < argc) o.frame_limit = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--for-phone" && i + 1 < argc) for_phone = argv[++i];
        else if (a == "--capture" && i + 1 < argc) capture = argv[++i];
        else if (a == "--capture-frames" && i + 1 < argc) capture_frames = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--screen" && i + 1 < argc) {  // e.g. 2412x1080: the display the game sees (widescreen)
            int sw = 0, sh = 0;
            if (std::sscanf(argv[++i], "%dx%d", &sw, &sh) == 2) wse::recomp::shim::set_screen_size(sw, sh);
        }
    }
    // The capture is set up through the device's construction hook, so the shared startup code
    // (and with it the translator fingerprint) is unchanged.
    static std::unique_ptr<wse::recomp::d3dcap::Writer> writer;
    if (!capture.empty()) {
        writer = std::make_unique<wse::recomp::d3dcap::Writer>(capture);
        if (!writer->ok()) {
            std::fprintf(stderr, "cannot write the capture %s\n", capture.c_str());
            return 2;
        }
        wse::recomp::shim::D3d9::created_hook() = [capture_frames](wse::recomp::shim::D3d9& d) {
            d.on_call = [&d, capture_frames](const wse::recomp::d3dcap::Call& c) {
                if (d.frames() <= capture_frames) writer->write(c);
            };
        };
    }
    std::error_code ec;
    fs::create_directories(work / "user", ec);
    setvbuf(stdout, nullptr, _IOLBF, 0);

    o.exe = read_file(find_ci(install, "mb_warband_wse2_x64.exe"));
    o.lua = read_file(find_ci(install, "lua5164.dll"));
    o.pdb = read_file(find_ci(install, "mb_warband_wse2_x64.pdb"));
    o.so_dir = (work / "so").string();
    if (!for_phone.empty()) {
        wse::recomp::set_code_target(wse::recomp::portable_android_target());
        o.translate_only = true;
        o.so_dir = for_phone.string();
    }
    static wse::platform::HostVfs install_vfs(install);
    static wse::platform::HostVfs user_vfs(work / "user");
    o.install = &install_vfs;
    o.user = &user_vfs;
    const int rc = wse::recomp::run_startup(o);
    if (!for_phone.empty() && rc == 0) {
        // Converted textures go in a subfolder; the phone puts them in its texture cache.
        (void)convert_textures_for_phone(install, o.module, for_phone / "texcache", std::max(1U, o.jobs));
    }
    return rc;
}
