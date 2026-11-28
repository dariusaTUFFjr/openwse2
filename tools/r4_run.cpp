// R4/R7 bring-up run (`docs/RECOMP_PLAN.md` "R4 design", "R7 — WSL x86-64 Bring-Up & Savegame Oracle").
// LOCAL ONLY: reads the user's own install and runs retail code translated on this machine; never
// run in CI, and nothing it reads or writes is committed. Linux x86-64 (WSL, `tools/wsl_gate.sh`
// builds it).
//
//   r4_run <install dir> [--work <dir>] [--retranslate] [--batch <n>] [--opt <0-3>] [--jobs <n>]
//          [--reuse] [--quiet] [--module <name>] [--frames <n>] [--for-phone <dir>]
//          [--capture <file> [--capture-frames <n>]] [--screen <w>x<h>]
//          [--vulkan] [--icd <auto|lavapipe|system|path>] [--window]
//          [--audio <capture|sdl|none>] [--script <file>]
//          [--dump-frames <dir>] [--dump-every <n>] [--log-imports <file>]

#include <sys/stat.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#ifdef WSE_HAS_SDL3
#include <SDL3/SDL.h>
#include "sdl_audio.h"
#endif

#include "../engine/src/recomp/r4_startup.h"
#include "../engine/src/recomp/recomp_llvm.h"
#include "d3d9_vulkan.h"
#include "driver_loader.h"
#include "wse/asset/audio_codecs.h"
#include "wse/asset/dds.h"
#include "wse/asset/etc2.h"
#include "wse/asset/texture_cache.h"
#include "wse/platform/audio.h"
#include "wse/platform/rhi.h"
#include "wse/platform/vfs.h"
#include "wse/recomp/d3d9_capture.h"
#include "wse/recomp/host_stack.h"
#include "wse/recomp/input_hub.h"
#include "wse/recomp/pdb_reader.h"
#include "wse/recomp/pe_image.h"
#include "wse/recomp/screen.h"
#include "wse/recomp/scripted_input.h"
#include "wse/recomp/session_replay.h"
#include "wse/recomp/shim_d3d9.h"
#include "wse/recomp/shim_fmod_steam.h"
#include "wse/recomp/shim_system.h"
#include "wse/recomp/shim_user32.h"

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

void set_env_var(const char* name, const char* value, bool overwrite) {
#ifdef _WIN32
    if (!overwrite && std::getenv(name) != nullptr) return;
    (void)_putenv_s(name, value);
#else
    (void)::setenv(name, value, overwrite ? 1 : 0);
#endif
}

std::string to_lower_ascii(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

bool ends_with_ci(std::string_view s, std::string_view suffix) {
    if (s.size() < suffix.size()) return false;
    return to_lower_ascii(std::string(s.substr(s.size() - suffix.size()))) ==
           to_lower_ascii(std::string(suffix));
}

// Streaming capture sink with simulated real-time playback drain so `FmodSteam::worker_loop()`
// mixes continuously across long runs instead of halting after the first 2,048 queued frames.
class StreamingCaptureAudioDevice final : public wse::platform::AudioDevice {
public:
    bool open(const wse::platform::AudioDeviceSpec& s) override {
        const std::lock_guard<std::mutex> lock(m_);
        if (s.sample_rate == 0U || s.channels != 2U) return false;
        spec_ = s;
        open_ = true;
        paused_ = false;
        submitted_frames_ = 0U;
        nonzero_frames_ = 0U;
        sum_sq_l_ = 0.0;
        sum_sq_r_ = 0.0;
        peak_abs_ = 0.0;
        hash_ = 14695981039346656037ULL;
        start_tp_ = std::chrono::steady_clock::now();
        return true;
    }
    void close() override {
        const std::lock_guard<std::mutex> lock(m_);
        open_ = false;
    }
    [[nodiscard]] bool is_open() const noexcept override {
        const std::lock_guard<std::mutex> lock(m_);
        return open_;
    }
    [[nodiscard]] wse::platform::AudioDeviceSpec spec() const noexcept override {
        const std::lock_guard<std::mutex> lock(m_);
        return spec_;
    }
    bool submit_frames(std::span<const float> interleaved_stereo) override {
        const std::lock_guard<std::mutex> lock(m_);
        if (!open_ || (interleaved_stereo.size() & 1U) != 0U) return false;
        const std::size_t n = interleaved_stereo.size() / 2U;
        for (std::size_t i = 0U; i < n; ++i) {
            const float l = interleaved_stereo[i * 2U];
            const float r = interleaved_stereo[i * 2U + 1U];
            const double dl = static_cast<double>(l);
            const double dr = static_cast<double>(r);
            sum_sq_l_ += dl * dl;
            sum_sq_r_ += dr * dr;
            peak_abs_ = std::max(peak_abs_, std::max(std::fabs(dl), std::fabs(dr)));
            if (std::fabs(dl) > 1e-6 || std::fabs(dr) > 1e-6) ++nonzero_frames_;
            for (const float v : {l, r}) {
                const float clamped = std::clamp(v, -1.0F, 1.0F);
                const auto q = static_cast<std::int16_t>(std::lround(clamped * 32767.0F));
                const auto u = static_cast<std::uint16_t>(q);
                hash_ ^= static_cast<std::uint64_t>(u & 0xFFU);
                hash_ *= 1099511628211ULL;
                hash_ ^= static_cast<std::uint64_t>((u >> 8U) & 0xFFU);
                hash_ *= 1099511628211ULL;
            }
        }
        submitted_frames_ += n;
        return true;
    }
    [[nodiscard]] std::size_t queued_frames() const noexcept override {
        const std::lock_guard<std::mutex> lock(m_);
        if (!open_ || paused_) return 0U;
        const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                    std::chrono::steady_clock::now() - start_tp_)
                                    .count();
        const std::uint64_t drained =
            elapsed_us > 0 ? (static_cast<std::uint64_t>(elapsed_us) * spec_.sample_rate) / 1000000ULL : 0ULL;
        return submitted_frames_ > drained ? static_cast<std::size_t>(submitted_frames_ - drained) : 0U;
    }
    void clear() override {
        const std::lock_guard<std::mutex> lock(m_);
        start_tp_ = std::chrono::steady_clock::now();
        submitted_frames_ = 0U;
    }
    void set_paused(bool paused) override {
        const std::lock_guard<std::mutex> lock(m_);
        paused_ = paused;
    }
    [[nodiscard]] const char* backend_name() const noexcept override { return "streaming_capture"; }

    struct Stats {
        std::size_t frames = 0U;
        std::size_t nonzero_frames = 0U;
        double rms_left = 0.0;
        double rms_right = 0.0;
        double peak = 0.0;
        std::uint64_t hash = 0U;
    };
    [[nodiscard]] Stats stats() const noexcept {
        const std::lock_guard<std::mutex> lock(m_);
        Stats s;
        s.frames = static_cast<std::size_t>(submitted_frames_);
        s.nonzero_frames = static_cast<std::size_t>(nonzero_frames_);
        if (submitted_frames_ > 0U) {
            s.rms_left = std::sqrt(sum_sq_l_ / static_cast<double>(submitted_frames_));
            s.rms_right = std::sqrt(sum_sq_r_ / static_cast<double>(submitted_frames_));
        }
        s.peak = peak_abs_;
        s.hash = hash_;
        return s;
    }

private:
    mutable std::mutex m_;
    wse::platform::AudioDeviceSpec spec_{};
    bool open_ = false;
    bool paused_ = false;
    std::chrono::steady_clock::time_point start_tp_{};
    std::uint64_t submitted_frames_ = 0U;
    std::uint64_t nonzero_frames_ = 0U;
    double sum_sq_l_ = 0.0;
    double sum_sq_r_ = 0.0;
    double peak_abs_ = 0.0;
    std::uint64_t hash_ = 14695981039346656037ULL;
};

bool write_ppm_rgba(const fs::path& p,
                    std::span<const std::uint8_t> rgba,
                    std::uint32_t w,
                    std::uint32_t h) {
    if (w == 0U || h == 0U || rgba.size() < static_cast<std::size_t>(w) * h * 4U) return false;
    std::ofstream f(p, std::ios::binary);
    if (!f) return false;
    f << "P6\n" << w << ' ' << h << "\n255\n";
    std::vector<std::uint8_t> rgb(static_cast<std::size_t>(w) * 3U);
    for (std::uint32_t y = 0; y < h; ++y) {
        const std::uint8_t* row = rgba.data() + static_cast<std::size_t>(y) * w * 4U;
        for (std::uint32_t x = 0; x < w; ++x) {
            rgb[x * 3U + 0U] = row[x * 4U + 0U];
            rgb[x * 3U + 1U] = row[x * 4U + 1U];
            rgb[x * 3U + 2U] = row[x * 4U + 2U];
        }
        f.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
    }
    return static_cast<bool>(f);
}

const char* scene_name(wse::recomp::shim::TouchScene s) noexcept {
    switch (s) {
    case wse::recomp::shim::TouchScene::main_menu: return "main_menu";
    case wse::recomp::shim::TouchScene::battle: return "battle";
    case wse::recomp::shim::TouchScene::orders: return "orders";
    case wse::recomp::shim::TouchScene::campaign_map: return "campaign_map";
    case wse::recomp::shim::TouchScene::party: return "party";
    case wse::recomp::shim::TouchScene::trade: return "trade";
    case wse::recomp::shim::TouchScene::dialog: return "dialog";
    case wse::recomp::shim::TouchScene::screens: return "screens";
    }
    return "?";
}

bool read_vfs_guest_path(wse::platform::Vfs* install_vfs,
                         wse::platform::Vfs* user_vfs,
                         const std::string& path,
                         std::vector<std::uint8_t>& out) {
    std::string p = path;
    std::string low = to_lower_ascii(p);
    static constexpr std::string_view kRoot = "c:\\wse2\\";
    if (low.rfind(kRoot, 0U) == 0U) p = p.substr(kRoot.size());
    for (char& c : p) {
        if (c == '\\') c = '/';
    }
    while (!p.empty() && p.front() == '/') p.erase(p.begin());
    return (install_vfs != nullptr && install_vfs->read_file(p, out)) ||
           (user_vfs != nullptr && user_vfs->read_file(p, out));
}

bool dump_pe_imports_file(const fs::path& out_path,
                          const std::vector<std::uint8_t>& exe_bytes,
                          const std::vector<std::uint8_t>& lua_bytes,
                          std::string& error_out) {
    std::ofstream out(out_path, std::ios::binary);
    if (!out) {
        error_out = "cannot open --log-imports output: " + out_path.string();
        return false;
    }
    auto dump_one = [&](const char* mod_label, const std::vector<std::uint8_t>& raw) {
        if (raw.empty()) return;
        wse::recomp::PeImage pe;
        std::string err;
        if (!wse::recomp::parse_pe(raw.data(), raw.size(), pe, &err)) {
            out << "# " << mod_label << ": parse_pe failed: " << err << "\n";
            return;
        }
        std::vector<std::uint8_t> mapped(pe.size_of_image, 0U);
        if (!wse::recomp::map_pe_sections(raw.data(), raw.size(), pe, mapped.data())) {
            out << "# " << mod_label << ": map_pe_sections failed\n";
            return;
        }
        std::vector<wse::recomp::PeImport> imports;
        if (!wse::recomp::read_pe_imports(mapped.data(), mapped.size(), pe.image_base, imports, &err)) {
            out << "# " << mod_label << ": read_pe_imports failed: " << err << "\n";
            return;
        }
        std::map<std::string, std::size_t> by_dll;
        for (const auto& imp : imports) ++by_dll[imp.dll];
        out << "[" << mod_label << "] total_imports=" << imports.size()
            << " dlls=" << by_dll.size() << "\n";
        for (const auto& [dll, cnt] : by_dll) {
            out << "  dll " << dll << ": " << cnt << " imports\n";
        }
        for (const auto& imp : imports) {
            out << mod_label << "\t" << imp.dll << "\t";
            if (!imp.name.empty()) {
                out << imp.name;
            } else {
                out << "#" << imp.ordinal;
            }
            out << "\t0x" << std::hex << imp.slot_va << std::dec << "\n";
        }
    };
    dump_one("mb_warband_wse2_x64.exe", exe_bytes);
    dump_one("lua5164.dll", lua_bytes);
    return static_cast<bool>(out);
}

bool init_vulkan_loader(const std::string& icd_arg,
                        const fs::path& work,
                        std::shared_ptr<wse::platform::VulkanDriverLoader>& out_loader,
                        wse::platform::VulkanDriverProbeResult& out_probe,
                        std::string& error_out) {
    set_env_var("MESA_SHADER_CACHE_DIR", (work / "mesa_cache").string().c_str(), false);

    auto try_system = [&](std::string& err) -> bool {
        auto l = std::make_shared<wse::platform::VulkanDriverLoader>();
        wse::platform::VulkanDriverProbeResult p;
        if (!l->load_system(&err)) return false;
        if (!l->probe_and_create_device(p) || !p.ok) {
            err = p.error.empty() ? "probe_and_create_device failed" : p.error;
            return false;
        }
        out_loader = std::move(l);
        out_probe = std::move(p);
        return true;
    };

    auto try_custom_so = [&](const std::string& so_path, std::string& err) -> bool {
        auto l = std::make_shared<wse::platform::VulkanDriverLoader>();
        wse::platform::VulkanDriverProbeResult p;
        if (!l->load_custom_driver(wse::platform::DriverMode::custom_icd, so_path, {}, &err)) return false;
        if (!l->probe_and_create_device(p) || !p.ok) {
            err = p.error.empty() ? "probe_and_create_device failed" : p.error;
            return false;
        }
        out_loader = std::move(l);
        out_probe = std::move(p);
        return true;
    };

    auto try_lavapipe = [&](std::string& err) -> bool {
        for (const char* so_cand : {"/usr/lib/x86_64-linux-gnu/libvulkan_lvp.so",
                                    "/usr/lib/libvulkan_lvp.so"}) {
            std::error_code ec;
            if (fs::exists(so_cand, ec) && try_custom_so(so_cand, err)) return true;
        }
        for (const char* json_cand : {"/usr/share/vulkan/icd.d/lvp_icd.x86_64.json",
                                      "/usr/share/vulkan/icd.d/lvp_icd.json"}) {
            std::error_code ec;
            if (fs::exists(json_cand, ec)) {
                set_env_var("VK_ICD_FILENAMES", json_cand, true);
                if (try_system(err)) return true;
            }
        }
        if (err.empty()) err = "Lavapipe ICD not found";
        return false;
    };

    const std::string mode = to_lower_ascii(icd_arg);
    if (mode == "lavapipe") {
        return try_lavapipe(error_out);
    }
    if (mode == "system") {
        return try_system(error_out);
    }
    if (mode == "auto" || mode.empty()) {
        std::string sys_err;
        if (try_system(sys_err)) return true;
        std::string lvp_err;
        if (try_lavapipe(lvp_err)) return true;
        error_out = "system (" + sys_err + ") and lavapipe (" + lvp_err + ") failed";
        return false;
    }
    if (ends_with_ci(icd_arg, ".json")) {
        set_env_var("VK_ICD_FILENAMES", icd_arg.c_str(), true);
        return try_system(error_out);
    }
    return try_custom_so(icd_arg, error_out);
}

#ifdef WSE_HAS_SDL3
struct FrameMailbox {
    std::mutex m;
    std::vector<std::uint8_t> rgba;
    std::uint32_t width = 0U;
    std::uint32_t height = 0U;
    std::uint64_t seq = 0U;
};

bool sdl_scancode_to_dik(SDL_Scancode sc, std::uint8_t& out_dik, std::uint32_t& out_vk) noexcept {
    namespace d = wse::recomp::shim::dik;
    if (sc >= SDL_SCANCODE_A && sc <= SDL_SCANCODE_Z) {
        static constexpr std::uint8_t kLetters[26] = {
            d::kA, d::kB, d::kC, d::kD, d::kE, d::kF, d::kG, d::kH, d::kI,
            d::kJ, d::kK, d::kL, d::kM, d::kN, d::kO, d::kP, d::kQ, d::kR,
            d::kS, d::kT, d::kU, d::kV, d::kW, d::kX, d::kY, d::kZ};
        const int idx = static_cast<int>(sc - SDL_SCANCODE_A);
        out_dik = kLetters[idx];
        out_vk = static_cast<std::uint32_t>('A' + idx);
        return true;
    }
    if (sc >= SDL_SCANCODE_1 && sc <= SDL_SCANCODE_9) {
        const int idx = static_cast<int>(sc - SDL_SCANCODE_1);
        out_dik = static_cast<std::uint8_t>(d::k1 + idx);
        out_vk = static_cast<std::uint32_t>('1' + idx);
        return true;
    }
    switch (sc) {
    case SDL_SCANCODE_0: out_dik = d::k0; out_vk = '0'; return true;
    case SDL_SCANCODE_ESCAPE: out_dik = d::kEscape; out_vk = 0x1BU; return true;
    case SDL_SCANCODE_MINUS: out_dik = d::kMinus; out_vk = 0xBDU; return true;
    case SDL_SCANCODE_EQUALS: out_dik = d::kEquals; out_vk = 0xBBU; return true;
    case SDL_SCANCODE_BACKSPACE: out_dik = d::kBackspace; out_vk = 0x08U; return true;
    case SDL_SCANCODE_TAB: out_dik = d::kTab; out_vk = 0x09U; return true;
    case SDL_SCANCODE_LEFTBRACKET: out_dik = d::kLBracket; out_vk = 0xDBU; return true;
    case SDL_SCANCODE_RIGHTBRACKET: out_dik = d::kRBracket; out_vk = 0xDDU; return true;
    case SDL_SCANCODE_RETURN: out_dik = d::kReturn; out_vk = 0x0DU; return true;
    case SDL_SCANCODE_KP_ENTER: out_dik = d::kNumPadEnter; out_vk = 0x0DU; return true;
    case SDL_SCANCODE_LCTRL: out_dik = d::kLCtrl; out_vk = 0x11U; return true;
    case SDL_SCANCODE_RCTRL: out_dik = d::kRCtrl; out_vk = 0x11U; return true;
    case SDL_SCANCODE_SEMICOLON: out_dik = d::kSemicolon; out_vk = 0xBAU; return true;
    case SDL_SCANCODE_APOSTROPHE: out_dik = d::kApostrophe; out_vk = 0xDEU; return true;
    case SDL_SCANCODE_GRAVE: out_dik = d::kGrave; out_vk = 0xC0U; return true;
    case SDL_SCANCODE_LSHIFT: out_dik = d::kLShift; out_vk = 0x10U; return true;
    case SDL_SCANCODE_RSHIFT: out_dik = d::kRShift; out_vk = 0x10U; return true;
    case SDL_SCANCODE_BACKSLASH: out_dik = d::kBackslash; out_vk = 0xDCU; return true;
    case SDL_SCANCODE_COMMA: out_dik = d::kComma; out_vk = 0xBCU; return true;
    case SDL_SCANCODE_PERIOD: out_dik = d::kPeriod; out_vk = 0xBEU; return true;
    case SDL_SCANCODE_SLASH: out_dik = d::kSlash; out_vk = 0xBFU; return true;
    case SDL_SCANCODE_LALT: out_dik = d::kLAlt; out_vk = 0x12U; return true;
    case SDL_SCANCODE_RALT: out_dik = d::kRAlt; out_vk = 0x12U; return true;
    case SDL_SCANCODE_SPACE: out_dik = d::kSpace; out_vk = 0x20U; return true;
    case SDL_SCANCODE_F1: out_dik = d::kF1; out_vk = 0x70U; return true;
    case SDL_SCANCODE_F2: out_dik = d::kF2; out_vk = 0x71U; return true;
    case SDL_SCANCODE_F3: out_dik = d::kF3; out_vk = 0x72U; return true;
    case SDL_SCANCODE_F4: out_dik = d::kF4; out_vk = 0x73U; return true;
    case SDL_SCANCODE_F5: out_dik = d::kF5; out_vk = 0x74U; return true;
    case SDL_SCANCODE_F6: out_dik = d::kF6; out_vk = 0x75U; return true;
    case SDL_SCANCODE_F7: out_dik = d::kF7; out_vk = 0x76U; return true;
    case SDL_SCANCODE_F8: out_dik = d::kF8; out_vk = 0x77U; return true;
    case SDL_SCANCODE_F9: out_dik = d::kF9; out_vk = 0x78U; return true;
    case SDL_SCANCODE_F10: out_dik = d::kF10; out_vk = 0x79U; return true;
    case SDL_SCANCODE_F11: out_dik = d::kF11; out_vk = 0x7AU; return true;
    case SDL_SCANCODE_F12: out_dik = d::kF12; out_vk = 0x7BU; return true;
    case SDL_SCANCODE_HOME: out_dik = d::kHome; out_vk = 0x24U; return true;
    case SDL_SCANCODE_UP: out_dik = d::kUp; out_vk = 0x26U; return true;
    case SDL_SCANCODE_PAGEUP: out_dik = d::kPgUp; out_vk = 0x21U; return true;
    case SDL_SCANCODE_LEFT: out_dik = d::kLeft; out_vk = 0x25U; return true;
    case SDL_SCANCODE_RIGHT: out_dik = d::kRight; out_vk = 0x27U; return true;
    case SDL_SCANCODE_END: out_dik = d::kEnd; out_vk = 0x23U; return true;
    case SDL_SCANCODE_DOWN: out_dik = d::kDown; out_vk = 0x28U; return true;
    case SDL_SCANCODE_PAGEDOWN: out_dik = d::kPgDown; out_vk = 0x22U; return true;
    case SDL_SCANCODE_INSERT: out_dik = d::kInsert; out_vk = 0x2DU; return true;
    case SDL_SCANCODE_DELETE: out_dik = d::kDelete; out_vk = 0x2EU; return true;
    default: return false;
    }
}
#endif

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
    // `skip`: names (lower case, relative to the folder) that a higher-priority folder supplies.
    auto add_dir = [&](const fs::path& dir, const std::set<std::string>& skip, std::set<std::string>* names) {
        std::error_code ec;
        if (!fs::is_directory(dir, ec)) return;
        for (fs::recursive_directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
            if (to_lower_ascii(it->path().extension().string()) != ".dds" || !it->is_regular_file(ec)) continue;
            const std::string rel = to_lower_ascii(fs::relative(it->path(), dir, ec).generic_string());
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
                     "[--reuse] [--quiet] [--module <name>] [--frames <n>] [--for-phone <dir>] "
                     "[--capture <file> [--capture-frames <n>]] [--screen <w>x<h>] "
                     "[--vulkan] [--icd <auto|lavapipe|system|path>] [--window] "
                     "[--audio <capture|sdl|none>] [--script <file>] "
                     "[--dump-frames <dir>] [--dump-every <n>] [--log-imports <file>]\n",
                     argv[0]);
        return 2;
    }
    const fs::path install = argv[1];
    fs::path work = "/root/wse/r4";
    wse::recomp::StartupOptions o;
    fs::path for_phone;
    bool no_texcache = false;
    std::string capture;
    std::uint64_t capture_frames = 60;
    bool want_vulkan = false;
    bool want_window = false;
    std::string icd_arg = "auto";
    std::string audio_arg;
    std::string script_path;
    std::string record_session_path;
    std::string replay_session_path;
    fs::path dump_frames_dir;
    std::uint64_t dump_every = 0U;
    fs::path log_imports_path;

    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--work" && i + 1 < argc) work = argv[++i];
        else if (a == "--retranslate") o.retranslate = true;
        else if (a == "--batch" && i + 1 < argc) o.batch_kib = std::strtoul(argv[++i], nullptr, 10);
        else if (a == "--opt" && i + 1 < argc) o.opt = std::atoi(argv[++i]);
        else if (a == "--cold-opt" && i + 1 < argc) o.cold_opt = std::atoi(argv[++i]);
        else if (a == "--all-o2") { o.opt = 2; o.cold_opt = 2; }
        else if (a == "--jobs" && i + 1 < argc) o.jobs = static_cast<unsigned>(std::max(1, std::atoi(argv[++i])));
        else if (a == "--reuse") o.reuse = true;
        else if (a == "--quiet") o.quiet = true;
        else if (a == "--strict-tso") o.strict_tso = true;
        else if (a == "--profile-in" && i + 1 < argc) o.profile_in = argv[++i];
        else if (a == "--profile-out" && i + 1 < argc) o.profile_out = argv[++i];
        else if (a == "--hot-opt" && i + 1 < argc) o.pgo_hot_opt = std::atoi(argv[++i]);
        else if (a == "--module" && i + 1 < argc) o.module = argv[++i];
        else if (a == "--frames" && i + 1 < argc) o.frame_limit = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--for-phone" && i + 1 < argc) for_phone = argv[++i];
        else if (a == "--no-texcache") no_texcache = true;
        else if (a == "--capture" && i + 1 < argc) capture = argv[++i];
        else if (a == "--capture-frames" && i + 1 < argc) capture_frames = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--screen" && i + 1 < argc) {  // e.g. 2412x1080: the display the game sees
            int sw = 0, sh = 0;
            if (std::sscanf(argv[++i], "%dx%d", &sw, &sh) == 2) wse::recomp::shim::set_screen_size(sw, sh);
        } else if (a == "--vulkan") want_vulkan = true;
        else if (a == "--icd" && i + 1 < argc) icd_arg = argv[++i];
        else if (a == "--window") {
            want_window = true;
            want_vulkan = true;
        } else if (a == "--audio" && i + 1 < argc) audio_arg = to_lower_ascii(argv[++i]);
        else if (a == "--script" && i + 1 < argc) script_path = argv[++i];
        else if (a == "--record-session" && i + 1 < argc) record_session_path = argv[++i];
        else if (a == "--replay-session" && i + 1 < argc) replay_session_path = argv[++i];
        else if (a == "--dump-frames" && i + 1 < argc) dump_frames_dir = argv[++i];
        else if (a == "--dump-every" && i + 1 < argc) dump_every = std::strtoull(argv[++i], nullptr, 10);
        else if (a == "--log-imports" && i + 1 < argc) log_imports_path = argv[++i];
    }

    if (audio_arg.empty()) {
        if (want_window) {
#ifdef WSE_HAS_SDL3
            audio_arg = "sdl";
#else
            audio_arg = "capture";
#endif
        } else if (want_vulkan) {
            audio_arg = "capture";
        } else {
            audio_arg = "none";
        }
    }

#ifndef WSE_HAS_SDL3
    if (want_window) {
        std::fprintf(stderr,
                     "r4_run: --window requires SDL3 (libsdl3-dev was not present at CMake configure time)\n");
        return 2;
    }
#endif

    std::error_code ec;
    fs::create_directories(work / "user", ec);
    if (!dump_frames_dir.empty()) fs::create_directories(dump_frames_dir, ec);
    setvbuf(stdout, nullptr, _IOLBF, 0);

    o.exe = read_file(find_ci(install, "mb_warband_wse2_x64.exe"));
    o.lua = read_file(find_ci(install, "lua5164.dll"));
    o.pdb = read_file(find_ci(install, "mb_warband_wse2_x64.pdb"));
    o.so_dir = (work / "so").string();
    if (o.crash_report_path.empty()) o.crash_report_path = (work / "crash_report.txt").string();
    if (!for_phone.empty()) {
        wse::recomp::set_code_target(wse::recomp::portable_android_target());
        o.translate_only = true;
        o.so_dir = for_phone.string();
    }

    if (!log_imports_path.empty()) {
        std::string err;
        if (!dump_pe_imports_file(log_imports_path, o.exe, o.lua, err)) {
            std::fprintf(stderr, "r4_run: %s\n", err.c_str());
            return 2;
        }
        std::printf("[r7] wrote PE imports to %s\n", log_imports_path.c_str());
    }

    // Resolve g_gameScreen VA from the PDB so we can monitor active window slot transitions.
    std::uint64_t game_screen_va = 0U;
    if (!o.exe.empty() && !o.pdb.empty()) {
        wse::recomp::PeImage pe;
        std::string pe_err;
        if (wse::recomp::parse_pe(o.exe.data(), o.exe.size(), pe, &pe_err)) {
            std::vector<wse::recomp::PdbPublic> publics;
            std::string pdb_err;
            if (wse::recomp::read_pdb_publics(o.pdb.data(), o.pdb.size(), publics, nullptr, &pdb_err)) {
                if (const auto va = wse::recomp::pdb_public_va(publics, "?g_gameScreen@@3VmbGameScreen@@A",
                                                               pe.sections, pe.image_base)) {
                    game_screen_va = *va;
                    std::printf("[r7] g_gameScreen VA = 0x%llx\n",
                                static_cast<unsigned long long>(game_screen_va));
                }
            }
        }
    }

    static wse::recomp::shim::ScriptedInputPlayer script_player;
    if (!script_path.empty()) {
        std::string err;
        if (!script_player.load_file(script_path, &err)) {
            std::fprintf(stderr, "r4_run: script load failed: %s\n", err.c_str());
            return 2;
        }
        std::printf("[r7] loaded script %s (%zu steps)\n", script_path.c_str(),
                    script_player.step_count());
    }
    if (!record_session_path.empty()) {
        if (!wse::recomp::shim::SessionLog::shared().start_record(record_session_path)) {
            std::fprintf(stderr, "r4_run: cannot open --record-session %s\n", record_session_path.c_str());
            return 2;
        }
        std::printf("[r8] recording deterministic session to %s\n", record_session_path.c_str());
    } else if (!replay_session_path.empty()) {
        if (!wse::recomp::shim::SessionLog::shared().start_replay(replay_session_path)) {
            std::fprintf(stderr, "r4_run: cannot load --replay-session %s\n", replay_session_path.c_str());
            return 2;
        }
        std::printf("[r8] replaying deterministic session from %s\n", replay_session_path.c_str());
    }

    // Optional Vulkan RHI backend for live D3D9 rendering in WSL (Lavapipe or WSLg system ICD).
    static std::shared_ptr<wse::platform::VulkanDriverLoader> vk_loader;
    static std::shared_ptr<wse::platform::VulkanD3d9Backend> vk_backend;
    if (for_phone.empty() && want_vulkan) {
        wse::platform::VulkanDriverProbeResult probe;
        std::string vk_err;
        if (!init_vulkan_loader(icd_arg, work, vk_loader, probe, vk_err)) {
            std::fprintf(stderr, "r4_run: Vulkan init failed (--icd %s): %s\n", icd_arg.c_str(),
                         vk_err.c_str());
            return 2;
        }
        std::printf("[r7] Vulkan ICD: device=\"%s\" api=%s driver=0x%x bc=%s loaded=%s\n",
                    probe.device_name.c_str(),
                    wse::platform::format_vulkan_version(probe.api_version).c_str(),
                    probe.driver_version,
                    probe.texture_compression_bc ? "yes" : "no",
                    probe.loaded_library.c_str());

        wse::platform::VulkanD3d9Config cfg{};
        cfg.pipeline_cache_path = (work / "r7_pipeline_cache.bin").string();
        cfg.prefer_bc_compression = probe.texture_compression_bc;
        cfg.prefer_etc2 = false;
        cfg.render_scale = 1.0F;
        vk_backend = std::make_shared<wse::platform::VulkanD3d9Backend>(vk_loader, std::move(cfg));
    }

    // Optional FMOD -> AudioDevice routing (`capture` for automated RMS checks, `sdl` for live window).
    static std::unique_ptr<StreamingCaptureAudioDevice> capture_audio;
    static std::unique_ptr<wse::platform::AudioDevice> sdl_audio;
    static wse::platform::AudioDevice* active_audio = nullptr;
    if (for_phone.empty() && audio_arg != "none") {
        wse::platform::AudioDeviceSpec spec{48000U, 2U};
        if (audio_arg == "sdl") {
#ifdef WSE_HAS_SDL3
            sdl_audio = wse::platform::create_sdl_audio_device();
            if (sdl_audio && sdl_audio->open(spec)) {
                active_audio = sdl_audio.get();
                std::printf("[r7] Audio output: SDL3 (%u Hz, %uch)\n", spec.sample_rate, spec.channels);
            } else {
                std::fprintf(stderr,
                             "[r7] warning: SDL3 audio open failed; falling back to CaptureAudioDevice\n");
                sdl_audio.reset();
                audio_arg = "capture";
            }
#else
            std::fprintf(stderr,
                         "[r7] warning: --audio sdl requested without SDL3; falling back to capture\n");
            audio_arg = "capture";
#endif
        }
        if (audio_arg == "capture") {
            capture_audio = std::make_unique<StreamingCaptureAudioDevice>();
            if (capture_audio->open(spec)) {
                active_audio = capture_audio.get();
                std::printf("[r7] Audio output: StreamingCaptureAudioDevice (%u Hz, %uch)\n",
                            spec.sample_rate, spec.channels);
            }
        }
    }

    static wse::platform::HostVfs install_vfs(install);
    static wse::platform::HostVfs user_vfs(work / "user");
    o.install = &install_vfs;
    o.user = &user_vfs;

    auto& hub = wse::recomp::shim::RecompInputHub::shared();
    const auto sc = wse::recomp::shim::screen_size();
    hub.set_screen_size(sc.width, sc.height);
    hub.set_surface_size(sc.width, sc.height);

    static std::mutex fmod_m;
    static wse::recomp::shim::FmodSteam* active_fmod = nullptr;
    auto stop_and_summarize_audio = []() {
        std::size_t sounds_n = 0U;
        std::size_t active_ch = 0U;
        {
            const std::lock_guard<std::mutex> lock(fmod_m);
            if (active_fmod != nullptr) {
                sounds_n = active_fmod->sounds_created();
                active_ch = active_fmod->active_channel_count();
                active_fmod->stop_worker();
                active_fmod = nullptr;
            }
        }
        if (capture_audio && capture_audio->is_open()) {
            const auto st = capture_audio->stats();
            std::printf(
                "[r7] audio: sounds_created=%zu active_ch=%zu captured_frames=%zu nonzero_frames=%zu rms_L=%.6f rms_R=%.6f peak=%.4f hash=0x%016llx\n",
                sounds_n, active_ch, st.frames, st.nonzero_frames, st.rms_left, st.rms_right,
                st.peak, static_cast<unsigned long long>(st.hash));
        } else {
            std::printf("[r7] audio: sounds_created=%zu active_ch=%zu\n", sounds_n, active_ch);
        }
        if (vk_backend) vk_backend->flush_pipeline_cache();
        wse::recomp::shim::SessionLog::shared().stop();
        std::fflush(stdout);
    };

    // Capture writer (set up before startup, preserving translator_fingerprint).
    static std::unique_ptr<wse::recomp::d3dcap::Writer> writer;
    if (!capture.empty()) {
        writer = std::make_unique<wse::recomp::d3dcap::Writer>(capture);
        if (!writer->ok()) {
            std::fprintf(stderr, "cannot write the capture %s\n", capture.c_str());
            return 2;
        }
    }

#ifdef WSE_HAS_SDL3
    static FrameMailbox mailbox;
    SDL_Window* sdl_win = nullptr;
    SDL_Renderer* sdl_ren = nullptr;
    SDL_Texture* sdl_tex = nullptr;
    if (for_phone.empty() && want_window) {
        if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD)) {
            std::fprintf(stderr, "r4_run: SDL_Init failed: %s\n", SDL_GetError());
            return 2;
        }
        sdl_win = SDL_CreateWindow("WSE2 Recomp (R7 WSL)", sc.width, sc.height, SDL_WINDOW_RESIZABLE);
        if (sdl_win == nullptr) {
            std::fprintf(stderr, "r4_run: SDL_CreateWindow failed: %s\n", SDL_GetError());
            SDL_Quit();
            return 2;
        }
        sdl_ren = SDL_CreateRenderer(sdl_win, nullptr);
        if (sdl_ren == nullptr) {
            std::fprintf(stderr, "r4_run: SDL_CreateRenderer failed: %s\n", SDL_GetError());
            SDL_DestroyWindow(sdl_win);
            SDL_Quit();
            return 2;
        }
        SDL_StartTextInput(sdl_win);
        std::printf("[r7] SDL3 live window open (%dx%d)\n", sc.width, sc.height);
    }
#endif

    if (for_phone.empty()) {
        wse::recomp::shim::User32::created_hook() = [&hub](wse::recomp::shim::User32& u) {
            u.input_hub = &hub;
            hub.set_window_msg_sink([&u](std::uint32_t msg, std::uint64_t wp, std::uint64_t lp) {
                (void)u.post_to_main(msg, wp, lp);
            });
        };
        wse::recomp::shim::SystemDlls::created_hook() = [&hub](wse::recomp::shim::SystemDlls& sys) {
            sys.input_hub = &hub;
        };
        wse::recomp::shim::FmodSteam::created_hook() = [](wse::recomp::shim::FmodSteam& fmod) {
            fmod.set_codec_backend(wse::asset::create_audio_codec_backend());
            fmod.read_file = [](const std::string& path, std::vector<std::uint8_t>& out) {
                return read_vfs_guest_path(&install_vfs, &user_vfs, path, out);
            };
            if (active_audio != nullptr) {
                fmod.set_audio_device(active_audio, true);
            }
            const std::lock_guard<std::mutex> lock(fmod_m);
            active_fmod = &fmod;
        };

        const std::uint64_t frame_limit = o.frame_limit;
        wse::recomp::shim::D3d9::created_hook() =
            [&hub, capture_frames, want_window, dump_frames_dir, dump_every, game_screen_va,
             frame_limit, stop_and_summarize_audio](wse::recomp::shim::D3d9& d) {
                d.input_hub = &hub;
                if (vk_backend) d.backend = vk_backend.get();
                if (writer) {
                    d.on_call = [&d, capture_frames](const wse::recomp::d3dcap::Call& c) {
                        if (d.frames() <= capture_frames) writer->write(c);
                    };
                }
                const bool need_readback = want_window || !dump_frames_dir.empty();
                if (need_readback) {
                    d.should_readback = [want_window, dump_frames_dir, dump_every](std::uint64_t frame) {
                        if (want_window) return true;
                        if (dump_frames_dir.empty()) return false;
                        if (script_player.has_pending_dump()) return true;
                        return dump_every > 0U && (frame % dump_every) == 0U;
                    };
                    d.on_present_rgba = [want_window, dump_frames_dir, dump_every](
                                            std::uint64_t frame,
                                            std::span<const std::uint8_t> rgba,
                                            std::uint32_t w,
                                            std::uint32_t h) {
                        if (!dump_frames_dir.empty()) {
                            const std::string tag = script_player.take_pending_dump_tag();
                            if (!tag.empty()) {
                                const fs::path out_ppm = dump_frames_dir / ("frame_" + tag + ".ppm");
                                if (write_ppm_rgba(out_ppm, rgba, w, h)) {
                                    std::printf("[r7] saved %s (%ux%u, frame=%llu)\n",
                                                out_ppm.string().c_str(), w, h,
                                                static_cast<unsigned long long>(frame));
                                }
                            }
                            if (dump_every > 0U && (frame % dump_every) == 0U) {
                                char name[64];
                                std::snprintf(name, sizeof(name), "frame_%06llu.ppm",
                                              static_cast<unsigned long long>(frame));
                                const fs::path out_ppm = dump_frames_dir / name;
                                if (write_ppm_rgba(out_ppm, rgba, w, h)) {
                                    std::printf("[r7] saved %s (%ux%u)\n",
                                                out_ppm.string().c_str(), w, h);
                                }
                            }
                        }
#ifdef WSE_HAS_SDL3
                        if (want_window) {
                            const std::lock_guard<std::mutex> lock(mailbox.m);
                            mailbox.rgba.assign(rgba.begin(), rgba.end());
                            mailbox.width = w;
                            mailbox.height = h;
                            ++mailbox.seq;
                        }
#else
                        (void)want_window;
#endif
                    };
                }

                d.on_after_present = [game_screen_va, frame_limit, &hub,
                                      stop_and_summarize_audio](std::uint64_t frames) {
                    static std::int32_t last_win = -999;
                    std::int32_t cur_win = -1;
                    if (game_screen_va >= 0x10000000ULL) {
                        cur_win = wse::recomp::shim::top_open_window(
                            reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(game_screen_va)));
                    }
                    if (cur_win != last_win) {
                        last_win = cur_win;
                        const auto sc_kind = wse::recomp::shim::scene_for_open_window(cur_win);
                        std::printf("[r7] frame=%llu win=%d (%s) cursor=%d\n",
                                    static_cast<unsigned long long>(frames), cur_win,
                                    scene_name(sc_kind), hub.cursor_visible() ? 1 : 0);
                    }

                    auto& slog = wse::recomp::shim::SessionLog::shared();
                    if (slog.mode() == wse::recomp::shim::SessionMode::record) {
                        slog.record_frame_mark(frames);
                    } else if (slog.mode() == wse::recomp::shim::SessionMode::replay) {
                        for (const auto& ev : slog.pop_replay_inputs_for_frame(frames)) {
                            using Tag = wse::recomp::shim::SessionEventTag;
                            switch (ev.tag) {
                            case Tag::input_key: hub.inject_key(static_cast<std::uint8_t>(ev.a), ev.b != 0); break;
                            case Tag::input_mouse_abs: hub.inject_mouse_move_abs(ev.a, ev.b); break;
                            case Tag::input_mouse_delta: hub.inject_mouse_delta(ev.a, ev.b); break;
                            case Tag::input_mouse_btn: hub.inject_mouse_button(ev.a, ev.b != 0); break;
                            case Tag::input_mouse_wheel: hub.inject_mouse_wheel(ev.a); break;
                            case Tag::input_touch: hub.on_touch_event(ev.a, ev.b, ev.fx, ev.fy); break;
                            default: break;
                            }
                        }
                    }

                    if (!script_player.empty()) {
                        std::string log_line;
                        bool stop_now = false;
                        script_player.step_frame(hub, cur_win, log_line, stop_now);
                        if (!log_line.empty()) {
                            std::printf("[script frame=%llu win=%d] %s\n",
                                        static_cast<unsigned long long>(frames), cur_win,
                                        log_line.c_str());
                        }
                        if (stop_now) {
                            stop_and_summarize_audio();
                            if (script_player.timed_out()) {
                                std::fprintf(stderr,
                                             "RESULT fail script timeout waiting for window %d at frame %llu (cur win=%d)\n",
                                             script_player.timed_out_window(),
                                             static_cast<unsigned long long>(frames), cur_win);
                                std::_Exit(wse::recomp::kStartupFault);
                            }
                            std::printf("RESULT pass script %llu frames\n",
                                        static_cast<unsigned long long>(frames));
                            std::_Exit(wse::recomp::kStartupFrameLimit);
                        }
                    }

                    if (frame_limit != 0U && frames >= frame_limit) {
                        stop_and_summarize_audio();
                    }
                };
            };
    }

    // On a host thread with the same stack as the phone's game thread (`wse/recomp/host_stack.h`).
    int rc = 0;
    std::atomic<bool> game_done{false};
    wse::recomp::HostThread game([&] {
        rc = wse::recomp::run_startup(o);
        game_done.store(true, std::memory_order_release);
    });

#ifdef WSE_HAS_SDL3
    if (for_phone.empty() && want_window && sdl_win != nullptr && sdl_ren != nullptr) {
        std::uint64_t drawn_seq = 0U;
        std::uint32_t tex_w = 0U, tex_h = 0U;
        std::vector<std::uint8_t> upload_buf;
        bool last_rel_mouse = false;
        bool user_quit = false;

        while (!game_done.load(std::memory_order_acquire)) {
            SDL_Event ev;
            while (SDL_PollEvent(&ev)) {
                if (ev.type == SDL_EVENT_QUIT) {
                    user_quit = true;
                    break;
                }
                if (ev.type == SDL_EVENT_KEY_DOWN || ev.type == SDL_EVENT_KEY_UP) {
                    if (ev.key.repeat) continue;
                    std::uint8_t dik_code = 0U;
                    std::uint32_t vk_code = 0U;
                    if (sdl_scancode_to_dik(ev.key.scancode, dik_code, vk_code)) {
                        hub.inject_key(dik_code, ev.type == SDL_EVENT_KEY_DOWN, vk_code);
                    }
                } else if (ev.type == SDL_EVENT_TEXT_INPUT) {
                    if (ev.text.text != nullptr) hub.inject_text_utf8(ev.text.text);
                } else if (ev.type == SDL_EVENT_MOUSE_MOTION) {
                    int ww = sc.width, wh = sc.height;
                    SDL_GetWindowSize(sdl_win, &ww, &wh);
                    if (!hub.cursor_visible()) {
                        hub.inject_mouse_delta(static_cast<int>(ev.motion.xrel),
                                               static_cast<int>(ev.motion.yrel));
                    } else {
                        const int gx = static_cast<int>(ev.motion.x * static_cast<float>(sc.width) /
                                                        static_cast<float>(std::max(1, ww)));
                        const int gy = static_cast<int>(ev.motion.y * static_cast<float>(sc.height) /
                                                        static_cast<float>(std::max(1, wh)));
                        hub.inject_mouse_move_abs(gx, gy);
                    }
                } else if (ev.type == SDL_EVENT_MOUSE_BUTTON_DOWN || ev.type == SDL_EVENT_MOUSE_BUTTON_UP) {
                    int btn = -1;
                    if (ev.button.button == SDL_BUTTON_LEFT) btn = 0;
                    else if (ev.button.button == SDL_BUTTON_RIGHT) btn = 1;
                    else if (ev.button.button == SDL_BUTTON_MIDDLE) btn = 2;
                    if (btn >= 0) {
                        hub.inject_mouse_button(btn, ev.type == SDL_EVENT_MOUSE_BUTTON_DOWN);
                    }
                } else if (ev.type == SDL_EVENT_MOUSE_WHEEL) {
                    hub.inject_mouse_wheel(static_cast<int>(ev.wheel.y * 120.0F));
                }
            }
            if (user_quit) break;

            const bool want_rel = !hub.cursor_visible();
            if (want_rel != last_rel_mouse) {
                SDL_SetWindowRelativeMouseMode(sdl_win, want_rel);
                last_rel_mouse = want_rel;
            }

            std::uint64_t cur_seq = 0U;
            std::uint32_t w = 0U, h = 0U;
            {
                const std::lock_guard<std::mutex> lock(mailbox.m);
                cur_seq = mailbox.seq;
                if (cur_seq != drawn_seq && !mailbox.rgba.empty()) {
                    upload_buf = mailbox.rgba;
                    w = mailbox.width;
                    h = mailbox.height;
                }
            }
            if (cur_seq != drawn_seq && w > 0U && h > 0U) {
                if (sdl_tex == nullptr || tex_w != w || tex_h != h) {
                    if (sdl_tex != nullptr) SDL_DestroyTexture(sdl_tex);
                    sdl_tex = SDL_CreateTexture(sdl_ren, SDL_PIXELFORMAT_RGBA32,
                                                SDL_TEXTUREACCESS_STREAMING,
                                                static_cast<int>(w), static_cast<int>(h));
                    tex_w = w;
                    tex_h = h;
                }
                if (sdl_tex != nullptr) {
                    SDL_UpdateTexture(sdl_tex, nullptr, upload_buf.data(), static_cast<int>(w * 4U));
                    SDL_RenderClear(sdl_ren);
                    SDL_RenderTexture(sdl_ren, sdl_tex, nullptr, nullptr);
                    SDL_RenderPresent(sdl_ren);
                }
                drawn_seq = cur_seq;
            } else {
                SDL_Delay(4);
            }
        }

        if (sdl_tex != nullptr) SDL_DestroyTexture(sdl_tex);
        SDL_StopTextInput(sdl_win);
        SDL_DestroyRenderer(sdl_ren);
        SDL_DestroyWindow(sdl_win);
        if (user_quit) {
            stop_and_summarize_audio();
            sdl_audio.reset();
            SDL_Quit();
            std::printf("[r7] SDL3 window closed by user\n");
            std::_Exit(0);
        }
        SDL_Quit();
    }
#endif

    game.join();
    if (!for_phone.empty() && !no_texcache && rc == 0) {
        // Converted textures go in a subfolder; the phone puts them in its texture cache.
        (void)convert_textures_for_phone(install, o.module, for_phone / "texcache", std::max(1U, o.jobs));
    } else if (for_phone.empty()) {
        stop_and_summarize_audio();
    }
    return rc;
}
