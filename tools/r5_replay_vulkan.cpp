// Phase 5 R5a step 4 (`docs/RECOMP_PLAN.md` "R5 — Graphics"): replays a D3D9 capture
// (`r4_run --capture`, `wse/recomp/d3d9_capture.h`) onto `VulkanD3d9Backend` (System Vulkan,
// PanVK, Turnip, or Mock ICD), writes chosen frames as P6 PPMs, and optionally compares them
// pixel-by-pixel against the real-D3D9 reference frames produced by `tools/r5_replay_d3d9.cpp`.
//
// Portable C++20 (Linux / Windows). CI compiles it so it stays in sync, and never runs it on
// retail files.
//
//   r5_replay_vulkan <capture> <install dir> <out dir>
//       [--frames 250,1000,1250,3000]
//       [--ref <d3d9_ref_dir>]
//       [--save-diff]
//       [--no-bc]
//       [--max-tex <px>]
//       [--pipeline-cache <path>]
//       [--driver system|mock|panvk|turnip]
//       [--driver-path <so_or_dll>]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#include <string>
#include <vector>

#include "d3d9_vulkan.h"
#include "driver_loader.h"
#include "wse/recomp/d3d9_backend.h"
#include "wse/recomp/d3d9_capture.h"

namespace fs = std::filesystem;
namespace cap = wse::recomp::d3dcap;

namespace {

bool read_all_bytes(const fs::path& p, std::vector<std::uint8_t>& out) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    const std::streamoff sz = f.tellg();
    if (sz < 0) return false;
    f.seekg(0, std::ios::beg);
    out.resize(static_cast<std::size_t>(sz));
    if (sz > 0) f.read(reinterpret_cast<char*>(out.data()), sz);
    return static_cast<bool>(f);
}

std::string to_lower_ascii(std::string s) {
    for (char& ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    return s;
}

// Resolves a Windows-style relative path (with `\` separators and arbitrary casing) under `root`.
fs::path resolve_install_path(const fs::path& root, const std::string& raw) {
    std::string norm = raw;
    for (char& ch : norm) {
        if (ch == '\\') ch = '/';
    }
    while (!norm.empty() && (norm.front() == '/' || norm.front() == '.')) {
        if (norm.size() >= 2 && norm[0] == '.' && norm[1] == '/') {
            norm.erase(0, 2);
        } else if (norm[0] == '/') {
            norm.erase(0, 1);
        } else {
            break;
        }
    }
    fs::path direct = root / norm;
    std::error_code ec;
    if (fs::exists(direct, ec)) return direct;

    // Walk component-by-component case-insensitively (needed on Linux/ext4).
    fs::path cur = root;
    std::size_t pos = 0;
    while (pos < norm.size()) {
        const std::size_t slash = norm.find('/', pos);
        const std::string comp = norm.substr(pos, slash == std::string::npos ? std::string::npos : slash - pos);
        pos = (slash == std::string::npos) ? norm.size() : slash + 1;
        if (comp.empty() || comp == ".") continue;
        fs::path next = cur / comp;
        if (fs::exists(next, ec)) {
            cur = std::move(next);
            continue;
        }
        const std::string want = to_lower_ascii(comp);
        bool found = false;
        if (fs::is_directory(cur, ec)) {
            for (const fs::directory_entry& de : fs::directory_iterator(cur, ec)) {
                if (to_lower_ascii(de.path().filename().string()) == want) {
                    cur = de.path();
                    found = true;
                    break;
                }
            }
        }
        if (!found) return direct;
    }
    return cur;
}

bool write_ppm_rgba(const fs::path& path, std::uint32_t w, std::uint32_t h,
                    const std::vector<std::uint8_t>& rgba) {
    if (rgba.size() < static_cast<std::size_t>(w) * h * 4U) return false;
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f << "P6\n" << w << " " << h << "\n255\n";
    std::vector<std::uint8_t> row(static_cast<std::size_t>(w) * 3U);
    for (std::uint32_t y = 0; y < h; ++y) {
        const std::uint8_t* src = rgba.data() + static_cast<std::size_t>(y) * w * 4U;
        for (std::uint32_t x = 0; x < w; ++x) {
            row[static_cast<std::size_t>(x) * 3U + 0U] = src[static_cast<std::size_t>(x) * 4U + 0U];
            row[static_cast<std::size_t>(x) * 3U + 1U] = src[static_cast<std::size_t>(x) * 4U + 1U];
            row[static_cast<std::size_t>(x) * 3U + 2U] = src[static_cast<std::size_t>(x) * 4U + 2U];
        }
        f.write(reinterpret_cast<const char*>(row.data()), static_cast<std::streamsize>(row.size()));
    }
    return static_cast<bool>(f);
}

bool read_ppm(const fs::path& path, std::uint32_t& w, std::uint32_t& h, std::vector<std::uint8_t>& rgb) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::string magic;
    std::uint32_t maxval = 0;
    f >> magic >> w >> h >> maxval;
    if (!f || magic != "P6" || maxval != 255U || w == 0 || h == 0) return false;
    char ws = 0;
    f.get(ws);
    const std::size_t n = static_cast<std::size_t>(w) * h * 3U;
    rgb.resize(n);
    f.read(reinterpret_cast<char*>(rgb.data()), static_cast<std::streamsize>(n));
    return static_cast<bool>(f);
}

struct DiffStats {
    bool ok = false;
    double mean_abs = 0.0;
    double rmse = 0.0;
    double psnr_db = 0.0;
    int max_diff = 0;
    double pct_le_2 = 0.0;
    double pct_le_4 = 0.0;
    double pct_le_8 = 0.0;
    double pct_le_16 = 0.0;
};

DiffStats compare_frame_with_ref(const fs::path& ref_path, std::uint32_t w, std::uint32_t h,
                                 const std::vector<std::uint8_t>& rgba,
                                 const fs::path* diff_out_path) {
    DiffStats st{};
    std::uint32_t rw = 0, rh = 0;
    std::vector<std::uint8_t> ref_rgb;
    if (!read_ppm(ref_path, rw, rh, ref_rgb) || rw != w || rh != h) return st;

    const std::size_t npix = static_cast<std::size_t>(w) * h;
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    std::size_t le2 = 0, le4 = 0, le8 = 0, le16 = 0;
    std::vector<std::uint8_t> diff_rgba;
    if (diff_out_path != nullptr) diff_rgba.resize(npix * 4U, 255U);

    for (std::size_t i = 0; i < npix; ++i) {
        const int dr = std::abs(static_cast<int>(rgba[i * 4U + 0U]) - static_cast<int>(ref_rgb[i * 3U + 0U]));
        const int dg = std::abs(static_cast<int>(rgba[i * 4U + 1U]) - static_cast<int>(ref_rgb[i * 3U + 1U]));
        const int db = std::abs(static_cast<int>(rgba[i * 4U + 2U]) - static_cast<int>(ref_rgb[i * 3U + 2U]));
        sum_abs += static_cast<double>(dr + dg + db);
        sum_sq += static_cast<double>(dr * dr + dg * dg + db * db);
        const int m = std::max({dr, dg, db});
        if (m > st.max_diff) st.max_diff = m;
        if (m <= 2) ++le2;
        if (m <= 4) ++le4;
        if (m <= 8) ++le8;
        if (m <= 16) ++le16;
        if (diff_out_path != nullptr) {
            diff_rgba[i * 4U + 0U] = static_cast<std::uint8_t>(std::min(255, dr * 8));
            diff_rgba[i * 4U + 1U] = static_cast<std::uint8_t>(std::min(255, dg * 8));
            diff_rgba[i * 4U + 2U] = static_cast<std::uint8_t>(std::min(255, db * 8));
        }
    }
    if (diff_out_path != nullptr) (void)write_ppm_rgba(*diff_out_path, w, h, diff_rgba);

    const double nch = static_cast<double>(npix) * 3.0;
    st.ok = true;
    st.mean_abs = sum_abs / nch;
    st.rmse = std::sqrt(sum_sq / nch);
    st.psnr_db = (st.rmse > 0.0) ? (20.0 * std::log10(255.0 / st.rmse)) : 99.99;
    st.pct_le_2 = 100.0 * static_cast<double>(le2) / static_cast<double>(npix);
    st.pct_le_4 = 100.0 * static_cast<double>(le4) / static_cast<double>(npix);
    st.pct_le_8 = 100.0 * static_cast<double>(le8) / static_cast<double>(npix);
    st.pct_le_16 = 100.0 * static_cast<double>(le16) / static_cast<double>(npix);
    return st;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
                     "usage: %s <capture> <install dir> <out dir> [--frames 250,1000,1250,3000] "
                     "[--ref <ref_dir>] [--save-diff] [--no-bc] [--max-tex <px>] "
                     "[--pipeline-cache <file>] [--driver system|mock|panvk|turnip] [--driver-path <path>] "
                     "[--screen <w>x<h>] [--render-scale <s>] [--auto-res] [--sharpen <x>] [--fxaa]\n",
                     argv[0]);
        return 2;
    }

    const fs::path capture_path = fs::absolute(argv[1]);
    const fs::path install_dir = fs::absolute(argv[2]);
    const fs::path out_dir = fs::absolute(argv[3]);

    std::set<std::uint64_t> save_frames;
    bool save_all_tenth = true;
    fs::path ref_dir;
    bool save_diff = false;
    bool no_bc = false;
    bool no_bgra_vtx = false;
    bool eager_shaders = false;
    std::uint32_t max_tex = 0;
    std::string pipeline_cache;
    std::string driver_kind = "system";
    std::string driver_path;
    float render_scale = 1.0F;
    bool auto_res = false;
    float sharpen = 0.0F;
    bool fxaa = false;

    for (int i = 4; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--frames" && i + 1 < argc) {
            save_all_tenth = false;
            const std::string list = argv[++i];
            for (std::size_t p = 0; p < list.size();) {
                const std::size_t q = list.find(',', p);
                save_frames.insert(std::strtoull(list.substr(p, q - p).c_str(), nullptr, 10));
                p = (q == std::string::npos) ? list.size() : q + 1;
            }
        } else if (arg == "--ref" && i + 1 < argc) {
            ref_dir = fs::absolute(argv[++i]);
        } else if (arg == "--save-diff") {
            save_diff = true;
        } else if (arg == "--no-bc") {
            no_bc = true;
        } else if (arg == "--no-bgra-vtx") {
            no_bgra_vtx = true;
        } else if (arg == "--eager-shaders") {
            eager_shaders = true;
        } else if (arg == "--max-tex" && i + 1 < argc) {
            max_tex = static_cast<std::uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (arg == "--pipeline-cache" && i + 1 < argc) {
            pipeline_cache = argv[++i];
        } else if (arg == "--driver" && i + 1 < argc) {
            driver_kind = argv[++i];
        } else if (arg == "--driver-path" && i + 1 < argc) {
            driver_path = argv[++i];
        } else if (arg == "--screen" && i + 1 < argc) {  // the r4_run --screen of the captured run
            int sw = 0, sh = 0;
            if (std::sscanf(argv[++i], "%dx%d", &sw, &sh) == 2) wse::recomp::shim::set_screen_size(sw, sh);
        } else if (arg == "--render-scale" && i + 1 < argc) {
            render_scale = std::strtof(argv[++i], nullptr);
        } else if (arg == "--auto-res") {
            auto_res = true;
        } else if (arg == "--sharpen" && i + 1 < argc) {
            sharpen = std::strtof(argv[++i], nullptr);
        } else if (arg == "--fxaa") {
            fxaa = true;
        }
    }

    std::vector<std::uint8_t> cap_bytes;
    if (!read_all_bytes(capture_path, cap_bytes)) {
        std::fprintf(stderr, "failed to read capture: %s\n", capture_path.string().c_str());
        return 1;
    }
    std::vector<cap::Call> calls;
    const bool whole = cap::decode(cap_bytes, calls);
    std::uint64_t total_presents = 0;
    for (const cap::Call& c : calls) {
        if (c.method == "Present") ++total_presents;
    }
    std::printf("%zu calls, %llu frames%s\n", calls.size(),
                static_cast<unsigned long long>(total_presents),
                whole ? "" : " (truncated tail skipped)");

    fs::create_directories(out_dir);

    auto loader = std::make_shared<wse::platform::VulkanDriverLoader>();
    std::string load_err;
    bool loaded = false;
    if (driver_kind == "mock") {
        wse::platform::VulkanMockConfig mc{};
        mc.mode = wse::platform::DriverMode::panvk;
        mc.device_name = "Mock PanVK (r5_replay_vulkan)";
        loaded = loader->load_mock_icd(mc);
    } else if (driver_kind == "panvk") {
        loaded = loader->load_custom_driver(wse::platform::DriverMode::panvk, driver_path, {}, &load_err);
    } else if (driver_kind == "turnip") {
        loaded = loader->load_custom_driver(wse::platform::DriverMode::turnip, driver_path, {}, &load_err);
    } else {
        loaded = loader->load_system(&load_err);
    }
    if (!loaded) {
        std::fprintf(stderr, "Vulkan driver load (%s) failed: %s\n", driver_kind.c_str(), load_err.c_str());
        return 1;
    }
    wse::platform::VulkanDriverProbeResult probe{};
    if (!loader->probe_and_create_device(probe) || !probe.ok) {
        std::fprintf(stderr, "Vulkan probe_and_create_device failed: %s\n", probe.error.c_str());
        return 1;
    }

    wse::platform::VulkanD3d9Config cfg{};
    cfg.prefer_bc_compression = !no_bc;
    cfg.prefer_bgra_vertex_format = !no_bgra_vtx;
    cfg.eager_shader_compile = eager_shaders;
    cfg.max_texture_size = max_tex;
    cfg.pipeline_cache_path = pipeline_cache;
    cfg.render_scale = std::clamp(render_scale, 0.5F, 1.0F);
    cfg.auto_resolution = auto_res;
    cfg.sharpness = std::max(0.0F, sharpen);
    cfg.fxaa = fxaa;
    wse::platform::VulkanD3d9Backend backend(loader, std::move(cfg));

    wse::recomp::D3d9CaptureReplayer replayer(backend);
    std::size_t missing_files = 0;
    replayer.read_file = [&](const std::string& rel, std::vector<std::uint8_t>& out) -> bool {
        const fs::path p = resolve_install_path(install_dir, rel);
        if (read_all_bytes(p, out)) return true;
        ++missing_files;
        return false;
    };
    replayer.want_frame = [&](std::uint64_t frame) -> bool {
        if (!save_all_tenth) return save_frames.count(frame) != 0;
        return (frame % 10ULL == 0ULL) || (frame == total_presents);
    };
    replayer.on_frame = [&](std::uint64_t frame, std::uint32_t w, std::uint32_t h,
                            const std::vector<std::uint8_t>& rgba) {
        char name[64];
        std::snprintf(name, sizeof(name), "frame_%04llu.ppm", static_cast<unsigned long long>(frame));
        const fs::path out_path = out_dir / name;
        if (!write_ppm_rgba(out_path, w, h, rgba)) {
            std::fprintf(stderr, "failed to write %s\n", out_path.string().c_str());
            return;
        }
        std::printf("saved %s (%ux%u)", name, w, h);
        if (!ref_dir.empty()) {
            const fs::path ref_path = ref_dir / name;
            char diff_name[64];
            std::snprintf(diff_name, sizeof(diff_name), "diff_%04llu.ppm", static_cast<unsigned long long>(frame));
            const fs::path diff_path = out_dir / diff_name;
            const DiffStats st = compare_frame_with_ref(ref_path, w, h, rgba, save_diff ? &diff_path : nullptr);
            if (st.ok) {
                std::printf(" | vs ref: mean_abs=%.2f rmse=%.2f psnr=%.2fdB max=%d <=2:%.2f%% <=4:%.2f%% <=8:%.2f%% <=16:%.2f%%",
                            st.mean_abs, st.rmse, st.psnr_db, st.max_diff,
                            st.pct_le_2, st.pct_le_4, st.pct_le_8, st.pct_le_16);
            } else {
                std::printf(" | ref missing or size mismatch: %s", ref_path.string().c_str());
            }
        }
        std::printf("\n");
        std::fflush(stdout);
    };

    const auto t0 = std::chrono::steady_clock::now();
    bool printed_report = false;
    std::string err;
    for (const cap::Call& c : calls) {
        if (!replayer.run_call(c, &err)) {
            std::fprintf(stderr, "replay error at call %zu (%s::%s): %s\n",
                         replayer.calls(), c.iface.c_str(), c.method.c_str(), err.c_str());
            return 1;
        }
        if (!printed_report && backend.report().ok) {
            std::printf("%s\n", backend.report().format_summary().c_str());
            std::fflush(stdout);
            printed_report = true;
        }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();

    std::printf("done: %zu calls, %llu frames, %zu draws, %zu shaders (%zu failed), %zu pipelines, %zu samplers, %zu desc_sets, %zu missing files, %.2fs\n",
                replayer.calls(), static_cast<unsigned long long>(replayer.frames()),
                replayer.draws(), backend.compiled_shader_count(), backend.failed_shader_count(),
                backend.cached_pipeline_count(), backend.cached_sampler_count(),
                backend.cached_descriptor_set_count(), missing_files, secs);
    if (!backend.last_error().empty()) {
        std::printf("last backend note: %s\n", backend.last_error().c_str());
    }
    return 0;
}
