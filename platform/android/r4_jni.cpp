// Phase 5 R4 startup run and R5a live Vulkan run on the phone (`docs/RECOMP_PLAN.md` "R4 design",
// "R5 — Graphics"): JNI entries on `R4Native` called inside the isolated `:recomp` process
// (`R4StartupService` for headless R4, `RecompGameActivity` for live R5a Vulkan presentation).
//
// Compiled into libwse_r0.so (with the translator and `wse_vulkan`) only when WSE_ENABLE_LLVM is ON.

#include <android/log.h>
#include <android/native_window.h>
#include <android/native_window_jni.h>
#include <dlfcn.h>
#include <jni.h>
#include <sys/auxv.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "../../engine/src/recomp/r4_startup.h"
#include "../../engine/src/recomp/recomp_llvm.h"
#include "d3d9_vulkan.h"
#include "driver_loader.h"
#include "saf_jni_util.h"
#include "wse/platform/rhi.h"
#include "wse/platform/saf_vfs.h"
#include "wse/platform/vfs.h"
#include "wse/recomp/shim_d3d9.h"

namespace {

// AArch64 Linux AT_HWCAP bits (asm/hwcap.h): LSE atomics and RCpc (LDAPR).
constexpr unsigned long kHwcapAtomics = 1UL << 8;
constexpr unsigned long kHwcapLrcpc = 1UL << 15;
// Frame at which Native reaches Warband's main menu (measured in R5a step 3).
constexpr std::uint64_t kMenuStartFrame = 1250U;

using wse::platform::android_jni::clear_if_exception;
using wse::platform::android_jni::jstring_to_utf8;
using wse::platform::android_jni::parse_child_record;
using wse::platform::android_jni::utf8_to_jstring;

constexpr const char* kTag = "WSE2-R4";

// The install folder through `WseSafBridge`'s instance methods, from any thread: each call gets
// the calling thread's JNIEnv, attaching it to the JVM the first time (guest threads are host
// threads; the run ends the process, so none is detached). Read-only: the run never writes the
// install (writes go to the writable layer). Callers serialize (`LockedVfs`).
class ThreadJniSafBackend final : public wse::platform::SafTreeBackend {
public:
    ThreadJniSafBackend(JavaVM* vm, JNIEnv* env, jobject bridge) : vm_(vm) {
        bridge_ = env->NewGlobalRef(bridge);
        jclass cls = env->GetObjectClass(bridge);
        mid_root_ = env->GetMethodID(cls, "rootDocumentId", "()Ljava/lang/String;");
        mid_list_ = env->GetMethodID(cls, "listChildren", "(Ljava/lang/String;)[Ljava/lang/String;");
        mid_read_ = env->GetMethodID(cls, "readDocument", "(Ljava/lang/String;)[B");
        env->DeleteLocalRef(cls);
        if (clear_if_exception(env)) mid_root_ = mid_list_ = mid_read_ = nullptr;
    }
    [[nodiscard]] bool valid() const noexcept {
        return bridge_ != nullptr && mid_root_ != nullptr && mid_list_ != nullptr && mid_read_ != nullptr;
    }

    [[nodiscard]] std::string root_document_id() const override {
        JNIEnv* env = this_env();
        if (env == nullptr || !valid()) return {};
        auto jroot = static_cast<jstring>(env->CallObjectMethod(bridge_, mid_root_));
        if (clear_if_exception(env) || jroot == nullptr) return {};
        std::string out = jstring_to_utf8(env, jroot);
        env->DeleteLocalRef(jroot);
        return out;
    }

    [[nodiscard]] bool list_children(std::string_view parent, std::vector<wse::platform::SafChildEntry>& out) override {
        out.clear();
        JNIEnv* env = this_env();
        if (env == nullptr || !valid()) return false;
        jstring jparent = utf8_to_jstring(env, parent);
        if (jparent == nullptr) return false;
        auto jarr = static_cast<jobjectArray>(env->CallObjectMethod(bridge_, mid_list_, jparent));
        env->DeleteLocalRef(jparent);
        if (clear_if_exception(env) || jarr == nullptr) return false;
        const jsize n = env->GetArrayLength(jarr);
        out.reserve(static_cast<std::size_t>(n));
        for (jsize i = 0; i < n; ++i) {
            auto jelem = static_cast<jstring>(env->GetObjectArrayElement(jarr, i));
            if (clear_if_exception(env)) {
                env->DeleteLocalRef(jarr);
                out.clear();
                return false;
            }
            if (jelem == nullptr) continue;
            const std::string line = jstring_to_utf8(env, jelem);
            env->DeleteLocalRef(jelem);
            wse::platform::SafChildEntry e{};
            if (parse_child_record(line, e)) out.push_back(std::move(e));
        }
        env->DeleteLocalRef(jarr);
        return true;
    }

    [[nodiscard]] bool read_document(std::string_view id, std::vector<std::uint8_t>& out) override {
        out.clear();
        JNIEnv* env = this_env();
        if (env == nullptr || !valid()) return false;
        jstring jid = utf8_to_jstring(env, id);
        if (jid == nullptr) return false;
        auto jbytes = static_cast<jbyteArray>(env->CallObjectMethod(bridge_, mid_read_, jid));
        env->DeleteLocalRef(jid);
        if (clear_if_exception(env) || jbytes == nullptr) return false;
        const jsize len = env->GetArrayLength(jbytes);
        out.resize(static_cast<std::size_t>(len));
        if (len > 0) env->GetByteArrayRegion(jbytes, 0, len, reinterpret_cast<jbyte*>(out.data()));
        env->DeleteLocalRef(jbytes);
        if (clear_if_exception(env)) {
            out.clear();
            return false;
        }
        return true;
    }

    // The install is read-only for the run.
    [[nodiscard]] bool write_document(std::string_view, std::string_view, std::string_view, std::span<const std::uint8_t>,
                                      std::string& out_id) override {
        out_id.clear();
        return false;
    }
    [[nodiscard]] bool create_directory(std::string_view, std::string_view, std::string& out_id) override {
        out_id.clear();
        return false;
    }
    [[nodiscard]] bool delete_document(std::string_view) override { return false; }

private:
    [[nodiscard]] JNIEnv* this_env() const {
        void* raw = nullptr;
        if (vm_->GetEnv(&raw, JNI_VERSION_1_6) == JNI_OK && raw != nullptr) return static_cast<JNIEnv*>(raw);
        JNIEnv* env = nullptr;
        return vm_->AttachCurrentThread(&env, nullptr) == JNI_OK ? env : nullptr;
    }

    JavaVM* vm_ = nullptr;
    jobject bridge_ = nullptr;
    jmethodID mid_root_ = nullptr;
    jmethodID mid_list_ = nullptr;
    jmethodID mid_read_ = nullptr;
};

// Install reads (count, bytes, wall time including the wait for the lock), for the load-time
// breakdown in the report's `fps:` lines.
std::atomic<std::uint64_t> g_file_ns{0}, g_file_count{0}, g_file_bytes{0};

// Serializes every call of a Vfs that is not thread-safe (`SafVfs`): guest threads read files
// concurrently (resource loading threads).
class LockedVfs final : public wse::platform::Vfs {
public:
    explicit LockedVfs(wse::platform::Vfs& inner) : inner_(inner) {}
    bool read_file(std::string_view p, std::vector<std::uint8_t>& out) override {
        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = locked([&] { return inner_.read_file(p, out); });
        g_file_ns += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
        ++g_file_count;
        g_file_bytes += out.size();
        return ok;
    }
    bool write_file(std::string_view p, std::span<const std::uint8_t> d) override { return locked([&] { return inner_.write_file(p, d); }); }
    bool remove_file(std::string_view p) override { return locked([&] { return inner_.remove_file(p); }); }
    bool list_dir(std::string_view p, std::vector<wse::platform::VfsEntry>& out) override {
        return locked([&] { return inner_.list_dir(p, out); });
    }
    bool is_file(std::string_view p) override { return locked([&] { return inner_.is_file(p); }); }
    bool is_dir(std::string_view p) override { return locked([&] { return inner_.is_dir(p); }); }
    bool make_dir(std::string_view p) override { return locked([&] { return inner_.make_dir(p); }); }
    bool remove_dir(std::string_view p) override { return locked([&] { return inner_.remove_dir(p); }); }
    bool rename_file(std::string_view f, std::string_view t) override { return locked([&] { return inner_.rename_file(f, t); }); }
    bool file_size(std::string_view p, std::uint64_t& size) override { return locked([&] { return inner_.file_size(p, size); }); }
    bool read_text(std::string_view p, std::string& out) override { return locked([&] { return inner_.read_text(p, out); }); }

private:
    template <typename F>
    bool locked(F&& f) {
        const std::lock_guard<std::mutex> lock(m_);
        return f();
    }
    wse::platform::Vfs& inner_;
    std::mutex m_;
};

// Optional HUD status bridge into `RecompGameActivity` (`R4Native.HudListener`).
class HudNotifier {
public:
    HudNotifier(JavaVM* vm, JNIEnv* env, jobject listener) : vm_(vm) {
        if (vm_ == nullptr || env == nullptr || listener == nullptr) return;
        listener_ = env->NewGlobalRef(listener);
        jclass cls = env->GetObjectClass(listener);
        mid_update_ = env->GetMethodID(cls, "onHudUpdate", "(Ljava/lang/String;)V");
        env->DeleteLocalRef(cls);
        if (clear_if_exception(env)) mid_update_ = nullptr;
    }

    void update(std::string_view text) const {
        if (vm_ == nullptr || listener_ == nullptr || mid_update_ == nullptr) return;
        JNIEnv* env = nullptr;
        void* raw = nullptr;
        if (vm_->GetEnv(&raw, JNI_VERSION_1_6) == JNI_OK && raw != nullptr) {
            env = static_cast<JNIEnv*>(raw);
        } else if (vm_->AttachCurrentThread(&env, nullptr) != JNI_OK || env == nullptr) {
            return;
        }
        jstring js = utf8_to_jstring(env, text);
        if (js != nullptr) {
            env->CallVoidMethod(listener_, mid_update_, js);
            env->DeleteLocalRef(js);
            (void)clear_if_exception(env);
        }
    }

private:
    JavaVM* vm_ = nullptr;
    jobject listener_ = nullptr;
    jmethodID mid_update_ = nullptr;
};

std::mutex g_report_m;
std::FILE* g_report = nullptr;
std::string g_progress_path;

void report(const std::string& text) {
    const std::lock_guard<std::mutex> lock(g_report_m);
    if (g_report != nullptr) {
        std::fputs(text.c_str(), g_report);
        std::fflush(g_report);
    }
    __android_log_write(ANDROID_LOG_INFO, kTag, text.c_str());
}

std::string format_fixed1(double v) {
    char buf[32]{};
    std::snprintf(buf, sizeof(buf), "%.1f", v);
    return std::string(buf);
}

std::string_view driver_label(wse::platform::DriverMode mode) noexcept {
    switch (mode) {
    case wse::platform::DriverMode::panvk: return "PanVK";
    case wse::platform::DriverMode::turnip: return "Turnip";
    case wse::platform::DriverMode::system:
    default: return "System Vulkan";
    }
}

std::string_view short_present_path(std::string_view p) noexcept {
    if (p.find("swapchain") != std::string_view::npos) return "swapchain";
    if (p.find("ahb+ASurfaceControl") != std::string_view::npos) return "ahb+ASurfaceControl";
    if (p.find("ANativeWindow_lock") != std::string_view::npos) return "ANativeWindow_lock";
    return p;
}

// Live Vulkan R5a state shared across the render thread and UI `SurfaceHolder.Callback`.
std::mutex g_vk_m;
std::shared_ptr<wse::platform::VulkanD3d9Backend> g_vk_backend;
ANativeWindow* g_pending_window = nullptr;

std::mutex g_stats_m;
std::uint64_t g_frames_presented = 0U;
double g_last_inst_fps = 0.0;
double g_last_avg_fps = 0.0;
double g_menu_avg_fps = 0.0;
bool g_menu_started = false;
std::chrono::steady_clock::time_point g_menu_start_tp{};
std::chrono::steady_clock::time_point g_last_log_tp{};
std::chrono::steady_clock::time_point g_last_hud_tp{};
std::atomic<bool> g_reported_driver_summary{false};
std::atomic<bool> g_saved_menu_cache{false};
std::atomic<bool> g_finished_report{false};

int run_startup_common(JNIEnv* env, jobject bridge, const std::string& copy_dir, const std::filesystem::path& work,
                       const std::string& module_name, jint jobs, jlong frames,
                       const std::shared_ptr<HudNotifier>& hud) {
    namespace fs = std::filesystem;

    // The install: the SAF folder (through the launcher's bridge) or a copy in app storage.
    static std::unique_ptr<wse::platform::Vfs> install_base;
    static std::unique_ptr<LockedVfs> install;
    if (!copy_dir.empty()) {
        report("install: the copy in app storage (" + copy_dir + ")\n");
        install_base = std::make_unique<wse::platform::HostVfs>(fs::path(copy_dir));
    } else {
        JavaVM* vm = nullptr;
        if (bridge == nullptr || env->GetJavaVM(&vm) != JNI_OK) {
            report("setup: no install folder\nRESULT setup-error\n");
            if (hud) hud->update("Setup error: no Warband folder selected.");
            return wse::recomp::kStartupSetupError;
        }
        auto backend = std::make_shared<ThreadJniSafBackend>(vm, env, bridge);
        if (!backend->valid()) {
            report("setup: the install folder bridge is missing a method\nRESULT setup-error\n");
            if (hud) hud->update("Setup error: install folder bridge invalid.");
            return wse::recomp::kStartupSetupError;
        }
        if (hud) hud->update("Indexing Warband folder…");
        report("install: the SAF folder (indexing it)\n");
        auto saf = std::make_unique<wse::platform::SafVfs>(backend, false);
        if (!saf->rebuild_index()) {
            report("setup: indexing the install folder failed (permission revoked or folder removed?)\nRESULT setup-error\n");
            if (hud) hud->update("Setup error: failed to index Warband folder.");
            return wse::recomp::kStartupSetupError;
        }
        report("install: " + std::to_string(saf->index_file_count()) + " files indexed\n");
        install_base = std::move(saf);
    }
    install = std::make_unique<LockedVfs>(*install_base);
    static wse::platform::HostVfs user(work / "user");

    wse::recomp::StartupOptions o;
    if (hud) hud->update("Reading mb_warband_wse2_x64.exe…");
    if (!install->read_file("mb_warband_wse2_x64.exe", o.exe) || !install->read_file("lua5164.dll", o.lua) ||
        !install->read_file("mb_warband_wse2_x64.pdb", o.pdb)) {
        report("setup: mb_warband_wse2_x64.exe, lua5164.dll and mb_warband_wse2_x64.pdb must be in the install's root\n"
               "RESULT setup-error\n");
        if (hud) hud->update("Missing WSE2 files in Warband folder.");
        return wse::recomp::kStartupSetupError;
    }
    o.install = install.get();
    o.user = &user;
    o.so_dir = (work / "so").string();
    o.module = module_name.empty() ? std::string("Native") : module_name;
    o.jobs = static_cast<unsigned>(jobs < 1 ? 1 : jobs);
    o.opt = 2;
    o.quiet = false;
    o.frame_limit = frames <= 0 ? 0 : static_cast<std::uint64_t>(frames);
    o.log = report;
    o.progress = [hud](const std::string& image, std::size_t done, std::size_t total) {
        std::ofstream(g_progress_path) << image << " " << done << " " << total << "\n";
        if (hud && total > 0 && done < total) {
            hud->update("Preparing " + image + ": " + std::to_string(done) + "/" + std::to_string(total));
        } else if (hud && total > 0 && done == total) {
            hud->update("Loading Warband…");
        }
    };
    report(std::to_string(o.jobs) + " translation threads, -O2, module " + o.module +
           (o.frame_limit == 0 ? std::string(", live run (no frame limit)\n")
                               : (", stop after " + std::to_string(o.frame_limit) + " frames\n")));
    // The shared phone target (the one a PC prepares batches for) whenever this CPU has LSE and
    // RCpc. Asked of the kernel here, because LLVM 18's host feature list never reports "rcpc"
    // (it does not map /proc/cpuinfo's "lrcpc"), so `lifter_llvm.cpp`'s own check always fell back
    // to the host CPU, and no PC-prepared batch matched (seen on the user's Cortex-A78 phone,
    // 2026-11-22). Kept out of the lifter sources on purpose: they make the translator fingerprint.
    const unsigned long hwcap = getauxval(AT_HWCAP);
    if ((hwcap & kHwcapAtomics) != 0 && (hwcap & kHwcapLrcpc) != 0) {
        wse::recomp::set_code_target(wse::recomp::portable_android_target());
    }
    if (hud) hud->update("Loading Warband (" + o.module + ")…");
    return wse::recomp::run_startup(o);  // returns only when nothing could be started
}

}  // namespace

extern "C" JNIEXPORT jint JNICALL Java_org_wse2_android_launcher_R4Native_nativeRun(JNIEnv* env, jclass, jobject bridge,
                                                                                    jstring jcopy_dir, jstring jwork_dir,
                                                                                    jstring jmodule, jint jobs, jlong frames) {
    namespace fs = std::filesystem;
    const fs::path work = fs::path(jstring_to_utf8(env, jwork_dir)) / "r4";
    std::error_code ec;
    fs::create_directories(work / "user", ec);
    {
        const std::lock_guard<std::mutex> lock(g_report_m);
        g_report = std::fopen((work / "report.txt").c_str(), "w");
    }
    g_progress_path = (work / "progress.txt").string();
    wse::recomp::shim::D3d9::created_hook() = nullptr;
    return run_startup_common(env, bridge, jstring_to_utf8(env, jcopy_dir), work, jstring_to_utf8(env, jmodule),
                              jobs, frames, nullptr);
}

extern "C" JNIEXPORT jint JNICALL Java_org_wse2_android_launcher_R4Native_nativeRunVulkan(
    JNIEnv* env, jclass, jobject bridge, jstring jcopy_dir, jstring jwork_dir, jstring jmodule, jint jobs,
    jlong frames, jobject jsurface, jstring jdriver_mode, jstring jcustom_icd_path, jstring jnative_lib_dir,
    jstring jshader_cache_dir, jint max_texture_size, jobject jhud_listener) {
    namespace fs = std::filesystem;
    const fs::path work = fs::path(jstring_to_utf8(env, jwork_dir)) / "r4";
    std::error_code ec;
    fs::create_directories(work / "user", ec);
    {
        const std::lock_guard<std::mutex> lock(g_report_m);
        g_report = std::fopen((work / "report.txt").c_str(), "w");
    }
    g_progress_path = (work / "progress.txt").string();
    g_reported_driver_summary.store(false);
    g_saved_menu_cache.store(false);
    g_finished_report.store(false);

    JavaVM* vm = nullptr;
    (void)env->GetJavaVM(&vm);
    auto hud = std::make_shared<HudNotifier>(vm, env, jhud_listener);

    const std::string mode_str = jstring_to_utf8(env, jdriver_mode);
    const std::string custom_icd = jstring_to_utf8(env, jcustom_icd_path);
    const std::string native_lib_dir = jstring_to_utf8(env, jnative_lib_dir);
    const std::string shader_cache_dir = jstring_to_utf8(env, jshader_cache_dir);
    wse::platform::DriverMode mode = wse::platform::DriverMode::system;
    (void)wse::platform::parse_driver_mode(mode_str, mode);
    const std::string drv_label(driver_label(mode));

    report("r5a-vulkan: mode=" + std::string(wse::platform::driver_mode_name(mode)) +
           " max_tex=" + std::to_string(std::max<jint>(0, max_texture_size)) + "\n");
    hud->update("Initializing " + drv_label + "…");

    auto loader = std::make_shared<wse::platform::VulkanDriverLoader>();
    std::string vk_err;
    bool loaded = false;
    if (mode != wse::platform::DriverMode::system) {
        if (custom_icd.empty()) {
            report("vulkan: custom GPU driver '" + drv_label +
                   "' is selected, but no driver package has been imported yet.\n"
                   "Use Advanced > Import driver package first, or switch GPU Driver to System Vulkan.\n"
                   "RESULT setup-error\n");
            hud->update("No " + drv_label + " driver package imported.");
            return wse::recomp::kStartupSetupError;
        }
        (void)wse::platform::apply_mesa_driver_environment(mode, shader_cache_dir);
        std::string hw_stub = native_lib_dir.empty() ? std::string("libpanvk_hw.so")
                                                     : (native_lib_dir + "/libpanvk_hw.so");
        if (::access(hw_stub.c_str(), R_OK) != 0) {
            hw_stub = "libpanvk_hw.so";
        }
        report("vulkan: loading custom ICD " + custom_icd + " (hw_stub=" + hw_stub + ")\n");
        loaded = loader->load_custom_driver(mode, custom_icd, hw_stub, &vk_err);
    } else {
        loaded = loader->load_system(&vk_err);
    }
    if (!loaded) {
        report("vulkan: driver load failed: " + vk_err + "\nRESULT vulkan-error\n");
        hud->update("Vulkan load failed: " + vk_err);
        return wse::recomp::kStartupSetupError;
    }

    wse::platform::VulkanDriverProbeResult probe{};
    if (!loader->probe_and_create_device(probe)) {
        report("vulkan: probe/device failed: " + probe.error + "\nRESULT vulkan-error\n");
        hud->update("Vulkan device failed: " + probe.error);
        return wse::recomp::kStartupSetupError;
    }

    report("vulkan-probe: mode=" + std::string(wse::platform::driver_mode_name(probe.mode)) +
           " device=" + probe.device_name +
           " api=" + wse::platform::format_vulkan_version(probe.api_version) +
           " driver=" + wse::platform::format_vulkan_version(probe.driver_version) +
           " direct_icd=" + (probe.used_direct_icd ? "yes" : "no") +
           " hal=" + (probe.used_hal_module ? "yes" : "no") +
           " bc=" + (probe.texture_compression_bc ? "yes" : "no") +
           " wsi=" + (probe.supports_android_surface_wsi ? "yes" : "no") +
           " swapchain=" + (probe.swapchain_extension_enabled ? "yes" : "no") +
           " ahb=" + (probe.ahb_external_memory_enabled ? "yes" : "no") +
           " foreign_q=" + (probe.queue_family_foreign_enabled ? "yes" : "no") +
           " ext_sem_fd=" + (probe.external_semaphore_fd_enabled ? "yes" : "no") +
           (probe.presentation_note.empty() ? std::string{} : (" note=" + probe.presentation_note)) + "\n");

    ANativeWindow* init_win = (jsurface != nullptr) ? ANativeWindow_fromSurface(env, jsurface) : nullptr;
    {
        const std::lock_guard<std::mutex> lock(g_vk_m);
        if (init_win == nullptr && g_pending_window != nullptr) {
            init_win = g_pending_window;
            g_pending_window = nullptr;
        }
    }

    wse::platform::VulkanD3d9Config cfg{};
    cfg.pipeline_cache_path = (work / "r5_pipeline_cache.bin").string();
    cfg.max_texture_size = static_cast<std::uint32_t>(std::max<jint>(0, max_texture_size));
    cfg.native_window = init_win;
    cfg.on_frame_presented = [hud, drv_label](std::uint64_t frame, double inst_fps, double avg_fps,
                                              std::string_view present_path) {
        std::shared_ptr<wse::platform::VulkanD3d9Backend> bk;
        {
            const std::lock_guard<std::mutex> lock(g_vk_m);
            bk = g_vk_backend;
        }
        if (bk && !g_reported_driver_summary.exchange(true)) {
            report(bk->report().format_summary() + "\n");
        }
        if (bk && frame >= kMenuStartFrame && !g_saved_menu_cache.exchange(true)) {
            bk->flush_pipeline_cache();
        }

        const auto now = std::chrono::steady_clock::now();
        double menu_avg = 0.0;
        bool should_log = false;
        bool should_hud = false;
        {
            const std::lock_guard<std::mutex> lock(g_stats_m);
            g_frames_presented = frame;
            g_last_inst_fps = inst_fps;
            g_last_avg_fps = avg_fps;
            if (frame >= kMenuStartFrame) {
                if (!g_menu_started) {
                    g_menu_started = true;
                    g_menu_start_tp = now;
                } else {
                    const double ms = std::chrono::duration<double>(now - g_menu_start_tp).count();
                    if (ms >= 0.25) {
                        g_menu_avg_fps = static_cast<double>(frame - kMenuStartFrame) / ms;
                    }
                }
            }
            menu_avg = g_menu_avg_fps;
            const bool milestone = (frame == 1U || frame == 60U || frame == 250U || frame == 500U ||
                                    frame == 1000U || frame == kMenuStartFrame);
            if (milestone || std::chrono::duration<double>(now - g_last_log_tp).count() >= 2.0) {
                g_last_log_tp = now;
                should_log = true;
            }
            if (frame == 1U || std::chrono::duration<double>(now - g_last_hud_tp).count() >= 0.5) {
                g_last_hud_tp = now;
                should_hud = true;
            }
        }

        if (should_log) {
            std::string line = "fps: frame=" + std::to_string(frame) +
                               " inst=" + format_fixed1(inst_fps) +
                               " avg=" + format_fixed1(avg_fps);
            if (menu_avg > 0.0) {
                line += " menu_avg=" + format_fixed1(menu_avg);
            }
            if (bk) {
                line += " shaders=" + std::to_string(bk->compiled_shader_count()) + "/" +
                        std::to_string(bk->failed_shader_count()) +
                        " pipelines=" + std::to_string(bk->cached_pipeline_count());
            }
            line += " path=" + std::string(present_path) + "\n";
            // Where the time goes: install reads, texture preparation, the game thread's own CPU
            // time (this callback runs on it) and the whole app's CPU time, against wall time.
            {
                static const auto run_start = now;
                timespec ts{};
                const double game_cpu = clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) == 0
                                            ? static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) / 1e9
                                            : 0.0;
                rusage ru{};
                double app_cpu = 0.0;
                if (getrusage(RUSAGE_SELF, &ru) == 0) {
                    app_cpu = static_cast<double>(ru.ru_utime.tv_sec + ru.ru_stime.tv_sec) +
                              static_cast<double>(ru.ru_utime.tv_usec + ru.ru_stime.tv_usec) / 1e6;
                }
                std::string load = "load: wall=" + format_fixed1(std::chrono::duration<double>(now - run_start).count()) +
                                   "s files=" + format_fixed1(static_cast<double>(g_file_ns.load()) / 1e9) + "s/" +
                                   std::to_string(g_file_count.load()) + "/" +
                                   std::to_string(g_file_bytes.load() >> 20) + "MiB";
                if (bk) {
                    const auto tt = bk->texture_timings();
                    load += " textures=" + format_fixed1(tt.seconds) + "s/" + std::to_string(tt.textures) + "/" +
                            std::to_string(tt.bytes >> 20) + "MiB";
                }
                load += " game_cpu=" + format_fixed1(game_cpu) + "s app_cpu=" + format_fixed1(app_cpu) + "s\n";
                line += load;
            }
            report(line);
        }
        if (should_hud && hud) {
            std::string pill = drv_label + " · " + format_fixed1(inst_fps) + " fps · f" +
                               std::to_string(frame) + " · " + std::string(short_present_path(present_path));
            hud->update(pill);
        }
    };

    auto vk_backend = std::make_shared<wse::platform::VulkanD3d9Backend>(loader, std::move(cfg));
    if (init_win != nullptr) {
        ANativeWindow_release(init_win);
        init_win = nullptr;
    }
    ANativeWindow* late_win = nullptr;
    {
        const std::lock_guard<std::mutex> lock(g_vk_m);
        g_vk_backend = vk_backend;
        late_win = g_pending_window;
        g_pending_window = nullptr;
    }
    if (late_win != nullptr) {
        vk_backend->set_native_window(late_win);
        ANativeWindow_release(late_win);
    }

    // Hook into `shim::D3d9`'s constructor without modifying `r4_startup.cpp` (preserves translator_fingerprint).
    wse::recomp::shim::D3d9::created_hook() = [](wse::recomp::shim::D3d9& d3d) {
        const std::lock_guard<std::mutex> lock(g_vk_m);
        if (g_vk_backend) d3d.backend = g_vk_backend.get();
    };

    return run_startup_common(env, bridge, jstring_to_utf8(env, jcopy_dir), work, jstring_to_utf8(env, jmodule),
                              jobs, frames, hud);
}

extern "C" JNIEXPORT void JNICALL Java_org_wse2_android_launcher_R4Native_nativeSetSurface(JNIEnv* env, jclass,
                                                                                           jobject jsurface) {
    ANativeWindow* win = (jsurface != nullptr) ? ANativeWindow_fromSurface(env, jsurface) : nullptr;
    std::shared_ptr<wse::platform::VulkanD3d9Backend> bk;
    {
        const std::lock_guard<std::mutex> lock(g_vk_m);
        if (g_pending_window != nullptr) {
            ANativeWindow_release(g_pending_window);
            g_pending_window = nullptr;
        }
        bk = g_vk_backend;
        if (!bk) {
            g_pending_window = win;
            return;
        }
    }
    bk->set_native_window(win);
    if (win != nullptr) {
        ANativeWindow_release(win);
    }
}

extern "C" JNIEXPORT void JNICALL Java_org_wse2_android_launcher_R4Native_nativeFinishReport(JNIEnv*, jclass) {
    if (g_finished_report.exchange(true)) return;

    std::shared_ptr<wse::platform::VulkanD3d9Backend> bk;
    {
        const std::lock_guard<std::mutex> lock(g_vk_m);
        bk = g_vk_backend;
    }
    if (bk) {
        bk->flush_pipeline_cache();
        bk->set_native_window(nullptr);
        if (!g_reported_driver_summary.exchange(true) && bk->report().ok) {
            report(bk->report().format_summary() + "\n");
        }
    }

    std::uint64_t frames = 0U;
    double inst_fps = 0.0;
    double avg_fps = 0.0;
    double menu_fps = 0.0;
    {
        const std::lock_guard<std::mutex> lock(g_stats_m);
        frames = g_frames_presented;
        inst_fps = g_last_inst_fps;
        avg_fps = g_last_avg_fps;
        menu_fps = g_menu_avg_fps;
    }

    if (frames > 0U) {
        std::string summary = "r5a-summary: frames=" + std::to_string(frames) +
                              " inst_fps=" + format_fixed1(inst_fps) +
                              " avg_fps=" + format_fixed1(avg_fps);
        if (menu_fps > 0.0) summary += " menu_avg_fps=" + format_fixed1(menu_fps);
        if (bk) {
            summary += " path=" + bk->active_presentation_path() +
                       " shaders=" + std::to_string(bk->compiled_shader_count()) + "/" +
                       std::to_string(bk->failed_shader_count()) +
                       " pipelines=" + std::to_string(bk->cached_pipeline_count());
        }
        report(summary + "\nRESULT pass-r5a " + std::to_string(frames) + " frames\n");
    } else {
        report("RESULT stopped-r5a 0 frames\n");
    }
}
