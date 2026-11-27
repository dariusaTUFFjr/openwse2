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
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <atomic>
#include <chrono>
#include <condition_variable>
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
#include "aaudio_device.h"
#include "d3d9_vulkan.h"
#include "driver_loader.h"
#include "perf_hint.h"
#include "saf_jni_util.h"
#include "wse/asset/audio_codecs.h"
#include "wse/platform/caching_vfs.h"
#include "wse/platform/mod_controls_detector.h"
#include "wse/platform/performance.h"
#include "wse/platform/rhi.h"
#include "wse/platform/saf_vfs.h"
#include "wse/platform/vfs.h"
#include "wse/recomp/host_stack.h"
#include "wse/recomp/input_hub.h"
#include "wse/recomp/pdb_reader.h"
#include "wse/recomp/pe_image.h"
#include "wse/recomp/screen.h"
#include "wse/recomp/shim_d3d9.h"
#include "wse/recomp/shim_fmod_steam.h"
#include "wse/recomp/shim_system.h"
#include "wse/recomp/shim_user32.h"

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
        if (clear_if_exception(env)) mid_root_ = mid_list_ = mid_read_ = nullptr;
        mid_open_fd_ = env->GetMethodID(cls, "openDocumentFd", "(Ljava/lang/String;)I");
        if (clear_if_exception(env)) mid_open_fd_ = nullptr;  // older bridge: read through Java
        env->DeleteLocalRef(cls);
    }
    ThreadJniSafBackend(const ThreadJniSafBackend&) = delete;
    ThreadJniSafBackend& operator=(const ThreadJniSafBackend&) = delete;
    ~ThreadJniSafBackend() override {
        if (vm_ != nullptr && bridge_ != nullptr) {
            JNIEnv* env = this_env();
            if (env != nullptr) {
                env->DeleteGlobalRef(bridge_);
            }
            bridge_ = nullptr;
        }
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
        // Java only opens the document; the bytes go straight from the descriptor into `out`
        // (one copy, nothing left for the garbage collector).
        if (mid_open_fd_ != nullptr) {
            const jint fd = env->CallIntMethod(bridge_, mid_open_fd_, jid);
            if (!clear_if_exception(env) && fd >= 0) {
                env->DeleteLocalRef(jid);
                const bool ok = read_fd(fd, out);
                ::close(fd);
                if (!ok) out.clear();
                return ok;
            }
        }
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
    jmethodID mid_open_fd_ = nullptr;

    // Reads a whole descriptor; the size from `fstat` is a hint (a provider may hand out a pipe).
    static bool read_fd(int fd, std::vector<std::uint8_t>& out) {
        struct stat st {};
        std::size_t cap = 1U << 16;
        if (::fstat(fd, &st) == 0 && st.st_size > 0) cap = static_cast<std::size_t>(st.st_size) + 1U;
        out.resize(cap);
        std::size_t got = 0;
        for (;;) {
            if (got == out.size()) out.resize(out.size() * 2U);
            const ssize_t n = ::read(fd, out.data() + got, out.size() - got);
            if (n == 0) break;
            if (n < 0) {
                if (errno == EINTR) continue;
                return false;
            }
            got += static_cast<std::size_t>(n);
        }
        out.resize(got);
        return true;
    }
};

// Install reads (count, bytes, wall time including the wait for the lock), for the load-time
// breakdown in the report's `fps:` lines.
std::atomic<std::uint64_t> g_file_ns{0}, g_file_count{0}, g_file_bytes{0};
bool g_strict_tso = false;
std::string g_profile_in;
std::string g_profile_out;
std::string g_crash_report_path;

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
    bool file_stamp(std::string_view p, std::uint64_t& size, std::int64_t& mtime_ms) override {
        return locked([&] { return inner_.file_stamp(p, size, mtime_ms); });
    }
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
        if (clear_if_exception(env)) mid_update_ = nullptr;
        mid_haptic_ = env->GetMethodID(cls, "onHapticPulse", "(II)V");
        if (clear_if_exception(env)) mid_haptic_ = nullptr;
        mid_ui_action_ = env->GetMethodID(cls, "onUiAction", "(Ljava/lang/String;)V");
        if (clear_if_exception(env)) mid_ui_action_ = nullptr;
        env->DeleteLocalRef(cls);
    }

    HudNotifier(const HudNotifier&) = delete;
    HudNotifier& operator=(const HudNotifier&) = delete;
    ~HudNotifier() {
        if (vm_ != nullptr && listener_ != nullptr) {
            JNIEnv* env = attach_env();
            if (env != nullptr) {
                env->DeleteGlobalRef(listener_);
            }
            listener_ = nullptr;
        }
    }

    void update(std::string_view text) const {
        if (vm_ == nullptr || listener_ == nullptr || mid_update_ == nullptr) return;
        JNIEnv* env = attach_env();
        if (env == nullptr) return;
        jstring js = utf8_to_jstring(env, text);
        if (js != nullptr) {
            env->CallVoidMethod(listener_, mid_update_, js);
            env->DeleteLocalRef(js);
            (void)clear_if_exception(env);
        }
    }

    void haptic_pulse(std::uint32_t duration_ms, std::uint8_t amplitude) const {
        if (vm_ == nullptr || listener_ == nullptr || mid_haptic_ == nullptr) return;
        JNIEnv* env = attach_env();
        if (env == nullptr) return;
        env->CallVoidMethod(listener_, mid_haptic_, static_cast<jint>(duration_ms), static_cast<jint>(amplitude));
        (void)clear_if_exception(env);
    }

    void ui_action(std::string_view action) const {
        if (vm_ == nullptr || listener_ == nullptr || mid_ui_action_ == nullptr) return;
        JNIEnv* env = attach_env();
        if (env == nullptr) return;
        jstring js = utf8_to_jstring(env, action);
        if (js != nullptr) {
            env->CallVoidMethod(listener_, mid_ui_action_, js);
            env->DeleteLocalRef(js);
            (void)clear_if_exception(env);
        }
    }

private:
    [[nodiscard]] JNIEnv* attach_env() const {
        JNIEnv* env = nullptr;
        void* raw = nullptr;
        if (vm_->GetEnv(&raw, JNI_VERSION_1_6) == JNI_OK && raw != nullptr) {
            return static_cast<JNIEnv*>(raw);
        }
        return (vm_->AttachCurrentThread(&env, nullptr) == JNI_OK) ? env : nullptr;
    }

    JavaVM* vm_ = nullptr;
    jobject listener_ = nullptr;
    jmethodID mid_update_ = nullptr;
    jmethodID mid_haptic_ = nullptr;
    jmethodID mid_ui_action_ = nullptr;
};

std::mutex g_report_m;
std::FILE* g_report = nullptr;
std::string g_progress_path;
std::atomic<std::uint64_t> g_va_game_screen{0U};
std::atomic<std::uint64_t> g_va_cheat_mode{0U};

// Polls Warband's live `g_gameScreen->m_openWindows` and `rglConfig::General::bCheatMode`
// from fixed-address guest memory (resolved from the user's PDB at startup) to drive automatic
// per-scene touch layouts and live cheat-menu visibility.
void poll_guest_scene_and_cheats() noexcept {
    const std::uint64_t cheat_va = g_va_cheat_mode.load(std::memory_order_acquire);
    if (cheat_va >= 0x10000000ULL) {
        const auto* p_cheat = reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(cheat_va));
        const bool live_cheat = (*p_cheat != 0U);
        auto& hub = wse::recomp::shim::RecompInputHub::shared();
        if (hub.cheats_enabled() != live_cheat) {
            hub.set_cheats_enabled(live_cheat);
        }
    }

    const std::uint64_t gs_va = g_va_game_screen.load(std::memory_order_acquire);
    if (gs_va < 0x10000000ULL) return;
    // `g_gameScreen->m_openWindows` (+0x130) and the window slots: input_hub.h, with the evidence.
    const std::int32_t top = wse::recomp::shim::top_open_window(
        reinterpret_cast<const std::uint8_t*>(static_cast<std::uintptr_t>(gs_va)));
    if (top < 0) return;
    wse::recomp::shim::RecompInputHub::shared().set_inferred_scene(wse::recomp::shim::scene_for_open_window(top));
}

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

std::string format_fixed2(double v) {
    char buf[32]{};
    std::snprintf(buf, sizeof(buf), "%.2f", v);
    return std::string(buf);
}

struct ThermalProbe {
    void* lib = nullptr;
    void* mgr = nullptr;
    using AcquireFn = void* (*)();
    using GetStatusFn = std::int32_t (*)(void*);
    using ReleaseFn = void (*)(void*);
    AcquireFn acquire = nullptr;
    GetStatusFn get_status = nullptr;
    ReleaseFn release = nullptr;
    bool initialized = false;

    wse::platform::ThermalStatus poll() noexcept {
        if (!initialized) {
            initialized = true;
            lib = dlopen("libandroid.so", RTLD_LAZY | RTLD_LOCAL);
            if (lib != nullptr) {
                acquire = reinterpret_cast<AcquireFn>(dlsym(lib, "AThermal_acquireManager"));
                get_status = reinterpret_cast<GetStatusFn>(dlsym(lib, "AThermal_getCurrentThermalStatus"));
                release = reinterpret_cast<ReleaseFn>(dlsym(lib, "AThermal_releaseManager"));
                if (acquire != nullptr && get_status != nullptr) {
                    mgr = acquire();
                }
            }
        }
        if (mgr == nullptr || get_status == nullptr) return wse::platform::ThermalStatus::None;
        return wse::platform::sanitize_thermal_status(get_status(mgr));
    }
};

std::string_view driver_label(wse::platform::DriverMode mode) noexcept {
    switch (mode) {
    case wse::platform::DriverMode::panvk: return "PanVK";
    case wse::platform::DriverMode::turnip: return "Turnip";
    case wse::platform::DriverMode::system:
    default: return "System Vulkan";
    }
}

// Live Vulkan R5a state shared across the render thread and UI `SurfaceHolder.Callback`.
std::mutex g_vk_m;
std::shared_ptr<wse::platform::VulkanD3d9Backend> g_vk_backend;
ANativeWindow* g_pending_window = nullptr;
std::unique_ptr<wse::platform::android_perf::PerformanceHintSession> g_adpf_session;
std::atomic<bool> g_notified_first_frame{false};

// Live R6 audio & VFS handles shared with `FmodSteam::created_hook()` and `nativeSetPaused`.
std::mutex g_audio_m;
wse::recomp::shim::FmodSteam* g_active_fmod = nullptr;
std::unique_ptr<wse::platform::AAudioDevice> g_aaudio_dev;
wse::platform::Vfs* g_audio_install_vfs = nullptr;
wse::platform::Vfs* g_audio_user_vfs = nullptr;
std::shared_ptr<HudNotifier> g_active_hud;
float g_audio_master_volume = 1.0F;
wse::recomp::shim::CombatHapticsSettings g_haptics_cfg{};

// Background pause state: the game thread blocks at the end of `on_frame_presented`
// while the activity is paused or has no surface, and wakes immediately on resume or shutdown.
std::mutex g_pause_m;
std::condition_variable g_pause_cv;
bool g_activity_paused = false;
bool g_has_surface = false;
bool g_shutting_down = false;
std::chrono::steady_clock::duration g_paused_duration{};

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
std::atomic<wse::platform::CachingVfs*> g_install_cache{nullptr};

int run_startup_common(JNIEnv* env, jobject bridge, const std::string& copy_dir, const std::filesystem::path& work,
                       const std::string& module_name, jint jobs, jlong frames,
                       const std::shared_ptr<HudNotifier>& hud) {
    namespace fs = std::filesystem;

    // The install: the SAF folder (through the launcher's bridge) or a copy in app storage.
    static std::unique_ptr<wse::platform::Vfs> install_base;
    static std::unique_ptr<LockedVfs> install;
    static std::unique_ptr<wse::platform::CachingVfs> install_cache;
    const bool use_install_cache = copy_dir.empty();
    if (!copy_dir.empty()) {
        report("install: the copy in app storage (" + copy_dir + ")\n");
        install_base = std::make_unique<wse::platform::HostVfs>(fs::path(copy_dir));
    } else {
        JavaVM* vm = nullptr;
        if (bridge == nullptr || env->GetJavaVM(&vm) != JNI_OK) {
            report("setup: no install folder\nRESULT setup-error\n");
            if (hud) hud->update("No Warband folder is selected. Pick it in the app first.");
            return wse::recomp::kStartupSetupError;
        }
        auto backend = std::make_shared<ThreadJniSafBackend>(vm, env, bridge);
        if (!backend->valid()) {
            report("setup: the install folder bridge is missing a method\nRESULT setup-error\n");
            if (hud) hud->update("Can't read the Warband folder. Pick it again in the app.");
            return wse::recomp::kStartupSetupError;
        }
        if (hud) hud->update("Looking through your Warband folder…");
        report("install: the SAF folder (indexing it)\n");
        auto saf = std::make_unique<wse::platform::SafVfs>(backend, false);
        if (!saf->rebuild_index()) {
            report("setup: indexing the install folder failed (permission revoked or folder removed?)\nRESULT setup-error\n");
            if (hud) hud->update("Can't read the Warband folder. Pick it again in the app.");
            return wse::recomp::kStartupSetupError;
        }
        report("install: " + std::to_string(saf->index_file_count()) + " files indexed\n");
        install_base = std::move(saf);
    }
    install = std::make_unique<LockedVfs>(*install_base);
    install_cache.reset();
    wse::platform::Vfs* game_install = install.get();
    if (use_install_cache) {
        install_cache = std::make_unique<wse::platform::CachingVfs>(*install, work / "install-cache");
        game_install = install_cache.get();
    }
    g_install_cache.store(install_cache.get(), std::memory_order_release);
    // Permanent user storage (`filesDir/wse2/user`, P14.3): kept outside `filesDir/recomp/r4/`
    // so cache clears never touch saves or config. Falls back to `work / "user"` if no parent dir.
    const fs::path files_root = work.parent_path().parent_path();
    const fs::path user_root = files_root.empty() ? (work / "user") : (files_root / "wse2" / "user");
    {
        std::error_code uec;
        fs::create_directories(user_root, uec);
    }
    static wse::platform::HostVfs user(user_root);
    {
        const std::lock_guard<std::mutex> lock(g_audio_m);
        g_audio_install_vfs = game_install;
        g_audio_user_vfs = &user;
    }

    const std::string resolved_mod = module_name.empty() ? std::string("Native") : module_name;
    const auto detected_mods = wse::platform::detect_mod_controls(*game_install, "Modules/" + resolved_mod, resolved_mod);
    if (!detected_mods.buttons.empty() || detected_mods.crouch_supported) {
        report("mod-controls: " + detected_mods.summary_text() + "\n");
    }
    wse::recomp::shim::RecompInputHub::shared().apply_detected_mod_controls(detected_mods);

    // Auto-enable the Cheats overlay menu in every scene when cheats are enabled in WSE2's
    // `rgl_config.ini` (`bCheatMode=true`) or legacy `rgl_config.txt` (`cheat_mode = 1`).
    {
        bool cheats_on = false;
        std::string rgl_cfg;
        auto check_cfg = [&](std::string_view rel_path, std::string_view key) {
            if (cheats_on) return;
            rgl_cfg.clear();
            if (user.read_text(std::string(rel_path), rgl_cfg)) {
                const std::string low = wse::platform::mod_detect_detail::to_lower_ascii(rgl_cfg);
                bool flag = false;
                if (wse::platform::mod_detect_detail::parse_ini_bool(low, key, flag) && flag) {
                    cheats_on = true;
                }
            }
        };
        check_cfg("Documents/Mount&Blade Warband WSE2/rgl_config.ini", "bcheatmode");
        check_cfg("Documents/Mount&Blade Warband WSE2/rgl_config.txt", "cheat_mode");
        check_cfg("Documents/Mount&Blade Warband/rgl_config.txt", "cheat_mode");
        wse::recomp::shim::RecompInputHub::shared().set_cheats_enabled(cheats_on);
        if (cheats_on) {
            report("cheats-menu: enabled (cheat mode active in config)\n");
        }
    }

    wse::recomp::StartupOptions o;
    o.strict_tso = g_strict_tso;
    o.profile_in = g_profile_in;
    o.profile_out = g_profile_out;
    o.crash_report_path = g_crash_report_path.empty() ? (user_root / "crash_report.txt").string() : g_crash_report_path;
    if (hud) hud->update("Reading the game…");
    if (!game_install->read_file("mb_warband_wse2_x64.exe", o.exe) ||
        !game_install->read_file("lua5164.dll", o.lua) ||
        !game_install->read_file("mb_warband_wse2_x64.pdb", o.pdb)) {
        report("setup: mb_warband_wse2_x64.exe, lua5164.dll and mb_warband_wse2_x64.pdb must be in the install's root\n"
               "RESULT setup-error\n");
        if (hud) hud->update("WSE2 isn't installed in this Warband folder.");
        return wse::recomp::kStartupSetupError;
    }

    // Resolve live PDB globals (`g_gameScreen` and `rglConfig::General::bCheatMode`) for automatic
    // per-scene touch layout switching and live cheat-menu detection (without touching r4_startup.cpp).
    {
        wse::recomp::PeImage pe{};
        std::vector<wse::recomp::PdbPublic> publics;
        std::string pdb_err;
        if (wse::recomp::parse_pe(o.exe.data(), o.exe.size(), pe, &pdb_err) &&
            wse::recomp::read_pdb_publics(o.pdb.data(), o.pdb.size(), publics, nullptr, &pdb_err)) {
            if (auto va = wse::recomp::pdb_public_va(publics, "?g_gameScreen@@3VmbGameScreen@@A",
                                                     pe.sections, pe.image_base)) {
                g_va_game_screen.store(*va, std::memory_order_release);
            }
            if (auto va = wse::recomp::pdb_public_va(publics, "?bCheatMode@General@rglConfig@@2_NA",
                                                     pe.sections, pe.image_base)) {
                g_va_cheat_mode.store(*va, std::memory_order_release);
            }
        }
    }
    o.install = game_install;
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
            hud->update("Preparing the game: " + std::to_string(done) + " of " + std::to_string(total));
        } else if (hud && total > 0 && done == total) {
            hud->update("Loading Warband…");
        }
    };
    report(std::to_string(o.jobs) + " translation threads, -O2, module " + o.module +
           (o.frame_limit == 0 ? std::string(", live run (no frame limit)\n")
                               : (", stop after " + std::to_string(o.frame_limit) + " frames\n")));
    // The shared phone target (the one a PC prepares batches for) whenever this CPU has LSE and
    // RCpc. `lifter_llvm.cpp` now asks the kernel itself (LLVM 18's host feature list never
    // reports "rcpc", seen on the user's Cortex-A78 phone, 2026-11-22); this stays as the same
    // check made explicitly, so the target never depends on which side runs first.
    const unsigned long hwcap = getauxval(AT_HWCAP);
    if ((hwcap & kHwcapAtomics) != 0 && (hwcap & kHwcapLrcpc) != 0) {
        wse::recomp::set_code_target(wse::recomp::portable_android_target());
    }
    if (hud) hud->update("Loading Warband (" + o.module + ")…");
    // This thread runs the game: it was started with R4Native.GAME_STACK_BYTES; record its bounds
    // and give it an alternate signal stack (wse/recomp/host_stack.h).
    wse::recomp::prepare_current_host_thread();
    return wse::recomp::run_startup(o);  // returns only when nothing could be started
}

}  // namespace

extern "C" JNIEXPORT jint JNICALL Java_org_wse2_android_launcher_R4Native_nativeRun(JNIEnv* env, jclass, jobject bridge,
                                                                                    jstring jcopy_dir, jstring jwork_dir,
                                                                                    jstring jmodule, jint jobs, jlong frames) {
    g_strict_tso = false;
    g_profile_in.clear();
    g_profile_out.clear();
    g_crash_report_path.clear();
    namespace fs = std::filesystem;
    const fs::path work = fs::path(jstring_to_utf8(env, jwork_dir)) / "r4";
    std::error_code ec;
    fs::create_directories(work, ec);
    {
        const std::lock_guard<std::mutex> lock(g_report_m);
        g_report = std::fopen((work / "report.txt").c_str(), "w");
    }
    g_progress_path = (work / "progress.txt").string();
    wse::recomp::shim::D3d9::created_hook() = nullptr;
    wse::recomp::shim::User32::created_hook() = nullptr;
    wse::recomp::shim::SystemDlls::created_hook() = nullptr;
    wse::recomp::shim::FmodSteam::created_hook() = nullptr;
    return run_startup_common(env, bridge, jstring_to_utf8(env, jcopy_dir), work, jstring_to_utf8(env, jmodule),
                              jobs, frames, nullptr);
}

extern "C" JNIEXPORT jint JNICALL Java_org_wse2_android_launcher_R4Native_nativeRunVulkan(
    JNIEnv* env, jclass, jobject bridge, jstring jcopy_dir, jstring jwork_dir, jstring jmodule, jint jobs,
    jlong frames, jobject jsurface, jstring jdriver_mode, jstring jcustom_icd_path, jstring jnative_lib_dir,
    jstring jshader_cache_dir, jint max_texture_size, jfloat render_scale, jboolean auto_resolution,
    jfloat sharpness, jboolean fxaa, jfloat target_fps, jboolean custom_driver_shared_buffers,
    jboolean strict_tso, jstring jprofile_in, jstring jprofile_out, jstring jcrash_report_path,
    jobject jhud_listener) {
    g_strict_tso = (strict_tso == JNI_TRUE);
    g_profile_in = jstring_to_utf8(env, jprofile_in);
    g_profile_out = jstring_to_utf8(env, jprofile_out);
    g_crash_report_path = jstring_to_utf8(env, jcrash_report_path);
    namespace fs = std::filesystem;
    const fs::path work = fs::path(jstring_to_utf8(env, jwork_dir)) / "r4";
    std::error_code ec;
    fs::create_directories(work, ec);
    {
        const std::lock_guard<std::mutex> lock(g_report_m);
        g_report = std::fopen((work / "report.txt").c_str(), "w");
    }
    g_progress_path = (work / "progress.txt").string();
    g_reported_driver_summary.store(false);
    g_saved_menu_cache.store(false);
    g_finished_report.store(false);
    g_notified_first_frame.store(false);

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
    hud->update("Starting " + drv_label + "…");

    auto loader = std::make_shared<wse::platform::VulkanDriverLoader>();
    std::string vk_err;
    bool loaded = false;
    if (mode != wse::platform::DriverMode::system) {
        if (custom_icd.empty()) {
            report("vulkan: custom GPU driver '" + drv_label +
                   "' is selected, but no driver package has been imported yet.\n"
                   "Use Advanced > Import driver package first, or switch GPU Driver to System Vulkan.\n"
                   "RESULT setup-error\n");
            hud->update("Import the " + drv_label + " driver in the app first.");
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
        hud->update(drv_label + " couldn't start: " + vk_err);
        return wse::recomp::kStartupSetupError;
    }

    wse::platform::VulkanDriverProbeResult probe{};
    if (!loader->probe_and_create_device(probe)) {
        report("vulkan: probe/device failed: " + probe.error + "\nRESULT vulkan-error\n");
        hud->update("The graphics driver couldn't start: " + probe.error);
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
    {
        const std::lock_guard<std::mutex> lock(g_pause_m);
        g_activity_paused = false;
        g_has_surface = (init_win != nullptr);
        g_shutting_down = false;
        g_paused_duration = std::chrono::steady_clock::duration::zero();
    }

    // Widescreen: the game draws at the screen's own shape (e.g. 2412 x 1080 on a 20:9 phone) when
    // the shims report it as the desktop; without a window it stays 1920 x 1080.
    if (init_win != nullptr) {
        const int sw = ANativeWindow_getWidth(init_win);
        const int sh = ANativeWindow_getHeight(init_win);
        wse::recomp::shim::set_screen_size(sw, sh);
        wse::recomp::shim::RecompInputHub::shared().set_screen_size(sw, sh);
    } else {
        const auto sz = wse::recomp::shim::screen_size();
        wse::recomp::shim::RecompInputHub::shared().set_screen_size(sz.width, sz.height);
    }
    {
        const std::lock_guard<std::mutex> lock(g_audio_m);
        g_active_hud = hud;
    }
    wse::recomp::shim::RecompInputHub::shared().set_ui_action_sink([hud](std::string_view act) {
        if (hud) hud->ui_action(act);
    });
    report("screen: " + std::to_string(wse::recomp::shim::screen_size().width) + "x" +
           std::to_string(wse::recomp::shim::screen_size().height) + "\n");

    wse::platform::VulkanD3d9Config cfg{};
    cfg.build_touch_overlay = [](wse::platform::GeometryFrame& out_frame, std::uint32_t w, std::uint32_t h) {
        return wse::recomp::shim::RecompInputHub::shared().build_overlay_geometry(out_frame, w, h);
    };
    cfg.query_menu_zoom = []() -> std::array<float, 3> {
        const auto mv = wse::recomp::shim::RecompInputHub::shared().menu_zoom_view();
        return {mv.zoom, mv.pan_x, mv.pan_y};
    };
    cfg.pipeline_cache_path = (work / "r5_pipeline_cache.bin").string();
    {
        const std::filesystem::path tex_cache_dir = work / "texcache";
        std::error_code tex_ec;
        std::filesystem::create_directories(tex_cache_dir, tex_ec);
        cfg.texture_cache_dir = tex_cache_dir.string();
    }
    cfg.max_texture_size = static_cast<std::uint32_t>(std::max<jint>(0, max_texture_size));
    cfg.native_window = init_win;
    cfg.custom_driver_shared_buffers = (custom_driver_shared_buffers == JNI_TRUE);
    cfg.render_scale = std::clamp(static_cast<float>(render_scale), 0.5F, 1.0F);
    cfg.auto_resolution = (auto_resolution == JNI_TRUE);
    const float resolved_target_fps = std::clamp(static_cast<float>(target_fps), 30.0F, 120.0F);
    cfg.min_render_scale = 0.5F;
    cfg.target_fps = resolved_target_fps;
    cfg.sharpness = std::max(0.0F, static_cast<float>(sharpness));
    cfg.fxaa = (fxaa == JNI_TRUE);
    {
        const std::lock_guard<std::mutex> lock(g_vk_m);
        const auto target_ns = static_cast<std::int64_t>(1'000'000'000.0 / static_cast<double>(resolved_target_fps));
        g_adpf_session = wse::platform::android_perf::PerformanceHintSession::create(target_ns);
        if (!g_adpf_session) {
            (void)wse::platform::android_perf::raise_current_thread_priority();
        }
    }
    cfg.on_frame_presented = [hud, drv_label](std::uint64_t frame, double inst_fps, double avg_fps,
                                              std::string_view present_path) {
        if (hud && !g_notified_first_frame.exchange(true)) {
            hud->ui_action("first_frame");
        }
        std::shared_ptr<wse::platform::VulkanD3d9Backend> bk;
        {
            const std::lock_guard<std::mutex> lock(g_vk_m);
            bk = g_vk_backend;
            if (g_adpf_session && bk) {
                const auto ft = bk->last_frame_timings();
                if (ft.frame_ms > 0.0) {
                    g_adpf_session->report_actual(static_cast<std::int64_t>(ft.frame_ms * 1e6));
                }
            }
        }
        if (bk && !g_reported_driver_summary.exchange(true)) {
            report(bk->report().format_summary() + "\n");
        }
        if (bk && frame >= kMenuStartFrame && !g_saved_menu_cache.exchange(true)) {
            bk->flush_pipeline_cache();
        }

        const auto now = std::chrono::steady_clock::now();
        {
            static ThermalProbe thermal_probe{};
            static auto last_thermal_tp = std::chrono::steady_clock::time_point::min();
            if (bk && (last_thermal_tp == std::chrono::steady_clock::time_point::min() ||
                       std::chrono::duration<double>(now - last_thermal_tp).count() >= 2.0)) {
                last_thermal_tp = now;
                bk->set_thermal_status(thermal_probe.poll());
            }
        }
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
                const auto ft = bk->last_frame_timings();
                line += " shaders=" + std::to_string(bk->compiled_shader_count()) + "/" +
                        std::to_string(bk->failed_shader_count()) +
                        " pipelines=" + std::to_string(bk->cached_pipeline_count()) +
                        " scale=" + format_fixed2(ft.render_scale) +
                        " cpu_ms=" + format_fixed1(ft.cpu_ms);
                if (ft.gpu_ms >= 0.0) {
                    line += " gpu_ms=" + format_fixed1(ft.gpu_ms);
                }
                line += " thermal=" + std::string(wse::platform::thermal_status_name(ft.thermal));
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
                std::chrono::steady_clock::duration paused_dur{};
                {
                    const std::lock_guard<std::mutex> lock(g_pause_m);
                    paused_dur = g_paused_duration;
                }
                const double wall_s =
                    std::max(0.0, std::chrono::duration<double>((now - run_start) - paused_dur).count());
                std::string load = "load: wall=" + format_fixed1(wall_s) +
                                   "s files=" + format_fixed1(static_cast<double>(g_file_ns.load()) / 1e9) + "s/" +
                                   std::to_string(g_file_count.load()) + "/" +
                                   std::to_string(g_file_bytes.load() >> 20) + "MiB";
                if (const auto* cache = g_install_cache.load(std::memory_order_acquire)) {
                    load += " copy_hits=" + std::to_string(cache->hits()) +
                            " copied=" + std::to_string(cache->copies()) + "/" +
                            std::to_string(cache->copied_bytes() >> 20) + "MiB";
                }
                if (bk) {
                    const auto tt = bk->texture_timings();
                    load += " textures=" + format_fixed1(tt.seconds) + "s/" + std::to_string(tt.textures) + "/" +
                            std::to_string(tt.bytes >> 20) + "MiB" +
                            " etc2=" + std::to_string(tt.etc2_textures) + "/" +
                            std::to_string(tt.etc2_cached) + "/" +
                            format_fixed1(tt.etc2_transcode_ms / 1000.0) + "s";
                }
                load += " game_cpu=" + format_fixed1(game_cpu) + "s app_cpu=" + format_fixed1(app_cpu) + "s\n";
                line += load;
            }
            report(line);
        }
        poll_guest_scene_and_cheats();
        if (should_hud && hud) {
            std::string pill = format_fixed1(inst_fps) + " FPS";
            if (bk) {
                const auto ft = bk->last_frame_timings();
                if (ft.cpu_ms > 0.0) {
                    pill += " · " + format_fixed1(ft.cpu_ms) + "ms";
                }
                if (ft.gpu_ms >= 0.0) {
                    pill += "/G:" + format_fixed1(ft.gpu_ms) + "ms";
                }
                if (ft.render_scale < 0.995F) {
                    pill += " · " + std::to_string(static_cast<int>(std::lround(ft.render_scale * 100.0F))) + "%";
                }
                if (ft.thermal != wse::platform::ThermalStatus::None) {
                    pill += " · " + std::string(wse::platform::thermal_status_name(ft.thermal));
                }
            }
            hud->update(pill);
        }
        {
            std::unique_lock<std::mutex> lock(g_pause_m);
            if ((g_activity_paused || !g_has_surface) && !g_shutting_down) {
                const auto pause_t0 = std::chrono::steady_clock::now();
                g_pause_cv.wait(lock, [] {
                    return (!g_activity_paused && g_has_surface) || g_shutting_down;
                });
                g_paused_duration += std::chrono::steady_clock::now() - pause_t0;
            }
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
        const int lw = ANativeWindow_getWidth(late_win);
        const int lh = ANativeWindow_getHeight(late_win);
        if (lw > 0 && lh > 0) {
            wse::recomp::shim::set_screen_size(lw, lh);
            wse::recomp::shim::RecompInputHub::shared().set_screen_size(lw, lh);
            wse::recomp::shim::RecompInputHub::shared().set_surface_size(lw, lh);
        }
        {
            const std::lock_guard<std::mutex> lock(g_pause_m);
            g_has_surface = true;
        }
        g_pause_cv.notify_all();
        vk_backend->set_native_window(late_win);
        ANativeWindow_release(late_win);
    }

    // Wire XInput controller rumble into Android haptics when available.
    if (hud) {
        wse::recomp::shim::RecompInputHub::shared().set_rumble_sink(
            [hud](std::uint16_t left_motor, std::uint16_t right_motor) {
                const std::uint32_t peak = std::max<std::uint32_t>(left_motor, right_motor);
                if (peak < 2048U) return;
                const auto amp = static_cast<std::uint8_t>(std::clamp<std::uint32_t>(peak >> 8U, 1U, 255U));
                hud->haptic_pulse(30U, amp);
            });
    }

    // Hook into the R4/R5/R6 shims without modifying `r4_startup.cpp` (preserves translator_fingerprint).
    wse::recomp::shim::D3d9::created_hook() = [](wse::recomp::shim::D3d9& d3d) {
        const std::lock_guard<std::mutex> lock(g_vk_m);
        if (g_vk_backend) d3d.backend = g_vk_backend.get();
        d3d.input_hub = &wse::recomp::shim::RecompInputHub::shared();
    };
    wse::recomp::shim::User32::created_hook() = [](wse::recomp::shim::User32& u) {
        u.input_hub = &wse::recomp::shim::RecompInputHub::shared();
        wse::recomp::shim::RecompInputHub::shared().set_window_msg_sink(
            [&u](std::uint32_t msg, std::uint64_t wp, std::uint64_t lp) { (void)u.post_to_main(msg, wp, lp); });
    };
    wse::recomp::shim::SystemDlls::created_hook() = [](wse::recomp::shim::SystemDlls& sys) {
        sys.input_hub = &wse::recomp::shim::RecompInputHub::shared();
    };
    wse::recomp::shim::FmodSteam::created_hook() = [](wse::recomp::shim::FmodSteam& fmod) {
        const std::lock_guard<std::mutex> lock(g_audio_m);
        g_active_fmod = &fmod;
        fmod.set_codec_backend(wse::asset::create_audio_codec_backend());
        fmod.set_master_volume(g_audio_master_volume);
        fmod.set_haptics_settings(g_haptics_cfg);
        wse::platform::Vfs* ivfs = g_audio_install_vfs;
        wse::platform::Vfs* uvfs = g_audio_user_vfs;
        fmod.read_file = [ivfs, uvfs](const std::string& path, std::vector<std::uint8_t>& out) -> bool {
            std::string p = path;
            std::string low = p;
            for (char& c : low) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            static constexpr std::string_view kRoot = "c:\\wse2\\";
            if (low.rfind(kRoot, 0U) == 0U) p = p.substr(kRoot.size());
            for (char& c : p) {
                if (c == '\\') c = '/';
            }
            while (!p.empty() && p.front() == '/') p.erase(p.begin());
            return (ivfs != nullptr && ivfs->read_file(p, out)) || (uvfs != nullptr && uvfs->read_file(p, out));
        };
        std::shared_ptr<HudNotifier> hud_cb = g_active_hud;
        fmod.on_haptic = [hud_cb](const wse::recomp::shim::CombatHapticPulse& pulse) {
            if (hud_cb && pulse.duration_ms > 0U && pulse.amplitude > 0U) {
                hud_cb->haptic_pulse(pulse.duration_ms, pulse.amplitude);
            }
        };
        g_aaudio_dev = std::make_unique<wse::platform::AAudioDevice>();
        fmod.on_device_recover = []() -> bool {
            return g_aaudio_dev && g_aaudio_dev->recover_if_needed();
        };
        bool initially_paused = false;
        {
            const std::lock_guard<std::mutex> plock(g_pause_m);
            initially_paused = g_activity_paused;
        }
        if (g_aaudio_dev->open(wse::platform::AudioDeviceSpec{48000U, 2U})) {
            fmod.set_audio_device(g_aaudio_dev.get(), true);
            report("audio: Android AAudio opened at " + std::to_string(g_aaudio_dev->spec().sample_rate) + " Hz stereo\n");
        } else {
            // Start silent clock worker so FMOD channels & end callbacks still advance cleanly.
            fmod.start_worker();
            report("audio: AAudio open failed (continuing silent)\n");
        }
        if (initially_paused) {
            fmod.set_app_paused(true);
        }
    };

    return run_startup_common(env, bridge, jstring_to_utf8(env, jcopy_dir), work, jstring_to_utf8(env, jmodule),
                              jobs, frames, hud);
}

extern "C" JNIEXPORT void JNICALL Java_org_wse2_android_launcher_R4Native_nativeSetSurface(JNIEnv* env, jclass,
                                                                                           jobject jsurface) {
    ANativeWindow* win = (jsurface != nullptr) ? ANativeWindow_fromSurface(env, jsurface) : nullptr;
    {
        const std::lock_guard<std::mutex> lock(g_pause_m);
        g_has_surface = (win != nullptr);
    }
    g_pause_cv.notify_all();
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

extern "C" JNIEXPORT void JNICALL Java_org_wse2_android_launcher_R4Native_nativeSetPaused(JNIEnv*, jclass,
                                                                                          jboolean paused) {
    const bool is_paused = (paused == JNI_TRUE);
    if (is_paused) {
        wse::recomp::shim::RecompInputHub::shared().release_all();
    }
    {
        const std::lock_guard<std::mutex> lock(g_pause_m);
        g_activity_paused = is_paused;
    }
    {
        const std::lock_guard<std::mutex> lock(g_audio_m);
        if (g_active_fmod != nullptr) {
            g_active_fmod->set_app_paused(is_paused);
        }
    }
    g_pause_cv.notify_all();
}

extern "C" JNIEXPORT void JNICALL Java_org_wse2_android_launcher_R4Native_nativeFinishReport(JNIEnv*, jclass) {
    flush_pgo_profile_now();
    {
        const std::lock_guard<std::mutex> lock(g_pause_m);
        g_shutting_down = true;
    }
    g_pause_cv.notify_all();
    if (g_finished_report.exchange(true)) return;
    {
        const std::lock_guard<std::mutex> lock(g_audio_m);
        if (g_active_fmod != nullptr) {
            // The mixer thread stops first: it may be running `on_device_recover` right now.
            g_active_fmod->stop_worker();
            g_active_fmod->set_device_recover(nullptr);
            g_active_fmod->set_audio_device(nullptr, false);
            g_active_fmod = nullptr;
        }
        if (g_aaudio_dev) {
            g_aaudio_dev->close();
            g_aaudio_dev.reset();
        }
    }

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

extern "C" JNIEXPORT jlong JNICALL Java_org_wse2_android_launcher_R4Native_installCacheBytes(
    JNIEnv* env, jclass, jstring jwork_dir) {
    const std::string raw = jstring_to_utf8(env, jwork_dir);
    if (raw.empty()) return 0;
    const std::filesystem::path cache_root = std::filesystem::path(raw) / "r4" / "install-cache";
    return static_cast<jlong>(wse::platform::CachingVfs::disk_usage(cache_root));
}

extern "C" JNIEXPORT jboolean JNICALL Java_org_wse2_android_launcher_R4Native_clearInstallCache(
    JNIEnv* env, jclass, jstring jwork_dir) {
    const std::string raw = jstring_to_utf8(env, jwork_dir);
    if (raw.empty()) return JNI_FALSE;
    const std::filesystem::path cache_root = std::filesystem::path(raw) / "r4" / "install-cache";
    return wse::platform::CachingVfs::clear(cache_root) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jlong JNICALL Java_org_wse2_android_launcher_R4Native_textureCacheBytes(
    JNIEnv* env, jclass, jstring jwork_dir) {
    const std::string raw = jstring_to_utf8(env, jwork_dir);
    if (raw.empty()) return 0;
    const std::filesystem::path cache_root = std::filesystem::path(raw) / "r4" / "texcache";
    return static_cast<jlong>(wse::platform::CachingVfs::disk_usage(cache_root));
}

extern "C" JNIEXPORT jboolean JNICALL Java_org_wse2_android_launcher_R4Native_clearTextureCache(
    JNIEnv* env, jclass, jstring jwork_dir) {
    const std::string raw = jstring_to_utf8(env, jwork_dir);
    if (raw.empty()) return JNI_FALSE;
    const std::filesystem::path cache_root = std::filesystem::path(raw) / "r4" / "texcache";
    return wse::platform::CachingVfs::clear(cache_root) ? JNI_TRUE : JNI_FALSE;
}

// ---- Phase 5 R6: Touch, Keyboard, Mouse, Gamepad, Gyro & Audio JNI entries ----

extern "C" JNIEXPORT void JNICALL Java_org_wse2_android_launcher_R4Native_nativeTouchEvent(
    JNIEnv*, jclass, jint pointer_id, jint action, jfloat x, jfloat y) {
    wse::recomp::shim::RecompInputHub::shared().on_touch_event(
        static_cast<int>(pointer_id), static_cast<int>(action), static_cast<float>(x), static_cast<float>(y));
}

extern "C" JNIEXPORT void JNICALL Java_org_wse2_android_launcher_R4Native_nativeKeyEvent(
    JNIEnv*, jclass, jint dik_code, jint vk_code, jboolean down) {
    wse::recomp::shim::RecompInputHub::shared().inject_key(
        static_cast<std::uint8_t>(dik_code & 0xFF), down == JNI_TRUE, static_cast<std::uint32_t>(vk_code));
}

extern "C" JNIEXPORT void JNICALL Java_org_wse2_android_launcher_R4Native_nativeTextInput(
    JNIEnv* env, jclass, jstring jutf8) {
    const std::string utf8 = jstring_to_utf8(env, jutf8);
    if (!utf8.empty()) {
        wse::recomp::shim::RecompInputHub::shared().inject_text_utf8(utf8);
    }
}

extern "C" JNIEXPORT void JNICALL Java_org_wse2_android_launcher_R4Native_nativeMouseMove(
    JNIEnv*, jclass, jint x, jint y, jboolean relative) {
    if (relative == JNI_TRUE) {
        wse::recomp::shim::RecompInputHub::shared().inject_mouse_delta(static_cast<int>(x), static_cast<int>(y));
    } else {
        wse::recomp::shim::RecompInputHub::shared().inject_mouse_move_abs(static_cast<int>(x), static_cast<int>(y));
    }
}

extern "C" JNIEXPORT void JNICALL Java_org_wse2_android_launcher_R4Native_nativeMouseButton(
    JNIEnv*, jclass, jint button, jboolean down) {
    wse::recomp::shim::RecompInputHub::shared().inject_mouse_button(static_cast<int>(button), down == JNI_TRUE);
}

extern "C" JNIEXPORT void JNICALL Java_org_wse2_android_launcher_R4Native_nativeMouseWheel(
    JNIEnv*, jclass, jint delta) {
    wse::recomp::shim::RecompInputHub::shared().inject_mouse_wheel(static_cast<int>(delta));
}

extern "C" JNIEXPORT void JNICALL Java_org_wse2_android_launcher_R4Native_nativeGamepadState(
    JNIEnv*, jclass, jint user_index, jboolean connected, jint buttons, jint left_trigger, jint right_trigger,
    jint thumb_lx, jint thumb_ly, jint thumb_rx, jint thumb_ry) {
    wse::recomp::shim::XInputPadState st{};
    st.connected = (connected == JNI_TRUE);
    st.buttons = static_cast<std::uint16_t>(buttons & 0xFFFF);
    st.left_trigger = static_cast<std::uint8_t>(std::clamp<int>(left_trigger, 0, 255));
    st.right_trigger = static_cast<std::uint8_t>(std::clamp<int>(right_trigger, 0, 255));
    st.thumb_lx = static_cast<std::int16_t>(std::clamp<int>(thumb_lx, -32768, 32767));
    st.thumb_ly = static_cast<std::int16_t>(std::clamp<int>(thumb_ly, -32768, 32767));
    st.thumb_rx = static_cast<std::int16_t>(std::clamp<int>(thumb_rx, -32768, 32767));
    st.thumb_ry = static_cast<std::int16_t>(std::clamp<int>(thumb_ry, -32768, 32767));
    wse::recomp::shim::RecompInputHub::shared().update_gamepad(static_cast<std::uint32_t>(std::max<jint>(0, user_index)), st);
}

extern "C" JNIEXPORT void JNICALL Java_org_wse2_android_launcher_R4Native_nativeGyroSample(
    JNIEnv*, jclass, jfloat rx, jfloat ry, jfloat dt_sec, jboolean rotated_270) {
    wse::recomp::shim::RecompInputHub::shared().on_gyro_sample(
        static_cast<float>(rx), static_cast<float>(ry), static_cast<float>(dt_sec), rotated_270 == JNI_TRUE);
}

extern "C" JNIEXPORT void JNICALL Java_org_wse2_android_launcher_R4Native_nativeSetSurfaceSize(
    JNIEnv*, jclass, jint width, jint height) {
    if (width > 0 && height > 0) {
        wse::recomp::shim::RecompInputHub::shared().set_surface_size(static_cast<int>(width), static_cast<int>(height));
    }
}

extern "C" JNIEXPORT void JNICALL Java_org_wse2_android_launcher_R4Native_nativeSetSafeArea(
    JNIEnv*, jclass, jfloat left, jfloat top, jfloat right, jfloat bottom) {
    wse::platform::SafeAreaInsets in{};
    in.left = std::clamp(static_cast<float>(left), 0.0F, 0.25F);
    in.top = std::clamp(static_cast<float>(top), 0.0F, 0.25F);
    in.right = std::clamp(static_cast<float>(right), 0.0F, 0.25F);
    in.bottom = std::clamp(static_cast<float>(bottom), 0.0F, 0.25F);
    wse::recomp::shim::RecompInputHub::shared().set_safe_area(in);
}

extern "C" JNIEXPORT void JNICALL Java_org_wse2_android_launcher_R4Native_nativeSetTouchConfig(
    JNIEnv*, jclass, jfloat opacity, jfloat button_scale, jfloat look_sens, jboolean aim_hold,
    jboolean gyro_enabled, jboolean show_touch) {
    auto cfg = wse::recomp::shim::RecompInputHub::shared().config();
    cfg.opacity = static_cast<float>(opacity);
    cfg.button_scale = static_cast<float>(button_scale);
    cfg.look_sensitivity = static_cast<float>(look_sens);
    cfg.aim_hold = (aim_hold == JNI_TRUE);
    cfg.gyro_enabled = (gyro_enabled == JNI_TRUE);
    cfg.show_touch_controls = (show_touch == JNI_TRUE);
    wse::recomp::shim::RecompInputHub::shared().set_config(cfg);
}

extern "C" JNIEXPORT void JNICALL Java_org_wse2_android_launcher_R4Native_nativeSetAudioConfig(
    JNIEnv*, jclass, jfloat master_volume, jboolean haptics_enabled, jboolean when_hit,
    jboolean when_dealing, jint haptic_strength) {
    const std::lock_guard<std::mutex> lock(g_audio_m);
    g_audio_master_volume = std::clamp(static_cast<float>(master_volume), 0.0F, 1.0F);
    g_haptics_cfg.enabled = (haptics_enabled == JNI_TRUE);
    g_haptics_cfg.when_hit = (when_hit == JNI_TRUE);
    g_haptics_cfg.when_dealing = (when_dealing == JNI_TRUE);
    g_haptics_cfg.strength_percent = std::clamp<int>(static_cast<int>(haptic_strength), 0, 100);
    if (g_active_fmod != nullptr) {
        g_active_fmod->set_master_volume(g_audio_master_volume);
        g_active_fmod->set_haptics_settings(g_haptics_cfg);
    }
}

extern "C" JNIEXPORT jboolean JNICALL Java_org_wse2_android_launcher_R4Native_nativeIsCursorVisible(JNIEnv*, jclass) {
    return wse::recomp::shim::RecompInputHub::shared().cursor_visible() ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL Java_org_wse2_android_launcher_R4Native_nativeSetGraphicsConfig(
    JNIEnv*, jclass, jfloat render_scale, jboolean auto_resolution, jfloat sharpness, jboolean fxaa,
    jfloat target_fps) {
    std::shared_ptr<wse::platform::VulkanD3d9Backend> bk;
    {
        const std::lock_guard<std::mutex> lock(g_vk_m);
        bk = g_vk_backend;
        if (g_adpf_session && target_fps > 0.0F) {
            const auto ns = static_cast<std::int64_t>(1e9 / static_cast<double>(std::clamp(static_cast<float>(target_fps), 15.0F, 240.0F)));
            g_adpf_session->update_target(ns);
        }
    }
    if (bk) {
        bk->set_graphics_options(static_cast<float>(render_scale), auto_resolution == JNI_TRUE,
                                 static_cast<float>(sharpness), fxaa == JNI_TRUE, static_cast<float>(target_fps));
    }
}

extern "C" JNIEXPORT void JNICALL Java_org_wse2_android_launcher_R4Native_nativeSetLayoutEditMode(
    JNIEnv*, jclass, jboolean enabled) {
    wse::recomp::shim::RecompInputHub::shared().set_edit_mode(enabled == JNI_TRUE);
}

extern "C" JNIEXPORT void JNICALL Java_org_wse2_android_launcher_R4Native_nativeResetTouchLayout(JNIEnv*, jclass) {
    wse::recomp::shim::RecompInputHub::shared().reset_layout_to_defaults();
}

extern "C" JNIEXPORT jstring JNICALL Java_org_wse2_android_launcher_R4Native_nativeExportTouchLayoutJson(
    JNIEnv* env, jclass) {
    return utf8_to_jstring(env, wse::recomp::shim::RecompInputHub::shared().export_layout_json());
}

extern "C" JNIEXPORT jboolean JNICALL Java_org_wse2_android_launcher_R4Native_nativeImportTouchLayoutJson(
    JNIEnv* env, jclass, jstring jjson) {
    const std::string json = jstring_to_utf8(env, jjson);
    return wse::recomp::shim::RecompInputHub::shared().import_layout_json(json) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL Java_org_wse2_android_launcher_R4Native_nativeAppendReportLine(
    JNIEnv* env, jclass, jstring jline) {
    const std::string line = jstring_to_utf8(env, jline);
    if (!line.empty()) {
        report(line.back() == '\n' ? line : (line + "\n"));
    }
}
