// The two delay-loaded DLLs of v1.1.5.6 (Phase 5 R4 -> R6, `docs/RECOMP_PLAN.md` "R6 — Audio,
// Input, Touch Controls & Desktop Mode"):
//
//   fmodex64.dll     the 54 FMOD Ex 4.44.64 exports the exe imports: FMOD_System_Create and member
//                    functions of FMOD::System, Sound, Channel and Geometry (MSVC x64 names).
//                    R6 backs these with real 2D/3D stereo mixing, WAV/OGG/MP3 decoding &
//                    streaming (`wse/asset/audio.h`), an asynchronous audio worker thread,
//                    background pause/resume, safe `System::update()` channel-end callbacks, and
//                    nearby-combat haptic pulse classification (`wse/recomp/combat_haptics.h`).
//   steam_api64.dll  the 10 SteamAPI_* / SteamInternal_* functions (behaves as Steam not running).
//
// Header-only: uses built-in RIFF/WAVE decoding by default so standalone tests incur zero extra
// link dependencies; embedders pass `wse::asset::create_audio_codec_backend()` (from `wse_audio`)
// for OGG Vorbis + MP3 decoding and an `AudioDevice*` (e.g. Android AAudio or CaptureAudioDevice)
// for real-time audio output via `FmodSteam::created_hook()`.

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "wse/asset/audio.h"
#include "wse/platform/audio.h"
#include "wse/recomp/combat_haptics.h"
#include "wse/recomp/cpu_state.h"
#include "wse/recomp/host_stack.h"
#include "wse/recomp/runtime.h"
#include "wse/recomp/shim_crt.h"

namespace wse::recomp::shim {

inline constexpr std::uint32_t kFmodVersion = 0x00044464U;  // FMOD Ex 4.44.64
inline constexpr std::uint32_t kFmodSpeakerModeStereo = 2U;

// FMOD Ex 4.44 FMOD_RESULT codes used by the shim.
inline constexpr std::uint32_t kFmodOk = 0U;
inline constexpr std::uint32_t kFmodErrFileNotFound = 23U;
inline constexpr std::uint32_t kFmodErrFormat = 25U;
inline constexpr std::uint32_t kFmodErrInvalidHandle = 36U;
inline constexpr std::uint32_t kFmodErrInvalidParam = 37U;

// FMOD Ex 4.44 FMOD_MODE bits.
inline constexpr std::uint32_t kFmodLoopOff = 0x00000001U;
inline constexpr std::uint32_t kFmodLoopNormal = 0x00000002U;
inline constexpr std::uint32_t kFmodLoopBidi = 0x00000004U;
inline constexpr std::uint32_t kFmod2D = 0x00000008U;
inline constexpr std::uint32_t kFmod3D = 0x00000010U;
inline constexpr std::uint32_t kFmodCreateStream = 0x00000080U;
inline constexpr std::uint32_t kFmodOpenMemory = 0x00000800U;
inline constexpr std::uint32_t kFmodOpenRaw = 0x00001000U;
inline constexpr std::uint32_t kFmodOpenMemoryPoint = 0x10000000U;
inline constexpr std::uint32_t kFmod3DHeadRelative = 0x00040000U;
inline constexpr std::uint32_t kFmod3DLinearRolloff = 0x00200000U;
inline constexpr std::uint32_t kFmod3DIgnoreGeometry = 0x40000000U;

// FMOD Ex 4.44 FMOD_TIMEUNIT bits.
inline constexpr std::uint32_t kFmodTimeUnitMs = 0x00000001U;
inline constexpr std::uint32_t kFmodTimeUnitPcm = 0x00000002U;
inline constexpr std::uint32_t kFmodTimeUnitPcmBytes = 0x00000004U;

// FMOD Ex 4.44 FMOD_CHANNELINDEX sentinels & callback type.
inline constexpr std::int32_t kFmodChannelFree = -1;
inline constexpr std::int32_t kFmodChannelReuse = -2;
inline constexpr std::uint32_t kFmodCallbackTypeEnd = 0U;

class FmodSteam {
public:
    using ReadFileFn = std::function<bool(const std::string& path, std::vector<std::uint8_t>& out)>;
    using HapticFn = std::function<void(const CombatHapticPulse& pulse)>;
    using CreatedHook = std::function<void(FmodSteam&)>;

    static CreatedHook& created_hook() {
        static CreatedHook hook;
        return hook;
    }

    explicit FmodSteam(Runtime& rt) : rt_(rt) {
        init_channel_pool_locked(max_channels_);
        if (auto& hook = created_hook()) {
            hook(*this);
        }
    }
    FmodSteam(const FmodSteam&) = delete;
    FmodSteam& operator=(const FmodSteam&) = delete;
    ~FmodSteam() { stop_worker(); }

    // Embedder hooks (safe to configure from `created_hook()` or before `install()`).
    ReadFileFn read_file;
    HapticFn on_haptic;
    std::function<bool()> on_device_recover;

    void set_codec_backend(std::shared_ptr<wse::asset::AudioCodecBackend> backend) {
        const std::lock_guard<std::mutex> lock(m_);
        codec_backend_ = std::move(backend);
    }

    // Attaches an audio output device. Does not take ownership; caller keeps `device` alive until
    // `stop_worker()` or destruction. Setting `start_thread = true` launches the background mixer
    // thread (never mixes on the game thread).
    void set_audio_device(wse::platform::AudioDevice* device, bool start_thread = true) {
        stop_worker();
        {
            const std::lock_guard<std::mutex> lock(m_);
            audio_device_ = device;
            if (device != nullptr && device->is_open()) {
                output_sample_rate_ = device->spec().sample_rate;
            }
        }
        if (device != nullptr && device->is_open() && start_thread) {
            start_worker();
        }
    }

    void set_master_volume(float vol) noexcept {
        master_volume_.store(std::clamp(std::isfinite(vol) ? vol : 1.0F, 0.0F, 1.0F), std::memory_order_relaxed);
    }
    [[nodiscard]] float master_volume() const noexcept {
        return master_volume_.load(std::memory_order_relaxed);
    }

    void set_haptics_settings(const CombatHapticsSettings& settings) {
        const std::lock_guard<std::mutex> lock(m_);
        haptics_.set_settings(settings);
    }

    // Pauses or resumes audio output immediately when the app moves to/from background.
    void set_app_paused(bool paused) {
        {
            const std::lock_guard<std::mutex> lock(m_);
            if (app_paused_ == paused) {
                return;
            }
            app_paused_ = paused;
            if (audio_device_ != nullptr && audio_device_->is_open()) {
                audio_device_->set_paused(paused);
            }
        }
        worker_cv_.notify_all();
    }
    [[nodiscard]] bool app_paused() const {
        const std::lock_guard<std::mutex> lock(m_);
        return app_paused_;
    }

    void start_worker() {
        if (worker_.joinable()) {
            return;
        }
        {
            const std::lock_guard<std::mutex> lock(m_);
            stop_worker_ = false;
        }
        worker_ = HostThread([this] { worker_loop(); });
    }

    void stop_worker() {
        {
            const std::lock_guard<std::mutex> lock(m_);
            stop_worker_ = true;
        }
        worker_cv_.notify_all();
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    // Mixes `frame_count` interleaved stereo float frames synchronously (used by the background
    // worker and deterministic host tests).
    void mix_frames(std::size_t frame_count, std::vector<float>& out_interleaved_stereo) {
        out_interleaved_stereo.assign(frame_count * 2U, 0.0F);
        if (frame_count == 0U) {
            return;
        }
        const std::lock_guard<std::mutex> lock(m_);
        if (app_paused_) {
            return;
        }
        mix_locked(frame_count, out_interleaved_stereo.data());
    }

    // Pumps audio into the attached `AudioDevice` up to `target_queued_frames` (one synchronous
    // step for tests or non-threaded hosts).
    std::size_t pump_once(std::size_t target_queued_frames = 2048U, std::size_t max_block_frames = 512U) {
        const std::lock_guard<std::mutex> lock(m_);
        return pump_locked(target_queued_frames, max_block_frames);
    }

    [[nodiscard]] std::size_t sounds_created() const noexcept { return sounds_created_.load(std::memory_order_relaxed); }
    [[nodiscard]] std::size_t active_channel_count() const {
        const std::lock_guard<std::mutex> lock(m_);
        std::size_t n = 0U;
        for (const auto& c : channels_) {
            if (c.playing) ++n;
        }
        return n;
    }

    std::size_t install() {
        std::size_t n = 0;
        auto fmod = [&](const char* name, Runtime::ImportShim shim) {
            n += rt_.set_import("fmodex64.dll", name, std::move(shim));
        };
        auto ok = [](CpuState& s) { win64_return(s, kFmodOk); };

        // FMOD_System_Create(FMOD::System**)
        fmod("FMOD_System_Create", [this](CpuState& s) {
            const std::uint64_t out = win64_arg(s, 0);
            if (out == 0) {
                win64_return(s, kFmodErrInvalidParam);
                return;
            }
            crt_detail::put<std::uint64_t>(out, next_handle());
            win64_return(s, kFmodOk);
        });

        // System::init(this, maxchannels, flags, extradriverdata)
        fmod("?init@System@FMOD@@QEAA?AW4FMOD_RESULT@@HIPEAX@Z", [this](CpuState& s) {
            const auto max_ch = static_cast<std::int32_t>(win64_arg(s, 1));
            {
                const std::lock_guard<std::mutex> lock(m_);
                max_channels_ = std::clamp(max_ch > 0 ? max_ch : 64, 1, 256);
                init_channel_pool_locked(max_channels_);
            }
            win64_return(s, kFmodOk);
        });

        // System::close / release
        fmod("?close@System@FMOD@@QEAA?AW4FMOD_RESULT@@XZ", [this](CpuState& s) {
            {
                const std::lock_guard<std::mutex> lock(m_);
                init_channel_pool_locked(max_channels_);
                sounds_.clear();
                pending_end_callbacks_.clear();
            }
            win64_return(s, kFmodOk);
        });
        fmod("?release@System@FMOD@@QEAA?AW4FMOD_RESULT@@XZ", [this](CpuState& s) {
            {
                const std::lock_guard<std::mutex> lock(m_);
                init_channel_pool_locked(max_channels_);
                sounds_.clear();
                pending_end_callbacks_.clear();
            }
            win64_return(s, kFmodOk);
        });

        // System::setSoftwareFormat(this, samplerate, format, numoutputchannels, maxinputchannels, resamplemethod)
        fmod("?setSoftwareFormat@System@FMOD@@QEAA?AW4FMOD_RESULT@@HW4FMOD_SOUND_FORMAT@@HHW4FMOD_DSP_RESAMPLER@@@Z",
             [this](CpuState& s) {
                 const auto sr = static_cast<std::int32_t>(win64_arg(s, 1));
                 if (sr >= 8000 && sr <= static_cast<std::int32_t>(wse::asset::kMaxAudioSampleRate)) {
                     const std::lock_guard<std::mutex> lock(m_);
                     if (audio_device_ == nullptr || !audio_device_->is_open()) {
                         output_sample_rate_ = static_cast<std::uint32_t>(sr);
                     }
                 }
                 win64_return(s, kFmodOk);
             });

        // System::set3DSettings(this, dopplerscale, distancefactor, rolloffscale)
        fmod("?set3DSettings@System@FMOD@@QEAA?AW4FMOD_RESULT@@MMM@Z", [this](CpuState& s) {
            const float dist_factor = win64_arg_f32(s, 2);
            const float rolloff = win64_arg_f32(s, 3);
            const std::lock_guard<std::mutex> lock(m_);
            if (std::isfinite(dist_factor) && dist_factor > 1e-4F) {
                distance_factor_ = dist_factor;
            }
            if (std::isfinite(rolloff) && rolloff >= 0.0F) {
                rolloff_scale_ = rolloff;
            }
            win64_return(s, kFmodOk);
        });

        // System::set3DListenerAttributes(this, listener, pos, vel, forward, up)
        fmod("?set3DListenerAttributes@System@FMOD@@QEAA?AW4FMOD_RESULT@@HPEBUFMOD_VECTOR@@000@Z",
             [this](CpuState& s) {
                 const auto listener_id = static_cast<std::int32_t>(win64_arg(s, 1));
                 if (listener_id != 0) {
                     win64_return(s, kFmodOk);
                     return;
                 }
                 const std::uint64_t p_pos = win64_arg(s, 2);
                 const std::uint64_t p_vel = win64_arg(s, 3);
                 const std::uint64_t p_fwd = win64_arg(s, 4);
                 const std::uint64_t p_up = win64_arg(s, 5);
                 const std::lock_guard<std::mutex> lock(m_);
                 if (p_pos != 0) read_vec3(p_pos, listener_.position);
                 if (p_vel != 0) read_vec3(p_vel, listener_.velocity);
                 if (p_fwd != 0) {
                     read_vec3(p_fwd, listener_.forward);
                     if (listener_.forward.length() > 1e-6F) listener_.forward.normalize();
                 }
                 if (p_up != 0) {
                     read_vec3(p_up, listener_.up);
                     if (listener_.up.length() > 1e-6F) listener_.up.normalize();
                 }
                 // FMOD Ex uses a left-handed 3D coordinate system by default: right = up x forward.
                 wse::core::WseVec4 r = listener_.up.cross(listener_.forward);
                 if (r.length() > 1e-6F) {
                     r.normalize();
                     listener_.right = r;
                 }
                 win64_return(s, kFmodOk);
             });

        // System::update(this) — delivers finished-channel callbacks safely on the game thread.
        fmod("?update@System@FMOD@@QEAA?AW4FMOD_RESULT@@XZ", [this](CpuState& s) {
            std::vector<PendingCallback> cbs;
            {
                const std::lock_guard<std::mutex> lock(m_);
                cbs.swap(pending_end_callbacks_);
            }
            for (const PendingCallback& cb : cbs) {
                if (cb.fn_va != 0 && cb.channel != 0) {
                    (void)rt_.call_guest(s, cb.fn_va, {cb.channel, kFmodCallbackTypeEnd, 0, 0});
                }
            }
            win64_return(s, kFmodOk);
        });

        // System::createSound / createStream(this, name_or_data, mode, exinfo, Sound**)
        fmod("?createSound@System@FMOD@@QEAA?AW4FMOD_RESULT@@PEBDIPEAUFMOD_CREATESOUNDEXINFO@@PEAPEAVSound@2@@Z",
             [this](CpuState& s) { handle_create_sound(s, false); });
        fmod("?createStream@System@FMOD@@QEAA?AW4FMOD_RESULT@@PEBDIPEAUFMOD_CREATESOUNDEXINFO@@PEAPEAVSound@2@@Z",
             [this](CpuState& s) { handle_create_sound(s, true); });

        // System::createGeometry(this, maxpolygons, maxvertices, Geometry**)
        fmod("?createGeometry@System@FMOD@@QEAA?AW4FMOD_RESULT@@HHPEAPEAVGeometry@2@@Z", [this](CpuState& s) {
            const std::uint64_t out = win64_arg(s, 3);
            if (out != 0) crt_detail::put<std::uint64_t>(out, next_handle());
            win64_return(s, kFmodOk);
        });

        // System::playSound(this, channelid, Sound*, paused, Channel**)
        fmod("?playSound@System@FMOD@@QEAA?AW4FMOD_RESULT@@W4FMOD_CHANNELINDEX@@PEAVSound@2@_NPEAPEAVChannel@2@@Z",
             [this](CpuState& s) { handle_play_sound(s); });

        // System::getChannel(this, channelid, Channel**)
        fmod("?getChannel@System@FMOD@@QEAA?AW4FMOD_RESULT@@HPEAPEAVChannel@2@@Z", [this](CpuState& s) {
            const auto ch_idx = static_cast<std::int32_t>(win64_arg(s, 1));
            const std::uint64_t out = win64_arg(s, 2);
            if (out == 0) {
                win64_return(s, kFmodErrInvalidParam);
                return;
            }
            const std::lock_guard<std::mutex> lock(m_);
            if (ch_idx < 0 || static_cast<std::size_t>(ch_idx) >= channels_.size()) {
                crt_detail::put<std::uint64_t>(out, 0U);
                win64_return(s, kFmodErrInvalidParam);
                return;
            }
            Channel& c = channels_[static_cast<std::size_t>(ch_idx)];
            if (!c.handle_valid || c.handle == 0U) {
                c.handle = make_channel_handle(static_cast<std::size_t>(ch_idx), ++generation_);
                c.handle_valid = true;
            }
            crt_detail::put<std::uint64_t>(out, c.handle);
            win64_return(s, kFmodOk);
        });

        fmod("?getVersion@System@FMOD@@QEAA?AW4FMOD_RESULT@@PEAI@Z", [](CpuState& s) {
            if (win64_arg(s, 1) != 0) crt_detail::put<std::uint32_t>(win64_arg(s, 1), kFmodVersion);
            win64_return(s, kFmodOk);
        });
        fmod("?getNumDrivers@System@FMOD@@QEAA?AW4FMOD_RESULT@@PEAH@Z", [](CpuState& s) {
            if (win64_arg(s, 1) != 0) crt_detail::put<std::int32_t>(win64_arg(s, 1), 1);
            win64_return(s, kFmodOk);
        });
        // System::getDriverInfo(this, id, char* name, namelen, FMOD_GUID*)
        fmod("?getDriverInfo@System@FMOD@@QEAA?AW4FMOD_RESULT@@HPEADHPEAUFMOD_GUID@@@Z", [this](CpuState& s) {
            const char* label = "WSE2 Audio Mixer";
            {
                const std::lock_guard<std::mutex> lock(m_);
                if (audio_device_ != nullptr && audio_device_->is_open()) {
                    label = audio_device_->backend_name();
                }
            }
            const std::uint64_t name = win64_arg(s, 2);
            const auto len = static_cast<std::int32_t>(win64_arg(s, 3));
            if (name != 0 && len > 0) {
                const std::size_t slen = std::strlen(label);
                const std::size_t k = std::min<std::size_t>(static_cast<std::size_t>(len) - 1U, slen);
                std::memcpy(reinterpret_cast<void*>(static_cast<std::uintptr_t>(name)), label, k);
                crt_detail::put<char>(name + k, 0);
            }
            if (win64_arg(s, 4) != 0) {
                std::memset(reinterpret_cast<void*>(static_cast<std::uintptr_t>(win64_arg(s, 4))), 0, 16);
            }
            win64_return(s, kFmodOk);
        });
        // System::getDriverCaps(this, id, unsigned* caps, int* controlpaneloutputrate, FMOD_SPEAKERMODE*)
        fmod("?getDriverCaps@System@FMOD@@QEAA?AW4FMOD_RESULT@@HPEAIPEAHPEAW4FMOD_SPEAKERMODE@@@Z", [this](CpuState& s) {
            std::uint32_t sr = 48000U;
            {
                const std::lock_guard<std::mutex> lock(m_);
                sr = output_sample_rate_;
            }
            if (win64_arg(s, 2) != 0) crt_detail::put<std::uint32_t>(win64_arg(s, 2), 0U);
            if (win64_arg(s, 3) != 0) crt_detail::put<std::int32_t>(win64_arg(s, 3), static_cast<std::int32_t>(sr));
            if (win64_arg(s, 4) != 0) crt_detail::put<std::uint32_t>(win64_arg(s, 4), kFmodSpeakerModeStereo);
            win64_return(s, kFmodOk);
        });
        fmod("?getChannelsPlaying@System@FMOD@@QEAA?AW4FMOD_RESULT@@PEAH@Z", [this](CpuState& s) {
            if (win64_arg(s, 1) != 0) {
                const std::lock_guard<std::mutex> lock(m_);
                std::int32_t count = 0;
                for (const auto& c : channels_) {
                    if (c.playing) ++count;
                }
                crt_detail::put<std::int32_t>(win64_arg(s, 1), count);
            }
            win64_return(s, kFmodOk);
        });

        // Sound methods
        fmod("?getOpenState@Sound@FMOD@@QEAA?AW4FMOD_RESULT@@PEAW4FMOD_OPENSTATE@@PEAIPEA_N2@Z", [](CpuState& s) {
            if (win64_arg(s, 1) != 0) crt_detail::put<std::uint32_t>(win64_arg(s, 1), 0U);  // FMOD_OPENSTATE_READY
            if (win64_arg(s, 2) != 0) crt_detail::put<std::uint32_t>(win64_arg(s, 2), 100U);
            if (win64_arg(s, 3) != 0) crt_detail::put<std::uint8_t>(win64_arg(s, 3), 0U);
            if (win64_arg(s, 4) != 0) crt_detail::put<std::uint8_t>(win64_arg(s, 4), 0U);
            win64_return(s, kFmodOk);
        });
        fmod("?getLength@Sound@FMOD@@QEAA?AW4FMOD_RESULT@@PEAII@Z", [this](CpuState& s) {
            const std::uint64_t h = win64_arg(s, 0);
            const std::uint64_t out = win64_arg(s, 1);
            const auto unit = static_cast<std::uint32_t>(win64_arg(s, 2));
            if (out != 0) {
                const std::lock_guard<std::mutex> lock(m_);
                const auto it = sounds_.find(h);
                if (it != sounds_.end()) {
                    crt_detail::put<std::uint32_t>(out, sound_length_in_unit(*it->second, unit));
                } else {
                    crt_detail::put<std::uint32_t>(out, 1000U);
                }
            }
            win64_return(s, kFmodOk);
        });
        fmod("?getDefaults@Sound@FMOD@@QEAA?AW4FMOD_RESULT@@PEAM00PEAH@Z", [this](CpuState& s) {
            const std::lock_guard<std::mutex> lock(m_);
            const auto it = sounds_.find(win64_arg(s, 0));
            const float freq = (it != sounds_.end()) ? it->second->default_frequency : 44100.0F;
            const float vol = (it != sounds_.end()) ? it->second->default_volume : 1.0F;
            const float pan = (it != sounds_.end()) ? it->second->default_pan : 0.0F;
            const std::int32_t prio = (it != sounds_.end()) ? it->second->default_priority : 128;
            if (win64_arg(s, 1) != 0) crt_detail::put<float>(win64_arg(s, 1), freq);
            if (win64_arg(s, 2) != 0) crt_detail::put<float>(win64_arg(s, 2), vol);
            if (win64_arg(s, 3) != 0) crt_detail::put<float>(win64_arg(s, 3), pan);
            if (win64_arg(s, 4) != 0) crt_detail::put<std::int32_t>(win64_arg(s, 4), prio);
            win64_return(s, kFmodOk);
        });
        fmod("?setDefaults@Sound@FMOD@@QEAA?AW4FMOD_RESULT@@MMMH@Z", [this](CpuState& s) {
            const std::lock_guard<std::mutex> lock(m_);
            const auto it = sounds_.find(win64_arg(s, 0));
            if (it != sounds_.end()) {
                const float freq = win64_arg_f32(s, 1);
                const float vol = win64_arg_f32(s, 2);
                const float pan = win64_arg_f32(s, 3);
                const auto prio = static_cast<std::int32_t>(win64_arg(s, 4));
                if (std::isfinite(freq) && freq > 0.0F) it->second->default_frequency = freq;
                if (std::isfinite(vol)) it->second->default_volume = std::clamp(vol, 0.0F, 1.0F);
                if (std::isfinite(pan)) it->second->default_pan = std::clamp(pan, -1.0F, 1.0F);
                it->second->default_priority = std::clamp(prio, 0, 256);
            }
            win64_return(s, kFmodOk);
        });
        fmod("?set3DMinMaxDistance@Sound@FMOD@@QEAA?AW4FMOD_RESULT@@MM@Z", [this](CpuState& s) {
            const std::lock_guard<std::mutex> lock(m_);
            const auto it = sounds_.find(win64_arg(s, 0));
            if (it != sounds_.end()) {
                const float mn = win64_arg_f32(s, 1);
                const float mx = win64_arg_f32(s, 2);
                if (std::isfinite(mn) && mn > 1e-3F) it->second->min_dist = mn;
                if (std::isfinite(mx) && mx >= it->second->min_dist) it->second->max_dist = mx;
            }
            win64_return(s, kFmodOk);
        });
        fmod("?release@Sound@FMOD@@QEAA?AW4FMOD_RESULT@@XZ", [this](CpuState& s) {
            const std::uint64_t h = win64_arg(s, 0);
            const std::lock_guard<std::mutex> lock(m_);
            for (auto& c : channels_) {
                if (c.sound_handle == h) {
                    stop_channel_locked(c, false);
                }
            }
            sounds_.erase(h);
            win64_return(s, kFmodOk);
        });

        // Channel methods (all validate generation-checked pool handles; stale handles return FMOD_ERR_INVALID_HANDLE).
        fmod("?isPlaying@Channel@FMOD@@QEAA?AW4FMOD_RESULT@@PEA_N@Z", [this](CpuState& s) {
            const std::uint64_t out = win64_arg(s, 1);
            const std::lock_guard<std::mutex> lock(m_);
            Channel* c = resolve_channel_locked(win64_arg(s, 0));
            if (c == nullptr) {
                if (out != 0) crt_detail::put<std::uint8_t>(out, 0U);
                win64_return(s, kFmodErrInvalidHandle);
                return;
            }
            if (out != 0) {
                crt_detail::put<std::uint8_t>(out, c->playing ? 1U : 0U);
            }
            win64_return(s, kFmodOk);
        });
        fmod("?stop@Channel@FMOD@@QEAA?AW4FMOD_RESULT@@XZ", [this](CpuState& s) {
            const std::lock_guard<std::mutex> lock(m_);
            Channel* c = resolve_channel_locked(win64_arg(s, 0));
            if (c == nullptr) {
                win64_return(s, kFmodErrInvalidHandle);
                return;
            }
            stop_channel_locked(*c, true);
            win64_return(s, kFmodOk);
        });
        fmod("?setPaused@Channel@FMOD@@QEAA?AW4FMOD_RESULT@@_N@Z", [this](CpuState& s) {
            const bool paused = (win64_arg(s, 1) & 0xFFU) != 0U;
            CombatHapticPulse pulse{};
            HapticFn haptic_cb;
            {
                const std::lock_guard<std::mutex> lock(m_);
                Channel* c = resolve_channel_locked(win64_arg(s, 0));
                if (c == nullptr) {
                    win64_return(s, kFmodErrInvalidHandle);
                    return;
                }
                const bool was_paused = c->paused;
                c->paused = paused;
                if (was_paused && !paused && c->playing && !c->haptic_fired) {
                    pulse = maybe_trigger_haptic_locked(*c);
                    haptic_cb = on_haptic;
                }
            }
            if (pulse.kind != CombatHapticKind::none && haptic_cb) {
                haptic_cb(pulse);
            }
            win64_return(s, kFmodOk);
        });
        fmod("?setPosition@Channel@FMOD@@QEAA?AW4FMOD_RESULT@@II@Z", [this](CpuState& s) {
            const auto pos = static_cast<std::uint32_t>(win64_arg(s, 1));
            const auto unit = static_cast<std::uint32_t>(win64_arg(s, 2));
            const std::lock_guard<std::mutex> lock(m_);
            Channel* c = resolve_channel_locked(win64_arg(s, 0));
            if (c == nullptr) {
                win64_return(s, kFmodErrInvalidHandle);
                return;
            }
            if (c->sound != nullptr) {
                const double target_frame = unit_to_frame(*c->sound, pos, unit);
                if (c->sound->is_stream) {
                    seek_stream_channel_locked(*c, static_cast<std::size_t>(std::max(0.0, target_frame)));
                } else {
                    const std::size_t total = c->sound->total_frames;
                    c->cursor_frame =
                        total > 0U ? std::clamp(target_frame, 0.0, static_cast<double>(total - 1U)) : 0.0;
                }
            }
            win64_return(s, kFmodOk);
        });
        fmod("?getPosition@Channel@FMOD@@QEAA?AW4FMOD_RESULT@@PEAII@Z", [this](CpuState& s) {
            const std::uint64_t out = win64_arg(s, 1);
            const auto unit = static_cast<std::uint32_t>(win64_arg(s, 2));
            const std::lock_guard<std::mutex> lock(m_);
            Channel* c = resolve_channel_locked(win64_arg(s, 0));
            if (c == nullptr) {
                if (out != 0) crt_detail::put<std::uint32_t>(out, 0U);
                win64_return(s, kFmodErrInvalidHandle);
                return;
            }
            if (out != 0) {
                const std::uint32_t val =
                    (c->sound != nullptr) ? frame_to_unit(*c->sound, c->cursor_frame, unit) : 0U;
                crt_detail::put<std::uint32_t>(out, val);
            }
            win64_return(s, kFmodOk);
        });
        fmod("?setFrequency@Channel@FMOD@@QEAA?AW4FMOD_RESULT@@M@Z", [this](CpuState& s) {
            const float freq = win64_arg_f32(s, 1);
            const std::lock_guard<std::mutex> lock(m_);
            Channel* c = resolve_channel_locked(win64_arg(s, 0));
            if (c == nullptr) {
                win64_return(s, kFmodErrInvalidHandle);
                return;
            }
            if (std::isfinite(freq) && freq > 1.0F) {
                c->frequency = std::clamp(freq, 100.0F, 384000.0F);
            }
            win64_return(s, kFmodOk);
        });
        fmod("?set3DAttributes@Channel@FMOD@@QEAA?AW4FMOD_RESULT@@PEBUFMOD_VECTOR@@0@Z", [this](CpuState& s) {
            const std::uint64_t p_pos = win64_arg(s, 1);
            const std::uint64_t p_vel = win64_arg(s, 2);
            CombatHapticPulse pulse{};
            HapticFn haptic_cb;
            {
                const std::lock_guard<std::mutex> lock(m_);
                Channel* c = resolve_channel_locked(win64_arg(s, 0));
                if (c == nullptr) {
                    win64_return(s, kFmodErrInvalidHandle);
                    return;
                }
                if (p_pos != 0) {
                    read_vec3(p_pos, c->position);
                    c->has_3d_pos = true;
                }
                if (p_vel != 0) read_vec3(p_vel, c->velocity);
                if (c->playing && !c->paused && !c->haptic_fired && c->has_3d_pos) {
                    pulse = maybe_trigger_haptic_locked(*c);
                    haptic_cb = on_haptic;
                }
            }
            if (pulse.kind != CombatHapticKind::none && haptic_cb) {
                haptic_cb(pulse);
            }
            win64_return(s, kFmodOk);
        });
        fmod("?get3DAttributes@Channel@FMOD@@QEAA?AW4FMOD_RESULT@@PEAUFMOD_VECTOR@@0@Z", [this](CpuState& s) {
            const std::uint64_t p_pos = win64_arg(s, 1);
            const std::uint64_t p_vel = win64_arg(s, 2);
            const std::lock_guard<std::mutex> lock(m_);
            Channel* c = resolve_channel_locked(win64_arg(s, 0));
            if (c == nullptr) {
                win64_return(s, kFmodErrInvalidHandle);
                return;
            }
            if (p_pos != 0) write_vec3(p_pos, c->position);
            if (p_vel != 0) write_vec3(p_vel, c->velocity);
            win64_return(s, kFmodOk);
        });
        fmod("?setUserData@Channel@FMOD@@QEAA?AW4FMOD_RESULT@@PEAX@Z", [this](CpuState& s) {
            const std::lock_guard<std::mutex> lock(m_);
            Channel* c = resolve_channel_locked(win64_arg(s, 0));
            if (c == nullptr) {
                win64_return(s, kFmodErrInvalidHandle);
                return;
            }
            c->user = win64_arg(s, 1);
            win64_return(s, kFmodOk);
        });
        fmod("?getUserData@Channel@FMOD@@QEAA?AW4FMOD_RESULT@@PEAPEAX@Z", [this](CpuState& s) {
            const std::lock_guard<std::mutex> lock(m_);
            Channel* c = resolve_channel_locked(win64_arg(s, 0));
            if (c == nullptr) {
                win64_return(s, kFmodErrInvalidHandle);
                return;
            }
            if (win64_arg(s, 1) != 0) crt_detail::put<std::uint64_t>(win64_arg(s, 1), c->user);
            win64_return(s, kFmodOk);
        });
        fmod("?getCurrentSound@Channel@FMOD@@QEAA?AW4FMOD_RESULT@@PEAPEAVSound@2@@Z", [this](CpuState& s) {
            const std::lock_guard<std::mutex> lock(m_);
            Channel* c = resolve_channel_locked(win64_arg(s, 0));
            if (c == nullptr) {
                win64_return(s, kFmodErrInvalidHandle);
                return;
            }
            if (win64_arg(s, 1) != 0) {
                crt_detail::put<std::uint64_t>(win64_arg(s, 1), c->sound_handle);
            }
            win64_return(s, kFmodOk);
        });
        fmod("?getIndex@Channel@FMOD@@QEAA?AW4FMOD_RESULT@@PEAH@Z", [this](CpuState& s) {
            const std::lock_guard<std::mutex> lock(m_);
            Channel* c = resolve_channel_locked(win64_arg(s, 0));
            if (c == nullptr) {
                win64_return(s, kFmodErrInvalidHandle);
                return;
            }
            if (win64_arg(s, 1) != 0) crt_detail::put<std::int32_t>(win64_arg(s, 1), c->index);
            win64_return(s, kFmodOk);
        });
        fmod("?setMode@Channel@FMOD@@QEAA?AW4FMOD_RESULT@@I@Z", [this](CpuState& s) {
            const std::lock_guard<std::mutex> lock(m_);
            Channel* c = resolve_channel_locked(win64_arg(s, 0));
            if (c == nullptr) {
                win64_return(s, kFmodErrInvalidHandle);
                return;
            }
            c->mode = static_cast<std::uint32_t>(win64_arg(s, 1));
            win64_return(s, kFmodOk);
        });
        fmod("?getMode@Channel@FMOD@@QEAA?AW4FMOD_RESULT@@PEAI@Z", [this](CpuState& s) {
            const std::lock_guard<std::mutex> lock(m_);
            Channel* c = resolve_channel_locked(win64_arg(s, 0));
            if (c == nullptr) {
                win64_return(s, kFmodErrInvalidHandle);
                return;
            }
            if (win64_arg(s, 1) != 0) crt_detail::put<std::uint32_t>(win64_arg(s, 1), c->mode);
            win64_return(s, kFmodOk);
        });
        fmod("?setVolume@Channel@FMOD@@QEAA?AW4FMOD_RESULT@@M@Z", [this](CpuState& s) {
            const float vol = win64_arg_f32(s, 1);
            const std::lock_guard<std::mutex> lock(m_);
            Channel* c = resolve_channel_locked(win64_arg(s, 0));
            if (c == nullptr) {
                win64_return(s, kFmodErrInvalidHandle);
                return;
            }
            c->volume = std::clamp(std::isfinite(vol) ? vol : 1.0F, 0.0F, 4.0F);
            win64_return(s, kFmodOk);
        });
        fmod("?getVolume@Channel@FMOD@@QEAA?AW4FMOD_RESULT@@PEAM@Z", [this](CpuState& s) {
            const std::lock_guard<std::mutex> lock(m_);
            Channel* c = resolve_channel_locked(win64_arg(s, 0));
            if (c == nullptr) {
                win64_return(s, kFmodErrInvalidHandle);
                return;
            }
            if (win64_arg(s, 1) != 0) crt_detail::put<float>(win64_arg(s, 1), c->volume);
            win64_return(s, kFmodOk);
        });
        fmod("?setPriority@Channel@FMOD@@QEAA?AW4FMOD_RESULT@@H@Z", [this](CpuState& s) {
            const auto prio = static_cast<std::int32_t>(win64_arg(s, 1));
            const std::lock_guard<std::mutex> lock(m_);
            Channel* c = resolve_channel_locked(win64_arg(s, 0));
            if (c == nullptr) {
                win64_return(s, kFmodErrInvalidHandle);
                return;
            }
            c->priority = std::clamp(prio, 0, 256);
            win64_return(s, kFmodOk);
        });
        fmod("?setCallback@Channel@FMOD@@QEAA?AW4FMOD_RESULT@@P6A?AW43@PEAUFMOD_CHANNEL@@W4FMOD_CHANNEL_CALLBACKTYPE@@PEAX2@Z@Z",
             [this](CpuState& s) {
                 const std::lock_guard<std::mutex> lock(m_);
                 Channel* c = resolve_channel_locked(win64_arg(s, 0));
                 if (c == nullptr) {
                     win64_return(s, kFmodErrInvalidHandle);
                     return;
                 }
                 c->callback_va = win64_arg(s, 1);
                 win64_return(s, kFmodOk);
             });

        // Optional output/geometry/channel-budget configuration calls accepted as clean no-ops.
        for (const char* name : {
                 "?addPolygon@Geometry@FMOD@@QEAA?AW4FMOD_RESULT@@MM_NHPEBUFMOD_VECTOR@@PEAH@Z",
                 "?setOutput@System@FMOD@@QEAA?AW4FMOD_RESULT@@W4FMOD_OUTPUTTYPE@@@Z",
                 "?setDSPBufferSize@System@FMOD@@QEAA?AW4FMOD_RESULT@@IH@Z",
                 "?setSpeakerMode@System@FMOD@@QEAA?AW4FMOD_RESULT@@W4FMOD_SPEAKERMODE@@@Z",
                 "?setGeometrySettings@System@FMOD@@QEAA?AW4FMOD_RESULT@@M@Z",
                 "?setPosition@Geometry@FMOD@@QEAA?AW4FMOD_RESULT@@PEBUFMOD_VECTOR@@@Z",
                 "?setRotation@Geometry@FMOD@@QEAA?AW4FMOD_RESULT@@PEBUFMOD_VECTOR@@0@Z",
                 "?setScale@Geometry@FMOD@@QEAA?AW4FMOD_RESULT@@PEBUFMOD_VECTOR@@@Z",
                 "?setActive@Geometry@FMOD@@QEAA?AW4FMOD_RESULT@@_N@Z",
                 "?release@Geometry@FMOD@@QEAA?AW4FMOD_RESULT@@XZ",
                 "?setHardwareChannels@System@FMOD@@QEAA?AW4FMOD_RESULT@@H@Z",
                 "?setSoftwareChannels@System@FMOD@@QEAA?AW4FMOD_RESULT@@H@Z"}) {
            fmod(name, ok);
        }

        // Steam, as on a PC where Steam is not running.
        auto steam = [&](const char* name, Runtime::ImportShim shim) {
            n += rt_.set_import("steam_api64.dll", name, std::move(shim));
        };
        steam("SteamAPI_Init", [](CpuState& s) { win64_return(s, 0); });
        steam("SteamAPI_GetHSteamUser", [](CpuState& s) { win64_return(s, 0); });
        steam("SteamInternal_FindOrCreateUserInterface", [](CpuState& s) { win64_return(s, 0); });
        steam("SteamInternal_ContextInit", [](CpuState& s) {
            const std::uint64_t d = win64_arg(s, 0);
            win64_return(s, d != 0 ? d + 16 : 0);
        });
        for (const char* name : {"SteamAPI_RegisterCallResult", "SteamAPI_UnregisterCallResult", "SteamAPI_RunCallbacks",
                                 "SteamAPI_Shutdown", "SteamAPI_RegisterCallback", "SteamAPI_UnregisterCallback"}) {
            steam(name, ok);
        }
        return n;
    }

private:
    static constexpr std::uint64_t kChannelHandleTag = 0x7E00000000000000ULL;

    struct Sound {
        std::string name;
        std::uint32_t mode = kFmod2D | kFmodLoopOff;
        bool is_stream = false;
        std::uint32_t sample_rate = 44100U;
        std::uint16_t channels = 1U;
        std::size_t total_frames = 44100U;
        float default_frequency = 44100.0F;
        float default_volume = 1.0F;
        float default_pan = 0.0F;
        std::int32_t default_priority = 128;
        float min_dist = 2.0F;
        float max_dist = 80.0F;
        std::shared_ptr<const wse::asset::AudioPcmBuffer> pcm;
        std::shared_ptr<const std::vector<std::uint8_t>> stream_bytes;
        std::unique_ptr<wse::asset::AudioStreamDecoder> initial_stream;
    };

    struct Channel {
        std::uint64_t handle = 0U;
        bool handle_valid = false;
        std::uint64_t sound_handle = 0;
        std::shared_ptr<Sound> sound;
        std::uint64_t user = 0;
        std::uint64_t callback_va = 0;
        std::uint32_t mode = kFmod2D | kFmodLoopOff;
        float volume = 1.0F;
        float frequency = 44100.0F;
        std::int32_t priority = 128;
        std::int32_t index = 0;
        std::uint64_t generation = 0;
        bool playing = false;
        bool paused = false;
        bool has_3d_pos = false;
        bool haptic_fired = false;
        wse::core::WseVec4 position{};
        wse::core::WseVec4 velocity{};
        double cursor_frame = 0.0;
        bool has_gain = false;
        float gain_l = 0.0F;
        float gain_r = 0.0F;
        // Streaming state (used only when `sound->is_stream` is true).
        std::unique_ptr<wse::asset::AudioStreamDecoder> stream;
        std::vector<float> stream_buf;
        std::vector<float> stream_scratch;
        std::size_t stream_buf_frames = 0U;
        double stream_buf_cursor = 0.0;
        bool stream_eof = false;
    };

    struct PendingCallback {
        std::uint64_t channel = 0;
        std::uint64_t fn_va = 0;
    };

    static std::uint64_t make_channel_handle(std::size_t slot_idx, std::uint64_t gen) noexcept {
        return kChannelHandleTag | ((gen & 0x0000FFFFFFFFULL) << 16U) |
               static_cast<std::uint64_t>((slot_idx + 1U) & 0xFFFFU);
    }

    void init_channel_pool_locked(std::int32_t count) {
        const auto n = static_cast<std::size_t>(std::clamp(count, 1, 256));
        channels_.clear();
        channels_.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            channels_[i].index = static_cast<std::int32_t>(i);
        }
    }

    Channel* resolve_channel_locked(std::uint64_t handle) noexcept {
        if ((handle & 0xFF00000000000000ULL) != kChannelHandleTag) {
            return nullptr;
        }
        const std::size_t slot_1 = static_cast<std::size_t>(handle & 0xFFFFU);
        if (slot_1 == 0U || slot_1 > channels_.size()) {
            return nullptr;
        }
        Channel& c = channels_[slot_1 - 1U];
        if (!c.handle_valid || c.handle != handle) {
            return nullptr;
        }
        return &c;
    }

    static void read_vec3(std::uint64_t ptr, wse::core::WseVec4& out) noexcept {
        out.x = crt_detail::get<float>(ptr);
        out.y = crt_detail::get<float>(ptr + 4U);
        out.z = crt_detail::get<float>(ptr + 8U);
        out.w = 0.0F;
    }

    static void write_vec3(std::uint64_t ptr, const wse::core::WseVec4& v) noexcept {
        crt_detail::put<float>(ptr, v.x);
        crt_detail::put<float>(ptr + 4U, v.y);
        crt_detail::put<float>(ptr + 8U, v.z);
    }

    static std::string read_guest_cstr(std::uint64_t ptr) {
        if (ptr == 0) return {};
        const char* p = reinterpret_cast<const char*>(static_cast<std::uintptr_t>(ptr));
        std::size_t len = 0U;
        while (len < 4096U && p[len] != '\0') ++len;
        return std::string(p, len);
    }

    static std::uint32_t sound_length_in_unit(const Sound& snd, std::uint32_t unit) noexcept {
        const std::uint32_t sr = std::max<std::uint32_t>(1U, snd.sample_rate);
        if ((unit & kFmodTimeUnitPcm) != 0U) {
            return static_cast<std::uint32_t>(snd.total_frames);
        }
        if ((unit & kFmodTimeUnitPcmBytes) != 0U) {
            return static_cast<std::uint32_t>(snd.total_frames * std::max<std::uint16_t>(1U, snd.channels) * 2U);
        }
        return static_cast<std::uint32_t>((snd.total_frames * 1000ULL) / static_cast<std::uint64_t>(sr));
    }

    static double unit_to_frame(const Sound& snd, std::uint32_t pos, std::uint32_t unit) noexcept {
        const double sr = static_cast<double>(std::max<std::uint32_t>(1U, snd.sample_rate));
        if ((unit & kFmodTimeUnitPcm) != 0U) {
            return static_cast<double>(pos);
        }
        if ((unit & kFmodTimeUnitPcmBytes) != 0U) {
            const std::size_t bpf = std::max<std::size_t>(1U, snd.channels) * 2U;
            return static_cast<double>(pos) / static_cast<double>(bpf);
        }
        return (static_cast<double>(pos) * sr) / 1000.0;
    }

    static std::uint32_t frame_to_unit(const Sound& snd, double frame, std::uint32_t unit) noexcept {
        const double clamped = std::max(0.0, frame);
        const double sr = static_cast<double>(std::max<std::uint32_t>(1U, snd.sample_rate));
        if ((unit & kFmodTimeUnitPcm) != 0U) {
            return static_cast<std::uint32_t>(clamped);
        }
        if ((unit & kFmodTimeUnitPcmBytes) != 0U) {
            const std::size_t bpf = std::max<std::size_t>(1U, snd.channels) * 2U;
            return static_cast<std::uint32_t>(clamped * static_cast<double>(bpf));
        }
        return static_cast<std::uint32_t>((clamped * 1000.0) / sr);
    }

    void handle_create_sound(CpuState& s, bool force_stream) {
        const std::uint64_t name_or_data = win64_arg(s, 1);
        const auto mode = static_cast<std::uint32_t>(win64_arg(s, 2));
        const std::uint64_t exinfo = win64_arg(s, 3);
        const std::uint64_t out_sound = win64_arg(s, 4);
        if (out_sound == 0) {
            win64_return(s, kFmodErrInvalidParam);
            return;
        }

        const bool from_mem = (mode & (kFmodOpenMemory | kFmodOpenMemoryPoint)) != 0U;
        const bool is_stream = force_stream || ((mode & kFmodCreateStream) != 0U);
        std::vector<std::uint8_t> raw_bytes;
        std::string sound_name;

        if (from_mem) {
            std::uint32_t len = 0U;
            if (exinfo != 0) {
                len = crt_detail::get<std::uint32_t>(exinfo + 4U);  // FMOD_CREATESOUNDEXINFO::length
            }
            constexpr std::uint32_t kMaxMemSoundBytes = 128U * 1024U * 1024U;
            if (name_or_data == 0 || len == 0U || len > kMaxMemSoundBytes) {
                crt_detail::put<std::uint64_t>(out_sound, 0U);
                win64_return(s, kFmodErrInvalidParam);
                return;
            }
            raw_bytes.resize(len);
            std::memcpy(raw_bytes.data(),
                        reinterpret_cast<const void*>(static_cast<std::uintptr_t>(name_or_data)), len);
            sound_name = "<memory>";
        } else {
            sound_name = read_guest_cstr(name_or_data);
            if (read_file) {
                if (sound_name.empty() || !read_file(sound_name, raw_bytes)) {
                    // Return FMOD_ERR_FILE_NOTFOUND so Warband's 4-step module -> root
                    // sound/music fallback probes the next path instead of halting on silence.
                    crt_detail::put<std::uint64_t>(out_sound, 0U);
                    win64_return(s, kFmodErrFileNotFound);
                    return;
                }
            }
        }

        std::shared_ptr<wse::asset::AudioCodecBackend> codec;
        {
            const std::lock_guard<std::mutex> lock(m_);
            codec = codec_backend_;
        }

        auto snd = std::make_shared<Sound>();
        snd->name = sound_name;
        snd->mode = mode;
        if ((snd->mode & (kFmod2D | kFmod3D)) == 0U) {
            snd->mode |= kFmod2D;
        }
        snd->is_stream = is_stream;

        if (!raw_bytes.empty()) {
            if (is_stream) {
                auto shared_bytes = std::make_shared<std::vector<std::uint8_t>>(std::move(raw_bytes));
                std::string err;
                auto probe = wse::asset::open_audio_stream(*shared_bytes, sound_name, codec.get(), &err);
                if (probe) {
                    snd->sample_rate = probe->sample_rate() > 0U ? probe->sample_rate() : 44100U;
                    snd->channels = probe->channels() > 0U ? probe->channels() : 2U;
                    snd->total_frames = probe->total_frames() > 0U ? static_cast<std::size_t>(probe->total_frames())
                                                                   : static_cast<std::size_t>(snd->sample_rate);
                    snd->default_frequency = static_cast<float>(snd->sample_rate);
                    snd->stream_bytes = std::move(shared_bytes);
                    snd->initial_stream = std::move(probe);
                } else {
                    auto pcm = std::make_shared<wse::asset::AudioPcmBuffer>();
                    if (wse::asset::decode_audio_bytes(*shared_bytes, sound_name, codec.get(), *pcm, &err) &&
                        pcm->valid()) {
                        snd->is_stream = false;
                        snd->sample_rate = pcm->sample_rate;
                        snd->channels = pcm->channels;
                        snd->total_frames = pcm->frame_count();
                        snd->default_frequency = static_cast<float>(pcm->sample_rate);
                        snd->pcm = std::move(pcm);
                    }
                }
            } else {
                auto pcm = std::make_shared<wse::asset::AudioPcmBuffer>();
                std::string err;
                if (wse::asset::decode_audio_bytes(raw_bytes, sound_name, codec.get(), *pcm, &err) && pcm->valid()) {
                    snd->sample_rate = pcm->sample_rate;
                    snd->channels = pcm->channels;
                    snd->total_frames = pcm->frame_count();
                    snd->default_frequency = static_cast<float>(pcm->sample_rate);
                    snd->pcm = std::move(pcm);
                }
            }
        }

        const std::uint64_t handle = next_handle();
        {
            const std::lock_guard<std::mutex> lock(m_);
            sounds_[handle] = std::move(snd);
        }
        sounds_created_.fetch_add(1U, std::memory_order_relaxed);
        crt_detail::put<std::uint64_t>(out_sound, handle);
        win64_return(s, kFmodOk);
    }

    void handle_play_sound(CpuState& s) {
        const auto ch_id = static_cast<std::int32_t>(win64_arg(s, 1));
        const std::uint64_t sound_handle = win64_arg(s, 2);
        const bool paused = (win64_arg(s, 3) & 0xFFU) != 0U;
        const std::uint64_t out_ch = win64_arg(s, 4);

        CombatHapticPulse pulse{};
        HapticFn haptic_cb;
        std::uint64_t ch_handle = 0;

        {
            const std::lock_guard<std::mutex> lock(m_);
            if (channels_.empty()) {
                init_channel_pool_locked(max_channels_);
            }
            const auto sit = sounds_.find(sound_handle);
            std::shared_ptr<Sound> snd = (sit != sounds_.end()) ? sit->second : nullptr;
            const std::int32_t new_prio = snd ? snd->default_priority : 128;

            Channel* slot = nullptr;
            if (ch_id == kFmodChannelReuse && out_ch != 0) {
                const std::uint64_t existing = crt_detail::get<std::uint64_t>(out_ch);
                slot = resolve_channel_locked(existing);
            } else if (ch_id >= 0 && static_cast<std::size_t>(ch_id) < channels_.size()) {
                slot = &channels_[static_cast<std::size_t>(ch_id)];
            }

            if (slot == nullptr) {
                // Find an idle slot first.
                for (auto& c : channels_) {
                    if (!c.playing) {
                        slot = &c;
                        break;
                    }
                }
                // All slots busy: evict lowest-priority (highest numeric priority), oldest channel.
                if (slot == nullptr) {
                    slot = find_eviction_victim_locked(new_prio);
                }
                if (slot == nullptr) {
                    if (out_ch != 0) crt_detail::put<std::uint64_t>(out_ch, 0U);
                    win64_return(s, kFmodErrInvalidHandle);
                    return;
                }
            }

            if (slot->playing) {
                stop_channel_locked(*slot, true);
            }

            const std::int32_t saved_index = slot->index;
            *slot = Channel{};
            slot->index = saved_index;
            slot->generation = ++generation_;
            slot->handle = make_channel_handle(static_cast<std::size_t>(saved_index), slot->generation);
            slot->handle_valid = true;
            slot->sound_handle = sound_handle;
            slot->sound = snd;
            slot->mode = snd ? snd->mode : (kFmod2D | kFmodLoopOff);
            slot->volume = snd ? snd->default_volume : 1.0F;
            slot->frequency = snd ? snd->default_frequency : 44100.0F;
            slot->priority = new_prio;
            slot->playing = (snd != nullptr && (snd->pcm != nullptr || snd->stream_bytes != nullptr));
            slot->paused = paused;

            if (snd && snd->is_stream && snd->stream_bytes) {
                if (snd->initial_stream) {
                    slot->stream = std::move(snd->initial_stream);
                } else {
                    std::string err;
                    slot->stream =
                        wse::asset::open_audio_stream(*snd->stream_bytes, snd->name, codec_backend_.get(), &err);
                }
                if (!slot->stream) {
                    slot->playing = false;
                }
            }

            ch_handle = slot->handle;
            if (slot->playing && !slot->paused) {
                pulse = maybe_trigger_haptic_locked(*slot);
                haptic_cb = on_haptic;
            }
        }

        if (out_ch != 0) {
            crt_detail::put<std::uint64_t>(out_ch, ch_handle);
        }
        if (pulse.kind != CombatHapticKind::none && haptic_cb) {
            haptic_cb(pulse);
        }
        win64_return(s, kFmodOk);
    }

    CombatHapticPulse maybe_trigger_haptic_locked(Channel& c) {
        if (!c.sound || c.sound->name.empty() || !on_haptic) {
            return {};
        }
        const bool is_3d = ((c.mode & kFmod3D) != 0U) && ((c.mode & kFmod3DHeadRelative) == 0U);
        // Never fire haptics for unpositioned 3D sounds (wait until Channel::set3DAttributes provides position).
        if (is_3d && !c.has_3d_pos) {
            return {};
        }
        float dist = 0.0F;
        if (is_3d) {
            const wse::core::WseVec4 d = c.position - listener_.position;
            dist = d.length() * distance_factor_;
        }
        const auto now_ms = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch())
                .count());
        const CombatHapticPulse pulse = haptics_.evaluate(c.sound->name, is_3d, dist, c.volume, now_ms);
        if (pulse.kind != CombatHapticKind::none) {
            c.haptic_fired = true;
        }
        return pulse;
    }

    Channel* find_eviction_victim_locked(std::int32_t new_priority) noexcept {
        Channel* worst = nullptr;
        std::int32_t worst_prio = -1;
        std::uint64_t worst_gen = UINT64_MAX;
        for (auto& c : channels_) {
            if (!c.playing) return &c;
            if (c.priority > worst_prio || (c.priority == worst_prio && c.generation < worst_gen)) {
                worst_prio = c.priority;
                worst_gen = c.generation;
                worst = &c;
            }
        }
        if (worst != nullptr && new_priority <= worst_prio) {
            return worst;
        }
        return nullptr;
    }

    void stop_channel_locked(Channel& c, bool fire_callback) {
        const bool was_playing = c.playing;
        const std::uint64_t cb_va = c.callback_va;
        const std::uint64_t handle = c.handle;
        c.playing = false;
        c.sound.reset();
        c.sound_handle = 0U;
        c.stream.reset();
        c.stream_buf.clear();
        c.stream_buf_frames = 0U;
        c.stream_buf_cursor = 0.0;
        if (was_playing && fire_callback && cb_va != 0) {
            pending_end_callbacks_.push_back(PendingCallback{handle, cb_va});
        }
    }

    void seek_stream_channel_locked(Channel& c, std::size_t target_frame) {
        if (!c.sound || !c.sound->stream_bytes) return;
        std::string err;
        c.stream = wse::asset::open_audio_stream(*c.sound->stream_bytes, c.sound->name, codec_backend_.get(), &err);
        c.stream_buf.clear();
        c.stream_buf_frames = 0U;
        c.stream_buf_cursor = 0.0;
        c.stream_eof = (c.stream == nullptr);
        c.cursor_frame = 0.0;
        if (!c.stream || target_frame == 0U) return;

        // Advance sequentially up to `target_frame` in small chunks.
        std::size_t remaining = target_frame;
        std::vector<float> discard;
        while (remaining > 0U && !c.stream_eof) {
            const std::size_t want = std::min<std::size_t>(remaining, 2048U);
            const std::size_t got = c.stream->read_frames(want, discard);
            if (got == 0U) {
                c.stream_eof = true;
                break;
            }
            remaining -= got;
            c.cursor_frame += static_cast<double>(got);
        }
    }

    void compute_channel_gains_locked(const Channel& c, float& out_l, float& out_r) const noexcept {
        const float master = master_volume_.load(std::memory_order_relaxed);
        const float base_gain = std::max(0.0F, c.volume) * master;
        if (base_gain <= 1e-5F) {
            out_l = 0.0F;
            out_r = 0.0F;
            return;
        }
        const bool is_3d = ((c.mode & kFmod3D) != 0U);
        if (!is_3d) {
            const float pan = (c.sound != nullptr) ? std::clamp(c.sound->default_pan, -1.0F, 1.0F) : 0.0F;
            const float angle = (pan + 1.0F) * 0.25F * 3.14159265358979323846F;
            out_l = base_gain * std::cos(angle) * 1.41421356F;
            out_r = base_gain * std::sin(angle) * 1.41421356F;
            return;
        }

        const bool head_rel = ((c.mode & kFmod3DHeadRelative) != 0U);
        const wse::core::WseVec4 rel = head_rel ? c.position : (c.position - listener_.position);
        const float dist = rel.length() * distance_factor_;
        const float min_d = (c.sound != nullptr) ? std::max(1e-3F, c.sound->min_dist) : 2.0F;
        const float max_d = (c.sound != nullptr) ? std::max(min_d, c.sound->max_dist) : 80.0F;

        float atten = 1.0F;
        if (dist > min_d) {
            if ((c.mode & kFmod3DLinearRolloff) != 0U) {
                const float span = std::max(1e-3F, max_d - min_d);
                atten = std::clamp(1.0F - (dist - min_d) / span, 0.0F, 1.0F);
            } else {
                const float clamped_d = std::min(dist, max_d);
                const float denom = min_d + rolloff_scale_ * (clamped_d - min_d);
                atten = (denom > 1e-4F) ? std::clamp(min_d / denom, 0.0F, 1.0F) : 1.0F;
                if (dist >= max_d) {
                    atten *= 0.25F;
                }
            }
        }

        float pan = 0.0F;
        if (dist > 1e-4F) {
            const wse::core::WseVec4 dir{rel.x / (rel.length()), rel.y / (rel.length()), rel.z / (rel.length()), 0.0F};
            pan = std::clamp(dir.dot(listener_.right), -1.0F, 1.0F);
        }
        const float angle = (pan + 1.0F) * 0.25F * 3.14159265358979323846F;
        const float g = base_gain * atten * 1.41421356F;
        out_l = g * std::cos(angle);
        out_r = g * std::sin(angle);
    }

    void mix_locked(std::size_t frame_count, float* out) {
        const double out_sr = static_cast<double>(std::max<std::uint32_t>(8000U, output_sample_rate_));
        const float inv_n = 1.0F / static_cast<float>(frame_count);

        for (auto& c : channels_) {
            if (!c.playing || c.paused || !c.sound) {
                continue;
            }

            float target_l = 0.0F;
            float target_r = 0.0F;
            compute_channel_gains_locked(c, target_l, target_r);
            const float start_l = c.has_gain ? c.gain_l : target_l;
            const float start_r = c.has_gain ? c.gain_r : target_r;
            c.gain_l = target_l;
            c.gain_r = target_r;
            c.has_gain = true;

            const bool loop = ((c.mode & (kFmodLoopNormal | kFmodLoopBidi)) != 0U) &&
                              ((c.mode & kFmodLoopOff) == 0U);
            const double step = static_cast<double>(std::max(100.0F, c.frequency)) / out_sr;

            if (c.sound->is_stream) {
                mix_stream_channel_locked(c, frame_count, out, start_l, start_r,
                                          target_l - start_l, target_r - start_r, inv_n, step, loop);
            } else if (c.sound->pcm != nullptr && c.sound->pcm->valid()) {
                mix_pcm_channel_locked(c, frame_count, out, start_l, start_r,
                                       target_l - start_l, target_r - start_r, inv_n, step, loop);
            } else {
                stop_channel_locked(c, true);
            }
        }

        const std::size_t total_samples = frame_count * 2U;
        for (std::size_t i = 0; i < total_samples; ++i) {
            out[i] = wse::platform::soft_clip_sample(out[i]);
        }
    }

    void mix_pcm_channel_locked(Channel& c, std::size_t frame_count, float* out,
                                float start_l, float start_r, float dl, float dr, float inv_n,
                                double step, bool loop) {
        const auto& pcm = *c.sound->pcm;
        const std::size_t total = pcm.frame_count();
        const bool stereo = (pcm.channels >= 2U);
        const float* data = pcm.samples.data();

        for (std::size_t f = 0; f < frame_count; ++f) {
            if (c.cursor_frame >= static_cast<double>(total)) {
                if (loop) {
                    c.cursor_frame = std::fmod(c.cursor_frame, static_cast<double>(total));
                } else {
                    stop_channel_locked(c, true);
                    break;
                }
            }
            const auto i0 = static_cast<std::size_t>(c.cursor_frame);
            const std::size_t i1 = (i0 + 1U < total) ? (i0 + 1U) : (loop ? 0U : i0);
            const float frac = static_cast<float>(c.cursor_frame - static_cast<double>(i0));
            float sl = 0.0F;
            float sr = 0.0F;
            if (stereo) {
                sl = data[i0 * 2U] + (data[i1 * 2U] - data[i0 * 2U]) * frac;
                sr = data[i0 * 2U + 1U] + (data[i1 * 2U + 1U] - data[i0 * 2U + 1U]) * frac;
            } else {
                sl = sr = data[i0] + (data[i1] - data[i0]) * frac;
            }
            const float t = static_cast<float>(f) * inv_n;
            out[f * 2U] += sl * (start_l + dl * t);
            out[f * 2U + 1U] += sr * (start_r + dr * t);
            c.cursor_frame += step;
        }
    }

    bool refill_stream_locked(Channel& c, bool loop) {
        if (!c.stream) return false;
        constexpr std::size_t kChunk = 2048U;
        const std::size_t stride = (c.stream->channels() >= 2U) ? 2U : 1U;
        const bool had_prev = (c.stream_buf_frames > 0U && c.stream_buf.size() >= c.stream_buf_frames * stride);
        float prev0 = 0.0F;
        float prev1 = 0.0F;
        if (had_prev) {
            const std::size_t last = (c.stream_buf_frames - 1U) * stride;
            prev0 = c.stream_buf[last];
            if (stride == 2U) prev1 = c.stream_buf[last + 1U];
        }

        std::size_t got = c.stream->read_frames(kChunk, c.stream_scratch);
        if (got == 0U && loop) {
            if (c.stream->seek_start()) {
                c.cursor_frame = 0.0;
                got = c.stream->read_frames(kChunk, c.stream_scratch);
            }
        }
        if (got == 0U) {
            c.stream_buf_frames = 0U;
            c.stream_buf_cursor = 0.0;
            c.stream_eof = true;
            return false;
        }
        if (had_prev) {
            c.stream_buf.resize((got + 1U) * stride);
            c.stream_buf[0] = prev0;
            if (stride == 2U) c.stream_buf[1] = prev1;
            std::memcpy(c.stream_buf.data() + stride, c.stream_scratch.data(), got * stride * sizeof(float));
            c.stream_buf_frames = got + 1U;
        } else {
            c.stream_buf.swap(c.stream_scratch);
            c.stream_buf_frames = got;
        }
        c.stream_eof = false;
        return true;
    }

    void mix_stream_channel_locked(Channel& c, std::size_t frame_count, float* out,
                                   float start_l, float start_r, float dl, float dr, float inv_n,
                                   double step, bool loop) {
        if (!c.stream) {
            stop_channel_locked(c, true);
            return;
        }
        const bool stereo = (c.stream->channels() >= 2U);
        for (std::size_t f = 0; f < frame_count; ++f) {
            while (c.stream_buf_frames == 0U ||
                   c.stream_buf_cursor >= static_cast<double>(c.stream_buf_frames - 1U)) {
                const bool had_prev = (c.stream_buf_frames > 0U);
                const double carry = had_prev ? (c.stream_buf_cursor - static_cast<double>(c.stream_buf_frames - 1U))
                                              : 0.0;
                if (!refill_stream_locked(c, loop)) {
                    stop_channel_locked(c, true);
                    return;
                }
                c.stream_buf_cursor = std::clamp(carry, 0.0, static_cast<double>(c.stream_buf_frames - 1U));
            }
            const auto i0 = static_cast<std::size_t>(c.stream_buf_cursor);
            const std::size_t i1 = (i0 + 1U < c.stream_buf_frames) ? (i0 + 1U) : i0;
            const float frac = static_cast<float>(c.stream_buf_cursor - static_cast<double>(i0));
            const float* data = c.stream_buf.data();
            float sl = 0.0F;
            float sr = 0.0F;
            if (stereo) {
                sl = data[i0 * 2U] + (data[i1 * 2U] - data[i0 * 2U]) * frac;
                sr = data[i0 * 2U + 1U] + (data[i1 * 2U + 1U] - data[i0 * 2U + 1U]) * frac;
            } else {
                sl = sr = data[i0] + (data[i1] - data[i0]) * frac;
            }
            const float t = static_cast<float>(f) * inv_n;
            out[f * 2U] += sl * (start_l + dl * t);
            out[f * 2U + 1U] += sr * (start_r + dr * t);
            c.stream_buf_cursor += step;
            c.cursor_frame += step;
        }
    }

    // Advances channel cursors when no open output device is available (or during route recovery)
    // so `Channel::isPlaying()` and end callbacks never freeze.
    void advance_silent_locked(std::size_t frame_count) {
        if (frame_count == 0U || app_paused_) return;
        const double out_sr = static_cast<double>(std::max<std::uint32_t>(8000U, output_sample_rate_));
        for (auto& c : channels_) {
            if (!c.playing || c.paused || !c.sound) continue;
            const bool loop = ((c.mode & (kFmodLoopNormal | kFmodLoopBidi)) != 0U) &&
                              ((c.mode & kFmodLoopOff) == 0U);
            const double delta = (static_cast<double>(std::max(100.0F, c.frequency)) / out_sr) *
                                 static_cast<double>(frame_count);
            c.cursor_frame += delta;
            const std::size_t total = c.sound->total_frames;
            if (total > 0U && c.cursor_frame >= static_cast<double>(total)) {
                if (loop) {
                    c.cursor_frame = std::fmod(c.cursor_frame, static_cast<double>(total));
                } else {
                    stop_channel_locked(c, true);
                }
            }
        }
    }

    std::size_t pump_locked(std::size_t target_queued_frames, std::size_t max_block_frames) {
        if (app_paused_ || max_block_frames == 0U) {
            return 0U;
        }
        if (audio_device_ == nullptr || !audio_device_->is_open()) {
            if (on_device_recover && on_device_recover() && audio_device_ != nullptr && audio_device_->is_open()) {
                output_sample_rate_ = std::max<std::uint32_t>(8000U, audio_device_->spec().sample_rate);
            } else {
                return 0U;
            }
        }
        output_sample_rate_ = std::max<std::uint32_t>(8000U, audio_device_->spec().sample_rate);
        const std::size_t queued = audio_device_->queued_frames();
        if (queued >= target_queued_frames) {
            return 0U;
        }
        std::size_t remaining = target_queued_frames - queued;
        std::size_t submitted = 0U;
        while (remaining > 0U) {
            const std::size_t block = std::min(remaining, max_block_frames);
            pump_buf_.assign(block * 2U, 0.0F);
            mix_locked(block, pump_buf_.data());
            if (!audio_device_->submit_frames(pump_buf_)) {
                if (on_device_recover && on_device_recover() && audio_device_->is_open()) {
                    output_sample_rate_ = std::max<std::uint32_t>(8000U, audio_device_->spec().sample_rate);
                    if (audio_device_->submit_frames(pump_buf_)) {
                        submitted += block;
                        remaining -= block;
                        continue;
                    }
                }
                break;
            }
            submitted += block;
            remaining -= block;
        }
        return submitted;
    }

    void worker_loop() {
        std::unique_lock<std::mutex> lock(m_);
        auto last_tick = std::chrono::steady_clock::now();
        while (!stop_worker_) {
            const auto now = std::chrono::steady_clock::now();
            const std::size_t pumped = pump_locked(2048U, 512U);
            if (pumped == 0U && !app_paused_ && (audio_device_ == nullptr || !audio_device_->is_open())) {
                const auto us = std::chrono::duration_cast<std::chrono::microseconds>(now - last_tick).count();
                if (us > 0) {
                    const auto frames = static_cast<std::size_t>(
                        (static_cast<std::uint64_t>(us) * static_cast<std::uint64_t>(output_sample_rate_)) /
                        1000000ULL);
                    advance_silent_locked(std::min<std::size_t>(frames, 4096U));
                }
            }
            last_tick = now;
            worker_cv_.wait_for(lock, std::chrono::microseconds{4000}, [this] { return stop_worker_; });
        }
    }

    std::uint64_t next_handle_locked() noexcept {
        next_handle_ += 0x10U;
        return next_handle_;
    }

    std::uint64_t next_handle() {
        const std::lock_guard<std::mutex> lock(m_);
        return next_handle_locked();
    }

    Runtime& rt_;
    mutable std::mutex m_;
    std::condition_variable worker_cv_;
    HostThread worker_;
    bool stop_worker_ = false;
    bool app_paused_ = false;
    wse::platform::AudioDevice* audio_device_ = nullptr;
    std::shared_ptr<wse::asset::AudioCodecBackend> codec_backend_;
    std::uint32_t output_sample_rate_ = 48000U;
    std::int32_t max_channels_ = 64;
    float distance_factor_ = 1.0F;
    float rolloff_scale_ = 1.0F;
    std::atomic<float> master_volume_{1.0F};
    wse::platform::AudioListener listener_{};
    CombatHapticsClassifier haptics_{};
    std::map<std::uint64_t, std::shared_ptr<Sound>> sounds_;
    std::vector<Channel> channels_;
    std::vector<PendingCallback> pending_end_callbacks_;
    std::vector<float> pump_buf_;
    std::uint64_t next_handle_ = 0x7F000000U;
    std::uint64_t generation_ = 0U;
    std::atomic<std::size_t> sounds_created_{0U};
};

}  // namespace wse::recomp::shim
