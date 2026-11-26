// Deterministic Session Record & Replay (`docs/RECOMP_PLAN.md` "v1 must-haves": Record and replay).
//
// Records live inputs plus every non-deterministic clock (`QueryPerformanceCounter`,
// `GetTickCount`, `GetTickCount64`, `timeGetTime`) and `RtlGenRandom` response into a compact
// binary stream, and replays them identically for repeatable WSL/phone performance benchmarking
// and crash reproduction. Zero overhead when `SessionMode::off` (single relaxed atomic check).
//
// Clean-room, header-only, standalone-compilable.

#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

namespace wse::recomp::shim {

enum class SessionMode : std::uint8_t {
    off = 0,
    record = 1,
    replay = 2,
};

enum class SessionEventTag : std::uint8_t {
    qpc = 1,
    tick32 = 2,
    tick64 = 3,
    time_get_time = 4,
    rtl_random = 5,
    frame_mark = 6,
    input_key = 7,
    input_mouse_abs = 8,
    input_mouse_delta = 9,
    input_mouse_btn = 10,
    input_mouse_wheel = 11,
    input_touch = 12,
};

struct SessionInputEvent {
    SessionEventTag tag = SessionEventTag::frame_mark;
    std::int32_t a = 0;
    std::int32_t b = 0;
    float fx = 0.0F;
    float fy = 0.0F;
};

class SessionLog {
public:
    static constexpr char kMagic[8] = {'W', 'S', 'E', '2', 'R', 'P', 'L', '1'};

    static SessionLog& shared() noexcept {
        static SessionLog instance;
        return instance;
    }

    [[nodiscard]] SessionMode mode() const noexcept {
        return static_cast<SessionMode>(mode_.load(std::memory_order_relaxed));
    }

    [[nodiscard]] bool active() const noexcept {
        return mode_.load(std::memory_order_relaxed) != static_cast<std::uint8_t>(SessionMode::off);
    }

    bool start_record(const std::string& path) {
        const std::lock_guard<std::mutex> lock(m_);
        reset_locked();
        out_.open(path, std::ios::binary | std::ios::trunc);
        if (!out_) return false;
        out_.write(kMagic, sizeof(kMagic));
        const std::uint32_t version = 1U;
        out_.write(reinterpret_cast<const char*>(&version), sizeof(version));
        mode_.store(static_cast<std::uint8_t>(SessionMode::record), std::memory_order_release);
        return static_cast<bool>(out_);
    }

    bool start_replay(const std::string& path) {
        const std::lock_guard<std::mutex> lock(m_);
        reset_locked();
        std::ifstream in(path, std::ios::binary);
        if (!in) return false;
        char magic[8]{};
        std::uint32_t version = 0U;
        if (!in.read(magic, sizeof(magic)) || std::memcmp(magic, kMagic, sizeof(magic)) != 0 ||
            !in.read(reinterpret_cast<char*>(&version), sizeof(version)) || version != 1U) {
            return false;
        }
        while (in.peek() != std::char_traits<char>::eof()) {
            std::uint8_t raw_tag = 0U;
            if (!in.read(reinterpret_cast<char*>(&raw_tag), 1)) break;
            const auto tag = static_cast<SessionEventTag>(raw_tag);
            switch (tag) {
            case SessionEventTag::qpc: {
                std::int64_t v = 0;
                if (!in.read(reinterpret_cast<char*>(&v), sizeof(v))) return false;
                qpc_vals_.push_back(v);
                break;
            }
            case SessionEventTag::tick32: {
                std::uint32_t v = 0;
                if (!in.read(reinterpret_cast<char*>(&v), sizeof(v))) return false;
                tick32_vals_.push_back(v);
                break;
            }
            case SessionEventTag::tick64: {
                std::uint64_t v = 0;
                if (!in.read(reinterpret_cast<char*>(&v), sizeof(v))) return false;
                tick64_vals_.push_back(v);
                break;
            }
            case SessionEventTag::time_get_time: {
                std::uint32_t v = 0;
                if (!in.read(reinterpret_cast<char*>(&v), sizeof(v))) return false;
                tgt_vals_.push_back(v);
                break;
            }
            case SessionEventTag::rtl_random: {
                std::uint32_t n = 0;
                if (!in.read(reinterpret_cast<char*>(&n), sizeof(n)) || n > (16U << 20U)) return false;
                std::vector<std::uint8_t> chunk(n);
                if (n > 0 && !in.read(reinterpret_cast<char*>(chunk.data()), static_cast<std::streamsize>(n))) {
                    return false;
                }
                random_chunks_.push_back(std::move(chunk));
                break;
            }
            case SessionEventTag::frame_mark: {
                std::uint64_t frame = 0;
                if (!in.read(reinterpret_cast<char*>(&frame), sizeof(frame))) return false;
                replay_frames_.push_back({frame, {}});
                break;
            }
            case SessionEventTag::input_key:
            case SessionEventTag::input_mouse_abs:
            case SessionEventTag::input_mouse_delta:
            case SessionEventTag::input_mouse_btn:
            case SessionEventTag::input_mouse_wheel:
            case SessionEventTag::input_touch: {
                SessionInputEvent ev{};
                ev.tag = tag;
                if (!in.read(reinterpret_cast<char*>(&ev.a), sizeof(ev.a)) ||
                    !in.read(reinterpret_cast<char*>(&ev.b), sizeof(ev.b)) ||
                    !in.read(reinterpret_cast<char*>(&ev.fx), sizeof(ev.fx)) ||
                    !in.read(reinterpret_cast<char*>(&ev.fy), sizeof(ev.fy))) {
                    return false;
                }
                if (replay_frames_.empty()) {
                    replay_frames_.push_back({0U, {}});
                }
                replay_frames_.back().second.push_back(ev);
                break;
            }
            default:
                return false;
            }
        }
        mode_.store(static_cast<std::uint8_t>(SessionMode::replay), std::memory_order_release);
        return true;
    }

    void stop() {
        const std::lock_guard<std::mutex> lock(m_);
        reset_locked();
    }

    std::int64_t on_qpc(std::int64_t host_val) {
        const auto m = mode();
        if (m == SessionMode::off) return host_val;
        const std::lock_guard<std::mutex> lock(m_);
        if (m == SessionMode::record && out_) {
            const auto tag = static_cast<std::uint8_t>(SessionEventTag::qpc);
            out_.write(reinterpret_cast<const char*>(&tag), 1);
            out_.write(reinterpret_cast<const char*>(&host_val), sizeof(host_val));
            return host_val;
        }
        if (m == SessionMode::replay && qpc_idx_ < qpc_vals_.size()) {
            return qpc_vals_[qpc_idx_++];
        }
        return host_val;
    }

    std::uint32_t on_tick32(std::uint32_t host_val) {
        const auto m = mode();
        if (m == SessionMode::off) return host_val;
        const std::lock_guard<std::mutex> lock(m_);
        if (m == SessionMode::record && out_) {
            const auto tag = static_cast<std::uint8_t>(SessionEventTag::tick32);
            out_.write(reinterpret_cast<const char*>(&tag), 1);
            out_.write(reinterpret_cast<const char*>(&host_val), sizeof(host_val));
            return host_val;
        }
        if (m == SessionMode::replay && tick32_idx_ < tick32_vals_.size()) {
            return tick32_vals_[tick32_idx_++];
        }
        return host_val;
    }

    std::uint64_t on_tick64(std::uint64_t host_val) {
        const auto m = mode();
        if (m == SessionMode::off) return host_val;
        const std::lock_guard<std::mutex> lock(m_);
        if (m == SessionMode::record && out_) {
            const auto tag = static_cast<std::uint8_t>(SessionEventTag::tick64);
            out_.write(reinterpret_cast<const char*>(&tag), 1);
            out_.write(reinterpret_cast<const char*>(&host_val), sizeof(host_val));
            return host_val;
        }
        if (m == SessionMode::replay && tick64_idx_ < tick64_vals_.size()) {
            return tick64_vals_[tick64_idx_++];
        }
        return host_val;
    }

    std::uint32_t on_time_get_time(std::uint32_t host_val) {
        const auto m = mode();
        if (m == SessionMode::off) return host_val;
        const std::lock_guard<std::mutex> lock(m_);
        if (m == SessionMode::record && out_) {
            const auto tag = static_cast<std::uint8_t>(SessionEventTag::time_get_time);
            out_.write(reinterpret_cast<const char*>(&tag), 1);
            out_.write(reinterpret_cast<const char*>(&host_val), sizeof(host_val));
            return host_val;
        }
        if (m == SessionMode::replay && tgt_idx_ < tgt_vals_.size()) {
            return tgt_vals_[tgt_idx_++];
        }
        return host_val;
    }

    // Intercepts `RtlGenRandom`: in record mode writes generated bytes; in replay mode overwrites
    // `dst` with recorded bytes if the next chunk matches length.
    void on_rtl_random(std::uint8_t* dst, std::size_t n) {
        const auto m = mode();
        if (m == SessionMode::off || dst == nullptr) return;
        const std::lock_guard<std::mutex> lock(m_);
        if (m == SessionMode::record && out_) {
            const auto tag = static_cast<std::uint8_t>(SessionEventTag::rtl_random);
            const auto len = static_cast<std::uint32_t>(n);
            out_.write(reinterpret_cast<const char*>(&tag), 1);
            out_.write(reinterpret_cast<const char*>(&len), sizeof(len));
            if (n > 0) out_.write(reinterpret_cast<const char*>(dst), static_cast<std::streamsize>(n));
            return;
        }
        if (m == SessionMode::replay && rand_idx_ < random_chunks_.size()) {
            const auto& chunk = random_chunks_[rand_idx_++];
            const std::size_t copy_n = std::min(n, chunk.size());
            if (copy_n > 0) std::memcpy(dst, chunk.data(), copy_n);
        }
    }

    void record_frame_mark(std::uint64_t frame) {
        if (mode() != SessionMode::record) return;
        const std::lock_guard<std::mutex> lock(m_);
        if (!out_) return;
        const auto tag = static_cast<std::uint8_t>(SessionEventTag::frame_mark);
        out_.write(reinterpret_cast<const char*>(&tag), 1);
        out_.write(reinterpret_cast<const char*>(&frame), sizeof(frame));
    }

    void record_input(const SessionInputEvent& ev) {
        if (mode() != SessionMode::record) return;
        const std::lock_guard<std::mutex> lock(m_);
        if (!out_) return;
        const auto tag = static_cast<std::uint8_t>(ev.tag);
        out_.write(reinterpret_cast<const char*>(&tag), 1);
        out_.write(reinterpret_cast<const char*>(&ev.a), sizeof(ev.a));
        out_.write(reinterpret_cast<const char*>(&ev.b), sizeof(ev.b));
        out_.write(reinterpret_cast<const char*>(&ev.fx), sizeof(ev.fx));
        out_.write(reinterpret_cast<const char*>(&ev.fy), sizeof(ev.fy));
    }

    // Pops all recorded input events scheduled up to `frame` during replay.
    std::vector<SessionInputEvent> pop_replay_inputs_for_frame(std::uint64_t frame) {
        std::vector<SessionInputEvent> out;
        if (mode() != SessionMode::replay) return out;
        const std::lock_guard<std::mutex> lock(m_);
        while (frame_idx_ < replay_frames_.size() && replay_frames_[frame_idx_].first <= frame) {
            const auto& evs = replay_frames_[frame_idx_++].second;
            out.insert(out.end(), evs.begin(), evs.end());
        }
        return out;
    }

    [[nodiscard]] bool replay_finished() const {
        const std::lock_guard<std::mutex> lock(m_);
        return frame_idx_ >= replay_frames_.size();
    }

private:
    void reset_locked() {
        mode_.store(static_cast<std::uint8_t>(SessionMode::off), std::memory_order_release);
        if (out_.is_open()) {
            out_.flush();
            out_.close();
        }
        qpc_vals_.clear();
        tick32_vals_.clear();
        tick64_vals_.clear();
        tgt_vals_.clear();
        random_chunks_.clear();
        replay_frames_.clear();
        qpc_idx_ = tick32_idx_ = tick64_idx_ = tgt_idx_ = rand_idx_ = frame_idx_ = 0U;
    }

    std::atomic<std::uint8_t> mode_{static_cast<std::uint8_t>(SessionMode::off)};
    mutable std::mutex m_;
    std::ofstream out_;
    std::vector<std::int64_t> qpc_vals_;
    std::vector<std::uint32_t> tick32_vals_;
    std::vector<std::uint64_t> tick64_vals_;
    std::vector<std::uint32_t> tgt_vals_;
    std::vector<std::vector<std::uint8_t>> random_chunks_;
    std::vector<std::pair<std::uint64_t, std::vector<SessionInputEvent>>> replay_frames_;
    std::size_t qpc_idx_ = 0U;
    std::size_t tick32_idx_ = 0U;
    std::size_t tick64_idx_ = 0U;
    std::size_t tgt_idx_ = 0U;
    std::size_t rand_idx_ = 0U;
    std::size_t frame_idx_ = 0U;
};

}  // namespace wse::recomp::shim
