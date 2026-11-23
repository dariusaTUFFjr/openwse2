// Android API 29+ AAudio `AudioDevice` implementation for the `:recomp` process (`libwse_r0.so`).
//
// Port infrastructure: avoids SDL3 Java bootstrap inside `RecompGameActivity` and feeds AAudio's
// real-time callback from a single-producer/single-consumer lock-free float ring buffer so the
// audio callback never blocks on a mutex or allocates memory.

#pragma once

#ifdef __ANDROID__

#include <aaudio/AAudio.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

#include "wse/platform/audio.h"

namespace wse::platform {

class AAudioDevice final : public AudioDevice {
public:
    AAudioDevice() = default;
    AAudioDevice(const AAudioDevice&) = delete;
    AAudioDevice& operator=(const AAudioDevice&) = delete;
    ~AAudioDevice() override { close(); }

    bool open(const AudioDeviceSpec& requested) override {
        close();
        if (requested.sample_rate == 0U || requested.channels != 2U) {
            return false;
        }
        AAudioStreamBuilder* builder = nullptr;
        if (AAudio_createStreamBuilder(&builder) != AAUDIO_OK || builder == nullptr) {
            return false;
        }
        AAudioStreamBuilder_setDirection(builder, AAUDIO_DIRECTION_OUTPUT);
        AAudioStreamBuilder_setSharingMode(builder, AAUDIO_SHARING_MODE_SHARED);
        AAudioStreamBuilder_setFormat(builder, AAUDIO_FORMAT_PCM_FLOAT);
        AAudioStreamBuilder_setChannelCount(builder, 2);
        AAudioStreamBuilder_setSampleRate(builder, static_cast<int32_t>(requested.sample_rate));
        AAudioStreamBuilder_setPerformanceMode(builder, AAUDIO_PERFORMANCE_MODE_LOW_LATENCY);
        AAudioStreamBuilder_setUsage(builder, AAUDIO_USAGE_GAME);
        AAudioStreamBuilder_setDataCallback(builder, &AAudioDevice::data_callback, this);

        AAudioStream* stream = nullptr;
        const aaudio_result_t open_res = AAudioStreamBuilder_openStream(builder, &stream);
        AAudioStreamBuilder_delete(builder);
        if (open_res != AAUDIO_OK || stream == nullptr) {
            return false;
        }

        const int32_t actual_sr = AAudioStream_getSampleRate(stream);
        spec_.sample_rate = actual_sr > 0 ? static_cast<std::uint32_t>(actual_sr) : requested.sample_rate;
        spec_.channels = 2U;

        // Power-of-two ring buffer holding ~340 ms of stereo float audio at 48 kHz (16384 frames).
        ring_frames_ = 16384U;
        ring_mask_ = ring_frames_ - 1U;
        ring_.assign(ring_frames_ * 2U, 0.0F);
        read_pos_.store(0U, std::memory_order_relaxed);
        write_pos_.store(0U, std::memory_order_relaxed);
        paused_.store(false, std::memory_order_relaxed);
        stream_ = stream;

        if (AAudioStream_requestStart(stream_) != AAUDIO_OK) {
            AAudioStream_close(stream_);
            stream_ = nullptr;
            return false;
        }
        open_ = true;
        return true;
    }

    void close() override {
        open_ = false;
        if (stream_ != nullptr) {
            AAudioStream_requestStop(stream_);
            AAudioStream_close(stream_);
            stream_ = nullptr;
        }
        read_pos_.store(0U, std::memory_order_relaxed);
        write_pos_.store(0U, std::memory_order_relaxed);
    }

    [[nodiscard]] bool is_open() const noexcept override { return open_ && stream_ != nullptr; }
    [[nodiscard]] AudioDeviceSpec spec() const noexcept override { return spec_; }

    bool submit_frames(std::span<const float> interleaved_stereo) override {
        if (!is_open() || (interleaved_stereo.size() & 1U) != 0U) {
            return false;
        }
        const std::size_t frames = interleaved_stereo.size() / 2U;
        if (frames == 0U) {
            return true;
        }
        const std::size_t r = read_pos_.load(std::memory_order_acquire);
        const std::size_t w = write_pos_.load(std::memory_order_relaxed);
        const std::size_t used = w - r;
        const std::size_t avail = (used < ring_frames_) ? (ring_frames_ - used) : 0U;
        const std::size_t to_write = std::min(frames, avail);
        if (to_write == 0U) {
            return true;
        }
        const std::size_t idx = w & ring_mask_;
        const std::size_t first = std::min(to_write, ring_frames_ - idx);
        std::memcpy(&ring_[idx * 2U], interleaved_stereo.data(), first * 2U * sizeof(float));
        if (to_write > first) {
            std::memcpy(ring_.data(), interleaved_stereo.data() + first * 2U, (to_write - first) * 2U * sizeof(float));
        }
        write_pos_.store(w + to_write, std::memory_order_release);
        return true;
    }

    [[nodiscard]] std::size_t queued_frames() const noexcept override {
        const std::size_t r = read_pos_.load(std::memory_order_acquire);
        const std::size_t w = write_pos_.load(std::memory_order_acquire);
        return (w >= r) ? (w - r) : 0U;
    }

    void clear() override {
        read_pos_.store(write_pos_.load(std::memory_order_acquire), std::memory_order_release);
    }

    void set_paused(bool paused) override {
        paused_.store(paused, std::memory_order_release);
        if (stream_ == nullptr) {
            return;
        }
        if (paused) {
            AAudioStream_requestPause(stream_);
        } else {
            AAudioStream_requestStart(stream_);
        }
    }

    [[nodiscard]] const char* backend_name() const noexcept override { return "Android AAudio"; }

private:
    static aaudio_data_callback_result_t data_callback(AAudioStream*, void* user_data,
                                                       void* audio_data, int32_t num_frames) {
        auto* self = static_cast<AAudioDevice*>(user_data);
        auto* out = static_cast<float*>(audio_data);
        if (self == nullptr || out == nullptr || num_frames <= 0) {
            return AAUDIO_CALLBACK_RESULT_CONTINUE;
        }
        const auto frames = static_cast<std::size_t>(num_frames);
        if (self->paused_.load(std::memory_order_acquire)) {
            std::memset(out, 0, frames * 2U * sizeof(float));
            return AAUDIO_CALLBACK_RESULT_CONTINUE;
        }
        const std::size_t w = self->write_pos_.load(std::memory_order_acquire);
        const std::size_t r = self->read_pos_.load(std::memory_order_relaxed);
        const std::size_t avail = (w >= r) ? (w - r) : 0U;
        const std::size_t to_read = std::min(frames, avail);
        if (to_read > 0U) {
            const std::size_t idx = r & self->ring_mask_;
            const std::size_t first = std::min(to_read, self->ring_frames_ - idx);
            std::memcpy(out, &self->ring_[idx * 2U], first * 2U * sizeof(float));
            if (to_read > first) {
                std::memcpy(out + first * 2U, self->ring_.data(), (to_read - first) * 2U * sizeof(float));
            }
            self->read_pos_.store(r + to_read, std::memory_order_release);
        }
        if (to_read < frames) {
            std::memset(out + to_read * 2U, 0, (frames - to_read) * 2U * sizeof(float));
        }
        return AAUDIO_CALLBACK_RESULT_CONTINUE;
    }

    AAudioStream* stream_ = nullptr;
    AudioDeviceSpec spec_{};
    bool open_ = false;
    std::size_t ring_frames_ = 0U;
    std::size_t ring_mask_ = 0U;
    std::vector<float> ring_;
    std::atomic<std::size_t> read_pos_{0U};
    std::atomic<std::size_t> write_pos_{0U};
    std::atomic<bool> paused_{false};
};

}  // namespace wse::platform

#endif  // __ANDROID__
