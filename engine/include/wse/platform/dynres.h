// Platform seam: R5b thermal-aware dynamic resolution controller
// (`docs/RECOMP_PLAN.md` "R5 — Graphics").
//
// Port infrastructure, not a reconstruction: mirrors no retail v1.1.5.6 layout
// and carries no PE/PDB citations. Standard library only.
//
// Note: `wse::platform::ThermalStatus` is also defined with PascalCase
// enumerators (`None..Shutdown`) in `wse/platform/performance.h`; avoid
// including both headers in the same translation unit until wired together.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>

namespace wse::platform {

struct DynResConfig {
    float min_scale = 0.5F;
    float max_scale = 1.0F;
    float step = 0.05F;
    double target_frame_ms = 33.3;
    double headroom = 0.85;  // aim for 85% of budget
    int frames_to_decide = 30;
};

// Matches Android PowerManager.THERMAL_STATUS_* levels (0..6).
enum class ThermalStatus {
    none,
    light,
    moderate,
    severe,
    critical,
    emergency,
    shutdown,
};

// Dynamic resolution scale controller with hysteresis and thermal capping.
//
// Rules:
// - Initial scale (and after `reset()`) is `cfg.max_scale` (sanitized to
//   `[cfg.min_scale, 1.0]`).
// - Frame budget threshold is `budget_ms = cfg.target_frame_ms * cfg.headroom`.
//   Recovery threshold ("well below budget") is `recover_ms = budget_ms * 0.85`,
//   leaving a 15% deadband `[recover_ms, budget_ms]` to prevent oscillation.
// - Down-scaling: every `cfg.frames_to_decide` frames, if the window average
//   strictly exceeds `budget_ms`, `scale` decreases by `cfg.step` (clamped at
//   `cfg.min_scale`) and the recovery counter resets.
// - Up-scaling: when `scale` is below the active thermal cap, every
//   `2 * cfg.frames_to_decide` consecutive frames whose average is strictly
//   below `recover_ms` increases `scale` by `cfg.step` (clamped at the active
//   thermal cap).
// - Thermal capping (applied immediately on `update()`):
//   * `none` / `light`: cap = `cfg.max_scale` (lifts any prior thermal cap so
//     subsequent under-budget frames can recover upward)
//   * `moderate`: cap = `clamp(0.85, cfg.min_scale, cfg.max_scale)`
//   * `severe`: cap = `clamp(0.75, cfg.min_scale, cfg.max_scale)`
//   * `critical` / `emergency` / `shutdown`: cap = `cfg.min_scale`
// - `render_size(out_w, out_h, scale)` clamps `scale` to `[0, 1]`, multiplies
//   each output dimension by `scale`, rounds to the nearest multiple of 8, and
//   clamps to `[8, out_dim]` (or `out_dim` itself when `out_dim < 8`).
class DynamicResolution {
public:
    explicit DynamicResolution(DynResConfig cfg = {}) noexcept
        : cfg_(sanitize_config(cfg)), scale_(cfg_.max_scale) {}

    [[nodiscard]] float update(double frame_ms, ThermalStatus thermal) noexcept {
        const float cap = thermal_cap(thermal);
        if (scale_ > cap) {
            scale_ = cap;
            recover_sum_ = 0.0;
            recover_frames_ = 0;
        }

        if (std::isfinite(frame_ms)) {
            const double sample_ms = std::max(0.0, frame_ms);
            const double budget_ms = cfg_.target_frame_ms * cfg_.headroom;
            const double recover_ms = budget_ms * 0.85;

            window_sum_ += sample_ms;
            ++window_frames_;

            bool stepped_down = false;
            if (window_frames_ >= cfg_.frames_to_decide) {
                const double avg_ms = window_sum_ / static_cast<double>(window_frames_);
                window_sum_ = 0.0;
                window_frames_ = 0;
                if (avg_ms > budget_ms && scale_ > cfg_.min_scale) {
                    scale_ = snap_scale(std::max(cfg_.min_scale, scale_ - cfg_.step), cap);
                    recover_sum_ = 0.0;
                    recover_frames_ = 0;
                    stepped_down = true;
                }
            }

            if (!stepped_down) {
                if (scale_ < cap) {
                    recover_sum_ += sample_ms;
                    ++recover_frames_;
                    if (recover_frames_ >= 2 * cfg_.frames_to_decide) {
                        const double rec_avg =
                            recover_sum_ / static_cast<double>(recover_frames_);
                        recover_sum_ = 0.0;
                        recover_frames_ = 0;
                        if (rec_avg < recover_ms) {
                            scale_ = snap_scale(std::min(cap, scale_ + cfg_.step), cap);
                        }
                    }
                } else {
                    recover_sum_ = 0.0;
                    recover_frames_ = 0;
                }
            }
        }

        scale_ = snap_scale(std::clamp(scale_, cfg_.min_scale, cap), cap);
        return scale_;
    }

    [[nodiscard]] float scale() const noexcept {
        return scale_;
    }

    void reset() noexcept {
        scale_ = cfg_.max_scale;
        window_sum_ = 0.0;
        window_frames_ = 0;
        recover_sum_ = 0.0;
        recover_frames_ = 0;
    }

    [[nodiscard]] static std::pair<std::uint32_t, std::uint32_t> render_size(
        std::uint32_t out_w,
        std::uint32_t out_h,
        float scale) noexcept {
        const float s = std::isfinite(scale) ? std::clamp(scale, 0.0F, 1.0F) : 1.0F;
        return {scale_dim_multiple_of_8(out_w, s), scale_dim_multiple_of_8(out_h, s)};
    }

private:
    [[nodiscard]] static DynResConfig sanitize_config(DynResConfig cfg) noexcept {
        cfg.min_scale = std::isfinite(cfg.min_scale) ? std::clamp(cfg.min_scale, 0.1F, 1.0F) : 0.5F;
        cfg.max_scale =
            std::isfinite(cfg.max_scale) ? std::clamp(cfg.max_scale, cfg.min_scale, 1.0F) : 1.0F;
        if (!std::isfinite(cfg.step) || cfg.step <= 0.0F) {
            cfg.step = 0.05F;
        }
        if (!std::isfinite(cfg.target_frame_ms) || cfg.target_frame_ms <= 0.0) {
            cfg.target_frame_ms = 33.3;
        }
        if (!std::isfinite(cfg.headroom) || cfg.headroom <= 0.0 || cfg.headroom > 1.0) {
            cfg.headroom = 0.85;
        }
        if (cfg.frames_to_decide < 1) {
            cfg.frames_to_decide = 1;
        }
        return cfg;
    }

    [[nodiscard]] float thermal_cap(ThermalStatus thermal) const noexcept {
        switch (thermal) {
            case ThermalStatus::none:
            case ThermalStatus::light:
                return cfg_.max_scale;
            case ThermalStatus::moderate:
                return std::clamp(0.85F, cfg_.min_scale, cfg_.max_scale);
            case ThermalStatus::severe:
                return std::clamp(0.75F, cfg_.min_scale, cfg_.max_scale);
            case ThermalStatus::critical:
            case ThermalStatus::emergency:
            case ThermalStatus::shutdown:
                return cfg_.min_scale;
        }
        return cfg_.max_scale;
    }

    [[nodiscard]] float snap_scale(float s, float cap) const noexcept {
        if (std::fabs(s - cfg_.min_scale) < 1e-5F) {
            return cfg_.min_scale;
        }
        if (std::fabs(s - cap) < 1e-5F) {
            return cap;
        }
        return s;
    }

    [[nodiscard]] static std::uint32_t scale_dim_multiple_of_8(std::uint32_t dim,
                                                               float scale) noexcept {
        if (dim < 8U) {
            return dim;
        }
        const double scaled = static_cast<double>(dim) * static_cast<double>(scale);
        const auto rounded = static_cast<std::int64_t>(std::llround(scaled / 8.0)) * 8LL;
        const std::int64_t clamped = std::clamp(rounded, 8LL, static_cast<std::int64_t>(dim));
        return static_cast<std::uint32_t>(clamped);
    }

    DynResConfig cfg_{};
    float scale_ = 1.0F;
    double window_sum_ = 0.0;
    int window_frames_ = 0;
    double recover_sum_ = 0.0;
    int recover_frames_ = 0;
};

}  // namespace wse::platform
