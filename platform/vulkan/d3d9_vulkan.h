// Phase 5 R5a step 4 (`docs/RECOMP_PLAN.md` "R5 — Graphics"): Vulkan 1.1 D3D9 backend
// built directly over `VulkanDriverLoader` (System Vulkan, PanVK, Turnip, Android HAL, and
// Mock ICD). `<vulkan/vulkan.h>` stays private inside `d3d9_vulkan.cpp`.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "driver_loader.h"
#include "wse/platform/performance.h"
#include "wse/platform/rhi_geometry.h"
#include "wse/recomp/d3d9_backend.h"

namespace wse::platform {

struct ScaledRect {
    std::int32_t left = 0;
    std::int32_t top = 0;
    std::int32_t right = 0;
    std::int32_t bottom = 0;
};

// Rounded like `scale_rect`, so a full-size viewport fills exactly the region the upscaler reads.
[[nodiscard]] inline std::pair<std::uint32_t, std::uint32_t> scaled_render_extent(
    std::uint32_t width, std::uint32_t height, float scale) noexcept {
    if (width == 0U || height == 0U) return {width, height};
    const float s = std::isfinite(scale) ? std::clamp(scale, 0.5F, 1.0F) : 1.0F;
    if (s >= 1.0F) return {width, height};
    const auto sw = static_cast<std::uint32_t>(
        std::clamp<std::int64_t>(std::llround(static_cast<double>(width) * s),
                                 1LL, static_cast<std::int64_t>(width)));
    const auto sh = static_cast<std::uint32_t>(
        std::clamp<std::int64_t>(std::llround(static_cast<double>(height) * s),
                                 1LL, static_cast<std::int64_t>(height)));
    return {sw, sh};
}

[[nodiscard]] inline ScaledRect scale_rect(std::int32_t left, std::int32_t top,
                                           std::int32_t right, std::int32_t bottom,
                                           float scale,
                                           std::uint32_t max_w, std::uint32_t max_h) noexcept {
    const float s = std::isfinite(scale) ? std::clamp(scale, 0.5F, 1.0F) : 1.0F;
    const auto mw = static_cast<std::int32_t>(max_w);
    const auto mh = static_cast<std::int32_t>(max_h);
    auto sc = [s](std::int32_t v) noexcept -> std::int32_t {
        return static_cast<std::int32_t>(std::llround(static_cast<double>(v) * s));
    };
    const std::int32_t l = std::clamp(sc(left), 0, mw);
    const std::int32_t t = std::clamp(sc(top), 0, mh);
    const std::int32_t r = std::clamp(sc(right), l, mw);
    const std::int32_t b = std::clamp(sc(bottom), t, mh);
    return {l, t, r, b};
}

struct VulkanD3d9Config {
    std::string pipeline_cache_path{};     // optional on-disk VkPipelineCache file
    std::uint32_t max_texture_size = 0U;   // optional P10.7 mip-cap for RGBA8 fallback (0 = uncapped)
    bool prefer_bc_compression = true;     // upload BC1/BC2/BC3 natively when supported
    bool prefer_bgra_vertex_format = true; // use B8G8R8A8_UNORM vertex fetch when supported (false forces shader swizzle)
    bool eager_shader_compile = false;     // false: compile shaders lazily on first draw; true: compile at create_shader
    void* native_window = nullptr;         // Android ANativeWindow* (non-owning; null = offscreen)
    // Custom drivers (PanVK/Turnip) show frames through a CPU copy unless this is set: on the
    // reference phone (Mali-G68, PanVK FristOneRR v3.7.0) the zero-copy AHardwareBuffer path drew
    // grey static and stalled the GPU for seconds per frame (2026-11-22).
    bool custom_driver_shared_buffers = false;
    // Optional callback invoked once per presented frame (outside internal locks).
    std::function<void(std::uint64_t frame, double inst_fps, double avg_fps,
                       std::string_view presentation_path)>
        on_frame_presented{};
    std::string texture_cache_dir{};       // optional on-disk ETC2 cache directory (empty = no disk cache)
    bool prefer_etc2 = true;               // transcode DDS to ETC2 when BC is not used and ETC2 is supported
    bool simulate_etc2 = false;            // PC check: transcode to ETC2, decode back to RGBA8 and upload RGBA8
    unsigned etc2_threads = 0U;            // 0 = min(4, hardware_concurrency)
    // R5b FSR upscaling, sharpening, FXAA, and dynamic resolution.
    float render_scale = 1.0F;       // fixed scale in [0.5, 1.0] when auto_resolution is false
    bool auto_resolution = false;
    float min_render_scale = 0.5F;
    float target_fps = 60.0F;
    float sharpness = 0.0F;          // 0 = off; RCAS strength in FSR stops (e.g. 0.2)
    bool fxaa = false;
    // R6 full-resolution touch overlay pass (drawn after FSR/RCAS/FXAA so icons stay razor-sharp).
    std::function<bool(GeometryFrame& out_frame, std::uint32_t width, std::uint32_t height)>
        build_touch_overlay{};
    // R6 / P14.5 menu pinch-to-zoom query: returns {zoom, pan_x, pan_y}; zoom <= 1.01 disables.
    std::function<std::array<float, 3>()> query_menu_zoom{};
};

class VulkanD3d9Backend final : public wse::recomp::D3d9Backend {
public:
    explicit VulkanD3d9Backend(std::shared_ptr<VulkanDriverLoader> loader,
                               VulkanD3d9Config config = {});
    ~VulkanD3d9Backend() override;

    VulkanD3d9Backend(const VulkanD3d9Backend&) = delete;
    VulkanD3d9Backend& operator=(const VulkanD3d9Backend&) = delete;

    bool init(std::uint32_t width, std::uint32_t height, std::uint32_t color_fmt,
              std::uint32_t depth_fmt, std::string* error_out = nullptr) override;
    bool reset(std::uint32_t width, std::uint32_t height, std::uint32_t color_fmt,
               std::uint32_t depth_fmt, std::string* error_out = nullptr) override;
    [[nodiscard]] const wse::recomp::D3d9DriverReport& report() const noexcept override;

    void create_buffer(std::uint64_t id, bool is_index, std::uint32_t size,
                       bool dynamic, std::uint32_t fmt) override;
    void update_buffer(std::uint64_t id, std::uint32_t offset,
                       std::span<const std::uint8_t> bytes, std::uint32_t lock_flags) override;
    void destroy_buffer(std::uint64_t id) override;

    void create_texture(std::uint64_t id, int kind, std::uint32_t width, std::uint32_t height,
                        std::uint32_t depth, std::uint32_t levels, std::uint32_t fmt,
                        std::uint32_t usage) override;
    void upload_texture_level(std::uint64_t id, std::uint32_t face, std::uint32_t level,
                              std::span<const std::uint8_t> bytes, std::uint32_t row_pitch) override;
    void upload_texture_dds(std::uint64_t id, std::span<const std::uint8_t> dds_bytes) override;
    void destroy_texture(std::uint64_t id) override;

    void create_surface(std::uint64_t id, std::uint32_t width, std::uint32_t height,
                        std::uint32_t fmt, std::uint32_t usage,
                        std::uint64_t parent_tex, std::uint32_t face, std::uint32_t level) override;
    void destroy_surface(std::uint64_t id) override;

    void create_vertex_decl(std::uint64_t id,
                            std::span<const wse::recomp::D3d9VertexElement> elems) override;
    void destroy_vertex_decl(std::uint64_t id) override;

    void create_shader(std::uint64_t id, bool is_pixel,
                       std::span<const std::uint8_t> bytecode) override;
    void destroy_shader(std::uint64_t id) override;

    void clear(const wse::recomp::D3d9DrawState& state, std::uint32_t flags, std::uint32_t argb,
               float z, std::uint32_t stencil,
               std::span<const wse::recomp::D3d9Rect> rects) override;
    void draw_indexed(const wse::recomp::D3d9DrawState& state, std::uint32_t prim_type,
                      std::int32_t base_vertex, std::uint32_t min_v, std::uint32_t num_v,
                      std::uint32_t start_index, std::uint32_t prim_count) override;
    void draw_primitive(const wse::recomp::D3d9DrawState& state, std::uint32_t prim_type,
                        std::uint32_t start_vertex, std::uint32_t prim_count) override;
    bool present(std::vector<std::uint8_t>* out_rgba8 = nullptr,
                 std::uint32_t* out_width = nullptr,
                 std::uint32_t* out_height = nullptr) override;

    // Thread-safe Android ANativeWindow* attachment (pass nullptr on surfaceDestroyed).
    // Safe no-op on non-Android platforms.
    void set_native_window(void* native_window) noexcept;
    // Updates the thermal status used by DynamicResolution (default ThermalStatus::None).
    void set_thermal_status(ThermalStatus status) noexcept;
    // Active render scale in [0.5, 1.0] for the current/next frame.
    [[nodiscard]] float current_render_scale() const noexcept;
    // Persists VkPipelineCache to `config.pipeline_cache_path` (safe to call before process exit).
    void flush_pipeline_cache() noexcept;
    [[nodiscard]] std::string active_presentation_path() const;

    // Diagnostics for tests and reports.
    [[nodiscard]] std::size_t compiled_shader_count() const noexcept;
    // Time spent turning game textures into GPU images (decoding `.dds` to RGBA8 where the GPU
    // has no BC support, and staging the upload), for the load-time breakdown in reports.
    struct TextureTimings {
        double seconds = 0.0;
        std::uint64_t textures = 0;
        std::uint64_t bytes = 0;  // uploaded
        std::uint64_t etc2_textures = 0;
        std::uint64_t etc2_cached = 0;
        double etc2_transcode_ms = 0.0;
        std::uint64_t etc2_cache_read_failures = 0;
        std::uint64_t etc2_cache_write_failures = 0;
        std::uint64_t rgba8_fallbacks = 0;
        std::uint64_t level_rgba8_fallbacks = 0;
    };
    [[nodiscard]] TextureTimings texture_timings() const;
    [[nodiscard]] std::size_t failed_shader_count() const noexcept;
    [[nodiscard]] std::size_t cached_pipeline_count() const noexcept;
    [[nodiscard]] std::size_t cached_sampler_count() const noexcept;
    [[nodiscard]] std::size_t cached_descriptor_set_count() const noexcept;
    [[nodiscard]] const std::string& last_error() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace wse::platform
