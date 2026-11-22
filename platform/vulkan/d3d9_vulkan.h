// Phase 5 R5a step 4 (`docs/RECOMP_PLAN.md` "R5 — Graphics"): Vulkan 1.1 D3D9 backend
// built directly over `VulkanDriverLoader` (System Vulkan, PanVK, Turnip, Android HAL, and
// Mock ICD). `<vulkan/vulkan.h>` stays private inside `d3d9_vulkan.cpp`.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "driver_loader.h"
#include "wse/recomp/d3d9_backend.h"

namespace wse::platform {

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
    // Persists VkPipelineCache to `config.pipeline_cache_path` (safe to call before process exit).
    void flush_pipeline_cache() noexcept;
    [[nodiscard]] std::string active_presentation_path() const;

    // Diagnostics for tests and reports.
    [[nodiscard]] std::size_t compiled_shader_count() const noexcept;
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
