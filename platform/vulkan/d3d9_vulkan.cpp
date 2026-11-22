// Phase 5 R5a step 4 (`docs/RECOMP_PLAN.md` "R5 — Graphics"): Vulkan 1.1 D3D9 backend.
// Targets System Vulkan, direct-ICD PanVK (Mali), direct-ICD Turnip (Adreno), Android HAL,
// and Mock ICD using Vulkan 1.1 core features only (<= 2 descriptor sets, <= 16 samplers/stage,
// <= 23 varying locations, zero geometry/tessellation shaders, graceful fallbacks for missing
// BC compression, BGRA vertex fetch, anisotropy, wireframe, and VkPipelineCache).

#include "d3d9_vulkan.h"
#include "driver_loader_internal.h"

#include "wse/asset/dds.h"
#include "wse/asset/shader.h"
#include "wse/asset/shader_compiler.h"
#include "wse/recomp/d3d9_shader.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef __ANDROID__
#include <android/hardware_buffer.h>
#include <android/native_window.h>
#include <android/surface_control.h>
#endif

namespace wse::platform {

namespace {

using namespace wse::recomp;

constexpr std::uint32_t kMaxSamplers = 16U;
constexpr std::uint32_t kDummyVertexBinding = 15U;
constexpr VkDeviceSize kInitialUboRingBytes = 8ULL * 1024ULL * 1024ULL;    // 8 MiB, doubles as needed
constexpr VkDeviceSize kInitialDynGeomBytes = 256ULL * 1024ULL;            // unused by draws since the 2026-11-22 pass
// Frames the CPU may record ahead of the GPU (2026-11-22 performance pass): the game prepares
// frame N+1 while the GPU draws frame N. Each slot owns its command buffers, fence and host rings.
constexpr std::size_t kFramesInFlight = 2U;
// Upload staging per frame slot: grows as needed; past this size an upload first waits for the
// frame so far (long loads that never present would otherwise grow it without bound).
constexpr VkDeviceSize kInitialStagingBytes = 4ULL * 1024ULL * 1024ULL;
constexpr VkDeviceSize kStagingFlushBytes = 64ULL * 1024ULL * 1024ULL;
// D3DLOCK_* flags passed to `update_buffer`.
constexpr std::uint32_t kLockNoOverwrite = 0x1000U;
constexpr std::uint32_t kLockDiscard = 0x2000U;
// A dynamic buffer keeps at most this many renamed copies before it waits for the GPU.
constexpr std::size_t kMaxDynInstances = 16U;
constexpr VkDeviceSize kSharedAllocBlockBytes = 16ULL * 1024ULL * 1024ULL; // 16 MiB device blocks

struct StageUboData {
    float c[256][4];
    std::int32_t i[16][4];
    std::int32_t b[16][4];
    float posfix[4];
};
static_assert(sizeof(StageUboData) == 4624U, "StageUboData must match d3d9_shader.h std140 block");

[[nodiscard]] VkDeviceSize align_up(VkDeviceSize v, VkDeviceSize a) noexcept {
    if (a <= 1U) return v;
    const VkDeviceSize r = v % a;
    return r == 0U ? v : v + (a - r);
}

[[nodiscard]] std::uint64_t fnv1a_bytes(std::span<const std::uint8_t> data) noexcept {
    std::uint64_t h = 14695981039346656037ULL;
    for (std::uint8_t b : data) {
        h ^= b;
        h *= 1099511628211ULL;
    }
    return h;
}

void hash_mix(std::uint64_t& h, std::uint64_t v) noexcept {
    h ^= v;
    h *= 1099511628211ULL;
}

void set_err(std::string* out, std::string msg) {
    if (out != nullptr) *out = std::move(msg);
}

[[nodiscard]] bool has_stencil_aspect(VkFormat fmt) noexcept {
    return fmt == VK_FORMAT_D24_UNORM_S8_UINT || fmt == VK_FORMAT_D32_SFLOAT_S8_UINT ||
           fmt == VK_FORMAT_D16_UNORM_S8_UINT;
}

[[nodiscard]] VkCompareOp map_compare_op(std::uint32_t d3d_cmp) noexcept {
    switch (d3d_cmp) {
    case 1: return VK_COMPARE_OP_NEVER;
    case 2: return VK_COMPARE_OP_LESS;
    case 3: return VK_COMPARE_OP_EQUAL;
    case 4: return VK_COMPARE_OP_LESS_OR_EQUAL;
    case 5: return VK_COMPARE_OP_GREATER;
    case 6: return VK_COMPARE_OP_NOT_EQUAL;
    case 7: return VK_COMPARE_OP_GREATER_OR_EQUAL;
    case 8: return VK_COMPARE_OP_ALWAYS;
    default: return VK_COMPARE_OP_LESS_OR_EQUAL;
    }
}

[[nodiscard]] VkBlendFactor map_blend_factor(std::uint32_t d3d_blend) noexcept {
    switch (d3d_blend) {
    case 1: return VK_BLEND_FACTOR_ZERO;
    case 2: return VK_BLEND_FACTOR_ONE;
    case 3: return VK_BLEND_FACTOR_SRC_COLOR;
    case 4: return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
    case 5: return VK_BLEND_FACTOR_SRC_ALPHA;
    case 6: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    case 7: return VK_BLEND_FACTOR_DST_ALPHA;
    case 8: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
    case 9: return VK_BLEND_FACTOR_DST_COLOR;
    case 10: return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
    case 11: return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
    case 14: return VK_BLEND_FACTOR_CONSTANT_COLOR;
    case 15: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
    default: return VK_BLEND_FACTOR_ONE;
    }
}

[[nodiscard]] VkBlendOp map_blend_op(std::uint32_t d3d_op) noexcept {
    switch (d3d_op) {
    case 1: return VK_BLEND_OP_ADD;
    case 2: return VK_BLEND_OP_SUBTRACT;
    case 3: return VK_BLEND_OP_REVERSE_SUBTRACT;
    case 4: return VK_BLEND_OP_MIN;
    case 5: return VK_BLEND_OP_MAX;
    default: return VK_BLEND_OP_ADD;
    }
}

[[nodiscard]] VkSamplerAddressMode map_address_mode(std::uint32_t m) noexcept {
    switch (m) {
    case 1: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    case 2: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    case 3: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    case 4: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    case 5: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;  // MIRRORONCE core fallback
    default: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    }
}

[[nodiscard]] VkFilter map_filter(std::uint32_t f) noexcept {
    return (f == 2U || f == 3U) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
}

[[nodiscard]] VkPrimitiveTopology map_topology(std::uint32_t prim_type) noexcept {
    switch (prim_type) {
    case 1: return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    case 2: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    case 3: return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
    case 4: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    case 5: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
    case 6: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
    default: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    }
}

[[nodiscard]] std::uint32_t vertex_count_for_prims(std::uint32_t prim_type, std::uint32_t prim_count) noexcept {
    if (prim_count == 0U) return 0U;
    switch (prim_type) {
    case 1: return prim_count;
    case 2: return prim_count * 2U;
    case 3: return prim_count + 1U;
    case 4: return prim_count * 3U;
    case 5: return prim_count + 2U;
    case 6: return prim_count + 2U;
    default: return 0U;
    }
}

[[nodiscard]] VkFormat map_decl_type(std::uint8_t type, bool bgra_vertex_ok, bool& needs_bgra_swizzle) noexcept {
    needs_bgra_swizzle = false;
    switch (type) {
    case 0: return VK_FORMAT_R32_SFLOAT;                 // FLOAT1
    case 1: return VK_FORMAT_R32G32_SFLOAT;              // FLOAT2
    case 2: return VK_FORMAT_R32G32B32_SFLOAT;           // FLOAT3
    case 3: return VK_FORMAT_R32G32B32A32_SFLOAT;        // FLOAT4
    case 4:                                              // D3DCOLOR (BGRA8 UNORM)
        if (bgra_vertex_ok) return VK_FORMAT_B8G8R8A8_UNORM;
        needs_bgra_swizzle = true;
        return VK_FORMAT_R8G8B8A8_UNORM;
    case 5: return VK_FORMAT_R8G8B8A8_USCALED;           // UBYTE4
    case 6: return VK_FORMAT_R16G16_SSCALED;             // SHORT2
    case 7: return VK_FORMAT_R16G16B16A16_SSCALED;       // SHORT4
    case 8: return VK_FORMAT_R8G8B8A8_UNORM;             // UBYTE4N
    case 9: return VK_FORMAT_R16G16_SNORM;               // SHORT2N
    case 10: return VK_FORMAT_R16G16B16A16_SNORM;        // SHORT4N
    case 11: return VK_FORMAT_R16G16_UNORM;              // USHORT2N
    case 12: return VK_FORMAT_R16G16B16A16_UNORM;        // USHORT4N
    case 15: return VK_FORMAT_R16G16_SFLOAT;             // FLOAT16_2
    case 16: return VK_FORMAT_R16G16B16A16_SFLOAT;       // FLOAT16_4
    default: return VK_FORMAT_R32G32B32A32_SFLOAT;
    }
}

[[nodiscard]] bool is_dxt_d3dfmt(std::uint32_t fmt) noexcept {
    return fmt == 0x31545844U || fmt == 0x32545844U || fmt == 0x33545844U ||
           fmt == 0x34545844U || fmt == 0x35545844U;
}

[[nodiscard]] VkFormat dxt_d3dfmt_to_vk(std::uint32_t fmt) noexcept {
    if (fmt == 0x31545844U) return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
    if (fmt == 0x32545844U || fmt == 0x33545844U) return VK_FORMAT_BC2_UNORM_BLOCK;
    if (fmt == 0x34545844U || fmt == 0x35545844U) return VK_FORMAT_BC3_UNORM_BLOCK;
    return VK_FORMAT_R8G8B8A8_UNORM;
}

[[nodiscard]] VkFormat dds_bc_to_vk(wse::asset::DdsFormat fmt) noexcept {
    switch (fmt) {
    case wse::asset::DdsFormat::BC1_UNORM: return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
    case wse::asset::DdsFormat::BC2_UNORM: return VK_FORMAT_BC2_UNORM_BLOCK;
    case wse::asset::DdsFormat::BC3_UNORM: return VK_FORMAT_BC3_UNORM_BLOCK;
    default: return VK_FORMAT_UNDEFINED;
    }
}

// Converts a CPU-locked D3D9 surface level into tightly packed RGBA8 pixels.
void convert_d3d9_level_to_rgba8(std::uint32_t fmt, std::uint32_t w, std::uint32_t h,
                                 std::span<const std::uint8_t> src, std::uint32_t row_pitch,
                                 std::vector<std::uint8_t>& out) {
    out.assign(static_cast<std::size_t>(w) * h * 4U, 0);
    if (src.empty() || w == 0U || h == 0U) return;
    for (std::uint32_t y = 0; y < h; ++y) {
        const std::size_t row_off = static_cast<std::size_t>(y) * row_pitch;
        if (row_off >= src.size()) break;
        const std::uint8_t* r = src.data() + row_off;
        const std::size_t rem = src.size() - row_off;
        std::uint8_t* d = out.data() + static_cast<std::size_t>(y) * w * 4U;
        switch (fmt) {
        case 21:  // D3DFMT_A8R8G8B8 (BGRA)
            for (std::uint32_t x = 0; x < w && static_cast<std::size_t>(x) * 4U + 4U <= rem; ++x) {
                d[4 * x + 0] = r[4 * x + 2];
                d[4 * x + 1] = r[4 * x + 1];
                d[4 * x + 2] = r[4 * x + 0];
                d[4 * x + 3] = r[4 * x + 3];
            }
            break;
        case 22:  // D3DFMT_X8R8G8B8 (BGRX)
            for (std::uint32_t x = 0; x < w && static_cast<std::size_t>(x) * 4U + 4U <= rem; ++x) {
                d[4 * x + 0] = r[4 * x + 2];
                d[4 * x + 1] = r[4 * x + 1];
                d[4 * x + 2] = r[4 * x + 0];
                d[4 * x + 3] = 255U;
            }
            break;
        case 32:  // D3DFMT_A8B8G8R8 (RGBA)
            for (std::uint32_t x = 0; x < w && static_cast<std::size_t>(x) * 4U + 4U <= rem; ++x) {
                std::memcpy(d + 4 * x, r + 4 * x, 4);
            }
            break;
        case 23:  // D3DFMT_R5G6B5
            for (std::uint32_t x = 0; x < w && static_cast<std::size_t>(x) * 2U + 2U <= rem; ++x) {
                const std::uint16_t p = static_cast<std::uint16_t>(r[2 * x] | (r[2 * x + 1] << 8));
                const std::uint8_t r5 = static_cast<std::uint8_t>((p >> 11) & 0x1FU);
                const std::uint8_t g6 = static_cast<std::uint8_t>((p >> 5) & 0x3FU);
                const std::uint8_t b5 = static_cast<std::uint8_t>(p & 0x1FU);
                d[4 * x + 0] = static_cast<std::uint8_t>((r5 << 3) | (r5 >> 2));
                d[4 * x + 1] = static_cast<std::uint8_t>((g6 << 2) | (g6 >> 4));
                d[4 * x + 2] = static_cast<std::uint8_t>((b5 << 3) | (b5 >> 2));
                d[4 * x + 3] = 255U;
            }
            break;
        case 28:  // D3DFMT_A8
            for (std::uint32_t x = 0; x < w && x < rem; ++x) {
                d[4 * x + 0] = 0U;
                d[4 * x + 1] = 0U;
                d[4 * x + 2] = 0U;
                d[4 * x + 3] = r[x];
            }
            break;
        case 50:  // D3DFMT_L8
            for (std::uint32_t x = 0; x < w && x < rem; ++x) {
                d[4 * x + 0] = r[x];
                d[4 * x + 1] = r[x];
                d[4 * x + 2] = r[x];
                d[4 * x + 3] = 255U;
            }
            break;
        case 51:  // D3DFMT_A8L8
            for (std::uint32_t x = 0; x < w && static_cast<std::size_t>(x) * 2U + 2U <= rem; ++x) {
                d[4 * x + 0] = r[2 * x];
                d[4 * x + 1] = r[2 * x];
                d[4 * x + 2] = r[2 * x];
                d[4 * x + 3] = r[2 * x + 1];
            }
            break;
        default:
            for (std::uint32_t x = 0; x < w && static_cast<std::size_t>(x) * 4U + 4U <= rem; ++x) {
                d[4 * x + 0] = r[4 * x + 2];
                d[4 * x + 1] = r[4 * x + 1];
                d[4 * x + 2] = r[4 * x + 0];
                d[4 * x + 3] = r[4 * x + 3];
            }
            break;
        }
    }
}

struct VkProcs {
    PFN_vkGetPhysicalDeviceProperties get_phys_props = nullptr;
    PFN_vkGetPhysicalDeviceFeatures get_phys_features = nullptr;
    PFN_vkGetPhysicalDeviceMemoryProperties get_phys_mem_props = nullptr;
    PFN_vkGetPhysicalDeviceFormatProperties get_phys_format_props = nullptr;

    PFN_vkDeviceWaitIdle device_wait_idle = nullptr;
    PFN_vkAllocateMemory allocate_memory = nullptr;
    PFN_vkFreeMemory free_memory = nullptr;
    PFN_vkMapMemory map_memory = nullptr;
    PFN_vkUnmapMemory unmap_memory = nullptr;
    PFN_vkFlushMappedMemoryRanges flush_memory = nullptr;
    PFN_vkInvalidateMappedMemoryRanges invalidate_memory = nullptr;

    PFN_vkCreateBuffer create_buffer = nullptr;
    PFN_vkDestroyBuffer destroy_buffer = nullptr;
    PFN_vkGetBufferMemoryRequirements get_buffer_mem_reqs = nullptr;
    PFN_vkBindBufferMemory bind_buffer_memory = nullptr;

    PFN_vkCreateImage create_image = nullptr;
    PFN_vkDestroyImage destroy_image = nullptr;
    PFN_vkGetImageMemoryRequirements get_image_mem_reqs = nullptr;
    PFN_vkBindImageMemory bind_image_memory = nullptr;
    PFN_vkCreateImageView create_image_view = nullptr;
    PFN_vkDestroyImageView destroy_image_view = nullptr;
    PFN_vkCreateSampler create_sampler = nullptr;
    PFN_vkDestroySampler destroy_sampler = nullptr;

    PFN_vkCreateShaderModule create_shader_module = nullptr;
    PFN_vkDestroyShaderModule destroy_shader_module = nullptr;
    PFN_vkCreatePipelineCache create_pipeline_cache = nullptr;
    PFN_vkDestroyPipelineCache destroy_pipeline_cache = nullptr;
    PFN_vkGetPipelineCacheData get_pipeline_cache_data = nullptr;
    PFN_vkCreateDescriptorSetLayout create_desc_set_layout = nullptr;
    PFN_vkDestroyDescriptorSetLayout destroy_desc_set_layout = nullptr;
    PFN_vkCreateDescriptorPool create_desc_pool = nullptr;
    PFN_vkDestroyDescriptorPool destroy_desc_pool = nullptr;
    PFN_vkResetDescriptorPool reset_desc_pool = nullptr;
    PFN_vkAllocateDescriptorSets allocate_desc_sets = nullptr;
    PFN_vkUpdateDescriptorSets update_desc_sets = nullptr;
    PFN_vkCreatePipelineLayout create_pipeline_layout = nullptr;
    PFN_vkDestroyPipelineLayout destroy_pipeline_layout = nullptr;
    PFN_vkCreateRenderPass create_render_pass = nullptr;
    PFN_vkDestroyRenderPass destroy_render_pass = nullptr;
    PFN_vkCreateFramebuffer create_framebuffer = nullptr;
    PFN_vkDestroyFramebuffer destroy_framebuffer = nullptr;
    PFN_vkCreateGraphicsPipelines create_graphics_pipelines = nullptr;
    PFN_vkDestroyPipeline destroy_pipeline = nullptr;

    PFN_vkAllocateCommandBuffers allocate_cmd_buffers = nullptr;
    PFN_vkFreeCommandBuffers free_cmd_buffers = nullptr;
    PFN_vkBeginCommandBuffer begin_cmd_buffer = nullptr;
    PFN_vkEndCommandBuffer end_cmd_buffer = nullptr;
    PFN_vkResetCommandBuffer reset_cmd_buffer = nullptr;
    PFN_vkQueueSubmit queue_submit = nullptr;
    PFN_vkCreateFence create_fence = nullptr;
    PFN_vkDestroyFence destroy_fence = nullptr;
    PFN_vkWaitForFences wait_for_fences = nullptr;
    PFN_vkResetFences reset_fences = nullptr;
    PFN_vkCreateSemaphore create_semaphore = nullptr;
    PFN_vkDestroySemaphore destroy_semaphore = nullptr;

    PFN_vkCmdPipelineBarrier cmd_pipeline_barrier = nullptr;
    PFN_vkCmdCopyBuffer cmd_copy_buffer = nullptr;
    PFN_vkCmdCopyBufferToImage cmd_copy_buffer_to_image = nullptr;
    PFN_vkCmdCopyImageToBuffer cmd_copy_image_to_buffer = nullptr;
    PFN_vkCmdCopyImage cmd_copy_image = nullptr;
    PFN_vkCmdBlitImage cmd_blit_image = nullptr;
    PFN_vkCmdClearColorImage cmd_clear_color_image = nullptr;
    PFN_vkCmdBeginRenderPass cmd_begin_render_pass = nullptr;
    PFN_vkCmdEndRenderPass cmd_end_render_pass = nullptr;
    PFN_vkCmdClearAttachments cmd_clear_attachments = nullptr;
    PFN_vkCmdBindPipeline cmd_bind_pipeline = nullptr;
    PFN_vkCmdSetViewport cmd_set_viewport = nullptr;
    PFN_vkCmdSetScissor cmd_set_scissor = nullptr;
    PFN_vkCmdSetDepthBias cmd_set_depth_bias = nullptr;
    PFN_vkCmdSetBlendConstants cmd_set_blend_constants = nullptr;
    PFN_vkCmdBindDescriptorSets cmd_bind_desc_sets = nullptr;
    PFN_vkCmdBindVertexBuffers cmd_bind_vertex_buffers = nullptr;
    PFN_vkCmdBindIndexBuffer cmd_bind_index_buffer = nullptr;
    PFN_vkCmdDraw cmd_draw = nullptr;
    PFN_vkCmdDrawIndexed cmd_draw_indexed = nullptr;

#ifdef __ANDROID__
    // Optional WSI + Android AHardwareBuffer presentation entry points.
    PFN_vkCreateAndroidSurfaceKHR create_android_surface = nullptr;
    PFN_vkDestroySurfaceKHR destroy_surface = nullptr;
    PFN_vkGetPhysicalDeviceSurfaceSupportKHR get_phys_surface_support = nullptr;
    PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR get_phys_surface_caps = nullptr;
    PFN_vkGetPhysicalDeviceSurfaceFormatsKHR get_phys_surface_formats = nullptr;
    PFN_vkGetPhysicalDeviceSurfacePresentModesKHR get_phys_surface_present_modes = nullptr;
    PFN_vkCreateSwapchainKHR create_swapchain = nullptr;
    PFN_vkDestroySwapchainKHR destroy_swapchain = nullptr;
    PFN_vkGetSwapchainImagesKHR get_swapchain_images = nullptr;
    PFN_vkAcquireNextImageKHR acquire_next_image = nullptr;
    PFN_vkQueuePresentKHR queue_present = nullptr;
    PFN_vkGetAndroidHardwareBufferPropertiesANDROID get_ahb_props = nullptr;
#endif

    void resolve(const VulkanRawHandles& h) noexcept {
        auto ri = [&](const char* name, auto& fn) {
            fn = (h.gipa != nullptr && h.instance != VK_NULL_HANDLE)
                     ? reinterpret_cast<std::remove_reference_t<decltype(fn)>>(h.gipa(h.instance, name))
                     : nullptr;
        };
        auto rd = [&](const char* name, auto& fn) {
            fn = nullptr;
            if (h.gdpa != nullptr && h.device != VK_NULL_HANDLE) {
                fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(h.gdpa(h.device, name));
            }
            if (fn == nullptr && h.gipa != nullptr && h.instance != VK_NULL_HANDLE) {
                fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(h.gipa(h.instance, name));
            }
        };
        ri("vkGetPhysicalDeviceProperties", get_phys_props);
        ri("vkGetPhysicalDeviceFeatures", get_phys_features);
        ri("vkGetPhysicalDeviceMemoryProperties", get_phys_mem_props);
        ri("vkGetPhysicalDeviceFormatProperties", get_phys_format_props);

        rd("vkDeviceWaitIdle", device_wait_idle);
        rd("vkAllocateMemory", allocate_memory);
        rd("vkFreeMemory", free_memory);
        rd("vkMapMemory", map_memory);
        rd("vkUnmapMemory", unmap_memory);
        rd("vkFlushMappedMemoryRanges", flush_memory);
        rd("vkInvalidateMappedMemoryRanges", invalidate_memory);

        rd("vkCreateBuffer", create_buffer);
        rd("vkDestroyBuffer", destroy_buffer);
        rd("vkGetBufferMemoryRequirements", get_buffer_mem_reqs);
        rd("vkBindBufferMemory", bind_buffer_memory);

        rd("vkCreateImage", create_image);
        rd("vkDestroyImage", destroy_image);
        rd("vkGetImageMemoryRequirements", get_image_mem_reqs);
        rd("vkBindImageMemory", bind_image_memory);
        rd("vkCreateImageView", create_image_view);
        rd("vkDestroyImageView", destroy_image_view);
        rd("vkCreateSampler", create_sampler);
        rd("vkDestroySampler", destroy_sampler);

        rd("vkCreateShaderModule", create_shader_module);
        rd("vkDestroyShaderModule", destroy_shader_module);
        rd("vkCreatePipelineCache", create_pipeline_cache);
        rd("vkDestroyPipelineCache", destroy_pipeline_cache);
        rd("vkGetPipelineCacheData", get_pipeline_cache_data);
        rd("vkCreateDescriptorSetLayout", create_desc_set_layout);
        rd("vkDestroyDescriptorSetLayout", destroy_desc_set_layout);
        rd("vkCreateDescriptorPool", create_desc_pool);
        rd("vkDestroyDescriptorPool", destroy_desc_pool);
        rd("vkResetDescriptorPool", reset_desc_pool);
        rd("vkAllocateDescriptorSets", allocate_desc_sets);
        rd("vkUpdateDescriptorSets", update_desc_sets);
        rd("vkCreatePipelineLayout", create_pipeline_layout);
        rd("vkDestroyPipelineLayout", destroy_pipeline_layout);
        rd("vkCreateRenderPass", create_render_pass);
        rd("vkDestroyRenderPass", destroy_render_pass);
        rd("vkCreateFramebuffer", create_framebuffer);
        rd("vkDestroyFramebuffer", destroy_framebuffer);
        rd("vkCreateGraphicsPipelines", create_graphics_pipelines);
        rd("vkDestroyPipeline", destroy_pipeline);

        rd("vkAllocateCommandBuffers", allocate_cmd_buffers);
        rd("vkFreeCommandBuffers", free_cmd_buffers);
        rd("vkBeginCommandBuffer", begin_cmd_buffer);
        rd("vkEndCommandBuffer", end_cmd_buffer);
        rd("vkResetCommandBuffer", reset_cmd_buffer);
        rd("vkQueueSubmit", queue_submit);
        rd("vkCreateFence", create_fence);
        rd("vkDestroyFence", destroy_fence);
        rd("vkWaitForFences", wait_for_fences);
        rd("vkResetFences", reset_fences);
        rd("vkCreateSemaphore", create_semaphore);
        rd("vkDestroySemaphore", destroy_semaphore);

        rd("vkCmdPipelineBarrier", cmd_pipeline_barrier);
        rd("vkCmdCopyBuffer", cmd_copy_buffer);
        rd("vkCmdCopyBufferToImage", cmd_copy_buffer_to_image);
        rd("vkCmdCopyImageToBuffer", cmd_copy_image_to_buffer);
        rd("vkCmdCopyImage", cmd_copy_image);
        rd("vkCmdBlitImage", cmd_blit_image);
        rd("vkCmdClearColorImage", cmd_clear_color_image);
        rd("vkCmdBeginRenderPass", cmd_begin_render_pass);
        rd("vkCmdEndRenderPass", cmd_end_render_pass);
        rd("vkCmdClearAttachments", cmd_clear_attachments);
        rd("vkCmdBindPipeline", cmd_bind_pipeline);
        rd("vkCmdSetViewport", cmd_set_viewport);
        rd("vkCmdSetScissor", cmd_set_scissor);
        rd("vkCmdSetDepthBias", cmd_set_depth_bias);
        rd("vkCmdSetBlendConstants", cmd_set_blend_constants);
        rd("vkCmdBindDescriptorSets", cmd_bind_desc_sets);
        rd("vkCmdBindVertexBuffers", cmd_bind_vertex_buffers);
        rd("vkCmdBindIndexBuffer", cmd_bind_index_buffer);
        rd("vkCmdDraw", cmd_draw);
        rd("vkCmdDrawIndexed", cmd_draw_indexed);

#ifdef __ANDROID__
        ri("vkCreateAndroidSurfaceKHR", create_android_surface);
        ri("vkDestroySurfaceKHR", destroy_surface);
        ri("vkGetPhysicalDeviceSurfaceSupportKHR", get_phys_surface_support);
        ri("vkGetPhysicalDeviceSurfaceCapabilitiesKHR", get_phys_surface_caps);
        ri("vkGetPhysicalDeviceSurfaceFormatsKHR", get_phys_surface_formats);
        ri("vkGetPhysicalDeviceSurfacePresentModesKHR", get_phys_surface_present_modes);
        rd("vkCreateSwapchainKHR", create_swapchain);
        rd("vkDestroySwapchainKHR", destroy_swapchain);
        rd("vkGetSwapchainImagesKHR", get_swapchain_images);
        rd("vkAcquireNextImageKHR", acquire_next_image);
        rd("vkQueuePresentKHR", queue_present);
        rd("vkGetAndroidHardwareBufferPropertiesANDROID", get_ahb_props);
#endif
    }

    [[nodiscard]] bool has_core_gpu() const noexcept {
        return get_phys_props && get_phys_mem_props && get_phys_format_props && device_wait_idle &&
               allocate_memory && free_memory && map_memory && unmap_memory && flush_memory &&
               invalidate_memory && create_buffer && destroy_buffer && get_buffer_mem_reqs &&
               bind_buffer_memory && create_image && destroy_image && get_image_mem_reqs &&
               bind_image_memory && create_image_view && destroy_image_view && create_sampler &&
               destroy_sampler && create_shader_module && destroy_shader_module &&
               create_desc_set_layout && destroy_desc_set_layout && create_desc_pool &&
               destroy_desc_pool && reset_desc_pool && allocate_desc_sets && update_desc_sets &&
               create_pipeline_layout && destroy_pipeline_layout && create_render_pass &&
               destroy_render_pass && create_framebuffer && destroy_framebuffer &&
               create_graphics_pipelines && destroy_pipeline && allocate_cmd_buffers &&
               free_cmd_buffers && begin_cmd_buffer && end_cmd_buffer && reset_cmd_buffer &&
               queue_submit && create_fence && destroy_fence && wait_for_fences && reset_fences &&
               cmd_pipeline_barrier && cmd_copy_buffer && cmd_copy_buffer_to_image &&
               cmd_copy_image_to_buffer && cmd_begin_render_pass && cmd_end_render_pass &&
               cmd_clear_attachments && cmd_bind_pipeline && cmd_set_viewport && cmd_set_scissor &&
               cmd_bind_desc_sets && cmd_bind_vertex_buffers && cmd_bind_index_buffer &&
               cmd_draw && cmd_draw_indexed;
    }
};

struct MemAlloc {
    std::size_t block = 0;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    VkDeviceSize size = 0;
    void* mapped = nullptr;
    bool coherent = false;

    [[nodiscard]] bool valid() const noexcept { return memory != VK_NULL_HANDLE; }
};

class D3d9MemAllocator {
public:
    void init(VkDevice dev, const VkProcs* vk, VkPhysicalDevice phys, VkDeviceSize atom_size) noexcept {
        dev_ = dev;
        vk_ = vk;
        atom_ = std::max<VkDeviceSize>(1U, atom_size);
        std::memset(&props_, 0, sizeof(props_));
        if (vk_ != nullptr && vk_->get_phys_mem_props != nullptr && phys != VK_NULL_HANDLE) {
            vk_->get_phys_mem_props(phys, &props_);
        }
    }

    [[nodiscard]] bool alloc(const VkMemoryRequirements& reqs, bool is_image,
                             VkMemoryPropertyFlags req_flags, VkMemoryPropertyFlags pref_flags,
                             MemAlloc& out, std::string* err) {
        out = MemAlloc{};
        if (dev_ == VK_NULL_HANDLE || vk_ == nullptr || reqs.size == 0U) {
            set_err(err, "invalid Vulkan memory allocation request");
            return false;
        }
        const std::uint32_t ti = find_type(reqs.memoryTypeBits, req_flags, pref_flags);
        if (ti == UINT32_MAX) {
            set_err(err, "no compatible Vulkan memory type");
            return false;
        }
        const VkMemoryPropertyFlags flags = props_.memoryTypes[ti].propertyFlags;
        const bool host_vis = (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0U;
        const bool coherent = (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0U;
        VkDeviceSize align = std::max<VkDeviceSize>(1U, reqs.alignment);
        VkDeviceSize size = reqs.size;
        if (host_vis && !coherent) {
            align = std::max(align, atom_);
            size = align_up(size, atom_);
        }

        const bool try_shared = size <= kSharedAllocBlockBytes / 2U;
        if (try_shared) {
            for (std::size_t bi = 0; bi < blocks_.size(); ++bi) {
                Block& b = blocks_[bi];
                if (b.memory == VK_NULL_HANDLE || b.dedicated || b.type_index != ti || b.is_image != is_image) {
                    continue;
                }
                if (carve(bi, b, size, align, out)) return true;
            }
        }

        bool dedicated = !try_shared;
        VkDeviceSize bsize = dedicated ? size : kSharedAllocBlockBytes;
        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = bsize;
        ai.memoryTypeIndex = ti;
        VkDeviceMemory mem = VK_NULL_HANDLE;
        VkResult r = vk_->allocate_memory(dev_, &ai, nullptr, &mem);
        if ((r != VK_SUCCESS || mem == VK_NULL_HANDLE) && !dedicated) {
            dedicated = true;
            bsize = size;
            ai.allocationSize = size;
            r = vk_->allocate_memory(dev_, &ai, nullptr, &mem);
        }
        if (r != VK_SUCCESS || mem == VK_NULL_HANDLE) {
            set_err(err, "vkAllocateMemory failed: " + std::to_string(static_cast<int>(r)));
            return false;
        }
        void* mapped = nullptr;
        if (host_vis) {
            const VkResult mr = vk_->map_memory(dev_, mem, 0U, VK_WHOLE_SIZE, 0U, &mapped);
            if (mr != VK_SUCCESS || mapped == nullptr) {
                vk_->free_memory(dev_, mem, nullptr);
                set_err(err, "vkMapMemory failed: " + std::to_string(static_cast<int>(mr)));
                return false;
            }
        }

        Block blk{};
        blk.memory = mem;
        blk.size = bsize;
        blk.type_index = ti;
        blk.is_image = is_image;
        blk.host_vis = host_vis;
        blk.coherent = coherent;
        blk.dedicated = dedicated;
        blk.mapped = mapped;
        if (!dedicated && bsize > size) blk.free.push_back({size, bsize - size});

        std::size_t bi = 0;
        if (!free_slots_.empty()) {
            bi = free_slots_.back();
            free_slots_.pop_back();
            blocks_[bi] = std::move(blk);
        } else {
            bi = blocks_.size();
            blocks_.push_back(std::move(blk));
        }

        out = MemAlloc{bi, mem, 0U, size, mapped, coherent};
        return true;
    }

    void free(MemAlloc& a) noexcept {
        if (!a.valid() || dev_ == VK_NULL_HANDLE || a.block >= blocks_.size()) {
            a = MemAlloc{};
            return;
        }
        Block& b = blocks_[a.block];
        if (b.memory != a.memory) {
            a = MemAlloc{};
            return;
        }
        if (b.dedicated) {
            free_block(a.block);
            a = MemAlloc{};
            return;
        }
        b.free.push_back({a.offset, a.size});
        std::sort(b.free.begin(), b.free.end(), [](const Range& x, const Range& y) { return x.off < y.off; });
        std::size_t w = 0;
        for (std::size_t i = 0; i < b.free.size(); ++i) {
            if (w > 0 && b.free[w - 1].off + b.free[w - 1].len == b.free[i].off) {
                b.free[w - 1].len += b.free[i].len;
            } else {
                b.free[w++] = b.free[i];
            }
        }
        b.free.resize(w);
        if (b.free.size() == 1U && b.free[0].off == 0U && b.free[0].len == b.size) {
            bool sibling = false;
            for (std::size_t j = 0; j < blocks_.size(); ++j) {
                if (j != a.block && blocks_[j].memory != VK_NULL_HANDLE && !blocks_[j].dedicated &&
                    blocks_[j].type_index == b.type_index && blocks_[j].is_image == b.is_image) {
                    sibling = true;
                    break;
                }
            }
            if (sibling) free_block(a.block);
        }
        a = MemAlloc{};
    }

    void flush(const MemAlloc& a, VkDeviceSize local_off, VkDeviceSize len) const noexcept {
        if (!a.valid() || a.coherent || len == 0U || dev_ == VK_NULL_HANDLE || a.block >= blocks_.size()) return;
        const Block& b = blocks_[a.block];
        const VkDeviceSize start = ((a.offset + local_off) / atom_) * atom_;
        const VkDeviceSize end = std::min(b.size, align_up(a.offset + local_off + len, atom_));
        VkMappedMemoryRange rng{};
        rng.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        rng.memory = b.memory;
        rng.offset = start;
        rng.size = end >= b.size ? VK_WHOLE_SIZE : (end - start);
        (void)vk_->flush_memory(dev_, 1U, &rng);
    }

    void invalidate(const MemAlloc& a, VkDeviceSize local_off, VkDeviceSize len) const noexcept {
        if (!a.valid() || a.coherent || len == 0U || dev_ == VK_NULL_HANDLE || a.block >= blocks_.size()) return;
        const Block& b = blocks_[a.block];
        const VkDeviceSize start = ((a.offset + local_off) / atom_) * atom_;
        const VkDeviceSize end = std::min(b.size, align_up(a.offset + local_off + len, atom_));
        VkMappedMemoryRange rng{};
        rng.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        rng.memory = b.memory;
        rng.offset = start;
        rng.size = end >= b.size ? VK_WHOLE_SIZE : (end - start);
        (void)vk_->invalidate_memory(dev_, 1U, &rng);
    }

    [[nodiscard]] std::uint32_t find_memory_type(std::uint32_t bits, VkMemoryPropertyFlags req,
                                                 VkMemoryPropertyFlags pref) const noexcept {
        return find_type(bits, req, pref);
    }

    void release_all() noexcept {
        if (dev_ != VK_NULL_HANDLE && vk_ != nullptr) {
            for (std::size_t i = 0; i < blocks_.size(); ++i) free_block(i);
        }
        blocks_.clear();
        free_slots_.clear();
    }

private:
    struct Range {
        VkDeviceSize off = 0, len = 0;
    };
    struct Block {
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize size = 0;
        std::uint32_t type_index = 0;
        bool is_image = false, host_vis = false, coherent = false, dedicated = false;
        void* mapped = nullptr;
        std::vector<Range> free;
    };

    [[nodiscard]] std::uint32_t find_type(std::uint32_t bits, VkMemoryPropertyFlags req,
                                          VkMemoryPropertyFlags pref) const noexcept {
        const VkMemoryPropertyFlags both = req | pref;
        for (std::uint32_t i = 0; i < props_.memoryTypeCount; ++i) {
            if ((bits & (1U << i)) != 0U && (props_.memoryTypes[i].propertyFlags & both) == both) return i;
        }
        for (std::uint32_t i = 0; i < props_.memoryTypeCount; ++i) {
            if ((bits & (1U << i)) != 0U && (props_.memoryTypes[i].propertyFlags & req) == req) return i;
        }
        return UINT32_MAX;
    }

    [[nodiscard]] bool carve(std::size_t bi, Block& b, VkDeviceSize sz, VkDeviceSize al, MemAlloc& out) {
        for (std::size_t ri = 0; ri < b.free.size(); ++ri) {
            const Range r = b.free[ri];
            const VkDeviceSize aligned = align_up(r.off, al);
            if (aligned < r.off || aligned > r.off + r.len || sz > (r.off + r.len) - aligned) continue;
            const VkDeviceSize pre = aligned - r.off;
            const VkDeviceSize suf_start = aligned + sz;
            const VkDeviceSize suf = (r.off + r.len) - suf_start;
            b.free.erase(b.free.begin() + static_cast<std::ptrdiff_t>(ri));
            if (pre > 0U) {
                b.free.insert(b.free.begin() + static_cast<std::ptrdiff_t>(ri), Range{r.off, pre});
                ++ri;
            }
            if (suf > 0U) {
                b.free.insert(b.free.begin() + static_cast<std::ptrdiff_t>(ri), Range{suf_start, suf});
            }
            out = MemAlloc{bi, b.memory, aligned, sz,
                           b.mapped != nullptr ? static_cast<std::uint8_t*>(b.mapped) + aligned : nullptr,
                           b.coherent};
            return true;
        }
        return false;
    }

    void free_block(std::size_t bi) noexcept {
        Block& b = blocks_[bi];
        if (b.memory != VK_NULL_HANDLE) {
            if (b.mapped != nullptr && vk_->unmap_memory != nullptr) vk_->unmap_memory(dev_, b.memory);
            if (vk_->free_memory != nullptr) vk_->free_memory(dev_, b.memory, nullptr);
        }
        b = Block{};
        free_slots_.push_back(bi);
    }

    VkDevice dev_ = VK_NULL_HANDLE;
    const VkProcs* vk_ = nullptr;
    VkDeviceSize atom_ = 1U;
    VkPhysicalDeviceMemoryProperties props_{};
    std::vector<Block> blocks_;
    std::vector<std::size_t> free_slots_;
};

struct SamplerKey {
    std::uint32_t addr_u = 1, addr_v = 1, addr_w = 1;
    std::uint32_t mag = 1, min = 1, mip = 0;
    std::uint32_t max_aniso = 1;
    std::uint32_t border = 0;
    std::uint32_t lod_bias_bits = 0;
    bool operator==(const SamplerKey& o) const noexcept = default;
};

struct SamplerKeyHash {
    std::size_t operator()(const SamplerKey& k) const noexcept {
        std::uint64_t h = 14695981039346656037ULL;
        hash_mix(h, k.addr_u); hash_mix(h, k.addr_v); hash_mix(h, k.addr_w);
        hash_mix(h, k.mag); hash_mix(h, k.min); hash_mix(h, k.mip);
        hash_mix(h, k.max_aniso); hash_mix(h, k.border); hash_mix(h, k.lod_bias_bits);
        return static_cast<std::size_t>(h);
    }
};

struct SamplerSetKey {
    std::array<std::uint64_t, kMaxSamplers> views{};
    std::array<std::uint64_t, kMaxSamplers> samplers{};
    bool operator==(const SamplerSetKey& o) const noexcept = default;
};

struct SamplerSetKeyHash {
    std::size_t operator()(const SamplerSetKey& k) const noexcept {
        std::uint64_t h = 14695981039346656037ULL;
        for (std::uint32_t i = 0; i < kMaxSamplers; ++i) {
            hash_mix(h, k.views[i]);
            hash_mix(h, k.samplers[i]);
        }
        return static_cast<std::size_t>(h);
    }
};

struct PipelineKey {
    std::uint64_t vs_hash = 0;
    std::uint64_t ps_hash = 0;
    std::uint64_t decl_hash = 0;
    std::array<std::uint32_t, 16> stream_strides{};
    std::uint16_t instance_mask = 0;
    std::uint16_t stream_present_mask = 0;
    VkFormat color_fmt = VK_FORMAT_R8G8B8A8_UNORM;
    VkFormat depth_fmt = VK_FORMAT_UNDEFINED;
    std::uint8_t prim_type = 4;
    std::uint8_t cull_mode = 3;
    std::uint8_t fill_mode = 3;
    std::uint8_t depth_enable = 1;
    std::uint8_t depth_write = 1;
    std::uint8_t depth_func = 4;
    std::uint8_t depth_bias_enable = 0;
    std::uint8_t blend_enable = 0;
    std::uint8_t src_blend = 2;
    std::uint8_t dst_blend = 1;
    std::uint8_t blend_op = 1;
    std::uint8_t sep_alpha = 0;
    std::uint8_t src_alpha = 2;
    std::uint8_t dst_alpha = 1;
    std::uint8_t op_alpha = 1;
    std::uint8_t color_mask = 0xF;
    bool operator==(const PipelineKey& o) const noexcept = default;
};

struct PipelineKeyHash {
    std::size_t operator()(const PipelineKey& k) const noexcept {
        std::uint64_t h = 14695981039346656037ULL;
        hash_mix(h, k.vs_hash);
        hash_mix(h, k.ps_hash);
        hash_mix(h, k.decl_hash);
        for (std::uint32_t s : k.stream_strides) hash_mix(h, s);
        hash_mix(h, (static_cast<std::uint64_t>(k.instance_mask) << 16) | k.stream_present_mask);
        hash_mix(h, static_cast<std::uint64_t>(k.color_fmt));
        hash_mix(h, static_cast<std::uint64_t>(k.depth_fmt));
        const std::uint64_t packed =
            static_cast<std::uint64_t>(k.prim_type) | (static_cast<std::uint64_t>(k.cull_mode) << 4) |
            (static_cast<std::uint64_t>(k.fill_mode) << 8) | (static_cast<std::uint64_t>(k.depth_enable) << 12) |
            (static_cast<std::uint64_t>(k.depth_write) << 13) | (static_cast<std::uint64_t>(k.depth_func) << 14) |
            (static_cast<std::uint64_t>(k.depth_bias_enable) << 18) | (static_cast<std::uint64_t>(k.blend_enable) << 19) |
            (static_cast<std::uint64_t>(k.src_blend) << 20) | (static_cast<std::uint64_t>(k.dst_blend) << 24) |
            (static_cast<std::uint64_t>(k.blend_op) << 28) | (static_cast<std::uint64_t>(k.sep_alpha) << 32) |
            (static_cast<std::uint64_t>(k.src_alpha) << 33) | (static_cast<std::uint64_t>(k.dst_alpha) << 37) |
            (static_cast<std::uint64_t>(k.op_alpha) << 41) | (static_cast<std::uint64_t>(k.color_mask) << 45);
        hash_mix(h, packed);
        return static_cast<std::size_t>(h);
    }
};

struct RenderPassKey {
    VkFormat color_fmt = VK_FORMAT_R8G8B8A8_UNORM;
    VkFormat depth_fmt = VK_FORMAT_UNDEFINED;
    bool clear_color = false;
    bool clear_depth = false;
    bool operator==(const RenderPassKey& o) const noexcept = default;
};

struct RenderPassKeyHash {
    std::size_t operator()(const RenderPassKey& k) const noexcept {
        std::uint64_t h = 14695981039346656037ULL;
        hash_mix(h, static_cast<std::uint64_t>(k.color_fmt));
        hash_mix(h, static_cast<std::uint64_t>(k.depth_fmt));
        hash_mix(h, (k.clear_color ? 1U : 0U) | (k.clear_depth ? 2U : 0U));
        return static_cast<std::size_t>(h);
    }
};

}  // namespace

struct VulkanD3d9Backend::Impl {
    std::shared_ptr<VulkanDriverLoader> loader;
    VulkanD3d9Config config;
    VulkanRawHandles h{};
    VkProcs vk{};
    D3d9MemAllocator alloc{};
    D3d9DriverReport report{};
    std::string last_error;
    std::shared_ptr<wse::asset::SpirvCompilerBackend> compiler;

    bool initialized = false;
    bool gpu_ready = false;

    VkPhysicalDeviceProperties phys_props{};
    VkPhysicalDeviceFeatures phys_features{};
    VkDeviceSize min_ubo_align = 256U;
    VkDeviceSize ubo_stride = 4864U;
    std::uint32_t sampler_binding_count = kMaxSamplers;
    VkFormat default_depth_fmt = VK_FORMAT_D24_UNORM_S8_UINT;

    VkCommandBuffer upload_cmd = VK_NULL_HANDLE;
    VkCommandBuffer frame_cmd = VK_NULL_HANDLE;
    VkFence upload_fence = VK_NULL_HANDLE;
    VkFence frame_fence = VK_NULL_HANDLE;
    bool frame_recording = false;
    bool render_pass_active = false;
    bool frame_touched_rt = false;
    bool pending_clear_color = false;
    bool pending_clear_depth = false;
    std::array<float, 4> clear_color_val{0.0F, 0.0F, 0.0F, 1.0F};
    float clear_depth_val = 1.0F;
    std::uint32_t clear_stencil_val = 0U;

    VkPipelineCache pipeline_cache = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl_ubo = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl_samplers = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;

    VkDescriptorPool ubo_desc_pool = VK_NULL_HANDLE;
    VkDescriptorSet ubo_desc_set = VK_NULL_HANDLE;
    std::vector<VkDescriptorPool> sampler_desc_pools;
    std::unordered_map<SamplerSetKey, VkDescriptorSet, SamplerSetKeyHash> sampler_set_cache;

    // Growable per-frame UBO ring buffer.
    struct RingBuf {
        VkBuffer buf = VK_NULL_HANDLE;
        MemAlloc mem{};
        VkDeviceSize cap = 0;
        VkDeviceSize cursor = 0;
    };
    RingBuf ubo_ring{};
    RingBuf dyn_geom_ring{};
    struct RetiredBuf {
        VkBuffer buf = VK_NULL_HANDLE;
        MemAlloc mem{};
    };
    std::vector<RetiredBuf> retired_bufs;

    // Frames in flight. `frame_cmd`, `frame_fence`, `up_cmd`, `ubo_ring`, `dyn_geom_ring`,
    // `stage_ring`, `ubo_desc_set` and `retired_bufs` always hold the current slot's; the other
    // slots are parked in `slots` and swapped in by `advance_frame_slot()`.
    struct FrameSlot {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkCommandBuffer up_cmd = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        bool in_flight = false;
        std::uint64_t fence_frame = 0;
        RingBuf ubo{};
        RingBuf geom{};
        RingBuf stage{};
        VkDescriptorSet ubo_set = VK_NULL_HANDLE;
        std::vector<RetiredBuf> retired;
#ifdef __ANDROID__
        VkSemaphore sc_acquire = VK_NULL_HANDLE;
        VkSemaphore sc_done = VK_NULL_HANDLE;
#endif
    };
    std::array<FrameSlot, kFramesInFlight> slots{};
    std::size_t cur_slot = 0;
    VkCommandBuffer up_cmd = VK_NULL_HANDLE;  // uploads submitted just before this frame's draws
    bool up_recording = false;
    RingBuf stage_ring{};
    bool frame_in_flight = false;      // `frame_fence` is pending
    std::uint64_t fence_frame = 0;     // the frame `frame_fence` covers
    std::uint64_t completed_frame = 0; // every frame up to this one has finished on the GPU

    // Staging upload & readback buffers.
    VkBuffer staging_buf = VK_NULL_HANDLE;
    MemAlloc staging_mem{};
    VkDeviceSize staging_cap = 0;
    VkBuffer readback_buf = VK_NULL_HANDLE;
    MemAlloc readback_mem{};
    VkDeviceSize readback_cap = 0;

    // 16-byte dummy vertex buffer `{0, 0, 0, 1}` for missing attributes.
    VkBuffer dummy_vb = VK_NULL_HANDLE;
    MemAlloc dummy_vb_mem{};

    // 1x1 dummy textures (2D white, Cube white, 3D white) + default sampler.
    struct DummyTex {
        VkImage img = VK_NULL_HANDLE;
        MemAlloc mem{};
        VkImageView view = VK_NULL_HANDLE;
    };
    DummyTex dummy_2d{};
    DummyTex dummy_cube{};
    DummyTex dummy_3d{};
    VkSampler default_sampler = VK_NULL_HANDLE;

    // Backbuffer + auto depth-stencil.
    std::uint32_t bb_width = 0, bb_height = 0;
    VkImage bb_img = VK_NULL_HANDLE;
    MemAlloc bb_mem{};
    VkImageView bb_view = VK_NULL_HANDLE;
    VkImageLayout bb_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImage depth_img = VK_NULL_HANDLE;
    MemAlloc depth_mem{};
    VkImageView depth_view = VK_NULL_HANDLE;
    VkImageLayout depth_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkFramebuffer bb_fb = VK_NULL_HANDLE;

    struct CompiledShader {
        std::vector<std::uint8_t> bytecode;
        bool is_pixel = false;
        bool attempted = false;
        VkShaderModule mod = VK_NULL_HANDLE;
        d3d9sh::Shader info;
        bool valid = false;
    };
    std::unordered_map<std::uint64_t, CompiledShader> shader_by_hash;
    std::unordered_map<std::uint64_t, std::uint64_t> shader_id_to_hash;

    struct BufferObj {
        bool is_index = false;
        bool is_index32 = false;
        bool dynamic = false;
        std::uint32_t size = 0;
        std::vector<std::uint8_t> shadow;
        bool dirty = false;
        // Static buffer storage:
        VkBuffer vk_buf = VK_NULL_HANDLE;
        MemAlloc vk_mem{};
        // Changed bytes not yet copied to `vk_buf` (static buffers).
        std::size_t dirty_lo = SIZE_MAX, dirty_hi = 0;
        std::uint64_t last_used_frame = 0;  // the frame whose draws last read `vk_buf`
        // Dynamic buffers: persistent host-visible copies, renamed like a D3D9 driver does. A
        // NOOVERWRITE lock writes the current copy in place; a DISCARD (or a plain lock of a copy
        // the GPU may still read) moves to a free copy.
        struct DynInst {
            VkBuffer buf = VK_NULL_HANDLE;
            MemAlloc mem{};
            std::uint64_t last_used = 0;  // frame of the last draw that read it (0: never)
        };
        std::vector<DynInst> insts;
        int cur = -1;
    };
    std::unordered_map<std::uint64_t, BufferObj> buffers;

    struct TextureObj {
        int kind = 0;  // 0 2D, 1 Cube, 2 Volume
        std::uint32_t width = 1, height = 1, depth = 1, levels = 1, d3d_fmt = 21, usage = 0;
        VkFormat vk_fmt = VK_FORMAT_R8G8B8A8_UNORM;
        bool is_bc = false;
        VkImage img = VK_NULL_HANDLE;
        MemAlloc mem{};
        VkImageView view = VK_NULL_HANDLE;
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
        std::uint64_t last_used_frame = 0;  // the frame whose draws last sampled it
    };
    std::unordered_map<std::uint64_t, TextureObj> textures;

    struct VertexDeclObj {
        std::vector<D3d9VertexElement> elems;
        std::uint64_t hash = 0;
    };
    std::unordered_map<std::uint64_t, VertexDeclObj> decls;

    std::unordered_map<RenderPassKey, VkRenderPass, RenderPassKeyHash> render_passes;
    std::unordered_map<SamplerKey, VkSampler, SamplerKeyHash> sampler_cache;
    std::unordered_map<PipelineKey, VkPipeline, PipelineKeyHash> pipeline_cache_map;

    std::uint64_t frame_index = 1;
    VkPipeline bound_pipeline = VK_NULL_HANDLE;
    VkDescriptorSet bound_sampler_set = VK_NULL_HANDLE;

    mutable std::mutex api_mutex;

    // Frame timing / FPS telemetry for on_frame_presented callback.
    bool fps_started = false;
    std::chrono::steady_clock::time_point fps_first_tp{};
    std::chrono::steady_clock::time_point fps_win_tp{};
    std::uint64_t presented_frames = 0;
    std::uint64_t fps_win_frames = 0;
    double last_inst_fps = 0.0;

#ifdef __ANDROID__
    enum class AndroidPresentPath { none, swapchain, ahb_surface_control, native_window_cpu };
    AndroidPresentPath android_path = AndroidPresentPath::none;
    ANativeWindow* active_window = nullptr;

    // Path A: VK_KHR_android_surface + VK_KHR_swapchain (System Vulkan).
    VkSurfaceKHR vk_surface = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkFormat sc_format = VK_FORMAT_R8G8B8A8_UNORM;
    VkExtent2D sc_extent{0U, 0U};
    std::vector<VkImage> sc_images;
    bool sc_dirty = false;

    // Path B: Zero-copy AHardwareBuffer + ASurfaceControl (PanVK / Turnip).
    static constexpr std::size_t kAhbRingSize = 3U;
    struct AhbSlot {
        AHardwareBuffer* ahb = nullptr;
        VkImage img = VK_NULL_HANDLE;
        VkDeviceMemory mem = VK_NULL_HANDLE;
        VkFormat fmt = VK_FORMAT_R8G8B8A8_UNORM;
    };
    std::array<AhbSlot, kAhbRingSize> ahb_slots{};
    std::size_t ahb_slot_idx = 0U;
    std::uint32_t ahb_width = 0U;
    std::uint32_t ahb_height = 0U;
    ASurfaceControl* ahb_surface_control = nullptr;

    // Path C: Universal CPU readback + ANativeWindow_lock fallback.
    std::uint32_t cpu_win_w = 0U;
    std::uint32_t cpu_win_h = 0U;

    static void compute_letterbox_rect(std::int32_t src_w, std::int32_t src_h,
                                       std::int32_t dst_w, std::int32_t dst_h,
                                       std::int32_t& out_x, std::int32_t& out_y,
                                       std::int32_t& out_w, std::int32_t& out_h) noexcept {
        if (src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) {
            out_x = 0;
            out_y = 0;
            out_w = std::max<std::int32_t>(1, dst_w);
            out_h = std::max<std::int32_t>(1, dst_h);
            return;
        }
        const std::int64_t lhs = static_cast<std::int64_t>(dst_w) * src_h;
        const std::int64_t rhs = static_cast<std::int64_t>(dst_h) * src_w;
        if (lhs > rhs) {
            out_h = dst_h;
            out_w = static_cast<std::int32_t>(std::max<std::int64_t>(1, rhs / src_h));
            out_x = (dst_w - out_w) / 2;
            out_y = 0;
        } else {
            out_w = dst_w;
            out_h = static_cast<std::int32_t>(std::max<std::int64_t>(1, lhs / src_w));
            out_x = 0;
            out_y = (dst_h - out_h) / 2;
        }
    }

    void destroy_swapchain_only_locked() noexcept {
        if (swapchain != VK_NULL_HANDLE && h.device != VK_NULL_HANDLE && vk.destroy_swapchain != nullptr) {
            vk.destroy_swapchain(h.device, swapchain, nullptr);
        }
        swapchain = VK_NULL_HANDLE;
        sc_images.clear();
        sc_extent = {0U, 0U};
        sc_dirty = false;
    }

    void destroy_ahb_ring_locked() noexcept {
        // Queued frames may still write these images, and pending entries name their slots.
        if (gpu_ready && h.device != VK_NULL_HANDLE && vk.device_wait_idle != nullptr) (void)vk.device_wait_idle(h.device);
        ahb_pending_count = 0;
        if (ahb_surface_control != nullptr) {
            if (ASurfaceTransaction* tx = ASurfaceTransaction_create()) {
                ASurfaceTransaction_setVisibility(tx, ahb_surface_control, ASURFACE_TRANSACTION_VISIBILITY_HIDE);
                ASurfaceTransaction_reparent(tx, ahb_surface_control, nullptr);
                ASurfaceTransaction_apply(tx);
                ASurfaceTransaction_delete(tx);
            }
            ASurfaceControl_release(ahb_surface_control);
            ahb_surface_control = nullptr;
        }
        for (AhbSlot& s : ahb_slots) {
            if (s.img != VK_NULL_HANDLE && h.device != VK_NULL_HANDLE && vk.destroy_image != nullptr) {
                vk.destroy_image(h.device, s.img, nullptr);
            }
            if (s.mem != VK_NULL_HANDLE && h.device != VK_NULL_HANDLE && vk.free_memory != nullptr) {
                vk.free_memory(h.device, s.mem, nullptr);
            }
            if (s.ahb != nullptr) {
                AHardwareBuffer_release(s.ahb);
            }
            s = AhbSlot{};
        }
        ahb_slot_idx = 0U;
        ahb_width = 0U;
        ahb_height = 0U;
    }

    void teardown_android_presentation_locked() noexcept {
        if (gpu_ready && h.device != VK_NULL_HANDLE && vk.device_wait_idle != nullptr) {
            (void)vk.device_wait_idle(h.device);
        }
        destroy_swapchain_only_locked();
        for (FrameSlot& s : slots) {
            for (VkSemaphore* sem : {&s.sc_acquire, &s.sc_done}) {
                if (*sem != VK_NULL_HANDLE && h.device != VK_NULL_HANDLE && vk.destroy_semaphore != nullptr) {
                    vk.destroy_semaphore(h.device, *sem, nullptr);
                }
                *sem = VK_NULL_HANDLE;
            }
        }
        if (vk_surface != VK_NULL_HANDLE && h.instance != VK_NULL_HANDLE && vk.destroy_surface != nullptr) {
            vk.destroy_surface(h.instance, vk_surface, nullptr);
        }
        vk_surface = VK_NULL_HANDLE;
        destroy_ahb_ring_locked();
        cpu_win_w = 0U;
        cpu_win_h = 0U;
        android_path = AndroidPresentPath::none;
    }

    bool create_swapchain_for_surface_locked() {
        if (vk_surface == VK_NULL_HANDLE || vk.get_phys_surface_caps == nullptr ||
            vk.get_phys_surface_formats == nullptr || vk.create_swapchain == nullptr ||
            vk.get_swapchain_images == nullptr) {
            return false;
        }
        VkSurfaceCapabilitiesKHR caps{};
        if (vk.get_phys_surface_caps(h.physical_device, vk_surface, &caps) != VK_SUCCESS) {
            return false;
        }
        if ((caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT) == 0U) {
            report.missing_features.push_back("swapchain lacks TRANSFER_DST_BIT");
            return false;
        }

        std::uint32_t fmt_cnt = 0U;
        if (vk.get_phys_surface_formats(h.physical_device, vk_surface, &fmt_cnt, nullptr) != VK_SUCCESS ||
            fmt_cnt == 0U) {
            return false;
        }
        std::vector<VkSurfaceFormatKHR> fmts(fmt_cnt);
        if (vk.get_phys_surface_formats(h.physical_device, vk_surface, &fmt_cnt, fmts.data()) != VK_SUCCESS ||
            fmt_cnt == 0U) {
            return false;
        }
        fmts.resize(fmt_cnt);
        VkSurfaceFormatKHR chosen_fmt = fmts[0];
        for (const VkSurfaceFormatKHR& f : fmts) {
            if (f.format == VK_FORMAT_R8G8B8A8_UNORM && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                chosen_fmt = f;
                break;
            }
            if (f.format == VK_FORMAT_B8G8R8A8_UNORM && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                chosen_fmt = f;
            }
        }

        VkExtent2D extent = caps.currentExtent;
        if (extent.width == UINT32_MAX || extent.height == UINT32_MAX || extent.width == 0U || extent.height == 0U) {
            const std::uint32_t ww = active_window != nullptr
                                         ? static_cast<std::uint32_t>(std::max<std::int32_t>(1, ANativeWindow_getWidth(active_window)))
                                         : bb_width;
            const std::uint32_t wh = active_window != nullptr
                                         ? static_cast<std::uint32_t>(std::max<std::int32_t>(1, ANativeWindow_getHeight(active_window)))
                                         : bb_height;
            extent.width = std::clamp(ww, caps.minImageExtent.width, caps.maxImageExtent.width);
            extent.height = std::clamp(wh, caps.minImageExtent.height, caps.maxImageExtent.height);
        }
        if (extent.width == 0U || extent.height == 0U) return false;

        std::uint32_t img_count = std::max<std::uint32_t>(caps.minImageCount, 3U);
        if (caps.maxImageCount > 0U && img_count > caps.maxImageCount) {
            img_count = caps.maxImageCount;
        }

        VkSurfaceTransformFlagBitsKHR pre_transform = caps.currentTransform;
        if ((caps.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR) != 0U) {
            pre_transform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
        }

        VkCompositeAlphaFlagBitsKHR comp_alpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        if ((caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR) != 0U) {
            comp_alpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        } else if ((caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR) != 0U) {
            comp_alpha = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
        } else if ((caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR) != 0U) {
            comp_alpha = VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR;
        } else if ((caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR) != 0U) {
            comp_alpha = VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR;
        }

        VkSwapchainKHR old_sc = swapchain;
        VkSwapchainCreateInfoKHR sci{};
        sci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        sci.surface = vk_surface;
        sci.minImageCount = img_count;
        sci.imageFormat = chosen_fmt.format;
        sci.imageColorSpace = chosen_fmt.colorSpace;
        sci.imageExtent = extent;
        sci.imageArrayLayers = 1U;
        sci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        sci.preTransform = pre_transform;
        sci.compositeAlpha = comp_alpha;
        sci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
        sci.clipped = VK_TRUE;
        sci.oldSwapchain = old_sc;

        VkSwapchainKHR next_sc = VK_NULL_HANDLE;
        const VkResult r = vk.create_swapchain(h.device, &sci, nullptr, &next_sc);
        if (old_sc != VK_NULL_HANDLE && vk.destroy_swapchain != nullptr) {
            vk.destroy_swapchain(h.device, old_sc, nullptr);
        }
        swapchain = VK_NULL_HANDLE;
        sc_images.clear();
        if (r != VK_SUCCESS || next_sc == VK_NULL_HANDLE) {
            report.missing_features.push_back("vkCreateSwapchainKHR failed (" + std::to_string(static_cast<int>(r)) + ")");
            return false;
        }

        std::uint32_t sc_cnt = 0U;
        if (vk.get_swapchain_images(h.device, next_sc, &sc_cnt, nullptr) != VK_SUCCESS || sc_cnt == 0U) {
            vk.destroy_swapchain(h.device, next_sc, nullptr);
            return false;
        }
        sc_images.resize(sc_cnt);
        if (vk.get_swapchain_images(h.device, next_sc, &sc_cnt, sc_images.data()) != VK_SUCCESS || sc_cnt == 0U) {
            sc_images.clear();
            vk.destroy_swapchain(h.device, next_sc, nullptr);
            return false;
        }
        sc_images.resize(sc_cnt);
        swapchain = next_sc;
        sc_format = chosen_fmt.format;
        sc_extent = extent;
        sc_dirty = false;
        return true;
    }

    bool setup_swapchain_locked() {
        if (active_window == nullptr || vk.create_android_surface == nullptr || vk.destroy_surface == nullptr ||
            vk.get_phys_surface_support == nullptr || vk.create_semaphore == nullptr ||
            vk.destroy_semaphore == nullptr || vk.acquire_next_image == nullptr || vk.queue_present == nullptr) {
            report.missing_features.push_back("missing Android WSI/swapchain procs");
            return false;
        }
        VkAndroidSurfaceCreateInfoKHR asci{};
        asci.sType = VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR;
        asci.window = active_window;
        const VkResult sr = vk.create_android_surface(h.instance, &asci, nullptr, &vk_surface);
        if (sr != VK_SUCCESS || vk_surface == VK_NULL_HANDLE) {
            vk_surface = VK_NULL_HANDLE;
            report.missing_features.push_back("vkCreateAndroidSurfaceKHR failed (" +
                                              std::to_string(static_cast<int>(sr)) + ")");
            return false;
        }
        VkBool32 supported = VK_FALSE;
        if (vk.get_phys_surface_support(h.physical_device, h.graphics_queue_family, vk_surface, &supported) !=
                VK_SUCCESS ||
            supported != VK_TRUE) {
            teardown_android_presentation_locked();
            report.missing_features.push_back("graphics queue lacks surface present support");
            return false;
        }
        VkSemaphoreCreateInfo sem_ci{};
        sem_ci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        for (FrameSlot& s : slots) {
            if (vk.create_semaphore(h.device, &sem_ci, nullptr, &s.sc_acquire) != VK_SUCCESS ||
                vk.create_semaphore(h.device, &sem_ci, nullptr, &s.sc_done) != VK_SUCCESS) {
                teardown_android_presentation_locked();
                return false;
            }
        }
        if (!create_swapchain_for_surface_locked()) {
            teardown_android_presentation_locked();
            return false;
        }
        return true;
    }

    bool setup_ahb_surface_control_locked() {
        if (active_window == nullptr || !h.ahb_external_memory || vk.get_ahb_props == nullptr ||
            bb_width == 0U || bb_height == 0U) {
            return false;
        }
        destroy_ahb_ring_locked();

        ahb_surface_control = ASurfaceControl_createFromWindow(active_window, "WSE2-R5");
        if (ahb_surface_control == nullptr) {
            report.missing_features.push_back("ASurfaceControl_createFromWindow failed -> ANativeWindow_lock");
            return false;
        }

        for (std::size_t i = 0; i < kAhbRingSize; ++i) {
            AhbSlot& slot = ahb_slots[i];
            AHardwareBuffer_Desc desc{};
            desc.width = bb_width;
            desc.height = bb_height;
            desc.layers = 1U;
            desc.format = AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM;
            desc.usage = AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE |
                         AHARDWAREBUFFER_USAGE_GPU_FRAMEBUFFER |
                         AHARDWAREBUFFER_USAGE_COMPOSER_OVERLAY;
            if (AHardwareBuffer_allocate(&desc, &slot.ahb) != 0 || slot.ahb == nullptr) {
                desc.usage = AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE | AHARDWAREBUFFER_USAGE_GPU_FRAMEBUFFER;
                if (AHardwareBuffer_allocate(&desc, &slot.ahb) != 0 || slot.ahb == nullptr) {
                    report.missing_features.push_back("AHardwareBuffer_allocate failed -> ANativeWindow_lock");
                    destroy_ahb_ring_locked();
                    return false;
                }
            }

            VkAndroidHardwareBufferFormatPropertiesANDROID fmt_props{};
            fmt_props.sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_FORMAT_PROPERTIES_ANDROID;
            VkAndroidHardwareBufferPropertiesANDROID ahb_props{};
            ahb_props.sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID;
            ahb_props.pNext = &fmt_props;
            const VkResult pr = vk.get_ahb_props(h.device, slot.ahb, &ahb_props);
            if (pr != VK_SUCCESS || ahb_props.allocationSize == 0U || ahb_props.memoryTypeBits == 0U) {
                report.missing_features.push_back("vkGetAndroidHardwareBufferPropertiesANDROID failed (" +
                                                  std::to_string(static_cast<int>(pr)) + ") -> ANativeWindow_lock");
                destroy_ahb_ring_locked();
                return false;
            }

            slot.fmt = (fmt_props.format != VK_FORMAT_UNDEFINED) ? fmt_props.format : VK_FORMAT_R8G8B8A8_UNORM;
            VkExternalMemoryImageCreateInfo ext_img{};
            ext_img.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
            ext_img.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;

            VkImageCreateInfo ici{};
            ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            ici.pNext = &ext_img;
            ici.imageType = VK_IMAGE_TYPE_2D;
            ici.format = slot.fmt;
            ici.extent = {bb_width, bb_height, 1U};
            ici.mipLevels = 1U;
            ici.arrayLayers = 1U;
            ici.samples = VK_SAMPLE_COUNT_1_BIT;
            ici.tiling = VK_IMAGE_TILING_OPTIMAL;
            ici.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            const VkResult ir = vk.create_image(h.device, &ici, nullptr, &slot.img);
            if (ir != VK_SUCCESS || slot.img == VK_NULL_HANDLE) {
                report.missing_features.push_back("vkCreateImage(AHB) failed (" +
                                                  std::to_string(static_cast<int>(ir)) + ") -> ANativeWindow_lock");
                destroy_ahb_ring_locked();
                return false;
            }

            std::uint32_t mem_type = alloc.find_memory_type(ahb_props.memoryTypeBits, 0U,
                                                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (mem_type == UINT32_MAX) {
                for (std::uint32_t bit = 0; bit < 32U; ++bit) {
                    if ((ahb_props.memoryTypeBits & (1U << bit)) != 0U) {
                        mem_type = bit;
                        break;
                    }
                }
            }
            if (mem_type == UINT32_MAX) {
                report.missing_features.push_back("no memoryType for AHB import -> ANativeWindow_lock");
                destroy_ahb_ring_locked();
                return false;
            }

            VkImportAndroidHardwareBufferInfoANDROID import_info{};
            import_info.sType = VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID;
            import_info.buffer = slot.ahb;

            VkMemoryDedicatedAllocateInfo ded_info{};
            ded_info.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
            ded_info.pNext = &import_info;
            ded_info.image = slot.img;

            VkMemoryAllocateInfo mai{};
            mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            mai.pNext = &ded_info;
            mai.allocationSize = ahb_props.allocationSize;
            mai.memoryTypeIndex = mem_type;

            const VkResult ar = vk.allocate_memory(h.device, &mai, nullptr, &slot.mem);
            if (ar != VK_SUCCESS || slot.mem == VK_NULL_HANDLE ||
                vk.bind_image_memory(h.device, slot.img, slot.mem, 0U) != VK_SUCCESS) {
                report.missing_features.push_back("vkAllocateMemory/Bind(AHB) failed (" +
                                                  std::to_string(static_cast<int>(ar)) + ") -> ANativeWindow_lock");
                destroy_ahb_ring_locked();
                return false;
            }
        }
        ahb_width = bb_width;
        ahb_height = bb_height;
        ahb_slot_idx = 0U;
        return true;
    }

    void setup_android_presentation_locked() {
        teardown_android_presentation_locked();
        if (!gpu_ready || active_window == nullptr) return;

        // Tier 1: System Vulkan WSI (VK_KHR_android_surface + VK_KHR_swapchain).
        if (h.probe.presentation_path == PresentationPath::android_surface_wsi &&
            h.probe.swapchain_extension_enabled) {
            if (setup_swapchain_locked()) {
                android_path = AndroidPresentPath::swapchain;
                report.presentation_path = "android_surface_wsi(swapchain)";
                return;
            }
        }

        // Tier 2: Zero-copy AHardwareBuffer + ASurfaceControl (PanVK / Turnip), opt-in: see
        // `VulkanD3d9Config::custom_driver_shared_buffers`.
        if (!config.custom_driver_shared_buffers && h.ahb_external_memory) {
            report.missing_features.push_back("shared AHardwareBuffer frames off for custom drivers -> ANativeWindow_lock");
        } else if (h.ahb_external_memory && vk.get_ahb_props != nullptr) {
            if (setup_ahb_surface_control_locked()) {
                android_path = AndroidPresentPath::ahb_surface_control;
                report.presentation_path = "offscreen_hardware_buffer_bridge(ahb+ASurfaceControl)";
                return;
            }
        } else if (h.probe.presentation_path == PresentationPath::offscreen_hardware_buffer_bridge) {
            report.missing_features.push_back(
                "VK_ANDROID_external_memory_android_hardware_buffer unavailable -> ANativeWindow_lock");
        }

        // Tier 3: Universal CPU readback + ANativeWindow_lock fallback.
        android_path = AndroidPresentPath::native_window_cpu;
        report.presentation_path = "native_window_cpu_fallback(ANativeWindow_lock)";
    }

    bool present_to_android_locked(std::vector<std::uint8_t>* out_rgba8) {
        if (active_window == nullptr || android_path == AndroidPresentPath::none) return false;

        const VkDeviceSize rb_bytes = static_cast<VkDeviceSize>(bb_width) * bb_height * 4U;

        if (android_path == AndroidPresentPath::swapchain) {
            if (sc_dirty || swapchain == VK_NULL_HANDLE) {
                if (vk.device_wait_idle != nullptr) (void)vk.device_wait_idle(h.device);
                if (!create_swapchain_for_surface_locked()) {
                    // Fall back to CPU window upload if swapchain cannot be recreated.
                    teardown_android_presentation_locked();
                    android_path = AndroidPresentPath::native_window_cpu;
                    report.presentation_path = "native_window_cpu_fallback(ANativeWindow_lock)";
                    return present_to_android_locked(out_rgba8);
                }
            }

            std::uint32_t img_idx = 0U;
            // One acquire/render-done semaphore pair per frame slot: with two frames in flight
            // the previous frame's submit may still be waiting on its own pair.
            const VkSemaphore acquire_sem = slots[cur_slot].sc_acquire;
            const VkSemaphore done_sem = slots[cur_slot].sc_done;
            VkResult ar = vk.acquire_next_image(h.device, swapchain, 1'000'000'000ULL,
                                                acquire_sem, VK_NULL_HANDLE, &img_idx);
            if (ar == VK_ERROR_OUT_OF_DATE_KHR) {
                if (vk.device_wait_idle != nullptr) (void)vk.device_wait_idle(h.device);
                if (!create_swapchain_for_surface_locked()) return true;
                ar = vk.acquire_next_image(h.device, swapchain, 1'000'000'000ULL,
                                           acquire_sem, VK_NULL_HANDLE, &img_idx);
            }
            if ((ar != VK_SUCCESS && ar != VK_SUBOPTIMAL_KHR) || img_idx >= sc_images.size()) {
                if (ar == VK_ERROR_SURFACE_LOST_KHR) teardown_android_presentation_locked();
                return true;
            }
            if (ar == VK_SUBOPTIMAL_KHR) sc_dirty = true;

            if (!ensure_frame_recording()) return false;
            if (render_pass_active) {
                vk.cmd_end_render_pass(frame_cmd);
                render_pass_active = false;
            }
            const bool do_rb = (out_rgba8 != nullptr) && ensure_readback(rb_bytes, &last_error);

            std::array<VkImageMemoryBarrier, 2> pre_bars{};
            pre_bars[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            pre_bars[0].oldLayout = bb_layout;
            pre_bars[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            pre_bars[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            pre_bars[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            pre_bars[0].image = bb_img;
            pre_bars[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0U, 1U, 0U, 1U};
            pre_bars[0].srcAccessMask =
                (bb_layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) ? VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT : 0U;
            pre_bars[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

            pre_bars[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            pre_bars[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            pre_bars[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            pre_bars[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            pre_bars[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            pre_bars[1].image = sc_images[img_idx];
            pre_bars[1].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0U, 1U, 0U, 1U};
            pre_bars[1].srcAccessMask = 0U;
            pre_bars[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

            // TRANSFER is in the first scope so the swapchain image's layout change waits for
            // the acquire semaphore (waited at TRANSFER below).
            vk.cmd_pipeline_barrier(frame_cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                    VK_PIPELINE_STAGE_TRANSFER_BIT, 0U, 0U, nullptr, 0U, nullptr,
                                    2U, pre_bars.data());

            if (do_rb) {
                VkBufferImageCopy cpy{};
                cpy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0U, 0U, 1U};
                cpy.imageExtent = {bb_width, bb_height, 1U};
                vk.cmd_copy_image_to_buffer(frame_cmd, bb_img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                            readback_buf, 1U, &cpy);
            }

            std::int32_t dst_x = 0, dst_y = 0, dst_w = 0, dst_h = 0;
            compute_letterbox_rect(static_cast<std::int32_t>(bb_width), static_cast<std::int32_t>(bb_height),
                                   static_cast<std::int32_t>(sc_extent.width), static_cast<std::int32_t>(sc_extent.height),
                                   dst_x, dst_y, dst_w, dst_h);
            const bool letterboxed = (dst_x > 0 || dst_y > 0 ||
                                      static_cast<std::uint32_t>(dst_w) < sc_extent.width ||
                                      static_cast<std::uint32_t>(dst_h) < sc_extent.height);
            if (letterboxed && vk.cmd_clear_color_image != nullptr) {
                VkClearColorValue black{};
                black.float32[0] = 0.0F;
                black.float32[1] = 0.0F;
                black.float32[2] = 0.0F;
                black.float32[3] = 1.0F;
                VkImageSubresourceRange rng{VK_IMAGE_ASPECT_COLOR_BIT, 0U, 1U, 0U, 1U};
                vk.cmd_clear_color_image(frame_cmd, sc_images[img_idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                         &black, 1U, &rng);
                VkImageMemoryBarrier clr_bar{};
                clr_bar.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                clr_bar.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                clr_bar.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                clr_bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                clr_bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                clr_bar.image = sc_images[img_idx];
                clr_bar.subresourceRange = rng;
                clr_bar.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                clr_bar.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                vk.cmd_pipeline_barrier(frame_cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                        0U, 0U, nullptr, 0U, nullptr, 1U, &clr_bar);
            }

            if (!letterboxed && static_cast<std::uint32_t>(dst_w) == bb_width &&
                static_cast<std::uint32_t>(dst_h) == bb_height && sc_format == VK_FORMAT_R8G8B8A8_UNORM &&
                vk.cmd_copy_image != nullptr) {
                VkImageCopy ic{};
                ic.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0U, 0U, 1U};
                ic.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0U, 0U, 1U};
                ic.extent = {bb_width, bb_height, 1U};
                vk.cmd_copy_image(frame_cmd, bb_img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                  sc_images[img_idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1U, &ic);
            } else if (vk.cmd_blit_image != nullptr) {
                VkImageBlit bl{};
                bl.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0U, 0U, 1U};
                bl.srcOffsets[0] = {0, 0, 0};
                bl.srcOffsets[1] = {static_cast<std::int32_t>(bb_width), static_cast<std::int32_t>(bb_height), 1};
                bl.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0U, 0U, 1U};
                bl.dstOffsets[0] = {dst_x, dst_y, 0};
                bl.dstOffsets[1] = {dst_x + dst_w, dst_y + dst_h, 1};
                vk.cmd_blit_image(frame_cmd, bb_img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                  sc_images[img_idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1U, &bl, VK_FILTER_LINEAR);
            }

            std::array<VkImageMemoryBarrier, 2> post_bars{};
            post_bars[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            post_bars[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            post_bars[0].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            post_bars[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            post_bars[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            post_bars[0].image = bb_img;
            post_bars[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0U, 1U, 0U, 1U};
            post_bars[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            post_bars[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

            post_bars[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            post_bars[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            post_bars[1].newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            post_bars[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            post_bars[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            post_bars[1].image = sc_images[img_idx];
            post_bars[1].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0U, 1U, 0U, 1U};
            post_bars[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            post_bars[1].dstAccessMask = 0U;

            vk.cmd_pipeline_barrier(frame_cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0U, 0U, nullptr, 0U, nullptr,
                                    2U, post_bars.data());
            bb_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

            // No wait here: the next frame records while this one draws (readback alone waits).
            if (!submit_frame_locked(acquire_sem, VK_PIPELINE_STAGE_TRANSFER_BIT, done_sem, do_rb)) return false;

            VkPresentInfoKHR pi{};
            pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
            pi.waitSemaphoreCount = 1U;
            pi.pWaitSemaphores = &done_sem;
            pi.swapchainCount = 1U;
            pi.pSwapchains = &swapchain;
            pi.pImageIndices = &img_idx;
            const VkResult pr = vk.queue_present(h.graphics_queue, &pi);
            if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR) sc_dirty = true;

            if (do_rb) {
                alloc.invalidate(readback_mem, 0U, rb_bytes);
                out_rgba8->resize(static_cast<std::size_t>(rb_bytes));
                std::memcpy(out_rgba8->data(), readback_mem.mapped, static_cast<std::size_t>(rb_bytes));
            }
            return true;
        }

        if (android_path == AndroidPresentPath::ahb_surface_control) {
            if (ahb_width != bb_width || ahb_height != bb_height) {
                if (!setup_ahb_surface_control_locked()) {
                    android_path = AndroidPresentPath::native_window_cpu;
                    report.presentation_path = "native_window_cpu_fallback(ANativeWindow_lock)";
                    return present_to_android_locked(out_rgba8);
                }
            }

            AhbSlot& slot = ahb_slots[ahb_slot_idx];
            ahb_slot_idx = (ahb_slot_idx + 1U) % kAhbRingSize;
            if (!ensure_frame_recording()) return false;
            if (render_pass_active) {
                vk.cmd_end_render_pass(frame_cmd);
                render_pass_active = false;
            }
            const bool do_rb = (out_rgba8 != nullptr) && ensure_readback(rb_bytes, &last_error);

            std::array<VkImageMemoryBarrier, 2> pre_bars{};
            pre_bars[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            pre_bars[0].oldLayout = bb_layout;
            pre_bars[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            pre_bars[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            pre_bars[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            pre_bars[0].image = bb_img;
            pre_bars[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0U, 1U, 0U, 1U};
            pre_bars[0].srcAccessMask =
                (bb_layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) ? VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT : 0U;
            pre_bars[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

            pre_bars[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            pre_bars[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            pre_bars[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            pre_bars[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            pre_bars[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            pre_bars[1].image = slot.img;
            pre_bars[1].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0U, 1U, 0U, 1U};
            pre_bars[1].srcAccessMask = 0U;
            pre_bars[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

            vk.cmd_pipeline_barrier(frame_cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                    VK_PIPELINE_STAGE_TRANSFER_BIT, 0U, 0U, nullptr, 0U, nullptr,
                                    2U, pre_bars.data());

            if (do_rb) {
                VkBufferImageCopy cpy{};
                cpy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0U, 0U, 1U};
                cpy.imageExtent = {bb_width, bb_height, 1U};
                vk.cmd_copy_image_to_buffer(frame_cmd, bb_img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                            readback_buf, 1U, &cpy);
            }

            if (slot.fmt == VK_FORMAT_R8G8B8A8_UNORM && vk.cmd_copy_image != nullptr) {
                VkImageCopy ic{};
                ic.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0U, 0U, 1U};
                ic.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0U, 0U, 1U};
                ic.extent = {bb_width, bb_height, 1U};
                vk.cmd_copy_image(frame_cmd, bb_img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                  slot.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1U, &ic);
            } else if (vk.cmd_blit_image != nullptr) {
                VkImageBlit bl{};
                bl.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0U, 0U, 1U};
                bl.srcOffsets[0] = {0, 0, 0};
                bl.srcOffsets[1] = {static_cast<std::int32_t>(bb_width), static_cast<std::int32_t>(bb_height), 1};
                bl.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0U, 0U, 1U};
                bl.dstOffsets[0] = {0, 0, 0};
                bl.dstOffsets[1] = {static_cast<std::int32_t>(bb_width), static_cast<std::int32_t>(bb_height), 1};
                vk.cmd_blit_image(frame_cmd, bb_img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                  slot.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1U, &bl, VK_FILTER_NEAREST);
            }

            std::array<VkImageMemoryBarrier, 2> post_bars{};
            post_bars[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            post_bars[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            post_bars[0].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            post_bars[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            post_bars[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            post_bars[0].image = bb_img;
            post_bars[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0U, 1U, 0U, 1U};
            post_bars[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            post_bars[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

            post_bars[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            post_bars[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            post_bars[1].newLayout = VK_IMAGE_LAYOUT_GENERAL;
            post_bars[1].srcQueueFamilyIndex = h.queue_family_foreign ? h.graphics_queue_family : VK_QUEUE_FAMILY_IGNORED;
            post_bars[1].dstQueueFamilyIndex = h.queue_family_foreign ? VK_QUEUE_FAMILY_FOREIGN_EXT : VK_QUEUE_FAMILY_IGNORED;
            post_bars[1].image = slot.img;
            post_bars[1].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0U, 1U, 0U, 1U};
            post_bars[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            post_bars[1].dstAccessMask = 0U;

            vk.cmd_pipeline_barrier(frame_cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                    VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0U, 0U, nullptr, 0U, nullptr,
                                    2U, post_bars.data());
            bb_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

            // The compositor gets no GPU fence, so a buffer is handed over only once its frame has
            // finished: `advance_frame_slot()` waits for the previous frame, then
            // `show_finished_ahb_locked()` shows it (one frame of latency, no stall).
            if (!submit_frame_locked(VK_NULL_HANDLE, 0U, VK_NULL_HANDLE, do_rb)) return false;
            ahb_pending[ahb_pending_count < ahb_pending.size() ? ahb_pending_count++ : ahb_pending.size() - 1U] =
                {&slot - ahb_slots.data(), fence_frame};

            if (do_rb) {
                alloc.invalidate(readback_mem, 0U, rb_bytes);
                out_rgba8->resize(static_cast<std::size_t>(rb_bytes));
                std::memcpy(out_rgba8->data(), readback_mem.mapped, static_cast<std::size_t>(rb_bytes));
            }
            return true;
        }

        // Tier 3: CPU readback + ANativeWindow_lock upload.
        return present_cpu_window_locked(rb_bytes, out_rgba8);
    }

    // AHB slots whose frames were submitted but not yet shown (oldest first).
    struct AhbPending {
        std::ptrdiff_t slot = 0;
        std::uint64_t frame = 0;
    };
    std::array<AhbPending, kFramesInFlight + 1U> ahb_pending{};
    std::size_t ahb_pending_count = 0;

    // Shows every pending AHB slot whose frame has finished on the GPU.
    void show_finished_ahb_locked() {
        std::size_t done = 0;
        while (done < ahb_pending_count && ahb_pending[done].frame <= completed_frame) {
            show_ahb_slot_locked(ahb_slots[static_cast<std::size_t>(ahb_pending[done].slot)]);
            ++done;
        }
        for (std::size_t i = done; i < ahb_pending_count; ++i) ahb_pending[i - done] = ahb_pending[i];
        ahb_pending_count -= done;
    }

    void show_ahb_slot_locked(AhbSlot& slot) {
        {
            if (ahb_surface_control != nullptr) {
                const std::int32_t win_w = std::max<std::int32_t>(1, ANativeWindow_getWidth(active_window));
                const std::int32_t win_h = std::max<std::int32_t>(1, ANativeWindow_getHeight(active_window));
                std::int32_t dst_x = 0, dst_y = 0, dst_w = 0, dst_h = 0;
                compute_letterbox_rect(static_cast<std::int32_t>(bb_width), static_cast<std::int32_t>(bb_height),
                                       win_w, win_h, dst_x, dst_y, dst_w, dst_h);
                const ARect src_rc{0, 0, static_cast<std::int32_t>(bb_width), static_cast<std::int32_t>(bb_height)};
                const ARect dst_rc{dst_x, dst_y, dst_x + dst_w, dst_y + dst_h};
                if (ASurfaceTransaction* tx = ASurfaceTransaction_create()) {
                    ASurfaceTransaction_setVisibility(tx, ahb_surface_control, ASURFACE_TRANSACTION_VISIBILITY_SHOW);
                    ASurfaceTransaction_setZOrder(tx, ahb_surface_control, 0);
                    ASurfaceTransaction_setBufferTransparency(tx, ahb_surface_control,
                                                              ASURFACE_TRANSACTION_TRANSPARENCY_OPAQUE);
                    ASurfaceTransaction_setBuffer(tx, ahb_surface_control, slot.ahb, -1);
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
                    ASurfaceTransaction_setGeometry(tx, ahb_surface_control, src_rc, dst_rc, 0);
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
                    ASurfaceTransaction_apply(tx);
                    ASurfaceTransaction_delete(tx);
                }
            }
        }
    }

    // Tier 3: CPU readback + ANativeWindow_lock upload (waits for the frame: it copies the pixels).
    bool present_cpu_window_locked(VkDeviceSize rb_bytes, std::vector<std::uint8_t>* out_rgba8) {
        if (!ensure_frame_recording()) return false;
        if (render_pass_active) {
            vk.cmd_end_render_pass(frame_cmd);
            render_pass_active = false;
        }
        const bool rb_ok = ensure_readback(rb_bytes, &last_error);
        if (rb_ok) {
            VkImageMemoryBarrier bar{};
            bar.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            bar.oldLayout = bb_layout;
            bar.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bar.image = bb_img;
            bar.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0U, 1U, 0U, 1U};
            bar.srcAccessMask =
                (bb_layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) ? VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT : 0U;
            bar.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vk.cmd_pipeline_barrier(frame_cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                    VK_PIPELINE_STAGE_TRANSFER_BIT, 0U, 0U, nullptr, 0U, nullptr, 1U, &bar);
            VkBufferImageCopy cpy{};
            cpy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0U, 0U, 1U};
            cpy.imageExtent = {bb_width, bb_height, 1U};
            vk.cmd_copy_image_to_buffer(frame_cmd, bb_img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                        readback_buf, 1U, &cpy);
            bar.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            bar.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            bar.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            bar.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            vk.cmd_pipeline_barrier(frame_cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0U, 0U, nullptr, 0U, nullptr, 1U, &bar);
            bb_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        }
        if (!submit_frame_locked(VK_NULL_HANDLE, 0U, VK_NULL_HANDLE, true)) return false;

        if (rb_ok) {
            alloc.invalidate(readback_mem, 0U, rb_bytes);
            if (cpu_win_w != bb_width || cpu_win_h != bb_height) {
                (void)ANativeWindow_setBuffersGeometry(active_window, static_cast<std::int32_t>(bb_width),
                                                       static_cast<std::int32_t>(bb_height), WINDOW_FORMAT_RGBA_8888);
                cpu_win_w = bb_width;
                cpu_win_h = bb_height;
            }
            ANativeWindow_Buffer wb{};
            if (ANativeWindow_lock(active_window, &wb, nullptr) == 0 && wb.bits != nullptr) {
                const std::uint32_t copy_w =
                    std::min<std::uint32_t>(bb_width, static_cast<std::uint32_t>(std::max<std::int32_t>(0, wb.width)));
                const std::uint32_t copy_h =
                    std::min<std::uint32_t>(bb_height, static_cast<std::uint32_t>(std::max<std::int32_t>(0, wb.height)));
                const auto* src = static_cast<const std::uint8_t*>(readback_mem.mapped);
                auto* dst = static_cast<std::uint8_t*>(wb.bits);
                const std::size_t src_stride = static_cast<std::size_t>(bb_width) * 4U;
                const std::size_t dst_stride = static_cast<std::size_t>(std::max<std::int32_t>(0, wb.stride)) * 4U;
                const std::size_t row_bytes = static_cast<std::size_t>(copy_w) * 4U;
                if (row_bytes > 0U && dst_stride >= row_bytes) {
                    for (std::uint32_t y = 0; y < copy_h; ++y) {
                        std::memcpy(dst + y * dst_stride, src + y * src_stride, row_bytes);
                    }
                }
                (void)ANativeWindow_unlockAndPost(active_window);
            }
            if (out_rgba8 != nullptr) {
                out_rgba8->resize(static_cast<std::size_t>(rb_bytes));
                std::memcpy(out_rgba8->data(), readback_mem.mapped, static_cast<std::size_t>(rb_bytes));
            }
        }
        return true;
    }
#endif

    Impl(std::shared_ptr<VulkanDriverLoader> ldr, VulkanD3d9Config cfg)
        : loader(std::move(ldr)), config(std::move(cfg)) {
#ifdef __ANDROID__
        if (config.native_window != nullptr) {
            active_window = static_cast<ANativeWindow*>(config.native_window);
            ANativeWindow_acquire(active_window);
        }
#endif
    }

    ~Impl() {
        const std::lock_guard<std::mutex> lock(api_mutex);
        shutdown();
#ifdef __ANDROID__
        if (active_window != nullptr) {
            ANativeWindow_release(active_window);
            active_window = nullptr;
        }
#endif
    }

    void shutdown() noexcept {
        if (!initialized) return;
        if (gpu_ready && h.device != VK_NULL_HANDLE) {
            if (vk.device_wait_idle != nullptr) (void)vk.device_wait_idle(h.device);
#ifdef __ANDROID__
            teardown_android_presentation_locked();
#endif
            save_pipeline_cache();
            stash_slot();
            for (FrameSlot& s : slots) {
                for (RetiredBuf& rb : s.retired) {
                    if (rb.buf != VK_NULL_HANDLE) vk.destroy_buffer(h.device, rb.buf, nullptr);
                    alloc.free(rb.mem);
                }
                s.retired.clear();
                s.in_flight = false;
            }

            for (auto& [k, pip] : pipeline_cache_map) {
                (void)k;
                if (pip != VK_NULL_HANDLE) vk.destroy_pipeline(h.device, pip, nullptr);
            }
            pipeline_cache_map.clear();

            if (pipeline_cache != VK_NULL_HANDLE && vk.destroy_pipeline_cache != nullptr) {
                vk.destroy_pipeline_cache(h.device, pipeline_cache, nullptr);
                pipeline_cache = VK_NULL_HANDLE;
            }
            if (pipeline_layout != VK_NULL_HANDLE) {
                vk.destroy_pipeline_layout(h.device, pipeline_layout, nullptr);
                pipeline_layout = VK_NULL_HANDLE;
            }
            for (VkDescriptorPool p : sampler_desc_pools) {
                if (p != VK_NULL_HANDLE) vk.destroy_desc_pool(h.device, p, nullptr);
            }
            sampler_desc_pools.clear();
            sampler_set_cache.clear();
            if (ubo_desc_pool != VK_NULL_HANDLE) {
                vk.destroy_desc_pool(h.device, ubo_desc_pool, nullptr);
                ubo_desc_pool = VK_NULL_HANDLE;
                ubo_desc_set = VK_NULL_HANDLE;
            }
            if (dsl_samplers != VK_NULL_HANDLE) {
                vk.destroy_desc_set_layout(h.device, dsl_samplers, nullptr);
                dsl_samplers = VK_NULL_HANDLE;
            }
            if (dsl_ubo != VK_NULL_HANDLE) {
                vk.destroy_desc_set_layout(h.device, dsl_ubo, nullptr);
                dsl_ubo = VK_NULL_HANDLE;
            }

            destroy_backbuffer();
            for (auto& [k, rp] : render_passes) {
                (void)k;
                if (rp != VK_NULL_HANDLE) vk.destroy_render_pass(h.device, rp, nullptr);
            }
            render_passes.clear();

            for (auto& [id, tex] : textures) {
                (void)id;
                destroy_texture_gpu(tex);
            }
            textures.clear();

            destroy_dummy_tex(dummy_2d);
            destroy_dummy_tex(dummy_cube);
            destroy_dummy_tex(dummy_3d);

            for (auto& [k, smp] : sampler_cache) {
                (void)k;
                if (smp != VK_NULL_HANDLE) vk.destroy_sampler(h.device, smp, nullptr);
            }
            sampler_cache.clear();
            default_sampler = VK_NULL_HANDLE;

            for (auto& [id, buf] : buffers) {
                (void)id;
                destroy_buffer_gpu(buf);
            }
            buffers.clear();

            for (FrameSlot& s : slots) {
                destroy_ring(s.ubo);
                destroy_ring(s.geom);
                destroy_ring(s.stage);
                s.ubo_set = VK_NULL_HANDLE;
            }
            if (staging_buf != VK_NULL_HANDLE) {
                vk.destroy_buffer(h.device, staging_buf, nullptr);
                staging_buf = VK_NULL_HANDLE;
            }
            alloc.free(staging_mem);
            staging_cap = 0;
            if (readback_buf != VK_NULL_HANDLE) {
                vk.destroy_buffer(h.device, readback_buf, nullptr);
                readback_buf = VK_NULL_HANDLE;
            }
            alloc.free(readback_mem);
            readback_cap = 0;
            if (dummy_vb != VK_NULL_HANDLE) {
                vk.destroy_buffer(h.device, dummy_vb, nullptr);
                dummy_vb = VK_NULL_HANDLE;
            }
            alloc.free(dummy_vb_mem);

            for (auto& [hash, sh] : shader_by_hash) {
                (void)hash;
                if (sh.mod != VK_NULL_HANDLE) vk.destroy_shader_module(h.device, sh.mod, nullptr);
            }
            shader_by_hash.clear();
            shader_id_to_hash.clear();

            if (upload_fence != VK_NULL_HANDLE) {
                vk.destroy_fence(h.device, upload_fence, nullptr);
                upload_fence = VK_NULL_HANDLE;
            }
            for (FrameSlot& s : slots) {
                if (s.fence != VK_NULL_HANDLE) vk.destroy_fence(h.device, s.fence, nullptr);
                s.fence = VK_NULL_HANDLE;
            }
            if (h.command_pool != VK_NULL_HANDLE) {
                std::vector<VkCommandBuffer> cbs;
                if (upload_cmd != VK_NULL_HANDLE) cbs.push_back(upload_cmd);
                for (FrameSlot& s : slots) {
                    if (s.cmd != VK_NULL_HANDLE) cbs.push_back(s.cmd);
                    if (s.up_cmd != VK_NULL_HANDLE) cbs.push_back(s.up_cmd);
                    s.cmd = s.up_cmd = VK_NULL_HANDLE;
                }
                if (!cbs.empty()) vk.free_cmd_buffers(h.device, h.command_pool, static_cast<std::uint32_t>(cbs.size()), cbs.data());
                upload_cmd = VK_NULL_HANDLE;
            }
            load_slot();  // all null now
            alloc.release_all();
        }
        gpu_ready = false;
        initialized = false;
    }

    void destroy_ring(RingBuf& r) noexcept {
        if (r.buf != VK_NULL_HANDLE) {
            vk.destroy_buffer(h.device, r.buf, nullptr);
            r.buf = VK_NULL_HANDLE;
        }
        alloc.free(r.mem);
        r.cap = 0;
        r.cursor = 0;
    }

    void free_retired() noexcept {
        for (RetiredBuf& rb : retired_bufs) {
            if (rb.buf != VK_NULL_HANDLE) vk.destroy_buffer(h.device, rb.buf, nullptr);
            alloc.free(rb.mem);
        }
        retired_bufs.clear();
    }

    void destroy_buffer_gpu(BufferObj& b) noexcept {
        if (b.vk_buf != VK_NULL_HANDLE) vk.destroy_buffer(h.device, b.vk_buf, nullptr);
        b.vk_buf = VK_NULL_HANDLE;
        alloc.free(b.vk_mem);
        for (BufferObj::DynInst& d : b.insts) {
            if (d.buf != VK_NULL_HANDLE) vk.destroy_buffer(h.device, d.buf, nullptr);
            alloc.free(d.mem);
        }
        b.insts.clear();
        b.cur = -1;
    }

    // --- frames in flight ------------------------------------------------------------------------
    void stash_slot() {
        FrameSlot& s = slots[cur_slot];
        s.cmd = frame_cmd;
        s.up_cmd = up_cmd;
        s.fence = frame_fence;
        s.in_flight = frame_in_flight;
        s.fence_frame = fence_frame;
        s.ubo = ubo_ring;
        s.geom = dyn_geom_ring;
        s.stage = stage_ring;
        s.ubo_set = ubo_desc_set;
        s.retired = std::move(retired_bufs);
        retired_bufs.clear();
    }
    void load_slot() {
        FrameSlot& s = slots[cur_slot];
        frame_cmd = s.cmd;
        up_cmd = s.up_cmd;
        frame_fence = s.fence;
        frame_in_flight = s.in_flight;
        fence_frame = s.fence_frame;
        ubo_ring = s.ubo;
        dyn_geom_ring = s.geom;
        stage_ring = s.stage;
        ubo_desc_set = s.ubo_set;
        retired_bufs = std::move(s.retired);
        s.retired.clear();
    }
    // Waits for the current slot's last submit and frees what it retired.
    void wait_current_slot() {
        if (frame_in_flight) {
            (void)vk.wait_for_fences(h.device, 1U, &frame_fence, VK_TRUE, 10'000'000'000ULL);
            frame_in_flight = false;
            completed_frame = std::max(completed_frame, fence_frame);
        }
        free_retired();
    }
    // Ends and submits the current slot's uploads and frame commands (uploads first), signalling
    // `frame_fence`. `wait`: block until they finish (readback, mid-frame flush).
    bool submit_frame_locked(VkSemaphore wait_sem, VkPipelineStageFlags wait_stage, VkSemaphore signal_sem, bool wait) {
        std::array<VkCommandBuffer, 2> cbs{};
        std::uint32_t n = 0;
        if (up_recording) {
            up_recording = false;
            if (vk.end_cmd_buffer(up_cmd) == VK_SUCCESS) cbs[n++] = up_cmd;
        }
        if (frame_recording) {
            if (render_pass_active) {
                vk.cmd_end_render_pass(frame_cmd);
                render_pass_active = false;
            }
            frame_recording = false;
            if (vk.end_cmd_buffer(frame_cmd) != VK_SUCCESS) return false;
            cbs[n++] = frame_cmd;
        }
        if (n == 0U && wait_sem == VK_NULL_HANDLE && signal_sem == VK_NULL_HANDLE) return true;
        if (frame_in_flight) wait_current_slot();  // a mid-frame flush may have left it pending
        if (vk.reset_fences(h.device, 1U, &frame_fence) != VK_SUCCESS) return false;
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        if (wait_sem != VK_NULL_HANDLE) {
            si.waitSemaphoreCount = 1U;
            si.pWaitSemaphores = &wait_sem;
            si.pWaitDstStageMask = &wait_stage;
        }
        si.commandBufferCount = n;
        si.pCommandBuffers = cbs.data();
        if (signal_sem != VK_NULL_HANDLE) {
            si.signalSemaphoreCount = 1U;
            si.pSignalSemaphores = &signal_sem;
        }
        if (vk.queue_submit(h.graphics_queue, 1U, &si, frame_fence) != VK_SUCCESS) return false;
        frame_in_flight = true;
        fence_frame = frame_index;
        bound_pipeline = VK_NULL_HANDLE;
        bound_sampler_set = VK_NULL_HANDLE;
        if (wait) wait_current_slot();
        return true;
    }
    // After a present: park this slot, take the next one and wait for its previous frame (the
    // frame before this one), so at most `kFramesInFlight` frames are queued.
    void advance_frame_slot() {
        // A present that returned early (no swapchain image) leaves commands recorded: submit them,
        // since recording state is not parked with the slot.
        if (frame_recording || up_recording) (void)submit_frame_locked(VK_NULL_HANDLE, 0U, VK_NULL_HANDLE, false);
        stash_slot();
        cur_slot = (cur_slot + 1U) % kFramesInFlight;
        load_slot();
        wait_current_slot();
        ubo_ring.cursor = 0U;
        dyn_geom_ring.cursor = 0U;
        stage_ring.cursor = 0U;
    }
    // Waits until the GPU has finished everything (before destroying or replacing a resource that
    // a queued frame may still read). Rare: object destruction and device reset.
    void wait_all_locked() {
        flush_frame_if_recording();
        for (std::size_t i = 0; i < kFramesInFlight; ++i) {
            if (i == cur_slot) continue;
            FrameSlot& s = slots[i];
            if (s.in_flight) {
                (void)vk.wait_for_fences(h.device, 1U, &s.fence, VK_TRUE, 10'000'000'000ULL);
                s.in_flight = false;
                completed_frame = std::max(completed_frame, s.fence_frame);
            }
            for (RetiredBuf& rb : s.retired) {
                if (rb.buf != VK_NULL_HANDLE) vk.destroy_buffer(h.device, rb.buf, nullptr);
                alloc.free(rb.mem);
            }
            s.retired.clear();
        }
        wait_current_slot();
    }

    // --- uploads without stalls ----------------------------------------------------------------
    // Copies `bytes` into the current slot's staging ring. Call before `begin_upload_locked`: a
    // full ring first waits for the frame so far.
    bool stage_bytes(std::span<const std::uint8_t> bytes, VkBuffer& out_buf, VkDeviceSize& out_off) {
        const VkDeviceSize aligned = align_up(std::max<VkDeviceSize>(bytes.size(), 16U), 16U);
        if (stage_ring.cursor + aligned > stage_ring.cap) {
            if (stage_ring.cap >= kStagingFlushBytes && stage_ring.cursor > 0U) {
                flush_frame_if_recording();  // the staged copies are done: reuse the ring
                stage_ring.cursor = 0U;
            }
            if (stage_ring.cursor + aligned > stage_ring.cap) {
                const VkDeviceSize cap = std::max(stage_ring.cap * 2U, align_up(aligned + stage_ring.cursor, kInitialStagingBytes));
                RingBuf next{};
                if (!create_host_ring(next, cap, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &last_error)) return false;
                if (stage_ring.buf != VK_NULL_HANDLE) retired_bufs.push_back({stage_ring.buf, stage_ring.mem});
                stage_ring = next;
            }
        }
        out_buf = stage_ring.buf;
        out_off = stage_ring.cursor;
        std::memcpy(static_cast<std::uint8_t*>(stage_ring.mem.mapped) + out_off, bytes.data(), bytes.size());
        alloc.flush(stage_ring.mem, out_off, bytes.size());
        stage_ring.cursor += aligned;
        return true;
    }
    // The command buffer an upload records into: the slot's upload buffer, submitted just before
    // this frame's draws, when no draw of this frame has read the resource yet; else the frame's
    // own commands at this point (outside the render pass), so earlier draws keep the old data.
    VkCommandBuffer begin_upload_locked(bool used_this_frame) {
        if (!ensure_frame_recording()) return VK_NULL_HANDLE;
        if (used_this_frame) {
            if (render_pass_active) {
                vk.cmd_end_render_pass(frame_cmd);
                render_pass_active = false;
                bound_pipeline = VK_NULL_HANDLE;
                bound_sampler_set = VK_NULL_HANDLE;
            }
            return frame_cmd;
        }
        if (!up_recording) {
            if (vk.reset_cmd_buffer(up_cmd, 0U) != VK_SUCCESS) return VK_NULL_HANDLE;
            VkCommandBufferBeginInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            if (vk.begin_cmd_buffer(up_cmd, &bi) != VK_SUCCESS) return VK_NULL_HANDLE;
            up_recording = true;
        }
        return up_cmd;
    }
    // Replacing an existing image: queued frames may still sample it, and cached sampler sets
    // name its view.
    void retire_texture_image_locked(TextureObj& t) {
        if (t.img == VK_NULL_HANDLE) return;
        wait_all_locked();
        sampler_set_cache.clear();
        for (VkDescriptorPool p : sampler_desc_pools) (void)vk.reset_desc_pool(h.device, p, 0U);
        destroy_texture_gpu(t);
    }

    // --- dynamic buffers ---------------------------------------------------------------------
    [[nodiscard]] bool dyn_busy(const BufferObj::DynInst& d) const noexcept {
        return d.last_used != 0U && (d.last_used == frame_index || d.last_used > completed_frame);
    }
    // A copy no queued frame reads (a new one when all are busy). -1 on failure.
    int dyn_acquire(BufferObj& b) {
        for (int pass = 0; pass < 2; ++pass) {
            for (std::size_t i = 0; i < b.insts.size(); ++i) {
                if (static_cast<int>(i) != b.cur && !dyn_busy(b.insts[i])) return static_cast<int>(i);
            }
            if (b.insts.size() < kMaxDynInstances) break;
            wait_all_locked();  // rare: very many discards in flight
        }
        if (b.insts.size() >= kMaxDynInstances) return -1;
        RingBuf r{};
        if (!create_host_ring(r, std::max<VkDeviceSize>(b.size, 16U),
                              VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, &last_error)) {
            return -1;
        }
        b.insts.push_back({r.buf, r.mem, 0U});
        return static_cast<int>(b.insts.size() - 1U);
    }
    void dyn_write(BufferObj& b, BufferObj::DynInst& d, std::size_t off, std::size_t n) {
        if (off >= b.shadow.size() || d.mem.mapped == nullptr) return;
        n = std::min(n, b.shadow.size() - off);
        n = std::min<std::size_t>(n, std::max<std::size_t>(b.size, 16U) - std::min<std::size_t>(off, b.size));
        if (n == 0U) return;
        std::memcpy(static_cast<std::uint8_t*>(d.mem.mapped) + off, b.shadow.data() + off, n);
        alloc.flush(d.mem, off, n);
    }

    void destroy_dummy_tex(DummyTex& d) noexcept {
        if (d.view != VK_NULL_HANDLE) {
            vk.destroy_image_view(h.device, d.view, nullptr);
            d.view = VK_NULL_HANDLE;
        }
        if (d.img != VK_NULL_HANDLE) {
            vk.destroy_image(h.device, d.img, nullptr);
            d.img = VK_NULL_HANDLE;
        }
        alloc.free(d.mem);
    }

    void destroy_texture_gpu(TextureObj& t) noexcept {
        if (t.view != VK_NULL_HANDLE) {
            vk.destroy_image_view(h.device, t.view, nullptr);
            t.view = VK_NULL_HANDLE;
        }
        if (t.img != VK_NULL_HANDLE) {
            vk.destroy_image(h.device, t.img, nullptr);
            t.img = VK_NULL_HANDLE;
        }
        alloc.free(t.mem);
        t.layout = VK_IMAGE_LAYOUT_UNDEFINED;
    }

    void destroy_backbuffer() noexcept {
        if (bb_fb != VK_NULL_HANDLE) {
            vk.destroy_framebuffer(h.device, bb_fb, nullptr);
            bb_fb = VK_NULL_HANDLE;
        }
        if (bb_view != VK_NULL_HANDLE) {
            vk.destroy_image_view(h.device, bb_view, nullptr);
            bb_view = VK_NULL_HANDLE;
        }
        if (bb_img != VK_NULL_HANDLE) {
            vk.destroy_image(h.device, bb_img, nullptr);
            bb_img = VK_NULL_HANDLE;
        }
        alloc.free(bb_mem);
        bb_layout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (depth_view != VK_NULL_HANDLE) {
            vk.destroy_image_view(h.device, depth_view, nullptr);
            depth_view = VK_NULL_HANDLE;
        }
        if (depth_img != VK_NULL_HANDLE) {
            vk.destroy_image(h.device, depth_img, nullptr);
            depth_img = VK_NULL_HANDLE;
        }
        alloc.free(depth_mem);
        depth_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    }

    void load_pipeline_cache() {
        if (vk.create_pipeline_cache == nullptr) {
            report.pipeline_cache_active = false;
            report.missing_features.push_back("VkPipelineCache(missing proc)");
            return;
        }
        std::vector<std::uint8_t> init_data;
        if (!config.pipeline_cache_path.empty()) {
            std::ifstream in(config.pipeline_cache_path, std::ios::binary);
            if (in) {
                init_data.assign((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            }
        }
        VkPipelineCacheCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
        ci.initialDataSize = init_data.size();
        ci.pInitialData = init_data.empty() ? nullptr : init_data.data();
        if (vk.create_pipeline_cache(h.device, &ci, nullptr, &pipeline_cache) == VK_SUCCESS &&
            pipeline_cache != VK_NULL_HANDLE) {
            report.pipeline_cache_active = true;
        } else {
            pipeline_cache = VK_NULL_HANDLE;
            report.pipeline_cache_active = false;
            report.missing_features.push_back("VkPipelineCache(fallback none)");
        }
    }

    void save_pipeline_cache() noexcept {
        if (pipeline_cache == VK_NULL_HANDLE || config.pipeline_cache_path.empty() ||
            vk.get_pipeline_cache_data == nullptr) {
            return;
        }
        std::size_t sz = 0;
        if (vk.get_pipeline_cache_data(h.device, pipeline_cache, &sz, nullptr) != VK_SUCCESS || sz == 0) return;
        std::vector<std::uint8_t> data(sz);
        if (vk.get_pipeline_cache_data(h.device, pipeline_cache, &sz, data.data()) != VK_SUCCESS) return;
        std::ofstream out(config.pipeline_cache_path, std::ios::binary | std::ios::trunc);
        if (out) out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(sz));
    }

    bool submit_upload(VkCommandBuffer cmd, std::string* err = nullptr) {
        if (vk.reset_fences(h.device, 1U, &upload_fence) != VK_SUCCESS) {
            set_err(err, "vkResetFences(upload) failed");
            return false;
        }
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1U;
        si.pCommandBuffers = &cmd;
        if (vk.queue_submit(h.graphics_queue, 1U, &si, upload_fence) != VK_SUCCESS) {
            set_err(err, "vkQueueSubmit(upload) failed");
            return false;
        }
        if (vk.wait_for_fences(h.device, 1U, &upload_fence, VK_TRUE, 10'000'000'000ULL) != VK_SUCCESS) {
            set_err(err, "vkWaitForFences(upload) timed out");
            return false;
        }
        return true;
    }

    bool ensure_staging(VkDeviceSize needed, std::string* err = nullptr) {
        if (staging_buf != VK_NULL_HANDLE && staging_cap >= needed) return true;
        if (staging_buf != VK_NULL_HANDLE) {
            vk.destroy_buffer(h.device, staging_buf, nullptr);
            staging_buf = VK_NULL_HANDLE;
        }
        alloc.free(staging_mem);
        staging_cap = 0;
        const VkDeviceSize cap = std::max<VkDeviceSize>(needed, 1024ULL * 1024ULL);
        VkBufferCreateInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size = cap;
        bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vk.create_buffer(h.device, &bi, nullptr, &staging_buf) != VK_SUCCESS) {
            set_err(err, "vkCreateBuffer(staging) failed");
            return false;
        }
        VkMemoryRequirements reqs{};
        vk.get_buffer_mem_reqs(h.device, staging_buf, &reqs);
        if (!alloc.alloc(reqs, false, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging_mem, err) ||
            vk.bind_buffer_memory(h.device, staging_buf, staging_mem.memory, staging_mem.offset) != VK_SUCCESS) {
            vk.destroy_buffer(h.device, staging_buf, nullptr);
            staging_buf = VK_NULL_HANDLE;
            alloc.free(staging_mem);
            return false;
        }
        staging_cap = cap;
        return true;
    }

    bool ensure_readback(VkDeviceSize needed, std::string* err = nullptr) {
        if (readback_buf != VK_NULL_HANDLE && readback_cap >= needed) return true;
        if (readback_buf != VK_NULL_HANDLE) {
            vk.destroy_buffer(h.device, readback_buf, nullptr);
            readback_buf = VK_NULL_HANDLE;
        }
        alloc.free(readback_mem);
        readback_cap = 0;
        VkBufferCreateInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size = needed;
        bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vk.create_buffer(h.device, &bi, nullptr, &readback_buf) != VK_SUCCESS) {
            set_err(err, "vkCreateBuffer(readback) failed");
            return false;
        }
        VkMemoryRequirements reqs{};
        vk.get_buffer_mem_reqs(h.device, readback_buf, &reqs);
        if (!alloc.alloc(reqs, false, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                         VK_MEMORY_PROPERTY_HOST_CACHED_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         readback_mem, err) ||
            vk.bind_buffer_memory(h.device, readback_buf, readback_mem.memory, readback_mem.offset) != VK_SUCCESS) {
            vk.destroy_buffer(h.device, readback_buf, nullptr);
            readback_buf = VK_NULL_HANDLE;
            alloc.free(readback_mem);
            return false;
        }
        readback_cap = needed;
        return true;
    }

    bool create_host_ring(RingBuf& ring, VkDeviceSize cap, VkBufferUsageFlags usage, std::string* err = nullptr) {
        VkBufferCreateInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size = cap;
        bi.usage = usage;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkBuffer buf = VK_NULL_HANDLE;
        if (vk.create_buffer(h.device, &bi, nullptr, &buf) != VK_SUCCESS) {
            set_err(err, "vkCreateBuffer(ring) failed");
            return false;
        }
        VkMemoryRequirements reqs{};
        vk.get_buffer_mem_reqs(h.device, buf, &reqs);
        MemAlloc mem{};
        if (!alloc.alloc(reqs, false, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, mem, err) ||
            vk.bind_buffer_memory(h.device, buf, mem.memory, mem.offset) != VK_SUCCESS) {
            vk.destroy_buffer(h.device, buf, nullptr);
            alloc.free(mem);
            return false;
        }
        ring = RingBuf{buf, mem, cap, 0};
        return true;
    }

    bool bind_ubo_desc_set(VkBuffer ubo_buf, VkDescriptorSet& out_set, std::string* err = nullptr) {
        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = ubo_desc_pool;
        ai.descriptorSetCount = 1U;
        ai.pSetLayouts = &dsl_ubo;
        VkDescriptorSet ds = VK_NULL_HANDLE;
        if (vk.allocate_desc_sets(h.device, &ai, &ds) != VK_SUCCESS) {
            set_err(err, "vkAllocateDescriptorSets(ubo) failed");
            return false;
        }
        VkDescriptorBufferInfo bi{};
        bi.buffer = ubo_buf;
        bi.offset = 0;
        bi.range = sizeof(StageUboData);
        std::array<VkWriteDescriptorSet, 2> w{};
        for (std::uint32_t i = 0; i < 2U; ++i) {
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = ds;
            w[i].dstBinding = i;
            w[i].descriptorCount = 1U;
            w[i].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
            w[i].pBufferInfo = &bi;
        }
        vk.update_desc_sets(h.device, 2U, w.data(), 0U, nullptr);
        out_set = ds;
        return true;
    }

    bool alloc_ubo_pair(VkDeviceSize& vs_off, VkDeviceSize& ps_off) {
        const VkDeviceSize need = ubo_stride * 2U;
        if (ubo_ring.cursor + need > ubo_ring.cap) {
            const VkDeviceSize new_cap = std::max(ubo_ring.cap * 2U, align_up(need, 1024ULL * 1024ULL));
            RingBuf next{};
            if (!create_host_ring(next, new_cap, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &last_error)) return false;
            if (ubo_ring.buf != VK_NULL_HANDLE) retired_bufs.push_back({ubo_ring.buf, ubo_ring.mem});
            ubo_ring = next;
            if (!bind_ubo_desc_set(ubo_ring.buf, ubo_desc_set, &last_error)) return false;
        }
        vs_off = ubo_ring.cursor;
        ps_off = ubo_ring.cursor + ubo_stride;
        ubo_ring.cursor += need;
        return true;
    }

    bool alloc_dyn_geom(VkDeviceSize bytes, VkBuffer& out_buf, VkDeviceSize& out_off) {
        const VkDeviceSize aligned = align_up(std::max<VkDeviceSize>(bytes, 16U), 256U);
        if (dyn_geom_ring.cursor + aligned > dyn_geom_ring.cap) {
            const VkDeviceSize new_cap = std::max(dyn_geom_ring.cap * 2U, align_up(aligned * 2U, 4ULL * 1024ULL * 1024ULL));
            RingBuf next{};
            if (!create_host_ring(next, new_cap,
                                  VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                  &last_error)) {
                return false;
            }
            if (dyn_geom_ring.buf != VK_NULL_HANDLE) retired_bufs.push_back({dyn_geom_ring.buf, dyn_geom_ring.mem});
            dyn_geom_ring = next;
        }
        out_buf = dyn_geom_ring.buf;
        out_off = dyn_geom_ring.cursor;
        dyn_geom_ring.cursor += aligned;
        return true;
    }

    VkRenderPass get_render_pass(const RenderPassKey& key) {
        if (const auto it = render_passes.find(key); it != render_passes.end()) return it->second;
        std::array<VkAttachmentDescription, 2> atts{};
        atts[0].format = key.color_fmt;
        atts[0].samples = VK_SAMPLE_COUNT_1_BIT;
        atts[0].loadOp = key.clear_color ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
        atts[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        atts[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        atts[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        atts[0].initialLayout = key.clear_color ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        atts[0].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference color_ref{0U, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkAttachmentReference depth_ref{1U, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

        std::uint32_t att_cnt = 1U;
        const bool has_d = key.depth_fmt != VK_FORMAT_UNDEFINED;
        if (has_d) {
            att_cnt = 2U;
            atts[1].format = key.depth_fmt;
            atts[1].samples = VK_SAMPLE_COUNT_1_BIT;
            atts[1].loadOp = key.clear_depth ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
            atts[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            atts[1].stencilLoadOp = key.clear_depth ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
            atts[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE;
            atts[1].initialLayout =
                key.clear_depth ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            atts[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        }

        VkSubpassDescription sub{};
        sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sub.colorAttachmentCount = 1U;
        sub.pColorAttachments = &color_ref;
        sub.pDepthStencilAttachment = has_d ? &depth_ref : nullptr;

        VkSubpassDependency dep{};
        dep.srcSubpass = VK_SUBPASS_EXTERNAL;
        dep.dstSubpass = 0U;
        dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dep.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo rpci{};
        rpci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rpci.attachmentCount = att_cnt;
        rpci.pAttachments = atts.data();
        rpci.subpassCount = 1U;
        rpci.pSubpasses = &sub;
        rpci.dependencyCount = 1U;
        rpci.pDependencies = &dep;

        VkRenderPass rp = VK_NULL_HANDLE;
        if (vk.create_render_pass(h.device, &rpci, nullptr, &rp) != VK_SUCCESS) {
            last_error = "vkCreateRenderPass failed";
            return VK_NULL_HANDLE;
        }
        render_passes.emplace(key, rp);
        return rp;
    }

    bool create_backbuffer(std::uint32_t w, std::uint32_t ht, std::string* err) {
        destroy_backbuffer();
        bb_width = std::max<std::uint32_t>(1U, w);
        bb_height = std::max<std::uint32_t>(1U, ht);

        // Color target (RGBA8 UNORM).
        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = VK_FORMAT_R8G8B8A8_UNORM;
        ici.extent = {bb_width, bb_height, 1U};
        ici.mipLevels = 1U;
        ici.arrayLayers = 1U;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vk.create_image(h.device, &ici, nullptr, &bb_img) != VK_SUCCESS) {
            set_err(err, "vkCreateImage(backbuffer) failed");
            return false;
        }
        VkMemoryRequirements reqs{};
        vk.get_image_mem_reqs(h.device, bb_img, &reqs);
        if (!alloc.alloc(reqs, true, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0U, bb_mem, err) ||
            vk.bind_image_memory(h.device, bb_img, bb_mem.memory, bb_mem.offset) != VK_SUCCESS) {
            destroy_backbuffer();
            return false;
        }
        VkImageViewCreateInfo ivci{};
        ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ivci.image = bb_img;
        ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ivci.format = VK_FORMAT_R8G8B8A8_UNORM;
        ivci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0U, 1U, 0U, 1U};
        if (vk.create_image_view(h.device, &ivci, nullptr, &bb_view) != VK_SUCCESS) {
            set_err(err, "vkCreateImageView(backbuffer) failed");
            destroy_backbuffer();
            return false;
        }
        bb_layout = VK_IMAGE_LAYOUT_UNDEFINED;

        // Depth target.
        ici.format = default_depth_fmt;
        ici.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        if (vk.create_image(h.device, &ici, nullptr, &depth_img) != VK_SUCCESS) {
            set_err(err, "vkCreateImage(depth) failed");
            destroy_backbuffer();
            return false;
        }
        vk.get_image_mem_reqs(h.device, depth_img, &reqs);
        if (!alloc.alloc(reqs, true, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0U, depth_mem, err) ||
            vk.bind_image_memory(h.device, depth_img, depth_mem.memory, depth_mem.offset) != VK_SUCCESS) {
            destroy_backbuffer();
            return false;
        }
        ivci.image = depth_img;
        ivci.format = default_depth_fmt;
        ivci.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0U, 1U, 0U, 1U};
        if (vk.create_image_view(h.device, &ivci, nullptr, &depth_view) != VK_SUCCESS) {
            set_err(err, "vkCreateImageView(depth) failed");
            destroy_backbuffer();
            return false;
        }
        depth_layout = VK_IMAGE_LAYOUT_UNDEFINED;

        VkRenderPass compat_rp = get_render_pass({VK_FORMAT_R8G8B8A8_UNORM, default_depth_fmt, true, true});
        if (compat_rp == VK_NULL_HANDLE) {
            set_err(err, last_error);
            destroy_backbuffer();
            return false;
        }
        std::array<VkImageView, 2> views{bb_view, depth_view};
        VkFramebufferCreateInfo fbci{};
        fbci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbci.renderPass = compat_rp;
        fbci.attachmentCount = 2U;
        fbci.pAttachments = views.data();
        fbci.width = bb_width;
        fbci.height = bb_height;
        fbci.layers = 1U;
        if (vk.create_framebuffer(h.device, &fbci, nullptr, &bb_fb) != VK_SUCCESS) {
            set_err(err, "vkCreateFramebuffer(backbuffer) failed");
            destroy_backbuffer();
            return false;
        }
        return true;
    }

    bool create_dummy_texture(int kind, DummyTex& dst, std::string* err) {
        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = (kind == 2) ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
        ici.flags = (kind == 1) ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0U;
        ici.format = VK_FORMAT_R8G8B8A8_UNORM;
        ici.extent = {1U, 1U, 1U};
        ici.mipLevels = 1U;
        ici.arrayLayers = (kind == 1) ? 6U : 1U;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vk.create_image(h.device, &ici, nullptr, &dst.img) != VK_SUCCESS) {
            set_err(err, "vkCreateImage(dummy) failed");
            return false;
        }
        VkMemoryRequirements reqs{};
        vk.get_image_mem_reqs(h.device, dst.img, &reqs);
        if (!alloc.alloc(reqs, true, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0U, dst.mem, err) ||
            vk.bind_image_memory(h.device, dst.img, dst.mem.memory, dst.mem.offset) != VK_SUCCESS) {
            destroy_dummy_tex(dst);
            return false;
        }
        VkImageViewCreateInfo ivci{};
        ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ivci.image = dst.img;
        ivci.viewType = (kind == 1) ? VK_IMAGE_VIEW_TYPE_CUBE : (kind == 2) ? VK_IMAGE_VIEW_TYPE_3D : VK_IMAGE_VIEW_TYPE_2D;
        ivci.format = VK_FORMAT_R8G8B8A8_UNORM;
        ivci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0U, 1U, 0U, ici.arrayLayers};
        if (vk.create_image_view(h.device, &ivci, nullptr, &dst.view) != VK_SUCCESS) {
            set_err(err, "vkCreateImageView(dummy) failed");
            destroy_dummy_tex(dst);
            return false;
        }
        const std::vector<std::uint8_t> px(ici.arrayLayers * 4U, 255U);
        if (!ensure_staging(px.size(), err)) return false;
        std::memcpy(staging_mem.mapped, px.data(), px.size());
        alloc.flush(staging_mem, 0, px.size());

        (void)vk.reset_cmd_buffer(upload_cmd, 0U);
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vk.begin_cmd_buffer(upload_cmd, &bi) != VK_SUCCESS) return false;
        VkImageMemoryBarrier bar{};
        bar.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        bar.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        bar.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bar.image = dst.img;
        bar.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0U, 1U, 0U, ici.arrayLayers};
        bar.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vk.cmd_pipeline_barrier(upload_cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                0U, 0U, nullptr, 0U, nullptr, 1U, &bar);
        VkBufferImageCopy cpy{};
        cpy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0U, 0U, ici.arrayLayers};
        cpy.imageExtent = {1U, 1U, 1U};
        vk.cmd_copy_buffer_to_image(upload_cmd, staging_buf, dst.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1U, &cpy);
        bar.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        bar.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        bar.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vk.cmd_pipeline_barrier(upload_cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                0U, 0U, nullptr, 0U, nullptr, 1U, &bar);
        if (vk.end_cmd_buffer(upload_cmd) != VK_SUCCESS) return false;
        return submit_upload(upload_cmd, err);
    }

    bool init(std::uint32_t w, std::uint32_t ht, std::string* err) {
        shutdown();
        report = D3d9DriverReport{};
        if (!loader || !vulkan_loader_extract_handles(*loader, h)) {
            report.ok = false;
            report.error = "VulkanDriverLoader has no active VkDevice";
            set_err(err, report.error);
            return false;
        }
        vk.resolve(h);
        report.is_mock = h.is_mock;
        report.driver_mode = std::string(driver_mode_name(h.probe.mode));
        report.device_name = h.probe.device_name;
        report.api_version = format_vulkan_version(h.probe.api_version);
        report.driver_version = format_vulkan_version(h.probe.driver_version);
        report.presentation_path =
            (h.probe.presentation_path == PresentationPath::android_surface_wsi)
                ? "android_surface_wsi"
                : "offscreen_hardware_buffer_bridge";
        report.vendor_id = h.probe.vendor_id;
        report.device_id = h.probe.device_id;
        report.max_image_dimension_2d = h.probe.max_image_dimension_2d;
        report.texture_compression_bc = h.probe.texture_compression_bc;

        compiler = wse::asset::create_glslang_shader_compiler();
        if (!compiler) {
            report.error = "glslang shader compiler unavailable";
            set_err(err, report.error);
            return false;
        }

        if (vk.get_phys_props != nullptr && h.physical_device != VK_NULL_HANDLE) {
            vk.get_phys_props(h.physical_device, &phys_props);
            report.max_image_dimension_2d = phys_props.limits.maxImageDimension2D;
            report.max_bound_descriptor_sets = phys_props.limits.maxBoundDescriptorSets;
            report.max_per_stage_samplers = phys_props.limits.maxPerStageDescriptorSamplers;
            report.max_vertex_output_components = phys_props.limits.maxVertexOutputComponents;
            min_ubo_align = std::max<VkDeviceSize>(16U, phys_props.limits.minUniformBufferOffsetAlignment);
            ubo_stride = align_up(sizeof(StageUboData), min_ubo_align);
        }
        if (vk.get_phys_features != nullptr && h.physical_device != VK_NULL_HANDLE) {
            vk.get_phys_features(h.physical_device, &phys_features);
        }

        // Mock / non-executing device: record capability report and return ok without allocating GPU resources.
        if (h.is_mock || !h.can_execute_gpu || !vk.has_core_gpu()) {
            if (!report.texture_compression_bc) report.missing_features.push_back("textureCompressionBC -> RGBA8");
            if (!h.is_mock && !vk.has_core_gpu()) report.missing_features.push_back("missing core Vulkan GPU procs");
            report.ok = true;
            initialized = true;
            gpu_ready = false;
            bb_width = std::max<std::uint32_t>(1U, w);
            bb_height = std::max<std::uint32_t>(1U, ht);
            return true;
        }

        // Check Vulkan 1.1 limits & optional formats required for PanVK / Turnip / System Vulkan.
        if (report.max_bound_descriptor_sets < 2U) {
            report.missing_features.push_back("maxBoundDescriptorSets<2");
        }
        if (report.max_per_stage_samplers > 0U && report.max_per_stage_samplers < kMaxSamplers) {
            sampler_binding_count = report.max_per_stage_samplers;
            report.missing_features.push_back("maxPerStageDescriptorSamplers=" +
                                              std::to_string(report.max_per_stage_samplers));
        } else {
            sampler_binding_count = kMaxSamplers;
        }
        if (report.max_vertex_output_components > 0U && report.max_vertex_output_components < 64U) {
            report.missing_features.push_back("maxVertexOutputComponents=" +
                                              std::to_string(report.max_vertex_output_components));
        }

        // Probe BC1/BC2/BC3 sampling support.
        bool bc_ok = (phys_features.textureCompressionBC == VK_TRUE);
        if (bc_ok) {
            for (VkFormat f : {VK_FORMAT_BC1_RGBA_UNORM_BLOCK, VK_FORMAT_BC2_UNORM_BLOCK, VK_FORMAT_BC3_UNORM_BLOCK}) {
                VkFormatProperties fp{};
                vk.get_phys_format_props(h.physical_device, f, &fp);
                if ((fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) == 0U) bc_ok = false;
            }
        }
        report.texture_compression_bc = bc_ok && config.prefer_bc_compression;
        if (!bc_ok) {
            report.missing_features.push_back("textureCompressionBC -> RGBA8");
        } else if (!config.prefer_bc_compression) {
            report.missing_features.push_back("textureCompressionBC disabled -> RGBA8");
        }

        // Probe BGRA8 vertex attribute support (D3DDECLTYPE_D3DCOLOR).
        {
            VkFormatProperties fp{};
            vk.get_phys_format_props(h.physical_device, VK_FORMAT_B8G8R8A8_UNORM, &fp);
            const bool hw_bgra = (fp.bufferFeatures & VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT) != 0U;
            report.bgra_vertex_format = hw_bgra && config.prefer_bgra_vertex_format;
            if (!hw_bgra) {
                report.missing_features.push_back("B8G8R8A8_UNORM vertex -> shader swizzle");
            } else if (!config.prefer_bgra_vertex_format) {
                report.missing_features.push_back("B8G8R8A8_UNORM vertex disabled -> shader swizzle");
            }
        }
        if (!h.sampler_anisotropy) report.missing_features.push_back("samplerAnisotropy -> 1x");
        if (!h.fill_mode_non_solid) report.missing_features.push_back("fillModeNonSolid -> solid");

        // Probe depth format.
        default_depth_fmt = VK_FORMAT_UNDEFINED;
        for (VkFormat f : {VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT_S8_UINT,
                           VK_FORMAT_D32_SFLOAT, VK_FORMAT_D16_UNORM}) {
            VkFormatProperties fp{};
            vk.get_phys_format_props(h.physical_device, f, &fp);
            if ((fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0U) {
                default_depth_fmt = f;
                break;
            }
        }
        if (default_depth_fmt == VK_FORMAT_UNDEFINED) {
            report.error = "no supported Vulkan depth format";
            set_err(err, report.error);
            return false;
        }

        alloc.init(h.device, &vk, h.physical_device, phys_props.limits.nonCoherentAtomSize);
        initialized = true;
        gpu_ready = true;

        // Allocate upload & frame command buffers + fences.
        {
            VkCommandBufferAllocateInfo cbai{};
            cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            cbai.commandPool = h.command_pool;
            cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cbai.commandBufferCount = 2U;
            VkCommandBuffer cbs[2]{};
            if (vk.allocate_cmd_buffers(h.device, &cbai, cbs) != VK_SUCCESS) {
                report.error = "vkAllocateCommandBuffers failed";
                set_err(err, report.error);
                shutdown();
                return false;
            }
            upload_cmd = cbs[0];
            frame_cmd = cbs[1];
            VkFenceCreateInfo fci{};
            fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            if (vk.create_fence(h.device, &fci, nullptr, &upload_fence) != VK_SUCCESS ||
                vk.create_fence(h.device, &fci, nullptr, &frame_fence) != VK_SUCCESS) {
                report.error = "vkCreateFence failed";
                set_err(err, report.error);
                shutdown();
                return false;
            }
            // Slot 0 is the current one (`frame_cmd`/`frame_fence` above); the rest are parked.
            cbai.commandBufferCount = 1U;
            if (vk.allocate_cmd_buffers(h.device, &cbai, &up_cmd) != VK_SUCCESS) {
                report.error = "vkAllocateCommandBuffers failed";
                set_err(err, report.error);
                shutdown();
                return false;
            }
            for (std::size_t i = 1; i < kFramesInFlight; ++i) {
                FrameSlot& s = slots[i];
                VkCommandBuffer two[2]{};
                cbai.commandBufferCount = 2U;
                if (vk.allocate_cmd_buffers(h.device, &cbai, two) != VK_SUCCESS ||
                    vk.create_fence(h.device, &fci, nullptr, &s.fence) != VK_SUCCESS) {
                    report.error = "frame slot allocation failed";
                    set_err(err, report.error);
                    shutdown();
                    return false;
                }
                s.cmd = two[0];
                s.up_cmd = two[1];
            }
        }

        load_pipeline_cache();

        // Descriptor set layouts: set 0 (VS + PS dynamic UBOs), set 1 (16 combined image samplers).
        {
            std::array<VkDescriptorSetLayoutBinding, 2> b0{};
            b0[0] = {0U, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1U, VK_SHADER_STAGE_VERTEX_BIT, nullptr};
            b0[1] = {1U, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1U, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
            VkDescriptorSetLayoutCreateInfo ci0{};
            ci0.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            ci0.bindingCount = 2U;
            ci0.pBindings = b0.data();
            if (vk.create_desc_set_layout(h.device, &ci0, nullptr, &dsl_ubo) != VK_SUCCESS) {
                report.error = "vkCreateDescriptorSetLayout(ubo) failed";
                set_err(err, report.error);
                shutdown();
                return false;
            }

            std::vector<VkDescriptorSetLayoutBinding> b1(sampler_binding_count);
            for (std::uint32_t i = 0; i < sampler_binding_count; ++i) {
                b1[i] = {i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1U,
                         VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
            }
            VkDescriptorSetLayoutCreateInfo ci1{};
            ci1.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            ci1.bindingCount = sampler_binding_count;
            ci1.pBindings = b1.data();
            if (vk.create_desc_set_layout(h.device, &ci1, nullptr, &dsl_samplers) != VK_SUCCESS) {
                report.error = "vkCreateDescriptorSetLayout(samplers) failed";
                set_err(err, report.error);
                shutdown();
                return false;
            }

            std::array<VkDescriptorSetLayout, 2> sets{dsl_ubo, dsl_samplers};
            VkPipelineLayoutCreateInfo plci{};
            plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            plci.setLayoutCount = 2U;
            plci.pSetLayouts = sets.data();
            if (vk.create_pipeline_layout(h.device, &plci, nullptr, &pipeline_layout) != VK_SUCCESS) {
                report.error = "vkCreatePipelineLayout failed";
                set_err(err, report.error);
                shutdown();
                return false;
            }
        }

        // UBO descriptor pool (allows growth reallocations).
        {
            VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 64U};
            VkDescriptorPoolCreateInfo pci{};
            pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            pci.maxSets = 32U;
            pci.poolSizeCount = 1U;
            pci.pPoolSizes = &ps;
            if (vk.create_desc_pool(h.device, &pci, nullptr, &ubo_desc_pool) != VK_SUCCESS) {
                report.error = "vkCreateDescriptorPool(ubo) failed";
                set_err(err, report.error);
                shutdown();
                return false;
            }
        }

        bool rings_ok = create_host_ring(ubo_ring, kInitialUboRingBytes, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, err) &&
                        bind_ubo_desc_set(ubo_ring.buf, ubo_desc_set, err) &&
                        create_host_ring(dyn_geom_ring, kInitialDynGeomBytes,
                                         VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, err);
        for (std::size_t i = 1; rings_ok && i < kFramesInFlight; ++i) {
            FrameSlot& s = slots[i];
            rings_ok = create_host_ring(s.ubo, kInitialUboRingBytes, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, err) &&
                       bind_ubo_desc_set(s.ubo.buf, s.ubo_set, err) &&
                       create_host_ring(s.geom, kInitialDynGeomBytes,
                                        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, err);
        }
        if (!rings_ok) {
            report.error = err != nullptr ? *err : "ring allocation failed";
            shutdown();
            return false;
        }

        // Dummy 16-byte vertex buffer {0, 0, 0, 1}.
        {
            VkBufferCreateInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bi.size = 16U;
            bi.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
            bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            if (vk.create_buffer(h.device, &bi, nullptr, &dummy_vb) != VK_SUCCESS) {
                shutdown();
                return false;
            }
            VkMemoryRequirements reqs{};
            vk.get_buffer_mem_reqs(h.device, dummy_vb, &reqs);
            if (!alloc.alloc(reqs, false, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, dummy_vb_mem, err) ||
                vk.bind_buffer_memory(h.device, dummy_vb, dummy_vb_mem.memory, dummy_vb_mem.offset) != VK_SUCCESS) {
                shutdown();
                return false;
            }
            const float v[4] = {0.0F, 0.0F, 0.0F, 1.0F};
            std::memcpy(dummy_vb_mem.mapped, v, 16);
            alloc.flush(dummy_vb_mem, 0, 16);
        }

        default_sampler = get_sampler(D3d9SamplerState{});
        if (default_sampler == VK_NULL_HANDLE ||
            !create_dummy_texture(0, dummy_2d, err) ||
            !create_dummy_texture(1, dummy_cube, err) ||
            !create_dummy_texture(2, dummy_3d, err) ||
            !create_backbuffer(w, ht, err)) {
            report.error = err != nullptr ? *err : "backbuffer/dummy init failed";
            shutdown();
            return false;
        }

#ifdef __ANDROID__
        if (active_window != nullptr) {
            setup_android_presentation_locked();
        }
#endif

        report.ok = true;
        return true;
    }

    VkSampler get_sampler(const D3d9SamplerState& s) {
        SamplerKey key{};
        key.addr_u = s.address_u;
        key.addr_v = s.address_v;
        key.addr_w = s.address_w;
        key.mag = s.mag_filter;
        key.min = s.min_filter;
        key.mip = s.mip_filter;
        key.max_aniso = h.sampler_anisotropy ? s.max_anisotropy : 1U;
        key.border = s.border_color;
        std::memcpy(&key.lod_bias_bits, &s.lod_bias, 4);
        if (const auto it = sampler_cache.find(key); it != sampler_cache.end()) return it->second;

        VkSamplerCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        ci.magFilter = map_filter(s.mag_filter);
        ci.minFilter = map_filter(s.min_filter);
        ci.mipmapMode = (s.mip_filter == 2U) ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
        ci.addressModeU = map_address_mode(s.address_u);
        ci.addressModeV = map_address_mode(s.address_v);
        ci.addressModeW = map_address_mode(s.address_w);
        ci.mipLodBias = s.lod_bias;
        const bool want_aniso = h.sampler_anisotropy && (s.min_filter == 3U || s.mag_filter == 3U) && s.max_anisotropy > 1U;
        ci.anisotropyEnable = want_aniso ? VK_TRUE : VK_FALSE;
        ci.maxAnisotropy = want_aniso
                               ? std::min<float>(static_cast<float>(s.max_anisotropy), phys_props.limits.maxSamplerAnisotropy)
                               : 1.0F;
        ci.minLod = 0.0F;
        ci.maxLod = (s.mip_filter == 0U) ? 0.0F : VK_LOD_CLAMP_NONE;
        ci.borderColor = (s.border_color == 0U)             ? VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK
                         : (s.border_color == 0xFF000000U) ? VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK
                                                            : VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        VkSampler smp = VK_NULL_HANDLE;
        if (vk.create_sampler(h.device, &ci, nullptr, &smp) != VK_SUCCESS) return VK_NULL_HANDLE;
        sampler_cache.emplace(key, smp);
        return smp;
    }

    VkDescriptorSet get_sampler_set(const D3d9DrawState& st, const d3d9sh::Shader& vs, const d3d9sh::Shader& ps) {
        SamplerSetKey key{};
        std::array<VkDescriptorImageInfo, kMaxSamplers> infos{};
        for (std::uint32_t i = 0; i < sampler_binding_count; ++i) {
            const int need_dim = ps.samplers[i] != 0 ? ps.samplers[i] : vs.samplers[i];
            VkImageView view = (need_dim == 3) ? dummy_cube.view : (need_dim == 4) ? dummy_3d.view : dummy_2d.view;
            VkSampler smp = default_sampler;
            if (need_dim != 0 && st.textures[i] != 0) {
                const auto tit = textures.find(st.textures[i]);
                if (tit != textures.end()) tit->second.last_used_frame = frame_index;
                if (tit != textures.end() && tit->second.view != VK_NULL_HANDLE) {
                    const bool kind_match = (need_dim == 3 && tit->second.kind == 1) ||
                                            (need_dim == 4 && tit->second.kind == 2) ||
                                            (need_dim != 3 && need_dim != 4 && tit->second.kind == 0);
                    if (kind_match) view = tit->second.view;
                }
                smp = get_sampler(st.samplers[i]);
                if (smp == VK_NULL_HANDLE) smp = default_sampler;
            }
            infos[i] = {smp, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            std::uint64_t vh = 0, sh = 0;
            std::memcpy(&vh, &view, sizeof(view));
            std::memcpy(&sh, &smp, sizeof(smp));
            key.views[i] = vh;
            key.samplers[i] = sh;
        }
        if (const auto it = sampler_set_cache.find(key); it != sampler_set_cache.end()) return it->second;

        auto alloc_from_pool = [&](VkDescriptorPool pool) -> VkDescriptorSet {
            VkDescriptorSetAllocateInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            ai.descriptorPool = pool;
            ai.descriptorSetCount = 1U;
            ai.pSetLayouts = &dsl_samplers;
            VkDescriptorSet ds = VK_NULL_HANDLE;
            return vk.allocate_desc_sets(h.device, &ai, &ds) == VK_SUCCESS ? ds : VK_NULL_HANDLE;
        };

        VkDescriptorSet ds = VK_NULL_HANDLE;
        if (!sampler_desc_pools.empty()) ds = alloc_from_pool(sampler_desc_pools.back());
        if (ds == VK_NULL_HANDLE) {
            constexpr std::uint32_t kSetsPerPool = 256U;
            VkDescriptorPoolSize ps_sz{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kSetsPerPool * sampler_binding_count};
            VkDescriptorPoolCreateInfo pci{};
            pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            pci.maxSets = kSetsPerPool;
            pci.poolSizeCount = 1U;
            pci.pPoolSizes = &ps_sz;
            VkDescriptorPool pool = VK_NULL_HANDLE;
            if (vk.create_desc_pool(h.device, &pci, nullptr, &pool) != VK_SUCCESS) return VK_NULL_HANDLE;
            sampler_desc_pools.push_back(pool);
            ds = alloc_from_pool(pool);
            if (ds == VK_NULL_HANDLE) return VK_NULL_HANDLE;
        }

        std::array<VkWriteDescriptorSet, kMaxSamplers> writes{};
        for (std::uint32_t i = 0; i < sampler_binding_count; ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = ds;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1U;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[i].pImageInfo = &infos[i];
        }
        vk.update_desc_sets(h.device, sampler_binding_count, writes.data(), 0U, nullptr);
        sampler_set_cache.emplace(key, ds);
        return ds;
    }

    // Submits what this frame has recorded so far and waits for it (rare since the 2026-11-22
    // pass: object destruction, window changes, a full staging ring).
    void flush_frame_if_recording() {
        if (!gpu_ready || (!frame_recording && !up_recording)) return;
        (void)submit_frame_locked(VK_NULL_HANDLE, 0U, VK_NULL_HANDLE, true);
    }

    bool ensure_frame_recording() {
        if (!gpu_ready) return false;
        if (frame_recording) return true;
        if (vk.reset_cmd_buffer(frame_cmd, 0U) != VK_SUCCESS) return false;
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vk.begin_cmd_buffer(frame_cmd, &bi) != VK_SUCCESS) return false;
        frame_recording = true;
        bound_pipeline = VK_NULL_HANDLE;
        bound_sampler_set = VK_NULL_HANDLE;
        return true;
    }

    bool ensure_render_pass_active() {
        if (!ensure_frame_recording()) return false;
        if (render_pass_active) return true;

        const bool do_clear_c = pending_clear_color || bb_layout == VK_IMAGE_LAYOUT_UNDEFINED;
        const bool do_clear_d = pending_clear_depth || depth_layout == VK_IMAGE_LAYOUT_UNDEFINED;
        VkRenderPass rp = get_render_pass({VK_FORMAT_R8G8B8A8_UNORM, default_depth_fmt, do_clear_c, do_clear_d});
        if (rp == VK_NULL_HANDLE) return false;

        std::array<VkClearValue, 2> cv{};
        cv[0].color.float32[0] = clear_color_val[0];
        cv[0].color.float32[1] = clear_color_val[1];
        cv[0].color.float32[2] = clear_color_val[2];
        cv[0].color.float32[3] = clear_color_val[3];
        cv[1].depthStencil = {clear_depth_val, clear_stencil_val};

        VkRenderPassBeginInfo rpbi{};
        rpbi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpbi.renderPass = rp;
        rpbi.framebuffer = bb_fb;
        rpbi.renderArea = {{0, 0}, {bb_width, bb_height}};
        rpbi.clearValueCount = 2U;
        rpbi.pClearValues = cv.data();
        vk.cmd_begin_render_pass(frame_cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
        render_pass_active = true;
        frame_touched_rt = true;
        pending_clear_color = false;
        pending_clear_depth = false;
        bb_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        depth_layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        bound_pipeline = VK_NULL_HANDLE;
        bound_sampler_set = VK_NULL_HANDLE;
        return true;
    }

    void upload_static_buffer(BufferObj& b) {
        if (!gpu_ready || b.shadow.empty()) return;
        const bool fresh = (b.vk_buf == VK_NULL_HANDLE);
        if (b.vk_buf == VK_NULL_HANDLE) {
            VkBufferCreateInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bi.size = std::max<std::size_t>(b.shadow.size(), 16U);
            bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                       (b.is_index ? VK_BUFFER_USAGE_INDEX_BUFFER_BIT : VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
            bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            if (vk.create_buffer(h.device, &bi, nullptr, &b.vk_buf) != VK_SUCCESS) return;
            VkMemoryRequirements reqs{};
            vk.get_buffer_mem_reqs(h.device, b.vk_buf, &reqs);
            if (!alloc.alloc(reqs, false, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0U, b.vk_mem, &last_error) ||
                vk.bind_buffer_memory(h.device, b.vk_buf, b.vk_mem.memory, b.vk_mem.offset) != VK_SUCCESS) {
                vk.destroy_buffer(h.device, b.vk_buf, nullptr);
                b.vk_buf = VK_NULL_HANDLE;
                alloc.free(b.vk_mem);
                return;
            }
        }
        // Only the bytes changed since the last upload, recorded without waiting for the GPU.
        std::size_t lo = fresh ? 0U : std::min(b.dirty_lo, b.shadow.size());
        std::size_t hi = fresh ? b.shadow.size() : std::min(b.dirty_hi, b.shadow.size());
        if (hi <= lo) {
            b.dirty = false;
            return;
        }
        VkBuffer sbuf = VK_NULL_HANDLE;
        VkDeviceSize soff = 0;
        if (!stage_bytes(std::span<const std::uint8_t>(b.shadow.data() + lo, hi - lo), sbuf, soff)) return;
        const VkCommandBuffer cmd = begin_upload_locked(b.last_used_frame == frame_index);
        if (cmd == VK_NULL_HANDLE) return;
        VkMemoryBarrier mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        if (!fresh) {  // queued draws may still read the old bytes
            mb.srcAccessMask = 0U;
            mb.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            vk.cmd_pipeline_barrier(cmd, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0U, 1U, &mb,
                                    0U, nullptr, 0U, nullptr);
        }
        VkBufferCopy cpy{soff, lo, hi - lo};
        vk.cmd_copy_buffer(cmd, sbuf, b.vk_buf, 1U, &cpy);
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDEX_READ_BIT;
        vk.cmd_pipeline_barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0U, 1U, &mb,
                                0U, nullptr, 0U, nullptr);
        b.dirty = false;
        b.dirty_lo = SIZE_MAX;
        b.dirty_hi = 0U;
    }

    bool resolve_buffer_for_draw(std::uint64_t id, VkBuffer& out_buf, VkDeviceSize& out_off, bool& out_index32) {
        const auto it = buffers.find(id);
        if (it == buffers.end()) return false;
        BufferObj& b = it->second;
        out_index32 = b.is_index32;
        if (b.dynamic) {
            if (b.shadow.empty()) return false;
            if (b.cur < 0) {  // first draw, or a rename that could not get a copy
                const int ni = dyn_acquire(b);
                if (ni < 0) return false;
                b.cur = ni;
                dyn_write(b, b.insts[static_cast<std::size_t>(ni)], 0U, b.shadow.size());
            }
            BufferObj::DynInst& d = b.insts[static_cast<std::size_t>(b.cur)];
            d.last_used = frame_index;
            b.dirty = false;
            out_buf = d.buf;
            out_off = 0U;
            return true;
        }
        if (b.dirty || b.vk_buf == VK_NULL_HANDLE) upload_static_buffer(b);
        if (b.vk_buf == VK_NULL_HANDLE) return false;
        b.last_used_frame = frame_index;
        out_buf = b.vk_buf;
        out_off = 0U;
        return true;
    }

    bool create_texture_image_gpu(TextureObj& t, VkFormat vk_fmt, std::uint32_t w, std::uint32_t ht,
                                  std::uint32_t levels, int kind) {
        destroy_texture_gpu(t);
        t.kind = kind;
        t.width = std::max<std::uint32_t>(1U, w);
        t.height = std::max<std::uint32_t>(1U, ht);
        t.levels = std::max<std::uint32_t>(1U, levels);
        t.vk_fmt = vk_fmt;
        t.is_bc = (vk_fmt == VK_FORMAT_BC1_RGBA_UNORM_BLOCK || vk_fmt == VK_FORMAT_BC2_UNORM_BLOCK ||
                   vk_fmt == VK_FORMAT_BC3_UNORM_BLOCK);
        const std::uint32_t layers = (kind == 1) ? 6U : 1U;
        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.flags = (kind == 1) ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0U;
        ici.format = vk_fmt;
        ici.extent = {t.width, t.height, 1U};
        ici.mipLevels = t.levels;
        ici.arrayLayers = layers;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vk.create_image(h.device, &ici, nullptr, &t.img) != VK_SUCCESS) return false;
        VkMemoryRequirements reqs{};
        vk.get_image_mem_reqs(h.device, t.img, &reqs);
        if (!alloc.alloc(reqs, true, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0U, t.mem, &last_error) ||
            vk.bind_image_memory(h.device, t.img, t.mem.memory, t.mem.offset) != VK_SUCCESS) {
            destroy_texture_gpu(t);
            return false;
        }
        VkImageViewCreateInfo ivci{};
        ivci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ivci.image = t.img;
        ivci.viewType = (kind == 1) ? VK_IMAGE_VIEW_TYPE_CUBE : VK_IMAGE_VIEW_TYPE_2D;
        ivci.format = vk_fmt;
        ivci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0U, t.levels, 0U, layers};
        if (vk.create_image_view(h.device, &ivci, nullptr, &t.view) != VK_SUCCESS) {
            destroy_texture_gpu(t);
            return false;
        }
        t.layout = VK_IMAGE_LAYOUT_UNDEFINED;
        return true;
    }

    VkPipeline get_pipeline(const PipelineKey& key, const CompiledShader& vs, const CompiledShader& ps,
                            const VertexDeclObj* decl) {
        if (const auto it = pipeline_cache_map.find(key); it != pipeline_cache_map.end()) return it->second;

        std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vs.mod;
        stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = ps.mod;
        stages[1].pName = "main";

        std::vector<VkVertexInputAttributeDescription> attrs;
        std::array<bool, 16> used_bindings{};
        bool uses_dummy_binding = false;
        for (const d3d9sh::Semantic& in : vs.info.inputs) {
            const D3d9VertexElement* found = nullptr;
            if (decl != nullptr) {
                for (const D3d9VertexElement& el : decl->elems) {
                    if (el.usage == in.usage && el.usage_index == in.index) {
                        found = &el;
                        break;
                    }
                }
            }
            if (found != nullptr && found->stream < kDummyVertexBinding &&
                ((key.stream_present_mask >> found->stream) & 1U) != 0U) {
                bool dummy_swz = false;
                VkVertexInputAttributeDescription a{};
                a.location = in.reg;
                a.binding = found->stream;
                a.format = map_decl_type(found->type, report.bgra_vertex_format, dummy_swz);
                a.offset = found->offset;
                attrs.push_back(a);
                used_bindings[found->stream] = true;
            } else {
                VkVertexInputAttributeDescription a{};
                a.location = in.reg;
                a.binding = kDummyVertexBinding;
                a.format = VK_FORMAT_R32G32B32A32_SFLOAT;
                a.offset = 0U;
                attrs.push_back(a);
                uses_dummy_binding = true;
            }
        }
        std::vector<VkVertexInputBindingDescription> bindings;
        for (std::uint32_t s = 0; s < kDummyVertexBinding; ++s) {
            if (!used_bindings[s]) continue;
            VkVertexInputBindingDescription b{};
            b.binding = s;
            b.stride = key.stream_strides[s];
            b.inputRate = ((key.instance_mask >> s) & 1U) != 0U ? VK_VERTEX_INPUT_RATE_INSTANCE
                                                                : VK_VERTEX_INPUT_RATE_VERTEX;
            bindings.push_back(b);
        }
        if (uses_dummy_binding) {
            bindings.push_back({kDummyVertexBinding, 0U, VK_VERTEX_INPUT_RATE_INSTANCE});
        }

        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vi.vertexBindingDescriptionCount = static_cast<std::uint32_t>(bindings.size());
        vi.pVertexBindingDescriptions = bindings.data();
        vi.vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attrs.size());
        vi.pVertexAttributeDescriptions = attrs.data();

        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = map_topology(key.prim_type);

        VkPipelineViewportStateCreateInfo vp{};
        vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vp.viewportCount = 1U;
        vp.scissorCount = 1U;

        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode = (key.fill_mode == 2U && h.fill_mode_non_solid)   ? VK_POLYGON_MODE_LINE
                         : (key.fill_mode == 1U && h.fill_mode_non_solid) ? VK_POLYGON_MODE_POINT
                                                                          : VK_POLYGON_MODE_FILL;
        if (key.cull_mode == 1U) {
            rs.cullMode = VK_CULL_MODE_NONE;
            rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        } else if (key.cull_mode == 2U) {  // D3DCULL_CW
            rs.cullMode = VK_CULL_MODE_BACK_BIT;
            rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        } else {                           // D3DCULL_CCW
            rs.cullMode = VK_CULL_MODE_BACK_BIT;
            rs.frontFace = VK_FRONT_FACE_CLOCKWISE;
        }
        rs.depthBiasEnable = key.depth_bias_enable ? VK_TRUE : VK_FALSE;
        rs.lineWidth = 1.0F;

        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo ds{};
        ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.depthTestEnable = key.depth_enable ? VK_TRUE : VK_FALSE;
        ds.depthWriteEnable = (key.depth_enable && key.depth_write) ? VK_TRUE : VK_FALSE;
        ds.depthCompareOp = map_compare_op(key.depth_func);

        VkPipelineColorBlendAttachmentState cba{};
        cba.blendEnable = key.blend_enable ? VK_TRUE : VK_FALSE;
        cba.srcColorBlendFactor = map_blend_factor(key.src_blend);
        cba.dstColorBlendFactor = map_blend_factor(key.dst_blend);
        cba.colorBlendOp = map_blend_op(key.blend_op);
        if (key.sep_alpha) {
            cba.srcAlphaBlendFactor = map_blend_factor(key.src_alpha);
            cba.dstAlphaBlendFactor = map_blend_factor(key.dst_alpha);
            cba.alphaBlendOp = map_blend_op(key.op_alpha);
        } else {
            cba.srcAlphaBlendFactor = cba.srcColorBlendFactor;
            cba.dstAlphaBlendFactor = cba.dstColorBlendFactor;
            cba.alphaBlendOp = cba.colorBlendOp;
        }
        cba.colorWriteMask = static_cast<VkColorComponentFlags>(key.color_mask & 0xFU);

        VkPipelineColorBlendStateCreateInfo cb{};
        cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cb.attachmentCount = 1U;
        cb.pAttachments = &cba;

        static constexpr std::array<VkDynamicState, 4> kDyn{
            VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
            VK_DYNAMIC_STATE_DEPTH_BIAS, VK_DYNAMIC_STATE_BLEND_CONSTANTS};
        VkPipelineDynamicStateCreateInfo dyn{};
        dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dyn.dynamicStateCount = static_cast<std::uint32_t>(kDyn.size());
        dyn.pDynamicStates = kDyn.data();

        VkRenderPass compat_rp = get_render_pass({key.color_fmt, key.depth_fmt, false, false});
        if (compat_rp == VK_NULL_HANDLE) return VK_NULL_HANDLE;

        VkGraphicsPipelineCreateInfo gp{};
        gp.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        gp.stageCount = 2U;
        gp.pStages = stages.data();
        gp.pVertexInputState = &vi;
        gp.pInputAssemblyState = &ia;
        gp.pViewportState = &vp;
        gp.pRasterizationState = &rs;
        gp.pMultisampleState = &ms;
        gp.pDepthStencilState = &ds;
        gp.pColorBlendState = &cb;
        gp.pDynamicState = &dyn;
        gp.layout = pipeline_layout;
        gp.renderPass = compat_rp;
        gp.subpass = 0U;

        VkPipeline pip = VK_NULL_HANDLE;
        if (vk.create_graphics_pipelines(h.device, pipeline_cache, 1U, &gp, nullptr, &pip) != VK_SUCCESS ||
            pip == VK_NULL_HANDLE) {
            last_error = "vkCreateGraphicsPipelines failed";
            return VK_NULL_HANDLE;
        }
        pipeline_cache_map.emplace(key, pip);
        return pip;
    }

    bool compile_shader_if_needed(CompiledShader& cs) {
        if (cs.attempted) return cs.valid;
        cs.attempted = true;
        if (cs.bytecode.empty()) return false;
        if (!d3d9sh::translate(cs.bytecode, cs.info)) {
            last_error = "d3d9sh::translate failed: " + cs.info.error;
            return false;
        }
        if (!compiler) return false;
        std::vector<std::uint32_t> spv;
        std::string log;
        const auto stage = cs.is_pixel ? wse::asset::SpirvStage::Fragment : wse::asset::SpirvStage::Vertex;
        if (!compiler->compile_vulkan_glsl_to_spirv(cs.info.glsl, stage, "main", spv, nullptr, &log) || spv.empty()) {
            last_error = "glslang failed: " + log;
            return false;
        }
        if (gpu_ready) {
            VkShaderModuleCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            ci.codeSize = spv.size() * sizeof(std::uint32_t);
            ci.pCode = spv.data();
            if (vk.create_shader_module(h.device, &ci, nullptr, &cs.mod) == VK_SUCCESS && cs.mod != VK_NULL_HANDLE) {
                cs.valid = true;
            }
        } else {
            cs.valid = true;
        }
        // Release raw bytecode once compiled to save RAM on mobile.
        cs.bytecode.clear();
        cs.bytecode.shrink_to_fit();
        return cs.valid;
    }

    bool prepare_draw(const D3d9DrawState& st, std::uint32_t prim_type,
                      std::uint32_t& out_instance_count) {
        if (!gpu_ready) return false;
        const auto vs_hid = shader_id_to_hash.find(st.vs);
        const auto ps_hid = shader_id_to_hash.find(st.ps);
        if (vs_hid == shader_id_to_hash.end() || ps_hid == shader_id_to_hash.end()) return false;
        const auto vs_it = shader_by_hash.find(vs_hid->second);
        const auto ps_it = shader_by_hash.find(ps_hid->second);
        if (vs_it == shader_by_hash.end() || !compile_shader_if_needed(vs_it->second) ||
            ps_it == shader_by_hash.end() || !compile_shader_if_needed(ps_it->second)) {
            return false;
        }
        const CompiledShader& vs = vs_it->second;
        const CompiledShader& ps = ps_it->second;
        const VertexDeclObj* decl = nullptr;
        if (const auto dit = decls.find(st.decl); dit != decls.end()) decl = &dit->second;

        // Resolve active vertex buffers BEFORE opening the render pass (static uploads may submit).
        std::array<VkBuffer, 16> stream_bufs{};
        std::array<VkDeviceSize, 16> stream_offs{};
        std::uint16_t present_mask = 0;
        std::uint16_t inst_mask = 0;
        out_instance_count = 1U;
        for (std::uint32_t s = 0; s < kDummyVertexBinding; ++s) {
            const D3d9StreamBinding& sb = st.streams[s];
            if (sb.buffer == 0) continue;
            VkBuffer vkb = VK_NULL_HANDLE;
            VkDeviceSize vko = 0;
            bool dummy_i32 = false;
            if (resolve_buffer_for_draw(sb.buffer, vkb, vko, dummy_i32)) {
                stream_bufs[s] = vkb;
                stream_offs[s] = vko + sb.offset;
                present_mask |= static_cast<std::uint16_t>(1U << s);
            }
            if ((sb.freq & 0x40000000U) != 0U) {  // D3DSTREAMSOURCE_INDEXEDDATA
                out_instance_count = std::max<std::uint32_t>(1U, sb.freq & 0x3FFFFFFFU);
            } else if ((sb.freq & 0x80000000U) != 0U) {  // D3DSTREAMSOURCE_INSTANCEDATA
                inst_mask |= static_cast<std::uint16_t>(1U << s);
            }
        }

        // Compute BGRA swizzle mask if driver lacks B8G8R8A8_UNORM vertex fetch.
        std::uint32_t bgra_swz_mask = 0U;
        if (!report.bgra_vertex_format && decl != nullptr) {
            for (const d3d9sh::Semantic& in : vs.info.inputs) {
                for (const D3d9VertexElement& el : decl->elems) {
                    if (el.usage == in.usage && el.usage_index == in.index) {
                        bool swz = false;
                        (void)map_decl_type(el.type, false, swz);
                        if (swz && in.reg < 24U) bgra_swz_mask |= (1U << in.reg);
                        break;
                    }
                }
            }
        }

        if (!ensure_render_pass_active()) return false;

        PipelineKey pkey{};
        pkey.vs_hash = vs_hid->second;
        pkey.ps_hash = ps_hid->second;
        pkey.decl_hash = decl != nullptr ? decl->hash : 0U;
        for (std::uint32_t s = 0; s < 16U; ++s) pkey.stream_strides[s] = st.streams[s].stride;
        pkey.instance_mask = inst_mask;
        pkey.stream_present_mask = present_mask;
        pkey.color_fmt = VK_FORMAT_R8G8B8A8_UNORM;
        pkey.depth_fmt = default_depth_fmt;
        pkey.prim_type = static_cast<std::uint8_t>(prim_type);
        pkey.cull_mode = static_cast<std::uint8_t>(st.rs[22]);
        pkey.fill_mode = static_cast<std::uint8_t>(st.rs[8]);
        pkey.depth_enable = st.rs[7] != 0U ? 1U : 0U;
        pkey.depth_write = st.rs[14] != 0U ? 1U : 0U;
        pkey.depth_func = static_cast<std::uint8_t>(st.rs[23]);
        pkey.depth_bias_enable = (st.rs[175] != 0U || st.rs[195] != 0U) ? 1U : 0U;
        pkey.blend_enable = st.rs[27] != 0U ? 1U : 0U;
        pkey.src_blend = static_cast<std::uint8_t>(st.rs[19]);
        pkey.dst_blend = static_cast<std::uint8_t>(st.rs[20]);
        pkey.blend_op = static_cast<std::uint8_t>(st.rs[171]);
        pkey.sep_alpha = st.rs[206] != 0U ? 1U : 0U;
        pkey.src_alpha = static_cast<std::uint8_t>(st.rs[207]);
        pkey.dst_alpha = static_cast<std::uint8_t>(st.rs[208]);
        pkey.op_alpha = static_cast<std::uint8_t>(st.rs[209]);
        pkey.color_mask = static_cast<std::uint8_t>(st.rs[168] & 0xFU);

        VkPipeline pip = get_pipeline(pkey, vs, ps, decl);
        if (pip == VK_NULL_HANDLE) return false;
        if (pip != bound_pipeline) {
            vk.cmd_bind_pipeline(frame_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pip);
            bound_pipeline = pip;
        }

        // Dynamic viewport, scissor, depth bias, blend constants.
        const std::uint32_t vpw = st.viewport.width != 0U ? st.viewport.width : bb_width;
        const std::uint32_t vph = st.viewport.height != 0U ? st.viewport.height : bb_height;
        VkViewport vvp{};
        vvp.x = static_cast<float>(st.viewport.x);
        vvp.y = static_cast<float>(st.viewport.y);
        vvp.width = static_cast<float>(vpw);
        vvp.height = static_cast<float>(vph);
        vvp.minDepth = st.viewport.min_z;
        vvp.maxDepth = st.viewport.max_z;
        vk.cmd_set_viewport(frame_cmd, 0U, 1U, &vvp);

        VkRect2D sci{};
        if (st.rs[174] != 0U) {  // D3DRS_SCISSORTESTENABLE
            const std::int32_t l = std::clamp<std::int32_t>(st.scissor.left, 0, static_cast<std::int32_t>(bb_width));
            const std::int32_t t = std::clamp<std::int32_t>(st.scissor.top, 0, static_cast<std::int32_t>(bb_height));
            const std::int32_t r = std::clamp<std::int32_t>(st.scissor.right, l, static_cast<std::int32_t>(bb_width));
            const std::int32_t b = std::clamp<std::int32_t>(st.scissor.bottom, t, static_cast<std::int32_t>(bb_height));
            sci = {{l, t}, {static_cast<std::uint32_t>(r - l), static_cast<std::uint32_t>(b - t)}};
        } else {
            sci = {{static_cast<std::int32_t>(st.viewport.x), static_cast<std::int32_t>(st.viewport.y)}, {vpw, vph}};
        }
        vk.cmd_set_scissor(frame_cmd, 0U, 1U, &sci);

        if (vk.cmd_set_depth_bias != nullptr) {
            float slope = 0.0F, bias = 0.0F;
            std::memcpy(&slope, &st.rs[175], 4);
            std::memcpy(&bias, &st.rs[195], 4);
            vk.cmd_set_depth_bias(frame_cmd, bias, 0.0F, slope);
        }
        if (vk.cmd_set_blend_constants != nullptr) {
            const std::uint32_t bf = st.rs[193];
            const float c[4] = {((bf >> 16) & 0xFFU) / 255.0F, ((bf >> 8) & 0xFFU) / 255.0F,
                                (bf & 0xFFU) / 255.0F, ((bf >> 24) & 0xFFU) / 255.0F};
            vk.cmd_set_blend_constants(frame_cmd, c);
        }

        // Write VS + PS UBOs into the dynamic ring buffer.
        VkDeviceSize vs_off = 0, ps_off = 0;
        if (!alloc_ubo_pair(vs_off, ps_off)) return false;

        StageUboData vs_ubo{};
        std::memcpy(vs_ubo.c, st.vs_const.c.data(), sizeof(vs_ubo.c));
        std::memcpy(vs_ubo.i, st.vs_const.i.data(), sizeof(vs_ubo.i));
        std::memcpy(vs_ubo.b, st.vs_const.b.data(), sizeof(vs_ubo.b));
        vs_ubo.posfix[0] = vpw > 0U ? (1.0F / static_cast<float>(vpw)) : 0.0F;
        vs_ubo.posfix[1] = vph > 0U ? (1.0F / static_cast<float>(vph)) : 0.0F;
        vs_ubo.posfix[2] = static_cast<float>(bgra_swz_mask);
        vs_ubo.posfix[3] = 0.0F;

        StageUboData ps_ubo{};
        std::memcpy(ps_ubo.c, st.ps_const.c.data(), sizeof(ps_ubo.c));
        std::memcpy(ps_ubo.i, st.ps_const.i.data(), sizeof(ps_ubo.i));
        std::memcpy(ps_ubo.b, st.ps_const.b.data(), sizeof(ps_ubo.b));
        if (st.rs[15] != 0U && st.rs[25] >= 1U && st.rs[25] <= 7U) {  // D3DRS_ALPHATESTENABLE
            ps_ubo.posfix[0] = static_cast<float>(st.rs[25]);
            ps_ubo.posfix[1] = static_cast<float>(st.rs[24] & 0xFFU);
        }
        std::memcpy(static_cast<std::uint8_t*>(ubo_ring.mem.mapped) + vs_off, &vs_ubo, sizeof(vs_ubo));
        std::memcpy(static_cast<std::uint8_t*>(ubo_ring.mem.mapped) + ps_off, &ps_ubo, sizeof(ps_ubo));
        alloc.flush(ubo_ring.mem, vs_off, ubo_stride * 2U);

        const std::array<std::uint32_t, 2> dyn_offs{static_cast<std::uint32_t>(vs_off),
                                                    static_cast<std::uint32_t>(ps_off)};
        vk.cmd_bind_desc_sets(frame_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                              0U, 1U, &ubo_desc_set, 2U, dyn_offs.data());

        VkDescriptorSet smp_set = get_sampler_set(st, vs.info, ps.info);
        if (smp_set != VK_NULL_HANDLE && smp_set != bound_sampler_set) {
            vk.cmd_bind_desc_sets(frame_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                                  1U, 1U, &smp_set, 0U, nullptr);
            bound_sampler_set = smp_set;
        }

        // Bind active vertex streams + dummy fallback stream 15.
        for (std::uint32_t s = 0; s < kDummyVertexBinding; ++s) {
            if (((present_mask >> s) & 1U) != 0U) {
                vk.cmd_bind_vertex_buffers(frame_cmd, s, 1U, &stream_bufs[s], &stream_offs[s]);
            }
        }
        const VkDeviceSize zero_off = 0U;
        vk.cmd_bind_vertex_buffers(frame_cmd, kDummyVertexBinding, 1U, &dummy_vb, &zero_off);
        return true;
    }
};

VulkanD3d9Backend::VulkanD3d9Backend(std::shared_ptr<VulkanDriverLoader> loader, VulkanD3d9Config config)
    : impl_(std::make_unique<Impl>(std::move(loader), std::move(config))) {}

VulkanD3d9Backend::~VulkanD3d9Backend() = default;

bool VulkanD3d9Backend::init(std::uint32_t width, std::uint32_t height, std::uint32_t color_fmt,
                             std::uint32_t depth_fmt, std::string* error_out) {
    (void)color_fmt;
    (void)depth_fmt;
    const std::lock_guard<std::mutex> lock(impl_->api_mutex);
    return impl_->init(width, height, error_out);
}

bool VulkanD3d9Backend::reset(std::uint32_t width, std::uint32_t height, std::uint32_t color_fmt,
                              std::uint32_t depth_fmt, std::string* error_out) {
    (void)color_fmt;
    (void)depth_fmt;
    const std::lock_guard<std::mutex> lock(impl_->api_mutex);
    if (!impl_->initialized) return impl_->init(width, height, error_out);
    if (!impl_->gpu_ready) {
        impl_->bb_width = std::max<std::uint32_t>(1U, width);
        impl_->bb_height = std::max<std::uint32_t>(1U, height);
        return true;
    }
    impl_->flush_frame_if_recording();
    if (impl_->vk.device_wait_idle != nullptr) (void)impl_->vk.device_wait_idle(impl_->h.device);
    if (!impl_->create_backbuffer(width, height, error_out)) return false;
#ifdef __ANDROID__
    if (impl_->active_window != nullptr) {
        if (impl_->android_path == Impl::AndroidPresentPath::ahb_surface_control) {
            if (!impl_->setup_ahb_surface_control_locked()) {
                impl_->android_path = Impl::AndroidPresentPath::native_window_cpu;
                impl_->report.presentation_path = "native_window_cpu_fallback(ANativeWindow_lock)";
            }
        } else if (impl_->android_path == Impl::AndroidPresentPath::native_window_cpu) {
            impl_->cpu_win_w = 0U;
            impl_->cpu_win_h = 0U;
        }
    }
#endif
    return true;
}

const D3d9DriverReport& VulkanD3d9Backend::report() const noexcept { return impl_->report; }

void VulkanD3d9Backend::create_buffer(std::uint64_t id, bool is_index, std::uint32_t size,
                                      bool dynamic, std::uint32_t fmt) {
    const std::lock_guard<std::mutex> lock(impl_->api_mutex);
    const auto old = impl_->buffers.find(id);
    if (old != impl_->buffers.end()) {
        if (impl_->gpu_ready && (old->second.vk_buf != VK_NULL_HANDLE || !old->second.insts.empty())) {
            impl_->wait_all_locked();
            impl_->destroy_buffer_gpu(old->second);
        }
        impl_->buffers.erase(old);
    }
    Impl::BufferObj b{};
    b.is_index = is_index;
    b.is_index32 = (fmt == 102U);
    b.dynamic = dynamic;
    b.size = size;
    b.shadow.assign(size, 0U);
    b.dirty = true;
    impl_->buffers.emplace(id, std::move(b));
}

void VulkanD3d9Backend::update_buffer(std::uint64_t id, std::uint32_t offset,
                                      std::span<const std::uint8_t> bytes, std::uint32_t lock_flags) {
    const std::lock_guard<std::mutex> lock(impl_->api_mutex);
    const auto it = impl_->buffers.find(id);
    if (it == impl_->buffers.end() || bytes.empty()) return;
    Impl::BufferObj& b = it->second;
    const std::size_t end = static_cast<std::size_t>(offset) + bytes.size();
    if (b.shadow.size() < end) b.shadow.resize(end, 0U);
    std::memcpy(b.shadow.data() + offset, bytes.data(), bytes.size());
    if (!b.dynamic) {
        b.dirty = true;
        b.dirty_lo = std::min<std::size_t>(b.dirty_lo, offset);
        b.dirty_hi = std::max(b.dirty_hi, end);
        return;
    }
    // Dynamic: copy only what changed into a GPU-visible copy now, as a D3D9 driver would.
    if (!impl_->gpu_ready || b.cur < 0) return;  // the next draw copies the whole buffer
    Impl::BufferObj::DynInst& d = b.insts[static_cast<std::size_t>(b.cur)];
    if ((lock_flags & kLockNoOverwrite) != 0U || !impl_->dyn_busy(d)) {
        // NOOVERWRITE: the game promises queued draws do not read these bytes.
        impl_->dyn_write(b, d, offset, bytes.size());
        return;
    }
    const int ni = impl_->dyn_acquire(b);
    if (ni < 0) {
        b.cur = -1;
        return;
    }
    b.cur = ni;
    Impl::BufferObj::DynInst& fresh = b.insts[static_cast<std::size_t>(ni)];
    if ((lock_flags & kLockDiscard) != 0U) {
        impl_->dyn_write(b, fresh, offset, bytes.size());  // a discarded buffer's other bytes are undefined
    } else {
        impl_->dyn_write(b, fresh, 0U, b.shadow.size());
    }
}

void VulkanD3d9Backend::destroy_buffer(std::uint64_t id) {
    const std::lock_guard<std::mutex> lock(impl_->api_mutex);
    const auto it = impl_->buffers.find(id);
    if (it == impl_->buffers.end()) return;
    if (impl_->gpu_ready && (it->second.vk_buf != VK_NULL_HANDLE || !it->second.insts.empty())) {
        impl_->wait_all_locked();
        impl_->destroy_buffer_gpu(it->second);
    }
    impl_->buffers.erase(it);
}

void VulkanD3d9Backend::create_texture(std::uint64_t id, int kind, std::uint32_t width, std::uint32_t height,
                                       std::uint32_t depth, std::uint32_t levels, std::uint32_t fmt,
                                       std::uint32_t usage) {
    const std::lock_guard<std::mutex> lock(impl_->api_mutex);
    const auto old = impl_->textures.find(id);
    if (old != impl_->textures.end()) {
        if (impl_->gpu_ready) impl_->retire_texture_image_locked(old->second);
        impl_->textures.erase(old);
    }
    Impl::TextureObj t{};
    t.kind = kind;
    t.width = std::max<std::uint32_t>(1U, width);
    t.height = std::max<std::uint32_t>(1U, height);
    t.depth = std::max<std::uint32_t>(1U, depth);
    t.levels = std::max<std::uint32_t>(1U, levels);
    t.d3d_fmt = fmt;
    t.usage = usage;
    impl_->textures.emplace(id, std::move(t));
}

void VulkanD3d9Backend::upload_texture_level(std::uint64_t id, std::uint32_t face, std::uint32_t level,
                                             std::span<const std::uint8_t> bytes, std::uint32_t row_pitch) {
    const std::lock_guard<std::mutex> lock(impl_->api_mutex);
    if (!impl_->gpu_ready) return;
    const auto it = impl_->textures.find(id);
    if (it == impl_->textures.end()) return;
    Impl::TextureObj& t = it->second;
    if (level >= t.levels) return;
    const std::uint32_t layers = (t.kind == 1) ? 6U : 1U;
    if (face >= layers) return;

    const bool use_bc = impl_->config.prefer_bc_compression && impl_->report.texture_compression_bc &&
                        is_dxt_d3dfmt(t.d3d_fmt);
    const VkFormat vk_fmt = use_bc ? dxt_d3dfmt_to_vk(t.d3d_fmt) : VK_FORMAT_R8G8B8A8_UNORM;
    if (t.img == VK_NULL_HANDLE) {
        if (!impl_->create_texture_image_gpu(t, vk_fmt, t.width, t.height, t.levels, t.kind)) return;
    }

    const std::uint32_t lw = std::max<std::uint32_t>(1U, t.width >> level);
    const std::uint32_t lh = std::max<std::uint32_t>(1U, t.height >> level);
    std::vector<std::uint8_t> conv;
    std::span<const std::uint8_t> payload = bytes;
    if (!t.is_bc) {
        convert_d3d9_level_to_rgba8(t.d3d_fmt, lw, lh, bytes, row_pitch, conv);
        payload = conv;
    }
    VkBuffer sbuf = VK_NULL_HANDLE;
    VkDeviceSize soff = 0;
    if (payload.empty() || !impl_->stage_bytes(payload, sbuf, soff)) return;
    // Recorded with this frame (no wait): before its draws unless one already sampled the texture.
    const VkCommandBuffer cmd = impl_->begin_upload_locked(t.last_used_frame == impl_->frame_index);
    if (cmd == VK_NULL_HANDLE) return;

    VkImageMemoryBarrier bar{};
    bar.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    bar.oldLayout = t.layout;
    bar.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bar.image = t.img;
    bar.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0U, t.levels, 0U, layers};
    bar.srcAccessMask = 0U;  // write-after-read: an execution dependency is enough
    bar.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    impl_->vk.cmd_pipeline_barrier(cmd,
                                   (t.layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
                                       ? (VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT)
                                       : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                   VK_PIPELINE_STAGE_TRANSFER_BIT, 0U, 0U, nullptr, 0U, nullptr, 1U, &bar);

    VkBufferImageCopy cpy{};
    cpy.bufferOffset = soff;
    cpy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, face, 1U};
    cpy.imageExtent = {lw, lh, 1U};
    impl_->vk.cmd_copy_buffer_to_image(cmd, sbuf, t.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1U, &cpy);

    bar.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    bar.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    bar.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    impl_->vk.cmd_pipeline_barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                   VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                   0U, 0U, nullptr, 0U, nullptr, 1U, &bar);
    t.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

void VulkanD3d9Backend::upload_texture_dds(std::uint64_t id, std::span<const std::uint8_t> dds_bytes) {
    const std::lock_guard<std::mutex> lock(impl_->api_mutex);
    if (!impl_->gpu_ready || dds_bytes.empty()) return;
    const auto it = impl_->textures.find(id);
    if (it == impl_->textures.end()) return;

    wse::asset::DdsImageView view;
    const auto byte_span = std::span<const std::byte>(reinterpret_cast<const std::byte*>(dds_bytes.data()), dds_bytes.size());
    std::string dds_err;
    if (!wse::asset::parse_dds(byte_span, view, &dds_err) || view.mip_count == 0U || view.face_count == 0U) {
        impl_->last_error = "parse_dds failed: " + dds_err;
        return;
    }

    const int kind = (view.is_cubemap && view.face_count >= 6U) ? 1 : 0;
    const std::uint32_t layers = (kind == 1) ? 6U : 1U;
    const VkFormat bc_fmt = dds_bc_to_vk(view.format);
    const bool use_bc = impl_->config.prefer_bc_compression && impl_->report.texture_compression_bc &&
                        bc_fmt != VK_FORMAT_UNDEFINED;

    // Optional P10.7 mip cap when decoding to RGBA8 fallback.
    std::uint32_t first_mip = 0U;
    if (!use_bc && impl_->config.max_texture_size > 0U) {
        while (first_mip + 1U < view.mip_count &&
               std::max(view.width >> first_mip, view.height >> first_mip) > impl_->config.max_texture_size) {
            ++first_mip;
        }
    }
    const VkFormat vk_fmt = use_bc ? bc_fmt : VK_FORMAT_R8G8B8A8_UNORM;

    Impl::TextureObj& t = it->second;
    impl_->retire_texture_image_locked(t);  // a new image replaces one queued frames may sample
    std::uint32_t out_w = std::max<std::uint32_t>(1U, view.width >> first_mip);
    std::uint32_t out_h = std::max<std::uint32_t>(1U, view.height >> first_mip);
    std::uint32_t out_levels = view.mip_count - first_mip;
    bool img_ok = impl_->create_texture_image_gpu(t, vk_fmt, out_w, out_h, out_levels, kind);
    // On RGBA8 fallback, gracefully drop top mip(s) if device memory runs tight instead of dropping the texture.
    while (!img_ok && !use_bc && first_mip + 1U < view.mip_count) {
        ++first_mip;
        out_w = std::max<std::uint32_t>(1U, view.width >> first_mip);
        out_h = std::max<std::uint32_t>(1U, view.height >> first_mip);
        out_levels = view.mip_count - first_mip;
        img_ok = impl_->create_texture_image_gpu(t, vk_fmt, out_w, out_h, out_levels, kind);
    }
    if (!img_ok) return;
    if (first_mip > 0U && impl_->last_error.find("vkAllocateMemory") != std::string::npos) {
        impl_->last_error.clear();
    }

    // Pack all faces and mip levels into one staging upload.
    std::vector<std::uint8_t> packed;
    std::vector<VkBufferImageCopy> copies;
    copies.reserve(static_cast<std::size_t>(layers) * out_levels);
    std::vector<std::uint8_t> rgba_tmp;
    for (std::uint32_t face = 0; face < layers; ++face) {
        for (std::uint32_t m = 0; m < out_levels; ++m) {
            const std::uint32_t src_mip = first_mip + m;
            const std::size_t sidx = static_cast<std::size_t>(face) * view.mip_count + src_mip;
            if (sidx >= view.slices.size()) continue;
            const wse::asset::DdsMipSlice& sl = view.slices[sidx];
            const std::size_t aligned_off = static_cast<std::size_t>(align_up(packed.size(), 16U));
            packed.resize(aligned_off, 0U);
            if (use_bc) {
                const auto* src_p = reinterpret_cast<const std::uint8_t*>(sl.data.data());
                packed.insert(packed.end(), src_p, src_p + sl.data.size());
            } else {
                rgba_tmp.clear();
                if (!wse::asset::decode_dds_mip_rgba8(view, src_mip, face, rgba_tmp, &dds_err)) continue;
                packed.insert(packed.end(), rgba_tmp.begin(), rgba_tmp.end());
            }
            VkBufferImageCopy cpy{};
            cpy.bufferOffset = aligned_off;
            cpy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, m, face, 1U};
            cpy.imageExtent = {std::max<std::uint32_t>(1U, sl.width), std::max<std::uint32_t>(1U, sl.height), 1U};
            copies.push_back(cpy);
        }
    }
    VkBuffer sbuf = VK_NULL_HANDLE;
    VkDeviceSize soff = 0;
    if (copies.empty() || !impl_->stage_bytes(packed, sbuf, soff)) return;
    for (VkBufferImageCopy& c : copies) c.bufferOffset += soff;
    // A new image: no draw of this frame can have sampled it, so it goes before the draws.
    const VkCommandBuffer cmd = impl_->begin_upload_locked(false);
    if (cmd == VK_NULL_HANDLE) return;

    VkImageMemoryBarrier bar{};
    bar.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    bar.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    bar.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bar.image = t.img;
    bar.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0U, out_levels, 0U, layers};
    bar.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    impl_->vk.cmd_pipeline_barrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                   VK_PIPELINE_STAGE_TRANSFER_BIT, 0U, 0U, nullptr, 0U, nullptr, 1U, &bar);
    impl_->vk.cmd_copy_buffer_to_image(cmd, sbuf, t.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                       static_cast<std::uint32_t>(copies.size()), copies.data());
    bar.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    bar.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    bar.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    impl_->vk.cmd_pipeline_barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                   VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                   0U, 0U, nullptr, 0U, nullptr, 1U, &bar);
    t.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

void VulkanD3d9Backend::destroy_texture(std::uint64_t id) {
    const std::lock_guard<std::mutex> lock(impl_->api_mutex);
    const auto it = impl_->textures.find(id);
    if (it == impl_->textures.end()) return;
    // Waits for queued frames and clears the sampler sets that name its view.
    if (impl_->gpu_ready) impl_->retire_texture_image_locked(it->second);
    impl_->textures.erase(it);
}

void VulkanD3d9Backend::create_surface(std::uint64_t id, std::uint32_t width, std::uint32_t height,
                                       std::uint32_t fmt, std::uint32_t usage,
                                       std::uint64_t parent_tex, std::uint32_t face, std::uint32_t level) {
    (void)id; (void)width; (void)height; (void)fmt; (void)usage; (void)parent_tex; (void)face; (void)level;
}

void VulkanD3d9Backend::destroy_surface(std::uint64_t id) { (void)id; }

void VulkanD3d9Backend::create_vertex_decl(std::uint64_t id, std::span<const D3d9VertexElement> elems) {
    const std::lock_guard<std::mutex> lock(impl_->api_mutex);
    Impl::VertexDeclObj d;
    d.elems.assign(elems.begin(), elems.end());
    std::uint64_t h = 14695981039346656037ULL;
    for (const D3d9VertexElement& e : d.elems) {
        const std::uint64_t packed = static_cast<std::uint64_t>(e.stream) |
                                     (static_cast<std::uint64_t>(e.offset) << 16) |
                                     (static_cast<std::uint64_t>(e.type) << 32) |
                                     (static_cast<std::uint64_t>(e.usage) << 40) |
                                     (static_cast<std::uint64_t>(e.usage_index) << 48);
        hash_mix(h, packed);
    }
    d.hash = h;
    impl_->decls[id] = std::move(d);
}

void VulkanD3d9Backend::destroy_vertex_decl(std::uint64_t id) {
    const std::lock_guard<std::mutex> lock(impl_->api_mutex);
    impl_->decls.erase(id);
}

void VulkanD3d9Backend::create_shader(std::uint64_t id, bool is_pixel, std::span<const std::uint8_t> bytecode) {
    const std::lock_guard<std::mutex> lock(impl_->api_mutex);
    if (bytecode.empty()) return;
    const std::uint64_t sh_hash = fnv1a_bytes(bytecode) ^ (is_pixel ? 0x9E3779B97F4A7C15ULL : 0ULL);
    impl_->shader_id_to_hash[id] = sh_hash;
    if (impl_->shader_by_hash.find(sh_hash) != impl_->shader_by_hash.end()) return;

    Impl::CompiledShader cs{};
    cs.bytecode.assign(bytecode.begin(), bytecode.end());
    cs.is_pixel = is_pixel;
    if (impl_->config.eager_shader_compile || !impl_->gpu_ready) {
        (void)impl_->compile_shader_if_needed(cs);
    }
    impl_->shader_by_hash.emplace(sh_hash, std::move(cs));
}

void VulkanD3d9Backend::destroy_shader(std::uint64_t id) {
    const std::lock_guard<std::mutex> lock(impl_->api_mutex);
    impl_->shader_id_to_hash.erase(id);
}

void VulkanD3d9Backend::clear(const D3d9DrawState& state, std::uint32_t flags, std::uint32_t argb,
                              float z, std::uint32_t stencil, std::span<const D3d9Rect> rects) {
    const std::lock_guard<std::mutex> lock(impl_->api_mutex);
    if (!impl_->gpu_ready || flags == 0U) return;
    const float r = static_cast<float>((argb >> 16) & 0xFFU) / 255.0F;
    const float g = static_cast<float>((argb >> 8) & 0xFFU) / 255.0F;
    const float b = static_cast<float>(argb & 0xFFU) / 255.0F;
    const float a = static_cast<float>((argb >> 24) & 0xFFU) / 255.0F;
    const bool clr_c = (flags & 1U) != 0U;
    const bool clr_d = (flags & 2U) != 0U;
    const bool clr_s = (flags & 4U) != 0U && has_stencil_aspect(impl_->default_depth_fmt);

    const std::uint32_t vpw = state.viewport.width != 0U ? state.viewport.width : impl_->bb_width;
    const std::uint32_t vph = state.viewport.height != 0U ? state.viewport.height : impl_->bb_height;
    const bool full_area = rects.empty() && state.rs[174] == 0U &&
                           state.viewport.x == 0U && state.viewport.y == 0U &&
                           vpw == impl_->bb_width && vph == impl_->bb_height;

    if (!impl_->render_pass_active && full_area && !impl_->frame_touched_rt) {
        if (clr_c) {
            impl_->clear_color_val = {r, g, b, a};
            impl_->pending_clear_color = true;
        }
        if (clr_d || clr_s) {
            impl_->clear_depth_val = z;
            impl_->clear_stencil_val = stencil;
            impl_->pending_clear_depth = true;
        }
        return;
    }

    if (!impl_->ensure_render_pass_active()) return;
    std::array<VkClearAttachment, 2> atts{};
    std::uint32_t att_cnt = 0;
    if (clr_c) {
        atts[att_cnt].aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        atts[att_cnt].colorAttachment = 0U;
        atts[att_cnt].clearValue.color.float32[0] = r;
        atts[att_cnt].clearValue.color.float32[1] = g;
        atts[att_cnt].clearValue.color.float32[2] = b;
        atts[att_cnt].clearValue.color.float32[3] = a;
        ++att_cnt;
    }
    if (clr_d || clr_s) {
        atts[att_cnt].aspectMask = (clr_d ? VK_IMAGE_ASPECT_DEPTH_BIT : 0U) |
                                   (clr_s ? VK_IMAGE_ASPECT_STENCIL_BIT : 0U);
        atts[att_cnt].clearValue.depthStencil = {z, stencil};
        ++att_cnt;
    }
    if (att_cnt == 0U) return;

    std::vector<VkClearRect> vk_rects;
    if (rects.empty()) {
        VkClearRect cr{};
        cr.rect = {{static_cast<std::int32_t>(state.viewport.x), static_cast<std::int32_t>(state.viewport.y)},
                   {std::min(vpw, impl_->bb_width - std::min(state.viewport.x, impl_->bb_width)),
                    std::min(vph, impl_->bb_height - std::min(state.viewport.y, impl_->bb_height))}};
        cr.baseArrayLayer = 0U;
        cr.layerCount = 1U;
        if (cr.rect.extent.width > 0U && cr.rect.extent.height > 0U) vk_rects.push_back(cr);
    } else {
        for (const D3d9Rect& rc : rects) {
            const std::int32_t l = std::clamp<std::int32_t>(rc.left, 0, static_cast<std::int32_t>(impl_->bb_width));
            const std::int32_t t = std::clamp<std::int32_t>(rc.top, 0, static_cast<std::int32_t>(impl_->bb_height));
            const std::int32_t ri = std::clamp<std::int32_t>(rc.right, l, static_cast<std::int32_t>(impl_->bb_width));
            const std::int32_t bt = std::clamp<std::int32_t>(rc.bottom, t, static_cast<std::int32_t>(impl_->bb_height));
            if (ri > l && bt > t) {
                vk_rects.push_back({{{l, t}, {static_cast<std::uint32_t>(ri - l), static_cast<std::uint32_t>(bt - t)}}, 0U, 1U});
            }
        }
    }
    if (!vk_rects.empty()) {
        impl_->vk.cmd_clear_attachments(impl_->frame_cmd, att_cnt, atts.data(),
                                        static_cast<std::uint32_t>(vk_rects.size()), vk_rects.data());
    }
}

void VulkanD3d9Backend::draw_indexed(const D3d9DrawState& state, std::uint32_t prim_type,
                                     std::int32_t base_vertex, std::uint32_t min_v, std::uint32_t num_v,
                                     std::uint32_t start_index, std::uint32_t prim_count) {
    (void)min_v;
    (void)num_v;
    const std::lock_guard<std::mutex> lock(impl_->api_mutex);
    const std::uint32_t index_count = vertex_count_for_prims(prim_type, prim_count);
    if (!impl_->gpu_ready || index_count == 0U || state.indices == 0U) return;

    VkBuffer ib = VK_NULL_HANDLE;
    VkDeviceSize ib_off = 0;
    bool is_u32 = false;
    if (!impl_->resolve_buffer_for_draw(state.indices, ib, ib_off, is_u32)) return;

    std::uint32_t inst_count = 1U;
    if (!impl_->prepare_draw(state, prim_type, inst_count)) return;
    impl_->vk.cmd_bind_index_buffer(impl_->frame_cmd, ib, ib_off, is_u32 ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_UINT16);
    impl_->vk.cmd_draw_indexed(impl_->frame_cmd, index_count, inst_count, start_index, base_vertex, 0U);
}

void VulkanD3d9Backend::draw_primitive(const D3d9DrawState& state, std::uint32_t prim_type,
                                       std::uint32_t start_vertex, std::uint32_t prim_count) {
    const std::lock_guard<std::mutex> lock(impl_->api_mutex);
    const std::uint32_t vert_count = vertex_count_for_prims(prim_type, prim_count);
    if (!impl_->gpu_ready || vert_count == 0U) return;
    std::uint32_t inst_count = 1U;
    if (!impl_->prepare_draw(state, prim_type, inst_count)) return;
    impl_->vk.cmd_draw(impl_->frame_cmd, vert_count, inst_count, start_vertex, 0U);
}

bool VulkanD3d9Backend::present(std::vector<std::uint8_t>* out_rgba8,
                                std::uint32_t* out_width, std::uint32_t* out_height) {
    std::uint64_t cb_frame = 0;
    double cb_inst_fps = 0.0;
    double cb_avg_fps = 0.0;
    std::string cb_path;
    bool fire_cb = false;
    bool ok = true;

    {
        const std::lock_guard<std::mutex> lock(impl_->api_mutex);
        if (out_width != nullptr) *out_width = impl_->bb_width;
        if (out_height != nullptr) *out_height = impl_->bb_height;
        if (!impl_->gpu_ready) {
            if (out_rgba8 != nullptr) {
                out_rgba8->assign(static_cast<std::size_t>(impl_->bb_width) * impl_->bb_height * 4U, 0U);
            }
            ok = impl_->initialized;
        } else {
#ifdef __ANDROID__
            const bool has_android_window =
                (impl_->active_window != nullptr && impl_->android_path != Impl::AndroidPresentPath::none);
#else
            const bool has_android_window = false;
#endif
            const bool need_present_img = (out_rgba8 != nullptr) || has_android_window;
            // Ensure a clear-only or first-frame present executes its render pass before readback/present.
            if (!impl_->frame_touched_rt &&
                (impl_->pending_clear_color || impl_->pending_clear_depth ||
                 (need_present_img && impl_->bb_layout == VK_IMAGE_LAYOUT_UNDEFINED) ||
                 out_rgba8 != nullptr)) {
                (void)impl_->ensure_render_pass_active();
            }

            bool handled_present = false;
#ifdef __ANDROID__
            if (has_android_window) {
                ok = impl_->present_to_android_locked(out_rgba8);
                handled_present = true;
            }
#endif
            if (!handled_present && impl_->frame_recording) {
                if (impl_->render_pass_active) {
                    impl_->vk.cmd_end_render_pass(impl_->frame_cmd);
                    impl_->render_pass_active = false;
                }
                const VkDeviceSize rb_bytes = static_cast<VkDeviceSize>(impl_->bb_width) * impl_->bb_height * 4U;
                const bool do_rb = (out_rgba8 != nullptr) && impl_->ensure_readback(rb_bytes, &impl_->last_error);
                if (do_rb) {
                    VkImageMemoryBarrier bar{};
                    bar.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                    bar.oldLayout = impl_->bb_layout;
                    bar.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                    bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                    bar.image = impl_->bb_img;
                    bar.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0U, 1U, 0U, 1U};
                    bar.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                    bar.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                    impl_->vk.cmd_pipeline_barrier(impl_->frame_cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                                   VK_PIPELINE_STAGE_TRANSFER_BIT, 0U, 0U, nullptr, 0U, nullptr, 1U, &bar);
                    VkBufferImageCopy cpy{};
                    cpy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0U, 0U, 1U};
                    cpy.imageExtent = {impl_->bb_width, impl_->bb_height, 1U};
                    impl_->vk.cmd_copy_image_to_buffer(impl_->frame_cmd, impl_->bb_img,
                                                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, impl_->readback_buf, 1U, &cpy);
                    bar.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                    bar.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                    bar.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                    bar.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                    impl_->vk.cmd_pipeline_barrier(impl_->frame_cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                                   VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0U, 0U, nullptr, 0U, nullptr,
                                                   1U, &bar);
                    impl_->bb_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
                }
                // No wait unless the caller wants the pixels: the next frame records meanwhile.
                if (!impl_->submit_frame_locked(VK_NULL_HANDLE, 0U, VK_NULL_HANDLE, do_rb)) {
                    ok = false;
                } else if (do_rb) {
                    impl_->alloc.invalidate(impl_->readback_mem, 0U, rb_bytes);
                    out_rgba8->resize(static_cast<std::size_t>(rb_bytes));
                    std::memcpy(out_rgba8->data(), impl_->readback_mem.mapped, static_cast<std::size_t>(rb_bytes));
                }
            }
            if (!handled_present && !impl_->frame_recording && impl_->up_recording) {
                (void)impl_->submit_frame_locked(VK_NULL_HANDLE, 0U, VK_NULL_HANDLE, false);
            }

            impl_->frame_touched_rt = false;
            impl_->pending_clear_color = false;
            impl_->pending_clear_depth = false;
            impl_->bound_pipeline = VK_NULL_HANDLE;
            impl_->bound_sampler_set = VK_NULL_HANDLE;
            ++impl_->frame_index;
            // Take the next frame slot; waits only for the frame before this one.
            impl_->advance_frame_slot();
#ifdef __ANDROID__
            impl_->show_finished_ahb_locked();
#endif
        }

        if (ok) {
            const auto now = std::chrono::steady_clock::now();
            ++impl_->presented_frames;
            ++impl_->fps_win_frames;
            if (!impl_->fps_started) {
                impl_->fps_started = true;
                impl_->fps_first_tp = now;
                impl_->fps_win_tp = now;
                impl_->last_inst_fps = 0.0;
            } else {
                const double win_s = std::chrono::duration<double>(now - impl_->fps_win_tp).count();
                if (win_s >= 0.5) {
                    impl_->last_inst_fps = static_cast<double>(impl_->fps_win_frames) / win_s;
                    impl_->fps_win_tp = now;
                    impl_->fps_win_frames = 0U;
                }
            }
            const double total_s = std::chrono::duration<double>(now - impl_->fps_first_tp).count();
            cb_avg_fps = (total_s > 0.001) ? (static_cast<double>(impl_->presented_frames) / total_s) : 0.0;
            cb_inst_fps = (impl_->last_inst_fps > 0.0) ? impl_->last_inst_fps : cb_avg_fps;
            cb_frame = impl_->presented_frames;
            cb_path = impl_->report.presentation_path;
            fire_cb = static_cast<bool>(impl_->config.on_frame_presented);
        }
    }

    if (fire_cb) {
        impl_->config.on_frame_presented(cb_frame, cb_inst_fps, cb_avg_fps, cb_path);
    }
    return ok;
}

void VulkanD3d9Backend::set_native_window(void* native_window) noexcept {
    const std::lock_guard<std::mutex> lock(impl_->api_mutex);
#ifdef __ANDROID__
    auto* next_win = static_cast<ANativeWindow*>(native_window);
    if (next_win != nullptr) ANativeWindow_acquire(next_win);
    if (impl_->gpu_ready) {
        impl_->flush_frame_if_recording();
        impl_->teardown_android_presentation_locked();
    }
    if (impl_->active_window != nullptr) {
        ANativeWindow_release(impl_->active_window);
    }
    impl_->active_window = next_win;
    if (impl_->gpu_ready && impl_->active_window != nullptr) {
        impl_->setup_android_presentation_locked();
    }
#else
    (void)native_window;
#endif
}

void VulkanD3d9Backend::flush_pipeline_cache() noexcept {
    const std::lock_guard<std::mutex> lock(impl_->api_mutex);
    if (impl_->gpu_ready) {
        impl_->save_pipeline_cache();
    }
}

std::string VulkanD3d9Backend::active_presentation_path() const {
    const std::lock_guard<std::mutex> lock(impl_->api_mutex);
    return impl_->report.presentation_path;
}

std::size_t VulkanD3d9Backend::compiled_shader_count() const noexcept {
    std::size_t n = 0;
    for (const auto& [k, s] : impl_->shader_by_hash) {
        (void)k;
        if (s.valid) ++n;
    }
    return n;
}
std::size_t VulkanD3d9Backend::failed_shader_count() const noexcept {
    std::size_t n = 0;
    for (const auto& [k, s] : impl_->shader_by_hash) {
        (void)k;
        if (s.attempted && !s.valid) ++n;
    }
    return n;
}
std::size_t VulkanD3d9Backend::cached_pipeline_count() const noexcept { return impl_->pipeline_cache_map.size(); }
std::size_t VulkanD3d9Backend::cached_sampler_count() const noexcept { return impl_->sampler_cache.size(); }
std::size_t VulkanD3d9Backend::cached_descriptor_set_count() const noexcept { return impl_->sampler_set_cache.size(); }
const std::string& VulkanD3d9Backend::last_error() const noexcept { return impl_->last_error; }

}  // namespace wse::platform
