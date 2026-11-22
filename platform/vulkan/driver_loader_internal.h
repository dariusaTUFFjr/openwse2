// Internal Vulkan handle bridge between `driver_loader.cpp` and `d3d9_vulkan.cpp`.
// Private to `platform/vulkan/` — never include from `engine/include/`.

#pragma once

#include "driver_loader.h"

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif

#ifdef __ANDROID__
#ifndef VK_USE_PLATFORM_ANDROID_KHR
#define VK_USE_PLATFORM_ANDROID_KHR
#endif
#endif

#if __has_include("../../runtime/android/third_party/vulkan/vulkan.h")
#include "../../runtime/android/third_party/vulkan/vulkan.h"
#elif __has_include(<vulkan/vulkan.h>)
#include <vulkan/vulkan.h>
#else
#error "Missing Vulkan headers (expected vendored headers in runtime/android/third_party)"
#endif

namespace wse::platform {

struct VulkanRawHandles {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue graphics_queue = VK_NULL_HANDLE;
    VkCommandPool command_pool = VK_NULL_HANDLE;
    std::uint32_t graphics_queue_family = 0;
    PFN_vkGetInstanceProcAddr gipa = nullptr;
    PFN_vkGetDeviceProcAddr gdpa = nullptr;
    bool is_mock = false;
    bool can_execute_gpu = false;
    bool sampler_anisotropy = false;
    bool fill_mode_non_solid = false;
    bool ahb_external_memory = false;
    bool queue_family_foreign = false;
    bool external_semaphore_fd = false;
    std::uint64_t generation = 0;
    VulkanDriverProbeResult probe{};
};

[[nodiscard]] bool vulkan_loader_extract_handles(const VulkanDriverLoader& loader,
                                                 VulkanRawHandles& out) noexcept;

}  // namespace wse::platform
