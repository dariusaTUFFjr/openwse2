// Shared ETC2/EAC texture payload types (no etcpak dependency).
#pragma once

#include <cstdint>
#include <vector>

namespace wse::asset::etc2 {

enum class Format : std::uint32_t {
    rgb8 = 1,   // ETC2 RGB8 (8 B/block)
    rgba8 = 2,  // ETC2 RGBA8 EAC (16 B/block)
};

struct Level {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<std::uint8_t> blocks;
};

struct Image {
    Format format = Format::rgb8;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t levels = 0;
    std::uint32_t faces = 1;  // faces 6 for cube maps
    std::vector<Level> data;  // face-major: data[face * levels + level]
};

inline constexpr std::uint32_t block_size_bytes(Format format) noexcept {
    return (format == Format::rgba8) ? 16U : 8U;
}

}  // namespace wse::asset::etc2
