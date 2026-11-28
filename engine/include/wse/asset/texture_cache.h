// Header-only ETC2 disk texture cache key and binary container codec
// (no etcpak dependency).
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "wse/asset/etc2_types.h"

namespace wse::asset {

inline constexpr std::uint32_t kTextureCacheVersion = 1U;

namespace texture_cache_detail {

inline constexpr std::array<std::uint8_t, 8> kMagic = {
    'W', 'S', 'E', '2', 'T', 'X', 'C', '1',
};

inline bool set_error(std::string* error_out, std::string msg) {
    if (error_out != nullptr) {
        *error_out = std::move(msg);
    }
    return false;
}

inline void append_u32_le(std::vector<std::uint8_t>& out, std::uint32_t v) {
    out.push_back(static_cast<std::uint8_t>(v & 0xFFU));
    out.push_back(static_cast<std::uint8_t>((v >> 8U) & 0xFFU));
    out.push_back(static_cast<std::uint8_t>((v >> 16U) & 0xFFU));
    out.push_back(static_cast<std::uint8_t>((v >> 24U) & 0xFFU));
}

inline std::uint32_t read_u32_le(const std::uint8_t* p) noexcept {
    return static_cast<std::uint32_t>(p[0]) |
           (static_cast<std::uint32_t>(p[1]) << 8U) |
           (static_cast<std::uint32_t>(p[2]) << 16U) |
           (static_cast<std::uint32_t>(p[3]) << 24U);
}

inline std::uint64_t fnv1a_step_u32_le(std::uint64_t h, std::uint32_t v) noexcept {
    constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
    for (unsigned i = 0; i < 4U; ++i) {
        h ^= static_cast<std::uint64_t>((v >> (i * 8U)) & 0xFFU);
        h *= kFnvPrime;
    }
    return h;
}

}  // namespace texture_cache_detail

[[nodiscard]] inline std::string cache_key(std::span<const std::uint8_t> source_dds,
                                           std::uint32_t first_mip) {
    constexpr std::uint64_t kFnvOffsetBasis = 14695981039346656037ULL;
    constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
    std::uint64_t h = kFnvOffsetBasis;
    for (std::uint8_t b : source_dds) {
        h ^= static_cast<std::uint64_t>(b);
        h *= kFnvPrime;
    }
    h = texture_cache_detail::fnv1a_step_u32_le(h, first_mip);
    h = texture_cache_detail::fnv1a_step_u32_le(h, kTextureCacheVersion);
    // SplitMix64 avalanche so adjacent first_mip values diverge across all 64 bits.
    h ^= h >> 30U;
    h *= 0xbf58476d1ce4e5b9ULL;
    h ^= h >> 27U;
    h *= 0x94d049bb133111ebULL;
    h ^= h >> 31U;

    static constexpr char kHex[] = "0123456789abcdef";
    std::string out(21U, '\0');
    for (unsigned i = 0; i < 16U; ++i) {
        out[i] = kHex[(h >> ((15U - i) * 4U)) & 0xFULL];
    }
    out[16] = '.';
    out[17] = 'e';
    out[18] = 't';
    out[19] = 'c';
    out[20] = '2';
    return out;
}

[[nodiscard]] inline std::vector<std::uint8_t> encode(const etc2::Image& image) {
    if (image.width == 0U || image.height == 0U || image.levels == 0U || image.faces == 0U) {
        return {};
    }
    if (image.format != etc2::Format::rgb8 && image.format != etc2::Format::rgba8) {
        return {};
    }
    const std::size_t expected_entries =
        static_cast<std::size_t>(image.faces) * static_cast<std::size_t>(image.levels);
    if (image.data.size() != expected_entries) {
        return {};
    }
    std::size_t total_bytes = 32U;
    for (const etc2::Level& lvl : image.data) {
        if (lvl.blocks.size() > 0xFFFFFFFFULL) {
            return {};
        }
        total_bytes += 12U + lvl.blocks.size();
    }

    std::vector<std::uint8_t> out;
    out.reserve(total_bytes);
    out.insert(out.end(), texture_cache_detail::kMagic.begin(), texture_cache_detail::kMagic.end());
    texture_cache_detail::append_u32_le(out, kTextureCacheVersion);
    texture_cache_detail::append_u32_le(out, static_cast<std::uint32_t>(image.format));
    texture_cache_detail::append_u32_le(out, image.width);
    texture_cache_detail::append_u32_le(out, image.height);
    texture_cache_detail::append_u32_le(out, image.levels);
    texture_cache_detail::append_u32_le(out, image.faces);
    for (const etc2::Level& lvl : image.data) {
        texture_cache_detail::append_u32_le(out, lvl.width);
        texture_cache_detail::append_u32_le(out, lvl.height);
        texture_cache_detail::append_u32_le(out, static_cast<std::uint32_t>(lvl.blocks.size()));
        out.insert(out.end(), lvl.blocks.begin(), lvl.blocks.end());
    }
    return out;
}

[[nodiscard]] inline bool decode(std::span<const std::uint8_t> bytes,
                                 etc2::Image& out,
                                 std::string* error = nullptr) {
    out = etc2::Image{};
    if (bytes.size() < 32U) {
        return texture_cache_detail::set_error(error, "truncated texture cache header");
    }
    if (std::memcmp(bytes.data(), texture_cache_detail::kMagic.data(), 8U) != 0) {
        return texture_cache_detail::set_error(error, "invalid texture cache magic");
    }
    const std::uint32_t version = texture_cache_detail::read_u32_le(bytes.data() + 8U);
    if (version != kTextureCacheVersion) {
        return texture_cache_detail::set_error(error, "unsupported texture cache version");
    }
    const std::uint32_t fmt_raw = texture_cache_detail::read_u32_le(bytes.data() + 12U);
    if (fmt_raw != static_cast<std::uint32_t>(etc2::Format::rgb8) &&
        fmt_raw != static_cast<std::uint32_t>(etc2::Format::rgba8)) {
        return texture_cache_detail::set_error(error, "unsupported ETC2 format in texture cache");
    }
    const etc2::Format format = static_cast<etc2::Format>(fmt_raw);
    const std::uint32_t width = texture_cache_detail::read_u32_le(bytes.data() + 16U);
    const std::uint32_t height = texture_cache_detail::read_u32_le(bytes.data() + 20U);
    const std::uint32_t levels = texture_cache_detail::read_u32_le(bytes.data() + 24U);
    const std::uint32_t faces = texture_cache_detail::read_u32_le(bytes.data() + 28U);
    if (width == 0U || height == 0U || width > 16384U || height > 16384U) {
        return texture_cache_detail::set_error(error, "invalid texture cache dimensions");
    }
    if (levels == 0U || levels > 32U || faces == 0U || faces > 64U) {
        return texture_cache_detail::set_error(error, "invalid texture cache level or face count");
    }

    const std::uint64_t entry_count =
        static_cast<std::uint64_t>(faces) * static_cast<std::uint64_t>(levels);
    if ((bytes.size() - 32U) / 12U < entry_count) {
        return texture_cache_detail::set_error(error, "truncated texture cache level headers");
    }

    const std::uint64_t block_bytes = etc2::block_size_bytes(format);
    std::vector<etc2::Level> decoded;
    decoded.reserve(static_cast<std::size_t>(entry_count));

    std::size_t cursor = 32U;
    for (std::uint32_t f = 0; f < faces; ++f) {
        std::uint32_t expected_w = width;
        std::uint32_t expected_h = height;
        for (std::uint32_t l = 0; l < levels; ++l) {
            if (cursor > bytes.size() || bytes.size() - cursor < 12U) {
                return texture_cache_detail::set_error(error, "truncated texture cache level header");
            }
            const std::uint32_t lw = texture_cache_detail::read_u32_le(bytes.data() + cursor);
            const std::uint32_t lh = texture_cache_detail::read_u32_le(bytes.data() + cursor + 4U);
            const std::uint32_t byte_count =
                texture_cache_detail::read_u32_le(bytes.data() + cursor + 8U);
            cursor += 12U;

            if (lw != expected_w || lh != expected_h) {
                return texture_cache_detail::set_error(error, "texture cache level dimension mismatch");
            }
            const std::uint64_t bw = (static_cast<std::uint64_t>(lw) + 3ULL) / 4ULL;
            const std::uint64_t bh = (static_cast<std::uint64_t>(lh) + 3ULL) / 4ULL;
            const std::uint64_t expected_bytes = bw * bh * block_bytes;
            if (static_cast<std::uint64_t>(byte_count) != expected_bytes) {
                return texture_cache_detail::set_error(error, "texture cache level byte count mismatch");
            }
            if (static_cast<std::size_t>(byte_count) > bytes.size() - cursor) {
                return texture_cache_detail::set_error(error, "truncated texture cache level payload");
            }

            etc2::Level lvl;
            lvl.width = lw;
            lvl.height = lh;
            lvl.blocks.assign(bytes.data() + cursor, bytes.data() + cursor + byte_count);
            decoded.push_back(std::move(lvl));
            cursor += static_cast<std::size_t>(byte_count);

            expected_w = (std::max)(1U, expected_w >> 1U);
            expected_h = (std::max)(1U, expected_h >> 1U);
        }
    }

    if (cursor != bytes.size()) {
        return texture_cache_detail::set_error(error, "unexpected trailing bytes in texture cache");
    }

    out.format = format;
    out.width = width;
    out.height = height;
    out.levels = levels;
    out.faces = faces;
    out.data = std::move(decoded);
    return true;
}

// The first mip level a texture starts from under `max_size` (0 = no cap): the largest level
// whose width and height both fit, never past the last level. The Vulkan device uses it when it
// does not upload BC, and `r4_run --for-phone` uses it to name the converted textures it prepares,
// so the cache keys of both sides agree.
[[nodiscard]] inline std::uint32_t first_mip_for_cap(std::uint32_t width, std::uint32_t height,
                                                     std::uint32_t mip_count, std::uint32_t max_size) noexcept {
    std::uint32_t first = 0U;
    if (max_size == 0U) return first;
    while (first + 1U < mip_count && std::max(width >> first, height >> first) > max_size) ++first;
    return first;
}

struct TextureCachePruneResult {
    std::size_t removed_files = 0;
    std::uint64_t freed_bytes = 0;
    std::uint64_t remaining_bytes = 0;
};

[[nodiscard]] inline TextureCachePruneResult prune_texture_cache_lru(const std::filesystem::path& dir,
                                                                     std::uint64_t max_bytes) {
    namespace fs = std::filesystem;
    TextureCachePruneResult res{};
    if (max_bytes == 0U || dir.empty()) return res;
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) return res;
    struct Entry {
        fs::path path;
        std::uint64_t size = 0;
        fs::file_time_type mtime{};
    };
    std::vector<Entry> entries;
    std::uint64_t total = 0;
    for (const fs::directory_entry& de : fs::directory_iterator(dir, ec)) {
        if (!de.is_regular_file(ec)) continue;
        if (de.path().extension() != ".etc2") continue;
        const std::uint64_t sz = de.file_size(ec);
        if (ec) continue;
        const auto mt = de.last_write_time(ec);
        if (ec) continue;
        entries.push_back({de.path(), sz, mt});
        total += sz;
    }
    res.remaining_bytes = total;
    if (total <= max_bytes) return res;
    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
        if (a.mtime != b.mtime) return a.mtime < b.mtime;
        return a.path.filename() < b.path.filename();
    });
    const std::uint64_t target = (max_bytes * 85ULL) / 100ULL;
    for (const Entry& e : entries) {
        if (total <= target) break;
        if (fs::remove(e.path, ec)) {
            total -= e.size;
            ++res.removed_files;
            res.freed_bytes += e.size;
        }
    }
    res.remaining_bytes = total;
    return res;
}

}  // namespace wse::asset
