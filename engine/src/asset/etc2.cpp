// Clean-room DDS -> ETC2 RGB8 / ETC2 RGBA8 (EAC alpha) transcoder backed by etcpak.
// `ProcessRGB.hpp` stays private to this translation unit.

#include "wse/asset/etc2.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "ProcessRGB.hpp"

namespace wse::asset::etc2 {

namespace {

bool set_error(std::string* error_out, std::string msg) {
    if (error_out != nullptr) {
        *error_out = std::move(msg);
    }
    return false;
}

bool bc1_has_transparent_texel(const DdsImageView& view,
                               std::uint32_t first_mip,
                               std::string* error) {
    for (std::uint32_t f = 0; f < view.face_count; ++f) {
        for (std::uint32_t m = first_mip; m < view.mip_count; ++m) {
            const DdsMipSlice& slice = view.slices[static_cast<std::size_t>(f) * view.mip_count + m];
            const std::uint32_t w = slice.width;
            const std::uint32_t h = slice.height;
            const std::uint32_t blocks_x = (w + 3U) / 4U;
            const std::uint32_t blocks_y = (h + 3U) / 4U;
            const std::size_t required = static_cast<std::size_t>(blocks_x) * blocks_y * 8U;
            if (slice.data.size() < required || slice.data.data() == nullptr) {
                set_error(error, "DDS BC1 slice payload smaller than its dimensions require");
                return false;
            }
            const std::byte* src = slice.data.data();
            for (std::uint32_t by = 0; by < blocks_y; ++by) {
                const std::uint32_t rows = (std::min)(4U, h - by * 4U);
                for (std::uint32_t bx = 0; bx < blocks_x; ++bx) {
                    const std::uint32_t cols = (std::min)(4U, w - bx * 4U);
                    const std::byte* blk = src + (static_cast<std::size_t>(by) * blocks_x + bx) * 8U;
                    const std::uint16_t c0 = dds_detail::read_u16_le(blk);
                    const std::uint16_t c1 = dds_detail::read_u16_le(blk + 2);
                    if (c0 > c1) {
                        continue;
                    }
                    const std::uint32_t idx_bits = dds_detail::read_u32_le(blk + 4);
                    for (std::uint32_t py = 0; py < rows; ++py) {
                        for (std::uint32_t px = 0; px < cols; ++px) {
                            const std::uint32_t p = py * 4U + px;
                            if (((idx_bits >> (p * 2U)) & 0x3U) == 3U) {
                                return true;
                            }
                        }
                    }
                }
            }
        }
    }
    return false;
}

bool pick_target_format(const DdsImageView& view,
                        std::uint32_t first_mip,
                        Format& format_out,
                        std::string* error) {
    switch (view.format) {
        case DdsFormat::BC1_UNORM: {
            std::string bc1_err;
            const bool has_alpha = bc1_has_transparent_texel(view, first_mip, &bc1_err);
            if (!bc1_err.empty()) {
                return set_error(error, std::move(bc1_err));
            }
            format_out = has_alpha ? Format::rgba8 : Format::rgb8;
            return true;
        }
        case DdsFormat::BC2_UNORM:
        case DdsFormat::BC3_UNORM:
        case DdsFormat::B8G8R8A8_UNORM:
        case DdsFormat::R8G8B8A8_UNORM:
        case DdsFormat::B5G5R5A1_UNORM:
        case DdsFormat::B4G4R4A4_UNORM:
        case DdsFormat::A8_UNORM:
        case DdsFormat::A8L8_UNORM:
            format_out = Format::rgba8;
            return true;
        case DdsFormat::BC4_UNORM:
        case DdsFormat::BC5_UNORM:
        case DdsFormat::B8G8R8X8_UNORM:
        case DdsFormat::R8G8B8X8_UNORM:
        case DdsFormat::B8G8R8_UNORM:
        case DdsFormat::B5G6R5_UNORM:
        case DdsFormat::L8_UNORM:
        case DdsFormat::L16_UNORM:
            format_out = view.has_alpha ? Format::rgba8 : Format::rgb8;
            return true;
        default:
            return set_error(error, "unsupported DDS format for ETC2 transcoding");
    }
}

inline std::uint32_t pack_bgra(const std::uint8_t* p) noexcept {
    // etcpak expects BGRA bytes in memory (B=0, G=1, R=2, A=3) and reads A via v >> 24.
    return static_cast<std::uint32_t>(p[2]) |
           (static_cast<std::uint32_t>(p[1]) << 8U) |
           (static_cast<std::uint32_t>(p[0]) << 16U) |
           (static_cast<std::uint32_t>(p[3]) << 24U);
}

void rgba8_to_padded_bgra(const std::vector<std::uint8_t>& rgba,
                          std::uint32_t w,
                          std::uint32_t h,
                          std::uint32_t pw,
                          std::uint32_t ph,
                          std::vector<std::uint32_t>& bgra) {
    bgra.resize(static_cast<std::size_t>(pw) * ph);
    const std::uint8_t* src = rgba.data();
    std::uint32_t* dst = bgra.data();
    if (w == pw && h == ph) {
        const std::size_t count = static_cast<std::size_t>(w) * h;
        for (std::size_t i = 0; i < count; ++i) {
            dst[i] = pack_bgra(src + i * 4U);
        }
        return;
    }
    for (std::uint32_t y = 0; y < ph; ++y) {
        const std::uint32_t sy = (y < h) ? y : (h - 1U);
        const std::uint8_t* row_src = src + static_cast<std::size_t>(sy) * w * 4U;
        std::uint32_t* row_dst = dst + static_cast<std::size_t>(y) * pw;
        for (std::uint32_t x = 0; x < pw; ++x) {
            const std::uint32_t sx = (x < w) ? x : (w - 1U);
            row_dst[x] = pack_bgra(row_src + static_cast<std::size_t>(sx) * 4U);
        }
    }
}

void compress_strip(const std::uint32_t* bgra,
                    std::uint64_t* dst64,
                    std::uint32_t padded_w,
                    std::uint32_t blocks_x,
                    std::uint32_t by_start,
                    std::uint32_t by_end,
                    Format format) {
    const std::uint32_t rows = by_end - by_start;
    const std::uint32_t blocks = rows * blocks_x;
    const std::uint32_t* src_strip = bgra + static_cast<std::size_t>(by_start) * 4U * padded_w;
    const std::size_t u64s_per_block = (format == Format::rgba8) ? 2U : 1U;
    std::uint64_t* dst_strip =
        dst64 + static_cast<std::size_t>(by_start) * blocks_x * u64s_per_block;
    if (format == Format::rgba8) {
        CompressEtc2Rgba(src_strip, dst_strip, blocks, padded_w, true);
    } else {
        CompressEtc2Rgb(src_strip, dst_strip, blocks, padded_w, true);
    }
}

struct SliceWork {
    std::uint32_t src_mip = 0;
    std::uint32_t src_face = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t padded_w = 0;
    std::uint32_t padded_h = 0;
    std::uint32_t blocks_x = 0;
    std::uint32_t blocks_y = 0;
    std::vector<std::uint32_t> bgra;
    std::vector<std::uint64_t> dst64;
};

struct StripTask {
    std::size_t slice_idx = 0;
    std::uint32_t by_start = 0;
    std::uint32_t by_end = 0;
};

}  // namespace

bool transcode_dds(const DdsImageView& view,
                   std::uint32_t first_mip,
                   Image& out,
                   std::string* error,
                   unsigned threads) {
    out = Image{};
    if (view.width == 0U || view.height == 0U || view.mip_count == 0U || view.face_count == 0U) {
        return set_error(error, "invalid DDS dimensions or slice counts");
    }
    if (first_mip >= view.mip_count) {
        return set_error(error, "first_mip out of range");
    }
    const std::size_t expected_slices =
        static_cast<std::size_t>(view.face_count) * static_cast<std::size_t>(view.mip_count);
    if (view.slices.size() < expected_slices) {
        return set_error(error, "DDS slice table smaller than face_count * mip_count");
    }

    Format target_format = Format::rgb8;
    if (!pick_target_format(view, first_mip, target_format, error)) {
        return false;
    }

    const std::uint32_t kept_levels = view.mip_count - first_mip;
    const std::size_t total_out_slices =
        static_cast<std::size_t>(view.face_count) * static_cast<std::size_t>(kept_levels);
    const std::size_t u64s_per_block = (target_format == Format::rgba8) ? 2U : 1U;
    const std::size_t bytes_per_block = block_size_bytes(target_format);

    const DdsMipSlice& top_slice = view.slices[first_mip];
    out.format = target_format;
    out.width = top_slice.width;
    out.height = top_slice.height;
    out.levels = kept_levels;
    out.faces = view.face_count;
    out.data.resize(total_out_slices);

    const unsigned max_threads = (threads == 0U) ? 1U : threads;
    if (max_threads == 1U) {
        std::vector<std::uint8_t> rgba;
        std::vector<std::uint32_t> bgra;
        std::vector<std::uint64_t> dst64;
        for (std::uint32_t f = 0; f < view.face_count; ++f) {
            for (std::uint32_t l = 0; l < kept_levels; ++l) {
                const std::uint32_t mip = first_mip + l;
                const DdsMipSlice& slice =
                    view.slices[static_cast<std::size_t>(f) * view.mip_count + mip];
                if (!decode_dds_mip_rgba8(view, mip, f, rgba, error)) {
                    out = Image{};
                    return false;
                }
                const std::uint32_t pw = (slice.width + 3U) & ~3U;
                const std::uint32_t ph = (slice.height + 3U) & ~3U;
                const std::uint32_t blocks_x = pw / 4U;
                const std::uint32_t blocks_y = ph / 4U;
                rgba8_to_padded_bgra(rgba, slice.width, slice.height, pw, ph, bgra);

                const std::size_t total_blocks = static_cast<std::size_t>(blocks_x) * blocks_y;
                dst64.resize(total_blocks * u64s_per_block);
                compress_strip(bgra.data(), dst64.data(), pw, blocks_x, 0U, blocks_y, target_format);

                Level& lvl = out.data[static_cast<std::size_t>(f) * kept_levels + l];
                lvl.width = slice.width;
                lvl.height = slice.height;
                lvl.blocks.resize(total_blocks * bytes_per_block);
                std::memcpy(lvl.blocks.data(), dst64.data(), lvl.blocks.size());
            }
        }
        return true;
    }

    std::vector<SliceWork> slices(total_out_slices);
    for (std::uint32_t f = 0; f < view.face_count; ++f) {
        for (std::uint32_t l = 0; l < kept_levels; ++l) {
            const std::uint32_t mip = first_mip + l;
            const DdsMipSlice& src = view.slices[static_cast<std::size_t>(f) * view.mip_count + mip];
            SliceWork& sw = slices[static_cast<std::size_t>(f) * kept_levels + l];
            sw.src_mip = mip;
            sw.src_face = f;
            sw.width = src.width;
            sw.height = src.height;
            sw.padded_w = (src.width + 3U) & ~3U;
            sw.padded_h = (src.height + 3U) & ~3U;
            sw.blocks_x = sw.padded_w / 4U;
            sw.blocks_y = sw.padded_h / 4U;
            if (sw.width == 0U || sw.height == 0U) {
                out = Image{};
                return set_error(error, "zero-sized DDS mip slice");
            }
        }
    }

    // Phase 1: decode & pad each slice into BGRA (bounded by max_threads).
    std::atomic<std::size_t> next_slice{0};
    std::atomic<bool> decode_failed{false};
    std::mutex err_mutex;
    std::string decode_err;

    auto decode_worker = [&]() {
        std::vector<std::uint8_t> rgba;
        std::string local_err;
        while (!decode_failed.load(std::memory_order_relaxed)) {
            const std::size_t idx = next_slice.fetch_add(1U, std::memory_order_relaxed);
            if (idx >= slices.size()) {
                break;
            }
            SliceWork& sw = slices[idx];
            if (!decode_dds_mip_rgba8(view, sw.src_mip, sw.src_face, rgba, &local_err)) {
                if (!decode_failed.exchange(true, std::memory_order_relaxed)) {
                    std::lock_guard<std::mutex> lock(err_mutex);
                    decode_err = std::move(local_err);
                }
                break;
            }
            rgba8_to_padded_bgra(rgba, sw.width, sw.height, sw.padded_w, sw.padded_h, sw.bgra);
            const std::size_t total_blocks = static_cast<std::size_t>(sw.blocks_x) * sw.blocks_y;
            sw.dst64.resize(total_blocks * u64s_per_block);
        }
    };

    const unsigned decode_workers =
        (std::min)(max_threads, static_cast<unsigned>(slices.size()));
    if (decode_workers > 1U) {
        std::vector<std::thread> pool;
        pool.reserve(decode_workers - 1U);
        for (unsigned t = 1U; t < decode_workers; ++t) {
            pool.emplace_back(decode_worker);
        }
        decode_worker();
        for (std::thread& th : pool) {
            th.join();
        }
    } else {
        decode_worker();
    }

    if (decode_failed.load(std::memory_order_relaxed)) {
        out = Image{};
        return set_error(error, std::move(decode_err));
    }

    // Phase 2: split slices into independent block-row strips and compress in parallel.
    constexpr std::uint32_t kStripBlockRows = 8U;
    std::vector<StripTask> tasks;
    for (std::size_t i = 0; i < slices.size(); ++i) {
        const SliceWork& sw = slices[i];
        for (std::uint32_t by = 0; by < sw.blocks_y; by += kStripBlockRows) {
            const std::uint32_t by_end = (std::min)(sw.blocks_y, by + kStripBlockRows);
            tasks.push_back(StripTask{i, by, by_end});
        }
    }

    std::atomic<std::size_t> next_task{0};
    auto compress_worker = [&]() {
        while (true) {
            const std::size_t idx = next_task.fetch_add(1U, std::memory_order_relaxed);
            if (idx >= tasks.size()) {
                break;
            }
            const StripTask& task = tasks[idx];
            SliceWork& sw = slices[task.slice_idx];
            compress_strip(sw.bgra.data(), sw.dst64.data(), sw.padded_w, sw.blocks_x,
                           task.by_start, task.by_end, target_format);
        }
    };

    const unsigned compress_workers =
        (std::min)(max_threads, static_cast<unsigned>(tasks.size()));
    if (compress_workers > 1U) {
        std::vector<std::thread> pool;
        pool.reserve(compress_workers - 1U);
        for (unsigned t = 1U; t < compress_workers; ++t) {
            pool.emplace_back(compress_worker);
        }
        compress_worker();
        for (std::thread& th : pool) {
            th.join();
        }
    } else {
        compress_worker();
    }

    for (std::size_t i = 0; i < slices.size(); ++i) {
        SliceWork& sw = slices[i];
        Level& lvl = out.data[i];
        lvl.width = sw.width;
        lvl.height = sw.height;
        lvl.blocks.resize(static_cast<std::size_t>(sw.blocks_x) * sw.blocks_y * bytes_per_block);
        std::memcpy(lvl.blocks.data(), sw.dst64.data(), lvl.blocks.size());
    }

    return true;
}

}  // namespace wse::asset::etc2
