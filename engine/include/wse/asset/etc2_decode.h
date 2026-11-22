// Header-only reference ETC2 RGB8 + EAC alpha decoder (Khronos / OpenGL ES 3.0 spec:
// Individual, Differential, T, H, and Planar RGB modes + 16-table EAC alpha).
// No etcpak dependency.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <vector>

#include "wse/asset/etc2_types.h"

namespace wse::asset::etc2 {

namespace detail {

inline std::uint8_t clamp_u8(int v) noexcept {
    if (v < 0) return 0U;
    if (v > 255) return 255U;
    return static_cast<std::uint8_t>(v);
}

inline int sign_extend_3(std::uint32_t v) noexcept {
    return (v & 0x4U) != 0U ? static_cast<int>(v) - 8 : static_cast<int>(v);
}

inline std::uint8_t expand4(std::uint32_t v) noexcept {
    return static_cast<std::uint8_t>((v << 4U) | v);
}

inline std::uint8_t expand5(std::uint32_t v) noexcept {
    return static_cast<std::uint8_t>((v << 3U) | (v >> 2U));
}

inline std::uint8_t expand6(std::uint32_t v) noexcept {
    return static_cast<std::uint8_t>((v << 2U) | (v >> 4U));
}

inline std::uint8_t expand7(std::uint32_t v) noexcept {
    return static_cast<std::uint8_t>((v << 1U) | (v >> 6U));
}

}  // namespace detail

inline void decode_etc2_rgb8_block(const std::uint8_t* blk,
                                   std::uint8_t out_rgb[4][4][3]) noexcept {
    static constexpr int kModifier[8][4] = {
        {2, 8, -2, -8},
        {5, 17, -5, -17},
        {9, 29, -9, -29},
        {13, 42, -13, -42},
        {18, 60, -18, -60},
        {24, 80, -24, -80},
        {33, 106, -33, -106},
        {47, 183, -47, -183},
    };
    static constexpr int kTHDist[8] = {3, 6, 11, 16, 23, 32, 41, 64};

    const bool diff = ((blk[3] >> 1U) & 1U) != 0U;
    const bool flip = (blk[3] & 1U) != 0U;
    const std::uint32_t word_lo =
        (static_cast<std::uint32_t>(blk[4]) << 24U) |
        (static_cast<std::uint32_t>(blk[5]) << 16U) |
        (static_cast<std::uint32_t>(blk[6]) << 8U) |
        static_cast<std::uint32_t>(blk[7]);

    auto read_pixel_index = [&](int x, int y) noexcept -> std::uint32_t {
        const unsigned p = static_cast<unsigned>(4 * x + y);
        const std::uint32_t lsb = (word_lo >> p) & 1U;
        const std::uint32_t msb = (word_lo >> (16U + p)) & 1U;
        return (msb << 1U) | lsb;
    };

    if (!diff) {
        const std::uint8_t r[2] = {detail::expand4((blk[0] >> 4U) & 0xFU), detail::expand4(blk[0] & 0xFU)};
        const std::uint8_t g[2] = {detail::expand4((blk[1] >> 4U) & 0xFU), detail::expand4(blk[1] & 0xFU)};
        const std::uint8_t b[2] = {detail::expand4((blk[2] >> 4U) & 0xFU), detail::expand4(blk[2] & 0xFU)};
        const unsigned t[2] = {(blk[3] >> 5U) & 0x7U, (blk[3] >> 2U) & 0x7U};
        for (int y = 0; y < 4; ++y) {
            for (int x = 0; x < 4; ++x) {
                const unsigned sub = flip ? (y >= 2 ? 1U : 0U) : (x >= 2 ? 1U : 0U);
                const int mod = kModifier[t[sub]][read_pixel_index(x, y)];
                out_rgb[y][x][0] = detail::clamp_u8(static_cast<int>(r[sub]) + mod);
                out_rgb[y][x][1] = detail::clamp_u8(static_cast<int>(g[sub]) + mod);
                out_rgb[y][x][2] = detail::clamp_u8(static_cast<int>(b[sub]) + mod);
            }
        }
        return;
    }

    const int r1_5 = static_cast<int>((blk[0] >> 3U) & 0x1FU);
    const int g1_5 = static_cast<int>((blk[1] >> 3U) & 0x1FU);
    const int b1_5 = static_cast<int>((blk[2] >> 3U) & 0x1FU);
    const int r2_5 = r1_5 + detail::sign_extend_3(blk[0] & 0x7U);
    const int g2_5 = g1_5 + detail::sign_extend_3(blk[1] & 0x7U);
    const int b2_5 = b1_5 + detail::sign_extend_3(blk[2] & 0x7U);

    if (r2_5 < 0 || r2_5 > 31) {
        // T-mode
        const std::uint32_t r1_4 = ((static_cast<std::uint32_t>(blk[0]) >> 3U) & 0x3U) << 2U |
                                   (static_cast<std::uint32_t>(blk[0]) & 0x3U);
        const std::uint8_t r1 = detail::expand4(r1_4);
        const std::uint8_t g1 = detail::expand4((blk[1] >> 4U) & 0xFU);
        const std::uint8_t b1 = detail::expand4(blk[1] & 0xFU);
        const std::uint8_t r2 = detail::expand4((blk[2] >> 4U) & 0xFU);
        const std::uint8_t g2 = detail::expand4(blk[2] & 0xFU);
        const std::uint8_t b2 = detail::expand4((blk[3] >> 4U) & 0xFU);
        const unsigned d_idx = (((blk[3] >> 2U) & 0x3U) << 1U) | (blk[3] & 0x1U);
        const int dist = kTHDist[d_idx];
        const std::uint8_t paint[4][3] = {
            {r1, g1, b1},
            {detail::clamp_u8(r2 + dist), detail::clamp_u8(g2 + dist), detail::clamp_u8(b2 + dist)},
            {r2, g2, b2},
            {detail::clamp_u8(r2 - dist), detail::clamp_u8(g2 - dist), detail::clamp_u8(b2 - dist)},
        };
        for (int y = 0; y < 4; ++y) {
            for (int x = 0; x < 4; ++x) {
                const std::uint32_t idx = read_pixel_index(x, y);
                out_rgb[y][x][0] = paint[idx][0];
                out_rgb[y][x][1] = paint[idx][1];
                out_rgb[y][x][2] = paint[idx][2];
            }
        }
        return;
    }

    if (g2_5 < 0 || g2_5 > 31) {
        // H-mode
        const std::uint8_t r1 = detail::expand4((blk[0] >> 3U) & 0xFU);
        const std::uint8_t g1 = detail::expand4(((blk[0] & 0x7U) << 1U) | ((blk[1] >> 4U) & 0x1U));
        const std::uint8_t b1 =
            detail::expand4((blk[1] & 0x8U) | ((blk[1] & 0x3U) << 1U) | ((blk[2] >> 7U) & 0x1U));
        const std::uint8_t r2 = detail::expand4((blk[2] >> 3U) & 0xFU);
        const std::uint8_t g2 = detail::expand4(((blk[2] & 0x7U) << 1U) | ((blk[3] >> 7U) & 0x1U));
        const std::uint8_t b2 = detail::expand4((blk[3] >> 3U) & 0xFU);
        const std::uint32_t c1 =
            (static_cast<std::uint32_t>(r1) << 16U) |
            (static_cast<std::uint32_t>(g1) << 8U) |
            static_cast<std::uint32_t>(b1);
        const std::uint32_t c2 =
            (static_cast<std::uint32_t>(r2) << 16U) |
            (static_cast<std::uint32_t>(g2) << 8U) |
            static_cast<std::uint32_t>(b2);
        const unsigned d_idx = (((blk[3] >> 2U) & 0x1U) << 2U) |
                               ((blk[3] & 0x1U) << 1U) |
                               (c1 >= c2 ? 1U : 0U);
        const int dist = kTHDist[d_idx];
        const std::uint8_t paint[4][3] = {
            {detail::clamp_u8(r1 + dist), detail::clamp_u8(g1 + dist), detail::clamp_u8(b1 + dist)},
            {detail::clamp_u8(r1 - dist), detail::clamp_u8(g1 - dist), detail::clamp_u8(b1 - dist)},
            {detail::clamp_u8(r2 + dist), detail::clamp_u8(g2 + dist), detail::clamp_u8(b2 + dist)},
            {detail::clamp_u8(r2 - dist), detail::clamp_u8(g2 - dist), detail::clamp_u8(b2 - dist)},
        };
        for (int y = 0; y < 4; ++y) {
            for (int x = 0; x < 4; ++x) {
                const std::uint32_t idx = read_pixel_index(x, y);
                out_rgb[y][x][0] = paint[idx][0];
                out_rgb[y][x][1] = paint[idx][1];
                out_rgb[y][x][2] = paint[idx][2];
            }
        }
        return;
    }

    if (b2_5 < 0 || b2_5 > 31) {
        // Planar mode
        const int ro = detail::expand6((blk[0] >> 1U) & 0x3FU);
        const int go = detail::expand7(((blk[0] & 0x1U) << 6U) | ((blk[1] >> 1U) & 0x3FU));
        const int bo = detail::expand6(((blk[1] & 0x1U) << 5U) |
                                       (((blk[2] >> 3U) & 0x3U) << 3U) |
                                       ((blk[2] & 0x3U) << 1U) |
                                       ((blk[3] >> 7U) & 0x1U));
        const int rh = detail::expand6((((blk[3] >> 2U) & 0x1FU) << 1U) | (blk[3] & 0x1U));
        const int gh = detail::expand7((blk[4] >> 1U) & 0x7FU);
        const int bh = detail::expand6(((blk[4] & 0x1U) << 5U) | ((blk[5] >> 3U) & 0x1FU));
        const int rv = detail::expand6(((blk[5] & 0x7U) << 3U) | ((blk[6] >> 5U) & 0x7U));
        const int gv = detail::expand7(((blk[6] & 0x1FU) << 2U) | ((blk[7] >> 6U) & 0x3U));
        const int bv = detail::expand6(blk[7] & 0x3FU);
        for (int y = 0; y < 4; ++y) {
            for (int x = 0; x < 4; ++x) {
                out_rgb[y][x][0] = detail::clamp_u8((x * (rh - ro) + y * (rv - ro) + 4 * ro + 2) >> 2);
                out_rgb[y][x][1] = detail::clamp_u8((x * (gh - go) + y * (gv - go) + 4 * go + 2) >> 2);
                out_rgb[y][x][2] = detail::clamp_u8((x * (bh - bo) + y * (bv - bo) + 4 * bo + 2) >> 2);
            }
        }
        return;
    }

    // Differential mode
    const std::uint8_t r[2] = {detail::expand5(static_cast<std::uint32_t>(r1_5)),
                               detail::expand5(static_cast<std::uint32_t>(r2_5))};
    const std::uint8_t g[2] = {detail::expand5(static_cast<std::uint32_t>(g1_5)),
                               detail::expand5(static_cast<std::uint32_t>(g2_5))};
    const std::uint8_t b[2] = {detail::expand5(static_cast<std::uint32_t>(b1_5)),
                               detail::expand5(static_cast<std::uint32_t>(b2_5))};
    const unsigned t[2] = {(blk[3] >> 5U) & 0x7U, (blk[3] >> 2U) & 0x7U};
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            const unsigned sub = flip ? (y >= 2 ? 1U : 0U) : (x >= 2 ? 1U : 0U);
            const int mod = kModifier[t[sub]][read_pixel_index(x, y)];
            out_rgb[y][x][0] = detail::clamp_u8(static_cast<int>(r[sub]) + mod);
            out_rgb[y][x][1] = detail::clamp_u8(static_cast<int>(g[sub]) + mod);
            out_rgb[y][x][2] = detail::clamp_u8(static_cast<int>(b[sub]) + mod);
        }
    }
}

inline void decode_eac_alpha_block(const std::uint8_t* blk, std::uint8_t out_a[4][4]) noexcept {
    static constexpr int kEacAlpha[16][8] = {
        {-3, -6, -9, -15, 2, 5, 8, 14},
        {-3, -7, -10, -13, 2, 6, 9, 12},
        {-2, -5, -8, -13, 1, 4, 7, 12},
        {-2, -4, -6, -13, 1, 3, 5, 12},
        {-3, -6, -8, -12, 2, 5, 7, 11},
        {-3, -7, -9, -11, 2, 6, 8, 10},
        {-4, -7, -8, -11, 3, 6, 7, 10},
        {-3, -5, -8, -11, 2, 4, 7, 10},
        {-2, -6, -8, -10, 1, 5, 7, 9},
        {-2, -5, -8, -10, 1, 4, 7, 9},
        {-2, -4, -8, -10, 1, 3, 7, 9},
        {-2, -5, -7, -10, 1, 4, 6, 9},
        {-3, -4, -7, -10, 2, 3, 6, 9},
        {-1, -2, -3, -10, 0, 1, 2, 9},
        {-4, -6, -8, -9, 3, 5, 7, 8},
        {-3, -5, -7, -9, 2, 4, 6, 8},
    };
    const int base = static_cast<int>(blk[0]);
    const int mul = static_cast<int>((blk[1] >> 4U) & 0xFU);
    const unsigned tbl = blk[1] & 0xFU;
    const std::uint64_t bits48 =
        (static_cast<std::uint64_t>(blk[2]) << 40U) |
        (static_cast<std::uint64_t>(blk[3]) << 32U) |
        (static_cast<std::uint64_t>(blk[4]) << 24U) |
        (static_cast<std::uint64_t>(blk[5]) << 16U) |
        (static_cast<std::uint64_t>(blk[6]) << 8U) |
        static_cast<std::uint64_t>(blk[7]);
    for (int x = 0; x < 4; ++x) {
        for (int y = 0; y < 4; ++y) {
            const unsigned p = static_cast<unsigned>(4 * x + y);
            const unsigned idx = static_cast<unsigned>((bits48 >> (45U - 3U * p)) & 0x7ULL);
            out_a[y][x] = detail::clamp_u8(base + kEacAlpha[tbl][idx] * mul);
        }
    }
}

[[nodiscard]] inline bool decode_to_rgba8(const Image& image,
                                          std::uint32_t face,
                                          std::uint32_t level,
                                          std::vector<std::uint8_t>& out) {
    out.clear();
    if (image.format != Format::rgb8 && image.format != Format::rgba8) {
        return false;
    }
    if (face >= image.faces || level >= image.levels) {
        return false;
    }
    const std::size_t idx =
        static_cast<std::size_t>(face) * static_cast<std::size_t>(image.levels) + level;
    if (idx >= image.data.size()) {
        return false;
    }
    const Level& lvl = image.data[idx];
    const std::uint32_t w = lvl.width;
    const std::uint32_t h = lvl.height;
    if (w == 0U || h == 0U) {
        return false;
    }
    const std::uint64_t bx_count = (static_cast<std::uint64_t>(w) + 3ULL) / 4ULL;
    const std::uint64_t by_count = (static_cast<std::uint64_t>(h) + 3ULL) / 4ULL;
    const std::size_t stride = block_size_bytes(image.format);
    if (lvl.blocks.size() < bx_count * by_count * stride) {
        return false;
    }

    out.assign(static_cast<std::size_t>(w) * h * 4U, 255U);
    for (std::uint32_t by = 0; by < by_count; ++by) {
        for (std::uint32_t bx = 0; bx < bx_count; ++bx) {
            const std::uint8_t* blk =
                lvl.blocks.data() + (static_cast<std::size_t>(by) * bx_count + bx) * stride;
            std::uint8_t rgb[4][4][3]{};
            std::uint8_t a[4][4]{};
            if (image.format == Format::rgba8) {
                decode_eac_alpha_block(blk, a);
                decode_etc2_rgb8_block(blk + 8U, rgb);
            } else {
                decode_etc2_rgb8_block(blk, rgb);
            }
            const std::uint32_t rows = (std::min)(4U, h - by * 4U);
            const std::uint32_t cols = (std::min)(4U, w - bx * 4U);
            for (std::uint32_t py = 0; py < rows; ++py) {
                for (std::uint32_t px = 0; px < cols; ++px) {
                    const std::size_t dst_idx =
                        (static_cast<std::size_t>(by * 4U + py) * w + (bx * 4U + px)) * 4U;
                    out[dst_idx + 0U] = rgb[py][px][0];
                    out[dst_idx + 1U] = rgb[py][px][1];
                    out[dst_idx + 2U] = rgb[py][px][2];
                    out[dst_idx + 3U] = (image.format == Format::rgba8) ? a[py][px] : 255U;
                }
            }
        }
    }
    return true;
}

inline std::vector<std::uint8_t> decode_level_rgba8(const Level& lvl, Format fmt) {
    Image img;
    img.format = fmt;
    img.width = lvl.width;
    img.height = lvl.height;
    img.levels = 1U;
    img.faces = 1U;
    img.data.push_back(lvl);
    std::vector<std::uint8_t> out;
    (void)decode_to_rgba8(img, 0U, 0U, out);
    return out;
}

inline double compute_rgb_psnr_db(std::span<const std::uint8_t> ref_rgba,
                                  std::span<const std::uint8_t> test_rgba) {
    if (ref_rgba.size() != test_rgba.size() || ref_rgba.empty()) {
        return 0.0;
    }
    double sum_sq = 0.0;
    const std::size_t pixels = ref_rgba.size() / 4U;
    for (std::size_t i = 0; i < pixels; ++i) {
        for (std::size_t c = 0; c < 3U; ++c) {
            const double d = static_cast<double>(ref_rgba[i * 4U + c]) -
                             static_cast<double>(test_rgba[i * 4U + c]);
            sum_sq += d * d;
        }
    }
    const double mse = sum_sq / (3.0 * static_cast<double>(pixels));
    if (mse == 0.0) {
        return 99.0;
    }
    return 10.0 * std::log10((255.0 * 255.0) / mse);
}

inline int max_alpha_diff(std::span<const std::uint8_t> ref_rgba,
                          std::span<const std::uint8_t> test_rgba) {
    int worst = 0;
    const std::size_t pixels = ref_rgba.size() / 4U;
    for (std::size_t i = 0; i < pixels; ++i) {
        const int d = std::abs(static_cast<int>(ref_rgba[i * 4U + 3U]) -
                               static_cast<int>(test_rgba[i * 4U + 3U]));
        if (d > worst) {
            worst = d;
        }
    }
    return worst;
}

}  // namespace wse::asset::etc2
