// Clean-room DDS -> ETC2 RGB8 / ETC2 RGBA8 (EAC alpha) transcoder backed by etcpak.
#pragma once

#include <cstdint>
#include <string>

#include "wse/asset/dds.h"
#include "wse/asset/etc2_types.h"

namespace wse::asset::etc2 {

// Transcodes a parsed DDS (`wse/asset/dds.h`, use decode_dds_mip_rgba8 to get pixels) starting at
// mip `first_mip`. DXT1 without any transparent texel -> rgb8; DXT1 with 1-bit alpha, DXT3, DXT5,
// and uncompressed images with alpha -> rgba8; uncompressed without alpha -> rgb8.
// Level sizes smaller than 4 are padded to a 4x4 block by edge replication. `threads` > 1 may
// split work across std::threads (never more than requested). Returns false with *error on failure.
bool transcode_dds(const DdsImageView& view,
                   std::uint32_t first_mip,
                   Image& out,
                   std::string* error = nullptr,
                   unsigned threads = 1);

}  // namespace wse::asset::etc2
