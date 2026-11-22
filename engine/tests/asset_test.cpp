// Host unit tests for clean-room v1.1.5.6 .brf and .dds asset decoders
// (engine/include/wse/asset/brf.h and engine/include/wse/asset/dds.h).
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "wse/asset/brf.h"
#include "wse/asset/dds.h"
#include "wse/asset/etc2.h"
#include "wse/asset/shader.h"
#include "wse/asset/shader_compiler.h"
#include "wse/asset/texture_cache.h"
#include "wse/game/module_loader_common.h"
#include "wse/io/binary_stream.h"
#include "wse/platform/dynres.h"
#include "wse/platform/postfx_shaders.h"
#include "wse/platform/vfs.h"
#include "wse/recomp/d3d9_shader.h"

namespace {

int failures = 0;

void expect(bool cond, const char* msg) {
    if (!cond) {
        ++failures;
        std::printf("FAIL  %s\n", msg);
    }
}

std::vector<std::uint8_t> to_u8(std::span<const std::byte> bytes) {
    std::vector<std::uint8_t> out(bytes.size());
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        out[i] = std::to_integer<std::uint8_t>(bytes[i]);
    }
    return out;
}

void push_u16_le(wse::io::BinaryWriter& w, std::uint16_t v) {
    w.write_byte(static_cast<std::uint8_t>(v & 0xFFU));
    w.write_byte(static_cast<std::uint8_t>((v >> 8) & 0xFFU));
}

void push_u32_le(wse::io::BinaryWriter& w, std::uint32_t v) {
    w.write_u32(v);
}

std::uint16_t pack_rgb565(int r8, int g8, int b8) {
    const std::uint16_t r = static_cast<std::uint16_t>((r8 >> 3) & 0x1F);
    const std::uint16_t g = static_cast<std::uint16_t>((g8 >> 2) & 0x3F);
    const std::uint16_t b = static_cast<std::uint16_t>((b8 >> 3) & 0x1F);
    return static_cast<std::uint16_t>((r << 11) | (g << 5) | b);
}

void write_dds_header(wse::io::BinaryWriter& w,
                      std::uint32_t width,
                      std::uint32_t height,
                      std::uint32_t flags,
                      std::uint32_t mip_count,
                      std::uint32_t pf_flags,
                      std::uint32_t fourcc,
                      std::uint32_t bpp,
                      std::uint32_t r_mask,
                      std::uint32_t g_mask,
                      std::uint32_t b_mask,
                      std::uint32_t a_mask,
                      std::uint32_t caps = 0x1000U,
                      std::uint32_t caps2 = 0U,
                      std::uint32_t depth = 1U) {
    w.write_u32(wse::asset::dds_detail::fourcc('D', 'D', 'S', ' '));
    w.write_u32(124U);
    w.write_u32(flags);
    w.write_u32(height);
    w.write_u32(width);
    w.write_u32(0U);     // pitch
    w.write_u32(depth);  // depth
    w.write_u32(mip_count);
    for (int i = 0; i < 11; ++i) w.write_u32(0U);
    w.write_u32(32U);
    w.write_u32(pf_flags);
    w.write_u32(fourcc);
    w.write_u32(bpp);
    w.write_u32(r_mask);
    w.write_u32(g_mask);
    w.write_u32(b_mask);
    w.write_u32(a_mask);
    w.write_u32(caps);
    w.write_u32(caps2);
    w.write_u32(0U);
    w.write_u32(0U);
    w.write_u32(0U);
}

void write_vec3(wse::io::BinaryWriter& w, float x, float y, float z) {
    w.write_f32(x);
    w.write_f32(y);
    w.write_f32(z);
}

void test_brf_unversioned_all_eight_types() {
    wse::io::BinaryWriter w;

    // 1. texture
    w.write_string("texture");
    w.write_i32(1);
    w.write_string("sword_tex.dds");
    w.write_u32(0x10000007U);  // bit 0x10000000 masked off by rglTexture decode

    // 2. shader
    w.write_string("shader");
    w.write_i32(1);
    w.write_string("standart_shader");
    w.write_u32(0x80002001U);  // clears 0x2000 because 0x80002000 set
    w.write_u32(0x4U);
    w.write_string("tech_main");
    w.write_i32(2);
    w.write_string("fallback_1");
    w.write_string("fallback_2");
    w.write_i32(1);
    w.write_i32(3);
    w.write_u32(0x18U);  // (color_op & 0x1F) == 0x18 sets bit 0x40000 on shader.flags
    w.write_u32(0x1U);
    w.write_u32(0x2U);

    // 3. material
    w.write_string("material");
    w.write_i32(1);
    w.write_string("sword_mat");
    w.write_u32(0x12U);
    w.write_string("standart_shader");
    w.write_string("sword_tex.dds");
    w.write_string("none");
    w.write_string("none");
    w.write_string("none");
    w.write_string("none");
    w.write_f32(12.5f);
    write_vec3(w, 0.25f, 0.5f, 0.75f);

    // 4. mesh (unversioned -> no tangents, dual UVs)
    w.write_string("mesh");
    w.write_i32(1);
    w.write_string("sword_a");
    w.write_u32(0x30001U);  // even with rfver==0 bits, tangents/single_uv stay false!
    w.write_string("sword_mat");
    w.write_u32(3U);
    write_vec3(w, 0.0f, 0.0f, 0.0f);
    write_vec3(w, 1.0f, 0.0f, 0.0f);
    write_vec3(w, 0.0f, 1.0f, 0.0f);
    w.write_u32(1U);  // 1 vertex group
    w.write_i32(5);
    w.write_u32(2U);
    w.write_i32(0);
    w.write_f32(0.75f);
    w.write_i32(1);
    w.write_f32(0.25f);
    w.write_i32(1);  // 1 vertex key
    w.write_i32(10);
    w.write_u32(1U);
    write_vec3(w, 0.0f, 0.0f, 1.0f);
    w.write_u32(1U);
    write_vec3(w, 0.0f, 1.0f, 0.0f);
    w.write_u32(3U);  // 3 corners
    for (int i = 0; i < 3; ++i) {
        w.write_i32(i);
        w.write_u32(0xFFFFFFFFU);
        write_vec3(w, 0.0f, 0.0f, 1.0f);
        w.write_f32(0.1f * static_cast<float>(i));
        w.write_f32(0.2f * static_cast<float>(i));
        w.write_f32(0.3f * static_cast<float>(i));
        w.write_f32(0.4f * static_cast<float>(i));
    }
    w.write_u32(1U);
    w.write_i32(0);
    w.write_i32(1);
    w.write_i32(2);

    // 5. body (composite containing sphere, capsule, face, manifold)
    w.write_string("body");
    w.write_i32(1);
    w.write_string("bo_sword");
    w.write_string("composite");
    w.write_i32(4);
    w.write_string("sphere");
    w.write_f32(2.5f);
    write_vec3(w, 1.0f, 2.0f, 3.0f);
    w.write_u32(0x1U);
    w.write_string("capsule");
    w.write_f32(0.5f);
    write_vec3(w, 0.0f, 0.0f, 0.0f);
    write_vec3(w, 0.0f, 0.0f, 1.5f);
    w.write_u32(0x2U);
    w.write_string("face");
    w.write_i32(3);
    write_vec3(w, 0.0f, 0.0f, 0.0f);
    write_vec3(w, 1.0f, 0.0f, 0.0f);
    write_vec3(w, 0.0f, 1.0f, 0.0f);
    w.write_u32(0x4U);
    w.write_string("manifold");
    w.write_i32(3);
    write_vec3(w, 0.0f, 0.0f, 0.0f);
    write_vec3(w, 2.0f, 0.0f, 0.0f);
    write_vec3(w, 0.0f, 2.0f, 0.0f);
    w.write_i32(1);
    w.write_i32(-1);
    w.write_u32(0x8U);
    w.write_i32(3);
    w.write_i32(0);
    w.write_i32(1);
    w.write_i32(2);

    // 6. skeleton
    w.write_string("skeleton");
    w.write_i32(1);
    w.write_string("skel_human");
    w.write_i32(2);
    w.write_i32(-1);
    w.write_string("abdomen");
    w.write_u32(0U);
    write_vec3(w, 1.0f, 0.0f, 0.0f);
    write_vec3(w, 0.0f, 1.0f, 0.0f);
    write_vec3(w, 0.0f, 0.0f, 1.0f);
    write_vec3(w, 0.0f, 0.0f, 0.9f);
    w.write_i32(0);
    w.write_string("thorax");
    w.write_u32(1U);
    write_vec3(w, 1.0f, 0.0f, 0.0f);
    write_vec3(w, 0.0f, 1.0f, 0.0f);
    write_vec3(w, 0.0f, 0.0f, 1.0f);
    write_vec3(w, 0.0f, 0.0f, 0.25f);

    // 7. skeleton_anim (stream quaternion order: qx, qy, qz, qw; 0x14030d0e0 + 0x1402b6840)
    w.write_string("skeleton_anim");
    w.write_i32(1);
    w.write_string("anim_swing");
    w.write_u32(1U);
    w.write_i32(1);
    w.write_i32(15);
    w.write_f32(0.1f);  // qx
    w.write_f32(0.2f);  // qy
    w.write_f32(0.3f);  // qz
    w.write_f32(0.9f);  // qw
    w.write_i32(1);
    w.write_i32(15);
    write_vec3(w, 0.1f, 0.2f, 0.3f);

    // 8. physics_material
    w.write_string("physics_material");
    w.write_i32(1);
    w.write_string("phys_wood");
    w.write_u32(0x3U);
    w.write_f32(0.625f);

    w.write_string("end");

    wse::asset::BrfFile file;
    std::string err;
    expect(wse::asset::parse_brf(w.bytes(), file, &err), "parse unversioned brf");
    expect(file.rfver == 0, "unversioned rfver == 0");
    expect(file.textures.size() == 1 && file.textures[0].flags == 0x7U, "texture masks bit 0x10000000");
    expect(file.shaders.size() == 1 && file.shaders[0].flags == 0x80040001U && file.shaders[0].alternatives.size() == 2,
           "shader flag rules match 0x140302750");
    expect(file.materials.size() == 1 && file.materials[0].specular_coeff == 12.5f &&
               file.materials[0].texture_names[0] == "sword_tex.dds",
           "material parsed");
    expect(file.meshes.size() == 1 && !file.meshes[0].has_tangents && !file.meshes[0].single_uv &&
               file.meshes[0].vertex_groups.size() == 1 && file.meshes[0].vertex_groups[0].bone_no == 5 &&
               file.meshes[0].face_corners[2].uv2.y == 0.8f,
           "unversioned mesh parsed");
    expect(file.bodies.size() == 1 && file.bodies[0].root.kind == wse::asset::BrfBodyPartKind::Composite &&
               file.bodies[0].root.children.size() == 4 &&
               file.bodies[0].root.children[3].manifold_faces[0].vertex_indices.size() == 3,
           "composite body parsed");
    expect(file.skeletons.size() == 1 && file.skeletons[0].bones.size() == 2 &&
               file.skeletons[0].bones[1].parent_index == 0 && file.skeletons[0].bones[1].name == "thorax",
           "skeleton parsed");
    expect(file.animations.size() == 1 && file.animations[0].bone_rotations.size() == 1 &&
               file.animations[0].bone_rotations[0][0].rotation.x == 0.1f &&
               file.animations[0].bone_rotations[0][0].rotation.y == 0.2f &&
               file.animations[0].bone_rotations[0][0].rotation.z == 0.3f &&
               file.animations[0].bone_rotations[0][0].rotation.w == 0.9f,
           "skeleton_anim reads qx/qy/qz/qw into Quatf{x,y,z,w}");
    expect(file.physics_materials.size() == 1 && file.physics_materials[0].sound_occlusion == 0.625f,
           "physics_material reads m_soundOcclusion");
}

void test_brf_rfver1_metamesh_and_catalog() {
    // rfver 1 mesh with tangents + single_uv
    wse::io::BinaryWriter w;
    w.write_string("rfver ");
    w.write_i32(1);
    w.write_string("mesh");
    w.write_i32(7);

    const std::array<const char*, 7> names = {
        "castle", "castle.1", "castle.lod1", "castle.LOD2", "castle.lod3.1", "castle.lod4", "prop_x"
    };
    for (std::size_t n = 0; n < names.size(); ++n) {
        w.write_string(names[n]);
        w.write_u32(n == 0 ? 0x30000U : 0U);  // tangents + single_uv on mesh 0 only
        w.write_string("stone");
        w.write_u32(1U);
        write_vec3(w, 1.0f, 2.0f, 3.0f);
        w.write_u32(0U);  // vertex groups
        w.write_i32(0);   // vertex keys
        w.write_u32(3U);
        for (int c = 0; c < 3; ++c) {
            w.write_i32(0);
            w.write_u32(0U);
            write_vec3(w, 0.0f, 0.0f, 1.0f);
            if (n == 0) {
                write_vec3(w, 1.0f, 0.0f, 0.0f);
                w.write_bool(true);
            }
            w.write_f32(0.5f);
            w.write_f32(0.25f);
            if (n != 0) {
                w.write_f32(0.9f);
                w.write_f32(0.8f);
            }
        }
        w.write_u32(1U);
        w.write_i32(0);
        w.write_i32(1);
        w.write_i32(2);
    }
    w.write_string("end");

    wse::asset::BrfFile rf1;
    std::string err;
    expect(wse::asset::parse_brf(w.bytes(), rf1, &err), "parse rfver 1 brf");
    expect(rf1.rfver == 1 && rf1.meshes.size() == 7, "rfver 1 count");
    expect(rf1.meshes[0].has_tangents && rf1.meshes[0].single_uv &&
               rf1.meshes[0].face_corners[0].binormal_sign &&
               rf1.meshes[0].face_corners[0].uv2.x == 0.5f,
           "rfver 1 tangents and single_uv decode");

    const auto groups = wse::asset::group_meta_meshes(rf1.meshes);
    expect(groups.size() == 2 && groups[0].base_name == "castle" && groups[1].base_name == "prop_x",
           "MetaMesh grouping count and names");
    expect(groups[0].lod_mesh_indices[0].size() == 2 &&
               groups[0].lod_mesh_indices[1].size() == 1 &&
               groups[0].lod_mesh_indices[2].size() == 1 &&
               groups[0].lod_mesh_indices[3].size() == 1 &&
               groups[0].lod_mesh_indices[4].size() == 1,
           "MetaMesh LOD buckets [0..4]");

    // Verified createMetaMesh (0x1402f3030) edge cases: dotted non-LOD names land in LOD 0,
    // ".lod3_alt" still selects LOD 3, and prefixes shorter than 2 chars before '.' are not split.
    {
        std::vector<wse::asset::BrfMesh> edge_meshes(4);
        edge_meshes[0].name = "tree";
        edge_meshes[1].name = "tree.1";
        edge_meshes[2].name = "tree.lod3_alt";
        edge_meshes[3].name = "x.lod1";
        const auto edge_groups = wse::asset::group_meta_meshes(edge_meshes);
        expect(edge_groups.size() == 2 &&
                   edge_groups[0].base_name == "tree" &&
                   edge_groups[0].lod_mesh_indices[0].size() == 2 &&
                   edge_groups[0].lod_mesh_indices[3].size() == 1 &&
                   edge_groups[1].base_name == "x.lod1" &&
                   edge_groups[1].lod_mesh_indices[0].size() == 1,
               "MetaMesh createMetaMesh prefix >= 2 and dotted/prefix LOD rules");
    }

    // Rejections
    {
        wse::io::BinaryWriter bad;
        bad.write_string("bogus");
        bad.write_i32(0);
        bad.write_string("end");
        wse::asset::BrfFile out;
        expect(!wse::asset::parse_brf(bad.bytes(), out, &err), "rejects unknown resource type");
    }
    {
        wse::io::BinaryWriter bad;
        bad.write_string("shader");
        bad.write_i32(1);
        bad.write_string("s");
        bad.write_u32(0);
        bad.write_u32(0);
        bad.write_string("t");
        bad.write_i32(4);  // exceeds 3 alternatives
        wse::asset::BrfFile out;
        expect(!wse::asset::parse_brf(bad.bytes(), out, &err), "rejects > 3 shader alternatives");
    }
    {
        wse::io::BinaryWriter bad;
        bad.write_string("mesh");
        bad.write_i32(1);
        bad.write_string("m");
        bad.write_u32(0);
        bad.write_string("mat");
        bad.write_u32(1);
        write_vec3(bad, 0, 0, 0);
        bad.write_u32(0);
        bad.write_i32(0);
        bad.write_u32(1);
        bad.write_i32(99);  // invalid vertex index
        wse::asset::BrfFile out;
        expect(!wse::asset::parse_brf(bad.bytes(), out, &err), "rejects invalid vertex index");
    }
    {
        wse::io::BinaryWriter bad;
        bad.write_string("mesh");
        bad.write_i32(1);
        bad.write_string("m");
        bad.write_u32(0);
        bad.write_string("mat");
        bad.write_u32(1);
        write_vec3(bad, 0, 0, 0);
        bad.write_u32(1);
        bad.write_i32(64);  // bone_no >= 64
        wse::asset::BrfFile out;
        expect(!wse::asset::parse_brf(bad.bytes(), out, &err), "rejects >= 64 bone index");
    }
    {
        const auto truncated = std::span<const std::byte>(w.bytes()).subspan(0, 12);
        wse::asset::BrfFile out;
        expect(!wse::asset::parse_brf(truncated, out, &err), "rejects truncated brf");
    }

    // Catalog loading via MemoryVfs + module.ini
    wse::platform::MemoryVfs vfs;
    const auto rf1_u8 = to_u8(w.bytes());
    wse::io::BinaryWriter empty_brf;
    empty_brf.write_string("end");
    const auto empty_u8 = to_u8(empty_brf.bytes());

    vfs.write_file("warband/CommonRes/core_shaders.brf", empty_u8);
    vfs.write_file("warband/CommonRes/core_materials.brf", empty_u8);
    vfs.write_file("warband/CommonRes/shared_props.brf", rf1_u8);
    vfs.write_file("warband/Modules/Mod/Resource/mod_weapons.brf", rf1_u8);

    wse::game::ModuleIniData ini{};
    wse::game::parse_module_ini(
        "load_resource = shared_props\nload_mod_resource = mod_weapons\nload_resource_nofast = extra_missing\n",
        ini);
    expect(ini.load_resources.size() == 2 && ini.load_mod_resources.size() == 1,
           "module ini captures load_resource_nofast");

    wse::asset::BrfCatalog cat;
    expect(wse::asset::load_module_brf_catalog(vfs, "warband", "warband/Modules/Mod", ini, cat, &err),
           "load_module_brf_catalog succeeds");
    // 6 core + 1 optional wse2_shaders + 2 load_resources + 1 load_mod_resource = 10 entries
    expect(cat.entries.size() == 10 && cat.entries[0].loaded && !cat.entries[6].loaded &&
               cat.entries[7].loaded && cat.entries[9].loaded && cat.entries[9].from_module,
           "catalog loads core + common + mod brf and warns on missing non-optional");
    expect(cat.warnings.size() == 5, "missing non-optional core + nofast logged as warnings (not optional)");
}

bool contains(const std::string& haystack, std::string_view needle) {
    return haystack.find(needle) != std::string::npos;
}

void write_dx10_header(wse::io::BinaryWriter& w,
                       std::uint32_t dxgi_format,
                       std::uint32_t res_dim,
                       std::uint32_t misc,
                       std::uint32_t array_size) {
    w.write_u32(dxgi_format);
    w.write_u32(res_dim);
    w.write_u32(misc);
    w.write_u32(array_size);
    w.write_u32(0U);
}

void test_binary_stream_strings() {
    wse::io::BinaryWriter w;
    w.write_string("");
    w.write_string("abc");
    w.write_string(std::string_view("a\0b", 3));
    wse::io::BinaryReader r(w.bytes());
    std::string a, b, c;
    expect(r.read_string(a) && a.empty() && r.read_string(b) && b == "abc" &&
               r.read_string(c) && c.size() == 3 && c[1] == '\0' && r.at_end(),
           "BinaryWriter/Reader string round-trip (empty, ascii, embedded NUL)");

    wse::io::BinaryWriter big;
    big.write_string("hello");
    wse::io::BinaryReader rb(big.bytes());
    std::string out;
    expect(!rb.read_string(out, 4) && rb.cursor() == 4U, "read_string rejects length above max_len");

    wse::io::BinaryWriter neg;
    neg.write_i32(-1);
    wse::io::BinaryReader rn(neg.bytes());
    expect(!rn.read_string(out), "read_string rejects negative length");

    wse::io::BinaryWriter overrun;
    overrun.write_i32(10);
    overrun.write_byte('x');
    wse::io::BinaryReader ro(overrun.bytes());
    expect(!ro.read_string(out), "read_string rejects length past end of stream");

    std::array<std::byte, 2> buf{};
    wse::io::BinaryReader rs(big.bytes());
    expect(rs.skip(4U) && rs.read_bytes(buf) && buf[0] == std::byte{'h'} && rs.cursor() == 6U &&
               !rs.skip(100U) && rs.cursor() == 6U,
           "skip / read_bytes bounds");
}

void test_brf_hardening() {
    std::string err;
    {
        // Declared count far larger than the in-memory reservation cap: the header is
        // consistent with the stream size (4 B per record) but each BrfMesh is ~200 B.
        constexpr std::size_t kCount = 200000;
        wse::io::BinaryWriter w;
        w.write_string("mesh");
        w.write_i32(static_cast<std::int32_t>(kCount));
        // 0xFF padding makes the first record's name length -1, so parsing stops at
        // record 0 and the capacity reflects only the up-front reservation.
        const std::vector<std::byte> pad(kCount * 4U, std::byte{0xFF});
        w.write_bytes(pad);
        wse::asset::BrfFile out;
        expect(!wse::asset::parse_brf(w.bytes(), out, &err), "huge mesh count still fails to parse");
        expect(out.meshes.capacity() <= wse::asset::detail::kMaxReserveElements,
               "section reservation is capped regardless of declared count");
    }
    {
        // Specific count diagnostic is not overwritten by a generic message.
        wse::io::BinaryWriter w;
        w.write_string("mesh");
        w.write_i32(1);
        w.write_string("m");
        w.write_u32(0);
        w.write_string("mat");
        w.write_u32(1000000U);
        wse::asset::BrfFile out;
        err.clear();
        expect(!wse::asset::parse_brf(w.bytes(), out, &err) && contains(err, "mesh positions in m") &&
                   contains(err, "exceeds remaining stream bytes"),
               "count overflow keeps its specific error message");
    }
    {
        wse::io::BinaryWriter w;
        w.write_string("texture");
        w.write_i32(-2);
        wse::asset::BrfFile out;
        err.clear();
        expect(!wse::asset::parse_brf(w.bytes(), out, &err) && contains(err, "negative count"),
               "negative section count reported as negative");
    }
    {
        wse::io::BinaryWriter w;
        w.write_string("skeleton");
        w.write_i32(1);
        w.write_string("skel");
        wse::asset::BrfFile out;
        err.clear();
        expect(!wse::asset::parse_brf(w.bytes(), out, &err) && contains(err, "truncated skeleton bones in skel count"),
               "truncated count reported as truncated");
    }
    {
        // Vertex index 65536 is valid against the position table but cannot fit
        // rglMeshFaceCorner::m_vertexNo (unsigned short).
        constexpr std::uint32_t kPositions = 65537U;
        wse::io::BinaryWriter w;
        w.write_string("mesh");
        w.write_i32(1);
        w.write_string("big");
        w.write_u32(0);
        w.write_string("mat");
        w.write_u32(kPositions);
        for (std::uint32_t i = 0; i < kPositions; ++i) write_vec3(w, 0, 0, 0);
        w.write_u32(0);  // vertex groups
        w.write_i32(0);  // vertex keys
        w.write_u32(1);  // face corners
        w.write_i32(65536);
        w.write_u32(0);
        write_vec3(w, 0, 0, 1);
        w.write_f32(0);
        w.write_f32(0);
        w.write_f32(0);
        w.write_f32(0);
        w.write_u32(0);  // faces
        w.write_string("end");
        wse::asset::BrfFile out;
        err.clear();
        expect(!wse::asset::parse_brf(w.bytes(), out, &err) && contains(err, "Invalid vertex index 65536"),
               "rejects vertex index above unsigned short range");
    }
    {
        // Composite nesting beyond the depth limit is rejected rather than recursing.
        wse::io::BinaryWriter w;
        w.write_string("body");
        w.write_i32(1);
        w.write_string("deep");
        for (int i = 0; i < 10; ++i) {
            w.write_string("composite");
            w.write_i32(1);
        }
        w.write_string("sphere");
        w.write_f32(1);
        write_vec3(w, 0, 0, 0);
        w.write_u32(0);
        w.write_string("end");
        wse::asset::BrfFile out;
        err.clear();
        expect(!wse::asset::parse_brf(w.bytes(), out, &err) && contains(err, "depth limit"),
               "rejects composite body nesting beyond depth limit");
    }
}

void test_dds_hardening() {
    std::string err;
    const std::uint32_t argb_masks[4] = {0x00FF0000U, 0x0000FF00U, 0x000000FFU, 0xFF000000U};
    {
        // Legacy multi-slice volume texture (DDSCAPS2_VOLUME + DDSD_DEPTH, depth = 4).
        wse::io::BinaryWriter w;
        write_dds_header(w, 4, 4, 0x1007U | 0x800000U, 1, 0x41U, 0U, 32U, argb_masks[0], argb_masks[1],
                         argb_masks[2], argb_masks[3], 0x1000U, 0x200000U, 4U);
        const std::vector<std::byte> px(4U * 4U * 4U * 4U * 4U);
        w.write_bytes(px);
        wse::asset::DdsImageView img;
        err.clear();
        expect(!wse::asset::parse_dds(w.bytes(), img, &err) && contains(err, "volume"),
               "rejects legacy multi-slice volume DDS");
    }
    {
        // DDSCAPS2_VOLUME without DDSD_DEPTH is still a single slice when dwDepth is 0 or 1.
        for (const std::uint32_t depth : {0U, 1U}) {
            wse::io::BinaryWriter w;
            write_dds_header(w, 4, 4, 0x1007U, 1, 0x41U, 0U, 32U, argb_masks[0], argb_masks[1],
                             argb_masks[2], argb_masks[3], 0x1000U, 0x200000U, depth);
            const std::vector<std::byte> px(4U * 4U * 4U);
            w.write_bytes(px);
            wse::asset::DdsImageView img;
            err.clear();
            expect(wse::asset::parse_dds(w.bytes(), img, &err) && img.slices.size() == 1U,
                   "accepts single-slice DDSCAPS2_VOLUME without DDSD_DEPTH");
        }
    }
    {
        // DDSCAPS2_VOLUME without DDSD_DEPTH but dwDepth = 4 is a multi-slice volume.
        wse::io::BinaryWriter w;
        write_dds_header(w, 4, 4, 0x1007U, 1, 0x41U, 0U, 32U, argb_masks[0], argb_masks[1],
                         argb_masks[2], argb_masks[3], 0x1000U, 0x200000U, 4U);
        const std::vector<std::byte> px(4U * 4U * 4U * 4U);
        w.write_bytes(px);
        wse::asset::DdsImageView img;
        err.clear();
        expect(!wse::asset::parse_dds(w.bytes(), img, &err) && contains(err, "volume"),
               "rejects DDSCAPS2_VOLUME with dwDepth > 1 and no DDSD_DEPTH");
    }
    {
        // Plain 2D texture (no DDSD_DEPTH, no DDSCAPS2_VOLUME) with junk dwDepth: depth ignored.
        wse::io::BinaryWriter w;
        write_dds_header(w, 4, 4, 0x1007U, 1, 0x41U, 0U, 32U, argb_masks[0], argb_masks[1],
                         argb_masks[2], argb_masks[3], 0x1000U, 0U, 7U);
        const std::vector<std::byte> px(4U * 4U * 4U);
        w.write_bytes(px);
        wse::asset::DdsImageView img;
        err.clear();
        expect(wse::asset::parse_dds(w.bytes(), img, &err) && img.slices.size() == 1U &&
                   img.width == 4U && img.height == 4U,
               "ignores dwDepth on a 2D DDS without DDSD_DEPTH");
    }
    {
        // Single-slice DDSCAPS2_VOLUME + DDSD_DEPTH + depth == 1 (retail Warband cnt/font.dds).
        wse::io::BinaryWriter w;
        write_dds_header(w, 4, 4, 0x1007U | 0x800000U, 1, 0x41U, 0U, 32U, argb_masks[0], argb_masks[1],
                         argb_masks[2], argb_masks[3], 0x1000U, 0x200000U, 1U);
        const std::vector<std::byte> px(4U * 4U * 4U);
        w.write_bytes(px);
        wse::asset::DdsImageView img;
        err.clear();
        expect(wse::asset::parse_dds(w.bytes(), img, &err) && img.slices.size() == 1U,
               "accepts single-slice DDSCAPS2_VOLUME with DDSD_DEPTH and depth == 1");
    }
    {
        // DX10 TEXTURE3D with dwDepth = 4.
        wse::io::BinaryWriter w;
        write_dds_header(w, 4, 4, 0x1007U | 0x800000U, 1, 0x4U, wse::asset::dds_detail::fourcc('D', 'X', '1', '0'),
                         0, 0, 0, 0, 0, 0x1000U, 0x200000U, 4U);
        write_dx10_header(w, 28U, 4U, 0U, 1U);
        const std::vector<std::byte> px(4U * 4U * 4U * 4U);
        w.write_bytes(px);
        wse::asset::DdsImageView img;
        err.clear();
        expect(!wse::asset::parse_dds(w.bytes(), img, &err) && contains(err, "volume"),
               "rejects multi-slice DX10 TEXTURE3D");
    }
    {
        // DX10 TEXTURE3D with a single slice (legacy header depth 1, no DDSD_DEPTH) is 2D.
        wse::io::BinaryWriter w;
        write_dds_header(w, 4, 4, 0x1007U, 1, 0x4U, wse::asset::dds_detail::fourcc('D', 'X', '1', '0'),
                         0, 0, 0, 0, 0);
        write_dx10_header(w, 28U, 4U, 0U, 1U);
        const std::vector<std::byte> px(4U * 4U * 4U);
        w.write_bytes(px);
        wse::asset::DdsImageView img;
        std::vector<std::uint8_t> rgba;
        err.clear();
        expect(wse::asset::parse_dds(w.bytes(), img, &err) && img.slices.size() == 1U &&
                   wse::asset::decode_dds_mip_rgba8(img, 0, 0, rgba) && rgba.size() == 64U,
               "accepts single-slice DX10 TEXTURE3D");
    }
    {
        // DX10 cube array of 2 cubes with the legacy face bits also set: 12 faces.
        wse::io::BinaryWriter w;
        write_dds_header(w, 4, 4, 0x1007U, 1, 0x4U, wse::asset::dds_detail::fourcc('D', 'X', '1', '0'),
                         0, 0, 0, 0, 0, 0x1008U, 0x200U | 0xFC00U);
        write_dx10_header(w, 71U, 3U, 0x4U, 2U);
        const std::vector<std::byte> blocks(12U * 8U);
        w.write_bytes(blocks);
        wse::asset::DdsImageView img;
        std::vector<std::uint8_t> rgba;
        expect(wse::asset::parse_dds(w.bytes(), img, &err) && img.is_cubemap && img.face_count == 12U &&
                   img.slices.size() == 12U && wse::asset::decode_dds_mip_rgba8(img, 0, 11, rgba),
               "DX10 cube array counts six faces per array element");
        expect(!wse::asset::parse_dds(std::span<const std::byte>(w.bytes()).subspan(0, w.size() - 8U), img, &err),
               "DX10 cube array with a missing face is truncated");
    }
    {
        // Legacy partial cube map keeps the face-bit count.
        wse::io::BinaryWriter w;
        write_dds_header(w, 4, 4, 0x1007U, 1, 0x4U, wse::asset::dds_detail::fourcc('D', 'X', 'T', '1'),
                         0, 0, 0, 0, 0, 0x1008U, 0x200U | 0x400U | 0x800U);
        const std::vector<std::byte> blocks(2U * 8U);
        w.write_bytes(blocks);
        wse::asset::DdsImageView img;
        expect(wse::asset::parse_dds(w.bytes(), img, &err) && img.face_count == 2U,
               "legacy partial cube map counts face bits");
    }
    {
        // sRGB flag for DX10 *_SRGB formats only.
        for (const std::uint32_t dxgi : {71U, 72U}) {
            wse::io::BinaryWriter w;
            write_dds_header(w, 4, 4, 0x1007U, 1, 0x4U, wse::asset::dds_detail::fourcc('D', 'X', '1', '0'),
                             0, 0, 0, 0, 0);
            write_dx10_header(w, dxgi, 3U, 0U, 1U);
            const std::vector<std::byte> block(8U);
            w.write_bytes(block);
            wse::asset::DdsImageView img;
            expect(wse::asset::parse_dds(w.bytes(), img, &err) && img.format == wse::asset::DdsFormat::BC1_UNORM &&
                       img.is_srgb == (dxgi == 72U),
                   "DX10 sRGB flag set only for BC1_UNORM_SRGB");
        }
    }
    {
        // Truncated mip chain: header claims 3 mips, payload holds 2.
        wse::io::BinaryWriter w;
        write_dds_header(w, 4, 4, 0x1007U | 0x20000U, 3, 0x41U, 0U, 32U, argb_masks[0], argb_masks[1],
                         argb_masks[2], argb_masks[3]);
        const std::vector<std::byte> px((16U + 4U) * 4U);
        w.write_bytes(px);
        wse::asset::DdsImageView img;
        err.clear();
        expect(!wse::asset::parse_dds(w.bytes(), img, &err) && contains(err, "truncated DDS mip payload"),
               "rejects truncated mip chain");
    }
    {
        // Hand-modified views cannot make the decoder read out of bounds.
        wse::io::BinaryWriter w;
        write_dds_header(w, 4, 4, 0x1007U, 1, 0x41U, 0U, 32U, argb_masks[0], argb_masks[1],
                         argb_masks[2], argb_masks[3]);
        const std::vector<std::byte> px(64U);
        w.write_bytes(px);
        wse::asset::DdsImageView img;
        std::vector<std::uint8_t> rgba;
        expect(wse::asset::parse_dds(w.bytes(), img, &err), "parse 4x4 BGRA for tamper tests");
        wse::asset::DdsImageView short_slice = img;
        short_slice.slices[0].data = short_slice.slices[0].data.subspan(0, 32U);
        expect(!wse::asset::decode_dds_mip_rgba8(short_slice, 0, 0, rgba, &err),
               "decode rejects slice payload smaller than its dimensions");
        wse::asset::DdsImageView no_slices = img;
        no_slices.slices.clear();
        expect(!wse::asset::decode_dds_mip_rgba8(no_slices, 0, 0, rgba, &err),
               "decode rejects slice table that does not match face/mip count");
        wse::asset::DdsImageView grown = img;
        grown.slices[0].width = 8U;
        expect(!wse::asset::decode_dds_mip_rgba8(grown, 0, 0, rgba, &err),
               "decode rejects slice dimensions larger than its payload");
    }
}

void test_dds_formats_and_resolver() {
    // 1. DXT1 4-color opaque (c0 > c1) and 1-bit transparent (c0 <= c1)
    {
        wse::io::BinaryWriter w;
        write_dds_header(w, 4, 4, 0x1007U, 1, 0x4U, wse::asset::dds_detail::fourcc('D', 'X', 'T', '1'),
                         0, 0, 0, 0, 0);
        const std::uint16_t c0 = pack_rgb565(255, 255, 255);  // 0xFFFF
        const std::uint16_t c1 = pack_rgb565(0, 0, 0);        // 0x0000 -> c0 > c1
        push_u16_le(w, c0);
        push_u16_le(w, c1);
        push_u32_le(w, 0xE4U);  // px 0:0, px 1:1, px 2:2, px 3:3

        wse::asset::DdsImageView img;
        std::string err;
        expect(wse::asset::parse_dds(w.bytes(), img, &err), "parse DXT1 opaque");
        std::vector<std::uint8_t> rgba;
        expect(wse::asset::decode_dds_mip_rgba8(img, 0, 0, rgba, &err), "decode DXT1 opaque");
        expect(rgba.size() == 64 && rgba[0] == 255 && rgba[3] == 255 &&
                   rgba[4] == 0 && rgba[7] == 255 && rgba[8] == 170 && rgba[12] == 85,
               "DXT1 4-color palette matches spec");
    }
    {
        wse::io::BinaryWriter w;
        write_dds_header(w, 4, 4, 0x1007U, 1, 0x4U, wse::asset::dds_detail::fourcc('D', 'X', 'T', '1'),
                         0, 0, 0, 0, 0);
        push_u16_le(w, 0x0000U);  // c0 <= c1 -> index 3 is transparent
        push_u16_le(w, 0xFFFFU);
        push_u32_le(w, 0xE4U);
        wse::asset::DdsImageView img;
        std::vector<std::uint8_t> rgba;
        expect(wse::asset::parse_dds(w.bytes(), img) && wse::asset::decode_dds_mip_rgba8(img, 0, 0, rgba),
               "decode DXT1 1-bit alpha");
        expect(rgba[8] == 127 && rgba[11] == 255 && rgba[15] == 0, "DXT1 1-bit alpha transparent pixel");
    }

    // 2. DXT3 explicit 4-bit alpha (also c0 <= c1 still uses 4-color mode)
    {
        wse::io::BinaryWriter w;
        write_dds_header(w, 4, 4, 0x1007U, 1, 0x4U, wse::asset::dds_detail::fourcc('D', 'X', 'T', '3'),
                         0, 0, 0, 0, 0);
        push_u16_le(w, 0xF500U);  // row 0: px0=0, px1=0, px2=5 (->0x55), px3=15 (->0xFF)
        push_u16_le(w, 0U);
        push_u16_le(w, 0U);
        push_u16_le(w, 0U);
        push_u16_le(w, 0x0000U);
        push_u16_le(w, 0xFFFFU);
        push_u32_le(w, 0xE4U);
        wse::asset::DdsImageView img;
        std::vector<std::uint8_t> rgba;
        expect(wse::asset::parse_dds(w.bytes(), img) && wse::asset::decode_dds_mip_rgba8(img, 0, 0, rgba),
               "decode DXT3");
        expect(rgba[3] == 0 && rgba[11] == 0x55 && rgba[15] == 255 && rgba[12] == 170,
               "DXT3 alpha expansion and 4-color interpolation");
    }

    // 3. DXT5 both alpha modes + BC4
    {
        wse::io::BinaryWriter w;
        write_dds_header(w, 4, 4, 0x1007U, 1, 0x4U, wse::asset::dds_detail::fourcc('D', 'X', 'T', '5'),
                         0, 0, 0, 0, 0);
        // 8-alpha mode (a0=200 > a1=60): px0 idx0->200, px1 idx1->60, px2 idx2->180
        w.write_byte(200);
        w.write_byte(60);
        const std::uint64_t idx = (0ULL << 0) | (1ULL << 3) | (2ULL << 6);
        for (int b = 0; b < 6; ++b) w.write_byte(static_cast<std::uint8_t>((idx >> (b * 8)) & 0xFFULL));
        push_u16_le(w, 0xFFFFU);
        push_u16_le(w, 0xFFFFU);
        push_u32_le(w, 0U);
        wse::asset::DdsImageView img;
        std::vector<std::uint8_t> rgba;
        expect(wse::asset::parse_dds(w.bytes(), img) && wse::asset::decode_dds_mip_rgba8(img, 0, 0, rgba),
               "decode DXT5 8-alpha");
        expect(rgba[3] == 200 && rgba[7] == 60 && rgba[11] == 180, "DXT5 8-alpha interpolation");
    }
    {
        wse::io::BinaryWriter w;
        write_dds_header(w, 4, 4, 0x1007U, 1, 0x4U, wse::asset::dds_detail::fourcc('D', 'X', 'T', '5'),
                         0, 0, 0, 0, 0);
        // 6-alpha mode (a0=50 <= a1=200): idx6=0, idx7=255, idx2=(4*50+200)/5=80
        w.write_byte(50);
        w.write_byte(200);
        const std::uint64_t idx = (6ULL << 0) | (7ULL << 3) | (2ULL << 6);
        for (int b = 0; b < 6; ++b) w.write_byte(static_cast<std::uint8_t>((idx >> (b * 8)) & 0xFFULL));
        push_u16_le(w, 0xFFFFU);
        push_u16_le(w, 0xFFFFU);
        push_u32_le(w, 0U);
        wse::asset::DdsImageView img;
        std::vector<std::uint8_t> rgba;
        expect(wse::asset::parse_dds(w.bytes(), img) && wse::asset::decode_dds_mip_rgba8(img, 0, 0, rgba),
               "decode DXT5 6-alpha");
        expect(rgba[3] == 0 && rgba[7] == 255 && rgba[11] == 80, "DXT5 6-alpha interpolation");
    }

    // 4. Uncompressed B8G8R8A8 + B8G8R8 + L8 + mip chain slicing + DX10
    {
        wse::io::BinaryWriter w;
        write_dds_header(w, 2, 2, 0x21007U, 2, 0x41U, 0U, 32U,
                         0x00FF0000U, 0x0000FF00U, 0x000000FFU, 0xFF000000U);
        // mip 0 (4 pixels BGRA): [B, G, R, A]
        for (int i = 0; i < 4; ++i) {
            w.write_byte(10);
            w.write_byte(20);
            w.write_byte(30);
            w.write_byte(40);
        }
        // mip 1 (1x1): [B=1, G=2, R=3, A=255]
        w.write_byte(1);
        w.write_byte(2);
        w.write_byte(3);
        w.write_byte(255);

        wse::asset::DdsImageView img;
        std::vector<std::uint8_t> rgba0, rgba1;
        expect(wse::asset::parse_dds(w.bytes(), img) && img.mip_count == 2 && img.slices.size() == 2,
               "parse uncompressed BGRA with 2 mips");
        expect(wse::asset::decode_dds_mip_rgba8(img, 0, 0, rgba0) &&
                   wse::asset::decode_dds_mip_rgba8(img, 1, 0, rgba1),
               "decode mip 0 and mip 1");
        expect(rgba0[0] == 30 && rgba0[1] == 20 && rgba0[2] == 10 && rgba0[3] == 40 &&
                   rgba1[0] == 3 && rgba1[1] == 2 && rgba1[2] == 1 && rgba1[3] == 255,
               "BGRA channel swizzle");
    }
    {
        // DX10 BC1 2x2
        wse::io::BinaryWriter w;
        write_dds_header(w, 2, 2, 0x1007U, 1, 0x4U, wse::asset::dds_detail::fourcc('D', 'X', '1', '0'),
                         0, 0, 0, 0, 0);
        w.write_u32(71U);  // DXGI_FORMAT_BC1_UNORM
        w.write_u32(3U);   // TEXTURE2D
        w.write_u32(0U);
        w.write_u32(1U);
        w.write_u32(0U);
        push_u16_le(w, 0xFFFFU);
        push_u16_le(w, 0U);
        push_u32_le(w, 0U);
        wse::asset::DdsImageView img;
        std::vector<std::uint8_t> rgba;
        expect(wse::asset::parse_dds(w.bytes(), img) && img.has_dx10_header &&
                   img.format == wse::asset::DdsFormat::BC1_UNORM &&
                   wse::asset::decode_dds_mip_rgba8(img, 0, 0, rgba) && rgba.size() == 16,
               "parse and decode DX10 BC1_UNORM");
    }
    {
        // 16-bit luminance (L16_UNORM, used by Warband skybox_*_exp.dds)
        wse::io::BinaryWriter w;
        write_dds_header(w, 2, 1, 0x1007U, 1, 0x20000U, 0U, 16U, 0xFFFFU, 0U, 0U, 0U);
        push_u16_le(w, 0x1234U);
        push_u16_le(w, 0xABCDU);
        wse::asset::DdsImageView img;
        std::vector<std::uint8_t> rgba;
        expect(wse::asset::parse_dds(w.bytes(), img) &&
                   img.format == wse::asset::DdsFormat::L16_UNORM &&
                   !img.has_alpha &&
                   wse::asset::decode_dds_mip_rgba8(img, 0, 0, rgba) && rgba.size() == 8 &&
                   rgba[0] == 0x12 && rgba[1] == 0x12 && rgba[2] == 0x12 && rgba[3] == 255 &&
                   rgba[4] == 0xAB && rgba[7] == 255,
               "parse and decode L16_UNORM");
    }

    // 5. DDS error checks
    {
        std::array<std::byte, 4> bad{std::byte{'N'}, std::byte{'O'}, std::byte{'P'}, std::byte{'E'}};
        wse::asset::DdsImageView img;
        std::string err;
        expect(!wse::asset::parse_dds(bad, img, &err), "reject bad DDS magic");
    }

    // 6. Texture path resolution priority (tryGetTexture 0x1402f35a0 + initDeviceObjects 0x1403184d0)
    {
        expect(wse::asset::texture_frame_count(0U) == 1U &&
                   wse::asset::texture_frame_count(0x01000000U) == 4U &&
                   wse::asset::texture_frame_count(0x07000000U) == 28U &&
                   wse::asset::texture_frame_count(0x0F000000U) == 60U,
               "texture_frame_count matches ((flags >> 24) & 0xF) * 4");

        wse::platform::MemoryVfs vfs;
        const std::vector<std::uint8_t> dummy{1};
        vfs.write_file("warband/Textures/ui_icon.dds", dummy);
        vfs.write_file("warband/Textures/languages/fr/ui_icon.dds", dummy);
        vfs.write_file("warband/TexturesHD/ui_icon.dds", dummy);
        vfs.write_file("warband/Modules/Mod/Textures/ui_icon.dds", dummy);
        vfs.write_file("warband/Modules/Mod/Textures/languages/fr/ui_icon.dds", dummy);
        vfs.write_file("warband/Textures/languages/fr/lang_only.dds", dummy);
        vfs.write_file("warband/Textures/lang_only.dds", dummy);
        vfs.write_file("warband/Textures/base_only.tga", dummy);
        vfs.write_file("warband/TexturesHD/hd_only.dds", dummy);
        vfs.write_file("warband/Textures/rock.1.dds", dummy);

        vfs.write_file("warband/Textures/waterbump_0.dds", dummy);
        vfs.write_file("warband/TexturesHD/waterbump_0.dds", dummy);
        vfs.write_file("warband/Textures/languages/fr/waterbump_0.dds", dummy);
        vfs.write_file("warband/Modules/Mod/Textures/waterbump_0.dds", dummy);
        vfs.write_file("warband/Modules/Mod/Textures/languages/fr/waterbump_0.dds", dummy);
        vfs.write_file("warband/Textures/waterbump_5.dds", dummy);
        vfs.write_file("warband/Textures/languages/fr/anim_lang_0.dds", dummy);
        vfs.write_file("warband/TexturesHD/anim_lang_0.dds", dummy);

        expect(wse::asset::resolve_texture_path(vfs, "Textures\\ui_icon", "warband", "warband/Modules/Mod", "fr", true) ==
                   "warband/Modules/Mod/Textures/languages/fr/ui_icon.dds",
               "module localized texture takes highest priority");
        expect(wse::asset::resolve_texture_path(vfs, "ui_icon", "warband", "warband/Modules/Mod", "", true) ==
                   "warband/Modules/Mod/Textures/ui_icon.dds",
               "module texture overrides base TexturesHD, base language, and Textures");
        expect(wse::asset::resolve_texture_path(vfs, "ui_icon", "warband", "", "fr", true) ==
                   "warband/TexturesHD/ui_icon.dds",
               "TexturesHD overrides base Textures/languages when enabled");
        expect(wse::asset::resolve_texture_path(vfs, "lang_only", "warband", "", "fr", true) ==
                   "warband/Textures/languages/fr/lang_only.dds",
               "base Textures/languages used when absent from module and TexturesHD");
        expect(wse::asset::resolve_texture_path(vfs, "base_only", "warband", "", "", true) ==
                   "warband/Textures/base_only.tga",
               "probes .dds -> exact -> .jpg -> .tga");
        expect(wse::asset::resolve_texture_path(vfs, "rock.1", "warband", "", "", true) ==
                   "warband/Textures/rock.1.dds",
               "dotted texture stem probes .dds via tryGetTexture");
        expect(wse::asset::resolve_texture_path(vfs, "hd_only.dds", "warband", "", "", false).empty() &&
                   wse::asset::resolve_texture_path(vfs, "hd_only.dds", "warband", "", "", true) ==
                       "warband/TexturesHD/hd_only.dds",
               "TexturesHD searched only when enabled");

        expect(wse::asset::resolve_texture_path(vfs, "waterbump.dds", "warband", "warband/Modules/Mod", "fr", true, 0) ==
                   "warband/Modules/Mod/Textures/languages/fr/waterbump_0.dds",
               "indexed frame prefers module language");
        expect(wse::asset::resolve_texture_path(vfs, "waterbump", "warband", "warband/Modules/Mod", "", true, 0) ==
                   "warband/Modules/Mod/Textures/waterbump_0.dds",
               "indexed frame checks module Textures");
        expect(wse::asset::resolve_texture_path(vfs, "anim_lang.dds", "warband", "", "fr", true, 0) ==
                   "warband/Textures/languages/fr/anim_lang_0.dds",
               "indexed frame skips TexturesHD and uses base language");
        expect(wse::asset::resolve_texture_path(vfs, "waterbump.dds", "warband", "", "", true, 0) ==
                   "warband/Textures/waterbump_0.dds",
               "indexed frame skips TexturesHD and falls back to base Textures");
        expect(wse::asset::resolve_texture_path(vfs, "waterbump", "warband", "", "", false, 5) ==
                   "warband/Textures/waterbump_5.dds",
               "indexed frame formats <stem>_<frame>.dds");

        // An empty warband_root is the VFS root (vfs_join_path("", x) == x).
        wse::platform::MemoryVfs root_vfs;
        root_vfs.write_file("Textures/rootless.dds", dummy);
        root_vfs.write_file("TexturesHD/rootless_hd.dds", dummy);
        root_vfs.write_file("Textures/languages/fr/rootless_lang.dds", dummy);
        root_vfs.write_file("Textures/rootless_anim_2.dds", dummy);
        expect(wse::asset::resolve_texture_path(root_vfs, "rootless", "") == "Textures/rootless.dds" &&
                   wse::asset::resolve_texture_path(root_vfs, "rootless_hd", "", "", "", true) ==
                       "TexturesHD/rootless_hd.dds" &&
                   wse::asset::resolve_texture_path(root_vfs, "rootless_lang", "", "", "fr") ==
                       "Textures/languages/fr/rootless_lang.dds" &&
                   wse::asset::resolve_texture_path(root_vfs, "rootless_anim", "", "", "", false, 2) ==
                       "Textures/rootless_anim_2.dds",
               "empty warband_root probes base directories at the VFS root");
    }
}

// Counts is_file/is_dir calls on top of a MemoryVfs.
class CountingVfs final : public wse::platform::Vfs {
public:
    wse::platform::MemoryVfs inner;
    std::size_t is_file_calls = 0;
    std::size_t is_dir_calls = 0;

    bool read_file(std::string_view path, std::vector<std::uint8_t>& out) override {
        return inner.read_file(path, out);
    }
    bool write_file(std::string_view path, std::span<const std::uint8_t> data) override {
        return inner.write_file(path, data);
    }
    bool remove_file(std::string_view path) override { return inner.remove_file(path); }
    bool list_dir(std::string_view path, std::vector<wse::platform::VfsEntry>& out) override {
        return inner.list_dir(path, out);
    }
    bool is_file(std::string_view path) override {
        ++is_file_calls;
        return inner.is_file(path);
    }
    bool is_dir(std::string_view path) override {
        ++is_dir_calls;
        return inner.is_dir(path);
    }
};

std::vector<std::byte> dds_with_payload(std::uint32_t width, std::uint32_t height, std::uint32_t fourcc,
                                        const std::vector<std::byte>& payload) {
    wse::io::BinaryWriter w;
    write_dds_header(w, width, height, 0x1007U, 1, 0x4U, fourcc, 0, 0, 0, 0, 0);
    w.write_bytes(payload);
    return std::vector<std::byte>(w.bytes().begin(), w.bytes().end());
}

void test_dds_decode_blocks_and_resolver_probes() {
    using wse::asset::dds_detail::fourcc;
    // Deterministic block data: any 16 bytes are a valid BC2/BC3 block.
    std::vector<std::byte> blocks(4U * 16U);
    std::uint32_t seed = 0x12345678U;
    for (std::byte& b : blocks) {
        seed = seed * 1664525U + 1013904223U;
        b = static_cast<std::byte>(seed >> 24);
    }
    for (const std::uint32_t cc : {fourcc('D', 'X', 'T', '3'), fourcc('D', 'X', 'T', '5')}) {
        // 8x8 and 6x5 share the same 2x2 block grid; 6x5 must be the top-left crop of 8x8.
        const auto full_bytes = dds_with_payload(8, 8, cc, blocks);
        const auto crop_bytes = dds_with_payload(6, 5, cc, blocks);
        wse::asset::DdsImageView full_img, crop_img;
        std::vector<std::uint8_t> full, crop;
        const bool ok = wse::asset::parse_dds(full_bytes, full_img) &&
                        wse::asset::parse_dds(crop_bytes, crop_img) &&
                        wse::asset::decode_dds_mip_rgba8(full_img, 0, 0, full) &&
                        wse::asset::decode_dds_mip_rgba8(crop_img, 0, 0, crop) &&
                        full.size() == 8U * 8U * 4U && crop.size() == 6U * 5U * 4U;
        bool same = ok;
        for (std::size_t y = 0; same && y < 5U; ++y) {
            for (std::size_t x = 0; x < 6U * 4U; ++x) {
                if (crop[y * 6U * 4U + x] != full[y * 8U * 4U + x]) {
                    same = false;
                    break;
                }
            }
        }
        expect(same, "edge blocks of a non-multiple-of-4 image are clipped, not shifted");

        // A reused buffer of the same size is fully overwritten.
        std::vector<std::uint8_t> dirty(full.size(), 0xA5U);
        expect(wse::asset::decode_dds_mip_rgba8(full_img, 0, 0, dirty) && dirty == full,
               "decode into a reused dirty buffer matches a fresh decode");
        std::vector<std::uint8_t> larger(4096U, 0xA5U);
        expect(wse::asset::decode_dds_mip_rgba8(crop_img, 0, 0, larger) && larger == crop,
               "decode into a larger reused buffer shrinks it to the image");
    }
    {
        // Uncompressed path into a dirty buffer.
        wse::io::BinaryWriter w;
        write_dds_header(w, 3, 1, 0x1007U, 1, 0x41U, 0U, 32U, 0x00FF0000U, 0x0000FF00U, 0x000000FFU,
                         0xFF000000U);
        for (int i = 0; i < 12; ++i) w.write_byte(static_cast<std::uint8_t>(i * 7));
        wse::asset::DdsImageView img;
        std::vector<std::uint8_t> fresh, dirty(12U, 0xA5U);
        expect(wse::asset::parse_dds(w.bytes(), img) && wse::asset::decode_dds_mip_rgba8(img, 0, 0, fresh) &&
                   wse::asset::decode_dds_mip_rgba8(img, 0, 0, dirty) && dirty == fresh,
               "uncompressed decode into a reused dirty buffer matches a fresh decode");
    }
    {
        // DX10 TEXTURE3D must have array_size == 1 and no TEXTURECUBE flag.
        const std::pair<std::uint32_t, std::uint32_t> bad[] = {{0U, 2U}, {0x4U, 1U}};
        for (const auto& [misc, array_size] : bad) {
            wse::io::BinaryWriter w;
            write_dds_header(w, 4, 4, 0x1007U, 1, 0x4U, fourcc('D', 'X', '1', '0'), 0, 0, 0, 0, 0);
            write_dx10_header(w, 28U, 4U, misc, array_size);
            const std::vector<std::byte> px(4U * 4U * 4U * 6U * 2U);
            w.write_bytes(px);
            wse::asset::DdsImageView img;
            std::string err;
            expect(!wse::asset::parse_dds(w.bytes(), img, &err) && contains(err, "volume"),
                   "rejects DX10 TEXTURE3D with array_size > 1 or TEXTURECUBE");
        }
    }
    {
        // A missing directory costs one is_dir call, not one is_file per candidate.
        CountingVfs vfs;
        const std::vector<std::uint8_t> dummy{0};
        vfs.write_file("warband/Textures/stone.dds", dummy);
        vfs.write_file("warband/Textures/languages/fr/sign.dds", dummy);
        vfs.is_file_calls = 0;
        vfs.is_dir_calls = 0;
        // Missing: module Textures/languages/de, module Textures, TexturesHD, base languages/de.
        expect(wse::asset::resolve_texture_path(vfs, "stone", "warband", "warband/Modules/M", "de", true) ==
                   "warband/Textures/stone.dds",
               "resolver finds the base texture past four missing directories");
        expect(vfs.is_file_calls == 1U && vfs.is_dir_calls == 5U,
               "missing directories are skipped without per-candidate is_file probes");

        // A trailing separator on the language gives the same path as vfs_join_path.
        expect(wse::asset::resolve_texture_path(vfs, "sign", "warband", "", "fr/") ==
                   "warband/Textures/languages/fr/sign.dds",
               "language with a trailing separator joins without a double slash");
    }
}

void test_p5_shader_review_regressions() {
    using namespace wse::asset;

    // XML: a `<techniques>` root no longer swallows the first technique; comments,
    // look-alike tags, `>` in quoted values, and element-text children are handled.
    {
        std::vector<GlTechniqueEntry> out;
        parse_techniques_xml(
            "<?xml version=\"1.0\"?>\n<techniques>\n"
            "  <!-- <technique name=\"dead\"/> -->\n"
            "  <technique name=\"a\" note=\"x>y\">\n"
            "    <vertex_shaders name=\"wrong.glsl\"/>\n"
            "    <vertex_shader name=\"vs_a.glsl\"/>\n"
            "    <fragment_shader> fs_a.glsl </fragment_shader>\n"
            "  </technique>\n"
            "  <technique_group name=\"not_a_technique\"/>\n"
            "  <technique name=\"b\"/>\n"
            "</techniques>\n",
            "x.xml", out);
        expect(out.size() == 2U && out[0].name == "a" && out[0].vertex_shader == "vs_a.glsl" &&
                   out[0].fragment_shader == "fs_a.glsl" && out[1].name == "b" &&
                   out[1].vertex_shader.empty(),
               "technique XML parser keeps the first technique under a <techniques> root");
    }

    // Catalog: an install with no techniques fails; variants need their base.
    {
        wse::platform::MemoryVfs empty;
        ShaderTechniqueCatalog cat{};
        std::string err;
        expect(!load_warband_shader_technique_catalog(empty, "wb", cat, &err) &&
                   cat.techniques_by_lower.empty() && !err.empty(),
               "technique catalog reports failure when no GLShaders techniques exist");

        wse::platform::MemoryVfs vfs;
        vfs.write_text("wb/GLShaders/techniques.xml",
                       "<techniques><technique name=\"simple_shading\"/></techniques>");
        vfs.write_text("wb/GLShadersOptimized/vs_simple_shading.glsl",
                       "#version 120\nattribute vec4 inPosition;\nvoid main() {}\n");
        expect(load_warband_shader_technique_catalog(vfs, "wb", cat, &err) &&
                   cat.techniques_by_lower.size() == 2U && cat.find("simple_shading") != nullptr &&
                   cat.find("simple_shading_with_skin") != nullptr &&
                   cat.find("face_shader_skin_high") == nullptr && cat.warnings.size() == 4U,
               "WSE2 skinning variants are synthesized only from present base techniques");
        const GlTechniqueEntry* syn = cat.find("simple_shading_with_skin");
        expect(syn != nullptr &&
                   syn->vs_glsl.rfind("#version 120\n#define USE_SKINNING 1\n", 0U) == 0U &&
                   (detect_glsl_feature_flags(syn->vs_glsl, VertexDeclType::Standard) &
                    kSpirvFeatSkinning) != 0U,
               "synthesized variant gets a real #define USE_SKINNING after #version");

        ShaderTechniqueCatalog names_only{};
        TechniqueCatalogOptions opts{};
        opts.load_glsl_sources = false;
        expect(load_warband_shader_technique_catalog(vfs, "wb", names_only, nullptr, opts) &&
                   names_only.find("simple_shading") != nullptr &&
                   names_only.find("simple_shading")->vs_glsl.empty(),
               "names-only technique catalog skips GLSL reads");
    }

    // Feature detection: whole identifiers outside comments; #define counts, #ifdef does not.
    {
        const auto f = [](std::string_view src) {
            return detect_glsl_feature_flags(src, VertexDeclType::Standard);
        };
        expect(f("int discarded_count; // not_instanced\n/* discard */") == kSpirvFeatNone,
               "feature detection ignores substrings and comments");
        expect(f("// #define USE_SKINNING 1\n#ifdef USE_SKINNING\n#endif\n") == kSpirvFeatNone,
               "commented-out or tested macros do not set the skinning flag");
        expect(f("void main() { if (a < 0.5) discard; }") == kSpirvFeatAlphaTest &&
                   f("#define USE_SKINNING 1\n") == kSpirvFeatSkinning &&
                   f("uniform sampler2D NORMAL_TEXTURE;") == kSpirvFeatNormalMap,
               "feature detection matches whole identifiers case-insensitively");
    }

    // Alternative resolution is linear-time on a dense, fully unsuitable graph.
    {
        std::unordered_map<std::string, BrfShader> m;
        constexpr int kN = 64;
        for (int i = 0; i < kN; ++i) {
            BrfShader sh{"s" + std::to_string(i), 0U, 0U, "t", {}, {}};
            for (int j = 0; j < kN; ++j) {
                if (j != i) sh.alternatives.push_back("S" + std::to_string(j));
            }
            m[sh.name] = sh;
        }
        m["s63"].flags = kShaderFlagUsesHlsl;
        const auto t0 = std::chrono::steady_clock::now();
        const ShaderResolutionResult r = resolve_suitable_shader("s0", m, {3, true, false});
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
        expect(r.shader != nullptr && r.shader->name == "s63" && r.hop_depth == 1 && ms < 1000,
               "dense alternative graph resolves in linear time to the 1-hop suitable shader");

        SuitableShaderResolver resolver(m, {3, true, false});
        const ShaderResolutionResult& a = resolver.resolve("S0");
        const ShaderResolutionResult& b = resolver.resolve("s0");
        expect(&a == &b && resolver.cached_count() == 1U && a.shader == r.shader,
               "SuitableShaderResolver memoizes case-folded names");
    }

    // INI: BOM, duplicate keys (last wins in entries and maps alike).
    {
        ShaderPairTable t{};
        parse_wse2_shaders_ini("\xEF\xBB\xBF[SkinFallback]\na = b\nA = c\n", t);
        expect(t.entries.size() == 1U && t.entries[0].target_shader == "c" &&
                   t.skin_fallback_map["a"] == "c" && t.warnings.size() == 1U,
               "wse2_shaders.ini handles a UTF-8 BOM and duplicate keys consistently");
    }

    // PostFx: WSE2 conditional detection by directive, not substring.
    {
        expect(effect_source_has_wse2_conditional("#if WSE2\n#endif\n") &&
                   effect_source_has_wse2_conditional("  #ifdef WSE2\n#endif\n") &&
                   effect_source_has_wse2_conditional("#if defined(WSE2)\n#endif\n") &&
                   !effect_source_has_wse2_conditional("#if WSE2_LEGACY\n#endif\n") &&
                   !effect_source_has_wse2_conditional("// #if WSE2\n"),
               "WSE2 conditional detection parses preprocessor directives");
        wse::platform::MemoryVfs v;
        v.write_text("wb/PostFx.fx", "#ifdef WSE2\n#include \"postFX_WSE2.fx\"\n#endif\n");
        PostFxResolveResult r{};
        expect(resolve_postfx_effect_source(v, "wb", "", true, r) && !r.appended_wse2_include,
               "#ifdef WSE2 PostFx is not given a duplicate include");
    }

    // Material: resolved shader flag `kShaderFlagHasSpecular` (0x20, 0x1402e8734) gates
    // specular even when material specular fields are non-zero; without a shader, material
    // properties decide.
    {
        BrfMaterial mat{};
        mat.specular_coeff = 4.0f;
        const BrfShader sh_nospec{"x", kShaderFlagUsesHlsl, 0U, "t", {}, {}};
        const BrfShader sh_spec{"y", kShaderFlagUsesHlsl | kShaderFlagHasSpecular, 0U, "t", {}, {}};
        expect(!decode_material_render_state(mat).uses_specular &&
                   !decode_material_render_state(mat, &sh_nospec).uses_specular,
               "uses_specular is false when specular_rgb is black even with coeff > 0");

        mat.specular_rgb = {0.8f, 0.8f, 0.8f};
        expect(decode_material_render_state(mat).uses_specular &&
                   !decode_material_render_state(mat, &sh_nospec).uses_specular &&
                   decode_material_render_state(mat, &sh_spec).uses_specular,
               "resolved shader kShaderFlagHasSpecular (0x20) gates material specular");
    }

    // SPIR-V cache: truncated, corrupt, placeholder, missing-backend, or mismatched-source/decl
    // files are rejected and recompiled cleanly; hits read feature flags from `OpString`.
    {
        const auto compiler = create_glslang_shader_compiler();
        wse::platform::MemoryVfs v;
        const std::string src =
            "uniform sampler2D specular_texture; void main() { gl_FragColor = vec4(1.0); }";
        const std::string fname = SpirvShaderCache::compute_cache_filename(
            SpirvStage::Fragment, VertexDeclType::Standard, src);
        std::vector<std::uint8_t> trunc(24U, 0U);
        const std::uint32_t hdr[6] = {0x07230203U, 0x00010000U,
                                      shader_detail::kLegacyPlaceholderSpirvMagic,
                                      7U, 0U, (2U << 16U) | 17U};
        std::memcpy(trunc.data(), hdr, sizeof(hdr));
        v.write_file("c/" + fname, trunc);

        SpirvShaderCache no_backend{};
        std::vector<std::uint32_t> w;
        expect(!no_backend.get_or_compile(v, "c", "t", SpirvStage::Fragment,
                                          VertexDeclType::Standard, src, w) &&
                   !no_backend.last_error.empty(),
               "SpirvShaderCache without backend fails cleanly on cache miss/corrupt entry");

        SpirvShaderCache c(compiler);
        bool hit = true;
        std::uint32_t feats = 0;
        expect(c.get_or_compile(v, "c", "t", SpirvStage::Fragment, VertexDeclType::Standard, src,
                                w, &hit, &feats) &&
                   !hit && c.compile_count == 1U && inspect_spirv_module(w).has_value() &&
                   w[2] != shader_detail::kLegacyPlaceholderSpirvMagic,
               "truncated/placeholder cache file is rejected and recompiled to real SPIR-V 1.0");
        expect(!v.is_file("c/" + fname + ".partial"), "cache write leaves no .partial file");

        SpirvShaderCache c2{};  // No compiler needed when cache hits.
        std::uint32_t feats2 = 0;
        expect(c2.get_or_compile(v, "c", "t", SpirvStage::Fragment, VertexDeclType::Standard,
                                 src, w, &hit, &feats2) &&
                   hit && feats2 == feats && (feats2 & kSpirvFeatSpecular) != 0U,
               "cache hit returns feature flags decoded from the OpString tag");

        // A valid module for another source stored under this name is not reused.
        SpirvCompileOutput other_out{};
        expect(compile_warband_glsl_to_spirv(*compiler, "void main() {}", SpirvStage::Fragment,
                                             "t", VertexDeclType::Standard, other_out),
               "compile minimal other fragment shader");
        v.write_file("c/" + fname,
                     std::span<const std::uint8_t>(
                         reinterpret_cast<const std::uint8_t*>(other_out.spirv_words.data()),
                         other_out.spirv_words.size() * 4U));
        SpirvShaderCache c3(compiler);
        expect(c3.get_or_compile(v, "c", "t", SpirvStage::Fragment, VertexDeclType::Standard,
                                 src, w, &hit) &&
                   !hit && c3.compile_count == 1U,
               "cached module whose source hash differs is recompiled");

        // Changing VertexDeclType for identical source/cache slot must also invalidate.
        SpirvShaderCache c4(compiler);
        const std::string fname_bumped = SpirvShaderCache::compute_cache_filename(
            SpirvStage::Fragment, VertexDeclType::Bumped, src);
        v.write_file("c/" + fname_bumped,
                     std::span<const std::uint8_t>(
                         reinterpret_cast<const std::uint8_t*>(w.data()), w.size() * 4U));
        expect(c4.get_or_compile(v, "c", "t", SpirvStage::Fragment, VertexDeclType::Bumped,
                                 src, w, &hit) &&
                   !hit && c4.compile_count == 1U,
               "cached module whose decl_type differs is recompiled");
    }

    // Parallel warm-up (P5 lowering review 11): misses compile on several threads, the
    // calling thread stores them, duplicates compile once, failures are reported per request.
    {
        const auto compiler = create_glslang_shader_compiler();
        wse::platform::MemoryVfs v;
        const std::string vs = "attribute vec4 inPosition; void main() { gl_Position = inPosition; }";
        const std::string fs = "void main() { gl_FragColor = vec4(0.5); }";
        const std::string fs2 = "uniform vec4 tint; void main() { gl_FragColor = tint; }";
        const std::string bad = "void main() { this is not glsl }";
        using Req = SpirvShaderCache::WarmupRequest;
        const std::vector<Req> reqs{
            Req{"t_a", SpirvStage::Vertex, VertexDeclType::Standard, vs, fs},
            Req{"t_a", SpirvStage::Fragment, VertexDeclType::Standard, fs, vs},
            Req{"t_b", SpirvStage::Fragment, VertexDeclType::Standard, fs2, vs},
            Req{"t_a", SpirvStage::Fragment, VertexDeclType::Standard, fs, vs},  // duplicate
            Req{"t_bad", SpirvStage::Fragment, VertexDeclType::Standard, bad, vs},
        };
        SpirvShaderCache warm(compiler);
        const SpirvShaderCache::WarmupReport rep = warm.warm_up(v, "w", reqs, 4U);
        expect(rep.compiled == 3U && rep.failed == 1U && rep.hits == 0U &&
                   warm.compile_count == 3U && rep.errors.size() == 1U &&
                   rep.errors[0].first == 4U && !rep.errors[0].second.empty() &&
                   !warm.last_error.empty(),
               "warm_up compiles each distinct miss once and reports the failing request");

        SpirvShaderCache later{};  // no backend: everything warmed must hit
        std::vector<std::uint32_t> words;
        bool hit = false;
        ShaderStageReflection refl{};
        expect(later.get_or_compile(v, "w", "t_b", SpirvStage::Fragment, VertexDeclType::Standard,
                                    fs2, words, &hit, nullptr, &refl, vs) &&
                   hit && inspect_spirv_module(words).has_value(),
               "a warmed module (with its reflection sidecar) hits without a compiler");

        SpirvShaderCache again(compiler);
        const SpirvShaderCache::WarmupReport rep2 = again.warm_up(v, "w", reqs, 2U);
        expect(rep2.hits == 4U && rep2.compiled == 0U && rep2.failed == 1U &&
                   again.compile_count == 0U && again.cache_hits == 4U,
               "a second warm_up hits every warmed request and retries only the failure");

        SpirvShaderCache serial(compiler);
        wse::platform::MemoryVfs v2;
        const SpirvShaderCache::WarmupReport rep3 =
            serial.warm_up(v2, "w", std::span<const Req>(reqs.data(), 3U), 1U);
        std::vector<std::uint8_t> a;
        std::vector<std::uint8_t> b;
        const std::string name = SpirvShaderCache::compute_cache_filename(
            SpirvStage::Fragment, VertexDeclType::Standard, fs2, vs);
        expect(rep3.compiled == 3U && v.read_file("w/" + name, a) && v2.read_file("w/" + name, b) &&
                   a == b,
               "one thread and four threads write identical modules");
    }

    // ensure_vfs_dir_tree tolerates an uncreatable intermediate component.
    {
        wse::platform::MemoryVfs v;
        expect(shader_detail::ensure_vfs_dir_tree(v, "a/b/c") && v.is_dir("a/b/c"),
               "ensure_vfs_dir_tree creates nested directories");
    }
}

void test_p5_shaders_materials_and_spirv_cache() {
    using namespace wse::asset;

    // 1. Retail v1.1.5.6 effect search order (`rglEffectManager::initDeviceObjects`,
    //    `rglRenderer::initPostFxShaders`, `rglGraphicsContext::checkLoadingPostEffects`).
    {
        wse::platform::MemoryVfs vfs;
        vfs.write_text("wb/mb_wse2_3_0.fxo", "root_30");
        vfs.write_text("wb/mb_2a.fxo", "root_2a");
        vfs.write_text("wb/earlyz_wse2.fxo", "root_ez_wse2");
        vfs.write_text("wb/earlyZ.fxo", "root_ez");
        vfs.write_text("wb/PostFx.fx", "float4 ps_main() : COLOR0 { return 1; }");
        vfs.write_text("wb/postFX_WSE2.fx", "// root wse2");

        EffectConfig cfg3{true, 3, true};
        expect(resolve_module_effect_path(vfs, "wb", "wb/Modules/Mod", cfg3) ==
                   "wb/mb_wse2_3_0.fxo",
               "effect search finds root mb_wse2_3_0.fxo when module overrides are absent");

        vfs.write_text("wb/Modules/Mod/mb.fx", "mod_fx");
        vfs.write_text("wb/Modules/Mod/mb_wse2_2a.fxo", "mod_2a");
        vfs.write_text("wb/Modules/Mod/mb_wse2_3_0.fxo", "mod_30");
        expect(resolve_module_effect_path(vfs, "wb", "wb/Modules/Mod", cfg3) ==
                   "wb/Modules/Mod/mb_wse2_3_0.fxo",
               "module mb_wse2_3_0.fxo takes highest priority at SM3");

        EffectConfig cfg2{true, 2, true};
        expect(resolve_module_effect_path(vfs, "wb", "wb/Modules/Mod", cfg2) ==
                   "wb/Modules/Mod/mb_wse2_2a.fxo",
               "SM2 skips mb_wse2_3_0.fxo and selects mb_wse2_2a.fxo");

        EffectConfig cfg_nowse2{false, 3, true};
        expect(resolve_module_effect_path(vfs, "wb", "wb/Modules/Mod", cfg_nowse2) ==
                   "wb/Modules/Mod/mb.fx",
               "bUseWSE2Effect=false skips mb_wse2_* and falls back to module mb.fx");

        expect(resolve_earlyz_effect_path(vfs, "wb", "wb/Modules/Mod", true) ==
                   "wb/earlyz_wse2.fxo",
               "earlyZ search picks root earlyz_wse2.fxo before earlyZ.fxo");
        vfs.write_text("wb/Modules/Mod/earlyz_wse2.fxo", "mod_ez_wse2");
        expect(resolve_earlyz_effect_path(vfs, "wb", "wb/Modules/Mod", true) ==
                   "wb/Modules/Mod/earlyz_wse2.fxo",
               "module earlyz_wse2.fxo shadows root earlyz_wse2.fxo");
        expect(resolve_earlyz_effect_path(vfs, "wb", "wb/Modules/Mod", false) ==
                   "wb/earlyZ.fxo",
               "use_wse2_effect=false skips earlyz_wse2.fxo and picks root earlyZ.fxo");

        PostFxResolveResult pfx{};
        std::string err;
        expect(resolve_postfx_effect_source(vfs, "wb", "wb/Modules/Mod", true, pfx, &err) &&
                   pfx.resolved_path == "wb/PostFx.fx" && pfx.appended_wse2_include &&
                   !pfx.module_wse2_shadows_root &&
                   pfx.source_text.find("#if WSE2\n#include \"postFX_WSE2.fx\"\n#endif") !=
                       std::string::npos,
               "PostFx.fx auto-appends #if WSE2 include when absent");

        vfs.write_text("wb/Modules/Mod/postFX_WSE2.fx", "// mod wse2");
        vfs.write_text("wb/Modules/Mod/PostFx.fx", "#if WSE2\n#include \"postFX_WSE2.fx\"\n#endif\n");
        expect(resolve_postfx_effect_source(vfs, "wb", "wb/Modules/Mod", true, pfx, &err) &&
                   pfx.resolved_path == "wb/Modules/Mod/PostFx.fx" &&
                   !pfx.appended_wse2_include && pfx.module_wse2_shadows_root,
               "explicit #if WSE2 avoids duplicate append and flags module postFX_WSE2.fx shadow");
    }

    // 2. Optional `wse2_shaders.brf` + `wse2_shaders.ini` [SkinFallback]/[NoSkinFallback].
    {
        auto build_single_shader_brf = [](std::string_view name, std::uint32_t flags,
                                          std::string_view tech) {
            wse::io::BinaryWriter w;
            w.write_string("shader");
            w.write_i32(1);
            w.write_string(name);
            w.write_u32(flags);
            w.write_u32(0U);
            w.write_string(tech);
            w.write_i32(0);  // alternatives
            w.write_i32(0);  // techniques
            w.write_string("end");
            return to_u8(w.bytes());
        };

        wse::platform::MemoryVfs vfs;
        vfs.write_file("wb/CommonRes/wse2_shaders.brf",
                       build_single_shader_brf("root_wse2_sh", kShaderFlagUsesHlsl, "t"));
        vfs.write_file("wb/Modules/M/Resource/wse2_shaders.brf",
                       build_single_shader_brf("mod_wse2_sh", kShaderFlagUsesHlsl, "t"));
        BrfFile opt_brf{};
        std::string resolved_path;
        std::string err;
        expect(load_optional_wse2_shaders_brf(vfs, "wb", "wb/Modules/M", opt_brf,
                                              &resolved_path, &err) &&
                   resolved_path == "wb/Modules/M/Resource/wse2_shaders.brf" &&
                   opt_brf.shaders.size() == 1U && opt_brf.shaders[0].name == "mod_wse2_sh",
               "module Resource/wse2_shaders.brf shadows CommonRes/wse2_shaders.brf");

        ShaderPairTable raw_pairs{};
        parse_wse2_shaders_ini(
            "[SkinFallback]\n"
            "plain_sh = skinned_sh ; inline comment\n"
            "missing_src = skinned_sh # hash comment\n"
            "[NoSkinFallback]\n"
            "skinned_sh = plain_sh\n"
            "bad_flags = skinned_sh\n",
            raw_pairs);
        expect(raw_pairs.entries.size() == 4U &&
                   raw_pairs.skin_fallback_map["plain_sh"] == "skinned_sh" &&
                   raw_pairs.noskin_fallback_map["skinned_sh"] == "plain_sh",
               "parse_wse2_shaders_ini handles sections and strips inline comments");

        std::unordered_map<std::string, BrfShader> shaders;
        shaders["plain_sh"] = BrfShader{"plain_sh", kShaderFlagUsesHlsl, 0U, "tech_a", {}, {}};
        shaders["skinned_sh"] =
            BrfShader{"skinned_sh", kShaderFlagUsesHlsl | kShaderFlagUsesSkinning, 0U, "tech_b", {}, {}};
        shaders["bad_flags"] = BrfShader{"bad_flags", kShaderFlagUsesHlsl, 0U, "tech_a", {}, {}};

        ShaderPairTable valid_pairs{};
        ShaderQualityConfig qcfg{3, true, false};
        validate_shader_pairs(raw_pairs, shaders, qcfg, valid_pairs);
        expect(valid_pairs.entries.size() == 2U && valid_pairs.warnings.size() == 2U,
               "validate_shader_pairs accepts valid pairs and warns on missing/invalid-flag pairs");
    }

    // 3. Suitability, shortest-hop alternative resolution, def_shader fallback, VertexDeclType 0..4.
    {
        BrfShader hi_sh{"hi_sh", kShaderFlagUsesHlsl, kShaderReqHiQuality, "tech_ok", {"deep_a", "shallow_ok"}, {}};
        BrfShader disabled_sh{"disabled_sh", kShaderFlagUsesHlsl | kShaderFlagRuntimeDisabled, 0U, "tech_ok", {}, {}};
        BrfShader no_hlsl{"no_hlsl", 0U, 0U, "tech_ok", {}, {}};
        BrfShader inst_sh{"inst_sh", kShaderFlagUsesHlsl | kShaderFlagUsesInstancing, 0U, "tech_ok", {}, {}};

        expect(!is_shader_suitable(disabled_sh) && !is_shader_suitable(no_hlsl),
               "is_shader_suitable requires HLSL flag and rejects disabled shaders");
        expect(!is_shader_suitable(hi_sh, {2, true, true}, true) &&
                   is_shader_suitable(hi_sh, {3, true, true}, true),
               "is_shader_suitable checks shader_quality against m_requirementFlags");
        expect(!is_shader_suitable(hi_sh, {3, true, true}, false),
               "is_shader_suitable checks HLSL technique presence when required");
        expect(!is_shader_suitable(inst_sh, {3, false, false}, true),
               "is_shader_suitable checks instancing support");

        std::unordered_map<std::string, BrfShader> map;
        map["hi_sh"] = hi_sh;
        map["deep_a"] = BrfShader{"deep_a", kShaderFlagUsesHlsl | kShaderFlagRuntimeDisabled, 0U, "tech_ok", {"deep_b"}, {}};
        map["deep_b"] = BrfShader{"deep_b", kShaderFlagUsesHlsl, 0U, "tech_ok", {}, {}};
        map["shallow_ok"] = BrfShader{"shallow_ok", kShaderFlagUsesHlsl, 0U, "tech_ok", {}, {}};
        map["cyc_a"] = BrfShader{"cyc_a", 0U, 0U, "tech_ok", {"cyc_b"}, {}};
        map["cyc_b"] = BrfShader{"cyc_b", 0U, 0U, "tech_ok", {"cyc_a"}, {}};
        map["def_shader"] = BrfShader{"def_shader", kShaderFlagUsesHlsl, 0U, "tech_ok", {}, {}};

        const ShaderResolutionResult short_res =
            resolve_suitable_shader("hi_sh", map, {2, true, false});
        expect(short_res.shader != nullptr && short_res.shader->name == "shallow_ok" &&
                   short_res.hop_depth == 1 && !short_res.used_default_fallback,
               "resolve_suitable_shader picks shortest-hop suitable alternative");

        const ShaderResolutionResult cyc_res =
            resolve_suitable_shader("cyc_a", map, {3, true, false});
        expect(cyc_res.shader != nullptr && cyc_res.shader->name == "def_shader" &&
                   cyc_res.used_default_fallback,
               "cycle in alternatives safely falls back to def_shader");

        const BrfShader sh_std{"s0", kShaderFlagUsesHlsl, 0U, "t", {}, {}};
        const BrfShader sh_bump{"s1", kShaderFlagUsesHlsl | kShaderFlagUsesNormalMap, 0U, "t", {}, {}};
        const BrfShader sh_skin{"s2", kShaderFlagUsesHlsl | kShaderFlagUsesSkinning, 0U, "t", {}, {}};
        const BrfShader sh_bskin{"s3", kShaderFlagUsesHlsl | kShaderFlagUsesNormalMap | kShaderFlagUsesSkinning, 0U, "t", {}, {}};
        const BrfShader sh_spec{"s4", kShaderFlagUsesHlsl | kShaderFlagSpecial, 0U, "t", {}, {}};
        expect(get_mesh_vertex_declaration_type(sh_std, nullptr, false) == VertexDeclType::Standard &&
                   get_mesh_vertex_declaration_type(sh_bump, nullptr, false) == VertexDeclType::Bumped &&
                   get_mesh_vertex_declaration_type(sh_skin, nullptr, true) == VertexDeclType::Skinned &&
                   get_mesh_vertex_declaration_type(sh_bskin, nullptr, true) == VertexDeclType::BumpedSkinned &&
                   get_mesh_vertex_declaration_type(sh_spec, nullptr, false) == VertexDeclType::Special,
               "get_mesh_vertex_declaration_type returns 0..4 accurately");

        const BrfShader* active = nullptr;
        const VertexDeclType swapped =
            get_mesh_vertex_declaration_type(sh_std, &sh_skin, true, &active);
        expect(swapped == VertexDeclType::Skinned && active == &sh_skin,
               "mismatched skinning selects paired shader before computing VertexDeclType");
    }

    // 4. `decode_material_render_state` flags, blend modes, alpha test, and render order.
    {
        BrfMaterial mat{};
        mat.name = "test_cutout";
        mat.flags = kMaterialFlagNoFog | kMaterialFlagNoZWrite | (2U << 12U) | (0xFU << 24U);
        mat.texture_names = {"diff_a", "none", "diff_bump", "NONE", "diff_spec"};
        mat.specular_coeff = 8.0f;
        mat.specular_rgb = {0.9f, 0.9f, 0.9f};
        const BrfShader sh_bump{"bump",
                                kShaderFlagUsesHlsl | kShaderFlagUsesNormalMap |
                                    kShaderFlagHasSpecular,
                                0U, "t", {}, {}};

        const MaterialRenderState st_cut = decode_material_render_state(mat, &sh_bump);
        expect(st_cut.no_fog && !st_cut.depth_write && st_cut.depth_test &&
                   st_cut.blend_mode == MaterialBlendMode::Opaque &&
                   st_cut.alpha_test_mode == AlphaTestMode::GreaterEqual136 &&
                   st_cut.alpha_ref_u8 == 136U && st_cut.render_order == -1 &&
                   st_cut.diffuse_a == "diff_a" && st_cut.diffuse_b.empty() &&
                   st_cut.bumpmap == "diff_bump" && st_cut.envmap.empty() &&
                   st_cut.specular_map == "diff_spec" && st_cut.uses_normal_map &&
                   st_cut.uses_specular,
               "decode_material_render_state decodes opaque cutout, nibble render_order, and slots");

        // Retail setAlphaTestMode (0x1402a0260) keeps D3DRS_ALPHATESTENABLE active when blended.
        mat.flags = (1U << 8U) | (3U << 12U) | (2U << 24U);
        const MaterialRenderState st_blend = decode_material_render_state(mat, &sh_bump);
        expect(st_blend.blend_mode == MaterialBlendMode::AlphaBlend &&
                   st_blend.alpha_test_mode == AlphaTestMode::GreaterEqual251 &&
                   st_blend.alpha_ref_u8 == 251U && st_blend.render_order == 2,
               "blended material preserves D3DRS_ALPHAREF threshold cleanly");
    }

    // 5. `GLShaders` XML + WSE2 skinning variants + SPIR-V generator + persistent VFS cache.
    {
        wse::platform::MemoryVfs vfs;
        vfs.write_text("wb/GLShaders/techniques.xml",
                       "<techniques>\n"
                       "  <technique name=\"face_shader_high\">\n"
                       "    <vertex_shader name=\"vs_face.glsl\" />\n"
                       "    <fragment_shader name=\"fs_face.glsl\" />\n"
                       "  </technique>\n"
                       "  <technique name=\"faceshader_high_specular\" />\n"
                       "  <technique name=\"faceshader_simple\" />\n"
                       "  <technique name=\"simple_shading\" />\n"
                       "  <technique name=\"simple_shading_no_filter\" />\n"
                       "</techniques>\n");
        vfs.write_text(
            "wb/GLShadersOptimized/vs_face_shader_high.glsl",
            "uniform mat4 matWorldViewProj;\n"
            "uniform mat4 matWorldArray[32];\n"
            "uniform bool use_depth_effects;\n"
            "uniform int iLightIndices[4];\n"
            "attribute vec3 inPosition;\n"
            "attribute vec4 inTangent;\n"
            "attribute vec4 inBlendWeight;\n"
            "attribute vec4 inBlendIndices;\n"
            "varying vec2 texCoord0;\n"
            "void main() {\n"
            "    vec4 p = vec4(inPosition, 1.0);\n"
            "    #ifdef USE_SKINNING\n"
            "    int b0 = int(inBlendIndices.x);\n"
            "    p = (matWorldArray[b0] * p) * inBlendWeight.x;\n"
            "    #endif\n"
            "    float bias = use_depth_effects ? float(iLightIndices[0]) * 0.0 : 0.0;\n"
            "    gl_Position = matWorldViewProj * p + vec4(inTangent.xyz, bias) * 0.0;\n"
            "    texCoord0 = inPosition.xy;\n"
            "}\n");
        vfs.write_text(
            "wb/GLShadersOptimized/fs_face_shader_high.glsl",
            "uniform vec4 vMaterialColor;\n"
            "uniform sampler2D diffuse_texture;\n"
            "uniform sampler2D specular_texture;\n"
            "uniform sampler2D shadowmap_texture;\n"
            "varying vec2 texCoord0;\n"
            "void main() {\n"
            "    vec4 c = texture2D(diffuse_texture, texCoord0) * vMaterialColor\n"
            "           + texture2D(specular_texture, texCoord0)\n"
            "           + texture2DProj(shadowmap_texture, vec4(texCoord0, 0.5, 1.0)) * 0.0;\n"
            "    if (c.a < 0.5) discard;\n"
            "    gl_FragColor = c;\n"
            "}\n");

        ShaderTechniqueCatalog tech_cat{};
        std::string err;
        const GlTechniqueEntry* wse2_face = nullptr;
        expect(load_warband_shader_technique_catalog(vfs, "wb", tech_cat, &err) &&
                   tech_cat.techniques_by_lower.size() == 10U &&
                   (wse2_face = tech_cat.find("face_shader_skin_high")) != nullptr &&
                   wse2_face->synthesized_wse2_variant &&
                   wse2_face->vs_glsl.find("USE_SKINNING") != std::string::npos,
               "GLShaders catalog parses XML/GLSL and synthesizes the 5 WSE2 skinning variants");

        const auto compiler = create_glslang_shader_compiler();
        const auto maybe_dump_spv = [](std::string_view tag_name,
                                       std::span<const std::uint32_t> words) {
            const char* dump_dir = std::getenv("WSE_DUMP_SPIRV_DIR");
            if (dump_dir == nullptr || dump_dir[0] == '\0' || words.empty()) {
                return;
            }
            std::error_code ec;
            std::filesystem::create_directories(dump_dir, ec);
            const std::filesystem::path p =
                std::filesystem::path(dump_dir) / (std::string(tag_name) + ".spv");
            std::FILE* fp = std::fopen(p.string().c_str(), "wb");
            if (fp != nullptr) {
                std::fwrite(words.data(), sizeof(std::uint32_t), words.size(), fp);
                std::fclose(fp);
            }
        };

        if (wse2_face != nullptr) {
            SpirvShaderCache cache1(compiler);
            std::vector<std::uint32_t> vs_words1;
            std::vector<std::uint32_t> fs_words1;
            bool hit_vs1 = true;
            bool hit_fs1 = true;
            std::uint32_t vs_flags = 0;
            std::uint32_t fs_flags = 0;
            ShaderStageReflection vs_refl{};
            ShaderStageReflection fs_refl{};
            expect(cache1.get_or_compile(vfs, "wb/cache/spirv", wse2_face->name,
                                         SpirvStage::Vertex, VertexDeclType::BumpedSkinned,
                                         wse2_face->vs_glsl, vs_words1, &hit_vs1, &vs_flags,
                                         &vs_refl, wse2_face->fs_glsl) &&
                       cache1.get_or_compile(vfs, "wb/cache/spirv", wse2_face->name,
                                             SpirvStage::Fragment, VertexDeclType::BumpedSkinned,
                                             wse2_face->fs_glsl, fs_words1, &hit_fs1, &fs_flags,
                                             &fs_refl, wse2_face->vs_glsl),
                   "first SpirvShaderCache compile produces real SPIR-V 1.0");
            const auto vs_info = inspect_spirv_module(vs_words1);
            const auto fs_info = inspect_spirv_module(fs_words1);
            bool vs_has_ubo0 = !vs_refl.uniform_blocks.empty() &&
                               vs_refl.uniform_blocks[0].binding == 0U &&
                               vs_refl.uniform_blocks[0].size_bytes > 0U;
            bool fs_has_ubo1 = !fs_refl.uniform_blocks.empty() &&
                               fs_refl.uniform_blocks[0].binding == 1U;
            bool fs_has_diff_and_spec = false;
            for (const auto& s : fs_refl.samplers) {
                if (s.name == "specular_texture" && s.binding == 13U) {
                    fs_has_diff_and_spec = true;
                }
            }
            expect(!hit_vs1 && !hit_fs1 && cache1.compile_count == 2U &&
                       cache1.cache_hits == 0U && vs_info.has_value() && fs_info.has_value() &&
                       vs_info->generator_version == kWseSpirvGeneratorVersion &&
                       vs_words1[2] != shader_detail::kLegacyPlaceholderSpirvMagic &&
                       vs_has_ubo0 && fs_has_ubo1 && fs_has_diff_and_spec &&
                       (vs_flags & (kSpirvFeatNormalMap | kSpirvFeatSkinning)) ==
                           (kSpirvFeatNormalMap | kSpirvFeatSkinning) &&
                       (fs_flags & (kSpirvFeatSpecular | kSpirvFeatAlphaTest |
                                    kSpirvFeatShadowMap)) ==
                           (kSpirvFeatSpecular | kSpirvFeatAlphaTest | kSpirvFeatShadowMap),
                   "compiled SPIR-V 1.0 + reflection + OpString features match GLSL cleanly");
            maybe_dump_spv("wse2_face_skin_high_vs", vs_words1);
            maybe_dump_spv("wse2_face_skin_high_fs", fs_words1);

            // Second session over the same persistent VFS hits disk/VFS cache with zero
            // compiles, even with a null compiler backend.
            SpirvShaderCache cache2{};
            std::vector<std::uint32_t> vs_words2;
            std::vector<std::uint32_t> fs_words2;
            bool hit_vs2 = false;
            bool hit_fs2 = false;
            ShaderStageReflection vs_refl2{};
            ShaderStageReflection fs_refl2{};
            expect(cache2.get_or_compile(vfs, "wb/cache/spirv", wse2_face->name,
                                         SpirvStage::Vertex, VertexDeclType::BumpedSkinned,
                                         wse2_face->vs_glsl, vs_words2, &hit_vs2, nullptr,
                                         &vs_refl2, wse2_face->fs_glsl) &&
                       cache2.get_or_compile(vfs, "wb/cache/spirv", wse2_face->name,
                                             SpirvStage::Fragment, VertexDeclType::BumpedSkinned,
                                             wse2_face->fs_glsl, fs_words2, &hit_fs2, nullptr,
                                             &fs_refl2, wse2_face->vs_glsl) &&
                       hit_vs2 && hit_fs2 && cache2.compile_count == 0U &&
                       cache2.cache_hits == 2U && vs_words1 == vs_words2 &&
                       fs_words1 == fs_words2,
                   "second launch SpirvShaderCache hits persisted binaries with compile_count==0");
            // Cache hits return the reflection the compile produced (`.refl` sidecar).
            const auto same_reflection = [](const ShaderStageReflection& a,
                                             const ShaderStageReflection& b) {
                return serialize_shader_reflection(a, 0U) == serialize_shader_reflection(b, 0U);
            };
            expect(!vs_refl.uniform_blocks.empty() && same_reflection(vs_refl, vs_refl2) &&
                       !fs_refl.samplers.empty() && same_reflection(fs_refl, fs_refl2),
                   "cache hits return the same reflection as the compile");

            // A hit that needs reflection but has no sidecar recompiles (or fails cleanly
            // without a backend) instead of returning empty reflection.
            const std::string vs_fname = SpirvShaderCache::compute_cache_filename(
                SpirvStage::Vertex, VertexDeclType::BumpedSkinned, wse2_face->vs_glsl,
                wse2_face->fs_glsl);
            expect(vfs.remove_file("wb/cache/spirv/" +
                                   vs_fname.substr(0, vs_fname.size() - 4U) + ".refl"),
                   "remove VS reflection sidecar");
            SpirvShaderCache cache3{};
            ShaderStageReflection vs_refl3{};
            std::vector<std::uint32_t> vs_words3;
            const bool no_refl_hit = cache3.get_or_compile(
                vfs, "wb/cache/spirv", wse2_face->name, SpirvStage::Vertex,
                VertexDeclType::BumpedSkinned, wse2_face->vs_glsl, vs_words3, nullptr, nullptr,
                &vs_refl3, wse2_face->fs_glsl);
            SpirvShaderCache cache4(compiler);
            expect(!no_refl_hit &&
                       cache4.get_or_compile(vfs, "wb/cache/spirv", wse2_face->name,
                                             SpirvStage::Vertex, VertexDeclType::BumpedSkinned,
                                             wse2_face->vs_glsl, vs_words3, nullptr, nullptr,
                                             &vs_refl3, wse2_face->fs_glsl) &&
                       cache4.compile_count == 1U && same_reflection(vs_refl, vs_refl3),
                   "missing .refl sidecar is a miss when reflection is requested");
        }
    }

    // 6. Retail Warband corpus gate (Native + Phantasy Calradia) when installed locally.
    {
        std::error_code ec;
        // WSE_WARBAND_ROOT selects a local install; the GOG default keeps the old behaviour.
        const char* root_env = std::getenv("WSE_WARBAND_ROOT");
        const std::filesystem::path retail_root(
            (root_env != nullptr && root_env[0] != '\0') ? root_env
                                                          : "C:/GOG Games/Mount and Blade - Warband");
        if (std::filesystem::exists(retail_root / "Modules/Native/module.ini", ec)) {
            wse::platform::HostVfs read_vfs(retail_root);
            ShaderTechniqueCatalog tech_cat{};
            const GlTechniqueEntry* face_skin = nullptr;
            expect(load_warband_shader_technique_catalog(read_vfs, "", tech_cat) &&
                       tech_cat.techniques_by_lower.size() >= 271U &&
                       tech_cat.warnings.empty() &&
                       (face_skin = tech_cat.find("face_shader_skin_high")) != nullptr &&
                       face_skin->synthesized_wse2_variant &&
                       face_skin->vs_glsl.find("#define USE_SKINNING 1") != std::string::npos,
                   "retail GLShaders + GLShadersOptimized + all 5 WSE2 skinning variants load cleanly");

            std::unordered_set<std::string> tech_names;
            for (const auto& kv : tech_cat.techniques_by_lower) {
                tech_names.insert(kv.first);
            }

            const auto check_module = [&](const char* mod_rel, std::size_t expected_mats, bool check_native_mats) {
                std::string ini_text;
                if (!read_vfs.read_text(std::string(mod_rel) + "/module.ini", ini_text)) {
                    return false;
                }
                wse::game::ModuleIniData ini{};
                wse::game::parse_module_ini(ini_text, ini);
                BrfCatalog cat{};
                if (!load_module_brf_catalog(read_vfs, "", mod_rel, ini, cat)) {
                    return false;
                }
                std::unordered_map<std::string, BrfShader> shaders;
                std::size_t mat_count = 0;
                std::size_t mat_resolved = 0;
                bool saw_rose = false;
                bool saw_shields = false;
                for (const auto& e : cat.entries) {
                    if (!e.loaded) continue;
                    for (const BrfShader& sh : e.file.shaders) {
                        shaders.try_emplace(wse::platform::ascii_lower(sh.name), sh);
                    }
                    mat_count += e.file.materials.size();
                }
                SuitableShaderResolver resolver(shaders, {3, true, true}, &tech_names);
                for (const auto& e : cat.entries) {
                    if (!e.loaded) continue;
                    for (const BrfMaterial& mat : e.file.materials) {
                        const ShaderResolutionResult& res = resolver.resolve(mat.shader_name);
                        if (res.shader != nullptr &&
                            tech_cat.find(res.shader->technique_name) != nullptr) {
                            ++mat_resolved;
                        }
                        if (check_native_mats && res.shader != nullptr) {
                            const std::string mname = wse::platform::ascii_lower(mat.name);
                            if (mname == "prt_shd_rose_a") {
                                const MaterialRenderState st = decode_material_render_state(mat, res.shader);
                                saw_rose = (st.blend_mode == MaterialBlendMode::AlphaBlend &&
                                            st.alpha_test_mode == AlphaTestMode::GreaterEqual136 &&
                                            st.alpha_ref_u8 == 136U);
                            } else if (mname == "shields") {
                                const MaterialRenderState st_no = decode_material_render_state(mat, nullptr);
                                const MaterialRenderState st_sh = decode_material_render_state(mat, res.shader);
                                saw_shields = (st_no.uses_specular && !st_sh.uses_specular);
                            }
                        }
                    }
                }
                if (check_native_mats && (!saw_rose || !saw_shields)) {
                    return false;
                }
                return shaders.size() == 144U && mat_count == expected_mats &&
                       mat_resolved == expected_mats;
            };

            expect(check_module("Modules/Native", 633U, true),
                   "retail Native resolves 144/144 shaders, 633/633 materials, and blend+alpha_test / 0x20 gates");
            if (std::filesystem::exists(retail_root / "Modules/Phantasy Calradia/module.ini", ec)) {
                expect(check_module("Modules/Phantasy Calradia", 1963U, false),
                       "retail Phantasy Calradia resolves 144/144 shaders and 1963/1963 materials");
            }

            // Compile EVERY retail GLSL technique (271+ techniques = 542+ VS+FS stages)
            // into real Vulkan SPIR-V 1.0 and verify persistent VFS cache reuse.
            {
                const auto compiler = create_glslang_shader_compiler();
                wse::platform::MemoryVfs corpus_cache_vfs;
                SpirvShaderCache corpus_cache(compiler);
                std::size_t compiled_techs = 0;
                std::size_t failed_techs = 0;
                const char* dump_dir = std::getenv("WSE_DUMP_SPIRV_DIR");
                for (const auto& [lower_name, tech] : tech_cat.techniques_by_lower) {
                    if (tech.vs_glsl.empty() || tech.fs_glsl.empty()) {
                        ++failed_techs;
                        continue;
                    }
                    const VertexDeclType decl = tech.synthesized_wse2_variant
                                                    ? VertexDeclType::BumpedSkinned
                                                    : VertexDeclType::Standard;
                    std::vector<std::uint32_t> vs_spv;
                    std::vector<std::uint32_t> fs_spv;
                    const bool vs_ok = corpus_cache.get_or_compile(
                        corpus_cache_vfs, "spv", tech.name, SpirvStage::Vertex, decl, tech.vs_glsl,
                        vs_spv, nullptr, nullptr, nullptr, tech.fs_glsl);
                    const bool fs_ok = corpus_cache.get_or_compile(
                        corpus_cache_vfs, "spv", tech.name, SpirvStage::Fragment, decl, tech.fs_glsl,
                        fs_spv, nullptr, nullptr, nullptr, tech.vs_glsl);
                    if (vs_ok && fs_ok && inspect_spirv_module(vs_spv).has_value() &&
                        inspect_spirv_module(fs_spv).has_value()) {
                        ++compiled_techs;
                        if (dump_dir != nullptr && dump_dir[0] != '\0') {
                            std::filesystem::create_directories(dump_dir, ec);
                            const auto write_bin = [&](std::string_view suffix,
                                                       std::span<const std::uint32_t> words) {
                                const std::filesystem::path p =
                                    std::filesystem::path(dump_dir) /
                                    (lower_name + std::string(suffix));
                                if (std::FILE* fp = std::fopen(p.string().c_str(), "wb")) {
                                    std::fwrite(words.data(), sizeof(std::uint32_t), words.size(),
                                                fp);
                                    std::fclose(fp);
                                }
                            };
                            write_bin("_vs.spv", vs_spv);
                            write_bin("_fs.spv", fs_spv);
                        }
                    } else {
                        ++failed_techs;
                        std::printf("FAIL  corpus SPIR-V compile %s: %s\n", tech.name.c_str(),
                                    corpus_cache.last_error.c_str());
                    }
                }
                // Techniques sharing both GLSL files share cache entries (content-keyed).
                expect(failed_techs == 0U && compiled_techs >= 271U &&
                           corpus_cache.compile_count + corpus_cache.cache_hits ==
                               compiled_techs * 2U,
                       "retail Warband shader corpus compiles 271+ techniques to valid SPIR-V 1.0");

                SpirvShaderCache warm_cache{};
                std::size_t warm_hits = 0;
                for (const auto& [lower_name, tech] : tech_cat.techniques_by_lower) {
                    (void)lower_name;
                    const VertexDeclType decl = tech.synthesized_wse2_variant
                                                    ? VertexDeclType::BumpedSkinned
                                                    : VertexDeclType::Standard;
                    std::vector<std::uint32_t> vs_spv;
                    std::vector<std::uint32_t> fs_spv;
                    bool hv = false;
                    bool hf = false;
                    if (warm_cache.get_or_compile(corpus_cache_vfs, "spv", tech.name,
                                                  SpirvStage::Vertex, decl, tech.vs_glsl, vs_spv,
                                                  &hv, nullptr, nullptr, tech.fs_glsl) &&
                        warm_cache.get_or_compile(corpus_cache_vfs, "spv", tech.name,
                                                  SpirvStage::Fragment, decl, tech.fs_glsl, fs_spv,
                                                  &hf, nullptr, nullptr, tech.vs_glsl) &&
                        hv && hf) {
                        ++warm_hits;
                    }
                }
                expect(warm_hits == compiled_techs && warm_cache.compile_count == 0U,
                       "warm VFS cache hits every retail shader with compile_count==0");
            }
        }
    }
}

// Regressions for the P5 lowering / compiler review: each case failed to compile, or
// compiled into mismatched stages, before the fix. Every module produced here is also
// written to `WSE_DUMP_SPIRV_DIR` (when set) for the CI `spirv-val` step.
void test_p5_lowering_review_regressions() {
    using namespace wse::asset;
    const auto compiler = create_glslang_shader_compiler();
    const char* dump_dir = std::getenv("WSE_DUMP_SPIRV_DIR");
    const auto dump = [&](std::string_view name, const std::vector<std::uint32_t>& words) {
        if (dump_dir == nullptr || dump_dir[0] == '\0' || words.empty()) {
            return;
        }
        std::error_code ec;
        std::filesystem::create_directories(dump_dir, ec);
        const std::filesystem::path p =
            std::filesystem::path(dump_dir) / ("review_" + std::string(name) + ".spv");
        if (std::FILE* fp = std::fopen(p.string().c_str(), "wb")) {
            std::fwrite(words.data(), sizeof(std::uint32_t), words.size(), fp);
            std::fclose(fp);
        }
    };
    struct Pair {
        bool ok = false;
        SpirvCompileOutput vs;
        SpirvCompileOutput fs;
    };
    const auto compile_pair = [&](std::string_view name, std::string_view vs,
                                  std::string_view fs) {
        Pair p{};
        const bool vs_ok = compile_warband_glsl_to_spirv(*compiler, vs, SpirvStage::Vertex, name,
                                                         VertexDeclType::Standard, p.vs, fs);
        const bool fs_ok = compile_warband_glsl_to_spirv(*compiler, fs, SpirvStage::Fragment,
                                                         name, VertexDeclType::Standard, p.fs, vs);
        p.ok = vs_ok && fs_ok;
        if (!p.ok) {
            std::printf("      %.*s: vs=[%s] fs=[%s]\n", static_cast<int>(name.size()),
                        name.data(), p.vs.info_log.c_str(), p.fs.info_log.c_str());
        }
        dump(std::string(name) + "_vs", p.vs.spirv_words);
        dump(std::string(name) + "_fs", p.fs.spirv_words);
        return p;
    };
    const auto compile_fs = [&](std::string_view name, std::string_view fs) {
        SpirvCompileOutput out{};
        const bool ok = compile_warband_glsl_to_spirv(*compiler, fs, SpirvStage::Fragment, name,
                                                      VertexDeclType::Standard, out);
        if (!ok) {
            std::printf("      %.*s: %s\n", static_cast<int>(name.size()), name.data(),
                        out.info_log.c_str());
        }
        dump(name, out.spirv_words);
        return ok;
    };
    // `layout(location = N) <...> <name>;` in lowered GLSL, or -1.
    const auto location_of = [](std::string_view glsl, std::string_view name) -> int {
        std::size_t pos = 0;
        while (pos < glsl.size()) {
            const std::size_t eol = glsl.find('\n', pos);
            const std::string_view line =
                glsl.substr(pos, (eol == std::string_view::npos ? glsl.size() : eol) - pos);
            pos = (eol == std::string_view::npos) ? glsl.size() : eol + 1U;
            const std::string suffix = " " + std::string(name) + ";";
            const std::string_view head = "layout(location = ";
            if (line.starts_with(head) && line.ends_with(suffix)) {
                return std::atoi(std::string(line.substr(head.size())).c_str());
            }
        }
        return -1;
    };
    const auto binding_of = [](const ShaderStageReflection& r, std::string_view name) -> int {
        for (const auto& s : r.samplers) {
            if (s.name == name) {
                return static_cast<int>(s.binding);
            }
        }
        return -1;
    };
    const std::string kVsBase = "attribute vec3 inPosition;\n";
    const std::string kVsMain = "gl_Position = vec4(inPosition, 1.0);";

    // 1-2. Names outside the fixed tables (different order, subset, multi-location types)
    // land on the same location in both stages; samplers never share a binding.
    {
        const std::string vs = kVsBase +
                               "varying vec2 myA; varying mat3 tbn; varying vec3 myB;\n"
                               "varying vec4 extra[2];\n"
                               "uniform sampler2D vtex;\n"
                               "void main() { myA = vec2(0.0); tbn = mat3(1.0);\n"
                               "  myB = texture2DLod(vtex, vec2(0.0), 0.0).xyz;\n"
                               "  extra[0] = vec4(0.0); extra[1] = vec4(1.0); " +
                               kVsMain + " }\n";
        const std::string fs = "varying vec4 extra[2]; varying vec3 myB; varying mat3 tbn;\n"
                               "uniform sampler2D ftex;\n"
                               "void main() { gl_FragColor = texture2D(ftex, myB.xy)\n"
                               "  * vec4(tbn[2], 1.0) * extra[1]; }\n";
        const Pair p = compile_pair("cross_stage_unknown", vs, fs);
        bool same = p.ok;
        for (const char* n : {"myB", "tbn", "extra[2]"}) {
            const int a = location_of(p.vs.lowered_glsl, n);
            const int b = location_of(p.fs.lowered_glsl, n);
            same = same && a >= 0 && a == b;
        }
        const int tbn = location_of(p.vs.lowered_glsl, "tbn");
        const int myb = location_of(p.vs.lowered_glsl, "myB");
        const int extra = location_of(p.vs.lowered_glsl, "extra[2]");
        const bool no_overlap = !(myb >= tbn && myb < tbn + 3) &&
                                !(extra >= tbn && extra < tbn + 3) && !(myb >= extra && myb < extra + 2) &&
                                !(tbn >= extra && tbn < extra + 2);
        const int vtex = binding_of(p.vs.reflection, "vtex");
        const int ftex = binding_of(p.fs.reflection, "ftex");
        expect(same && no_overlap && vtex >= 14 && ftex >= 14 && vtex != ftex,
               "unknown varyings/samplers get one interface across both stages, sized by type");
    }

    // 3. Two fixed-table names sharing a slot (Color / VertexColor are both 3).
    {
        const std::string vs = kVsBase + "varying vec4 Color; varying vec4 VertexColor;\n"
                                         "void main() { Color = vec4(1.0); VertexColor = vec4(0.5); " +
                               kVsMain + " }\n";
        const std::string fs = "varying vec4 Color; varying vec4 VertexColor;\n"
                               "void main() { gl_FragColor = Color * VertexColor; }\n";
        const Pair p = compile_pair("fixed_slot_collision", vs, fs);
        const int c = location_of(p.vs.lowered_glsl, "Color");
        const int vc = location_of(p.vs.lowered_glsl, "VertexColor");
        expect(p.ok && c == 3 && vc >= 0 && vc != c &&
                   vc == location_of(p.fs.lowered_glsl, "VertexColor"),
               "colliding fixed varying names fall back to a shared dynamic slot");
    }

    // 5-6. Preprocessor: `#if`/`#else` alternatives, macros naming uniforms and legacy
    // built-ins, line continuations, spaced `# version`, and block comments.
    {
        expect(compile_fs("pp_alternatives",
                          "#define HQ 1\n#if HQ\nfloat f() { return 1.0; }\n#else\n"
                          "float f() { return 0.0; }\n#endif\n"
                          "void main() { gl_FragColor = vec4(f()); }\n"),
               "#if/#else alternative definitions are not both emitted");
        expect(compile_fs("pp_macro_uniform",
                          "uniform vec4 fog_color;\n#define FOG fog_color\n"
                          "uniform sampler2D diffuse_texture;\n"
                          "#define SAMPLE(uv) texture2D(diffuse_texture, uv)\n"
                          "#define OUTC gl_FragColor\n"
                          "void main() { OUTC = FOG * SAMPLE(vec2(0.0)); }\n"),
               "macros naming uniforms, texture2D, and gl_FragColor are lowered");
        expect(compile_fs("pp_line_continuation",
                          "#version 120\n#define M(a) \\\n  (a * 2.0)\nuniform float k;\n"
                          "void main() { gl_FragColor = vec4(M(k)); }\n"),
               "backslash line continuations in a 1.20 shader are spliced");
        expect(compile_fs("pp_spaced_version", "# version 120\nvoid main() { gl_FragColor = vec4(1.0); }\n"),
               "`# version` with a space is replaced, not duplicated");
        expect(compile_fs("block_comment_token", "uniform/**/float k;\n"
                                                 "void main() { gl_FragColor = vec4(k); }\n"),
               "a block comment separates tokens");

        // Same on the raw path (no preprocessing backend).
        const LoweredVulkanGlsl raw = lower_warband_glsl_to_vulkan450(
            "uniform vec4 fog_color;\n#define FOG fog_color\n#if 1\nfloat f() { return 1.0; }\n"
            "#else\nfloat f() { return 0.0; }\n#endif\n"
            "void main() { gl_FragColor = FOG * f(); }\n",
            SpirvStage::Fragment);
        std::vector<std::uint32_t> raw_spv;
        std::string raw_log;
        expect(raw.ok && raw.glsl.find("#define FOG _wse_fs_u._wse_u_fog_color") != std::string::npos &&
                   compiler->compile_vulkan_glsl_to_spirv(raw.glsl, SpirvStage::Fragment, "main",
                                                          raw_spv, nullptr, &raw_log),
               "raw lowering keeps #if blocks in place and renames macro bodies");
    }

    // 7-11. Declarations and scoping.
    {
        SpirvCompileOutput out{};
        const bool init_ok = compile_warband_glsl_to_spirv(
            *compiler, "uniform float k = 0.25, j;\nuniform vec2 v = vec2(1.0, 2.0);\n"
                       "void main() { gl_FragColor = vec4(k, j, v); }\n",
            SpirvStage::Fragment, "init", VertexDeclType::Standard, out);
        dump("uniform_initializer", out.spirv_words);
        const auto& defs = out.reflection.uniform_defaults;
        expect(init_ok && defs.size() == 2U && defs[0].name == "k" &&
                   defs[0].initializer_glsl == "0.25" && defs[1].name == "v" &&
                   defs[1].initializer_glsl == "vec2(1.0, 2.0)",
               "uniform initializers become reported defaults, not garbage block members");

        expect(compile_fs("bool_array", "uniform bool flags[2]; uniform bvec2 bv;\n"
                                        "void main() { gl_FragColor = vec4(flags[1] && bv.y ? 1.0 : 0.0); }\n"),
               "bool / bvec / bool-array uniforms compile inside the uniform block");
        expect(compile_fs("shadowing",
                          "uniform float k;\nstruct S { float k; };\n"
                          "float g(float k) { return k * 2.0; }\n"
                          "float h(float k);\n"
                          "void main() {\n  S s; s.k = k;\n  float a = k, k2 = 1.0;\n"
                          "  { float k = 2.0; a += k; }\n"
                          "  for (int i = 0; i < 2; ++i) { float k = 3.0; a += k; }\n"
                          "  gl_FragColor = vec4(s.k + g(k) + a + k + k2);\n}\n"
                          "float h(float k) { return k; }\n"),
               "locals, parameters, and struct members shadowing a uniform are not renamed");
        expect(compile_fs("struct_uniform", "struct L { vec3 d; float i; };\nuniform L light;\n"
                                            "void main() { gl_FragColor = vec4(light.d * light.i, 1.0); }\n"),
               "struct-typed uniforms are declared after their struct");
        expect(compile_fs("float_exponent", "uniform float e5;\n"
                                            "void main() { gl_FragColor = vec4(1.0e5 * e5, 1e-3, .5e+2, 0.0); }\n"),
               "float literals with exponents are not split into identifiers");
    }

    // 12. Legacy built-ins and qualifiers.
    {
        expect(compile_fs("shadow2d",
                          "uniform sampler2DShadow shadowmap_texture;\n"
                          "void main() {\n"
                          "  vec4 a = shadow2D(shadowmap_texture, vec3(0.5));\n"
                          "  vec4 b = shadow2DProj(shadowmap_texture, vec4(0.5), 0.0);\n"
                          "  gl_FragColor = a * b.r;\n}\n"),
               "shadow2D / shadow2DProj keep their GLSL 1.20 vec4 result");
        expect(compile_fs("texture_cube", "uniform samplerCube env_texture;\n"
                                          "void main() { gl_FragColor = textureCube(env_texture, vec3(1.0)); }\n"),
               "textureCube is lowered");
        SpirvCompileOutput fd{};
        const bool fd_ok = compile_warband_glsl_to_spirv(
            *compiler, "void main() { gl_FragData[0] = vec4(1.0); gl_FragData[1] = vec4(0.5); }\n",
            SpirvStage::Fragment, "frag_data", VertexDeclType::Standard, fd);
        dump("frag_data", fd.spirv_words);
        expect(fd_ok && fd.lowered_glsl.find("out vec4 _wse_frag_data[2];") != std::string::npos,
               "gl_FragData[n] becomes a location-0 output array sized by its highest index");
        SpirvCompileOutput both{};
        expect(!compile_warband_glsl_to_spirv(
                   *compiler, "void main() { gl_FragColor = vec4(1.0); gl_FragData[1] = vec4(0.5); }\n",
                   SpirvStage::Fragment, "both", VertexDeclType::Standard, both) &&
                   both.info_log.find("gl_FragData") != std::string::npos,
               "writing gl_FragColor and gl_FragData is reported");
        const Pair q = compile_pair(
            "interp_qualifiers",
            kVsBase + "invariant varying vec4 Color; centroid varying vec2 Tex0;\n"
                      "void main() { Color = vec4(1.0); Tex0 = vec2(0.0); " + kVsMain + " }\n",
            "invariant varying vec4 Color; centroid varying vec2 Tex0;\n"
            "void main() { gl_FragColor = Color * Tex0.x; }\n");
        expect(q.ok && q.vs.lowered_glsl.find(") invariant out vec4 Color;") != std::string::npos &&
                   q.vs.lowered_glsl.find(") centroid out vec2 Tex0;") != std::string::npos &&
                   q.fs.lowered_glsl.find(") in vec4 Color;") != std::string::npos &&
                   q.fs.lowered_glsl.find(") centroid in vec2 Tex0;") != std::string::npos,
               "invariant (outputs only) / centroid varyings keep their qualifiers");
    }

    // 14. Reflection: Warband names, byte sizes, array and matrix strides.
    {
        SpirvCompileOutput out{};
        const bool ok = compile_warband_glsl_to_spirv(
            *compiler, "uniform vec4 vMaterialColor; uniform mat4 matWorld; uniform float fa[3];\n"
                       "void main() { gl_FragColor = vMaterialColor * matWorld[0] * fa[2]; }\n",
            SpirvStage::Fragment, "refl", VertexDeclType::Standard, out);
        const ShaderUniformMemberInfo* color = nullptr;
        const ShaderUniformMemberInfo* mat = nullptr;
        const ShaderUniformMemberInfo* arr = nullptr;
        if (ok && !out.reflection.uniform_blocks.empty()) {
            for (const auto& m : out.reflection.uniform_blocks[0].members) {
                if (m.name == "vMaterialColor") color = &m;
                if (m.name == "matWorld") mat = &m;
                if (m.name == "fa") arr = &m;
            }
        }
        expect(color != nullptr && color->size_bytes == 16U && color->array_size == 0U &&
                   mat != nullptr && mat->size_bytes == 64U && mat->matrix_stride_bytes == 16U &&
                   arr != nullptr && arr->array_size == 3U && arr->array_stride_bytes == 16U &&
                   arr->size_bytes == 48U,
               "reflection reports Warband uniform names and std140 byte sizes / strides");
    }

    // Backend identity in the tag: a different compiler version recompiles cached modules.
    {
        struct RenamedBackend final : SpirvCompilerBackend {
            std::shared_ptr<SpirvCompilerBackend> inner;
            [[nodiscard]] const char* backend_name() const noexcept override { return "glslang"; }
            [[nodiscard]] std::string cache_identity() const override { return "glslang-99.0.0"; }
            [[nodiscard]] bool supports_preprocessing() const noexcept override { return true; }
            [[nodiscard]] bool preprocess_legacy_glsl(std::string_view s, SpirvStage st,
                                                      std::string& o, std::string* log) override {
                return inner->preprocess_legacy_glsl(s, st, o, log);
            }
            [[nodiscard]] bool compile_vulkan_glsl_to_spirv(std::string_view s, SpirvStage st,
                                                            std::string_view e,
                                                            std::vector<std::uint32_t>& w,
                                                            ShaderStageReflection* r,
                                                            std::string* log) override {
                return inner->compile_vulkan_glsl_to_spirv(s, st, e, w, r, log);
            }
        };
        auto renamed = std::make_shared<RenamedBackend>();
        renamed->inner = compiler;
        wse::platform::MemoryVfs v;
        const std::string src = "void main() { gl_FragColor = vec4(1.0); }";
        std::vector<std::uint32_t> w;
        bool hit = true;
        SpirvShaderCache first(compiler);
        SpirvShaderCache upgraded(renamed);
        SpirvShaderCache headless{};
        const bool a = first.get_or_compile(v, "c", "t", SpirvStage::Fragment,
                                            VertexDeclType::Standard, src, w);
        const auto info = inspect_spirv_module(w);
        const bool b = upgraded.get_or_compile(v, "c", "t", SpirvStage::Fragment,
                                               VertexDeclType::Standard, src, w, &hit);
        const bool c = headless.get_or_compile(v, "c", "t", SpirvStage::Fragment,
                                               VertexDeclType::Standard, src, w);
        expect(a && info && info->backend_identity == compiler->cache_identity() && b && !hit &&
                   upgraded.compile_count == 1U && c && headless.cache_hits == 1U,
               "a different backend identity recompiles; no backend accepts any identity");
    }

    // Corrupt reflection sidecars are rejected.
    {
        ShaderStageReflection r{};
        r.samplers.push_back({"diffuse_texture", 0U, 3U});
        r.uniform_defaults.push_back({"k", "0.25"});
        const std::vector<std::uint8_t> bytes = serialize_shader_reflection(r, 42U);
        std::vector<std::uint8_t> truncated(bytes.begin(), bytes.end() - 1);
        std::vector<std::uint8_t> huge_count = bytes;
        huge_count[16] = 0xFF;  // uniform block count
        const auto back = deserialize_shader_reflection(bytes, 42U);
        expect(back && back->samplers.size() == 1U && back->samplers[0].binding == 3U &&
                   back->uniform_defaults.size() == 1U &&
                   !deserialize_shader_reflection(bytes, 43U) &&
                   !deserialize_shader_reflection(truncated, 42U) &&
                   !deserialize_shader_reflection(huge_count, 42U),
               "reflection sidecar round-trips and rejects wrong key / truncation / bad counts");
    }

    // Compiles run concurrently on one backend.
    {
        std::array<bool, 4> ok{};
        std::vector<std::thread> threads;
        for (std::size_t t = 0; t < ok.size(); ++t) {
            threads.emplace_back([&, t] {
                bool all = true;
                for (int n = 0; n < 8; ++n) {
                    SpirvCompileOutput o{};
                    all = all && compile_warband_glsl_to_spirv(
                                     *compiler,
                                     "uniform vec4 c; uniform sampler2D diffuse_texture;\n"
                                     "varying vec2 Tex0;\n"
                                     "void main() { gl_FragColor = c * texture2D(diffuse_texture, Tex0); }\n",
                                     SpirvStage::Fragment, "mt", VertexDeclType::Standard, o);
                }
                ok[t] = all;
            });
        }
        for (std::thread& th : threads) {
            th.join();
        }
        expect(ok[0] && ok[1] && ok[2] && ok[3], "concurrent compiles on one backend succeed");
    }

    // Review of 9411644: type names are exact, so a parameter shadowing a `mat*` uniform
    // keeps its own name (it used to be rewritten into a block member: syntax error).
    {
        const char* fs0 = "#version 120\nvarying vec2 Tex;\nuniform sampler2D diffuse_texture;\n"
                          "void main(){ gl_FragColor = texture2D(diffuse_texture, Tex); }\n";
        const Pair p = compile_pair(
            "mat_param_shadow",
            "#version 120\nuniform mat4 matWorld;\nattribute vec3 inPosition;\nvarying vec2 Tex;\n"
            "vec4 xf(mat4 matWorld, vec3 p) { return matWorld * vec4(p, 1.0); }\n"
            "void main(){ Tex = vec2(0.0); gl_Position = xf(matWorld, inPosition); }\n",
            fs0);
        expect(p.ok && p.vs.lowered_glsl.find("vec4 xf(mat4 matWorld, vec3 p) { return matWorld *") !=
                           std::string::npos &&
                   p.vs.lowered_glsl.find("xf(_wse_vs_u._wse_u_matWorld, inPosition)") !=
                       std::string::npos,
               "parameter named like a mat* uniform shadows it instead of being renamed");
        expect(shader_detail::is_glsl_matrix_type_name("mat3x4") &&
                   shader_detail::is_glsl_matrix_type_name("dmat2") &&
                   !shader_detail::is_glsl_matrix_type_name("matWorld") &&
                   !shader_detail::is_glsl_matrix_type_name("mat5") &&
                   shader_detail::is_glsl_opaque_type_name("usampler2D") &&
                   shader_detail::is_glsl_opaque_type_name("sampler2DArrayShadow") &&
                   shader_detail::is_glsl_opaque_type_name("iimage2D") &&
                   !shader_detail::is_glsl_opaque_type_name("samplerSettings") &&
                   !shader_detail::is_glsl_opaque_type_name("int"),
               "matrix / opaque type-name checks are exact");
    }
    // `for (int k ...)` scopes `k` to the loop: the uniform `k` is renamed after it.
    {
        expect(compile_fs("for_scope",
                          "#version 120\nvarying vec2 Tex;\nuniform float k;\n"
                          "uniform sampler2D diffuse_texture;\nvoid main(){ float s = 0.0;\n"
                          "for (int k = 0; k < 2; ++k) { s += float(k); }\n"
                          "for (int k = 0; k < 2; ++k) s += float(k);\n"
                          "gl_FragColor = texture2D(diffuse_texture, Tex) * k * s; }\n"),
               "for-loop variable shadows a uniform only inside the loop");
    }
    // Integer samplers are opaque: declared with a binding, not as block members.
    {
        const Pair p = compile_pair(
            "usampler",
            "#version 130\nin vec3 inPosition;\nout vec2 Tex;\n"
            "void main(){ Tex = vec2(0.0); gl_Position = vec4(inPosition, 1.0); }\n",
            "#version 130\nin vec2 Tex;\nuniform usampler2D ids;\n"
            "void main(){ gl_FragColor = vec4(texture(ids, Tex)); }\n");
        expect(p.ok && p.fs.lowered_glsl.find(") uniform usampler2D ids;") != std::string::npos,
               "usampler2D uniform gets a sampler binding");
    }
    // GLSL 1.30 file-scope `in` / `out` lower like attribute / varying / color outputs.
    {
        const Pair p = compile_pair(
            "glsl130_io",
            "#version 130\nin vec3 inPosition;\nin vec2 inTexCoord;\nout vec2 Tex;\n"
            "flat out int layer;\n"
            "void main(){ Tex = inTexCoord; layer = 1; gl_Position = vec4(inPosition, 1.0); }\n",
            "#version 130\nin vec2 Tex;\nflat in int layer;\nuniform sampler2D diffuse_texture;\n"
            "out vec4 color;\nout vec4 glow;\n"
            "void main(){ color = texture(diffuse_texture, Tex); glow = vec4(float(layer)); }\n");
        expect(p.ok && location_of(p.vs.lowered_glsl, "inTexCoord") == 2 &&
                   location_of(p.vs.lowered_glsl, "Tex") == location_of(p.fs.lowered_glsl, "Tex") &&
                   location_of(p.vs.lowered_glsl, "layer") ==
                       location_of(p.fs.lowered_glsl, "layer") &&
                   location_of(p.fs.lowered_glsl, "color") == 0 &&
                   location_of(p.fs.lowered_glsl, "glow") == 1,
               "GLSL 1.30 in/out declarations get attribute, varying, and output locations");
        LoweredVulkanGlsl mixed = lower_warband_glsl_to_vulkan450(
            "#version 130\nout vec4 color;\nvoid main(){ color = vec4(1.0); gl_FragColor = color; }\n",
            SpirvStage::Fragment);
        expect(!mixed.ok && mixed.error.find("mixes") != std::string::npos,
               "fragment `out` variables together with gl_FragColor are rejected clearly");
    }
    // Parse warnings reach the info log of a successful compile.
    {
        SpirvCompileOutput out{};
        const bool ok = compile_warband_glsl_to_spirv(
            *compiler,
            "#version 120\n#extension GL_WSE_not_a_real_extension : enable\n"
            "void main(){ gl_FragColor = vec4(1.0); }\n",
            SpirvStage::Fragment, "warn", VertexDeclType::Standard, out);
        expect(ok && out.info_log.find("GL_WSE_not_a_real_extension") != std::string::npos,
               "successful compile keeps glslang parse warnings in info_log");
    }
    // Compatibility-profile built-ins lower onto synthesized declarations that both
    // stages agree on.
    {
        const char* vs =
            "#version 120\nvoid main(){ gl_TexCoord[1] = gl_MultiTexCoord0;\n"
            "gl_FrontColor = gl_Color; gl_FogFragCoord = (gl_ModelViewMatrix * gl_Vertex).z;\n"
            "gl_Position = ftransform(); }\n";
        const char* fs =
            "#version 120\nuniform sampler2D diffuse_texture;\nvoid main(){\n"
            "gl_FragColor = gl_Color * texture2D(diffuse_texture, gl_TexCoord[0].xy) *\n"
            "gl_FogFragCoord; }\n";
        const Pair p = compile_pair("compat_builtins", vs, fs);
        const auto member_named = [](const ShaderStageReflection& r, std::string_view n) {
            for (const auto& b : r.uniform_blocks) {
                for (const auto& m : b.members) {
                    if (m.name == n) {
                        return true;
                    }
                }
            }
            return false;
        };
        expect(p.ok && location_of(p.vs.lowered_glsl, "_wse_gl_Vertex") == 0 &&
                   location_of(p.vs.lowered_glsl, "_wse_gl_Color") == 3 &&
                   location_of(p.vs.lowered_glsl, "_wse_gl_FrontColor") ==
                       location_of(p.fs.lowered_glsl, "_wse_gl_FrontColor") &&
                   location_of(p.vs.lowered_glsl, "_wse_gl_FogFragCoord") ==
                       location_of(p.fs.lowered_glsl, "_wse_gl_FogFragCoord") &&
                   p.vs.lowered_glsl.find("out vec4 _wse_gl_TexCoord[2];") != std::string::npos &&
                   p.fs.lowered_glsl.find("in vec4 _wse_gl_TexCoord[2];") != std::string::npos &&
                   member_named(p.vs.reflection, "gl_ModelViewProjectionMatrix") &&
                   member_named(p.vs.reflection, "gl_ModelViewMatrix"),
               "gl_Vertex / gl_Color / gl_TexCoord[] / ftransform() lower to a shared interface");
        const Pair p2 = compile_pair(
            "compat_texmatrix",
            "#version 120\nvoid main(){ gl_TexCoord[0] = gl_TextureMatrix[0] * gl_MultiTexCoord0;\n"
            "gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * gl_Vertex; }\n",
            "#version 120\nvoid main(){ gl_FragColor = gl_TexCoord[0]; }\n");
        expect(p2.ok && p2.vs.lowered_glsl.find("mat4 _wse_u_gl_TextureMatrix[1];") !=
                            std::string::npos,
               "gl_TextureMatrix[] becomes a sized uniform-block array");
    }

    expect(infer_vertex_decl_from_glsl("attribute vec4 inTangent; attribute vec4 inBlendIndices;") ==
                   VertexDeclType::BumpedSkinned &&
               infer_vertex_decl_from_glsl("attribute vec4 inBlendWeight;") ==
                   VertexDeclType::Skinned &&
               infer_vertex_decl_from_glsl("attribute vec3 inPosition;") ==
                   VertexDeclType::Standard,
           "infer_vertex_decl_from_glsl reads the attributes a vertex shader uses");
}


// R5a step 2: D3D9 shader bytecode to GLSL (wse/recomp/d3d9_shader.h), on hand-assembled shaders:
// the GLSL says what the bytecode does, and glslang compiles it (SPIR-V written to
// WSE_DUMP_SPIRV_DIR for CI's spirv-val). Locally, all 1,157 shaders of the user's install
// translate and compile (2026-11-22).
void test_d3d9_shader_translation() {
    namespace sh = wse::recomp::d3d9sh;
    using namespace wse::asset;
    // vs_2_0: dcl_position v0; dcl_texcoord v1; def c10 (1, 0.5, 0, 2); dp4 oPos.x..w, v0, c0..c3;
    // mad_sat oT0, v1, c10.yyyy, c10.xxxx
    const std::uint32_t one = 0x3F800000U, half = 0x3F000000U, two = 0x40000000U;
    const std::vector<std::uint32_t> vs = {
        0xFFFE0200U,
        0x0200001FU, 0x80000000U, 0x900F0000U,
        0x0200001FU, 0x80000005U, 0x900F0001U,
        0x05000051U, 0xA00F000AU, one, half, 0, two,
        0x03000009U, 0xC0010000U, 0x90E40000U, 0xA0E40000U,
        0x03000009U, 0xC0020000U, 0x90E40000U, 0xA0E40001U,
        0x03000009U, 0xC0040000U, 0x90E40000U, 0xA0E40002U,
        0x03000009U, 0xC0080000U, 0x90E40000U, 0xA0E40003U,
        0x04000004U, 0xE01F0000U, 0x90E40001U, 0xA055000AU, 0xA000000AU,
        0x0000FFFFU,
    };
    // ps_2_0: dcl t0.xy; dcl_2d s0; texld r0, t0, s0; texkill r0; mov oC0, r0
    const std::vector<std::uint32_t> ps = {
        0xFFFF0200U,
        0x0200001FU, 0x80000000U, 0xB0030000U,
        0x0200001FU, 0x90000000U, 0xA00F0800U,
        0x03000042U, 0x800F0000U, 0xB0E40000U, 0xA0E40800U,
        0x01000041U, 0x800F0000U,
        0x02000001U, 0x800F0800U, 0x80E40000U,
        0x0000FFFFU,
    };
    // call (not in scope): refused by name.
    const std::vector<std::uint32_t> unsupported = {0xFFFE0200U, 0x01000019U, 0xF0E40000U, 0x0000FFFFU};

    sh::Shader v, p, bad;
    const bool vs_ok = sh::translate(std::span<const std::uint32_t>(vs), v);
    const bool ps_ok = sh::translate(std::span<const std::uint32_t>(ps), p);
    auto has = [](const sh::Shader& s, const char* text) { return s.glsl.find(text) != std::string::npos; };
    expect(vs_ok && !v.pixel && v.inputs.size() == 2 && v.inputs[1].usage == sh::kTexCoord && v.inputs[1].reg == 1 &&
               has(v, "layout(location = 1) in vec4 v1;") && has(v, "vec4(1.0, 0.5, 0.0, 2.0)") &&
               has(v, "wse_vin(v0, 0u)") &&
               has(v, "clamp(") && has(v, "gl_Position = oPos;") && has(v, "gl_Position.y = -gl_Position.y;") &&
               has(v, "layout(location = 0) out vec4 out_oT0;"),
           "d3d9 shader: vs_2_0 inputs by register, BGRA swizzle helper, def literal, saturate, position flip, texcoord varying");
    expect(ps_ok && p.pixel && p.samplers[0] == 2 && has(p, "uniform sampler2D s0;") && has(p, "texture(s0, (t0).xy)") &&
               has(p, "layout(constant_id = 0) const int wse_alpha_func = 0;") &&
               has(p, "if (wse_alpha_func != 0)") && has(p, "u.posfix.y") && !has(p, "u.posfix.x") &&
               has(p, "discard;") && has(p, "layout(location = 0) in vec4 t0;") && has(p, "layout(location = 0) out vec4 out_oC0;"),
           "d3d9 shader: ps_2_0 texcoord input links to the vertex output, texld, texkill, spec-constant alpha-test epilogue, colour output");
    expect(!sh::translate(std::span<const std::uint32_t>(unsupported), bad) && bad.error.find("unsupported instruction 25") != std::string::npos,
           "d3d9 shader: an instruction outside the scope is refused by name");
    expect(sh::link_location(sh::kTexCoord, 3) == 3 && sh::link_location(sh::kColor, 1) == 11 && sh::link_location(sh::kFog, 0) == 12,
           "d3d9 shader: varyings link by usage and index");

    const auto compiler = create_glslang_shader_compiler();
    std::vector<std::uint32_t> vs_spv, ps_spv;
    std::string log;
    const bool vs_compiles = vs_ok && compiler->compile_vulkan_glsl_to_spirv(v.glsl, SpirvStage::Vertex, "main", vs_spv, nullptr, &log);
    const bool ps_compiles = ps_ok && compiler->compile_vulkan_glsl_to_spirv(p.glsl, SpirvStage::Fragment, "main", ps_spv, nullptr, &log);
    expect(vs_compiles && ps_compiles && !vs_spv.empty() && !ps_spv.empty(), "d3d9 shader: the translated GLSL compiles to SPIR-V");
    const char* dump_dir = std::getenv("WSE_DUMP_SPIRV_DIR");
    if (dump_dir != nullptr && dump_dir[0] != '\0') {
        std::error_code ec;
        std::filesystem::create_directories(dump_dir, ec);
        for (const auto& [name, words] : {std::pair<const char*, const std::vector<std::uint32_t>*>{"d3d9_vs20.spv", &vs_spv},
                                          std::pair<const char*, const std::vector<std::uint32_t>*>{"d3d9_ps20.spv", &ps_spv}}) {
            if (words->empty()) continue;
            if (std::FILE* fp = std::fopen((std::filesystem::path(dump_dir) / name).string().c_str(), "wb")) {
                std::fwrite(words->data(), sizeof(std::uint32_t), words->size(), fp);
                std::fclose(fp);
            }
        }
    }
}

void test_r5b_postfx_and_dynres() {
    using namespace wse::asset;
    using namespace wse::platform;

    // 1. Every post-pass GLSL 450 shader compiles to SPIR-V (Vulkan 1.0/1.1) and dumps to
    //    `WSE_DUMP_SPIRV_DIR` when set (for CI `spirv-val`).
    {
        const auto compiler = create_glslang_shader_compiler();
        const char* dump_dir = std::getenv("WSE_DUMP_SPIRV_DIR");
        struct ShaderCase {
            const char* name;
            const char* spv_file;
            SpirvStage stage;
            std::string_view src;
        };
        const std::array<ShaderCase, 5> cases = {{
            {"kPostVertex", "post_vs.spv", SpirvStage::Vertex, kPostVertex},
            {"kFsrEasu", "post_fsr_easu.spv", SpirvStage::Fragment, kFsrEasu},
            {"kFsrRcas", "post_fsr_rcas.spv", SpirvStage::Fragment, kFsrRcas},
            {"kCas", "post_cas.spv", SpirvStage::Fragment, kCas},
            {"kFxaa", "post_fxaa.spv", SpirvStage::Fragment, kFxaa},
        }};
        for (const auto& sc : cases) {
            std::vector<std::uint32_t> spv;
            ShaderStageReflection refl{};
            std::string log;
            const bool ok = compiler->compile_vulkan_glsl_to_spirv(
                sc.src, sc.stage, "main", spv, &refl, &log);
            if (!ok) {
                std::printf("postfx compile error (%s): %s\n", sc.name, log.c_str());
            }
            const bool valid_magic = !spv.empty() && spv[0] == 0x07230203U;
            expect(ok && valid_magic, sc.name);
            if (sc.stage == SpirvStage::Fragment) {
                bool has_src = false;
                for (const auto& samp : refl.samplers) {
                    if (samp.name == "src" && samp.binding == 0U) {
                        has_src = true;
                    }
                }
                expect(has_src, "postfx fragment shader reflects sampler2D src at binding 0");
            }
            if (ok && dump_dir != nullptr && dump_dir[0] != '\0' && !spv.empty()) {
                std::error_code ec;
                std::filesystem::create_directories(dump_dir, ec);
                if (std::FILE* fp = std::fopen(
                        (std::filesystem::path(dump_dir) / sc.spv_file).string().c_str(), "wb")) {
                    std::fwrite(spv.data(), sizeof(std::uint32_t), spv.size(), fp);
                    std::fclose(fp);
                }
            }
        }
    }

    // 2. Hand-computed FSR1 `easu_constants` and `rcas_constants` verification.
    {
        auto near_f = [](float a, float b) { return std::fabs(a - b) <= 1e-6F; };

        // 1280x720 -> 1920x1080 (derived from FsrEasuCon):
        //   scale = 1280/1920 = 2/3, 720/1080 = 2/3
        //   con0 = (2/3, 2/3, 0.5*(2/3)-0.5 = -1/6, -1/6)
        //   con1 = (1/1280, 1/720, 1/1280, -1/720)
        //   con2 = (-1/1280, 2/720, 1/1280, 2/720)
        //   con3 = (0, 4/720, 0, 0)
        const EasuConstants ups =
            easu_constants(1280.0F, 720.0F, 1280.0F, 720.0F, 1920.0F, 1080.0F);
        expect(near_f(ups.con[0], 0.6666667F) && near_f(ups.con[1], 0.6666667F) &&
                   near_f(ups.con[2], -0.16666667F) && near_f(ups.con[3], -0.16666667F) &&
                   near_f(ups.con[4], 0.00078125F) && near_f(ups.con[5], 0.0013888889F) &&
                   near_f(ups.con[6], 0.00078125F) && near_f(ups.con[7], -0.0013888889F) &&
                   near_f(ups.con[8], -0.00078125F) && near_f(ups.con[9], 0.0027777778F) &&
                   near_f(ups.con[10], 0.00078125F) && near_f(ups.con[11], 0.0027777778F) &&
                   ups.con[12] == 0.0F && near_f(ups.con[13], 0.0055555556F) &&
                   ups.con[14] == 0.0F && ups.con[15] == 0.0F,
               "easu_constants 1280x720 -> 1920x1080 match hand-computed FsrEasuCon");

        // 1:1 1920x1080 -> 1920x1080:
        //   con0 = (1, 1, 0, 0)
        //   con1 = (1/1920, 1/1080, 1/1920, -1/1080)
        //   con2 = (-1/1920, 2/1080, 1/1920, 2/1080)
        //   con3 = (0, 4/1080, 0, 0)
        const EasuConstants one_to_one =
            easu_constants(1920.0F, 1080.0F, 1920.0F, 1080.0F, 1920.0F, 1080.0F);
        expect(near_f(one_to_one.con[0], 1.0F) && near_f(one_to_one.con[1], 1.0F) &&
                   near_f(one_to_one.con[2], 0.0F) && near_f(one_to_one.con[3], 0.0F) &&
                   near_f(one_to_one.con[4], 0.0005208333F) &&
                   near_f(one_to_one.con[5], 0.0009259259F) &&
                   near_f(one_to_one.con[6], 0.0005208333F) &&
                   near_f(one_to_one.con[7], -0.0009259259F) &&
                   near_f(one_to_one.con[8], -0.0005208333F) &&
                   near_f(one_to_one.con[9], 0.0018518519F) &&
                   near_f(one_to_one.con[10], 0.0005208333F) &&
                   near_f(one_to_one.con[11], 0.0018518519F) &&
                   one_to_one.con[12] == 0.0F && near_f(one_to_one.con[13], 0.0037037037F) &&
                   one_to_one.con[14] == 0.0F && one_to_one.con[15] == 0.0F,
               "easu_constants 1:1 1920x1080 match hand-computed FsrEasuCon");

        // Dynamic-resolution sub-viewport inside full texture:
        //   in_size = 1920x1080, in_viewport = 1280x720, out_size = 1920x1080
        const EasuConstants subview =
            easu_constants(1920.0F, 1080.0F, 1280.0F, 720.0F, 1920.0F, 1080.0F);
        expect(near_f(subview.con[0], 0.6666667F) && near_f(subview.con[2], -0.16666667F) &&
                   near_f(subview.con[4], 0.0005208333F) && near_f(subview.con[5], 0.0009259259F),
               "easu_constants separates input viewport scale from full input texture size");

        // FsrRcasCon:
        //   0 stops -> exp2(0)=1.0, fp16(1.0)=0x3C00 -> con[1]=bit_cast<float>(0x3C003C00)
        //   1 stop  -> exp2(-1)=0.5, fp16(0.5)=0x3800 -> con[1]=bit_cast<float>(0x38003800)
        //   2 stops -> exp2(-2)=0.25, fp16(0.25)=0x3400 -> con[1]=bit_cast<float>(0x34003400)
        //   0.5 stops -> exp2(-0.5)=0.70710678, fp16=0x39A8 -> con[1]=bit_cast<float>(0x39A839A8)
        const RcasConstants r0 = rcas_constants(0.0F);
        const RcasConstants r1 = rcas_constants(1.0F);
        const RcasConstants r2 = rcas_constants(2.0F);
        const RcasConstants r05 = rcas_constants(0.5F);
        expect(r0.con[0] == 1.0F && std::bit_cast<std::uint32_t>(r0.con[1]) == 0x3C003C00U &&
                   r0.con[2] == 0.0F && r0.con[3] == 0.0F,
               "rcas_constants(0.0) packs 1.0 and fp16 pair 0x3C003C00");
        expect(r1.con[0] == 0.5F && std::bit_cast<std::uint32_t>(r1.con[1]) == 0x38003800U &&
                   r1.con[2] == 0.0F && r1.con[3] == 0.0F,
               "rcas_constants(1.0) packs 0.5 and fp16 pair 0x38003800");
        expect(r2.con[0] == 0.25F && std::bit_cast<std::uint32_t>(r2.con[1]) == 0x34003400U &&
                   r2.con[2] == 0.0F && r2.con[3] == 0.0F,
               "rcas_constants(2.0) packs 0.25 and fp16 pair 0x34003400");
        expect(near_f(r05.con[0], 0.70710678F) &&
                   std::bit_cast<std::uint32_t>(r05.con[1]) == 0x39A839A8U,
               "rcas_constants(0.5) packs exp2(-0.5) and fp16 pair 0x39A839A8");
    }

    // 3. CPU reference of kFxaa's Rec.601 luma, contrast threshold, and orientation on 3x3 patches.
    {
        struct Rgb {
            float r, g, b;
        };
        struct FxaaDecision {
            bool active;
            bool is_horiz;
            float range;
        };
        auto luma = [](Rgb c) { return 0.299F * c.r + 0.587F * c.g + 0.114F * c.b; };
        auto eval_patch = [&](const std::array<Rgb, 9>& p, float edge_thresh,
                              float edge_thresh_min) -> FxaaDecision {
            const float lNW = luma(p[0]), lN = luma(p[1]), lNE = luma(p[2]);
            const float lW = luma(p[3]), lM = luma(p[4]), lE = luma(p[5]);
            const float lSW = luma(p[6]), lS = luma(p[7]), lSE = luma(p[8]);
            const float lMin = std::min({lM, lN, lS, lW, lE});
            const float lMax = std::max({lM, lN, lS, lW, lE});
            const float lRange = lMax - lMin;
            if (lRange < std::max(edge_thresh_min, lMax * edge_thresh)) {
                return {false, false, lRange};
            }
            const float edgeH = std::fabs(lNW + lSW - 2.0F * lW) +
                                2.0F * std::fabs(lN + lS - 2.0F * lM) +
                                std::fabs(lNE + lSE - 2.0F * lE);
            const float edgeV = std::fabs(lNW + lNE - 2.0F * lN) +
                                2.0F * std::fabs(lW + lE - 2.0F * lM) +
                                std::fabs(lSW + lSE - 2.0F * lS);
            return {true, edgeH >= edgeV, lRange};
        };

        const float thresh = 0.125F;
        const float thresh_min = 0.05F;
        const Rgb flat_c{0.5F, 0.5F, 0.5F};
        const std::array<Rgb, 9> flat = {flat_c, flat_c, flat_c, flat_c, flat_c,
                                         flat_c, flat_c, flat_c, flat_c};
        const FxaaDecision d_flat = eval_patch(flat, thresh, thresh_min);
        expect(!d_flat.active && d_flat.range == 0.0F, "fxaa skips flat 3x3 patch");

        // Dark step: lMax=0.04, lRange=0.04 < edge_threshold_min (0.05) even though 0.04 >= 0.04*0.125.
        const Rgb d0{0.0F, 0.0F, 0.0F}, d1{0.04F, 0.04F, 0.04F};
        const std::array<Rgb, 9> dark = {d0, d0, d0, d1, d1, d1, d1, d1, d1};
        expect(!eval_patch(dark, thresh, thresh_min).active,
               "fxaa skips dark low-contrast patch below edge_threshold_min");

        // Bright step: lMax=0.98, lRange=0.08 > 0.05, but 0.08 < 0.98*0.125=0.1225 -> skipped.
        const Rgb b0{0.90F, 0.90F, 0.90F}, b1{0.98F, 0.98F, 0.98F};
        const std::array<Rgb, 9> bright = {b0, b0, b0, b1, b1, b1, b1, b1, b1};
        expect(!eval_patch(bright, thresh, thresh_min).active,
               "fxaa skips bright moderate-contrast patch below lMax * edge_threshold");

        // Sharp horizontal edge (row 0 dark, rows 1-2 white): active and horizontal.
        const Rgb blk{0.0F, 0.0F, 0.0F}, wht{1.0F, 1.0F, 1.0F};
        const std::array<Rgb, 9> horiz = {blk, blk, blk, wht, wht, wht, wht, wht, wht};
        const FxaaDecision d_horiz = eval_patch(horiz, thresh, thresh_min);
        expect(d_horiz.active && d_horiz.is_horiz,
               "fxaa activates on sharp horizontal edge and classifies isHoriz=true");

        // Sharp vertical edge (col 0 dark, cols 1-2 white): active and vertical.
        const std::array<Rgb, 9> vert = {blk, wht, wht, blk, wht, wht, blk, wht, wht};
        const FxaaDecision d_vert = eval_patch(vert, thresh, thresh_min);
        expect(d_vert.active && !d_vert.is_horiz,
               "fxaa activates on sharp vertical edge and classifies isHoriz=false");
    }

    // 4. DynamicResolution controller and render_size tests.
    {
        auto near_s = [](float a, float b) { return std::fabs(a - b) <= 1e-4F; };

        // Steady load under budget stays at max_scale (1.0).
        {
            DynamicResolution dr;
            expect(near_s(dr.scale(), 1.0F), "dynres starts at max_scale");
            for (int i = 0; i < 200; ++i) {
                (void)dr.update(20.0, ThermalStatus::None);
            }
            expect(near_s(dr.scale(), 1.0F), "dynres steady under-budget load stays at max_scale");
        }

        // Sustained overload steps down one step every frames_to_decide until min_scale, then clamps.
        {
            DynamicResolution dr;
            for (int i = 0; i < 29; ++i) {
                (void)dr.update(35.0, ThermalStatus::None);
            }
            expect(near_s(dr.scale(), 1.0F), "dynres waits frames_to_decide before first step down");
            (void)dr.update(35.0, ThermalStatus::None);
            expect(near_s(dr.scale(), 0.95F), "dynres steps down 1 step at frame 30");
            (void)dr.update(35.0, ThermalStatus::None);
            expect(near_s(dr.scale(), 0.95F), "dynres resets window after stepping down");
            for (int i = 1; i < 30; ++i) {
                (void)dr.update(35.0, ThermalStatus::None);
            }
            expect(near_s(dr.scale(), 0.90F), "dynres steps down 2nd step at frame 60");
            for (int i = 0; i < 300; ++i) {
                (void)dr.update(35.0, ThermalStatus::None);
            }
            expect(near_s(dr.scale(), 0.50F), "dynres clamps at min_scale under sustained overload");
        }

        // Recovery requires 2x frames_to_decide frames well below budget per step.
        {
            DynamicResolution dr;
            for (int i = 0; i < 30; ++i) {
                (void)dr.update(35.0, ThermalStatus::None);
            }
            expect(near_s(dr.scale(), 0.95F), "dynres drops to 0.95 before recovery test");
            for (int i = 0; i < 59; ++i) {
                (void)dr.update(15.0, ThermalStatus::None);
            }
            expect(near_s(dr.scale(), 0.95F),
                   "dynres does not recover before 2 * frames_to_decide frames");
            (void)dr.update(15.0, ThermalStatus::None);
            expect(near_s(dr.scale(), 1.0F),
                   "dynres steps up 1 step at 2 * frames_to_decide frames well below budget");
        }

        // No oscillation on load exactly at the threshold (33.3 * 0.85 = 28.305 ms)
        // or inside the deadband.
        {
            DynamicResolution dr;
            const double threshold_ms = 33.3 * 0.85;
            bool changed = false;
            for (int i = 0; i < 300; ++i) {
                const float s = dr.update(threshold_ms, ThermalStatus::None);
                if (!near_s(s, 1.0F)) {
                    changed = true;
                }
            }
            expect(!changed, "dynres does not step down at exact threshold load");

            // Step down to 0.85, then run at exact threshold: stays at 0.85 without oscillating.
            for (int i = 0; i < 90; ++i) {
                (void)dr.update(35.0, ThermalStatus::None);
            }
            expect(near_s(dr.scale(), 0.85F), "dynres stepped to 0.85 for deadband test");
            bool oscillated = false;
            for (int i = 0; i < 300; ++i) {
                const float s = dr.update(threshold_ms, ThermalStatus::None);
                if (!near_s(s, 0.85F)) {
                    oscillated = true;
                }
            }
            expect(!oscillated, "dynres does not oscillate when load sits at threshold");
        }

        // Thermal caps: none/light=1.0, moderate=0.85, severe=0.75, critical/emergency/shutdown=min_scale,
        // and lifting thermal back to none/light allows recovery; reset() returns to max_scale.
        {
            DynamicResolution dr;
            expect(near_s(dr.update(15.0, ThermalStatus::Light), 1.0F),
                   "dynres thermal light allows max_scale");
            expect(near_s(dr.update(15.0, ThermalStatus::Moderate), 0.85F),
                   "dynres thermal moderate caps at 0.85");
            expect(near_s(dr.update(15.0, ThermalStatus::Severe), 0.75F),
                   "dynres thermal severe caps at 0.75");
            expect(near_s(dr.update(15.0, ThermalStatus::Critical), 0.50F),
                   "dynres thermal critical caps at min_scale");
            expect(near_s(dr.update(15.0, ThermalStatus::Emergency), 0.50F),
                   "dynres thermal emergency caps at min_scale");
            expect(near_s(dr.update(15.0, ThermalStatus::Shutdown), 0.50F),
                   "dynres thermal shutdown caps at min_scale");

            // Lifting cap to none allows recovery over 2 * frames_to_decide per 0.05 step.
            for (int i = 0; i < 600; ++i) {
                (void)dr.update(15.0, ThermalStatus::None);
            }
            expect(near_s(dr.scale(), 1.0F),
                   "dynres lifting thermal to none allows recovery back to max_scale");

            (void)dr.update(15.0, ThermalStatus::Critical);
            expect(near_s(dr.scale(), 0.50F), "dynres drops to 0.50 on critical before reset");
            dr.reset();
            expect(near_s(dr.scale(), 1.0F), "dynres reset restores max_scale");
        }

        // render_size multiple-of-8 rounding and clamping.
        {
            const auto s75 = DynamicResolution::render_size(1920U, 1080U, 0.75F);
            const auto s100 = DynamicResolution::render_size(1920U, 1080U, 1.0F);
            const auto s50 = DynamicResolution::render_size(1920U, 1080U, 0.5F);
            const auto s0 = DynamicResolution::render_size(1920U, 1080U, 0.0F);
            const auto s_over = DynamicResolution::render_size(1920U, 1080U, 1.5F);
            const auto s_odd = DynamicResolution::render_size(1366U, 768U, 1.0F);
            const auto s_tiny = DynamicResolution::render_size(16U, 16U, 0.1F);
            const auto s_sub8 = DynamicResolution::render_size(4U, 4U, 0.5F);
            const auto s_zero = DynamicResolution::render_size(0U, 0U, 1.0F);

            expect(s75.first == 1440U && s75.second == 808U,
                   "render_size(1920x1080, 0.75) rounds to 1440x808");
            expect(s100.first == 1920U && s100.second == 1080U,
                   "render_size(1920x1080, 1.0) keeps 1920x1080");
            expect(s50.first == 960U && s50.second == 544U,
                   "render_size(1920x1080, 0.5) rounds 540 to 544");
            expect(s0.first == 8U && s0.second == 8U,
                   "render_size clamps minimum to 8x8");
            expect(s_over.first == 1920U && s_over.second == 1080U,
                   "render_size clamps scale > 1.0 to output size");
            expect(s_odd.first == 1366U && s_odd.second == 768U,
                   "render_size never exceeds non-multiple-of-8 output width");
            expect(s_tiny.first == 8U && s_tiny.second == 8U,
                   "render_size clamps small positive scale to at least 8");
            expect(s_sub8.first == 4U && s_sub8.second == 4U,
                   "render_size clamps sub-8 output to output size");
            expect(s_zero.first == 0U && s_zero.second == 0U,
                   "render_size preserves zero output dimensions");
        }
    }
}

// Reference ETC2 RGB8 + EAC alpha decoder (Khronos / OpenGL ES 3.0 spec:
// Individual, Differential, T, H, and Planar RGB modes + 16-table EAC alpha).
namespace etc2_ref {

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
        const std::uint8_t r[2] = {expand4((blk[0] >> 4U) & 0xFU), expand4(blk[0] & 0xFU)};
        const std::uint8_t g[2] = {expand4((blk[1] >> 4U) & 0xFU), expand4(blk[1] & 0xFU)};
        const std::uint8_t b[2] = {expand4((blk[2] >> 4U) & 0xFU), expand4(blk[2] & 0xFU)};
        const unsigned t[2] = {(blk[3] >> 5U) & 0x7U, (blk[3] >> 2U) & 0x7U};
        for (int y = 0; y < 4; ++y) {
            for (int x = 0; x < 4; ++x) {
                const unsigned sub = flip ? (y >= 2 ? 1U : 0U) : (x >= 2 ? 1U : 0U);
                const int mod = kModifier[t[sub]][read_pixel_index(x, y)];
                out_rgb[y][x][0] = clamp_u8(static_cast<int>(r[sub]) + mod);
                out_rgb[y][x][1] = clamp_u8(static_cast<int>(g[sub]) + mod);
                out_rgb[y][x][2] = clamp_u8(static_cast<int>(b[sub]) + mod);
            }
        }
        return;
    }

    const int r1_5 = static_cast<int>((blk[0] >> 3U) & 0x1FU);
    const int g1_5 = static_cast<int>((blk[1] >> 3U) & 0x1FU);
    const int b1_5 = static_cast<int>((blk[2] >> 3U) & 0x1FU);
    const int r2_5 = r1_5 + sign_extend_3(blk[0] & 0x7U);
    const int g2_5 = g1_5 + sign_extend_3(blk[1] & 0x7U);
    const int b2_5 = b1_5 + sign_extend_3(blk[2] & 0x7U);

    if (r2_5 < 0 || r2_5 > 31) {
        // T-mode
        const std::uint32_t r1_4 = ((static_cast<std::uint32_t>(blk[0]) >> 3U) & 0x3U) << 2U |
                                   (static_cast<std::uint32_t>(blk[0]) & 0x3U);
        const std::uint8_t r1 = expand4(r1_4);
        const std::uint8_t g1 = expand4((blk[1] >> 4U) & 0xFU);
        const std::uint8_t b1 = expand4(blk[1] & 0xFU);
        const std::uint8_t r2 = expand4((blk[2] >> 4U) & 0xFU);
        const std::uint8_t g2 = expand4(blk[2] & 0xFU);
        const std::uint8_t b2 = expand4((blk[3] >> 4U) & 0xFU);
        const unsigned d_idx = (((blk[3] >> 2U) & 0x3U) << 1U) | (blk[3] & 0x1U);
        const int dist = kTHDist[d_idx];
        const std::uint8_t paint[4][3] = {
            {r1, g1, b1},
            {clamp_u8(r2 + dist), clamp_u8(g2 + dist), clamp_u8(b2 + dist)},
            {r2, g2, b2},
            {clamp_u8(r2 - dist), clamp_u8(g2 - dist), clamp_u8(b2 - dist)},
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
        const std::uint8_t r1 = expand4((blk[0] >> 3U) & 0xFU);
        const std::uint8_t g1 = expand4(((blk[0] & 0x7U) << 1U) | ((blk[1] >> 4U) & 0x1U));
        const std::uint8_t b1 =
            expand4((blk[1] & 0x8U) | ((blk[1] & 0x3U) << 1U) | ((blk[2] >> 7U) & 0x1U));
        const std::uint8_t r2 = expand4((blk[2] >> 3U) & 0xFU);
        const std::uint8_t g2 = expand4(((blk[2] & 0x7U) << 1U) | ((blk[3] >> 7U) & 0x1U));
        const std::uint8_t b2 = expand4((blk[3] >> 3U) & 0xFU);
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
            {clamp_u8(r1 + dist), clamp_u8(g1 + dist), clamp_u8(b1 + dist)},
            {clamp_u8(r1 - dist), clamp_u8(g1 - dist), clamp_u8(b1 - dist)},
            {clamp_u8(r2 + dist), clamp_u8(g2 + dist), clamp_u8(b2 + dist)},
            {clamp_u8(r2 - dist), clamp_u8(g2 - dist), clamp_u8(b2 - dist)},
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
        const int ro = expand6((blk[0] >> 1U) & 0x3FU);
        const int go = expand7(((blk[0] & 0x1U) << 6U) | ((blk[1] >> 1U) & 0x3FU));
        const int bo = expand6(((blk[1] & 0x1U) << 5U) |
                               (((blk[2] >> 3U) & 0x3U) << 3U) |
                               ((blk[2] & 0x3U) << 1U) |
                               ((blk[3] >> 7U) & 0x1U));
        const int rh = expand6((((blk[3] >> 2U) & 0x1FU) << 1U) | (blk[3] & 0x1U));
        const int gh = expand7((blk[4] >> 1U) & 0x7FU);
        const int bh = expand6(((blk[4] & 0x1U) << 5U) | ((blk[5] >> 3U) & 0x1FU));
        const int rv = expand6(((blk[5] & 0x7U) << 3U) | ((blk[6] >> 5U) & 0x7U));
        const int gv = expand7(((blk[6] & 0x1FU) << 2U) | ((blk[7] >> 6U) & 0x3U));
        const int bv = expand6(blk[7] & 0x3FU);
        for (int y = 0; y < 4; ++y) {
            for (int x = 0; x < 4; ++x) {
                out_rgb[y][x][0] = clamp_u8((x * (rh - ro) + y * (rv - ro) + 4 * ro + 2) >> 2);
                out_rgb[y][x][1] = clamp_u8((x * (gh - go) + y * (gv - go) + 4 * go + 2) >> 2);
                out_rgb[y][x][2] = clamp_u8((x * (bh - bo) + y * (bv - bo) + 4 * bo + 2) >> 2);
            }
        }
        return;
    }

    // Differential mode
    const std::uint8_t r[2] = {expand5(static_cast<std::uint32_t>(r1_5)),
                               expand5(static_cast<std::uint32_t>(r2_5))};
    const std::uint8_t g[2] = {expand5(static_cast<std::uint32_t>(g1_5)),
                               expand5(static_cast<std::uint32_t>(g2_5))};
    const std::uint8_t b[2] = {expand5(static_cast<std::uint32_t>(b1_5)),
                               expand5(static_cast<std::uint32_t>(b2_5))};
    const unsigned t[2] = {(blk[3] >> 5U) & 0x7U, (blk[3] >> 2U) & 0x7U};
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            const unsigned sub = flip ? (y >= 2 ? 1U : 0U) : (x >= 2 ? 1U : 0U);
            const int mod = kModifier[t[sub]][read_pixel_index(x, y)];
            out_rgb[y][x][0] = clamp_u8(static_cast<int>(r[sub]) + mod);
            out_rgb[y][x][1] = clamp_u8(static_cast<int>(g[sub]) + mod);
            out_rgb[y][x][2] = clamp_u8(static_cast<int>(b[sub]) + mod);
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
            out_a[y][x] = clamp_u8(base + kEacAlpha[tbl][idx] * mul);
        }
    }
}

inline std::vector<std::uint8_t> decode_level_rgba8(const wse::asset::etc2::Level& lvl,
                                                    wse::asset::etc2::Format fmt) {
    const std::uint32_t w = lvl.width;
    const std::uint32_t h = lvl.height;
    const std::uint32_t bx_count = (w + 3U) / 4U;
    const std::uint32_t by_count = (h + 3U) / 4U;
    const std::size_t stride = (fmt == wse::asset::etc2::Format::rgba8) ? 16U : 8U;
    std::vector<std::uint8_t> rgba(static_cast<std::size_t>(w) * h * 4U, 255U);
    for (std::uint32_t by = 0; by < by_count; ++by) {
        for (std::uint32_t bx = 0; bx < bx_count; ++bx) {
            const std::uint8_t* blk =
                lvl.blocks.data() + (static_cast<std::size_t>(by) * bx_count + bx) * stride;
            std::uint8_t rgb[4][4][3]{};
            std::uint8_t a[4][4]{};
            if (fmt == wse::asset::etc2::Format::rgba8) {
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
                    rgba[dst_idx + 0U] = rgb[py][px][0];
                    rgba[dst_idx + 1U] = rgb[py][px][1];
                    rgba[dst_idx + 2U] = rgb[py][px][2];
                    rgba[dst_idx + 3U] = (fmt == wse::asset::etc2::Format::rgba8) ? a[py][px] : 255U;
                }
            }
        }
    }
    return rgba;
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

}  // namespace etc2_ref

void write_bc1_gradient_mip(wse::io::BinaryWriter& w,
                            std::uint32_t width,
                            std::uint32_t height,
                            bool with_alpha,
                            bool force_transparent_texel) {
    const std::uint32_t bx_count = (width + 3U) / 4U;
    const std::uint32_t by_count = (height + 3U) / 4U;
    for (std::uint32_t by = 0; by < by_count; ++by) {
        for (std::uint32_t bx = 0; bx < bx_count; ++bx) {
            const int r0 = static_cast<int>((bx * 200U) / (std::max)(1U, bx_count));
            const int g0 = static_cast<int>((by * 200U) / (std::max)(1U, by_count));
            const int b0 = 64 + static_cast<int>(((bx + by) * 64U) / (std::max)(1U, bx_count + by_count));
            const int r1 = (std::min)(255, r0 + 16);
            const int g1 = (std::min)(255, g0 + 8);
            const int b1 = (std::min)(255, b0 + 16);
            std::uint16_t c0 = pack_rgb565(r0, g0, b0);
            std::uint16_t c1 = pack_rgb565(r1, g1, b1);
            // Smooth horizontal ramp across the 4 columns: 0 -> c0, 2 -> 2/3 c0 + 1/3 c1, 3 -> 1/3 c0 + 2/3 c1, 1 -> c1.
            std::uint32_t idx_bits = 0x78787878U;
            if (with_alpha) {
                if (c0 > c1) {
                    std::swap(c0, c1);
                }
                idx_bits = 0x55555555U;  // index 1 everywhere (no index 3)
                if (force_transparent_texel && bx == 0U && by == 0U) {
                    idx_bits = 0x55555553U;  // texel (0,0) uses index 3 -> transparent
                }
            } else {
                if (c0 <= c1) {
                    std::swap(c0, c1);
                }
            }
            push_u16_le(w, c0);
            push_u16_le(w, c1);
            push_u32_le(w, idx_bits);
        }
    }
}

void write_bc3_gradient_mip(wse::io::BinaryWriter& w, std::uint32_t width, std::uint32_t height) {
    const std::uint32_t bx_count = (width + 3U) / 4U;
    const std::uint32_t by_count = (height + 3U) / 4U;
    for (std::uint32_t by = 0; by < by_count; ++by) {
        for (std::uint32_t bx = 0; bx < bx_count; ++bx) {
            const std::uint8_t a0 = static_cast<std::uint8_t>((by * 200U) / (std::max)(1U, by_count) + 32U);
            const std::uint8_t a1 = static_cast<std::uint8_t>((std::max)(0, static_cast<int>(a0) - 24));
            w.write_byte(a0);
            w.write_byte(a1);
            // Smooth alpha indices across the 4x4 block (3 bits * 16 = 48 bits).
            const std::uint64_t abits = 0x765432107654ULL;
            for (int i = 0; i < 6; ++i) {
                w.write_byte(static_cast<std::uint8_t>((abits >> (i * 8)) & 0xFFU));
            }
            const int r0 = static_cast<int>((bx * 180U) / (std::max)(1U, bx_count));
            const int g0 = static_cast<int>((by * 180U) / (std::max)(1U, by_count));
            const std::uint16_t c0 = pack_rgb565(r0 + 16, g0 + 8, 136);
            const std::uint16_t c1 = pack_rgb565(r0, g0, 120);
            push_u16_le(w, c0);
            push_u16_le(w, c1);
            push_u32_le(w, 0x78787878U);
        }
    }
}

void write_bc2_mip(wse::io::BinaryWriter& w, std::uint32_t width, std::uint32_t height) {
    const std::uint32_t bx_count = (width + 3U) / 4U;
    const std::uint32_t by_count = (height + 3U) / 4U;
    for (std::uint32_t by = 0; by < by_count; ++by) {
        for (std::uint32_t bx = 0; bx < bx_count; ++bx) {
            const std::uint32_t a_base = (by * 12U) / (std::max)(1U, by_count);
            for (int r = 0; r < 4; ++r) {
                const std::uint32_t a = (std::min)(15U, a_base + static_cast<std::uint32_t>(r));
                const std::uint16_t row_a = static_cast<std::uint16_t>(
                    a | (a << 4U) | (a << 8U) | (a << 12U));
                push_u16_le(w, row_a);
            }
            const int r0 = 40 + static_cast<int>(bx * 32U);
            const int g0 = 80 + static_cast<int>(by * 32U);
            const std::uint16_t c0 = pack_rgb565(r0 + 16, g0 + 12, 140);
            const std::uint16_t c1 = pack_rgb565(r0, g0, 120);
            push_u16_le(w, c0);
            push_u16_le(w, c1);
            push_u32_le(w, 0x78787878U);
        }
    }
}

void test_etc2_transcode_and_texture_cache() {
    using namespace wse::asset;
    using namespace wse::asset::etc2;

    // 1. Reference decoder check across all 5 ETC2 RGB modes + EAC alpha, plus 4x4 BGRX transcode.
    {
        std::uint8_t rgb[4][4][3]{};
        // Individual mode (diff=0, flip=0): sub0=(0x44,0x88,0xCC), sub1=(0x22,0x66,0xAA), tbl=(0,1)
        // word_lo = 0: all pixels use index 0 -> +2 on sub0, +5 on sub1.
        const std::uint8_t ind_blk[8] = {0x42U, 0x86U, 0xCAU, 0x04U, 0U, 0U, 0U, 0U};
        etc2_ref::decode_etc2_rgb8_block(ind_blk, rgb);
        expect(rgb[0][0][0] == 0x46U && rgb[0][0][1] == 0x8AU && rgb[0][0][2] == 0xCEU &&
                   rgb[3][2][0] == 0x27U && rgb[3][2][1] == 0x6BU && rgb[3][2][2] == 0xAFU,
               "etc2 ref: individual mode decode");

        // Differential mode (diff=1, flip=1): r1=16(132), r2=17(140), g1=10(84), g2=11(92), b1=8(66), b2=9(74)
        // tbl=(0,0), all indices 0 (+2) -> y<2=(134,86,68), y>=2=(142,94,76).
        const std::uint8_t diff_blk[8] = {
            static_cast<std::uint8_t>((16U << 3U) | 1U),
            static_cast<std::uint8_t>((10U << 3U) | 1U),
            static_cast<std::uint8_t>((8U << 3U) | 1U),
            0x03U,
            0U, 0U, 0U, 0U,
        };
        etc2_ref::decode_etc2_rgb8_block(diff_blk, rgb);
        expect(rgb[0][0][0] == 134U && rgb[0][0][1] == 84U && rgb[0][0][2] == 68U &&
                   rgb[2][0][0] == 142U && rgb[2][0][1] == 92U && rgb[2][0][2] == 76U,
               "etc2 ref: differential mode decode");

        // T mode (diff=1, r1_5=0, dr=-4 -> r2_5=-4 overflows): r1=0, g1=0x22, b1=0x44, r2=0x88, g2=0xAA, b2=0xCC, d_idx=0 (dist=3)
        // px(0,0) idx=0 -> (0,0x22,0x44); px(1,0) idx=1 -> (0x8B,0xAD,0xCF); px(0,1) idx=2 -> (0x88,0xAA,0xCC); px(1,1) idx=3 -> (0x85,0xA7,0xC9)
        const std::uint8_t t_blk[8] = {0x04U, 0x24U, 0x8AU, 0xC2U, 0x00U, 0x22U, 0x00U, 0x30U};
        etc2_ref::decode_etc2_rgb8_block(t_blk, rgb);
        expect(rgb[0][0][0] == 0x00U && rgb[0][0][1] == 0x22U && rgb[0][0][2] == 0x44U &&
                   rgb[0][1][0] == 0x8BU && rgb[0][1][1] == 0xADU && rgb[0][1][2] == 0xCFU &&
                   rgb[1][0][0] == 0x88U && rgb[1][0][1] == 0xAAU && rgb[1][0][2] == 0xCCU &&
                   rgb[1][1][0] == 0x85U && rgb[1][1][1] == 0xA7U && rgb[1][1][2] == 0xC9U,
               "etc2 ref: T mode decode");

        // H mode (diff=1, r2_5 in 0..31, g1_5=0, dg=-1 -> g2_5=-1 overflows):
        // c1=(0x22,0x44,0x66), c2=(0x11,0x33,0x55) -> c1>=c2 so dist bit0=1, d_idx=1 (dist=6)
        // px(0,0) idx=0 -> (0x28,0x4A,0x6C); px(1,0) idx=2 -> (0x17,0x39,0x5B)
        const std::uint8_t h_blk[8] = {0x12U, 0x07U, 0x09U, 0xAAU, 0x00U, 0x10U, 0x00U, 0x00U};
        etc2_ref::decode_etc2_rgb8_block(h_blk, rgb);
        expect(rgb[0][0][0] == 0x28U && rgb[0][0][1] == 0x4AU && rgb[0][0][2] == 0x6CU &&
                   rgb[0][1][0] == 0x17U && rgb[0][1][1] == 0x39U && rgb[0][1][2] == 0x5BU,
               "etc2 ref: H mode decode");

        // Planar mode (diff=1, r2/g2 in 0..31, b1_5=0, db=-4 -> b2_5=-4 overflows):
        // ro=16(65), go=32(64), bo=16(65), rh=32(130), gh=64(129), bh=32(130), rv=16(65), gv=32(64), bv=16(65)
        const std::uint8_t p_blk[8] = {
            static_cast<std::uint8_t>((16U << 1U) | 0U),
            static_cast<std::uint8_t>((32U << 1U) | 0U),
            static_cast<std::uint8_t>((2U << 3U) | 4U),
            static_cast<std::uint8_t>((16U << 2U) | 2U),
            static_cast<std::uint8_t>((64U << 1U) | 1U),
            static_cast<std::uint8_t>((0U << 3U) | 2U),
            static_cast<std::uint8_t>((0U << 5U) | 8U),
            static_cast<std::uint8_t>((0U << 6U) | 16U),
        };
        etc2_ref::decode_etc2_rgb8_block(p_blk, rgb);
        expect(rgb[0][0][0] == 65U && rgb[0][0][1] == 64U && rgb[0][0][2] == 65U &&
                   rgb[0][2][0] == 98U && rgb[0][2][1] == 97U && rgb[0][2][2] == 98U,
               "etc2 ref: planar mode decode");

        // EAC alpha (base=128, mul=2, tbl=0 {-3,-6,-9,-15,2,5,8,14}): all indices 4 (+2*2 = 132) except p0=7 (+14*2=156)
        const std::uint8_t eac_blk[8] = {128U, 0x20U, 0xF2U, 0x49U, 0x24U, 0x92U, 0x49U, 0x24U};
        std::uint8_t alpha[4][4]{};
        etc2_ref::decode_eac_alpha_block(eac_blk, alpha);
        expect(alpha[0][0] == 156U && alpha[1][0] == 132U && alpha[3][3] == 132U,
               "etc2 ref: EAC alpha decode");

        // Encode via transcode_dds of an uncompressed 4x4 smooth gradient and check PSNR.
        wse::io::BinaryWriter dds_w;
        write_dds_header(dds_w, 4U, 4U, 0x100FU, 1U, 0x40U, 0U, 32U,
                         0x00FF0000U, 0x0000FF00U, 0x000000FFU, 0x00000000U);
        for (int y = 0; y < 4; ++y) {
            for (int x = 0; x < 4; ++x) {
                dds_w.write_byte(static_cast<std::uint8_t>(100 + (x + y) * 3));
                dds_w.write_byte(static_cast<std::uint8_t>(110 + (x + y) * 3));
                dds_w.write_byte(static_cast<std::uint8_t>(120 + (x + y) * 3));
                dds_w.write_byte(255U);
            }
        }
        DdsImageView view;
        std::string err;
        expect(parse_dds(dds_w.bytes(), view, &err), "etc2 ref test: parse 4x4 BGRX");
        Image img;
        expect(transcode_dds(view, 0U, img, &err), "etc2 ref test: transcode 4x4 BGRX");
        expect(img.format == Format::rgb8 && img.data.size() == 1U && img.data[0].blocks.size() == 8U,
               "etc2 ref test: BGRX -> rgb8 8 bytes");
        std::vector<std::uint8_t> src_rgba;
        expect(decode_dds_mip_rgba8(view, 0U, 0U, src_rgba, &err), "etc2 ref test: decode source");
        const auto dec_rgba = etc2_ref::decode_level_rgba8(img.data[0], img.format);
        expect(etc2_ref::compute_rgb_psnr_db(src_rgba, dec_rgba) >= 30.0,
               "etc2 ref test: 4x4 planar/diff decode PSNR >= 30 dB");
    }

    // 2. DXT1 opaque 64x64 with full mip chain (includes 2x2 and 1x1 mips).
    {
        wse::io::BinaryWriter w;
        write_dds_header(w, 64U, 64U, 0x21007U, 7U, 0x4U, dds_detail::fourcc('D', 'X', 'T', '1'),
                         0U, 0U, 0U, 0U, 0U, 0x401008U);
        std::uint32_t mw = 64U, mh = 64U;
        for (int m = 0; m < 7; ++m) {
            write_bc1_gradient_mip(w, mw, mh, false, false);
            mw = (std::max)(1U, mw >> 1U);
            mh = (std::max)(1U, mh >> 1U);
        }
        DdsImageView view;
        std::string err;
        expect(parse_dds(w.bytes(), view, &err), "etc2: parse opaque DXT1 64x64 mipmap");
        Image img;
        expect(transcode_dds(view, 0U, img, &err), "etc2: transcode opaque DXT1 64x64");
        expect(img.format == Format::rgb8 && img.width == 64U && img.height == 64U &&
                   img.levels == 7U && img.faces == 1U && img.data.size() == 7U,
               "etc2: opaque DXT1 -> rgb8, 7 levels");
        expect(img.data[5].width == 2U && img.data[5].height == 2U && img.data[5].blocks.size() == 8U &&
                   img.data[6].width == 1U && img.data[6].height == 1U && img.data[6].blocks.size() == 8U,
               "etc2: 2x2 and 1x1 mips padded to 1 4x4 block (8 bytes)");

        std::vector<std::uint8_t> src_rgba;
        expect(decode_dds_mip_rgba8(view, 0U, 0U, src_rgba, &err), "etc2: decode DXT1 mip 0");
        const auto dec_rgba = etc2_ref::decode_level_rgba8(img.data[0], img.format);
        const double psnr = etc2_ref::compute_rgb_psnr_db(src_rgba, dec_rgba);
        expect(psnr >= 30.0, "etc2: opaque DXT1 64x64 PSNR >= 30 dB");

        // Edge-replication check on 1x1 mip (every decoded pixel in the 4x4 block matches mip 6).
        std::vector<std::uint8_t> src_1x1;
        expect(decode_dds_mip_rgba8(view, 6U, 0U, src_1x1, &err), "etc2: decode DXT1 1x1 mip");
        const auto dec_1x1 = etc2_ref::decode_level_rgba8(img.data[6], img.format);
        expect(etc2_ref::compute_rgb_psnr_db(src_1x1, dec_1x1) >= 30.0,
               "etc2: 1x1 mip edge-replicated PSNR >= 30 dB");

        // first_mip skipping (skip mip 0 -> 32x32, 6 levels)
        Image skipped;
        expect(transcode_dds(view, 1U, skipped, &err), "etc2: transcode with first_mip=1");
        expect(skipped.width == 32U && skipped.height == 32U && skipped.levels == 6U &&
                   skipped.data.size() == 6U && skipped.data[0].blocks == img.data[1].blocks,
               "etc2: first_mip=1 skips mip 0 and matches mip 1..6 blocks");
        expect(!transcode_dds(view, 7U, skipped, &err) && !err.empty(),
               "etc2: out-of-range first_mip rejected");

        // threads 1 vs 4 produce identical bytes
        Image mt;
        expect(transcode_dds(view, 0U, mt, &err, 4U), "etc2: transcode with 4 threads");
        bool same_mt = (mt.format == img.format && mt.data.size() == img.data.size());
        for (std::size_t i = 0; same_mt && i < img.data.size(); ++i) {
            same_mt = (mt.data[i].blocks == img.data[i].blocks);
        }
        expect(same_mt, "etc2: threads=1 and threads=4 produce identical bytes");
    }

    // 3. DXT1 with c0 <= c1 but NO index-3 transparent texel -> still rgb8.
    //    DXT1 with index-3 transparent texel -> rgba8.
    {
        wse::io::BinaryWriter w_no_trans;
        write_dds_header(w_no_trans, 8U, 8U, 0x1007U, 1U, 0x4U, dds_detail::fourcc('D', 'X', 'T', '1'),
                         0U, 0U, 0U, 0U, 0U);
        write_bc1_gradient_mip(w_no_trans, 8U, 8U, true, false);
        DdsImageView v_no;
        expect(parse_dds(w_no_trans.bytes(), v_no), "etc2: parse DXT1 c0<=c1 without index 3");
        Image img_no;
        expect(transcode_dds(v_no, 0U, img_no) && img_no.format == Format::rgb8,
               "etc2: DXT1 without transparent texels -> rgb8");

        wse::io::BinaryWriter w_trans;
        write_dds_header(w_trans, 8U, 8U, 0x1007U, 1U, 0x4U, dds_detail::fourcc('D', 'X', 'T', '1'),
                         0U, 0U, 0U, 0U, 0U);
        write_bc1_gradient_mip(w_trans, 8U, 8U, true, true);
        DdsImageView v_yes;
        expect(parse_dds(w_trans.bytes(), v_yes), "etc2: parse DXT1 with 1-bit alpha");
        Image img_yes;
        expect(transcode_dds(v_yes, 0U, img_yes) && img_yes.format == Format::rgba8 &&
                   img_yes.data[0].blocks.size() == 4U * 16U,
               "etc2: DXT1 with transparent texel -> rgba8");
        std::vector<std::uint8_t> src_rgba;
        expect(decode_dds_mip_rgba8(v_yes, 0U, 0U, src_rgba), "etc2: decode DXT1 alpha");
        const auto dec_rgba = etc2_ref::decode_level_rgba8(img_yes.data[0], img_yes.format);
        expect(etc2_ref::max_alpha_diff(src_rgba, dec_rgba) <= 16,
               "etc2: DXT1 1-bit alpha EAC error <= 16");
    }

    // 4. DXT5 with alpha gradient -> rgba8, RGB PSNR >= 30 dB, alpha error <= 16.
    {
        wse::io::BinaryWriter w;
        write_dds_header(w, 32U, 32U, 0x1007U, 1U, 0x4U, dds_detail::fourcc('D', 'X', 'T', '5'),
                         0U, 0U, 0U, 0U, 0U);
        write_bc3_gradient_mip(w, 32U, 32U);
        DdsImageView view;
        expect(parse_dds(w.bytes(), view), "etc2: parse DXT5 32x32");
        Image img;
        expect(transcode_dds(view, 0U, img) && img.format == Format::rgba8 &&
                   img.data[0].blocks.size() == 64U * 16U,
               "etc2: DXT5 -> rgba8 (64 blocks * 16 B)");
        std::vector<std::uint8_t> src_rgba;
        expect(decode_dds_mip_rgba8(view, 0U, 0U, src_rgba), "etc2: decode DXT5");
        const auto dec_rgba = etc2_ref::decode_level_rgba8(img.data[0], img.format);
        expect(etc2_ref::compute_rgb_psnr_db(src_rgba, dec_rgba) >= 30.0 &&
                   etc2_ref::max_alpha_diff(src_rgba, dec_rgba) <= 16,
               "etc2: DXT5 RGB PSNR >= 30 dB and EAC alpha within +-16");
    }

    // 5. DXT3 -> rgba8, block byte count, alpha within +-16.
    {
        wse::io::BinaryWriter w;
        write_dds_header(w, 16U, 16U, 0x1007U, 1U, 0x4U, dds_detail::fourcc('D', 'X', 'T', '3'),
                         0U, 0U, 0U, 0U, 0U);
        write_bc2_mip(w, 16U, 16U);
        DdsImageView view;
        expect(parse_dds(w.bytes(), view), "etc2: parse DXT3 16x16");
        Image img;
        expect(transcode_dds(view, 0U, img) && img.format == Format::rgba8 &&
                   img.data[0].blocks.size() == 16U * 16U,
               "etc2: DXT3 -> rgba8 (16 blocks * 16 B)");
        std::vector<std::uint8_t> src_rgba;
        expect(decode_dds_mip_rgba8(view, 0U, 0U, src_rgba), "etc2: decode DXT3");
        const auto dec_rgba = etc2_ref::decode_level_rgba8(img.data[0], img.format);
        expect(etc2_ref::compute_rgb_psnr_db(src_rgba, dec_rgba) >= 30.0 &&
                   etc2_ref::max_alpha_diff(src_rgba, dec_rgba) <= 16,
               "etc2: DXT3 RGB PSNR >= 30 dB and EAC alpha within +-16");
    }

    // 6. Uncompressed A8R8G8B8 with 2x2 and 1x1 mips -> rgba8, PSNR >= 30 dB, alpha <= 16.
    {
        wse::io::BinaryWriter w;
        write_dds_header(w, 16U, 16U, 0x2100FU, 5U, 0x41U, 0U, 32U,
                         0x00FF0000U, 0x0000FF00U, 0x000000FFU, 0xFF000000U, 0x401008U);
        std::uint32_t mw = 16U, mh = 16U;
        for (int m = 0; m < 5; ++m) {
            for (std::uint32_t y = 0; y < mh; ++y) {
                for (std::uint32_t x = 0; x < mw; ++x) {
                    const std::uint8_t b = static_cast<std::uint8_t>(40U + (x * 96U) / (std::max)(1U, mw));
                    const std::uint8_t g = static_cast<std::uint8_t>(60U + (y * 96U) / (std::max)(1U, mh));
                    const std::uint8_t r = static_cast<std::uint8_t>(80U + ((x + y) * 48U) / (std::max)(1U, mw));
                    const std::uint8_t a = static_cast<std::uint8_t>(64U + ((x * 160U) / (std::max)(1U, mw)));
                    w.write_byte(b);
                    w.write_byte(g);
                    w.write_byte(r);
                    w.write_byte(a);
                }
            }
            mw = (std::max)(1U, mw >> 1U);
            mh = (std::max)(1U, mh >> 1U);
        }
        DdsImageView view;
        expect(parse_dds(w.bytes(), view), "etc2: parse A8R8G8B8 16x16 mipmap");
        Image img;
        expect(transcode_dds(view, 0U, img) && img.format == Format::rgba8 && img.levels == 5U &&
                   img.data[3].width == 2U && img.data[3].height == 2U &&
                   img.data[3].blocks.size() == 16U &&
                   img.data[4].width == 1U && img.data[4].height == 1U &&
                   img.data[4].blocks.size() == 16U,
               "etc2: A8R8G8B8 -> rgba8, 2x2 and 1x1 mips padded to 16 bytes");
        std::vector<std::uint8_t> src_rgba;
        expect(decode_dds_mip_rgba8(view, 0U, 0U, src_rgba), "etc2: decode A8R8G8B8 mip 0");
        const auto dec_rgba = etc2_ref::decode_level_rgba8(img.data[0], img.format);
        expect(etc2_ref::compute_rgb_psnr_db(src_rgba, dec_rgba) >= 30.0 &&
                   etc2_ref::max_alpha_diff(src_rgba, dec_rgba) <= 16,
               "etc2: A8R8G8B8 RGB PSNR >= 30 dB and EAC alpha within +-16");
    }

    // 7. Cube map (6 faces) + threads 1 vs 4.
    {
        wse::io::BinaryWriter w;
        write_dds_header(w, 8U, 8U, 0x21007U, 2U, 0x4U, dds_detail::fourcc('D', 'X', 'T', '1'),
                         0U, 0U, 0U, 0U, 0U, 0x401008U, 0xFE00U);
        for (int f = 0; f < 6; ++f) {
            write_bc1_gradient_mip(w, 8U, 8U, false, false);
            write_bc1_gradient_mip(w, 4U, 4U, false, false);
        }
        DdsImageView view;
        expect(parse_dds(w.bytes(), view) && view.face_count == 6U && view.mip_count == 2U,
               "etc2: parse 6-face DXT1 cubemap");
        Image img1, img4;
        expect(transcode_dds(view, 0U, img1, nullptr, 1U) &&
                   transcode_dds(view, 0U, img4, nullptr, 4U) &&
                   img1.faces == 6U && img1.levels == 2U && img1.data.size() == 12U,
               "etc2: cubemap transcodes 6 faces * 2 levels");
        bool cube_same = (img1.data.size() == img4.data.size());
        for (std::size_t i = 0; cube_same && i < img1.data.size(); ++i) {
            cube_same = (img1.data[i].blocks == img4.data[i].blocks);
        }
        expect(cube_same, "etc2: cubemap threads=1 and threads=4 give identical bytes");

        // texture_cache.h round-trip + corruption + cache_key tests
        const std::vector<std::uint8_t> encoded = encode(img1);
        expect(!encoded.empty(), "texture_cache: encode cubemap");
        Image decoded;
        std::string cache_err;
        expect(decode(encoded, decoded, &cache_err) && decoded.format == img1.format &&
                   decoded.width == img1.width && decoded.height == img1.height &&
                   decoded.levels == img1.levels && decoded.faces == img1.faces &&
                   decoded.data.size() == img1.data.size() &&
                   decoded.data[11].blocks == img1.data[11].blocks,
               "texture_cache: encode/decode round-trip");

        auto corrupt_u32 = [&](std::size_t offset, std::uint32_t value) {
            std::vector<std::uint8_t> bad = encoded;
            bad[offset + 0] = static_cast<std::uint8_t>(value & 0xFFU);
            bad[offset + 1] = static_cast<std::uint8_t>((value >> 8U) & 0xFFU);
            bad[offset + 2] = static_cast<std::uint8_t>((value >> 16U) & 0xFFU);
            bad[offset + 3] = static_cast<std::uint8_t>((value >> 24U) & 0xFFU);
            return bad;
        };
        std::vector<std::uint8_t> bad_magic = encoded;
        bad_magic[0] = 'X';
        std::vector<std::uint8_t> truncated(encoded.begin(), encoded.end() - 1);
        std::vector<std::uint8_t> trailing = encoded;
        trailing.push_back(0U);
        const std::vector<std::uint8_t> bad_version = corrupt_u32(8U, 99U);
        const std::vector<std::uint8_t> bad_format = corrupt_u32(12U, 7U);
        const std::vector<std::uint8_t> bad_dim = corrupt_u32(16U, 0U);
        const std::vector<std::uint8_t> huge_levels = corrupt_u32(24U, 0x7FFFFFFFU);
        const std::vector<std::uint8_t> huge_faces = corrupt_u32(28U, 0x7FFFFFFFU);
        const std::vector<std::uint8_t> bad_lvl_dim = corrupt_u32(32U, 16U);
        const std::vector<std::uint8_t> bad_lvl_bytes = corrupt_u32(40U, 999U);
        Image discard_out;
        expect(!decode(std::span<const std::uint8_t>(), discard_out) &&
                   !decode(bad_magic, discard_out) &&
                   !decode(truncated, discard_out) &&
                   !decode(trailing, discard_out) &&
                   !decode(bad_version, discard_out) &&
                   !decode(bad_format, discard_out) &&
                   !decode(bad_dim, discard_out) &&
                   !decode(huge_levels, discard_out) &&
                   !decode(huge_faces, discard_out) &&
                   !decode(bad_lvl_dim, discard_out) &&
                   !decode(bad_lvl_bytes, discard_out),
               "texture_cache: rejects empty, bad magic, truncation, trailing, version, format, oversized counts, and level mismatches");

        const auto src_u8 = to_u8(w.bytes());
        const std::string k0 = cache_key(src_u8, 0U);
        const std::string k0_again = cache_key(src_u8, 0U);
        const std::string k1 = cache_key(src_u8, 1U);
        std::vector<std::uint8_t> tweaked = src_u8;
        tweaked.back() ^= 0x1U;
        const std::string kt = cache_key(tweaked, 0U);
        auto is_valid_key = [](const std::string& k) {
            if (k.size() != 21U || k.substr(16U) != ".etc2") return false;
            for (std::size_t i = 0; i < 16U; ++i) {
                const char c = k[i];
                if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
            }
            return true;
        };
        expect(is_valid_key(k0) && k0 == k0_again && k0 != k1 && k0 != kt && k1 != kt,
               "texture_cache: cache_key is 16 lowercase hex + .etc2, stable, and sensitive to bytes and first_mip");
    }

    // 8. Retail Warband check (only when WSE_WARBAND_ROOT or default GOG install exists).
    {
        std::error_code ec;
        const char* root_env = std::getenv("WSE_WARBAND_ROOT");
        const std::filesystem::path retail_root(
            (root_env != nullptr && root_env[0] != '\0') ? root_env
                                                          : "C:/GOG Games/Mount and Blade - Warband");
        if (std::filesystem::exists(retail_root / "Modules/Native/module.ini", ec)) {
            std::vector<std::filesystem::path> dds_files;
            for (const std::filesystem::path& dir : {retail_root / "Modules/Native/Textures",
                                                     retail_root / "Textures"}) {
                if (!std::filesystem::exists(dir, ec)) {
                    continue;
                }
                for (const auto& entry : std::filesystem::recursive_directory_iterator(dir, ec)) {
                    if (!entry.is_regular_file(ec)) {
                        continue;
                    }
                    const std::string ext = wse::platform::ascii_lower(entry.path().extension().string());
                    if (ext == ".dds") {
                        dds_files.push_back(entry.path());
                    }
                }
            }
            std::sort(dds_files.begin(), dds_files.end());

            std::size_t count = 0;
            std::size_t retail_failures = 0;
            std::uint64_t total_in_bytes = 0;
            std::uint64_t total_out_bytes = 0;
            double total_psnr_db = 0.0;

            const unsigned retail_threads =
                (std::max)(1U, (std::min)(8U, std::thread::hardware_concurrency()));
            const auto t0 = std::chrono::steady_clock::now();
            std::vector<std::byte> file_bytes;
            std::vector<std::uint8_t> ref_rgba;
            for (const std::filesystem::path& path : dds_files) {
                std::ifstream fp(path, std::ios::binary | std::ios::ate);
                if (!fp) {
                    ++retail_failures;
                    continue;
                }
                const std::streamsize sz = fp.tellg();
                if (sz <= 0) {
                    ++retail_failures;
                    continue;
                }
                file_bytes.resize(static_cast<std::size_t>(sz));
                fp.seekg(0, std::ios::beg);
                if (!fp.read(reinterpret_cast<char*>(file_bytes.data()), sz)) {
                    ++retail_failures;
                    continue;
                }

                DdsImageView view;
                std::string err;
                if (!parse_dds(file_bytes, view, &err)) {
                    ++retail_failures;
                    std::printf("FAIL  retail parse_dds %s: %s\n", path.string().c_str(), err.c_str());
                    continue;
                }
                Image etc_img;
                if (!transcode_dds(view, 0U, etc_img, &err, retail_threads)) {
                    ++retail_failures;
                    std::printf("FAIL  retail transcode_dds %s: %s\n", path.string().c_str(), err.c_str());
                    continue;
                }
                if (!decode_dds_mip_rgba8(view, 0U, 0U, ref_rgba, &err)) {
                    ++retail_failures;
                    std::printf("FAIL  retail decode_dds_mip_rgba8 %s: %s\n", path.string().c_str(), err.c_str());
                    continue;
                }

                const auto dec_rgba = etc2_ref::decode_level_rgba8(etc_img.data[0], etc_img.format);
                total_psnr_db += etc2_ref::compute_rgb_psnr_db(ref_rgba, dec_rgba);
                total_in_bytes += static_cast<std::uint64_t>(file_bytes.size());
                for (const Level& lvl : etc_img.data) {
                    total_out_bytes += static_cast<std::uint64_t>(lvl.blocks.size());
                }
                ++count;
            }
            const double seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            const double mean_psnr = (count > 0U) ? (total_psnr_db / static_cast<double>(count)) : 0.0;
            std::printf("retail etc2: count=%zu in=%llu out=%llu time=%.3fs mean_psnr=%.2f dB failures=%zu\n",
                        count,
                        static_cast<unsigned long long>(total_in_bytes),
                        static_cast<unsigned long long>(total_out_bytes),
                        seconds,
                        mean_psnr,
                        retail_failures);
            expect(count > 0U && retail_failures == 0U && mean_psnr >= 30.0,
                   "retail Warband DDS -> ETC2 transcoding succeeded with mean PSNR >= 30 dB");
        }
    }
}

}  // namespace

namespace wse::selftest {

int run_asset_suite(const char* fixture_path) {
    (void)fixture_path;
    failures = 0;
    test_brf_unversioned_all_eight_types();
    test_brf_rfver1_metamesh_and_catalog();
    test_dds_formats_and_resolver();
    test_binary_stream_strings();
    test_brf_hardening();
    test_dds_hardening();
    test_dds_decode_blocks_and_resolver_probes();
    test_p5_shaders_materials_and_spirv_cache();
    test_p5_shader_review_regressions();
    test_p5_lowering_review_regressions();
    test_d3d9_shader_translation();
    test_r5b_postfx_and_dynres();
    test_etc2_transcode_and_texture_cache();

    if (failures == 0) {
        std::printf("asset tests: PASS\n");
        return 0;
    }
    std::printf("asset tests: FAIL (%d)\n", failures);
    return 1;
}

}  // namespace wse::selftest

#ifndef WSE_SELF_TEST_LIBRARY
int main() {
    return wse::selftest::run_asset_suite(nullptr);
}
#endif
