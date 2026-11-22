// Phase 5 R5a step 4 (`docs/RECOMP_PLAN.md` "R5 — Graphics"): D3D9-shaped rendering backend
// interface, shared D3D9 state/effect/stateblock logic, and capture replayer.
//
// Header-only and STL-only: `<vulkan/vulkan.h>` stays isolated inside `platform/vulkan/`
// (`d3d9_vulkan.{h,cpp}`) so portable engine headers and host tests incur zero Vulkan header
// dependencies. Both `shim::D3d9` (live recompiled game) and `tools/r5_replay_vulkan.cpp`
// (offline capture comparison against `r5_replay_d3d9.cpp`) drive the same `D3d9Backend` seam.

#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "wse/recomp/d3d9_capture.h"
#include "wse/recomp/d3d9_shader.h"
#include "wse/recomp/fx_effect.h"
#include "wse/recomp/screen.h"

namespace wse::recomp {

// D3DVERTEXELEMENT9 (8 bytes).
struct D3d9VertexElement {
    std::uint16_t stream = 0;
    std::uint16_t offset = 0;
    std::uint8_t type = 0;
    std::uint8_t method = 0;
    std::uint8_t usage = 0;
    std::uint8_t usage_index = 0;
};

struct D3d9SamplerState {
    std::uint32_t address_u = 1;       // D3DTADDRESS_WRAP
    std::uint32_t address_v = 1;       // D3DTADDRESS_WRAP
    std::uint32_t address_w = 1;       // D3DTADDRESS_WRAP
    std::uint32_t border_color = 0;    // ARGB
    std::uint32_t mag_filter = 1;      // D3DTEXF_POINT
    std::uint32_t min_filter = 1;      // D3DTEXF_POINT
    std::uint32_t mip_filter = 0;      // D3DTEXF_NONE
    float lod_bias = 0.0F;
    std::uint32_t max_mip_level = 0;
    std::uint32_t max_anisotropy = 1;
    std::uint32_t srgb_texture = 0;

    void set(std::uint32_t type, std::uint32_t value) noexcept {
        switch (type) {
        case 1: address_u = value; break;
        case 2: address_v = value; break;
        case 3: address_w = value; break;
        case 4: border_color = value; break;
        case 5: mag_filter = value; break;
        case 6: min_filter = value; break;
        case 7: mip_filter = value; break;
        case 8: std::memcpy(&lod_bias, &value, 4); break;
        case 9: max_mip_level = value; break;
        case 10: max_anisotropy = value; break;
        case 11: srgb_texture = value; break;
        default: break;
        }
    }
};

struct D3d9StageConstants {
    std::array<float, 1024> c{};        // 256 vec4 float registers
    std::array<std::int32_t, 64> i{};   // 16 ivec4 int registers
    std::array<std::int32_t, 64> b{};   // 16 ivec4 bool registers (stored in lane .x)
};

struct D3d9StreamBinding {
    std::uint64_t buffer = 0;
    std::uint32_t offset = 0;
    std::uint32_t stride = 0;
    std::uint32_t freq = 1;  // D3DSTREAMSOURCE_* flags | divider
};

struct D3d9Viewport {
    std::uint32_t x = 0;
    std::uint32_t y = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    float min_z = 0.0F;
    float max_z = 1.0F;
};

struct D3d9Rect {
    std::int32_t left = 0;
    std::int32_t top = 0;
    std::int32_t right = 0;
    std::int32_t bottom = 0;
};

// `D3d9DrawState::depth_stencil` when the game set no depth-stencil surface
// (`SetDepthStencilSurface(NULL)`); 0 means the device's automatic one.
inline constexpr std::uint64_t kNoDepthStencil = ~0ULL;

struct D3d9DrawState {
    std::uint64_t vs = 0;
    std::uint64_t ps = 0;
    std::uint64_t decl = 0;
    std::uint64_t indices = 0;
    std::uint64_t render_target = 0;  // 0 = backbuffer
    std::uint64_t depth_stencil = 0;  // 0 = default auto depth-stencil
    D3d9Viewport viewport{};
    D3d9Rect scissor{};
    std::array<D3d9StreamBinding, 16> streams{};
    std::array<std::uint64_t, 16> textures{};
    std::array<D3d9SamplerState, 16> samplers{};
    std::array<std::uint32_t, 256> rs{};
    D3d9StageConstants vs_const{};
    D3d9StageConstants ps_const{};

    D3d9DrawState() { reset_defaults(); }

    void reset_defaults() noexcept {
        vs = ps = decl = indices = 0;
        render_target = depth_stencil = 0;
        streams = {};
        textures = {};
        for (D3d9SamplerState& s : samplers) s = D3d9SamplerState{};
        rs.fill(0);
        // Official Direct3D 9 default render state values.
        rs[7] = 1;    // D3DRS_ZENABLE = TRUE
        rs[8] = 3;    // D3DRS_FILLMODE = D3DFILL_SOLID
        rs[9] = 2;    // D3DRS_SHADEMODE = D3DSHADE_GOURAUD
        rs[14] = 1;   // D3DRS_ZWRITEENABLE = TRUE
        rs[15] = 0;   // D3DRS_ALPHATESTENABLE = FALSE
        rs[19] = 2;   // D3DRS_SRCBLEND = D3DBLEND_ONE
        rs[20] = 1;   // D3DRS_DESTBLEND = D3DBLEND_ZERO
        rs[22] = 3;   // D3DRS_CULLMODE = D3DCULL_CCW
        rs[23] = 4;   // D3DRS_ZFUNC = D3DCMP_LESSEQUAL
        rs[24] = 0;   // D3DRS_ALPHAREF = 0
        rs[25] = 8;   // D3DRS_ALPHAFUNC = D3DCMP_ALWAYS
        rs[27] = 0;   // D3DRS_ALPHABLENDENABLE = FALSE
        rs[168] = 0xF;  // D3DRS_COLORWRITEENABLE = RGBA
        rs[171] = 1;  // D3DRS_BLENDOP = D3DBLENDOP_ADD
        rs[174] = 0;  // D3DRS_SCISSORTESTENABLE = FALSE
        rs[175] = 0;  // D3DRS_SLOPESCALEDEPTHBIAS = 0.0f
        rs[195] = 0;  // D3DRS_DEPTHBIAS = 0.0f
        rs[206] = 0;  // D3DRS_SEPARATEALPHABLENDENABLE = FALSE
        rs[207] = 2;  // D3DRS_SRCBLENDALPHA = D3DBLEND_ONE
        rs[208] = 1;  // D3DRS_DESTBLENDALPHA = D3DBLEND_ZERO
        rs[209] = 1;  // D3DRS_BLENDOPALPHA = D3DBLENDOP_ADD
        vs_const = {};
        ps_const = {};
    }
};

// Driver capability report written on Vulkan device init (`docs/RECOMP_PLAN.md` R5a step 5).
struct D3d9DriverReport {
    bool ok = false;
    bool is_mock = false;
    std::string driver_mode;
    std::string device_name;
    std::string api_version;
    std::string driver_version;
    std::string presentation_path;
    std::uint32_t vendor_id = 0;
    std::uint32_t device_id = 0;
    std::uint32_t max_image_dimension_2d = 0;
    std::uint32_t max_bound_descriptor_sets = 0;
    std::uint32_t max_per_stage_samplers = 0;
    std::uint32_t max_vertex_output_components = 0;
    bool texture_compression_bc = false;
    bool texture_compression_etc2 = false;
    bool bgra_vertex_format = false;
    bool pipeline_cache_active = false;
    std::vector<std::string> missing_features;
    std::string error;

    [[nodiscard]] std::string format_summary() const {
        std::string s = "Vulkan D3D9: " + device_name + " (" + driver_mode + ", API " + api_version +
                        ", driver " + driver_version + ", " + presentation_path + ")";
        s += "\n  limits: max2D=" + std::to_string(max_image_dimension_2d) +
             " sets=" + std::to_string(max_bound_descriptor_sets) +
             " samplers/stage=" + std::to_string(max_per_stage_samplers) +
             " vs_out_comp=" + std::to_string(max_vertex_output_components);
        s += std::string("\n  features: BC=") + (texture_compression_bc ? "yes" : "no(RGBA8)") +
             " ETC2=" + (texture_compression_etc2 ? "yes" : "no") +
             " bgra_vtx=" + (bgra_vertex_format ? "yes" : "no(swizzle)") +
             " pipeline_cache=" + (pipeline_cache_active ? "yes" : "no");
        if (!missing_features.empty()) {
            s += "\n  missing/fallbacks:";
            for (const std::string& m : missing_features) s += " [" + m + "]";
        }
        return s;
    }
};

// Abstract rendering backend implemented by `platform/vulkan/d3d9_vulkan.cpp`.
class D3d9Backend {
public:
    virtual ~D3d9Backend() = default;

    virtual bool init(std::uint32_t width, std::uint32_t height, std::uint32_t color_fmt,
                      std::uint32_t depth_fmt, std::string* error_out = nullptr) = 0;
    virtual bool reset(std::uint32_t width, std::uint32_t height, std::uint32_t color_fmt,
                       std::uint32_t depth_fmt, std::string* error_out = nullptr) = 0;
    [[nodiscard]] virtual const D3d9DriverReport& report() const noexcept = 0;

    virtual void create_buffer(std::uint64_t id, bool is_index, std::uint32_t size,
                               bool dynamic, std::uint32_t fmt) = 0;
    // The bytes the game wrote under one Lock/Unlock: `offset` and `bytes` cover the locked range
    // (or, for a capture replay, the bytes that changed in it); `lock_flags` are the Lock's
    // D3DLOCK_* flags (DISCARD 0x2000, NOOVERWRITE 0x1000), which say whether queued draws may
    // still read the buffer's old contents.
    virtual void update_buffer(std::uint64_t id, std::uint32_t offset,
                               std::span<const std::uint8_t> bytes, std::uint32_t lock_flags) = 0;
    virtual void destroy_buffer(std::uint64_t id) = 0;

    // kind: 0 = 2D texture, 1 = Cube texture, 2 = Volume texture.
    virtual void create_texture(std::uint64_t id, int kind, std::uint32_t width, std::uint32_t height,
                                std::uint32_t depth, std::uint32_t levels, std::uint32_t fmt,
                                std::uint32_t usage) = 0;
    virtual void upload_texture_level(std::uint64_t id, std::uint32_t face, std::uint32_t level,
                                      std::span<const std::uint8_t> bytes, std::uint32_t row_pitch) = 0;
    virtual void upload_texture_dds(std::uint64_t id, std::span<const std::uint8_t> dds_bytes) = 0;
    virtual void destroy_texture(std::uint64_t id) = 0;

    virtual void create_surface(std::uint64_t id, std::uint32_t width, std::uint32_t height,
                                std::uint32_t fmt, std::uint32_t usage,
                                std::uint64_t parent_tex, std::uint32_t face, std::uint32_t level) = 0;
    virtual void destroy_surface(std::uint64_t id) = 0;

    virtual void create_vertex_decl(std::uint64_t id, std::span<const D3d9VertexElement> elems) = 0;
    virtual void destroy_vertex_decl(std::uint64_t id) = 0;

    virtual void create_shader(std::uint64_t id, bool is_pixel, std::span<const std::uint8_t> bytecode) = 0;
    virtual void destroy_shader(std::uint64_t id) = 0;

    virtual void clear(const D3d9DrawState& state, std::uint32_t flags, std::uint32_t argb,
                       float z, std::uint32_t stencil, std::span<const D3d9Rect> rects) = 0;
    virtual void draw_indexed(const D3d9DrawState& state, std::uint32_t prim_type,
                              std::int32_t base_vertex, std::uint32_t min_v, std::uint32_t num_v,
                              std::uint32_t start_index, std::uint32_t prim_count) = 0;
    virtual void draw_primitive(const D3d9DrawState& state, std::uint32_t prim_type,
                                std::uint32_t start_vertex, std::uint32_t prim_count) = 0;
    virtual bool present(std::vector<std::uint8_t>* out_rgba8 = nullptr,
                         std::uint32_t* out_width = nullptr,
                         std::uint32_t* out_height = nullptr) = 0;
};

// --- Shared fx_2_0 effect -> D3D9 shader constants & samplers binder -------------------------

namespace d3d9_fx_detail {

inline void pack_ctab_entry(const d3d9sh::CtabEntry& e,
                            std::span<const std::uint8_t> val,
                            std::uint32_t src_rows,
                            std::uint32_t src_cols,
                            std::uint32_t src_type,
                            D3d9StageConstants& dst) {
    if (e.reg_count == 0 || val.empty()) return;
    auto read_word = [&](std::size_t idx) -> std::uint32_t {
        if (idx * 4U + 4U > val.size()) return 0U;
        std::uint32_t w = 0;
        std::memcpy(&w, val.data() + idx * 4U, 4);
        return w;
    };
    auto word_to_float = [&](std::uint32_t w, std::uint32_t ty) -> float {
        if (ty == 1U) return w != 0U ? 1.0F : 0.0F;
        if (ty == 2U) return static_cast<float>(static_cast<std::int32_t>(w));
        float f = 0.0F;
        std::memcpy(&f, &w, 4);
        return f;
    };
    const std::uint32_t val_ty = src_type != 0 ? src_type : e.type;
    const std::uint32_t elems = std::max<std::uint32_t>(1U, e.elements);

    if (e.reg_set == 0U) {  // bool registers b#
        const std::uint32_t n = std::min<std::uint32_t>(elems, e.reg_count);
        for (std::uint32_t k = 0; k < n && static_cast<std::uint32_t>(e.reg_index) + k < 16U; ++k) {
            dst.b[(static_cast<std::size_t>(e.reg_index) + k) * 4U] = read_word(k) != 0U ? 1 : 0;
        }
        return;
    }
    if (e.reg_set == 1U) {  // int4 registers i#
        const std::uint32_t comps = std::max<std::uint32_t>(1U, std::min<std::uint32_t>(4U, e.cols > 0 ? e.cols : 1U));
        const std::uint32_t n = std::min<std::uint32_t>(elems, e.reg_count);
        for (std::uint32_t k = 0; k < n && static_cast<std::uint32_t>(e.reg_index) + k < 16U; ++k) {
            const std::size_t rbase = (static_cast<std::size_t>(e.reg_index) + k) * 4U;
            dst.i[rbase] = dst.i[rbase + 1] = dst.i[rbase + 2] = dst.i[rbase + 3] = 0;
            for (std::uint32_t c = 0; c < comps; ++c) {
                dst.i[rbase + c] = static_cast<std::int32_t>(read_word(static_cast<std::size_t>(k) * comps + c));
            }
        }
        return;
    }
    if (e.reg_set != 2U) return;  // float4 registers c#

    if (e.cls == 2U || e.cls == 3U) {  // matrix_rows (2) or matrix_columns (3)
        const std::uint32_t rows = std::clamp<std::uint32_t>(src_rows != 0 ? src_rows : e.rows, 1U, 4U);
        const std::uint32_t cols = std::clamp<std::uint32_t>(src_cols != 0 ? src_cols : e.cols, 1U, 4U);
        const bool col_major = (e.cls == 3U);
        const std::uint32_t regs_per_elem = col_major ? std::clamp<std::uint32_t>(e.cols, 1U, 4U)
                                                      : std::clamp<std::uint32_t>(e.rows, 1U, 4U);
        const std::size_t words_per_elem = static_cast<std::size_t>(rows) * cols;
        std::uint32_t reg_written = 0;
        for (std::uint32_t el = 0; el < elems && reg_written < e.reg_count; ++el) {
            const std::size_t base_w = static_cast<std::size_t>(el) * words_per_elem;
            for (std::uint32_t v = 0; v < regs_per_elem && reg_written < e.reg_count; ++v, ++reg_written) {
                const std::uint32_t ridx = static_cast<std::uint32_t>(e.reg_index) + reg_written;
                if (ridx >= 256U) return;
                float* out4 = &dst.c[static_cast<std::size_t>(ridx) * 4U];
                out4[0] = out4[1] = out4[2] = out4[3] = 0.0F;
                if (col_major) {
                    // Column `v` of row-major stored value V[r][v].
                    for (std::uint32_t r = 0; r < rows; ++r) {
                        out4[r] = word_to_float(read_word(base_w + static_cast<std::size_t>(r) * cols + v), val_ty);
                    }
                } else {
                    // Row `v` of row-major stored value V[v][c].
                    for (std::uint32_t c = 0; c < cols; ++c) {
                        out4[c] = word_to_float(read_word(base_w + static_cast<std::size_t>(v) * cols + c), val_ty);
                    }
                }
            }
        }
        return;
    }

    // Scalar (cls 0) or vector (cls 1), single or array.
    const std::uint32_t src_w = std::clamp<std::uint32_t>(
        (src_rows != 0 && src_cols != 0) ? (src_rows * src_cols) : std::max<std::uint32_t>(1U, e.cols), 1U, 4U);
    const std::uint32_t n = std::min<std::uint32_t>(elems, e.reg_count);
    for (std::uint32_t el = 0; el < n; ++el) {
        const std::uint32_t ridx = static_cast<std::uint32_t>(e.reg_index) + el;
        if (ridx >= 256U) return;
        float* out4 = &dst.c[static_cast<std::size_t>(ridx) * 4U];
        out4[0] = out4[1] = out4[2] = out4[3] = 0.0F;
        for (std::uint32_t c = 0; c < src_w; ++c) {
            out4[c] = word_to_float(read_word(static_cast<std::size_t>(el) * src_w + c), val_ty);
        }
    }
}

}  // namespace d3d9_fx_detail

// Runtime representation of a parsed `fx_2_0` effect: pre-parsed pass shaders (CTAB + PRES),
// sampler-to-texture links, and live parameter values. Used by both `D3d9CaptureReplayer` and
// `shim::D3d9` when a `D3d9Backend` is attached.
struct D3d9EffectInstance {
    struct PassShader {
        std::uint64_t shader_id = 0;
        d3d9sh::Ctab ctab;
        d3d9sh::Preshader preshader;
    };
    struct PassCompiled {
        PassShader vs;
        PassShader ps;
    };

    fx::Effect effect;
    std::vector<std::vector<std::uint8_t>> values;
    std::vector<std::uint64_t> textures;
    std::vector<fx::ResolvedSampler> samplers;  // indexed by parameter index
    std::vector<std::vector<PassCompiled>> compiled;  // [tech][pass]
    std::uint32_t current_tech = 0;
    std::uint32_t current_pass = 0;
    bool in_pass = false;

    void init(fx::Effect parsed, D3d9Backend& backend, std::uint64_t& next_shader_id) {
        effect = std::move(parsed);
        values.clear();
        textures.clear();
        samplers.clear();
        compiled.clear();
        values.reserve(effect.parameters.size());
        textures.assign(effect.parameters.size(), 0);
        samplers.resize(effect.parameters.size());
        for (std::uint32_t i = 0; i < effect.parameters.size(); ++i) {
            values.push_back(effect.parameters[i].value);
            if (effect.parameters[i].type.is_sampler()) {
                samplers[i] = fx::resolve_sampler(effect, i);
            }
        }
        // Deduplicate identical shader bytecodes within this effect by pointer/size.
        std::map<std::pair<const std::uint8_t*, std::size_t>, PassShader> cache;
        auto compile_one = [&](std::span<const std::uint8_t> code, bool is_pixel) -> PassShader {
            if (code.empty()) return {};
            const auto key = std::make_pair(code.data(), code.size());
            if (const auto it = cache.find(key); it != cache.end()) return it->second;
            PassShader ps;
            ps.shader_id = ++next_shader_id;
            (void)d3d9sh::parse_ctab(code, ps.ctab);
            (void)d3d9sh::parse_preshader(code, ps.preshader);
            backend.create_shader(ps.shader_id, is_pixel, code);
            cache.emplace(key, ps);
            return ps;
        };
        compiled.resize(effect.techniques.size());
        for (std::uint32_t t = 0; t < effect.techniques.size(); ++t) {
            compiled[t].resize(effect.techniques[t].passes.size());
            for (std::uint32_t p = 0; p < effect.techniques[t].passes.size(); ++p) {
                const auto vs_bytes = fx::resolve_pass_shader(effect, t, p, 146U);
                const auto ps_bytes = fx::resolve_pass_shader(effect, t, p, 147U);
                compiled[t][p].vs = compile_one(vs_bytes, false);
                compiled[t][p].ps = compile_one(ps_bytes, true);
            }
        }
    }

    [[nodiscard]] int find_param(std::string_view name) const noexcept {
        for (std::size_t i = 0; i < effect.parameters.size(); ++i) {
            if (effect.parameters[i].type.name == name) return static_cast<int>(i);
        }
        return -1;
    }

    [[nodiscard]] int find_tech(std::string_view name) const noexcept {
        for (std::size_t i = 0; i < effect.techniques.size(); ++i) {
            if (effect.techniques[i].name == name) return static_cast<int>(i);
        }
        return -1;
    }

    void set_param_bytes(int pi, const void* src, std::size_t n) {
        if (pi < 0 || static_cast<std::size_t>(pi) >= values.size() || src == nullptr) return;
        std::vector<std::uint8_t>& v = values[static_cast<std::size_t>(pi)];
        const std::size_t count = std::min(n, v.size());
        if (count != 0) std::memcpy(v.data(), src, count);
    }

    // Commits the active pass's VS/PS shaders, CTAB constants, PRES preshader outputs, and samplers
    // onto `draw_state`.
    void commit(D3d9DrawState& draw_state) const {
        if (!in_pass || current_tech >= compiled.size() || current_pass >= compiled[current_tech].size()) return;
        const PassCompiled& pass = compiled[current_tech][current_pass];
        if (pass.vs.shader_id != 0) draw_state.vs = pass.vs.shader_id;
        if (pass.ps.shader_id != 0) draw_state.ps = pass.ps.shader_id;

        auto bind_ctab = [&](const d3d9sh::Ctab& ctab, D3d9StageConstants& sc, bool bind_samplers) {
            for (const d3d9sh::CtabEntry& e : ctab.entries) {
                const int pi = find_param(e.name);
                if (e.reg_set == 3U) {
                    if (bind_samplers && pi >= 0 && e.reg_index < 16U) {
                        const fx::ResolvedSampler& rs = samplers[static_cast<std::size_t>(pi)];
                        if (rs.texture_param >= 0 && static_cast<std::size_t>(rs.texture_param) < textures.size()) {
                            draw_state.textures[e.reg_index] = textures[static_cast<std::size_t>(rs.texture_param)];
                        }
                        for (const auto& [stype, sval] : rs.sampler_states) {
                            draw_state.samplers[e.reg_index].set(stype, sval);
                        }
                    }
                    continue;
                }
                if (pi >= 0) {
                    const fx::TypeDef& td = effect.parameters[static_cast<std::size_t>(pi)].type;
                    d3d9_fx_detail::pack_ctab_entry(e, values[static_cast<std::size_t>(pi)], td.rows, td.columns, td.type, sc);
                } else if (!e.default_value.empty()) {
                    d3d9_fx_detail::pack_ctab_entry(e, e.default_value, e.rows, e.cols, e.type, sc);
                }
            }
        };

        auto apply_stage = [&](const PassShader& sh, D3d9StageConstants& sc) {
            if (sh.shader_id == 0) return;
            bind_ctab(sh.ctab, sc, true);
            if (!sh.preshader.empty()) {
                D3d9StageConstants pin;
                bind_ctab(sh.preshader.ctab, pin, false);
                d3d9sh::eval_preshader(sh.preshader, pin.c, sc.c, sc.i, sc.b);
            }
        };

        apply_stage(pass.vs, draw_state.vs_const);
        apply_stage(pass.ps, draw_state.ps_const);
    }
};

// Replays a `d3dcap::Call` recording (`wse/recomp/d3d9_capture.h`) onto any `D3d9Backend`.
// Used by `tools/r5_replay_vulkan.cpp` (comparing against `tools/r5_replay_d3d9.cpp` on the user's
// PC) and by host tests with synthetic captures.
class D3d9CaptureReplayer {
public:
    explicit D3d9CaptureReplayer(D3d9Backend& backend) : backend_(backend) {}

    // Callback to read install files (`.dds`, `.fxo`) referenced by path in the capture.
    std::function<bool(const std::string& path, std::vector<std::uint8_t>& out)> read_file;

    // Called at every Present (frame 1, 2, ...). Return true to read back the frame RGBA8 into
    // `on_frame` (skipped on frames that don't need readback, keeping replay fast).
    std::function<bool(std::uint64_t frame)> want_frame;
    std::function<void(std::uint64_t frame, std::uint32_t w, std::uint32_t h, const std::vector<std::uint8_t>& rgba8)> on_frame;

    [[nodiscard]] std::uint64_t frames() const noexcept { return frame_; }
    [[nodiscard]] std::size_t draws() const noexcept { return draws_; }
    [[nodiscard]] std::size_t calls() const noexcept { return calls_; }
    [[nodiscard]] const D3d9DrawState& state() const noexcept { return state_; }

    bool run_call(const d3dcap::Call& c, std::string* error_out = nullptr) {
        ++calls_;
        const std::string& m = c.method;

        auto u64_out = [&](int arg_idx) -> std::uint64_t {
            const d3dcap::Data* d = c.find(d3dcap::kTagOut + static_cast<std::uint32_t>(arg_idx));
            if (d == nullptr || d->bytes.size() < 8) return 0;
            std::uint64_t v = 0;
            std::memcpy(&v, d->bytes.data(), 8);
            return v;
        };
        auto str_arg = [&](int arg_idx) -> std::string {
            const d3dcap::Data* d = c.find(d3dcap::kTagArg + static_cast<std::uint32_t>(arg_idx));
            if (d == nullptr) return {};
            return std::string(d->bytes.begin(), d->bytes.end());
        };

        // Device creation / Reset
        if ((c.iface == "IDirect3D9" && m == "CreateDevice") || (c.iface == "IDirect3DDevice9" && m == "Reset")) {
            const d3dcap::Data* d = c.find(d3dcap::kTagArg + (m == "Reset" ? 0U : 4U));
            std::uint32_t w = 1920, h = 1080, fmt = 22, dfmt = 75;
            if (d != nullptr && d->bytes.size() >= 52) {
                std::memcpy(&w, d->bytes.data(), 4);
                std::memcpy(&h, d->bytes.data() + 4, 4);
                std::memcpy(&fmt, d->bytes.data() + 8, 4);
                std::memcpy(&dfmt, d->bytes.data() + 48, 4);
            }
            // A zero size (windowed: "the window's size") is the screen the recording device
            // reported (`screen.h`; replay tools take it as --screen, since a capture of a
            // widescreen run has 0 here).
            if (w == 0) w = static_cast<std::uint32_t>(wse::recomp::shim::screen_size().width);
            if (h == 0) h = static_cast<std::uint32_t>(wse::recomp::shim::screen_size().height);
            if (fmt == 0) fmt = 22;
            if (dfmt == 0) dfmt = 75;
            state_.reset_defaults();
            state_.viewport = {0, 0, w, h, 0.0F, 1.0F};
            state_.scissor = {0, 0, static_cast<std::int32_t>(w), static_cast<std::int32_t>(h)};
            bb_size_ = {w, h};
            if (m == "CreateDevice") {
                if (!backend_.init(w, h, fmt, dfmt, error_out)) return false;
            } else {
                if (!backend_.reset(w, h, fmt, dfmt, error_out)) return false;
            }
            return true;
        }

        // State block recording: record device state mutations without mutating live state.
        if (c.iface == "IDirect3DDevice9" && m == "BeginStateBlock") {
            recording_sb_ = true;
            open_sb_.clear();
            return true;
        }
        if (c.iface == "IDirect3DDevice9" && m == "EndStateBlock") {
            recording_sb_ = false;
            const std::uint64_t sb = u64_out(0);
            if (sb != 0) state_blocks_[sb] = std::move(open_sb_);
            open_sb_.clear();
            return true;
        }
        if (c.iface == "IDirect3DStateBlock9" && m == "Apply") {
            const auto it = state_blocks_.find(c.self);
            if (it != state_blocks_.end()) {
                for (const RecordedOp& op : it->second) apply_recorded_op(op);
            }
            return true;
        }
        if (c.iface == "IDirect3DStateBlock9" && m == "Release") {
            // Keep state block if still referenced; D3D9 games rarely reuse captured addresses.
            return true;
        }

        // D3DX texture / effect creation
        if (c.iface == "d3dx9_43.dll") {
            if (m == "D3DXCreateTextureFromFileExA" || m == "D3DXCreateCubeTextureFromFileExA") {
                const bool is_cube = (m == "D3DXCreateCubeTextureFromFileExA");
                const std::uint64_t id = u64_out(is_cube ? 12 : 13);
                const std::string path = str_arg(1);
                if (id != 0 && read_file) {
                    std::vector<std::uint8_t> bytes;
                    if (read_file(path, bytes) && !bytes.empty()) {
                        backend_.create_texture(id, is_cube ? 1 : 0, 1, 1, 1, 1, 0, 0);
                        backend_.upload_texture_dds(id, bytes);
                    }
                }
                return true;
            }
            if (m == "D3DXCreateTextureFromFileInMemoryEx") {
                const std::uint64_t id = u64_out(14);
                const d3dcap::Data* d = c.find(d3dcap::kTagArg + 1);
                if (id != 0 && d != nullptr && !d->bytes.empty()) {
                    backend_.create_texture(id, 0, 1, 1, 1, 1, 0, 0);
                    backend_.upload_texture_dds(id, d->bytes);
                }
                return true;
            }
            if (m == "D3DXCreateTexture") {
                const std::uint64_t id = u64_out(7);
                if (id != 0) {
                    const auto w = std::max<std::uint32_t>(1U, static_cast<std::uint32_t>(c.args[1]));
                    const auto h = std::max<std::uint32_t>(1U, static_cast<std::uint32_t>(c.args[2]));
                    const auto lv = std::max<std::uint32_t>(1U, static_cast<std::uint32_t>(c.args[3]));
                    const auto usg = static_cast<std::uint32_t>(c.args[4]);
                    const auto fmt = static_cast<std::uint32_t>(c.args[5]);
                    tex_meta_[id] = {w, h, lv, fmt};
                    backend_.create_texture(id, 0, w, h, 1, lv, fmt == 0 || fmt == 0xFFFFFFFFU ? 21U : fmt, usg);
                }
                return true;
            }
            if (m == "D3DXCreateEffectFromFileA") {
                const std::uint64_t id = u64_out(6);
                const std::string path = str_arg(1);
                if (id != 0 && read_file) {
                    std::vector<std::uint8_t> bytes;
                    if (read_file(path, bytes) && bytes.size() >= 4) {
                        std::uint32_t tag = 0;
                        std::memcpy(&tag, bytes.data(), 4);
                        if (tag == fx::kFx20Tag) {
                            fx::Effect parsed;
                            if (fx::parse(bytes, parsed)) {
                                auto inst = std::make_unique<D3d9EffectInstance>();
                                inst->init(std::move(parsed), backend_, next_internal_shader_);
                                effects_[id] = std::move(inst);
                            }
                        }
                    }
                }
                return true;
            }
            return true;
        }

        // ID3DXEffect methods
        if (c.iface == "ID3DXEffect") {
            const auto eit = effects_.find(c.self);
            if (eit == effects_.end()) return true;
            D3d9EffectInstance& fx = *eit->second;

            auto resolve_p = [&]() -> int {
                if (const d3dcap::Data* dn = c.find(d3dcap::kTagName); dn != nullptr) {
                    return fx.find_param(std::string_view(reinterpret_cast<const char*>(dn->bytes.data()), dn->bytes.size()));
                }
                const auto it = effect_param_handles_.find({c.self, c.args[0]});
                return it != effect_param_handles_.end() ? it->second : -1;
            };
            auto resolve_t = [&]() -> int {
                if (const d3dcap::Data* dn = c.find(d3dcap::kTagName); dn != nullptr) {
                    return fx.find_tech(std::string_view(reinterpret_cast<const char*>(dn->bytes.data()), dn->bytes.size()));
                }
                const auto it = effect_tech_handles_.find({c.self, c.args[0]});
                return it != effect_tech_handles_.end() ? it->second : -1;
            };

            if (m == "GetParameterByName") {
                const std::string name = str_arg(1);
                const int idx = fx.find_param(name);
                if (idx >= 0 && c.ret != 0) effect_param_handles_[{c.self, c.ret}] = idx;
            } else if (m == "GetParameterBySemantic") {
                const std::string want = str_arg(1);
                for (std::size_t i = 0; i < fx.effect.parameters.size(); ++i) {
                    const std::string& have = fx.effect.parameters[i].type.semantic;
                    if (have.size() == want.size() && std::equal(have.begin(), have.end(), want.begin(),
                                                                 [](char a, char b) { return (a | 0x20) == (b | 0x20); })) {
                        if (c.ret != 0) effect_param_handles_[{c.self, c.ret}] = static_cast<int>(i);
                        break;
                    }
                }
            } else if (m == "GetParameter") {
                const auto idx = static_cast<std::uint32_t>(c.args[1]);
                if (c.args[0] == 0 && idx < fx.effect.parameters.size() && c.ret != 0) {
                    effect_param_handles_[{c.self, c.ret}] = static_cast<int>(idx);
                }
            } else if (m == "GetTechniqueByName") {
                const std::string name = str_arg(0);
                const int idx = fx.find_tech(name);
                if (idx >= 0 && c.ret != 0) effect_tech_handles_[{c.self, c.ret}] = idx;
            } else if (m == "GetTechnique") {
                const auto idx = static_cast<std::uint32_t>(c.args[0]);
                if (idx < fx.effect.techniques.size() && c.ret != 0) {
                    effect_tech_handles_[{c.self, c.ret}] = static_cast<int>(idx);
                }
            } else if (m == "SetTechnique") {
                const int t = resolve_t();
                if (t >= 0) fx.current_tech = static_cast<std::uint32_t>(t);
            } else if (m == "Begin") {
                fx.in_pass = false;
            } else if (m == "BeginPass") {
                fx.current_pass = static_cast<std::uint32_t>(c.args[0]);
                fx.in_pass = true;
                fx.commit(state_);
            } else if (m == "CommitChanges") {
                fx.commit(state_);
            } else if (m == "EndPass") {
                fx.in_pass = false;
            } else if (m == "End") {
                fx.in_pass = false;
            } else if (m == "SetBool" || m == "SetInt") {
                const auto v = static_cast<std::uint32_t>(c.args[1]);
                fx.set_param_bytes(resolve_p(), &v, 4);
            } else if (m == "SetFloat") {
                fx.set_param_bytes(resolve_p(), &c.xmm[1], 4);
            } else if (m == "SetVector" || m == "SetMatrix" || m == "SetValue" ||
                       m == "SetFloatArray" || m == "SetIntArray" || m == "SetBoolArray" ||
                       m == "SetVectorArray" || m == "SetMatrixArray") {
                if (const d3dcap::Data* d = c.find(d3dcap::kTagArg + 1); d != nullptr) {
                    fx.set_param_bytes(resolve_p(), d->bytes.data(), d->bytes.size());
                }
            } else if (m == "SetMatrixTranspose") {
                if (const d3dcap::Data* d = c.find(d3dcap::kTagArg + 1); d != nullptr && d->bytes.size() >= 64) {
                    float src[16], tr[16];
                    std::memcpy(src, d->bytes.data(), 64);
                    for (int r = 0; r < 4; ++r) {
                        for (int col = 0; col < 4; ++col) tr[r * 4 + col] = src[col * 4 + r];
                    }
                    fx.set_param_bytes(resolve_p(), tr, 64);
                }
            } else if (m == "SetMatrixTransposeArray") {
                if (const d3dcap::Data* d = c.find(d3dcap::kTagArg + 1); d != nullptr) {
                    const std::size_t count = d->bytes.size() / 64;
                    std::vector<float> tr(count * 16);
                    for (std::size_t k = 0; k < count; ++k) {
                        float src[16];
                        std::memcpy(src, d->bytes.data() + k * 64, 64);
                        for (int r = 0; r < 4; ++r) {
                            for (int col = 0; col < 4; ++col) tr[k * 16 + r * 4 + col] = src[col * 4 + r];
                        }
                    }
                    fx.set_param_bytes(resolve_p(), tr.data(), tr.size() * 4);
                }
            } else if (m == "SetTexture") {
                const int pi = resolve_p();
                if (pi >= 0 && static_cast<std::size_t>(pi) < fx.textures.size()) {
                    fx.textures[static_cast<std::size_t>(pi)] = c.args[1];
                }
            }
            return true;
        }

        // Buffers (Lock/Unlock with delta shadow)
        if (c.iface == "IDirect3DVertexBuffer9" || c.iface == "IDirect3DIndexBuffer9") {
            if (m == "Lock") {
                buf_lock_flags_[c.self] = static_cast<std::uint32_t>(c.args[3]);
                buf_lock_range_[c.self] = {static_cast<std::uint32_t>(c.args[0]), static_cast<std::uint32_t>(c.args[1])};
            } else if (m == "Unlock") {
                const d3dcap::Data* d = c.find(d3dcap::kTagLocked);
                if (d != nullptr && d->bytes.size() >= 8) {
                    std::uint32_t off = 0, n = 0;
                    std::memcpy(&off, d->bytes.data(), 4);
                    std::memcpy(&n, d->bytes.data() + 4, 4);
                    std::vector<std::uint8_t>& full = buf_shadow_[c.self];
                    const std::size_t avail = std::min<std::size_t>(n, d->bytes.size() - 8);
                    if (full.size() < static_cast<std::size_t>(off) + avail) {
                        full.resize(static_cast<std::size_t>(off) + avail, 0);
                    }
                    if (avail != 0) std::memcpy(full.data() + off, d->bytes.data() + 8, avail);
                    const std::uint32_t flags = buf_lock_flags_[c.self];
                    if ((flags & 0x2000U) != 0U) {  // D3DLOCK_DISCARD: the whole locked range is new
                        const auto [lo, size] = buf_lock_range_[c.self];
                        const std::size_t from = std::min<std::size_t>(lo, full.size());
                        const std::size_t n = size == 0 ? full.size() - from : std::min<std::size_t>(size, full.size() - from);
                        if (n != 0) backend_.update_buffer(c.self, static_cast<std::uint32_t>(from), std::span<const std::uint8_t>(full.data() + from, n), flags);
                    } else if (avail != 0) {
                        backend_.update_buffer(c.self, off, std::span<const std::uint8_t>(full.data() + off, avail), flags);
                    }
                }
            }
            return true;
        }

        // Textures / surfaces (UnlockRect)
        if (m == "UnlockRect") {
            const d3dcap::Data* d = c.find(d3dcap::kTagLocked);
            if (d != nullptr && d->bytes.size() > 4) {
                std::uint32_t index = 0;
                std::memcpy(&index, d->bytes.data(), 4);
                const auto it = tex_meta_.find(c.self);
                if (it != tex_meta_.end()) {
                    const TexMeta& tm = it->second;
                    const std::uint32_t level = (c.iface == "IDirect3DCubeTexture9")
                                                    ? (index % std::max<std::uint32_t>(1U, tm.levels))
                                                    : index;
                    const std::uint32_t face = (c.iface == "IDirect3DCubeTexture9")
                                                   ? (index / std::max<std::uint32_t>(1U, tm.levels))
                                                   : 0U;
                    const std::uint32_t lw = std::max<std::uint32_t>(1U, tm.width >> level);
                    const std::uint32_t rp = tight_pitch(tm.format, lw);
                    backend_.upload_texture_level(c.self, face, level,
                                                  std::span<const std::uint8_t>(d->bytes.data() + 4, d->bytes.size() - 4),
                                                  rp);
                }
            }
            return true;
        }

        // IDirect3DDevice9 calls
        if (c.iface == "IDirect3DDevice9") {
            if (m == "CreateVertexBuffer" || m == "CreateIndexBuffer") {
                const std::uint64_t id = u64_out(4);
                if (id != 0) {
                    const auto sz = static_cast<std::uint32_t>(c.args[0]);
                    const auto usage = static_cast<std::uint32_t>(c.args[1]);
                    const auto fmt = static_cast<std::uint32_t>(c.args[2]);
                    buf_shadow_[id].assign(sz, 0);
                    backend_.create_buffer(id, m == "CreateIndexBuffer", sz, (usage & 0x200U) != 0U, fmt);
                }
            } else if (m == "CreateTexture") {
                const std::uint64_t id = u64_out(6);
                if (id != 0) {
                    const auto w = std::max<std::uint32_t>(1U, static_cast<std::uint32_t>(c.args[0]));
                    const auto h = std::max<std::uint32_t>(1U, static_cast<std::uint32_t>(c.args[1]));
                    const auto lv = std::max<std::uint32_t>(1U, static_cast<std::uint32_t>(c.args[2]));
                    const auto usg = static_cast<std::uint32_t>(c.args[3]);
                    const auto fmt = static_cast<std::uint32_t>(c.args[4]);
                    tex_meta_[id] = {w, h, lv, fmt, usg};
                    backend_.create_texture(id, 0, w, h, 1, lv, fmt, usg);
                }
            } else if (m == "CreateCubeTexture") {
                const std::uint64_t id = u64_out(5);
                if (id != 0) {
                    const auto e = std::max<std::uint32_t>(1U, static_cast<std::uint32_t>(c.args[0]));
                    const auto lv = std::max<std::uint32_t>(1U, static_cast<std::uint32_t>(c.args[1]));
                    const auto usg = static_cast<std::uint32_t>(c.args[2]);
                    const auto fmt = static_cast<std::uint32_t>(c.args[3]);
                    tex_meta_[id] = {e, e, lv, fmt, usg};
                    backend_.create_texture(id, 1, e, e, 1, lv, fmt, usg);
                }
            } else if (m == "CreateRenderTarget" || m == "CreateDepthStencilSurface") {
                const std::uint64_t id = u64_out(6);
                if (id != 0) {
                    surface_size_[id] = {static_cast<std::uint32_t>(c.args[0]), static_cast<std::uint32_t>(c.args[1])};
                    backend_.create_surface(id, static_cast<std::uint32_t>(c.args[0]),
                                            static_cast<std::uint32_t>(c.args[1]),
                                            static_cast<std::uint32_t>(c.args[2]),
                                            m == "CreateRenderTarget" ? 1U : 2U, 0, 0, 0);
                }
            } else if (m == "GetSurfaceLevel" || m == "GetCubeMapSurface") {
                // A texture level (or cube face) as a surface the game may draw into.
                const bool cube = m == "GetCubeMapSurface";
                const std::uint64_t id = u64_out(cube ? 2 : 1);
                const auto tm = tex_meta_.find(c.self);
                if (id != 0 && tm != tex_meta_.end()) {
                    const auto face = cube ? static_cast<std::uint32_t>(c.args[0]) : 0U;
                    const auto level = static_cast<std::uint32_t>(c.args[cube ? 1 : 0]);
                    const std::uint32_t w = std::max<std::uint32_t>(1U, tm->second.width >> level);
                    const std::uint32_t hh = std::max<std::uint32_t>(1U, tm->second.height >> level);
                    surface_size_[id] = {w, hh};
                    backend_.create_surface(id, w, hh, tm->second.format, tm->second.usage, c.self, face, level);
                }
            } else if (m == "CreateVertexDeclaration") {
                const std::uint64_t id = u64_out(1);
                const d3dcap::Data* d = c.find(d3dcap::kTagArg);
                if (id != 0 && d != nullptr) {
                    std::vector<D3d9VertexElement> elems;
                    for (std::size_t off = 0; off + 8 <= d->bytes.size(); off += 8) {
                        D3d9VertexElement el{};
                        std::memcpy(&el.stream, d->bytes.data() + off, 2);
                        std::memcpy(&el.offset, d->bytes.data() + off + 2, 2);
                        el.type = d->bytes[off + 4];
                        el.method = d->bytes[off + 5];
                        el.usage = d->bytes[off + 6];
                        el.usage_index = d->bytes[off + 7];
                        if (el.stream == 0xFF) break;
                        elems.push_back(el);
                    }
                    backend_.create_vertex_decl(id, elems);
                }
            } else if (m == "CreateVertexShader" || m == "CreatePixelShader") {
                const std::uint64_t id = u64_out(1);
                const d3dcap::Data* d = c.find(d3dcap::kTagArg);
                if (id != 0 && d != nullptr) {
                    backend_.create_shader(id, m == "CreatePixelShader", d->bytes);
                }
            } else if (m == "SetRenderState") {
                emit_or_apply(RecordedOp{OpKind::render_state, static_cast<std::uint32_t>(c.args[0]),
                                         static_cast<std::uint32_t>(c.args[1]), 0, 0});
            } else if (m == "SetSamplerState") {
                emit_or_apply(RecordedOp{OpKind::sampler_state, static_cast<std::uint32_t>(c.args[0]),
                                         static_cast<std::uint32_t>(c.args[1]), static_cast<std::uint32_t>(c.args[2]), 0});
            } else if (m == "SetTexture") {
                emit_or_apply(RecordedOp{OpKind::texture, static_cast<std::uint32_t>(c.args[0]), 0, 0, c.args[1]});
            } else if (m == "SetVertexDeclaration") {
                emit_or_apply(RecordedOp{OpKind::decl, 0, 0, 0, c.args[0]});
            } else if (m == "SetVertexShader") {
                emit_or_apply(RecordedOp{OpKind::vs, 0, 0, 0, c.args[0]});
            } else if (m == "SetPixelShader") {
                emit_or_apply(RecordedOp{OpKind::ps, 0, 0, 0, c.args[0]});
            } else if (m == "SetIndices") {
                emit_or_apply(RecordedOp{OpKind::indices, 0, 0, 0, c.args[0]});
            } else if (m == "SetStreamSource") {
                emit_or_apply(RecordedOp{OpKind::stream, static_cast<std::uint32_t>(c.args[0]),
                                         static_cast<std::uint32_t>(c.args[2]), static_cast<std::uint32_t>(c.args[3]),
                                         c.args[1]});
            } else if (m == "SetStreamSourceFreq") {
                emit_or_apply(RecordedOp{OpKind::stream_freq, static_cast<std::uint32_t>(c.args[0]),
                                         static_cast<std::uint32_t>(c.args[1]), 0, 0});
            } else if (m == "SetViewport") {
                if (const d3dcap::Data* d = c.find(d3dcap::kTagArg); d != nullptr && d->bytes.size() >= 24) {
                    D3d9Viewport vp{};
                    std::memcpy(&vp.x, d->bytes.data(), 4);
                    std::memcpy(&vp.y, d->bytes.data() + 4, 4);
                    std::memcpy(&vp.width, d->bytes.data() + 8, 4);
                    std::memcpy(&vp.height, d->bytes.data() + 12, 4);
                    std::memcpy(&vp.min_z, d->bytes.data() + 16, 4);
                    std::memcpy(&vp.max_z, d->bytes.data() + 20, 4);
                    state_.viewport = vp;
                }
            } else if (m == "SetScissorRect") {
                if (const d3dcap::Data* d = c.find(d3dcap::kTagArg); d != nullptr && d->bytes.size() >= 16) {
                    D3d9Rect r{};
                    std::memcpy(&r, d->bytes.data(), 16);
                    state_.scissor = r;
                }
            } else if (m == "SetRenderTarget") {
                if (c.args[0] == 0) {
                    state_.render_target = (c.args[1] == backbuffer_id_) ? 0 : c.args[1];
                    // D3D9 resets the viewport (and the scissor rectangle) to the new target.
                    auto size = bb_size_;
                    if (const auto it = surface_size_.find(state_.render_target); state_.render_target != 0 && it != surface_size_.end()) {
                        size = it->second;
                    }
                    state_.viewport = {0, 0, size.first, size.second, 0.0F, 1.0F};
                    state_.scissor = {0, 0, static_cast<std::int32_t>(size.first), static_cast<std::int32_t>(size.second)};
                }
            } else if (m == "SetDepthStencilSurface") {
                state_.depth_stencil = c.args[0] == 0 ? kNoDepthStencil : (c.args[0] == depth_id_) ? 0 : c.args[0];
            } else if (m == "GetBackBuffer") {
                const std::uint64_t id = u64_out(3);
                if (id != 0) backbuffer_id_ = id;
            } else if (m == "GetDepthStencilSurface") {
                const std::uint64_t id = u64_out(0);
                if (id != 0) depth_id_ = id;
            } else if (m == "SetVertexShaderConstantF" || m == "SetPixelShaderConstantF") {
                if (const d3dcap::Data* d = c.find(d3dcap::kTagArg + 1); d != nullptr) {
                    auto& dst = (m == "SetVertexShaderConstantF") ? state_.vs_const.c : state_.ps_const.c;
                    const std::size_t start = static_cast<std::size_t>(c.args[0]) * 4U;
                    const std::size_t nflt = std::min<std::size_t>(d->bytes.size() / 4U,
                                                                   start < dst.size() ? dst.size() - start : 0U);
                    if (nflt != 0) std::memcpy(dst.data() + start, d->bytes.data(), nflt * 4U);
                }
            } else if (m == "SetVertexShaderConstantI" || m == "SetPixelShaderConstantI") {
                if (const d3dcap::Data* d = c.find(d3dcap::kTagArg + 1); d != nullptr) {
                    auto& dst = (m == "SetVertexShaderConstantI") ? state_.vs_const.i : state_.ps_const.i;
                    const std::size_t start = static_cast<std::size_t>(c.args[0]) * 4U;
                    const std::size_t nint = std::min<std::size_t>(d->bytes.size() / 4U,
                                                                   start < dst.size() ? dst.size() - start : 0U);
                    if (nint != 0) std::memcpy(dst.data() + start, d->bytes.data(), nint * 4U);
                }
            } else if (m == "SetVertexShaderConstantB" || m == "SetPixelShaderConstantB") {
                if (const d3dcap::Data* d = c.find(d3dcap::kTagArg + 1); d != nullptr) {
                    auto& dst = (m == "SetVertexShaderConstantB") ? state_.vs_const.b : state_.ps_const.b;
                    const std::size_t start = static_cast<std::size_t>(c.args[0]);
                    const std::size_t nbool = d->bytes.size() / 4U;
                    for (std::size_t k = 0; k < nbool && start + k < 16U; ++k) {
                        std::uint32_t v = 0;
                        std::memcpy(&v, d->bytes.data() + k * 4U, 4);
                        dst[(start + k) * 4U] = v != 0U ? 1 : 0;
                    }
                }
            } else if (m == "Clear") {
                const auto flags = static_cast<std::uint32_t>(c.args[2]);
                const auto argb = static_cast<std::uint32_t>(c.args[3]);
                float z = 1.0F;
                // D3D9 Clear(Count, pRects, Flags, Color, float Z, DWORD Stencil): Z is arg 4 (float in stack slot on Win64!).
                const auto zbits = static_cast<std::uint32_t>(c.args[4]);
                std::memcpy(&z, &zbits, 4);
                const auto stencil = static_cast<std::uint32_t>(c.args[5]);
                std::vector<D3d9Rect> rects;
                if (const d3dcap::Data* d = c.find(d3dcap::kTagArg + 1); d != nullptr && c.args[0] > 0) {
                    const std::size_t cnt = std::min<std::size_t>(static_cast<std::size_t>(c.args[0]), d->bytes.size() / 16U);
                    rects.resize(cnt);
                    if (cnt != 0) std::memcpy(rects.data(), d->bytes.data(), cnt * 16U);
                }
                backend_.clear(state_, flags, argb, z, stencil, rects);
            } else if (m == "DrawIndexedPrimitive") {
                ++draws_;
                backend_.draw_indexed(state_, static_cast<std::uint32_t>(c.args[0]),
                                      static_cast<std::int32_t>(c.args[1]),
                                      static_cast<std::uint32_t>(c.args[2]),
                                      static_cast<std::uint32_t>(c.args[3]),
                                      static_cast<std::uint32_t>(c.args[4]),
                                      static_cast<std::uint32_t>(c.args[5]));
            } else if (m == "DrawPrimitive") {
                ++draws_;
                backend_.draw_primitive(state_, static_cast<std::uint32_t>(c.args[0]),
                                        static_cast<std::uint32_t>(c.args[1]),
                                        static_cast<std::uint32_t>(c.args[2]));
            } else if (m == "Present") {
                ++frame_;
                const bool readback = want_frame && want_frame(frame_);
                std::vector<std::uint8_t> rgba;
                std::uint32_t w = 0, h = 0;
                if (!backend_.present(readback ? &rgba : nullptr, &w, &h)) {
                    if (error_out != nullptr) *error_out = "D3d9Backend::present failed";
                    return false;
                }
                if (readback && on_frame && !rgba.empty()) on_frame(frame_, w, h, rgba);
            }
            return true;
        }

        if (c.iface == "IDirect3DSwapChain9" && m == "Present") {
            ++frame_;
            const bool readback = want_frame && want_frame(frame_);
            std::vector<std::uint8_t> rgba;
            std::uint32_t w = 0, h = 0;
            if (!backend_.present(readback ? &rgba : nullptr, &w, &h)) {
                if (error_out != nullptr) *error_out = "D3d9Backend::present failed";
                return false;
            }
            if (readback && on_frame && !rgba.empty()) on_frame(frame_, w, h, rgba);
            return true;
        }

        return true;
    }

private:
    enum class OpKind { render_state, sampler_state, texture, decl, vs, ps, indices, stream, stream_freq };
    struct RecordedOp {
        OpKind kind = OpKind::render_state;
        std::uint32_t a = 0;
        std::uint32_t b = 0;
        std::uint32_t c = 0;
        std::uint64_t obj = 0;
    };
    struct TexMeta {
        std::uint32_t width = 1, height = 1, levels = 1, format = 21, usage = 0;
    };
    std::map<std::uint64_t, std::pair<std::uint32_t, std::uint32_t>> surface_size_;  // surface -> width, height
    std::pair<std::uint32_t, std::uint32_t> bb_size_{1920U, 1080U};

    static bool is_dxt(std::uint32_t f) noexcept {
        return f >= 0x31545844U && f <= 0x35545844U && (f & 0xFFFFFFU) == 0x545844U;
    }
    static std::uint32_t bpp(std::uint32_t f) noexcept {
        switch (f) {
        case 28: case 50: case 41: return 1;
        case 23: case 24: case 25: case 26: case 51: case 80: case 111: return 2;
        case 113: case 115: return 8;
        case 116: return 16;
        default: return 4;
        }
    }
    static std::uint32_t tight_pitch(std::uint32_t f, std::uint32_t w) noexcept {
        if (is_dxt(f)) return std::max<std::uint32_t>(1U, (w + 3U) / 4U) * (f == 0x31545844U ? 8U : 16U);
        return std::max<std::uint32_t>(1U, w) * bpp(f);
    }

    void emit_or_apply(const RecordedOp& op) {
        if (recording_sb_) {
            open_sb_.push_back(op);
        } else {
            apply_recorded_op(op);
        }
    }

    void apply_recorded_op(const RecordedOp& op) {
        switch (op.kind) {
        case OpKind::render_state:
            if (op.a < state_.rs.size()) state_.rs[op.a] = op.b;
            break;
        case OpKind::sampler_state:
            if (op.a < state_.samplers.size()) state_.samplers[op.a].set(op.b, op.c);
            break;
        case OpKind::texture:
            if (op.a < state_.textures.size()) state_.textures[op.a] = op.obj;
            break;
        case OpKind::decl: state_.decl = op.obj; break;
        case OpKind::vs: state_.vs = op.obj; break;
        case OpKind::ps: state_.ps = op.obj; break;
        case OpKind::indices: state_.indices = op.obj; break;
        case OpKind::stream:
            if (op.a < state_.streams.size()) {
                state_.streams[op.a].buffer = op.obj;
                state_.streams[op.a].offset = op.b;
                state_.streams[op.a].stride = op.c;
            }
            break;
        case OpKind::stream_freq:
            if (op.a < state_.streams.size()) state_.streams[op.a].freq = op.b;
            break;
        }
    }

    D3d9Backend& backend_;
    D3d9DrawState state_;
    std::uint64_t frame_ = 0;
    std::size_t draws_ = 0;
    std::size_t calls_ = 0;
    std::uint64_t next_internal_shader_ = 0xF000000000000000ULL;
    std::uint64_t backbuffer_id_ = 0;
    std::uint64_t depth_id_ = 0;
    bool recording_sb_ = false;
    std::vector<RecordedOp> open_sb_;
    std::map<std::uint64_t, std::vector<RecordedOp>> state_blocks_;
    std::map<std::uint64_t, std::vector<std::uint8_t>> buf_shadow_;
    std::map<std::uint64_t, std::uint32_t> buf_lock_flags_;
    std::map<std::uint64_t, std::pair<std::uint32_t, std::uint32_t>> buf_lock_range_;  // offset, size
    std::map<std::uint64_t, TexMeta> tex_meta_;
    std::map<std::uint64_t, std::unique_ptr<D3d9EffectInstance>> effects_;
    std::map<std::pair<std::uint64_t, std::uint64_t>, int> effect_param_handles_;
    std::map<std::pair<std::uint64_t, std::uint64_t>, int> effect_tech_handles_;
};

}  // namespace wse::recomp
