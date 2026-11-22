// D3D9 shader bytecode (vs/ps 2.0, 2.x, 3.0) to Vulkan GLSL 450, for Phase 5 R5a step 2
// (`docs/RECOMP_PLAN.md` "R5 — Graphics"). The R5 device compiles the text to SPIR-V with the
// app's glslang (`wse/asset/shader_compiler.h`) and caches it. Header-only.
//
// Scope: the 35 instructions the user's effects use (local census, 2026-11-22) plus their close
// relatives: mov add sub mul mad dp3 dp4 dp2add rcp rsq exp log pow min max slt sge frc lrp cmp
// nrm sincos mova abs crs sgn dsx dsy tex/texldp/texldb texldl texldd texkill if ifc else endif
// loop endloop rep endrep break breakc dcl def defi defb nop. Anything else (call, label, setp,
// predication, shader model 1) fails with its name.
//
// The interface the R5 device builds against:
//   constants   one std140 block per stage: `vec4 c[256]; ivec4 i[16]; ivec4 b[16]; vec4 posfix;`
//               vertex shader set 0 binding 0, pixel shader set 0 binding 1. `def`/`defi`/`defb`
//               values are compiled in and win over the block for direct reads (relative reads
//               of a `def` register read the block; none of the game's shaders do that).
//   samplers    set 1, binding = sampler register (sampler2D / samplerCube / sampler3D by dcl).
//   vertex in   location = input register; `Shader::inputs` lists each register's usage/index
//               so the device maps the vertex declaration onto it.
//   varyings    location = link_location(usage, index), the same in both stages, so a vertex
//               and a pixel shader link by semantic as in D3D9.
//   pixel out   oC0..3 at locations 0..3, oDepth to gl_FragDepth.
//   position    the vertex shader flips y (D3D9 clip space has y up, Vulkan y down) and adds
//               `posfix.xy * w` (the device sets it for D3D9's half-pixel offset); depth 0..1 is
//               already Vulkan's. vPos is gl_FragCoord.xy - 0.5 (D3D9 pixel centres are integer).
//
// D3D9 rules kept: rcp/rsq of 0 give infinity, log of 0 gives -infinity, pow(x, y) is
// 2^(y * log2|x|), mova rounds to nearest, cmp selects on >= 0, texkill tests the masked
// components, saturate clamps to [0, 1]. Precision hints (_pp, centroid) are ignored.

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <set>
#include <span>
#include <string>
#include <vector>

namespace wse::recomp::d3d9sh {

// D3DDECLUSAGE values.
enum Usage : std::uint32_t {
    kPosition = 0, kBlendWeight = 1, kBlendIndices = 2, kNormal = 3, kPSize = 4, kTexCoord = 5, kTangent = 6,
    kBinormal = 7, kTessFactor = 8, kPositionT = 9, kColor = 10, kFog = 11, kDepth = 12, kSample = 13,
};

struct Semantic {
    std::uint32_t reg = 0;
    std::uint32_t usage = 0;
    std::uint32_t index = 0;
};

struct Shader {
    bool pixel = false;
    int major = 0;
    int minor = 0;
    std::vector<Semantic> inputs;   // vertex shader inputs (by register) / pixel shader varyings
    std::vector<Semantic> outputs;  // vertex shader varyings
    std::array<int, 16> samplers{};  // 0 none, 2 2D, 3 cube, 4 volume
    std::string glsl;
    std::string error;  // why translation failed
};

// Where a varying lives, the same in both stages (D3D9 links by usage and index).
[[nodiscard]] inline int link_location(std::uint32_t usage, std::uint32_t index) noexcept {
    switch (usage) {
    case kTexCoord: return index < 10 ? static_cast<int>(index) : -1;
    case kColor: return index < 2 ? 10 + static_cast<int>(index) : -1;
    case kFog: return 12;
    case kNormal: return index < 2 ? 13 + static_cast<int>(index) : -1;
    case kTangent: return 15;
    case kBinormal: return 16;
    case kPosition: return index >= 1 && index < 3 ? 16 + static_cast<int>(index) : -1;  // 17, 18
    case kBlendWeight: return 19;
    case kBlendIndices: return 20;
    case kDepth: return 21;
    case kPSize: return 22;
    default: return -1;
    }
}

namespace detail {

// Register types (bits 28..30 and 11..12 of a parameter token).
enum RegType : std::uint32_t {
    kTemp = 0, kInput = 1, kConst = 2, kAddrOrTexture = 3, kRastOut = 4, kAttrOut = 5, kOutput = 6, kConstInt = 7,
    kColorOut = 8, kDepthOut = 9, kSampler = 10, kConst2 = 11, kConst3 = 12, kConst4 = 13, kConstBool = 14,
    kLoop = 15, kTempFloat16 = 16, kMisc = 17, kLabel = 18, kPredicate = 19,
};

struct Param {
    std::uint32_t token = 0;
    std::uint32_t type = 0;
    std::uint32_t num = 0;
    bool relative = false;
    std::uint32_t rel_type = kAddrOrTexture;  // a0 (vs) when the token does not say
    std::uint32_t rel_num = 0;
    std::uint32_t rel_comp = 0;
};

inline Param param(std::uint32_t t) {
    Param p;
    p.token = t;
    p.type = ((t >> 28) & 7U) | ((t >> 8) & 0x18U);
    p.num = t & 0x7FFU;
    p.relative = (t & (1U << 13)) != 0;
    return p;
}

inline const char* kComp = "xyzw";

struct Op {
    std::uint32_t opcode = 0;
    std::uint32_t controls = 0;
    std::vector<std::uint32_t> words;  // the instruction's tokens after the opcode token
};

inline bool has_dst(std::uint32_t op) {
    switch (op) {
    case 0: case 25: case 26: case 27: case 28: case 29: case 30: case 38: case 39: case 40: case 41: case 42: case 43: case 44: case 45:
        return false;  // nop call callnz loop ret endloop label rep endrep if ifc else endif break breakc
    default:
        return true;
    }
}

class Translator {
public:
    explicit Translator(Shader& out) : s_(out) {}

    bool run(std::span<const std::uint32_t> words) {
        if (words.empty()) return fail("empty shader");
        const std::uint32_t v = words[0];
        if ((v & 0xFFFE0000U) != 0xFFFE0000U) return fail("no version token");
        s_.pixel = (v >> 16) == 0xFFFFU;
        s_.major = static_cast<int>((v >> 8) & 0xFF);
        s_.minor = static_cast<int>(v & 0xFF);
        if (s_.major < 2 || s_.major > 3) return fail("only shader model 2 and 3");
        if (!decode(words)) return false;
        if (!declare()) return false;
        return emit();
    }

private:
    Shader& s_;
    std::vector<Op> ops_;
    std::string body_;
    int indent_ = 1;
    int loop_depth_ = 0;
    std::vector<int> block_kinds_;  // 0 if/else, 1 rep, 2 loop
    std::set<std::uint32_t> temps_, consts_rel_;
    std::array<std::string, 256> defs_{};       // def'd float constants (GLSL vec4 literal)
    std::array<std::string, 16> defi_{}, defb_{};
    std::set<std::uint32_t> written_out_;  // output register keys written
    std::vector<std::pair<std::uint32_t, Semantic>> out_decl_;  // vs3 o# / ps inputs: by register

    bool fail(std::string why) {
        s_.error = std::move(why);
        return false;
    }

    // --- decoding ------------------------------------------------------------------------
    bool decode(std::span<const std::uint32_t> w) {
        std::size_t i = 1;
        while (i < w.size()) {
            const std::uint32_t t = w[i];
            const std::uint32_t op = t & 0xFFFFU;
            if (op == 0xFFFFU) return true;  // end
            if (op == 0xFFFEU) {             // comment
                i += 1 + ((t >> 16) & 0x7FFFU);
                continue;
            }
            const std::uint32_t len = (t >> 24) & 0xFU;
            if ((t & (1U << 28)) != 0) return fail("predicated instruction");
            if (i + 1 + len > w.size()) return fail("truncated instruction");
            Op o;
            o.opcode = op;
            o.controls = (t >> 16) & 0xFFU;
            o.words.assign(w.begin() + static_cast<std::ptrdiff_t>(i + 1), w.begin() + static_cast<std::ptrdiff_t>(i + 1 + len));
            ops_.push_back(std::move(o));
            i += 1 + len;
        }
        return fail("no end token");
    }

    // Splits an instruction's words into its dst (if any) and sources, with relative tokens.
    bool operands(const Op& o, bool dst, Param& d, std::vector<Param>& src) const {
        std::size_t k = 0;
        if (dst) {
            if (o.words.empty()) return false;
            d = param(o.words[k++]);
            if (d.relative && k < o.words.size()) ++k;  // relative destination (vs_3_0 o#): not used
        }
        while (k < o.words.size()) {
            Param p = param(o.words[k++]);
            if (p.relative) {
                if (k >= o.words.size()) return false;
                const Param r = param(o.words[k++]);
                p.rel_type = r.type;
                p.rel_num = r.num;
                p.rel_comp = (r.token >> 16) & 3U;
            }
            src.push_back(p);
        }
        return true;
    }

    // --- declarations ----------------------------------------------------------------------
    static std::string flt(std::uint32_t bits) {
        float f = 0;
        std::memcpy(&f, &bits, 4);
        char buf[48];
        if (f != f) return "(0.0/0.0)";
        if (f > 3.4e38F) return "(1.0/0.0)";
        if (f < -3.4e38F) return "(-1.0/0.0)";
        std::snprintf(buf, sizeof(buf), "%.9g", static_cast<double>(f));
        std::string s(buf);
        if (s.find_first_of(".eE") == std::string::npos) s += ".0";
        return s;
    }

    bool declare() {
        for (const Op& o : ops_) {
            if (o.opcode == 31) {  // dcl: usage token, then the register
                if (o.words.size() < 2) return fail("bad dcl");
                const std::uint32_t u = o.words[0];
                const Param r = param(o.words[1]);
                if (r.type == kSampler) {
                    if (r.num >= 16) return fail("sampler out of range");
                    s_.samplers[r.num] = static_cast<int>((u >> 27) & 0xFU);
                } else {
                    Semantic sem{r.num, u & 0x1FU, (u >> 16) & 0xFU};
                    if (!s_.pixel && r.type == kInput) s_.inputs.push_back(sem);
                    else if (!s_.pixel && r.type == kOutput) s_.outputs.push_back(sem);
                    else if (s_.pixel && r.type == kInput && s_.major >= 3) s_.inputs.push_back(sem);
                    else if (s_.pixel && r.type == kInput) s_.inputs.push_back(Semantic{r.num, kColor, r.num});
                    else if (s_.pixel && r.type == kAddrOrTexture) s_.inputs.push_back(Semantic{r.num + 100U, kTexCoord, r.num});
                }
            } else if (o.opcode == 81) {  // def c#, 4 floats
                if (o.words.size() < 5) return fail("bad def");
                const Param r = param(o.words[0]);
                if (r.num >= 256) return fail("def out of range");
                defs_[r.num] = "vec4(" + flt(o.words[1]) + ", " + flt(o.words[2]) + ", " + flt(o.words[3]) + ", " + flt(o.words[4]) + ")";
            } else if (o.opcode == 48) {  // defi i#, 4 ints
                if (o.words.size() < 5) return fail("bad defi");
                const Param r = param(o.words[0]);
                if (r.num >= 16) return fail("defi out of range");
                defi_[r.num] = "ivec4(" + std::to_string(static_cast<std::int32_t>(o.words[1])) + ", " +
                               std::to_string(static_cast<std::int32_t>(o.words[2])) + ", " +
                               std::to_string(static_cast<std::int32_t>(o.words[3])) + ", " +
                               std::to_string(static_cast<std::int32_t>(o.words[4])) + ")";
            } else if (o.opcode == 47) {  // defb b#, 1 bool
                if (o.words.size() < 2) return fail("bad defb");
                const Param r = param(o.words[0]);
                if (r.num >= 16) return fail("defb out of range");
                defb_[r.num] = o.words[1] != 0 ? "true" : "false";
            }
        }
        return true;
    }

    // --- operand text ------------------------------------------------------------------------
    std::string reg_name(const Param& p, bool for_write) {
        const std::string n = std::to_string(p.num);
        switch (p.type) {
        case kTemp:
        case kTempFloat16:
            temps_.insert(p.num);
            return "r" + n;
        case kInput:
            return "v" + n;
        case kConst:
        case kConst2:
        case kConst3:
        case kConst4: {
            const std::uint32_t base = p.type == kConst ? 0 : p.type == kConst2 ? 2048 : p.type == kConst3 ? 4096 : 6144;
            if (p.relative) {
                const std::string idx = p.rel_type == kLoop ? loop_name() : "a0." + std::string(1, kComp[p.rel_comp]);
                return "u.c[clamp(" + idx + " + " + std::to_string(base + p.num) + ", 0, 255)]";
            }
            if (base + p.num < 256 && !defs_[base + p.num].empty()) return defs_[base + p.num];
            return "u.c[" + std::to_string(base + p.num) + "]";
        }
        case kAddrOrTexture:
            if (s_.pixel) return "t" + n;
            return for_write ? "a0" : "vec4(a0)";
        case kRastOut:
            written_out_.insert((kRastOut << 8) | p.num);
            return p.num == 0 ? "oPos" : p.num == 1 ? "oFog" : "oPts";
        case kAttrOut:
            written_out_.insert((kAttrOut << 8) | p.num);
            return "oD" + n;
        case kOutput:
            written_out_.insert((kOutput << 8) | p.num);
            return (s_.major >= 3 ? "o" : "oT") + n;
        case kColorOut:
            written_out_.insert((kColorOut << 8) | p.num);
            return "oC" + n;
        case kDepthOut:
            written_out_.insert(kDepthOut << 8);
            return "oDepth";
        case kConstInt:
            return defi_[p.num % 16].empty() ? "u.i[" + n + "]" : defi_[p.num % 16];
        case kConstBool:
            return defb_[p.num % 16].empty() ? "(u.b[" + n + "].x != 0)" : defb_[p.num % 16];
        case kLoop:
            return loop_name();
        case kMisc:
            return p.num == 0 ? "vPos" : "vFace";
        default:
            return "";
        }
    }

    std::string loop_name() const { return "aL" + std::to_string(loop_depth_ > 0 ? loop_depth_ - 1 : 0); }

    // A source as a vec4 expression (swizzle and modifier applied).
    std::string src(const Param& p) {
        std::string base = reg_name(p, false);
        if (p.type == kConstInt) base = "vec4(" + base + ")";
        const std::uint32_t sw = (p.token >> 16) & 0xFFU;
        std::string swz = ".";
        for (int c = 0; c < 4; ++c) swz += kComp[(sw >> (2 * c)) & 3U];
        std::string e = "(" + base + ")" + (swz == ".xyzw" ? "" : swz);
        switch ((p.token >> 24) & 0xFU) {
        case 0: return e;
        case 1: return "(-" + e + ")";
        case 2: return "(" + e + " - 0.5)";
        case 3: return "(0.5 - " + e + ")";
        case 4: return "(2.0 * " + e + " - 1.0)";
        case 5: return "(1.0 - 2.0 * " + e + ")";
        case 6: return "(1.0 - " + e + ")";
        case 7: return "(2.0 * " + e + ")";
        case 8: return "(-2.0 * " + e + ")";
        case 9: return "(" + e + " / " + e + ".zzzz)";
        case 10: return "(" + e + " / " + e + ".wwww)";
        case 11: return "abs" + e;
        case 12: return "(-abs" + e + ")";
        default: return e;
        }
    }
    std::string scalar(const Param& p) { return src(p) + ".x"; }

    void line(const std::string& text) {
        body_.append(static_cast<std::size_t>(indent_) * 4, ' ');
        body_ += text;
        body_ += '\n';
    }

    // dst.mask = saturate?(value).mask
    void write(const Param& d, const std::string& vec4_value) {
        const std::uint32_t mask = (d.token >> 16) & 0xFU;
        std::string m;
        for (int c = 0; c < 4; ++c) {
            if ((mask >> c) & 1U) m += kComp[c];
        }
        if (m.empty()) m = "xyzw";
        std::string v = vec4_value;
        if (((d.token >> 20) & 1U) != 0) v = "clamp(" + v + ", 0.0, 1.0)";
        const std::string name = reg_name(d, true);
        if (!s_.pixel && d.type == kAddrOrTexture) {
            line(name + "." + m + " = ivec4(floor(" + v + " + 0.5))." + m + ";");  // mova rounds to nearest
        } else {
            line(name + "." + m + " = (" + v + ")." + m + ";");
        }
    }

    static const char* cmp_op(std::uint32_t c) {
        switch (c & 7U) {
        case 1: return ">";
        case 2: return "==";
        case 3: return ">=";
        case 4: return "<";
        case 5: return "!=";
        case 6: return "<=";
        default: return nullptr;
        }
    }

    std::string sample(const Param& coord, const Param& sampler, std::uint32_t controls, int kind, const std::vector<Param>& rest) {
        if (sampler.num >= 16) return "";
        int dim = s_.samplers[sampler.num];
        if (dim == 0) dim = 2;  // undeclared (ps_2_0 implies 2D)
        const std::string sname = "s" + std::to_string(sampler.num);
        const std::string c = src(coord);
        const std::string uv = dim == 2 ? c + ".xy" : c + ".xyz";
        if (kind == 1) return "textureLod(" + sname + ", " + uv + ", " + c + ".w)";
        if (kind == 2 && rest.size() >= 2) {
            const std::string g = dim == 2 ? ".xy" : ".xyz";
            return "textureGrad(" + sname + ", " + uv + ", " + src(rest[0]) + g + ", " + src(rest[1]) + g + ")";
        }
        if ((controls & 1U) != 0) {  // texldp
            if (dim == 2) return "textureProj(" + sname + ", " + c + ")";
            return "texture(" + sname + ", " + c + ".xyz / " + c + ".w)";
        }
        if ((controls & 2U) != 0) return "texture(" + sname + ", " + uv + ", " + c + ".w)";  // texldb
        if (!s_.pixel) return "textureLod(" + sname + ", " + uv + ", 0.0)";
        return "texture(" + sname + ", " + uv + ")";
    }

    bool instruction(const Op& o) {
        Param d;
        std::vector<Param> sv;
        const bool dst = has_dst(o.opcode);
        if (o.opcode == 31 || o.opcode == 81 || o.opcode == 48 || o.opcode == 47 || o.opcode == 0) return true;
        if (!operands(o, dst, d, sv)) return fail("bad operands");
        auto need = [&](std::size_t n) { return sv.size() >= n; };
        auto bin = [&](const char* f) { return std::string(f) + "(" + src(sv[0]) + ", " + src(sv[1]) + ")"; };
        switch (o.opcode) {
        case 1: if (!need(1)) break; write(d, src(sv[0])); return true;                                     // mov
        case 46: if (!need(1)) break; write(d, src(sv[0])); return true;                                    // mova
        case 2: if (!need(2)) break; write(d, src(sv[0]) + " + " + src(sv[1])); return true;                 // add
        case 3: if (!need(2)) break; write(d, src(sv[0]) + " - " + src(sv[1])); return true;                 // sub
        case 4: if (!need(3)) break; write(d, src(sv[0]) + " * " + src(sv[1]) + " + " + src(sv[2])); return true;  // mad
        case 5: if (!need(2)) break; write(d, src(sv[0]) + " * " + src(sv[1])); return true;                 // mul
        case 6: if (!need(1)) break; write(d, "vec4(1.0 / " + scalar(sv[0]) + ")"); return true;              // rcp
        case 7: if (!need(1)) break; write(d, "vec4(inversesqrt(abs(" + scalar(sv[0]) + ")))"); return true;  // rsq
        case 8: if (!need(2)) break; write(d, "vec4(dot(" + src(sv[0]) + ".xyz, " + src(sv[1]) + ".xyz))"); return true;  // dp3
        case 9: if (!need(2)) break; write(d, "vec4(dot(" + src(sv[0]) + ", " + src(sv[1]) + "))"); return true;  // dp4
        case 10: if (!need(2)) break; write(d, bin("min")); return true;
        case 11: if (!need(2)) break; write(d, bin("max")); return true;
        case 12: if (!need(2)) break; write(d, "vec4(lessThan(" + src(sv[0]) + ", " + src(sv[1]) + "))"); return true;  // slt
        case 13: if (!need(2)) break; write(d, "vec4(greaterThanEqual(" + src(sv[0]) + ", " + src(sv[1]) + "))"); return true;  // sge
        case 14: if (!need(1)) break; write(d, "vec4(exp2(" + scalar(sv[0]) + "))"); return true;            // exp
        case 15: if (!need(1)) break; write(d, "vec4(log2(abs(" + scalar(sv[0]) + ")))"); return true;       // log
        case 18: if (!need(3)) break; write(d, "mix(" + src(sv[2]) + ", " + src(sv[1]) + ", " + src(sv[0]) + ")"); return true;  // lrp
        case 19: if (!need(1)) break; write(d, "fract(" + src(sv[0]) + ")"); return true;                    // frc
        case 32: if (!need(2)) break; write(d, "vec4(exp2(" + scalar(sv[1]) + " * log2(abs(" + scalar(sv[0]) + "))))"); return true;  // pow
        case 33: if (!need(2)) break; write(d, "vec4(cross(" + src(sv[0]) + ".xyz, " + src(sv[1]) + ".xyz), 0.0)"); return true;  // crs
        case 34: if (!need(1)) break; write(d, "sign(" + src(sv[0]) + ")"); return true;                     // sgn
        case 35: if (!need(1)) break; write(d, "abs(" + src(sv[0]) + ")"); return true;                      // abs
        case 36: if (!need(1)) break; write(d, src(sv[0]) + " * inversesqrt(dot(" + src(sv[0]) + ".xyz, " + src(sv[0]) + ".xyz))"); return true;  // nrm
        case 37: {                                                                                             // sincos
            if (!need(1)) break;
            const std::string a = scalar(sv[0]);
            write(d, "vec4(cos(" + a + "), sin(" + a + "), 0.0, 0.0)");
            return true;
        }
        case 88: if (!need(3)) break; write(d, "mix(" + src(sv[2]) + ", " + src(sv[1]) + ", greaterThanEqual(" + src(sv[0]) + ", vec4(0.0)))"); return true;  // cmp
        case 90: if (!need(3)) break; write(d, "vec4(dot(" + src(sv[0]) + ".xy, " + src(sv[1]) + ".xy) + " + scalar(sv[2]) + ")"); return true;  // dp2add
        case 91: if (!need(1)) break; write(d, "dFdx(" + src(sv[0]) + ")"); return true;                     // dsx
        case 92: if (!need(1)) break; write(d, "dFdy(" + src(sv[0]) + ")"); return true;                     // dsy
        case 66: {                                                                                             // texld (texldp, texldb)
            if (!need(2)) break;
            const std::string e = sample(sv[0], sv[1], o.controls, 0, {});
            if (e.empty()) return fail("bad sampler");
            write(d, e);
            return true;
        }
        case 79: case 95: {                                                                                    // texldl
            if (!need(2)) break;
            const std::string e = sample(sv[0], sv[1], 0, 1, {});
            if (e.empty()) return fail("bad sampler");
            write(d, e);
            return true;
        }
        case 77: case 93: {                                                                                    // texldd
            if (!need(4)) break;
            const std::string e = sample(sv[0], sv[1], 0, 2, {sv[2], sv[3]});
            if (e.empty()) return fail("bad sampler");
            write(d, e);
            return true;
        }
        case 65: {                                                                                             // texkill
            const std::uint32_t mask = (d.token >> 16) & 0xFU;
            std::string m;
            for (int c = 0; c < 4; ++c) {
                if ((mask >> c) & 1U) m += kComp[c];
            }
            if (m.empty()) m = "xyzw";
            const std::string r = reg_name(d, false);
            line(m.size() == 1 ? "if (" + r + "." + m + " < 0.0) discard;"
                               : "if (any(lessThan(" + r + "." + m + ", vec" + std::to_string(m.size()) + "(0.0)))) discard;");
            return true;
        }
        case 40: {                                                                                             // if b#
            if (!need(1)) break;
            line("if (" + reg_name(sv[0], false) + ") {");
            ++indent_;
            block_kinds_.push_back(0);
            return true;
        }
        case 41: {                                                                                             // ifc
            if (!need(2) || cmp_op(o.controls) == nullptr) break;
            line("if (" + scalar(sv[0]) + " " + cmp_op(o.controls) + " " + scalar(sv[1]) + ") {");
            ++indent_;
            block_kinds_.push_back(0);
            return true;
        }
        case 42:                                                                                               // else
            if (block_kinds_.empty() || block_kinds_.back() != 0) return fail("else outside if");
            --indent_;
            line("} else {");
            ++indent_;
            return true;
        case 43:                                                                                               // endif
            if (block_kinds_.empty() || block_kinds_.back() != 0) return fail("endif outside if");
            block_kinds_.pop_back();
            --indent_;
            line("}");
            return true;
        case 38: {                                                                                             // rep i#
            if (!need(1)) break;
            const std::string k = "rep" + std::to_string(block_kinds_.size());
            line("for (int " + k + " = 0; " + k + " < " + reg_name(sv[0], false) + ".x; ++" + k + ") {");
            ++indent_;
            block_kinds_.push_back(1);
            return true;
        }
        case 39:                                                                                               // endrep
            if (block_kinds_.empty() || block_kinds_.back() != 1) return fail("endrep outside rep");
            block_kinds_.pop_back();
            --indent_;
            line("}");
            return true;
        case 27: {                                                                                             // loop aL, i#
            if (!need(2)) break;
            const std::string iv = reg_name(sv[1], false);
            ++loop_depth_;
            loops_max_ = std::max(loops_max_, loop_depth_);
            const std::string al = loop_name();
            const std::string k = "lp" + std::to_string(loop_depth_);
            line(al + " = " + iv + ".y;");
            line("for (int " + k + " = 0; " + k + " < " + iv + ".x; ++" + k + ") {");
            ++indent_;
            block_kinds_.push_back(2);
            loop_steps_.push_back(iv + ".z");
            return true;
        }
        case 29: {                                                                                             // endloop
            if (block_kinds_.empty() || block_kinds_.back() != 2) return fail("endloop outside loop");
            line(loop_name() + " += " + loop_steps_.back() + ";");
            loop_steps_.pop_back();
            block_kinds_.pop_back();
            --indent_;
            line("}");
            --loop_depth_;
            return true;
        }
        case 44: line("break;"); return true;                                                                  // break
        case 45: {                                                                                             // breakc
            if (!need(2) || cmp_op(o.controls) == nullptr) break;
            line("if (" + scalar(sv[0]) + " " + cmp_op(o.controls) + " " + scalar(sv[1]) + ") break;");
            return true;
        }
        default:
            return fail("unsupported instruction " + std::to_string(o.opcode));
        }
        return fail("missing operands for instruction " + std::to_string(o.opcode));
    }
    int loops_max_ = 0;
    std::vector<std::string> loop_steps_;

    // --- the whole shader --------------------------------------------------------------------
    bool emit() {
        for (const Op& o : ops_) {
            if (!instruction(o)) return false;
        }
        if (!block_kinds_.empty()) return fail("unclosed block");
        std::string g = "#version 450\n";
        const char* binding = s_.pixel ? "1" : "0";
        g += std::string("layout(std140, set = 0, binding = ") + binding +
             ") uniform Constants { vec4 c[256]; ivec4 i[16]; ivec4 b[16]; vec4 posfix; } u;\n";
        for (std::size_t n = 0; n < 16; ++n) {
            if (s_.samplers[n] == 0 && !uses_sampler(static_cast<std::uint32_t>(n))) continue;
            const int dim = s_.samplers[n] == 0 ? 2 : s_.samplers[n];
            const char* type = dim == 3 ? "samplerCube" : dim == 4 ? "sampler3D" : "sampler2D";
            g += "layout(set = 1, binding = " + std::to_string(n) + ") uniform " + type + " s" + std::to_string(n) + ";\n";
        }
        std::string prologue;
        std::string epilogue;
        if (!s_.pixel) {
            for (const Semantic& in : s_.inputs) {
                g += "layout(location = " + std::to_string(in.reg) + ") in vec4 v" + std::to_string(in.reg) + ";\n";
            }
            prologue += "    ivec4 a0 = ivec4(0);\n";
            if (s_.major >= 3) {
                for (const Semantic& o : s_.outputs) {
                    const std::string name = "o" + std::to_string(o.reg);
                    prologue += "    vec4 " + name + " = vec4(0.0);\n";
                    if (o.usage == kPosition && o.index == 0) {
                        epilogue += "    gl_Position = " + name + ";\n";
                    } else if (o.usage == kPSize) {
                        epilogue += "    gl_PointSize = " + name + ".x;\n";
                    } else {
                        const int loc = link_location(o.usage, o.index);
                        if (loc < 0) return fail("vertex output with no link location");
                        g += "layout(location = " + std::to_string(loc) + ") out vec4 out_" + name + ";\n";
                        epilogue += "    out_" + name + " = " + name + ";\n";
                    }
                }
            } else {
                prologue += "    vec4 oPos = vec4(0.0); vec4 oFog = vec4(0.0); vec4 oPts = vec4(0.0);\n";
                epilogue += "    gl_Position = oPos;\n";
                for (int n = 0; n < 2; ++n) {
                    prologue += "    vec4 oD" + std::to_string(n) + " = vec4(0.0);\n";
                    if (written_out_.count((kAttrOut << 8) | static_cast<std::uint32_t>(n)) != 0) {
                        g += "layout(location = " + std::to_string(10 + n) + ") out vec4 out_oD" + std::to_string(n) + ";\n";
                        epilogue += "    out_oD" + std::to_string(n) + " = oD" + std::to_string(n) + ";\n";
                        s_.outputs.push_back(Semantic{static_cast<std::uint32_t>(n), kColor, static_cast<std::uint32_t>(n)});
                    }
                }
                for (int n = 0; n < 8; ++n) {
                    prologue += "    vec4 oT" + std::to_string(n) + " = vec4(0.0);\n";
                    if (written_out_.count((kOutput << 8) | static_cast<std::uint32_t>(n)) != 0) {
                        g += "layout(location = " + std::to_string(n) + ") out vec4 out_oT" + std::to_string(n) + ";\n";
                        epilogue += "    out_oT" + std::to_string(n) + " = oT" + std::to_string(n) + ";\n";
                        s_.outputs.push_back(Semantic{static_cast<std::uint32_t>(n), kTexCoord, static_cast<std::uint32_t>(n)});
                    }
                }
                if (written_out_.count((kRastOut << 8) | 1U) != 0) {
                    g += "layout(location = 12) out vec4 out_oFog;\n";
                    epilogue += "    out_oFog = oFog;\n";
                    s_.outputs.push_back(Semantic{1, kFog, 0});
                }
                if (written_out_.count((kRastOut << 8) | 2U) != 0) epilogue += "    gl_PointSize = oPts.x;\n";
            }
            // D3D9 clip space has y up; Vulkan's has y down. posfix: the half-pixel offset.
            epilogue += "    gl_Position.y = -gl_Position.y;\n";
            epilogue += "    gl_Position.xy += u.posfix.xy * gl_Position.w;\n";
        } else {
            for (const Semantic& in : s_.inputs) {
                const int loc = link_location(in.usage, in.index);
                if (loc < 0) return fail("pixel input with no link location");
                const std::string name = in.reg >= 100 ? "t" + std::to_string(in.reg - 100) : "v" + std::to_string(in.reg);
                g += "layout(location = " + std::to_string(loc) + ") in vec4 " + name + ";\n";
            }
            for (int n = 0; n < 4; ++n) {
                prologue += "    vec4 oC" + std::to_string(n) + " = vec4(0.0);\n";
                if (written_out_.count((kColorOut << 8) | static_cast<std::uint32_t>(n)) != 0) {
                    g += "layout(location = " + std::to_string(n) + ") out vec4 out_oC" + std::to_string(n) + ";\n";
                    epilogue += "    out_oC" + std::to_string(n) + " = oC" + std::to_string(n) + ";\n";
                }
            }
            if (written_out_.count(kDepthOut << 8) != 0) {
                prologue += "    vec4 oDepth = vec4(0.0);\n";
                epilogue += "    gl_FragDepth = oDepth.x;\n";
            }
            prologue += "    vec4 vPos = vec4(gl_FragCoord.xy - 0.5, 0.0, 0.0);\n";
            prologue += "    vec4 vFace = vec4(gl_FrontFacing ? 1.0 : -1.0);\n";
        }
        for (std::uint32_t t : temps_) prologue += "    vec4 r" + std::to_string(t) + " = vec4(0.0);\n";
        for (int l = 0; l < std::max(1, loops_max_); ++l) prologue += "    int aL" + std::to_string(l) + " = 0;\n";
        g += "void main() {\n" + prologue + body_ + epilogue + "}\n";
        s_.glsl = std::move(g);
        return true;
    }

    bool uses_sampler(std::uint32_t n) const {
        for (const Op& o : ops_) {
            if (o.opcode != 66 && o.opcode != 79 && o.opcode != 95 && o.opcode != 77 && o.opcode != 93) continue;
            for (std::uint32_t w : o.words) {
                const Param p = param(w);
                if (p.type == kSampler && p.num == n) return true;
            }
        }
        return false;
    }
};

}  // namespace detail

// Translates one shader (its tokens, from the version token to the end token). False with
// `out.error` when the shader uses something outside the scope above.
inline bool translate(std::span<const std::uint32_t> tokens, Shader& out) {
    out = Shader{};
    detail::Translator t(out);
    return t.run(tokens);
}

inline bool translate(std::span<const std::uint8_t> bytes, Shader& out) {
    std::vector<std::uint32_t> words(bytes.size() / 4);
    if (!words.empty()) std::memcpy(words.data(), bytes.data(), words.size() * 4);
    return translate(std::span<const std::uint32_t>(words), out);
}

}  // namespace wse::recomp::d3d9sh
