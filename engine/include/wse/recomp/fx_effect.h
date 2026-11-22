// D3DX effect files (`fx_2_0`, the `.fxo` files the game loads with D3DXCreateEffectFromFileA):
// a parser for Phase 5 R5a step 1 (`docs/RECOMP_PLAN.md` "R5 — Graphics"). It reads every
// parameter (type, default value, sampler states), technique, pass, pass state, object (shader
// bytecode and strings) and resource (the inline shader of a pass state, or the parameter a state
// names), so our ID3DXEffect can resolve handles, constants and shaders. Header-only.
//
// Layout, measured on the user's install (all seven effect files parse to their last byte, and the
// shader count matches an independent bytecode scan: 399 in mb_wse2_3_0.fxo, 377 in
// mb_wse2_2a.fxo; 2026-11-22). Offsets are relative to `base` = file start + 8.
//
//   header      u32 0xFEFF0901 (fx_2_0), u32 offset of the table
//   table       u32 parameters, u32 techniques, u32 (unused), u32 object slots,
//               parameters, techniques, u32 objects, u32 resources, objects, resources
//   parameter   u32 typedef offset, u32 value offset, u32 flags, u32 annotations, annotations
//   annotation  u32 typedef offset, u32 value offset
//   typedef     u32 type, u32 class, u32 name offset, u32 semantic offset, u32 elements, then
//               scalar/vector/matrix: u32 rows, u32 columns; struct: u32 members + member typedefs
//               (inline); object: nothing
//   technique   u32 name offset, u32 annotations, u32 passes, annotations, passes
//   pass        u32 name offset, u32 annotations, u32 states, annotations, states
//   state       u32 operation, u32 index, u32 typedef offset, u32 value offset
//   name        u32 length (with the NUL), bytes; at an offset
//   object      u32 id, u32 size, bytes padded to 4 (shader bytecode, or a string)
//   resource    u32 technique, u32 pass, u32 element, u32 state, u32 usage, then usage 1: a name
//               (u32 length, bytes padded) naming the parameter that holds the state's object;
//               usage 0 or 2: u32 size, bytes padded (an inline shader, or an expression)
//
// A numeric value holds rows * columns * max(1, elements) 4-byte words; an object value holds one
// object id per element; a sampler's value holds u32 states and that many states. A resource with
// technique 0xFFFFFFFF belongs to a parameter: `pass` is then the parameter index.
//
// Safety: every read is bounds-checked and recursion is limited; a malformed file returns false
// with an error instead of reading out of range.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <vector>

namespace wse::recomp::fx {

inline constexpr std::uint32_t kFx20Tag = 0xFEFF0901U;
inline constexpr std::uint32_t kNoTechnique = 0xFFFFFFFFU;

// D3DXPARAMETER_CLASS / D3DXPARAMETER_TYPE values used by the parser.
enum class ParamClass : std::uint32_t { scalar = 0, vector = 1, matrix_rows = 2, matrix_columns = 3, object = 4, structure = 5 };
enum ParamType : std::uint32_t {
    kVoid = 0, kBool = 1, kInt = 2, kFloat = 3, kString = 4, kTexture = 5, kTexture1D = 6, kTexture2D = 7,
    kTexture3D = 8, kTextureCube = 9, kSampler = 10, kSampler1D = 11, kSampler2D = 12, kSampler3D = 13,
    kSamplerCube = 14, kPixelShader = 15, kVertexShader = 16,
};

struct TypeDef {
    std::uint32_t type = 0;
    ParamClass cls = ParamClass::scalar;
    std::string name;
    std::string semantic;
    std::uint32_t elements = 0;  // 0 = not an array
    std::uint32_t rows = 0;
    std::uint32_t columns = 0;
    std::vector<TypeDef> members;  // a struct's members

    [[nodiscard]] bool is_sampler() const noexcept { return type >= kSampler && type <= kSamplerCube; }
    [[nodiscard]] std::uint32_t count() const noexcept { return elements == 0 ? 1 : elements; }
};

struct State {
    std::uint32_t operation = 0;  // which state (the effect framework's state table)
    std::uint32_t index = 0;      // e.g. the sampler or light number
    TypeDef type;
    std::vector<std::uint8_t> value;  // numeric words, or one u32 object id per element
};

struct Parameter {
    TypeDef type;
    std::uint32_t flags = 0;
    std::uint32_t annotations = 0;
    std::vector<std::uint8_t> value;  // default value: numeric words, or object ids
    std::vector<State> sampler_states;  // a sampler's states (Texture, filters, address modes, ...)
};

struct Pass {
    std::string name;
    std::uint32_t annotations = 0;
    std::vector<State> states;
};

struct Technique {
    std::string name;
    std::uint32_t annotations = 0;
    std::vector<Pass> passes;
};

struct Object {
    std::uint32_t id = 0;
    std::vector<std::uint8_t> data;
};

struct Resource {
    std::uint32_t technique = 0;  // kNoTechnique: a parameter's (then `pass` is its index)
    std::uint32_t pass = 0;
    std::uint32_t element = 0;
    std::uint32_t state = 0;
    std::uint32_t usage = 0;           // 1: `name`; 0 or 2: `data`
    std::string name;                  // the parameter holding the state's object
    std::vector<std::uint8_t> data;    // an inline shader (usage 0) or an expression (usage 2)
};

struct Effect {
    std::vector<Parameter> parameters;
    std::vector<Technique> techniques;
    std::uint32_t object_slots = 0;
    std::vector<Object> objects;
    std::vector<Resource> resources;

    [[nodiscard]] const Parameter* parameter(std::string_view name) const noexcept {
        for (const Parameter& p : parameters) {
            if (p.type.name == name) return &p;
        }
        return nullptr;
    }
    [[nodiscard]] const Technique* technique(std::string_view name) const noexcept {
        for (const Technique& t : techniques) {
            if (t.name == name) return &t;
        }
        return nullptr;
    }
};

namespace detail {

// Bounds-checked reader over the bytes after the 8-byte header.
class Reader {
public:
    explicit Reader(std::span<const std::uint8_t> base) : base_(base) {}

    [[nodiscard]] bool word_at(std::size_t off, std::uint32_t& out) const noexcept {
        if (off > base_.size() || base_.size() - off < 4) return false;
        std::memcpy(&out, base_.data() + off, 4);
        return true;
    }
    // Sequential read at `pos`.
    [[nodiscard]] bool next(std::size_t& pos, std::uint32_t& out) const noexcept {
        if (!word_at(pos, out)) return false;
        pos += 4;
        return true;
    }
    // A name at `off`: u32 length (with the NUL), bytes.
    [[nodiscard]] bool name_at(std::size_t off, std::string& out) const {
        std::uint32_t n = 0;
        if (!word_at(off, n)) return false;
        if (n > base_.size() - off - 4) return false;
        out.assign(reinterpret_cast<const char*>(base_.data() + off + 4), n);
        while (!out.empty() && out.back() == '\0') out.pop_back();
        return true;
    }
    // Size-prefixed bytes at `pos`, padded to 4; advances `pos`.
    [[nodiscard]] bool blob(std::size_t& pos, std::vector<std::uint8_t>& out) const {
        std::uint32_t n = 0;
        if (!next(pos, n)) return false;
        if (n > base_.size() - pos) return false;
        out.assign(base_.begin() + static_cast<std::ptrdiff_t>(pos), base_.begin() + static_cast<std::ptrdiff_t>(pos + n));
        const std::size_t padded = (static_cast<std::size_t>(n) + 3U) & ~std::size_t{3};
        if (padded > base_.size() - pos) return false;
        pos += padded;
        return true;
    }
    [[nodiscard]] bool bytes_at(std::size_t off, std::size_t n, std::vector<std::uint8_t>& out) const {
        if (off > base_.size() || n > base_.size() - off) return false;
        out.assign(base_.begin() + static_cast<std::ptrdiff_t>(off), base_.begin() + static_cast<std::ptrdiff_t>(off + n));
        return true;
    }
    [[nodiscard]] std::size_t size() const noexcept { return base_.size(); }

private:
    std::span<const std::uint8_t> base_;
};

inline constexpr int kMaxDepth = 8;
inline constexpr std::uint32_t kMaxCount = 1U << 20;  // no effect comes near this; guards loops

// The typedef at `off`; `size` gets the record's length (struct members are inline).
inline bool read_typedef(const Reader& r, std::size_t off, TypeDef& t, std::size_t& size, int depth) {
    if (depth > kMaxDepth) return false;
    std::uint32_t cls = 0;
    std::uint32_t name_off = 0;
    std::uint32_t sem_off = 0;
    std::size_t p = off;
    if (!r.next(p, t.type) || !r.next(p, cls) || !r.next(p, name_off) || !r.next(p, sem_off) || !r.next(p, t.elements)) return false;
    if (cls > 5 || t.elements > kMaxCount) return false;
    t.cls = static_cast<ParamClass>(cls);
    if (!r.name_at(name_off, t.name) || !r.name_at(sem_off, t.semantic)) return false;
    switch (t.cls) {
    case ParamClass::scalar:
    case ParamClass::vector:
    case ParamClass::matrix_rows:
    case ParamClass::matrix_columns:
        if (!r.next(p, t.rows) || !r.next(p, t.columns)) return false;
        if (t.rows > 4 || t.columns > 4) return false;
        break;
    case ParamClass::structure: {
        std::uint32_t members = 0;
        if (!r.next(p, members) || members > kMaxCount) return false;
        t.members.resize(members);
        for (TypeDef& m : t.members) {
            std::size_t msize = 0;
            if (!read_typedef(r, p, m, msize, depth + 1)) return false;
            p += msize;
        }
        break;
    }
    case ParamClass::object:
        break;
    }
    size = p - off;
    return true;
}

// Bytes a value of `t` takes (structs: their members in order).
inline std::size_t value_bytes(const TypeDef& t, int depth = 0) {
    if (depth > kMaxDepth) return 0;
    std::size_t one = 0;
    if (t.cls == ParamClass::structure) {
        for (const TypeDef& m : t.members) one += value_bytes(m, depth + 1);
    } else if (t.cls == ParamClass::object) {
        one = 4;  // an object id
    } else {
        one = 4U * t.rows * t.columns;
    }
    return one * t.count();
}

inline bool read_state(const Reader& r, std::size_t& pos, State& st) {
    std::uint32_t toff = 0;
    std::uint32_t voff = 0;
    if (!r.next(pos, st.operation) || !r.next(pos, st.index) || !r.next(pos, toff) || !r.next(pos, voff)) return false;
    std::size_t size = 0;
    if (!read_typedef(r, toff, st.type, size, 0)) return false;
    return r.bytes_at(voff, value_bytes(st.type), st.value);
}

inline bool skip_annotations(const Reader& r, std::size_t& pos, std::uint32_t n) {
    if (n > kMaxCount) return false;
    for (std::uint32_t i = 0; i < n; ++i) {
        std::uint32_t a = 0;
        std::uint32_t b = 0;
        if (!r.next(pos, a) || !r.next(pos, b)) return false;
    }
    return true;
}

inline bool fail(std::string* error, const char* what) {
    if (error != nullptr) *error = what;
    return false;
}

}  // namespace detail

// Parses an fx_2_0 effect. False (with `error`) when the file is not fx_2_0 or is malformed.
inline bool parse(std::span<const std::uint8_t> file, Effect& out, std::string* error = nullptr) {
    using detail::fail;
    out = Effect{};
    std::uint32_t tag = 0;
    std::uint32_t table = 0;
    if (file.size() < 8) return fail(error, "shorter than the fx header");
    std::memcpy(&tag, file.data(), 4);
    std::memcpy(&table, file.data() + 4, 4);
    if (tag != kFx20Tag) return fail(error, "not an fx_2_0 effect");
    const detail::Reader r(file.subspan(8));
    std::size_t pos = table;
    std::uint32_t nparam = 0;
    std::uint32_t ntech = 0;
    std::uint32_t unused = 0;
    if (!r.next(pos, nparam) || !r.next(pos, ntech) || !r.next(pos, unused) || !r.next(pos, out.object_slots)) {
        return fail(error, "truncated table");
    }
    if (nparam > detail::kMaxCount || ntech > detail::kMaxCount) return fail(error, "implausible counts");

    out.parameters.resize(nparam);
    for (Parameter& p : out.parameters) {
        std::uint32_t toff = 0;
        std::uint32_t voff = 0;
        if (!r.next(pos, toff) || !r.next(pos, voff) || !r.next(pos, p.flags) || !r.next(pos, p.annotations)) {
            return fail(error, "truncated parameter");
        }
        if (!detail::skip_annotations(r, pos, p.annotations)) return fail(error, "truncated annotations");
        std::size_t size = 0;
        if (!detail::read_typedef(r, toff, p.type, size, 0)) return fail(error, "bad parameter type");
        if (p.type.is_sampler()) {
            // Per element: u32 states, then the states.
            std::size_t v = voff;
            for (std::uint32_t e = 0; e < p.type.count(); ++e) {
                std::uint32_t n = 0;
                if (!r.next(v, n) || n > detail::kMaxCount) return fail(error, "bad sampler value");
                for (std::uint32_t i = 0; i < n; ++i) {
                    State st;
                    if (!detail::read_state(r, v, st)) return fail(error, "bad sampler state");
                    p.sampler_states.push_back(std::move(st));
                }
            }
        } else if (!r.bytes_at(voff, detail::value_bytes(p.type), p.value)) {
            return fail(error, "bad parameter value");
        }
    }

    out.techniques.resize(ntech);
    for (Technique& t : out.techniques) {
        std::uint32_t name_off = 0;
        std::uint32_t npass = 0;
        if (!r.next(pos, name_off) || !r.next(pos, t.annotations) || !r.next(pos, npass)) return fail(error, "truncated technique");
        if (!r.name_at(name_off, t.name)) return fail(error, "bad technique name");
        if (!detail::skip_annotations(r, pos, t.annotations) || npass > detail::kMaxCount) return fail(error, "bad technique");
        t.passes.resize(npass);
        for (Pass& ps : t.passes) {
            std::uint32_t pname = 0;
            std::uint32_t nstate = 0;
            if (!r.next(pos, pname) || !r.next(pos, ps.annotations) || !r.next(pos, nstate)) return fail(error, "truncated pass");
            if (!r.name_at(pname, ps.name)) return fail(error, "bad pass name");
            if (!detail::skip_annotations(r, pos, ps.annotations) || nstate > detail::kMaxCount) return fail(error, "bad pass");
            ps.states.resize(nstate);
            for (State& st : ps.states) {
                if (!detail::read_state(r, pos, st)) return fail(error, "bad pass state");
            }
        }
    }

    std::uint32_t nobj = 0;
    std::uint32_t nres = 0;
    if (!r.next(pos, nobj) || !r.next(pos, nres) || nobj > detail::kMaxCount || nres > detail::kMaxCount) {
        return fail(error, "truncated object table");
    }
    out.objects.resize(nobj);
    for (Object& o : out.objects) {
        if (!r.next(pos, o.id) || !r.blob(pos, o.data)) return fail(error, "truncated object");
    }
    out.resources.resize(nres);
    for (Resource& res : out.resources) {
        if (!r.next(pos, res.technique) || !r.next(pos, res.pass) || !r.next(pos, res.element) || !r.next(pos, res.state) ||
            !r.next(pos, res.usage)) {
            return fail(error, "truncated resource");
        }
        if (res.usage == 1) {
            std::vector<std::uint8_t> bytes;
            if (!r.blob(pos, bytes)) return fail(error, "truncated resource name");
            res.name.assign(bytes.begin(), bytes.end());
            while (!res.name.empty() && res.name.back() == '\0') res.name.pop_back();
        } else if (res.usage == 0 || res.usage == 2) {
            if (!r.blob(pos, res.data)) return fail(error, "truncated resource data");
        } else {
            return fail(error, "unknown resource usage");
        }
    }
    if (pos != r.size()) return fail(error, "bytes left after the resources");
    return true;
}

// True when `data` starts with a D3D9 shader version token (vs/ps 1.x-3.0).
[[nodiscard]] inline bool is_shader_bytecode(std::span<const std::uint8_t> data) noexcept {
    if (data.size() < 4) return false;
    std::uint32_t v = 0;
    std::memcpy(&v, data.data(), 4);
    return (v & 0xFFFE0000U) == 0xFFFE0000U && ((v >> 8) & 0xFF) >= 1 && ((v >> 8) & 0xFF) <= 3;
}

// Every shader in the effect: shader objects and inline pass shaders.
[[nodiscard]] inline std::size_t shader_count(const Effect& e) noexcept {
    std::size_t n = 0;
    for (const Object& o : e.objects) n += is_shader_bytecode(o.data) ? 1U : 0U;
    for (const Resource& r : e.resources) n += (r.usage == 0 && is_shader_bytecode(r.data)) ? 1U : 0U;
    return n;
}

}  // namespace wse::recomp::fx
