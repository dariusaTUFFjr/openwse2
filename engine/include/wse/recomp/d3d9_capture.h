// D3D9/D3DX call captures for Phase 5 R5a step 3 (`docs/RECOMP_PLAN.md` "R5 — Graphics"): the
// recording D3D9 device (`shim_d3d9.h`) reports every COM and D3DX call it answers; a capture
// stores them, with the data their pointers reach, so that a Windows tool can replay the same
// calls on the real d3d9.dll (reference frames) and our Vulkan device can replay them too.
// Header-only.
//
// A capture holds retail data (shader bytecode, uploaded vertices and textures), so it stays on
// the user's PC and is never committed (`AGENTS.md` §1). Textures and effects loaded from files are
// stored as their paths; the replay reads them from the same install.
//
// File: "WSE2D3DC", u32 version (1), then records until the end of the file:
//   u32 record bytes (after this word), str interface, str method, u64 self, u64 ret,
//   u64 args[16] (the arguments as passed: after `this` for a COM method, from the first for a
//   D3DX function), u32 xmm[3] (the low float bits of xmm1..xmm3), u32 data count, then per
//   data: u32 tag, u32 bytes, bytes.
//   str is u16 length + bytes. Objects are identified by the address the device gave them.
// Data tags: kTagArg + i (the bytes argument i points to), kTagOut + i (the object address
// argument i received), kTagName (a parameter or technique name passed instead of a handle),
// kTagLocked (on unlock: for a vertex or index buffer, u32 offset, u32 size and the bytes that
// changed since its previous unlock, the rest being as then; for a texture level or surface, u32
// index and the whole level).

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <mutex>
#include <span>
#include <string>
#include <vector>

namespace wse::recomp::d3dcap {

inline constexpr char kMagic[8] = {'W', 'S', 'E', '2', 'D', '3', 'D', 'C'};
inline constexpr std::uint32_t kVersion = 1;
inline constexpr std::uint32_t kTagArg = 0x100;     // + argument index
inline constexpr std::uint32_t kTagOut = 0x200;     // + argument index
inline constexpr std::uint32_t kTagName = 0x300;
inline constexpr std::uint32_t kTagLocked = 0x400;

struct Data {
    std::uint32_t tag = 0;
    std::vector<std::uint8_t> bytes;
};

struct Call {
    std::string iface;   // "IDirect3DDevice9", ..., or "d3dx9_43.dll" for D3DX functions
    std::string method;
    std::uint64_t self = 0;
    std::uint64_t ret = 0;
    std::uint64_t args[16] = {};
    std::uint32_t xmm[3] = {};
    std::vector<Data> data;

    [[nodiscard]] const Data* find(std::uint32_t tag) const noexcept {
        for (const Data& d : data) {
            if (d.tag == tag) return &d;
        }
        return nullptr;
    }
};

namespace detail {
inline void put32(std::vector<std::uint8_t>& b, std::uint32_t v) {
    const std::size_t at = b.size();
    b.resize(at + 4);
    std::memcpy(b.data() + at, &v, 4);
}
inline void put64(std::vector<std::uint8_t>& b, std::uint64_t v) {
    const std::size_t at = b.size();
    b.resize(at + 8);
    std::memcpy(b.data() + at, &v, 8);
}
inline void put_str(std::vector<std::uint8_t>& b, const std::string& s) {
    const auto n = static_cast<std::uint16_t>(std::min<std::size_t>(s.size(), 0xFFFF));
    b.push_back(static_cast<std::uint8_t>(n & 0xFF));
    b.push_back(static_cast<std::uint8_t>(n >> 8));
    b.insert(b.end(), s.begin(), s.begin() + n);
}
}  // namespace detail

// One record's bytes (without the leading size word).
inline std::vector<std::uint8_t> encode(const Call& c) {
    std::vector<std::uint8_t> b;
    detail::put_str(b, c.iface);
    detail::put_str(b, c.method);
    detail::put64(b, c.self);
    detail::put64(b, c.ret);
    for (std::uint64_t a : c.args) detail::put64(b, a);
    for (std::uint32_t x : c.xmm) detail::put32(b, x);
    detail::put32(b, static_cast<std::uint32_t>(c.data.size()));
    for (const Data& d : c.data) {
        detail::put32(b, d.tag);
        detail::put32(b, static_cast<std::uint32_t>(d.bytes.size()));
        b.insert(b.end(), d.bytes.begin(), d.bytes.end());
    }
    return b;
}

// Reads a whole capture. False when the file is not a capture or a record is cut short (the
// records before it are kept: a run that ended mid-write still replays).
inline bool decode(std::span<const std::uint8_t> file, std::vector<Call>& out) {
    out.clear();
    if (file.size() < 12 || std::memcmp(file.data(), kMagic, 8) != 0) return false;
    std::uint32_t version = 0;
    std::memcpy(&version, file.data() + 8, 4);
    if (version != kVersion) return false;
    std::size_t p = 12;
    while (p < file.size()) {
        std::uint32_t n = 0;
        if (file.size() - p < 4) return false;
        std::memcpy(&n, file.data() + p, 4);
        p += 4;
        if (file.size() - p < n) return false;
        const std::span<const std::uint8_t> r = file.subspan(p, n);
        p += n;
        std::size_t q = 0;
        bool ok = true;
        auto take = [&](void* dst, std::size_t k) {
            if (!ok || r.size() - q < k) {
                ok = false;
                return;
            }
            std::memcpy(dst, r.data() + q, k);
            q += k;
        };
        auto str = [&](std::string& s) {
            std::uint16_t len = 0;
            take(&len, 2);
            if (!ok || r.size() - q < len) {
                ok = false;
                return;
            }
            s.assign(reinterpret_cast<const char*>(r.data() + q), len);
            q += len;
        };
        Call c;
        str(c.iface);
        str(c.method);
        take(&c.self, 8);
        take(&c.ret, 8);
        take(c.args, sizeof(c.args));
        take(c.xmm, sizeof(c.xmm));
        std::uint32_t count = 0;
        take(&count, 4);
        for (std::uint32_t i = 0; ok && i < count; ++i) {
            Data d;
            std::uint32_t len = 0;
            take(&d.tag, 4);
            take(&len, 4);
            if (!ok || r.size() - q < len) {
                ok = false;
                break;
            }
            d.bytes.assign(r.begin() + static_cast<std::ptrdiff_t>(q), r.begin() + static_cast<std::ptrdiff_t>(q + len));
            q += len;
            c.data.push_back(std::move(d));
        }
        if (!ok || q != r.size()) return false;
        out.push_back(std::move(c));
    }
    return true;
}

// Appends records to a file as calls arrive (thread-safe; flushed at every Present so a run that
// is stopped still leaves whole frames).
class Writer {
public:
    explicit Writer(const std::string& path) : out_(path, std::ios::binary | std::ios::trunc) {
        out_.write(kMagic, 8);
        const std::uint32_t v = kVersion;
        out_.write(reinterpret_cast<const char*>(&v), 4);
    }
    [[nodiscard]] bool ok() const { return static_cast<bool>(out_); }
    void write(const Call& c) {
        const std::vector<std::uint8_t> b = encode(c);
        const auto n = static_cast<std::uint32_t>(b.size());
        const std::lock_guard<std::mutex> lock(m_);
        out_.write(reinterpret_cast<const char*>(&n), 4);
        out_.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
        if (c.method == "Present") out_.flush();
    }

private:
    std::ofstream out_;
    std::mutex m_;
};

}  // namespace wse::recomp::d3dcap
