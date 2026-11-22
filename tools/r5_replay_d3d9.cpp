// R5a step 3 PC reference (`docs/RECOMP_PLAN.md` "R5 — Graphics"). LOCAL ONLY, Windows: replays a
// capture (`r4_run --capture`, `wse/recomp/d3d9_capture.h`) on the real d3d9.dll and d3dx9_43.dll
// of the user's PC and writes the chosen frames as PPM images: the reference our Vulkan device is
// compared with. Never run in CI; the capture and the frames hold retail data and are never
// committed.
//
//   r5_replay_d3d9 <capture> <install dir> <out dir> [--frames 1,30,60]   (default: every 10th
//                                                                           frame and the last)
//
// Calls are made by vtable slot (`com_vtables.h`), so no DirectX SDK headers are needed.
// Objects the capture names by the recording device's addresses map to the real ones; effect
// handles map to the real handles; recorded data stands in for the game's pointers; buffer and
// texture contents are written at Unlock. Skipped: queries (Get*, Check*, Enum*, Test*), DirectInput,
// AddRef/Release/QueryInterface, and ID3DXEffect::SetStateManager (the game's state manager is its
// own code; without it D3DX sets the states on the device directly, which draws the same).
// The device is created windowed on a hidden window at the captured back-buffer size.

#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "../engine/include/wse/recomp/com_vtables.h"
#include "../engine/include/wse/recomp/d3d9_capture.h"

namespace cap = wse::recomp::d3dcap;
namespace com = wse::recomp::com;
using u64 = std::uint64_t;

namespace {

const com::Interface* interfaces[] = {
    &com::kIDirect3D9, &com::kIDirect3DDevice9, &com::kIDirect3DSwapChain9, &com::kIDirect3DTexture9, &com::kIDirect3DCubeTexture9,
    &com::kIDirect3DVolumeTexture9, &com::kIDirect3DSurface9, &com::kIDirect3DVolume9, &com::kIDirect3DVertexBuffer9,
    &com::kIDirect3DIndexBuffer9, &com::kIDirect3DVertexDeclaration9, &com::kIDirect3DVertexShader9, &com::kIDirect3DPixelShader9,
    &com::kIDirect3DStateBlock9, &com::kIDirect3DQuery9, &com::kID3DXBuffer, &com::kID3DXEffectPool, &com::kID3DXEffect,
};

int slot_of(const std::string& iface, const std::string& method) {
    for (const com::Interface* i : interfaces) {
        if (iface != i->name) continue;
        for (std::size_t k = 0; k < i->count; ++k) {
            if (method == i->methods[k]) return static_cast<int>(k);
        }
    }
    return -1;
}

using Fn = u64(__stdcall*)(void*, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64);
using SetFloatFn = long(__stdcall*)(void*, u64, float);
using Free = u64(__stdcall*)(u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64);

bool is_query(const std::string& m) {
    static const std::set<std::string> keep = {"GetBackBuffer", "GetSurfaceLevel", "GetCubeMapSurface", "GetRenderTarget",
                                               "GetDepthStencilSurface", "GetSwapChain", "GetParameterByName", "GetParameterBySemantic",
                                               "GetParameter", "GetParameterElement", "GetTechniqueByName", "GetTechnique", "GetPass",
                                               "GetPassByName", "GetCurrentTechnique", "GetAnnotation", "GetAnnotationByName"};
    if (keep.count(m) != 0) return false;
    for (const char* p : {"Get", "Check", "Enum", "Test", "Validate", "Evict", "FindNextValidTechnique", "IsParameterUsed"}) {
        if (m.rfind(p, 0) == 0) return true;
    }
    return false;
}

// Bytes per row and rows of a level, as the recording device packed them (shim_d3d9.h pitch_of).
std::uint32_t bpp(std::uint32_t f) {
    switch (f) {
    case 28: case 50: case 41: return 1;
    case 23: case 24: case 25: case 26: case 51: case 80: case 111: return 2;
    case 113: case 115: return 8;
    case 116: return 16;
    default: return 4;
    }
}
bool is_dxt(std::uint32_t f) { return f >= 0x31545844U && f <= 0x35545844U && (f & 0xFFFFFF) == 0x545844; }
std::uint32_t tight_pitch(std::uint32_t f, std::uint32_t w) {
    if (is_dxt(f)) return std::max<std::uint32_t>(1, (w + 3) / 4) * (f == 0x31545844U ? 8 : 16);
    return std::max<std::uint32_t>(1, w) * bpp(f);
}
std::uint32_t tight_rows(std::uint32_t f, std::uint32_t h) { return is_dxt(f) ? std::max<std::uint32_t>(1, (h + 3) / 4) : std::max<std::uint32_t>(1, h); }

struct Replay {
    HMODULE d3d9 = nullptr;
    HMODULE d3dx = nullptr;
    HWND hwnd = nullptr;
    std::map<u64, void*> objects;              // captured address -> real object
    std::map<u64, u64> handles;                // captured effect handle -> real handle
    std::map<u64, std::uint8_t*> buffer_locks;  // captured buffer -> real locked pointer
    std::map<std::pair<u64, std::uint32_t>, std::pair<std::uint8_t*, int>> rect_locks;  // (texture, index) -> bits, pitch
    std::vector<std::vector<std::uint8_t>> keep;  // strings and blobs kept alive for the call
    void* device = nullptr;
    std::set<u64> save_frames;
    bool save_all_tenth = true;
    u64 frame = 0;
    std::string out_dir;
    int skipped = 0, failed = 0, calls = 0;

    // A LockRect / UnlockRect pair's key: the level, (face, level) of a cube, 0 for a surface.
    static std::uint32_t rect_key(const cap::Call& c) {
        if (c.iface == "IDirect3DSurface9") return 0;
        if (c.iface == "IDirect3DCubeTexture9") return static_cast<std::uint32_t>(c.args[0] * 64 + c.args[1]);
        return static_cast<std::uint32_t>(c.args[0]);
    }
    u64 map_value(u64 v) {
        if (const auto it = objects.find(v); it != objects.end()) return reinterpret_cast<u64>(it->second);
        if (const auto it = handles.find(v); it != handles.end()) return it->second;
        return v;
    }
    void* blob(const cap::Data& d, bool zero_end) {
        keep.push_back(d.bytes);
        if (zero_end) keep.back().push_back(0);
        if (keep.back().empty()) keep.back().push_back(0);
        return keep.back().data();
    }

    void save(u64 n) {
        void* bb = nullptr;
        auto vt = *reinterpret_cast<void***>(device);
        auto get_bb = reinterpret_cast<long(__stdcall*)(void*, unsigned, unsigned, int, void**)>(vt[slot_of("IDirect3DDevice9", "GetBackBuffer")]);
        if (get_bb(device, 0, 0, 0, &bb) < 0 || bb == nullptr) return;
        std::uint32_t desc[8] = {};
        auto svt = *reinterpret_cast<void***>(bb);
        reinterpret_cast<long(__stdcall*)(void*, void*)>(svt[slot_of("IDirect3DSurface9", "GetDesc")])(bb, desc);
        const std::uint32_t w = desc[6], h = desc[7], fmt = desc[0];
        void* off = nullptr;
        auto create_off = reinterpret_cast<long(__stdcall*)(void*, unsigned, unsigned, std::uint32_t, std::uint32_t, void**, void*)>(
            vt[slot_of("IDirect3DDevice9", "CreateOffscreenPlainSurface")]);
        if (create_off(device, w, h, fmt, 2 /* D3DPOOL_SYSTEMMEM */, &off, nullptr) < 0) return;
        auto get_rtd = reinterpret_cast<long(__stdcall*)(void*, void*, void*)>(vt[slot_of("IDirect3DDevice9", "GetRenderTargetData")]);
        if (get_rtd(device, bb, off) >= 0) {
            struct { int pitch; std::uint8_t* bits; } lr{};
            auto ovt = *reinterpret_cast<void***>(off);
            if (reinterpret_cast<long(__stdcall*)(void*, void*, void*, unsigned)>(ovt[slot_of("IDirect3DSurface9", "LockRect")])(off, &lr, nullptr, 0x10) >= 0) {
                char name[64];
                std::snprintf(name, sizeof(name), "frame_%04llu.ppm", static_cast<unsigned long long>(n));
                std::ofstream f(std::filesystem::path(out_dir) / name, std::ios::binary);
                f << "P6\n" << w << " " << h << "\n255\n";
                for (std::uint32_t y = 0; y < h; ++y) {
                    const std::uint8_t* row = lr.bits + static_cast<std::ptrdiff_t>(y) * lr.pitch;
                    for (std::uint32_t x = 0; x < w; ++x) {
                        const char rgb[3] = {static_cast<char>(row[4 * x + 2]), static_cast<char>(row[4 * x + 1]), static_cast<char>(row[4 * x])};
                        f.write(rgb, 3);
                    }
                }
                reinterpret_cast<long(__stdcall*)(void*)>(ovt[slot_of("IDirect3DSurface9", "UnlockRect")])(off);
            }
            reinterpret_cast<unsigned long(__stdcall*)(void*)>(ovt[2])(off);  // Release
        }
        reinterpret_cast<unsigned long(__stdcall*)(void*)>(svt[2])(bb);
    }

    // The level size of a texture (or a surface) for an UnlockRect copy.
    bool level_desc(void* obj, const std::string& iface, std::uint32_t level, std::uint32_t& fmt, std::uint32_t& w, std::uint32_t& h) {
        std::uint32_t desc[8] = {};
        auto vt = *reinterpret_cast<void***>(obj);
        long hr = -1;
        if (iface == "IDirect3DSurface9") {
            hr = reinterpret_cast<long(__stdcall*)(void*, void*)>(vt[slot_of(iface, "GetDesc")])(obj, desc);
        } else {
            hr = reinterpret_cast<long(__stdcall*)(void*, unsigned, void*)>(vt[slot_of(iface, "GetLevelDesc")])(obj, level, desc);
        }
        fmt = desc[0];
        w = desc[6];
        h = desc[7];
        return hr >= 0;
    }

    bool trace = std::getenv("WSE_REPLAY_TRACE") != nullptr;  // print each call before making it

    void run(const cap::Call& c) {
        ++calls;
        keep.clear();
        if (trace) {
            std::fprintf(stderr, "%d %s::%s\n", calls, c.iface.c_str(), c.method.c_str());
            std::fflush(stderr);
        }
        if (c.iface.rfind("IDirectInput", 0) == 0 || c.iface == "DINPUT8.dll" || c.method == "AddRef" || c.method == "Release" ||
            c.method == "QueryInterface" || c.method == "SetStateManager" || is_query(c.method)) {
            ++skipped;
            return;
        }
        u64 a[16];
        for (int i = 0; i < 16; ++i) a[i] = map_value(c.args[i]);
        u64 outs[16] = {};
        for (const cap::Data& d : c.data) {
            if (d.tag >= cap::kTagArg && d.tag < cap::kTagArg + 16) {
                const bool text = c.method.find("ByName") != std::string::npos || c.method.find("BySemantic") != std::string::npos ||
                                  c.method.find("FromFile") != std::string::npos;
                a[d.tag - cap::kTagArg] = reinterpret_cast<u64>(blob(d, text));
            } else if (d.tag >= cap::kTagOut && d.tag < cap::kTagOut + 16) {
                a[d.tag - cap::kTagOut] = reinterpret_cast<u64>(&outs[d.tag - cap::kTagOut]);
            } else if (d.tag == cap::kTagName) {
                a[0] = reinterpret_cast<u64>(blob(d, true));
            }
        }
        const std::string& m = c.method;
        // Pointers the game passed that the real call must not see (out structs and optional inputs).
        if (m == "D3DXCreateTextureFromFileExA") a[11] = a[12] = 0;      // pSrcInfo, pPalette
        if (m == "D3DXCreateCubeTextureFromFileExA") a[10] = a[11] = 0;
        if (m == "D3DXCreateTextureFromFileInMemoryEx") a[12] = a[13] = 0;
        if (m == "D3DXCreateEffectFromFileA") a[2] = a[3] = a[5] = a[7] = 0;
        if (m == "Present") a[0] = a[1] = a[2] = a[3] = 0;
        if (m.rfind("Create", 0) == 0 && c.iface == "IDirect3DDevice9" && m != "CreateVertexDeclaration" && m != "CreateVertexShader" &&
            m != "CreatePixelShader" && m != "CreateStateBlock" && m != "CreateQuery") {
            for (int i = 0; i < 16; ++i) {
                if (c.find(cap::kTagOut + static_cast<std::uint32_t>(i)) != nullptr && i + 1 < 16) a[i + 1] = 0;  // pSharedHandle
            }
        }
        if (m == "Begin" && c.iface == "ID3DXEffect") a[0] = reinterpret_cast<u64>(&outs[15]);
        std::uint8_t pp[64] = {};
        if ((c.iface == "IDirect3D9" && m == "CreateDevice") || (c.iface == "IDirect3DDevice9" && m == "Reset")) {
            const cap::Data* d = c.find(cap::kTagArg + (m == "Reset" ? 0 : 4));
            if (d == nullptr || d->bytes.size() < 64) {
                ++failed;
                return;
            }
            std::memcpy(pp, d->bytes.data(), 64);
            const u64 hw = reinterpret_cast<u64>(hwnd);
            const std::uint32_t one = 1, zero = 0, discard = 1, immediate = 0x80000000U;
            std::memcpy(pp + 24, &discard, 4);
            std::memcpy(pp + 32, &hw, 8);
            std::memcpy(pp + 40, &one, 4);
            std::memcpy(pp + 56, &zero, 4);
            std::memcpy(pp + 60, &immediate, 4);
            std::uint32_t w = 0, h = 0;
            std::memcpy(&w, pp, 4);
            std::memcpy(&h, pp + 4, 4);
            // As the recording device sized it (shim_d3d9.h reset_buffers): a zero field takes the
            // device's default, 1920 x 1080.
            if (w == 0) w = 1920;
            if (h == 0) h = 1080;
            std::memcpy(pp, &w, 4);
            std::memcpy(pp + 4, &h, 4);
            SetWindowPos(hwnd, nullptr, 0, 0, static_cast<int>(w), static_cast<int>(h), SWP_NOMOVE | SWP_NOZORDER);
            a[m == "Reset" ? 0 : 4] = reinterpret_cast<u64>(pp);
            if (m == "CreateDevice") a[2] = reinterpret_cast<u64>(hwnd);
        }
        // Texture and buffer contents: lock for real, copy the captured bytes, unlock.
        if (m == "Unlock" || m == "UnlockRect") {
            const auto it = objects.find(c.self);
            const cap::Data* d = c.find(cap::kTagLocked);
            if (it == objects.end()) {
                ++failed;
                return;
            }
            if (d != nullptr && m == "Unlock" && d->bytes.size() >= 8) {
                if (const auto lk = buffer_locks.find(c.self); lk != buffer_locks.end() && lk->second != nullptr) {
                    std::uint32_t n = 0;
                    std::memcpy(&n, d->bytes.data() + 4, 4);
                    std::memcpy(lk->second, d->bytes.data() + 8, std::min<std::size_t>(n, d->bytes.size() - 8));
                }
            } else if (d != nullptr && m == "UnlockRect" && d->bytes.size() >= 4) {
                const auto lk = rect_locks.find({c.self, rect_key(c)});
                std::uint32_t fmt = 0, w = 0, h = 0;
                const std::uint32_t level = c.iface == "IDirect3DCubeTexture9" ? static_cast<std::uint32_t>(c.args[1])
                                            : c.iface == "IDirect3DSurface9"   ? 0
                                                                               : static_cast<std::uint32_t>(c.args[0]);
                if (lk != rect_locks.end() && level_desc(it->second, c.iface, level, fmt, w, h)) {
                    const std::uint32_t tp = tight_pitch(fmt, w), rows = tight_rows(fmt, h);
                    for (std::uint32_t r = 0; r < rows && 4 + static_cast<std::size_t>(r + 1) * tp <= d->bytes.size(); ++r) {
                        std::memcpy(lk->second.first + static_cast<std::ptrdiff_t>(r) * lk->second.second, d->bytes.data() + 4 + static_cast<std::size_t>(r) * tp,
                                    std::min<std::uint32_t>(tp, static_cast<std::uint32_t>(lk->second.second)));
                    }
                }
            }
        }
        struct { int pitch; std::uint8_t* bits; } locked{};
        u64 lock_ptr = 0;
        if (m == "Lock") a[2] = reinterpret_cast<u64>(&lock_ptr);
        if (m == "LockRect") {
            const int at = c.iface == "IDirect3DSurface9" ? 0 : c.iface == "IDirect3DCubeTexture9" ? 2 : 1;
            a[at] = reinterpret_cast<u64>(&locked);
            a[at + 1] = 0;  // the whole level
        }
        if (m == "Present" && device != nullptr) {
            ++frame;
            if (save_frames.count(frame) != 0 || (save_all_tenth && frame % 10 == 0)) save(frame);
        }

        if (trace && c.iface == "d3dx9_43.dll") {
            for (int i = 0; i < 8; ++i) std::fprintf(stderr, "    a[%d] = %llx (captured %llx)\n", i, static_cast<unsigned long long>(a[i]),
                                                     static_cast<unsigned long long>(c.args[i]));
            for (const cap::Data& d : c.data) std::fprintf(stderr, "    tag %x, %zu bytes\n", d.tag, d.bytes.size());
            std::fflush(stderr);
        }
        u64 ret = 0;
        if (c.iface == "d3d9.dll" || c.iface == "d3dx9_43.dll") {
            FARPROC p = GetProcAddress(c.iface == "d3d9.dll" ? d3d9 : d3dx, m.c_str());
            if (p == nullptr) {
                ++failed;
                return;
            }
            ret = reinterpret_cast<Free>(reinterpret_cast<void*>(p))(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11],
                                                                      a[12], a[13], a[14], a[15]);
            if (m == "Direct3DCreate9") {
                std::uint64_t captured = 0;
                if (const cap::Data* d = c.find(cap::kTagOut); d != nullptr && d->bytes.size() == 8) std::memcpy(&captured, d->bytes.data(), 8);
                objects[captured != 0 ? captured : c.ret] = reinterpret_cast<void*>(ret);
                return;
            }
        } else {
            const auto it = objects.find(c.self);
            const int slot = slot_of(c.iface, m);
            if (it == objects.end() || slot < 0) {
                ++failed;
                return;
            }
            void* obj = it->second;
            auto vt = *reinterpret_cast<void***>(obj);
            if (m == "SetFloat" && c.iface == "ID3DXEffect") {
                float f = 0;
                std::memcpy(&f, &c.xmm[1], 4);
                ret = static_cast<u64>(reinterpret_cast<SetFloatFn>(vt[slot])(obj, a[0], f));
            } else {
                ret = reinterpret_cast<Fn>(vt[slot])(obj, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13],
                                                     a[14], a[15]);
            }
            if (static_cast<long>(ret) < 0 && c.iface != "ID3DXEffect") ++failed;
            if (trace && static_cast<long>(ret) < 0) {
                std::fprintf(stderr, "    failed: hr %08lx\n", static_cast<unsigned long>(ret));
                if (m == "CreateDevice") {
                    for (int i = 0; i < 6; ++i) std::fprintf(stderr, "    a[%d] = %llx\n", i, static_cast<unsigned long long>(a[i]));
                    std::uint32_t w[16];
                    std::memcpy(w, pp, 64);
                    for (int i = 0; i < 16; ++i) std::fprintf(stderr, "    pp[%d] = %08x\n", i * 4, w[i]);
                }
                std::fflush(stderr);
            }
            if (m == "Lock") buffer_locks[c.self] = reinterpret_cast<std::uint8_t*>(lock_ptr);
            if (m == "LockRect") rect_locks[{c.self, rect_key(c)}] = {locked.bits, locked.pitch};
        }
        // Map what the call created, and effect handles it returned.
        for (int i = 0; i < 16; ++i) {
            const cap::Data* d = c.find(cap::kTagOut + static_cast<std::uint32_t>(i));
            if (d == nullptr || d->bytes.size() != 8) continue;
            std::uint64_t captured = 0;
            std::memcpy(&captured, d->bytes.data(), 8);
            if (captured != 0 && outs[i] != 0) objects[captured] = reinterpret_cast<void*>(outs[i]);
            if (m == "CreateDevice" && outs[i] != 0) device = reinterpret_cast<void*>(outs[i]);
        }
        if (c.iface == "ID3DXEffect" && (m.rfind("Get", 0) == 0) && c.ret != 0 && ret != 0) handles[c.ret] = ret;
    }
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <capture> <install dir> <out dir> [--frames 1,30,60]\n", argv[0]);
        return 2;
    }
    std::ifstream in(argv[1], std::ios::binary);
    std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::vector<cap::Call> calls;
    const bool whole = cap::decode(bytes, calls);
    std::printf("%zu calls%s\n", calls.size(), whole ? "" : " (the capture ends mid-record; replaying the whole ones)");
    Replay r;
    r.out_dir = argv[3];
    std::filesystem::create_directories(r.out_dir);
    for (int i = 4; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "--frames") {
            r.save_all_tenth = false;
            std::string list = argv[++i];
            for (std::size_t p = 0; p < list.size();) {
                const std::size_t q = list.find(',', p);
                r.save_frames.insert(std::strtoull(list.substr(p, q - p).c_str(), nullptr, 10));
                p = q == std::string::npos ? list.size() : q + 1;
            }
        }
    }
    SetCurrentDirectoryA(argv[2]);  // the game's working directory: its relative paths resolve
    r.d3d9 = LoadLibraryA("d3d9.dll");
    r.d3dx = LoadLibraryA("d3dx9_43.dll");
    if (r.d3d9 == nullptr || r.d3dx == nullptr) {
        std::fprintf(stderr, "d3d9.dll or d3dx9_43.dll is missing\n");
        return 1;
    }
    WNDCLASSA wc{};
    wc.lpfnWndProc = DefWindowProcA;
    wc.hInstance = GetModuleHandleA(nullptr);
    wc.lpszClassName = "wse2_replay";
    RegisterClassA(&wc);
    r.hwnd = CreateWindowA("wse2_replay", "wse2 replay", WS_OVERLAPPEDWINDOW, 0, 0, 640, 480, nullptr, nullptr, wc.hInstance, nullptr);
    for (const cap::Call& c : calls) r.run(c);
    if (r.device != nullptr) r.save(r.frame + 1);  // the last frame drawn
    std::printf("replayed %d calls: %d skipped (queries, input), %d failed; %llu frames presented\n", r.calls, r.skipped, r.failed,
                static_cast<unsigned long long>(r.frame));
    return 0;
}
