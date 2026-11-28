// The R4 startup runner. See `r4_startup.h` and `docs/RECOMP_PLAN.md` ("R4 design", "Phone run").

#include "r4_startup.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <malloc.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <ucontext.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <unordered_map>
#include <condition_variable>
#include <cctype>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "code_loader.h"
#include "discover.h"
#include "lift_r1.h"
#include "recomp_llvm.h"
#include "wse/platform/vfs.h"
#include "wse/recomp/kernel.h"
#include "wse/recomp/pdb_reader.h"
#include "wse/recomp/pe_image.h"
#include "wse/recomp/pgo_profile.h"
#include "wse/recomp/r0_inputs.h"
#include "wse/recomp/runtime.h"
#include "wse/recomp/seh.h"
#include "wse/recomp/shim_advapi.h"
#include "wse/recomp/shim_crt.h"
#include "wse/recomp/shim_d3d9.h"
#include "wse/recomp/shim_d3dx.h"
#include "wse/recomp/shim_files.h"
#include "wse/recomp/shim_fmod_steam.h"
#include "wse/recomp/shim_memory.h"
#include "wse/recomp/shim_modules.h"
#include "wse/recomp/shim_net.h"
#include "wse/recomp/shim_strings.h"
#include "wse/recomp/shim_system.h"
#include "wse/recomp/shim_time.h"
#include "wse/recomp/shim_user32.h"
#include "wse/recomp/teb.h"
#include "wse/recomp/threadpool.h"
#include "wse/recomp/threads.h"

#ifndef MAP_FIXED_NOREPLACE
#define MAP_FIXED_NOREPLACE 0x100000
#endif

namespace wse::recomp {
namespace {

namespace fs = std::filesystem;

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// The report sink of the run (one run per process). Text may hold several lines.
std::function<void(const std::string&)> g_log;

[[gnu::format(printf, 1, 2)]] void out(const char* fmt, ...) {
    char buf[2048];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (g_log) {
        g_log(buf);
    } else {
        std::fputs(buf, stdout);
        std::fflush(stdout);
    }
}

std::uint64_t fnv64(const std::vector<std::uint8_t>& b) {
    std::uint64_t h = 0xCBF29CE484222325ULL;
    for (const std::uint8_t x : b) {
        h = (h ^ x) * 0x100000001B3ULL;
    }
    return h;
}

// What a translated batch depends on: the image bytes, the options and the translator build.
std::string cache_key(const std::vector<std::uint8_t>& file, int opt, std::size_t batch, const std::string& translator) {
    char buf[96];
    std::snprintf(buf, sizeof(buf), "%016llx %zu opt%d batch%zu lifter ", static_cast<unsigned long long>(fnv64(file)),
                  file.size(), opt, batch);
    return buf + translator;
}

struct Image {
    std::string tag;
    const std::vector<std::uint8_t>* file = nullptr;
    PeImage img;
    std::uint8_t* mapped = nullptr;
};

// Maps an image at its preferred base, read-write (guest bytes are data: only translated code runs).
bool map_image(Image& m, std::string* err) {
    if (m.file == nullptr || m.file->empty()) {
        *err = "missing";
        return false;
    }
    if (!parse_pe(m.file->data(), m.file->size(), m.img, err)) return false;
    const std::size_t size = (m.img.size_of_image + 0xFFFU) & ~std::size_t{0xFFF};
    void* at = mmap(reinterpret_cast<void*>(m.img.image_base), size, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
    if (at != reinterpret_cast<void*>(m.img.image_base)) {
        if (at != MAP_FAILED) munmap(at, size);
        *err = "its preferred base is taken";
        return false;
    }
    m.mapped = static_cast<std::uint8_t*>(at);
    std::size_t relocs = 0;
    return map_pe_sections(m.file->data(), m.file->size(), m.img, m.mapped) &&
           apply_pe_relocations(m.mapped, m.img.size_of_image, m.img.image_base, &relocs, err);
}

// Entries of one image: PDB publics and `.pdata` (collect_function_entries), the exports and the
// entry point, relocated pointers into .text, offset-table handlers, then direct call targets to a
// fixed point, and code addresses taken with lea.
std::set<std::uint64_t> entries_of(const Image& m, const ImageView& view, const std::vector<PdbPublic>& publics) {
    std::set<std::uint64_t> entries = collect_function_entries(view, m.img, m.mapped, publics);
    std::deque<std::uint64_t> work;
    PeExports ex;
    std::string err;
    if (read_pe_exports(m.mapped, m.img.size_of_image, ex, &err)) {
        for (const PeExport& e : ex.entries) {
            const std::uint64_t va = m.img.image_base + e.rva;
            if (e.forwarder.empty() && view.in_text(va) && entries.insert(va).second) work.push_back(va);
        }
    }
    if (m.img.entry_rva != 0 && entries.insert(m.img.image_base + m.img.entry_rva).second) {
        work.push_back(m.img.image_base + m.img.entry_rva);
    }
    // Code reached only through a stored pointer (CRT init tables, callbacks, vtables; leaf
    // functions have no .pdata entry, and the PDB's publics miss static functions such as
    // dynamic initializers): every relocated pointer into .text.
    {
        std::size_t added = 0;
        for (const std::uint64_t v : read_pe_dir64_values(m.mapped, m.img.size_of_image)) {
            if (view.in_text(v) && entries.insert(v).second) {
                work.push_back(v);
                ++added;
            }
        }
        out("%s: %zu entries from relocated pointers into .text\n", m.tag.c_str(), added);
    }
    // Code reached only through a run-time table of 16-bit offsets (LuaJIT's bytecode handlers).
    const OffsetTables ot = find_offset_table_entries(view, m.img, entries);
    for (const OffsetTables::Table& t : ot.tables) {
        out("%s: offset table at 0x%llx, %zu entries from 0x%llx\n", m.tag.c_str(), static_cast<unsigned long long>(t.table),
            t.count, static_cast<unsigned long long>(t.function));
    }
    for (const std::uint64_t e : ot.entries) {
        if (entries.insert(e).second) work.push_back(e);
    }
    Discoverer disc(view);
    while (!work.empty()) {
        const std::uint64_t e = work.front();
        work.pop_front();
        for (const std::uint64_t t : disc.discover(e, entries).call_targets) {
            if (view.in_text(t) && entries.insert(t).second) work.push_back(t);
        }
    }
    // Code whose address a function takes with a rip-relative `lea` (callbacks, tables filled at
    // run time) and that no .pdata range covers, or that lies in an interpreter with an offset table.
    std::set<std::uint64_t> interpreters;
    for (const OffsetTables::Table& t : ot.tables) interpreters.insert(t.function);
    const std::set<std::uint64_t> taken = find_code_address_entries(view, m.img, entries, &interpreters);
    entries.insert(taken.begin(), taken.end());
    out("%s: %zu entries from code addresses taken with lea\n", m.tag.c_str(), taken.size());
    return entries;
}

struct Loaded {
    std::vector<std::unique_ptr<LoadedCode>> code;
    std::vector<std::pair<const FnRow*, std::size_t>> tables;
    std::vector<std::pair<const FastFnRow*, std::size_t>> fast_tables;
    std::size_t functions = 0;
};

// Loads one translated batch: on Android from a memfd, else with our own ELF loader (an app-data
// file may not be mapped executable there); elsewhere with dlopen of the file.
bool load_batch(const fs::path& so, LoadedCode& code, std::string* err) {
#if defined(__ANDROID__)
    std::ifstream in(so, std::ios::binary);
    const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (bytes.empty()) {
        *err = "empty or unreadable";
        return false;
    }
    std::string memfd_err;
    if (code.load_memfd(bytes, so.filename().string(), &memfd_err)) return true;
    std::string own_err;
    if (code.load_own(bytes, &own_err)) return true;
    *err = "memfd: " + memfd_err + "; own loader: " + own_err;
    return false;
#else
    // dlopen keeps a library loaded by path, so read-only is not needed; a later retranslation
    // must be able to replace the file.
    void* h = dlopen(so.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (h == nullptr) {
        const char* e = dlerror();
        *err = e != nullptr ? e : "dlopen failed";
        return false;
    }
    code.adopt_handle(h);
    return true;
#endif
}

// Translation memory per call instruction in a batch, at -O2. Measured in WSL on the user's
// v1.1.5.6 exe (2026-11-23, one batch at a time): window 38 (4,001 calls, most in one resumable
// function with 3,755) took 2.53 GB above the baseline and 874 s; window 8 (1,860 calls) 0.77 GB
// and 155 s. Calls predict a batch's cost (correlation 0.97 with the translated size over all
// 80 batches; instructions only 0.43): in a resumable function every call's return address is a
// resume point. The larger figure, about 0.65 MB per call, is used.
constexpr std::size_t kBytesPerCall = 680 * 1024;

// Gives memory a finished batch freed back to the system. The allocator otherwise keeps it per
// thread: in WSL, two large batches translated one after the other on different threads held
// 3.5 GB at the end instead of the 2.6 GB of the larger one (2026-11-23), and the next module's
// memory budget was computed from what was left.
void release_free_memory() {
#if defined(__ANDROID__)
#ifdef M_PURGE
    (void)mallopt(M_PURGE, 0);
#endif
#elif defined(__GLIBC__)
    (void)malloc_trim(0);
#endif
}

// What the system reports as available memory (Linux and Android: MemAvailable), or 1 GiB.
std::size_t available_memory_bytes() {
    std::ifstream in("/proc/meminfo");
    std::string key;
    std::size_t kib = 0;
    std::string unit;
    while (in >> key >> kib >> unit) {
        if (key == "MemAvailable:") return kib * 1024;
    }
    return std::size_t{1} << 30;
}

// Translates `m` in batches on `jobs` worker threads (a batch whose `.so` and key file match is
// reused, so an interrupted run resumes) and loads every batch.
bool translate(Image& m, const std::vector<PdbPublic>& publics, const StartupOptions& o, Loaded& out_loaded) {
    const auto t0 = std::chrono::steady_clock::now();
    const fs::path so_dir = o.so_dir;
    const std::size_t batch = o.batch_kib;
    const auto [text_b, text_e] = text_range(m.img);
    ImageView view(m.mapped, m.img.image_base, m.img.size_of_image, text_b, text_e);
    // Hot-subsystem PDB patterns (fallback when no PGO profile is loaded): core per-frame inner
    // loops optimize at `o.opt` (-O2); cold batches optimize at `o.cold_opt` (-O1). `lua5164`
    // stays hot. When a PGO profile is provided, the top ~5% hottest functions move into dedicated
    // `-O3` batches (`<tag>_hot_<n>`) and the remaining ~95% compile at `-O1`.
    std::set<std::uint64_t> hot_vas;
    auto is_hot_symbol = [](const std::string& name) -> bool {
        static constexpr const char* kHot[] = {
            "btCollision", "btDiscreteDynamics", "btSequentialImpulse", "btDbvt", "btGjk",
            "rglSkeleton", "rglManifold", "rglMatrix", "rglVector", "rglQuaternion",
            "lua", "WSE",
        };
        for (const char* p : kHot) {
            if (name.find(p) != std::string::npos) return true;
        }
        return false;
    };
    for (const PdbPublic& p : publics) {
        if (p.section == 0 || p.section > m.img.sections.size()) continue;
        if (is_hot_symbol(p.name)) {
            hot_vas.insert(m.img.image_base + m.img.sections[p.section - 1].virtual_address + p.offset);
        }
    }
    const int cold_opt = std::clamp(o.cold_opt, 0, o.opt);
    const int pgo_hot_opt = std::clamp(o.pgo_hot_opt, o.opt, 3);
    const std::string tso_tag = (o.strict_tso || m.tag == "lua5164") ? " tso=strict" : " tso=relaxed";
    const std::string pgo_hot_key = cache_key(*m.file, pgo_hot_opt, batch, translator_fingerprint()) + tso_tag;
    const std::string hot_key = cache_key(*m.file, o.opt, batch, translator_fingerprint()) + tso_tag;
    const std::string cold_key = cache_key(*m.file, cold_opt, batch, translator_fingerprint()) + tso_tag;
    const std::set<std::uint64_t> entries = entries_of(m, view, publics);
    std::set<std::uint64_t> pgo_hot_vas;
    if (m.tag == "exe" && !o.profile_in.empty()) {
        PgoProfile prof;
        std::string perr;
        if (load_pgo_profile(o.profile_in, prof, &perr)) {
            for (const std::uint64_t va : select_pgo_hot_vas(prof, entries.size())) {
                if (entries.count(va) != 0) pgo_hot_vas.insert(va);
            }
            if (!pgo_hot_vas.empty()) {
                out("%s: PGO profile loaded (%llu samples, %zu hot functions -> -O%d, rest -> -O%d)\n",
                    m.tag.c_str(), static_cast<unsigned long long>(prof.total_samples), pgo_hot_vas.size(),
                    pgo_hot_opt, cold_opt);
            }
        } else {
            out("%s: PGO profile ignored (%s)\n", m.tag.c_str(), perr.c_str());
        }
    }
    const EhContinuations eh = find_eh_continuations(view, m.img, entries);
    std::vector<PeImport> imports;
    std::string err;
    (void)read_pe_imports(m.mapped, m.img.size_of_image, m.img.image_base, imports, &err);
    ImageInfo info;
    info.base = m.img.image_base;
    info.relaxed_tso = !o.strict_tso && m.tag != "lua5164";
    for (const PeSection& s : m.img.sections) {
        // IMAGE_SCN_MEM_WRITE == 0x80000000: non-writable sections (.rdata, .pdata, .rsrc).
        if ((s.characteristics & 0x80000000U) == 0U && s.virtual_size != 0U) {
            const std::uint64_t sb = m.img.image_base + s.virtual_address;
            info.readonly_ranges.push_back({sb, sb + s.virtual_size});
        }
    }
    auto is_sync_import = [](const std::string& name) -> bool {
        static constexpr const char* kSync[] = {
            "CriticalSection", "SRWLock", "Mutex", "Semaphore", "Event", "Interlocked",
            "WaitFor", "ConditionVariable", "Wake", "Thread", "Tls", "Fls", "Yield", "SwitchToThread",
        };
        for (const char* p : kSync) {
            if (name.find(p) != std::string::npos) return true;
        }
        return false;
    };
    for (const PeImport& imp : imports) {
        info.iat_begin = info.iat_begin == 0 ? imp.slot_va : std::min(info.iat_begin, imp.slot_va);
        info.iat_end = std::max(info.iat_end, imp.slot_va + 8);
        if (is_sync_import(imp.name)) info.sync_iat_slots.insert(imp.slot_va);
    }
    for (const PdbPublic& p : publics) {
        if (p.section == 0 || p.section > m.img.sections.size()) continue;
        static constexpr const char* kStrictSym[] = {
            "Mutex", "Lock", "Atomic", "Spin", "Thread", "Pool", "Alloc", "malloc",
            "free", "Heap", "CriticalSection", "Condition", "Semaphore", "lua", "lj_", "tbb", "omp",
        };
        for (const char* k : kStrictSym) {
            if (p.name.find(k) != std::string::npos) {
                info.strict_tso_fns.insert(m.img.image_base + m.img.sections[p.section - 1].virtual_address + p.offset);
                break;
            }
        }
    }
    for (const PeFunctionRange& r : m.img.functions) {
        const std::uint8_t f = r.unwind < m.img.size_of_image ? static_cast<std::uint8_t>(m.mapped[r.unwind] >> 3) : 0;
        if ((f & 0x3) != 0 && (f & 0x4) == 0) info.resumable.insert(m.img.image_base + r.begin);
    }
    const std::vector<std::uint64_t> all(entries.begin(), entries.end());
    // Translated without optimisation: the game's settings file reader and writer. Each is one
    // giant resumable function (rglConfig::load 15,236 instructions with 3,755 calls, ::save 15,255
    // with 3,470 in v1.1.5.6) and together they took 25 minutes of one PC core at -O2 (2026-11-23),
    // most of a first translation; they run when the game starts and when settings are saved, never
    // per frame. Found by their PDB names, so any WSE2 build with a matching PDB gets the same.
    std::set<std::uint64_t> light;
    for (const PdbPublic& p : publics) {
        if (p.section == 0 || p.section > m.img.sections.size()) continue;
        if (p.name.rfind("?load@rglConfig@@", 0) == 0 || p.name.rfind("?save@rglConfig@@", 0) == 0) {
            light.insert(m.img.image_base + m.img.sections[p.section - 1].virtual_address + p.offset);
        }
    }
    if (!light.empty()) out("%s: %zu functions translated without optimisation (rglConfig load/save)\n", m.tag.c_str(), light.size());
    // How each function was discovered (jump tables, their targets, unresolved jumps), plus
    // fast-ABI eligibility and small-leaf cloning metadata (computed first so PGO hot batches
    // can split cleanly by call weight).
    std::map<std::uint64_t, std::array<std::size_t, 4>> sig;  // tables, targets, unresolved, function-bounded tables
    std::map<std::uint64_t, std::size_t> cost_of;               // call instructions per function (batch cost)
    std::set<std::uint64_t> small_leaf_vas;
    std::map<std::uint64_t, std::vector<std::uint64_t>> ext_call_targets;
    {
        Discoverer d0(view);
        d0.set_function_ranges(&m.img.functions);
        for (const std::uint64_t e : all) {
            const auto cont = eh.by_function.find(e);
            const DFunction f = d0.discover(e, entries, cont == eh.by_function.end() ? nullptr : &cont->second);
            std::array<std::size_t, 4> v{f.jump_tables.size(), 0, f.unresolved_indirect_jumps.size(), 0};
            for (const JumpTable& jt : f.jump_tables) {
                v[1] += jt.targets.size();
                v[3] += jt.by_function_range ? 1 : 0;
            }
            sig[e] = v;
            std::size_t calls = 0;
            for (const auto& [va, ins] : f.insns) {
                (void)va;
                if (ins.ins.mnemonic == ZYDIS_MNEMONIC_CALL) ++calls;
            }
            cost_of[e] = calls + 1;
            if (light.count(e) == 0 && is_fast_abi_eligible(f, info)) {
                info.fast_abi_vas.insert(e);
                if (f.insns.size() <= 24) small_leaf_vas.insert(e);
            }
            if (!f.call_targets.empty()) {
                ext_call_targets[e].assign(f.call_targets.begin(), f.call_targets.end());
            }
        }
    }
    out("%s: %zu fast-ABI functions (%zu small inlineable leaves)\n", m.tag.c_str(), info.fast_abi_vas.size(),
        small_leaf_vas.size());
    // Batches are address windows of `batch` KiB of .text (named by window), except when a PGO
    // profile promotes the top ~5% hottest functions into dedicated `<tag>_hot_<k>` batches at -O3.
    const std::uint64_t window = static_cast<std::uint64_t>(batch) * 1024;
    std::vector<std::string> batch_name;
    std::vector<std::vector<std::uint64_t>> batch_vas;
    std::vector<int> batch_opt;
    {
        std::map<std::uint64_t, std::vector<std::uint64_t>> win_map;
        std::vector<std::uint64_t> hot_list;
        for (const std::uint64_t va : all) {
            const bool giant_resumable = info.resumable.count(va) != 0 && cost_of[va] > 1000U;
            if (pgo_hot_vas.count(va) != 0 && light.count(va) == 0 && !giant_resumable) {
                hot_list.push_back(va);
            } else {
                win_map[(va - text_b) / window].push_back(va);
            }
        }
        if (!hot_list.empty()) {
            constexpr std::size_t kHotBatchMaxFns = 16;
            constexpr std::size_t kHotBatchMaxCalls = 600;
            std::vector<std::uint64_t> cur;
            std::size_t cur_calls = 0;
            std::size_t idx = 0;
            for (const std::uint64_t va : hot_list) {
                const std::size_t w = cost_of[va];
                if (!cur.empty() && (cur.size() >= kHotBatchMaxFns || cur_calls + w > kHotBatchMaxCalls)) {
                    batch_name.push_back(m.tag + "_hot_" + std::to_string(idx++));
                    batch_vas.push_back(std::move(cur));
                    batch_opt.push_back(pgo_hot_opt);
                    cur.clear();
                    cur_calls = 0;
                }
                cur.push_back(va);
                cur_calls += w;
            }
            if (!cur.empty()) {
                batch_name.push_back(m.tag + "_hot_" + std::to_string(idx++));
                batch_vas.push_back(std::move(cur));
                batch_opt.push_back(pgo_hot_opt);
            }
        }
        for (auto& [w, vec] : win_map) {
            batch_name.push_back(m.tag + "_w" + std::to_string(w));
            batch_vas.push_back(std::move(vec));
            batch_opt.push_back(pgo_hot_vas.empty() ? o.opt : cold_opt);
        }
    }
    std::vector<std::size_t> todo;
    const std::size_t n_batches = batch_name.size();
    std::vector<fs::path> sos(n_batches);
    std::vector<std::size_t> weight(n_batches, 0);
    std::size_t cold_batches = 0;
    for (std::size_t b = 0; b < n_batches; ++b) {
        bool has_hot = !pgo_hot_vas.empty() ? (batch_opt[b] == pgo_hot_opt) : hot_vas.empty();
        for (const std::uint64_t va : batch_vas[b]) {
            weight[b] += cost_of[va];
            if (pgo_hot_vas.empty() && !has_hot && hot_vas.count(va) != 0) has_hot = true;
        }
        if (pgo_hot_vas.empty() && !has_hot && cold_opt != o.opt) {
            batch_opt[b] = cold_opt;
            ++cold_batches;
        } else if (!pgo_hot_vas.empty() && batch_opt[b] == cold_opt) {
            ++cold_batches;
        }
    }
    std::vector<std::vector<std::uint64_t>> batch_inline_leaves(n_batches);
    for (std::size_t b = 0; b < n_batches; ++b) {
        if (batch_opt[b] < 2 || weight[b] > 1200U) continue;
        std::set<std::uint64_t> in_batch(batch_vas[b].begin(), batch_vas[b].end());
        std::set<std::uint64_t> leaves;
        for (const std::uint64_t va : batch_vas[b]) {
            const auto it = ext_call_targets.find(va);
            if (it == ext_call_targets.end()) continue;
            for (const std::uint64_t t : it->second) {
                if (in_batch.count(t) == 0 && small_leaf_vas.count(t) != 0) {
                    leaves.insert(t);
                    if (leaves.size() >= 64U) break;
                }
            }
            if (leaves.size() >= 64U) break;
        }
        batch_inline_leaves[b].assign(leaves.begin(), leaves.end());
    }
    auto batch_hash = [&](std::size_t b) {
        std::uint64_t h = 0xCBF29CE484222325ULL;
        for (const std::uint64_t va : batch_vas[b]) {
            h = (h ^ va) * 0x100000001B3ULL;
            for (const std::size_t x : sig[va]) h = (h ^ x) * 0x100000001B3ULL;
            h = (h ^ static_cast<std::uint64_t>(info.fast_abi_vas.count(va))) * 0x100000001B3ULL;
        }
        for (const std::uint64_t leaf : batch_inline_leaves[b]) {
            h = (h ^ leaf) * 0x100000001B3ULL;
        }
        return std::to_string(h);
    };
    auto list_hash = [&](std::size_t b) {
        std::uint64_t h = 0xCBF29CE484222325ULL;
        for (const std::uint64_t va : batch_vas[b]) h = (h ^ va) * 0x100000001B3ULL;
        return std::to_string(h);
    };
    auto legacy_ok = [&](std::size_t b) {
        for (const std::uint64_t va : batch_vas[b]) {
            if (sig[va][3] != 0) return false;
        }
        return true;
    };
    if (cold_batches != 0) {
        out("%s: %zu/%zu cold batches scheduled at -O%d (hot batches at -O%d)\n", m.tag.c_str(), cold_batches,
            n_batches, cold_opt, pgo_hot_vas.empty() ? o.opt : pgo_hot_opt);
    }
    // Files of batches that no longer exist (a different batch size, an older translator that
    // batched differently, an interrupted link's object file) are removed, so old translations do
    // not pile up in app storage.
    {
        std::set<std::string> keep;
        for (const std::string& n : batch_name) {
            keep.insert(n + ".so");
            keep.insert(n + ".key");
        }
        const std::string prefix_w = m.tag + "_w";
        const std::string prefix_h = m.tag + "_hot_";
        std::error_code ec;
        std::size_t pruned = 0;
        for (const fs::directory_entry& de : fs::directory_iterator(so_dir, ec)) {
            const std::string f = de.path().filename().string();
            if ((f.rfind(prefix_w, 0) != 0 && f.rfind(prefix_h, 0) != 0) || keep.count(f) != 0) continue;
            const std::string ext = de.path().extension().string();
            if (ext != ".so" && ext != ".key" && ext != ".o") continue;
            std::error_code rec;
            if (fs::remove(de.path(), rec)) ++pruned;
        }
        if (pruned != 0) out("%s: removed %zu files of batches that no longer exist\n", m.tag.c_str(), pruned);
    }
    auto key_for_opt = [&](int opt_lv) -> const std::string& {
        if (opt_lv == pgo_hot_opt && pgo_hot_opt != o.opt) return pgo_hot_key;
        if (opt_lv == o.opt) return hot_key;
        return cold_key;
    };
    for (std::size_t b = 0; b < n_batches; ++b) {
        sos[b] = so_dir / (batch_name[b] + ".so");
        const fs::path key_file = so_dir / (batch_name[b] + ".key");
        std::ifstream in(key_file);
        std::string line;
        std::string fns;
        std::error_code ec;
        const bool have = std::getline(in, line) && fs::exists(sos[b], ec);
        const bool legacy = have && (!std::getline(in, fns) || fns == list_hash(b));
        const bool same_fns = legacy ? legacy_ok(b) : fns == batch_hash(b);
        const std::string& bkey = key_for_opt(batch_opt[b]);
        if (o.retranslate || !have || !same_fns || (line != bkey && !o.reuse)) {
            todo.push_back(b);
        } else if (legacy) {
            in.close();
            std::ofstream(key_file) << line << "\n" << batch_hash(b) << "\n";
        }
    }
    // Largest first: the costly batches start early instead of holding up the end (on the phone
    // the three largest stalled the run at 77/80, 2026-11-21), and the memory budget sees them
    // while it can still run them apart.
    std::stable_sort(todo.begin(), todo.end(), [&](std::size_t a, std::size_t b) { return weight[a] > weight[b]; });
    // Memory budget: the call instructions of the batches being translated at once stay under half
    // of what the system reports as available (kBytesPerCall each). A batch larger than the whole
    // budget still runs, alone.
    const std::size_t budget = std::max<std::size_t>(1, available_memory_bytes() / 2 / kBytesPerCall);
    std::size_t largest = 0;
    for (const std::size_t b : todo) largest = std::max(largest, weight[b]);
    {
        std::vector<std::size_t> order(n_batches);
        for (std::size_t b = 0; b < n_batches; ++b) order[b] = b;
        std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t c) { return weight[a] > weight[c]; });
        std::string heaviest;
        for (std::size_t k = 0; k < order.size() && k < 3; ++k) {
            heaviest += (k == 0 ? "" : ", ") + batch_name[order[k]] + " " + std::to_string(weight[order[k]]);
        }
        out("%s: costliest batches (calls): %s\n", m.tag.c_str(), heaviest.c_str());
    }
    out("%s: %zu functions, %zu resumable, catch continuations in %zu functions; %zu batches (%zu KiB windows), %zu to translate on %u threads (largest %zu calls, budget %zu calls at once)\n",
        m.tag.c_str(), entries.size(), info.resumable.size(), eh.by_function.size(), n_batches, batch, todo.size(), o.jobs,
        largest, budget);
    if (o.progress) o.progress(m.tag, 0, todo.size());
    std::atomic<std::size_t> next{0};
    std::atomic<std::size_t> done{0};
    std::atomic<bool> failed{false};
    std::mutex print_m;
    std::mutex budget_m;
    std::condition_variable budget_cv;
    std::size_t in_flight = 0;
    auto worker = [&]() {
        Discoverer disc(view);
        disc.set_function_ranges(&m.img.functions);
        for (std::size_t i = next++; i < todo.size() && !failed; i = next++) {
            const std::size_t b = todo[i];
            {
                std::unique_lock<std::mutex> lock(budget_m);
                budget_cv.wait(lock, [&] { return in_flight == 0 || in_flight + weight[b] <= budget || failed; });
                in_flight += weight[b];
            }
            struct Release {
                std::mutex& m;
                std::condition_variable& cv;
                std::size_t& in_flight;
                std::size_t w;
                ~Release() {
                    release_free_memory();
                    {
                        const std::lock_guard<std::mutex> lock(m);
                        in_flight -= w;
                    }
                    cv.notify_all();
                }
            } release{budget_m, budget_cv, in_flight, weight[b]};
            const std::vector<std::uint64_t>& vas = batch_vas[b];
            R1Translator tr(info);
            tr.set_light(light);
            std::string why;
            std::string e = tr.init_error();
            tr.declare(vas);
            for (const std::uint64_t leaf : batch_inline_leaves[b]) {
                if (e.empty()) tr.add_inline_leaf(disc.discover(leaf, entries, nullptr));
            }
            for (const std::uint64_t va : vas) {
                if (info.fast_abi_vas.count(va) == 0) continue;
                const auto cont = eh.by_function.find(va);
                if (e.empty()) (void)tr.lift(disc.discover(va, entries, cont == eh.by_function.end() ? nullptr : &cont->second), &why);
            }
            for (const std::uint64_t va : vas) {
                if (info.fast_abi_vas.count(va) != 0) continue;
                const auto cont = eh.by_function.find(va);
                if (e.empty()) (void)tr.lift(disc.discover(va, entries, cont == eh.by_function.end() ? nullptr : &cont->second), &why);
            }
            std::vector<char> obj;
            std::string log;
            const fs::path obj_path = so_dir / (batch_name[b] + ".o");
            if (e.empty() && !tr.emit_object(batch_opt[b], obj, nullptr, &e) && e.empty()) e = "emit failed";
            if (e.empty()) {
                std::ofstream(obj_path, std::ios::binary).write(obj.data(), static_cast<std::streamsize>(obj.size()));
                std::error_code ec;
                fs::remove(sos[b], ec);  // a stale copy may be read-only
                if (!link_shared_object({obj_path.string()}, sos[b].string(), &log)) e = "link failed: " + log;
                fs::remove(obj_path, ec);
            }
            const std::lock_guard<std::mutex> lock(print_m);
            if (!e.empty()) {
                out("%s batch %zu: %s\n", m.tag.c_str(), b, e.c_str());
                failed = true;
                return;
            }
            const std::string& bkey = key_for_opt(batch_opt[b]);
            std::ofstream(so_dir / (batch_name[b] + ".key")) << bkey << "\n" << batch_hash(b) << "\n";
            const std::size_t d = ++done;
            out("  %s batch %zu done (%zu/%zu, %.0f s)\n", m.tag.c_str(), b, d, todo.size(), ms_since(t0) / 1000.0);
            if (o.progress) o.progress(m.tag, d, todo.size());
        }
    };
    std::vector<std::thread> pool;
    for (unsigned j = 0; j < std::max(1U, o.jobs); ++j) pool.emplace_back(worker);
    for (std::thread& t : pool) t.join();
    budget_cv.notify_all();
    if (failed) return false;
    if (o.translate_only) {
        out("%s: %zu batches ready in %s (%.0f s)\n", m.tag.c_str(), n_batches, so_dir.string().c_str(), ms_since(t0) / 1000.0);
        return true;
    }
    for (const fs::path& so : sos) {
        auto code = std::make_unique<LoadedCode>();
        std::string lerr;
        if (!load_batch(so, *code, &lerr)) {
            out("%s: cannot load %s: %s (translate again)\n", m.tag.c_str(), so.c_str(), lerr.c_str());
            return false;
        }
        const auto* rows = static_cast<const FnRow*>(code->sym("wse_fn_table"));
        const auto* count = static_cast<const std::uint64_t*>(code->sym("wse_fn_table_count"));
        if (rows == nullptr || count == nullptr) {
            out("%s: %s has no function table (translate again)\n", m.tag.c_str(), so.c_str());
            return false;
        }
        out_loaded.tables.emplace_back(rows, static_cast<std::size_t>(*count));
        out_loaded.functions += static_cast<std::size_t>(*count);
        const auto* frows = static_cast<const FastFnRow*>(code->sym("wse_fast_table"));
        const auto* fcount = static_cast<const std::uint64_t*>(code->sym("wse_fast_table_count"));
        if (frows != nullptr && fcount != nullptr && *fcount != 0) {
            out_loaded.fast_tables.emplace_back(frows, static_cast<std::size_t>(*fcount));
        }
        out_loaded.code.push_back(std::move(code));
    }
    // Peak memory so far (KiB on Linux and Android): translation threads are the main cost, and the
    // phone's limit is not known in advance.
    struct rusage ru {};
    (void)getrusage(RUSAGE_SELF, &ru);
    out("%s: %zu translated functions loaded from %zu objects (%.1f s); peak memory %ld MiB\n", m.tag.c_str(),
        out_loaded.functions, sos.size(), ms_since(t0) / 1000.0, static_cast<long>(ru.ru_maxrss / 1024));
    return true;
}

// Import tracing: first calls in order, and the last few calls of the run. The game makes
// hundreds of thousands of import calls a minute on several threads, so the per-call path takes no
// lock: a ring of the latest slots, and one atomic "seen" flag per import slot (indexed once before
// the run); only an import's first call takes `g_trace_m`, to record and print it.
std::mutex g_trace_m;
std::set<std::uint64_t> g_seen;              // slots without an index (none expected), under g_trace_m
std::vector<std::uint64_t> g_first;          // under g_trace_m
std::atomic<std::size_t> g_distinct{0};
std::atomic<std::uint64_t> g_calls{0};
constexpr std::size_t kRecent = 16;
std::array<std::atomic<std::uint64_t>, kRecent> g_ring{};
std::atomic<std::uint64_t> g_ring_pos{0};
std::unordered_map<std::uint64_t, std::size_t> g_slot_index;  // filled before the run, then read-only
std::unique_ptr<std::atomic<bool>[]> g_slot_seen;
Runtime* g_rt = nullptr;
std::mutex g_raise_m;
std::string g_last_raise;  // the last RaiseException's report (under g_raise_m)

// R9 PGO 1 kHz SIGPROF sampler: frozen before guest threads run so the signal handler is lock-free.
struct SampleRange {
    std::uintptr_t begin = 0;
    std::uintptr_t end = 0;
    std::uint64_t va = 0;
};
std::vector<SampleRange> g_pgo_ranges;
std::unique_ptr<std::atomic<std::uint64_t>[]> g_pgo_hits;
std::string g_pgo_out_path;
PgoProfile g_pgo_baseline;  // what the output file held before this run (samples accumulate)

void init_pgo_sampler(const Loaded& lx, const std::string& out_path) {
    if (out_path.empty()) return;
    g_pgo_baseline = PgoProfile{};
    if (std::error_code bec; fs::is_regular_file(out_path, bec)) {
        PgoProfile prev;
        if (load_pgo_profile(out_path, prev, nullptr)) g_pgo_baseline = std::move(prev);
    }
    std::vector<SampleRange> ranges;
    auto index_tables = [&](const auto& tables) {
        for (const auto& [rows, n] : tables) {
            std::vector<std::pair<std::uintptr_t, std::uint64_t>> batch_addrs;
            batch_addrs.reserve(n);
            for (std::size_t i = 0; i < n; ++i) {
                if (rows[i].fn != nullptr) {
                    batch_addrs.emplace_back(reinterpret_cast<std::uintptr_t>(rows[i].fn), rows[i].va);
                }
            }
            std::sort(batch_addrs.begin(), batch_addrs.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });
            for (std::size_t i = 0; i < batch_addrs.size(); ++i) {
                const std::uintptr_t b = batch_addrs[i].first;
                const std::uintptr_t e = (i + 1 < batch_addrs.size()) ? batch_addrs[i + 1].first : (b + 0x10000U);
                if (e > b) ranges.push_back({b, e, batch_addrs[i].second});
            }
        }
    };
    index_tables(lx.tables);
    index_tables(lx.fast_tables);
    if (ranges.empty()) return;
    std::sort(ranges.begin(), ranges.end(), [](const SampleRange& a, const SampleRange& b) { return a.begin < b.begin; });
    g_pgo_ranges = std::move(ranges);
    g_pgo_hits = std::make_unique<std::atomic<std::uint64_t>[]>(g_pgo_ranges.size());
    for (std::size_t i = 0; i < g_pgo_ranges.size(); ++i) g_pgo_hits[i].store(0, std::memory_order_relaxed);
    g_pgo_out_path = out_path;
}

void on_sigprof(int, siginfo_t*, void* uctx) {
    if (g_pgo_ranges.empty() || !g_pgo_hits || uctx == nullptr) return;
    std::uintptr_t pc = 0;
#if defined(__x86_64__)
    pc = static_cast<std::uintptr_t>(static_cast<ucontext_t*>(uctx)->uc_mcontext.gregs[REG_RIP]);
#elif defined(__aarch64__)
    pc = static_cast<std::uintptr_t>(static_cast<ucontext_t*>(uctx)->uc_mcontext.pc);
#endif
    if (pc == 0) return;
    auto it = std::upper_bound(g_pgo_ranges.begin(), g_pgo_ranges.end(), pc,
                               [](std::uintptr_t v, const SampleRange& r) { return v < r.begin; });
    if (it == g_pgo_ranges.begin()) return;
    --it;
    if (pc < it->end) {
        const std::size_t idx = static_cast<std::size_t>(it - g_pgo_ranges.begin());
        g_pgo_hits[idx].fetch_add(1, std::memory_order_relaxed);
    }
}

void start_pgo_timer() {
    if (g_pgo_out_path.empty() || g_pgo_ranges.empty()) return;
    struct sigaction sa {};
    sa.sa_sigaction = &on_sigprof;
    sa.sa_flags = SA_SIGINFO | SA_RESTART;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGPROF, &sa, nullptr);
    struct itimerval itv {};
    itv.it_interval.tv_usec = 1000;
    itv.it_value.tv_usec = 1000;
    setitimer(ITIMER_PROF, &itv, nullptr);
}

std::mutex g_pgo_flush_m;

void flush_pgo_profile() {
    const std::lock_guard<std::mutex> lock(g_pgo_flush_m);
    if (g_pgo_out_path.empty() || g_pgo_ranges.empty() || !g_pgo_hits) return;
    PgoProfile session;
    for (std::size_t i = 0; i < g_pgo_ranges.size(); ++i) {
        const std::uint64_t c = g_pgo_hits[i].load(std::memory_order_relaxed);
        if (c > 0) {
            session.hits[g_pgo_ranges[i].va] += c;
            session.total_samples += c;
        }
    }
    if (session.total_samples == 0) return;
    // This session's counts on top of the file as it was at start (flushes repeat, so the
    // baseline is never the file just written).
    PgoProfile prof = g_pgo_baseline;
    merge_pgo_profile(prof, session);
    (void)save_pgo_profile(g_pgo_out_path, prof, nullptr);
}

// The last calls, oldest first. Lock-free, so a crash handler can call it.
std::vector<std::uint64_t> recent_slots() {
    std::vector<std::uint64_t> r;
    const std::uint64_t end = g_ring_pos.load(std::memory_order_acquire);
    const std::uint64_t begin = end > 12 ? end - 12 : 0;
    for (std::uint64_t i = begin; i < end; ++i) r.push_back(g_ring[i % kRecent].load(std::memory_order_relaxed));
    return r;
}

void print_recent() {
    out("last imports called (oldest first):\n");
    for (const std::uint64_t slot : recent_slots()) {
        out("  %s\n", g_rt->import_name(slot).c_str());
    }
    out("%llu import calls, %zu distinct\n", static_cast<unsigned long long>(g_calls.load()), g_distinct.load());
}

// R9 Wave 5: frozen PDB symbol table + missing-DLL warnings for crash_report.txt.
struct PdbSym {
    std::uint64_t va = 0;
    std::string name;
};
std::vector<PdbSym> g_pdb_syms;
std::uint64_t g_crash_text_b = 0;
std::uint64_t g_crash_text_e = 0;
std::uint64_t g_crash_exe_base = 0;
const std::uint8_t* g_crash_exe_mapped = nullptr;
std::uint64_t g_crash_main_stack_top = 0;
std::string g_crash_report_path;
std::string g_crash_module;
std::mutex g_missing_dll_m;
std::vector<std::string> g_missing_dll_warnings;

std::string symbolize_va(std::uint64_t va) {
    if (g_pdb_syms.empty()) return "?";
    if (g_crash_text_e > g_crash_text_b && (va < g_crash_text_b || va >= g_crash_text_e)) return "?";
    auto it = std::upper_bound(g_pdb_syms.begin(), g_pdb_syms.end(), va,
                               [](std::uint64_t v, const PdbSym& s) { return v < s.va; });
    if (it == g_pdb_syms.begin()) return "?";
    --it;
    if (it->va == va) return it->name;
    if (va - it->va > 0x100000U) return "?";
    char off[32];
    std::snprintf(off, sizeof(off), "+0x%llx", static_cast<unsigned long long>(va - it->va));
    return it->name + off;
}

void write_crash_report(const std::string& summary) {
    // Write minimal async-signal-safe header to disk first, before any locks, allocations, or guest reads.
    if (!g_crash_report_path.empty() && !summary.empty()) {
        const int fd = ::open(g_crash_report_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd >= 0) {
            std::size_t off = 0;
            while (off < summary.size()) {
                const ssize_t n = ::write(fd, summary.data() + off, summary.size() - off);
                if (n <= 0) break;
                off += static_cast<std::size_t>(n);
            }
            ::close(fd);
        }
    }
    std::string body = summary;
    if (const CpuState* st = Runtime::current_thread_state()) {
        char line[512];
        std::snprintf(line, sizeof(line), "module: %s\nguest rip: 0x%llx (%s)\nguest rsp: 0x%llx\n",
                      g_crash_module.empty() ? "?" : g_crash_module.c_str(),
                      static_cast<unsigned long long>(st->rip), symbolize_va(st->rip).c_str(),
                      static_cast<unsigned long long>(st->gpr[4]));
        body += line;
        const std::uint64_t rsp = st->gpr[4];
        std::uint64_t top = 0;
        if (st->teb != 0) std::memcpy(&top, reinterpret_cast<const void*>(st->teb + teb::tebo::stack_base), 8);
        if (top <= rsp) top = g_crash_main_stack_top > rsp ? g_crash_main_stack_top : rsp;
        if (g_crash_exe_mapped != nullptr && g_crash_text_e > g_crash_text_b + 6) {
            body += "guest stack (exe return addresses):\n";
            int shown = 0;
            for (std::uint64_t q = rsp; q + 8 <= top && q < rsp + 0x8000 && shown < 24; q += 8) {
                std::uint64_t v = 0;
                std::memcpy(&v, reinterpret_cast<const void*>(q), 8);
                if (v < g_crash_text_b + 6 || v >= g_crash_text_e) continue;
                const std::uint8_t* c = g_crash_exe_mapped + (v - g_crash_exe_base);
                const bool after_call = c[-5] == 0xE8 || (c[-6] == 0xFF && (c[-5] & 0x38) == 0x10) ||
                                        (c[-2] == 0xFF && (c[-1] & 0x38) == 0x10) ||
                                        (c[-3] == 0xFF && (c[-2] & 0x38) == 0x10);
                if (!after_call) continue;
                std::snprintf(line, sizeof(line), "  [rsp+0x%llx] 0x%llx %s\n",
                              static_cast<unsigned long long>(q - rsp), static_cast<unsigned long long>(v),
                              symbolize_va(v).c_str());
                body += line;
                ++shown;
            }
        }
    }
    body += "last imports called (oldest first):\n";
    for (const std::uint64_t slot : recent_slots()) {
        body += "  " + (g_rt != nullptr ? g_rt->import_name(slot) : std::string("?")) + "\n";
    }
    {
        const std::lock_guard<std::mutex> lock(g_missing_dll_m);
        if (!g_missing_dll_warnings.empty()) {
            body += "missing third-party DLL warnings:\n";
            for (const std::string& w : g_missing_dll_warnings) body += "  " + w + "\n";
        }
    }
    out("%s", body.c_str());
    if (!g_crash_report_path.empty()) {
        const int fd = ::open(g_crash_report_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd >= 0) {
            std::size_t off = 0;
            while (off < body.size()) {
                const ssize_t n = ::write(fd, body.data() + off, body.size() - off);
                if (n <= 0) break;
                off += static_cast<std::size_t>(n);
            }
            ::close(fd);
        }
    }
}

// A host fault: the address, the host pc and the function holding it (translated functions are
// exported as wse_fn_<guest VA>, so dladdr names the guest function when the batch was loaded by
// the system loader). Diagnostics only: formatting in a signal handler is not async-signal-safe,
// which is acceptable at the point of a crash.
// If reporting blocks (the crash hit while that thread held the log's or a shim's lock), the
// watchdog ends the process after 3 s with a short line written straight to stderr, instead of the
// app hanging with no report. A second crash while reporting ends the process at once.
void on_watchdog(int) {
    static const char kMsg[] = "\nRESULT host-signal (the crash report did not finish)\n";
    (void)!write(2, kMsg, sizeof(kMsg) - 1);
    std::_Exit(kStartupHostSignal);
}

void on_signal(int sig, siginfo_t* info, void* uctx) {
    static std::atomic<int> entered{0};
    if (entered.exchange(1) != 0) std::_Exit(kStartupHostSignal);
    struct sigaction wd {};
    wd.sa_handler = &on_watchdog;
    sigemptyset(&wd.sa_mask);
    sigaction(SIGALRM, &wd, nullptr);
    alarm(3);
    std::uint64_t pc = 0;
#if defined(__x86_64__)
    pc = static_cast<std::uint64_t>(static_cast<ucontext_t*>(uctx)->uc_mcontext.gregs[REG_RIP]);
#elif defined(__aarch64__)
    pc = static_cast<std::uint64_t>(static_cast<ucontext_t*>(uctx)->uc_mcontext.pc);
#else
    (void)uctx;
#endif
    char sig_buf[512];
    Dl_info dl{};
    if (pc != 0 && dladdr(reinterpret_cast<void*>(static_cast<std::uintptr_t>(pc)), &dl) != 0) {
        std::snprintf(sig_buf, sizeof(sig_buf),
                      "\nhost signal %d in translated code or a shim: address 0x%llx, host pc 0x%llx = %s+0x%llx in %s\n",
                      sig, static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(info->si_addr)),
                      static_cast<unsigned long long>(pc), dl.dli_sname != nullptr ? dl.dli_sname : "?",
                      static_cast<unsigned long long>(pc - reinterpret_cast<std::uintptr_t>(dl.dli_saddr)),
                      dl.dli_fname != nullptr ? dl.dli_fname : "?");
    } else {
        std::snprintf(sig_buf, sizeof(sig_buf),
                      "\nhost signal %d in translated code or a shim: address 0x%llx, host pc 0x%llx\n",
                      sig, static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(info->si_addr)),
                      static_cast<unsigned long long>(pc));
    }
    write_crash_report(sig_buf);
    out("RESULT host-signal\n");
    std::_Exit(kStartupHostSignal);
}

}  // namespace

void flush_pgo_profile_now() {
    flush_pgo_profile();
}

int run_startup(const StartupOptions& o) {
    g_log = o.log;
    std::string err;
    if (o.install == nullptr || o.user == nullptr || o.so_dir.empty()) {
        out("setup: no install, writable layer or translation directory\n");
        return kStartupSetupError;
    }
    {
        std::error_code ec;
        fs::create_directories(o.so_dir, ec);
        if (ec) {
            out("setup: cannot create %s: %s\n", o.so_dir.c_str(), ec.message().c_str());
            return kStartupSetupError;
        }
    }
    Image exe{"exe", &o.exe, {}, nullptr};
    Image lua{"lua5164", &o.lua, {}, nullptr};
    if (!map_image(exe, &err)) {
        out("exe: %s\n", err.c_str());
        return kStartupSetupError;
    }
    if (!map_image(lua, &err)) {
        out("lua5164.dll: %s\n", err.c_str());
        return kStartupSetupError;
    }
    std::vector<PdbPublic> publics;
    PdbGuidAge ga;
    PeCodeView cv;
    if (!read_pe_codeview(o.exe.data(), o.exe.size(), exe.img, cv, &err) ||
        !read_pdb_publics(o.pdb.data(), o.pdb.size(), publics, &ga, &err) || ga.guid != cv.guid) {
        out("PDB: %s\n", err.empty() ? "does not belong to this exe" : err.c_str());
        return kStartupSetupError;
    }
    out("exe at 0x%llx (%u bytes), lua5164 at 0x%llx, PDB matches (%zu publics)\n",
        static_cast<unsigned long long>(exe.img.image_base), exe.img.size_of_image,
        static_cast<unsigned long long>(lua.img.image_base), publics.size());

    // Kept until the process ends: translated code and every shim object outlive the run.
    static Loaded lx;
    static Loaded ll;
    out("translator: %s\n", translator_fingerprint().c_str());
    if (!translate(exe, publics, o, lx) || !translate(lua, {}, o, ll)) {
        return kStartupSetupError;
    }
    if (o.translate_only) {
        out("RESULT translated\n");
        return 0;
    }

    static Runtime rt;
    g_rt = &rt;
    g_crash_report_path = o.crash_report_path;
    g_crash_module = o.module;
    g_crash_text_b = text_range(exe.img).first;
    g_crash_text_e = text_range(exe.img).second;
    g_crash_exe_base = exe.img.image_base;
    g_crash_exe_mapped = exe.mapped;
    g_pdb_syms.clear();
    g_pdb_syms.reserve(publics.size());
    for (const PdbPublic& p : publics) {
        if (p.section >= 1 && p.section <= exe.img.sections.size()) {
            g_pdb_syms.push_back({exe.img.image_base + exe.img.sections[p.section - 1].virtual_address + p.offset, p.name});
        }
    }
    std::sort(g_pdb_syms.begin(), g_pdb_syms.end(), [](const PdbSym& a, const PdbSym& b) { return a.va < b.va; });
    static std::atomic<bool> faulted{false};
    rt.on_fault = [](const Fault& f) {
        if (faulted.exchange(true)) return;
        char fbuf[512];
        std::snprintf(fbuf, sizeof(fbuf), "\nFAULT: %s at 0x%llx (%s) (rip 0x%llx)%s%s\n", fault_name(f.kind),
                      static_cast<unsigned long long>(f.va), symbolize_va(f.va).c_str(),
                      static_cast<unsigned long long>(f.rip), f.what.empty() ? "" : ": ", f.what.c_str());
        std::string extra;
        {
            const std::lock_guard<std::mutex> lock(g_raise_m);
            if (f.what == "unhandled guest exception" && !g_last_raise.empty()) {
                extra = g_last_raise;
            }
        }
        write_crash_report(std::string(fbuf) + extra);
        out("RESULT fault\n");
        std::_Exit(kStartupFault);
    };
    const bool quiet = o.quiet;
    rt.trace_import = [quiet](std::uint64_t slot) {
        g_calls.fetch_add(1, std::memory_order_relaxed);
        g_ring[g_ring_pos.fetch_add(1, std::memory_order_acq_rel) % kRecent].store(slot, std::memory_order_relaxed);
        const auto it = g_slot_index.find(slot);
        if (it != g_slot_index.end()) {
            std::atomic<bool>& seen = g_slot_seen[it->second];
            if (seen.load(std::memory_order_relaxed) || seen.exchange(true)) return;
        }
        const std::lock_guard<std::mutex> lock(g_trace_m);
        if (it == g_slot_index.end() && !g_seen.insert(slot).second) return;
        g_first.push_back(slot);
        g_distinct.store(g_first.size(), std::memory_order_relaxed);
        if (!quiet) out("  import #%zu %s\n", g_first.size(), g_rt->import_name(slot).c_str());
    };
    for (const auto& [rows, n] : lx.tables) rt.add_table(rows, n);
    for (const auto& [rows, n] : ll.tables) rt.add_table(rows, n);
    for (const auto& [frows, n] : lx.fast_tables) rt.add_fast_table(frows, n);
    for (const auto& [frows, n] : ll.fast_tables) rt.add_fast_table(frows, n);
    {
        std::size_t patched_edges = 0;
        std::size_t total_edges = 0;
        std::size_t fast_patched = 0;
        std::size_t fast_total = 0;
        auto patch_loaded = [&](const Loaded& loaded) {
            for (const auto& code : loaded.code) {
                const auto* cnt = static_cast<const std::uint64_t*>(code->sym("wse_ext_call_count"));
                if (cnt != nullptr && *cnt != 0) {
                    auto* slots = static_cast<GuestFn*>(code->sym("wse_ext_calls"));
                    const auto* vas = static_cast<const std::uint64_t*>(code->sym("wse_ext_call_vas"));
                    total_edges += static_cast<std::size_t>(*cnt);
                    patched_edges += rt.patch_external_calls(slots, vas, static_cast<std::size_t>(*cnt));
                }
                const auto* fcnt = static_cast<const std::uint64_t*>(code->sym("wse_ext_fast_call_count"));
                if (fcnt != nullptr && *fcnt != 0) {
                    auto* fslots = static_cast<void**>(code->sym("wse_ext_fast_calls"));
                    const auto* fvas = static_cast<const std::uint64_t*>(code->sym("wse_ext_fast_call_vas"));
                    fast_total += static_cast<std::size_t>(*fcnt);
                    fast_patched += rt.patch_external_fast_calls(fslots, fvas, static_cast<std::size_t>(*fcnt));
                }
            }
        };
        patch_loaded(lx);
        patch_loaded(ll);
        if (total_edges != 0 || fast_total != 0) {
            out("linked %zu/%zu direct cross-batch call edges (%zu/%zu fast-ABI)\n", patched_edges, total_edges,
                fast_patched, fast_total);
        }
    }

    static shim::Modules modules(rt);
    {
        auto seen_missing = std::make_shared<std::set<std::string>>();
        platform::Vfs* ivfs = o.install;
        platform::Vfs* uvfs = o.user;
        std::string mod_name = o.module;
        modules.on_missing_library = [seen_missing, ivfs, uvfs, mod_name](const std::string& requested,
                                                                          const std::string& canon) {
            static constexpr const char* kBenignProbes[] = {
                "dwmapi.dll", "uxtheme.dll", "nvapi64.dll", "aticfx64.dll", "xinput1_4.dll",
                "dbghelp.dll", "wer.dll", "shcore.dll", "propsys.dll", "sechost.dll",
            };
            for (const char* b : kBenignProbes) {
                if (canon == b) return;
            }
            if (canon.rfind("d3dx9_", 0) == 0 || canon.rfind("msvcr", 0) == 0 ||
                canon.rfind("api-ms-", 0) == 0 || canon.rfind("ext-ms-", 0) == 0) {
                return;
            }
            const std::lock_guard<std::mutex> lock(g_missing_dll_m);
            if (!seen_missing->insert(canon).second) return;
            const bool on_disk = (ivfs != nullptr && (ivfs->is_file(canon) ||
                                                      ivfs->is_file("Modules/" + mod_name + "/" + canon))) ||
                                 (uvfs != nullptr && (uvfs->is_file(canon) ||
                                                      uvfs->is_file("Modules/" + mod_name + "/" + canon)));
            std::string msg = "Mod requested native x64 DLL '" + requested + "' (" + canon +
                              (on_disk ? ", present on disk" : ", not bundled") +
                              ") which cannot be loaded on ARM64";
            out("  WARNING: %s\n", msg.c_str());
            if (g_missing_dll_warnings.size() < 16U) g_missing_dll_warnings.push_back(std::move(msg));
        };
    }
    if (modules.add_image("C:\\WSE2\\mb_warband_wse2_x64.exe", exe.img, exe.mapped, true, &err) == nullptr ||
        modules.add_image("C:\\WSE2\\lua5164.dll", lua.img, lua.mapped, false, &err) == nullptr) {
        out("modules: %s\n", err.c_str());
        return kStartupSetupError;
    }
    std::vector<std::string> unresolved;
    const std::size_t cross = modules.link(&unresolved);
    out("linked %zu imports across modules%s\n", cross, unresolved.empty() ? "" : "; unresolved:");
    for (const std::string& u : unresolved) out("  %s\n", u.c_str());
    rt.fill_iat();
    {
        const std::vector<PeImport>& list = rt.import_list();
        g_slot_seen = std::make_unique<std::atomic<bool>[]>(list.size());
        for (std::size_t i = 0; i < list.size(); ++i) {
            g_slot_index.emplace(list[i].slot_va, i);
            g_slot_seen[i].store(false, std::memory_order_relaxed);
        }
    }
    // LuaJIT's JIT compiles Lua to x86 code at run time, which nothing here can run (and translating
    // during play is not allowed): every state the exe creates gets luaJIT_setmode(L, 0,
    // LUAJIT_MODE_ENGINE | LUAJIT_MODE_OFF) and runs in the interpreter.
    if (shim::Module* lm = modules.find("lua5164.dll", false)) {
        const std::uint64_t newstate = modules.proc_address(*lm, "luaL_newstate", 0, 0);
        const std::uint64_t setmode = modules.proc_address(*lm, "luaJIT_setmode", 0, 0);
        std::size_t wrapped = 0;
        for (const PeImport& imp : rt.import_list()) {
            if (imp.name == "luaL_newstate" && imp.slot_va >= exe.img.image_base &&
                imp.slot_va < exe.img.image_base + exe.img.size_of_image && newstate != 0 && setmode != 0) {
                rt.set_import(imp.slot_va, [newstate, setmode](CpuState& s) {
                    bool ok = false;
                    const std::uint64_t L = rt.call_guest(s, newstate, {}, &ok);
                    if (s.unwind_rsp != 0) return;
                    if (L != 0) (void)rt.call_guest(s, setmode, {L, 0, 0}, &ok);
                    if (s.unwind_rsp == 0) win64_return(s, L);
                });
                ++wrapped;
            }
        }
        out("LuaJIT: the JIT is switched off in every new state (%zu luaL_newstate imports wrapped)\n", wrapped);
    }

    static seh::Exceptions exceptions(rt, exe.img, exe.mapped);
    exceptions.add_image(lua.img, lua.mapped);
    static teb::Process process(rt, exe.img, exe.mapped);
    if (!process.init(&err)) {
        out("TEB/TLS: %s\n", err.c_str());
        return kStartupSetupError;
    }
    static kernel::Kernel kern(rt);
    static threads::Threads thr(rt, process, kern);
    static threadpool::Pool pool(rt, process, kern);
    static shim::Time time_shims(rt);
    shim::ProcessConfig process_config;
    process_config.command_line = "mb_warband_wse2_x64.exe --module \"" + o.module + "\"";
    out("command line: %s\n", process_config.command_line.c_str());
    static shim::Crt crt(rt, process_config);
    crt.on_exit = [](std::uint32_t code) {
        flush_pgo_profile();
        out("\nExitProcess(%u)\n", code);
        print_recent();
        out("RESULT exit-process %u\n", code);
        std::_Exit(kStartupExitProcess);
    };
    crt.on_debug_string = [](const std::string& text) { out("  debug: %s", text.c_str()); };
    static shim::Memory memory(rt);
    static shim::Strings strings(rt, memory);
    shim::FilesConfig files_cfg;
    files_cfg.install = o.install;
    files_cfg.user = o.user;
    static shim::Files files(rt, kern, files_cfg);
    static shim::Advapi advapi(rt, o.user);
    // The module, as the Warband launcher records it: the exe reads last_module_warband (REG_SZ)
    // from this key at startup.
    {
        std::vector<std::uint8_t> sz(o.module.begin(), o.module.end());
        sz.push_back(0);
        advapi.set_value("HKEY_CURRENT_USER\\Software\\MountAndBladeWarbandKeys", "last_module_warband", 1, std::move(sz));
    }
    static shim::User32 user32(rt);
    user32.on_message_box = [](const std::string& caption, const std::string& text) {
        out("  MessageBox [%s] %s\n", caption.c_str(), text.c_str());
    };
    static shim::Net net(rt);
    static shim::D3dxMath d3dx(rt);
    static shim::D3d9 d3d9(rt);
    static shim::FmodSteam fmod_steam(rt);
    // D3DX reads texture headers: a guest path in the install ("C:\WSE2\..." or relative).
    platform::Vfs* install_vfs = o.install;
    platform::Vfs* user_vfs = o.user;
    d3d9.read_file = [install_vfs, user_vfs](const std::string& path, std::vector<std::uint8_t>& bytes) {
        std::string p = path;
        static const std::string kRoot = "c:\\wse2\\";
        std::string low = p;
        for (char& c : low) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (low.rfind(kRoot, 0) == 0) p = p.substr(kRoot.size());
        for (char& c : p) c = c == '\\' ? '/' : c;
        return install_vfs->read_file(p, bytes) || user_vfs->read_file(p, bytes);
    };
    d3d9.on_first_call = [quiet](const std::string& m) {
        if (!quiet) out("  recorded %s\n", m.c_str());
    };
    static const auto t_start = std::chrono::steady_clock::now();
    if (o.frame_limit != 0) {
        const std::uint64_t limit = o.frame_limit;
        d3d9.on_present = [limit](std::uint64_t frames) {
            if (frames < limit) return;
            flush_pgo_profile();
            const std::lock_guard<std::mutex> lock(g_trace_m);
            out("\n%llu frames presented in %.1f s with no fault: %llu import calls, %zu distinct\n",
                static_cast<unsigned long long>(frames), ms_since(t_start) / 1000.0,
                static_cast<unsigned long long>(g_calls.load()), g_distinct.load());
            out("RESULT pass %llu frames\n", static_cast<unsigned long long>(frames));
            std::_Exit(kStartupFrameLimit);
        };
    }
    static shim::SystemDlls sysdlls(rt);
    sysdlls.on_launch = [](const std::string& what) { out("  launch refused: %s\n", what.c_str()); };
    const std::size_t bound = exceptions.install() + process.install() + kern.install() + thr.install() + pool.install() +
                              time_shims.install() + crt.install() + memory.install() + strings.install() + files.install() +
                              modules.install() + advapi.install() + user32.install() + net.install() + sysdlls.install() +
                              d3dx.install() + d3d9.install() + fmod_steam.install();
    std::size_t total = 0;
    std::size_t missing = 0;
    std::map<std::string, std::vector<std::string>> missing_by_dll;
    for (const PeImport& imp : rt.import_list()) {
        ++total;
        const std::string key = imp.name.empty() ? "#" + std::to_string(imp.ordinal) : imp.name;
        if (rt.named_shim(imp.dll, key) == nullptr && imp.dll != "lua5164.dll") {
            ++missing;
            missing_by_dll[imp.dll].push_back(key);
        }
    }
    out("%zu import slots bound to shims (%zu slots, %zu with no shim yet in %zu DLLs)\n", bound, total, missing,
        missing_by_dll.size());
    if (missing <= 32) {
        for (const auto& [dll, names] : missing_by_dll) {
            for (const std::string& n : names) out("  no shim: %s!%s\n", dll.c_str(), n.c_str());
        }
    }

    // Diagnostics for a guest exception nobody handles: RaiseException records its code and
    // parameters, the exe return addresses on the guest stack (named by the nearest PDB public),
    // and the printable strings in the raising frames; on_fault prints them.
    static std::uint64_t main_stack_top = 0;  // set once the main stack is prepared
    static std::map<std::uint64_t, std::string> names;
    for (const PdbPublic& p : publics) {
        if (p.section >= 1 && p.section <= exe.img.sections.size()) {
            names[exe.img.image_base + exe.img.sections[p.section - 1].virtual_address + p.offset] = p.name;
        }
    }
    static const std::uint64_t exe_text_b = text_range(exe.img).first;
    static const std::uint64_t exe_text_e = text_range(exe.img).second;
    static const std::uint64_t exe_base = exe.img.image_base;
    static const std::uint8_t* const exe_mapped = exe.mapped;
    auto name_of = [](std::uint64_t va) {
        auto it = names.upper_bound(va);
        if (it == names.begin()) return std::string("?");
        --it;
        char off[24];
        std::snprintf(off, sizeof(off), "+0x%llx", static_cast<unsigned long long>(va - it->first));
        return it->first == va ? it->second : it->second + off;
    };
    if (const Runtime::ImportShim* raise = rt.named_shim("KERNEL32.dll", "RaiseException")) {
        const Runtime::ImportShim original = *raise;
        rt.set_import("KERNEL32.dll", "RaiseException", [original, name_of](CpuState& st) {
            std::string r;
            char line[512];
            const std::uint64_t args = win64_arg(st, 3);
            const auto nargs = static_cast<std::uint32_t>(win64_arg(st, 2));
            std::snprintf(line, sizeof(line), "last RaiseException: code 0x%08llx flags 0x%llx, %u parameters",
                          static_cast<unsigned long long>(win64_arg(st, 0) & 0xFFFFFFFFULL),
                          static_cast<unsigned long long>(win64_arg(st, 1)), nargs);
            r += line;
            for (std::uint32_t i = 0; i < nargs && i < 4 && args != 0; ++i) {
                std::uint64_t v = 0;
                std::memcpy(&v, reinterpret_cast<const void*>(args + 8ULL * i), 8);
                std::snprintf(line, sizeof(line), " 0x%llx", static_cast<unsigned long long>(v));
                r += line;
            }
            r += "\n  guest stack (exe return addresses):\n";
            const std::uint64_t rsp = st.gpr[4];
            // The scan stays inside the thread's stack: the TEB's StackBase (guest threads), else
            // the main stack's top; with neither above rsp, nothing is read.
            std::uint64_t top = 0;
            if (st.teb != 0) std::memcpy(&top, reinterpret_cast<const void*>(st.teb + teb::tebo::stack_base), 8);
            if (top <= rsp) top = main_stack_top > rsp ? main_stack_top : rsp;
            int shown = 0;
            for (std::uint64_t q = rsp; q + 8 <= top && q < rsp + 0x8000 && shown < 24; q += 8) {
                std::uint64_t v = 0;
                std::memcpy(&v, reinterpret_cast<const void*>(q), 8);
                if (v < exe_text_b + 6 || v >= exe_text_e) continue;
                const std::uint8_t* c = exe_mapped + (v - exe_base);
                const bool after_call = c[-5] == 0xE8 || (c[-6] == 0xFF && (c[-5] & 0x38) == 0x10) ||
                                        (c[-2] == 0xFF && (c[-1] & 0x38) == 0x10) || (c[-3] == 0xFF && (c[-2] & 0x38) == 0x10);
                if (!after_call) continue;
                std::snprintf(line, sizeof(line), "    [rsp+0x%llx] 0x%llx %s\n", static_cast<unsigned long long>(q - rsp),
                              static_cast<unsigned long long>(v), name_of(v).c_str());
                r += line;
                ++shown;
            }
            r += "  strings in the raising frames:\n";
            std::string run;
            int strings_found = 0;
            for (std::uint64_t q = rsp; q < top && q < rsp + 0x4000 && strings_found < 12; ++q) {
                const char ch = *reinterpret_cast<const char*>(q);
                if (ch >= 0x20 && ch < 0x7F) {
                    run += ch;
                    continue;
                }
                if (run.size() >= 12) {
                    r += "    \"" + run.substr(0, 300) + "\"\n";
                    ++strings_found;
                }
                run.clear();
            }
            {
                const std::lock_guard<std::mutex> lock(g_raise_m);
                g_last_raise = r;
            }
            original(st);
        });
    }

    // Which module the game loads: the first distinct paths under Modules\ that CreateFile and
    // FindFirstFile are given (narrow and wide spellings).
    auto module_paths = std::make_shared<std::set<std::string>>();
    for (const char* fn : {"CreateFileA", "CreateFileW", "FindFirstFileA", "FindFirstFileW", "FindFirstFileExW"}) {
        const Runtime::ImportShim* orig = rt.named_shim("KERNEL32.dll", fn);
        if (orig == nullptr) continue;
        const bool wide = fn[std::strlen(fn) - 1] == 'W';
        rt.set_import("KERNEL32.dll", fn, [original = *orig, wide, module_paths](CpuState& st) {
            std::string p;
            if (const std::uint64_t a = win64_arg(st, 0)) {
                if (wide) {
                    for (auto* w = reinterpret_cast<const char16_t*>(a); *w != 0 && p.size() < 400; ++w) {
                        p += *w < 0x80 ? static_cast<char>(*w) : '?';
                    }
                } else {
                    p = reinterpret_cast<const char*>(a);
                }
            }
            std::string low = p;
            for (char& c : low) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            if (low.find("modules") != std::string::npos) {
                const std::lock_guard<std::mutex> lock(g_trace_m);
                if (module_paths->size() < 40 && module_paths->insert(p).second) out("  module file: %s\n", p.c_str());
            }
            original(st);
        });
    }

    // Registry reads: the first 30 key / value names and results (the module choice is one).
    auto reg_count = std::make_shared<int>(0);
    for (const char* fn : {"RegOpenKeyExA", "RegOpenKeyExW", "RegQueryValueExA", "RegQueryValueExW", "RegCreateKeyExA",
                           "RegCreateKeyExW", "RegGetValueA", "RegGetValueW"}) {
        const Runtime::ImportShim* orig = rt.named_shim("ADVAPI32.dll", fn);
        if (orig == nullptr) continue;
        const bool wide = fn[std::strlen(fn) - 1] == 'W';
        const std::string fname = fn;
        rt.set_import("ADVAPI32.dll", fn, [original = *orig, wide, fname, reg_count](CpuState& st) {
            auto text = [&](std::uint64_t a) {
                std::string p;
                if (a == 0) return std::string("(null)");
                if (wide) {
                    for (auto* w = reinterpret_cast<const char16_t*>(a); *w != 0 && p.size() < 200; ++w) p += *w < 0x80 ? static_cast<char>(*w) : '?';
                } else {
                    p = reinterpret_cast<const char*>(a);
                }
                return p;
            };
            const std::uint64_t hkey = win64_arg(st, 0);
            const std::string name = text(win64_arg(st, 1));
            const std::string extra = fname.rfind("RegGetValue", 0) == 0 ? " / " + text(win64_arg(st, 2)) : "";
            original(st);
            const std::lock_guard<std::mutex> lock(g_trace_m);
            if ((*reg_count)++ < 30) {
                out("  registry: %s(0x%llx, %s%s) = %llu\n", fname.c_str(), static_cast<unsigned long long>(hkey), name.c_str(),
                    extra.c_str(), static_cast<unsigned long long>(st.gpr[0]));
            }
        });
    }

    for (const int sig : {SIGSEGV, SIGBUS, SIGILL, SIGFPE}) {
        struct sigaction sa {};
        sa.sa_sigaction = &on_signal;
        sa.sa_flags = SA_SIGINFO;
        sigemptyset(&sa.sa_mask);
        sigaction(sig, &sa, nullptr);
    }
    static GuestStack stack(8 * 1024 * 1024);
    static CpuState s;
    stack.prepare(s);
    main_stack_top = s.gpr[4] + 8;  // prepare leaves rsp 8 below the aligned top
    g_crash_main_stack_top = main_stack_top;
    rt.attach(s);
    teb::GuestThread& main_thread = process.create_thread(0, 0);
    teb::Process::attach(s, main_thread);
    out("\n-- TLS callbacks, DllMain, entry point --\n");
    if (!process.run_tls_callbacks(s, teb::kDllProcessAttach)) out("TLS callbacks failed\n");
    std::string failed;
    if (!modules.attach_dlls(s, &failed)) {
        out("DllMain of %s failed\n", failed.c_str());
        out("RESULT fault\n");
        std::_Exit(kStartupFault);
    }
    const auto t0 = std::chrono::steady_clock::now();
    if (!o.profile_out.empty()) {
        init_pgo_sampler(lx, o.profile_out);
        start_pgo_timer();
        out("PGO profiler: sampling at 1 kHz -> %s\n", o.profile_out.c_str());
    }
    // Heartbeat: a steady-state loop prints no first calls, so report the call count, the frames
    // presented and the latest imports (a stall shows what it waits on).
    if (o.heartbeat_s != 0) {
        const unsigned every = o.heartbeat_s;
        std::thread([t0, every]() {
            std::uint64_t last_calls = 0;
            for (;;) {
                std::this_thread::sleep_for(std::chrono::seconds(every));
                flush_pgo_profile();
                const std::uint64_t calls = g_calls.load();
                std::string latest;
                for (const std::uint64_t slot : recent_slots()) latest += " " + g_rt->import_name(slot);
                out("\n[%.0f s] %llu import calls (+%llu), %llu frames presented; latest:%s\n", ms_since(t0) / 1000.0,
                    static_cast<unsigned long long>(calls), static_cast<unsigned long long>(calls - last_calls),
                    static_cast<unsigned long long>(d3d9.frames()), latest.c_str());
                last_calls = calls;
            }
        }).detach();
    }
    const bool ran = rt.run(s, exe.img.image_base + exe.img.entry_rva);
    flush_pgo_profile();
    out("\nentry point returned (%s) rax 0x%llx after %.1f s, %llu frames presented\n", ran ? "ok" : "fault",
        static_cast<unsigned long long>(s.gpr[0]), ms_since(t0) / 1000.0, static_cast<unsigned long long>(d3d9.frames()));
    print_recent();
    out("RESULT returned\n");
    std::_Exit(kStartupReturned);
}

}  // namespace wse::recomp
