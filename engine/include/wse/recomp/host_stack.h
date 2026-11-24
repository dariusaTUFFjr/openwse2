#pragma once

// Host stacks for threads that run translated code (Phase 5, docs/RECOMP_PLAN.md).
//
// Translated functions call each other natively, so a guest call chain lives on the HOST stack,
// and a translated frame is larger than the x64 frame it replaces (more so for functions
// translated without optimisation, such as rglConfig::load / ::save, each one giant function).
// The default host stack (about 1 MiB for an Android Java thread or a bionic pthread) ran out on
// the phone at startup (APK bbf4531: signal 11 at HostVfs::resolve+0x4, the first store of a
// prologue, right after rglConfig::load called CreateFileW).
//
// Every host thread that runs guest code therefore starts through `HostThread` with
// `kHostStackBytes` reserved (address space only; pages are committed as the stack grows), and
// records its stack bounds plus an alternate signal stack, so a stack overflow is reported as one
// instead of crashing again inside the signal handler. Threads created elsewhere (the Android
// game thread, started from Kotlin with the same size) call `prepare_current_host_thread()`.

#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#ifdef near
#undef near
#endif
#ifdef far
#undef far
#endif
#else
#include <pthread.h>
#include <signal.h>
#include <sys/mman.h>
#endif

namespace wse::recomp {

// 64 MiB of reserved stack per thread that runs translated code. Kotlin's game and startup
// threads use the same number (RecompGameActivity / R4StartupService).
inline constexpr std::size_t kHostStackBytes = std::size_t{64} << 20;

namespace host_stack_detail {

struct Bounds {
    std::uintptr_t lo = 0;  // lowest usable address
    std::uintptr_t hi = 0;  // one past the highest
};

inline Bounds& bounds() {
    static thread_local Bounds b;
    return b;
}

#ifndef _WIN32
inline constexpr std::size_t kAltStackBytes = std::size_t{128} << 10;

// The alternate signal stack of this thread, removed again before its memory goes.
struct AltStack {
    void* mem = nullptr;
    ~AltStack() {
        if (mem == nullptr) return;
        stack_t off{};
        off.ss_flags = SS_DISABLE;
        (void)sigaltstack(&off, nullptr);
        (void)munmap(mem, kAltStackBytes);
    }
};

inline AltStack& alt_stack() {
    static thread_local AltStack a;
    return a;
}
#endif

}  // namespace host_stack_detail

// Records this thread's stack bounds and gives it an alternate signal stack. Idempotent.
inline void prepare_current_host_thread() {
    auto& b = host_stack_detail::bounds();
#ifdef _WIN32
    if (b.hi == 0) {
        // The reservation's base is the bottom; the TIB's StackBase is the top.
        MEMORY_BASIC_INFORMATION mbi{};
        if (VirtualQuery(&mbi, &mbi, sizeof(mbi)) != 0) {
            b.lo = reinterpret_cast<std::uintptr_t>(mbi.AllocationBase);
            b.hi = reinterpret_cast<std::uintptr_t>(reinterpret_cast<NT_TIB*>(NtCurrentTeb())->StackBase);
        }
    }
#else
    if (b.hi == 0) {
        pthread_attr_t attr;
        if (pthread_getattr_np(pthread_self(), &attr) == 0) {
            void* addr = nullptr;
            std::size_t size = 0;
            if (pthread_attr_getstack(&attr, &addr, &size) == 0 && addr != nullptr) {
                b.lo = reinterpret_cast<std::uintptr_t>(addr);
                b.hi = b.lo + size;
            }
            pthread_attr_destroy(&attr);
        }
    }
    auto& alt = host_stack_detail::alt_stack();
    if (alt.mem == nullptr) {
        void* mem = mmap(nullptr, host_stack_detail::kAltStackBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mem != MAP_FAILED) {
            stack_t ss{};
            ss.ss_sp = mem;
            ss.ss_size = host_stack_detail::kAltStackBytes;
            if (sigaltstack(&ss, nullptr) == 0) {
                alt.mem = mem;
            } else {
                (void)munmap(mem, host_stack_detail::kAltStackBytes);
            }
        }
    }
#endif
}

// This thread's stack size in bytes, 0 when `prepare_current_host_thread` has not run or failed.
inline std::size_t current_host_stack_bytes() {
    const auto& b = host_stack_detail::bounds();
    return b.hi > b.lo ? static_cast<std::size_t>(b.hi - b.lo) : 0;
}

// True when `addr` lies at the bottom end of this thread's stack (the guard below it, or the last
// 64 KiB), the address of a host stack overflow. Reads only a thread_local: usable in a signal
// handler of the faulting thread.
inline bool is_host_stack_overflow(std::uintptr_t addr) {
    const auto& b = host_stack_detail::bounds();
    if (b.hi <= b.lo) return false;
    constexpr std::uintptr_t kBelow = std::uintptr_t{1} << 20;
    constexpr std::uintptr_t kAbove = std::uintptr_t{64} << 10;
    const std::uintptr_t from = b.lo > kBelow ? b.lo - kBelow : 0;
    return addr >= from && addr < b.lo + kAbove;
}

// A joinable host thread with a `stack_bytes` stack, prepared as above. Move-only, like
// std::thread; destroying a joinable one terminates, like std::thread.
class HostThread {
public:
    HostThread() = default;

    explicit HostThread(std::function<void()> fn, std::size_t stack_bytes = kHostStackBytes) {
        auto* job = new std::function<void()>(std::move(fn));
#ifdef _WIN32
        handle_ = CreateThread(nullptr, stack_bytes, &HostThread::entry, job, STACK_SIZE_PARAM_IS_A_RESERVATION, nullptr);
        if (handle_ == nullptr) {
            delete job;
            std::terminate();
        }
#else
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setstacksize(&attr, stack_bytes);
        const int rc = pthread_create(&thread_, &attr, &HostThread::entry, job);
        pthread_attr_destroy(&attr);
        if (rc != 0) {
            delete job;
            std::terminate();
        }
        started_ = true;
#endif
    }

    HostThread(const HostThread&) = delete;
    HostThread& operator=(const HostThread&) = delete;

    HostThread(HostThread&& o) noexcept { swap(o); }
    HostThread& operator=(HostThread&& o) noexcept {
        if (this != &o) {
            if (joinable()) std::terminate();
            swap(o);
        }
        return *this;
    }

    ~HostThread() {
        if (joinable()) std::terminate();
    }

    bool joinable() const {
#ifdef _WIN32
        return handle_ != nullptr;
#else
        return started_;
#endif
    }

    void join() {
#ifdef _WIN32
        if (handle_ == nullptr) return;
        WaitForSingleObject(handle_, INFINITE);
        CloseHandle(handle_);
        handle_ = nullptr;
#else
        if (!started_) return;
        pthread_join(thread_, nullptr);
        started_ = false;
#endif
    }

private:
    void swap(HostThread& o) noexcept {
#ifdef _WIN32
        std::swap(handle_, o.handle_);
#else
        std::swap(thread_, o.thread_);
        std::swap(started_, o.started_);
#endif
    }

#ifdef _WIN32
    static DWORD WINAPI entry(LPVOID p) {
        run(p);
        return 0;
    }
    HANDLE handle_ = nullptr;
#else
    static void* entry(void* p) {
        run(p);
        return nullptr;
    }
    pthread_t thread_{};
    bool started_ = false;
#endif

    static void run(void* p) {
        std::unique_ptr<std::function<void()>> job(static_cast<std::function<void()>*>(p));
        prepare_current_host_thread();
        (*job)();
    }
};

}  // namespace wse::recomp
