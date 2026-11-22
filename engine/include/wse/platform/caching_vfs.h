// Platform seam: thread-safe read-through disk cache over a slow `Vfs` (SAF).
//
// Port infrastructure, not a reconstruction: carries no retail v1.1.5.6 citation.
// Copies each file to `<cache_root>/<normalized_path>` the first time the game
// reads it, keyed by `(normalized_path, size, mtime_ms)` in `manifest.tsv`, so
// later runs read local files directly without Android SAF binder IPC.

#pragma once

#include <atomic>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

#include "wse/platform/vfs.h"

namespace wse::platform {

class CachingVfs final : public Vfs {
public:
    // `inner`: the slow install (SAF). `cache_root`: a host directory in app storage it owns.
    CachingVfs(Vfs& inner, std::filesystem::path cache_root)
        : inner_(inner), cache_root_(std::move(cache_root)) {
        load_manifest();
    }

    // Stats for the report: reads served from the copy, files copied, bytes copied.
    [[nodiscard]] std::uint64_t hits() const noexcept {
        return hits_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] std::uint64_t copies() const noexcept {
        return copies_.load(std::memory_order_relaxed);
    }
    [[nodiscard]] std::uint64_t copied_bytes() const noexcept {
        return copied_bytes_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] bool read_file(std::string_view path,
                                 std::vector<std::uint8_t>& out) override {
        return read_impl(path, out, [&](std::vector<std::uint8_t>& buf) {
            return inner_.read_file(path, buf);
        });
    }

    [[nodiscard]] bool read_text(std::string_view path, std::string& out) override {
        return read_impl(path, out, [&](std::string& buf) {
            return inner_.read_text(path, buf);
        });
    }

    [[nodiscard]] bool write_file(std::string_view path,
                                  std::span<const std::uint8_t> data) override {
        return inner_.write_file(path, data);
    }

    [[nodiscard]] bool remove_file(std::string_view path) override {
        return inner_.remove_file(path);
    }

    [[nodiscard]] bool list_dir(std::string_view path,
                                std::vector<VfsEntry>& out) override {
        return inner_.list_dir(path, out);
    }

    [[nodiscard]] bool is_file(std::string_view path) override {
        return inner_.is_file(path);
    }

    [[nodiscard]] bool is_dir(std::string_view path) override {
        return inner_.is_dir(path);
    }

    [[nodiscard]] bool make_dir(std::string_view path) override {
        return inner_.make_dir(path);
    }

    [[nodiscard]] bool remove_dir(std::string_view path) override {
        return inner_.remove_dir(path);
    }

    [[nodiscard]] bool rename_file(std::string_view from, std::string_view to) override {
        return inner_.rename_file(from, to);
    }

    [[nodiscard]] bool file_size(std::string_view path, std::uint64_t& size) override {
        return inner_.file_size(path, size);
    }

    [[nodiscard]] bool file_stamp(std::string_view path,
                                  std::uint64_t& size,
                                  std::int64_t& mtime_ms) override {
        return inner_.file_stamp(path, size, mtime_ms);
    }

    [[nodiscard]] static std::uint64_t disk_usage(const std::filesystem::path& cache_root) {
        std::error_code ec;
        if (!std::filesystem::exists(cache_root, ec) || ec) {
            return 0;
        }
        std::uint64_t total = 0;
        std::filesystem::recursive_directory_iterator it(
            cache_root, std::filesystem::directory_options::skip_permission_denied, ec);
        if (ec) {
            return 0;
        }
        for (const std::filesystem::recursive_directory_iterator end; it != end;
             it.increment(ec)) {
            if (ec) {
                break;
            }
            std::error_code fe;
            if (it->is_regular_file(fe) && !fe) {
                const std::uint64_t sz = it->file_size(fe);
                if (!fe) {
                    total += sz;
                }
            }
        }
        return total;
    }

    [[nodiscard]] static bool clear(const std::filesystem::path& cache_root) {
        std::error_code ec;
        if (!std::filesystem::exists(cache_root, ec)) {
            return !ec;
        }
        if (!std::filesystem::is_directory(cache_root, ec) || ec) {
            return std::filesystem::remove(cache_root, ec) && !ec;
        }
        bool ok = true;
        for (std::filesystem::directory_iterator it(cache_root, ec), end;
             !ec && it != end; it.increment(ec)) {
            std::error_code rem_ec;
            std::filesystem::remove_all(it->path(), rem_ec);
            if (rem_ec) {
                ok = false;
            }
        }
        return ok && !ec;
    }

private:
    struct Stamp {
        std::uint64_t size = 0;
        std::int64_t mtime_ms = 0;
    };

    template <typename Out, typename ReadInner>
    bool read_impl(std::string_view path, Out& out, ReadInner&& read_inner) {
        out.clear();
        const auto norm = normalize_path(path);
        if (!norm || norm->empty()) {
            return false;
        }

        std::uint64_t size = 0;
        std::int64_t mtime_ms = 0;
        if (!inner_.file_stamp(path, size, mtime_ms)) {
            return false;
        }

        if (*norm != "manifest.tsv") {
            bool stamp_matches = false;
            {
                const std::lock_guard<std::mutex> lock(mu_);
                const auto it = manifest_.find(*norm);
                stamp_matches = (it != manifest_.end() && it->second.size == size &&
                                 it->second.mtime_ms == mtime_ms);
            }
            if (stamp_matches) {
                const std::filesystem::path dst = local_path(cache_root_, *norm);
                std::error_code ec;
                if (std::filesystem::is_regular_file(dst, ec) && !ec &&
                    std::filesystem::file_size(dst, ec) == size && !ec &&
                    read_local_file(dst, size, out)) {
                    hits_.fetch_add(1, std::memory_order_relaxed);
                    return true;
                }
                out.clear();
            }
        }

        if (!read_inner(out)) {
            out.clear();
            return false;
        }

        if (*norm != "manifest.tsv" && static_cast<std::uint64_t>(out.size()) == size) {
            store_copy(*norm, size, mtime_ms, out.data(), out.size());
        }
        return true;
    }

    void load_manifest() {
        std::ifstream in(cache_root_ / "manifest.tsv", std::ios::binary);
        if (!in.is_open()) {
            return;
        }
        std::string line;
        while (std::getline(in, line)) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            if (line.empty()) {
                continue;
            }
            const std::size_t t1 = line.find('\t');
            if (t1 == std::string::npos) {
                continue;
            }
            const std::size_t t2 = line.find('\t', t1 + 1U);
            if (t2 == std::string::npos || line.find('\t', t2 + 1U) != std::string::npos) {
                continue;
            }
            const std::string_view path_sv(line.data(), t1);
            const std::string_view size_sv(line.data() + t1 + 1U, t2 - (t1 + 1U));
            const std::string_view mtime_sv(line.data() + t2 + 1U, line.size() - (t2 + 1U));
            if (path_sv.empty() || size_sv.empty() || mtime_sv.empty()) {
                continue;
            }
            const auto norm = normalize_path(path_sv);
            if (!norm || norm->empty() || *norm == "manifest.tsv") {
                continue;
            }
            std::uint64_t size = 0;
            const auto [sptr, sec] =
                std::from_chars(size_sv.data(), size_sv.data() + size_sv.size(), size);
            if (sec != std::errc{} || sptr != size_sv.data() + size_sv.size()) {
                continue;
            }
            std::int64_t mtime_ms = 0;
            const auto [mptr, mec] =
                std::from_chars(mtime_sv.data(), mtime_sv.data() + mtime_sv.size(), mtime_ms);
            if (mec != std::errc{} || mptr != mtime_sv.data() + mtime_sv.size()) {
                continue;
            }
            manifest_[*norm] = Stamp{size, mtime_ms};
        }
    }

    void store_copy(const std::string& norm,
                    std::uint64_t size,
                    std::int64_t mtime_ms,
                    const void* data,
                    std::size_t byte_count) {
        const std::lock_guard<std::mutex> copy_lock(copy_mu_);

        // Another thread may have finished copying this exact version while we read `inner_`.
        const std::filesystem::path dst = local_path(cache_root_, norm);
        {
            const std::lock_guard<std::mutex> lock(mu_);
            const auto it = manifest_.find(norm);
            if (it != manifest_.end() && it->second.size == size &&
                it->second.mtime_ms == mtime_ms) {
                std::error_code ec;
                if (std::filesystem::is_regular_file(dst, ec) && !ec &&
                    std::filesystem::file_size(dst, ec) == size && !ec) {
                    return;
                }
            }
        }

        std::error_code ec;
        std::filesystem::create_directories(dst.parent_path(), ec);
        if (ec) {
            return;
        }

        std::filesystem::path tmp = dst;
        tmp += ".tmp";
        {
            std::ofstream ofs(tmp, std::ios::binary | std::ios::trunc);
            if (!ofs.is_open()) {
                return;
            }
            if (byte_count > 0) {
                if (byte_count > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
                    ofs.close();
                    std::filesystem::remove(tmp, ec);
                    return;
                }
                ofs.write(static_cast<const char*>(data),
                          static_cast<std::streamsize>(byte_count));
            }
            ofs.flush();
            if (!ofs.good()) {
                ofs.close();
                std::filesystem::remove(tmp, ec);
                return;
            }
        }

        std::filesystem::rename(tmp, dst, ec);
        if (ec) {
            std::filesystem::remove(tmp, ec);
            return;
        }

        {
            const std::lock_guard<std::mutex> lock(mu_);
            std::ofstream mfs(cache_root_ / "manifest.tsv", std::ios::binary | std::ios::app);
            if (!mfs.is_open()) {
                return;
            }
            mfs << norm << '\t' << size << '\t' << mtime_ms << '\n';
            mfs.flush();
            if (!mfs.good()) {
                return;
            }
            manifest_[norm] = Stamp{size, mtime_ms};
        }

        copies_.fetch_add(1, std::memory_order_relaxed);
        copied_bytes_.fetch_add(static_cast<std::uint64_t>(byte_count),
                                 std::memory_order_relaxed);
    }

    template <typename Out>
    static bool read_local_file(const std::filesystem::path& path,
                                std::uint64_t expected_size,
                                Out& out) {
        if (expected_size >
            static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max())) {
            return false;
        }
        std::ifstream in(path, std::ios::binary | std::ios::ate);
        if (!in.is_open()) {
            return false;
        }
        const std::ifstream::pos_type end_pos = in.tellg();
        if (end_pos < 0 || static_cast<std::uint64_t>(end_pos) != expected_size) {
            return false;
        }
        in.seekg(0, std::ios::beg);
        if (!in) {
            return false;
        }
        out.resize(static_cast<std::size_t>(expected_size));
        if (expected_size > 0) {
            in.read(reinterpret_cast<char*>(out.data()),
                    static_cast<std::streamsize>(expected_size));
            if (!in || static_cast<std::uint64_t>(in.gcount()) != expected_size) {
                out.clear();
                return false;
            }
        }
        return true;
    }

    static std::filesystem::path local_path(const std::filesystem::path& root,
                                            std::string_view norm) {
        std::filesystem::path p = root;
        std::size_t pos = 0;
        while (pos < norm.size()) {
            const std::size_t slash = norm.find('/', pos);
            const std::string_view comp =
                (slash == std::string_view::npos) ? norm.substr(pos) : norm.substr(pos, slash - pos);
            if (!comp.empty()) {
                p /= comp;
            }
            if (slash == std::string_view::npos) {
                break;
            }
            pos = slash + 1U;
        }
        return p;
    }

    Vfs& inner_;
    std::filesystem::path cache_root_;
    std::mutex mu_;
    std::mutex copy_mu_;
    std::unordered_map<std::string, Stamp> manifest_;
    std::atomic<std::uint64_t> hits_{0};
    std::atomic<std::uint64_t> copies_{0};
    std::atomic<std::uint64_t> copied_bytes_{0};
};

}  // namespace wse::platform
