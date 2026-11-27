// Portable guest-VA sampling profile for R9 PGO (`docs/RECOMP_PLAN.md` "R9"): maps host PC
// samples back to guest VAs so a profile recorded on WSL or on-device drives hot-5% -O3 /
// cold-95% -O1 batching on any target. Header-only.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace wse::recomp {

struct PgoProfile {
    std::map<std::uint64_t, std::uint64_t> hits;
    std::uint64_t total_samples = 0;
};

[[nodiscard]] inline bool load_pgo_profile(const std::string& path, PgoProfile& out, std::string* err = nullptr) {
    out = PgoProfile{};
    std::ifstream in(path);
    if (!in) {
        if (err != nullptr) *err = "cannot open profile: " + path;
        return false;
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        std::string va_str;
        std::uint64_t count = 0;
        if (!(iss >> va_str >> count) || count == 0) continue;
        std::uint64_t va = 0;
        try {
            va = std::stoull(va_str, nullptr, 16);
        } catch (...) {
            continue;
        }
        if (va == 0) continue;
        out.hits[va] += count;
        out.total_samples += count;
    }
    if (out.hits.empty()) {
        if (err != nullptr) *err = "profile contains no samples";
        return false;
    }
    return true;
}

[[nodiscard]] inline bool save_pgo_profile(const std::string& path, const PgoProfile& prof, std::string* err = nullptr) {
    namespace fs = std::filesystem;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> sorted;
    sorted.reserve(prof.hits.size());
    std::uint64_t total = 0;
    for (const auto& [va, count] : prof.hits) {
        if (count == 0) continue;
        sorted.emplace_back(va, count);
        total += count;
    }
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second) return a.second > b.second;
        return a.first < b.first;
    });
    const fs::path dst(path);
    std::error_code ec;
    if (!dst.parent_path().empty()) fs::create_directories(dst.parent_path(), ec);
    const fs::path tmp(path + ".tmp");
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) {
            if (err != nullptr) *err = "cannot write profile: " + path;
            return false;
        }
        out << "# wse_pgo v1 samples=" << total << " functions=" << sorted.size() << "\n";
        for (const auto& [va, count] : sorted) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "0x%llx %llu\n", static_cast<unsigned long long>(va),
                          static_cast<unsigned long long>(count));
            out << buf;
        }
        if (!out) {
            if (err != nullptr) *err = "flush failed for profile: " + path;
            return false;
        }
    }
    fs::rename(tmp, dst, ec);
    if (ec) {
        fs::remove(tmp, ec);
        if (err != nullptr) *err = "rename failed for profile: " + ec.message();
        return false;
    }
    return true;
}

[[nodiscard]] inline std::set<std::uint64_t> select_pgo_hot_vas(const PgoProfile& prof,
                                                                std::size_t total_functions,
                                                                double coverage_target = 0.85,
                                                                double max_fraction = 0.05) {
    std::set<std::uint64_t> hot;
    if (prof.hits.empty() || prof.total_samples == 0) return hot;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> sorted(prof.hits.begin(), prof.hits.end());
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second) return a.second > b.second;
        return a.first < b.first;
    });
    std::size_t max_count = static_cast<std::size_t>(static_cast<double>(std::max<std::size_t>(total_functions, sorted.size())) *
                                                     std::clamp(max_fraction, 0.01, 1.0));
    if (sorted.size() >= 32U) max_count = std::max<std::size_t>(max_count, 32U);
    max_count = std::max<std::size_t>(max_count, 1U);
    const std::uint64_t target_hits = static_cast<std::uint64_t>(
        static_cast<double>(prof.total_samples) * std::clamp(coverage_target, 0.1, 1.0));
    std::uint64_t acc = 0;
    for (const auto& [va, count] : sorted) {
        if (hot.size() >= max_count) break;
        hot.insert(va);
        acc += count;
        if (acc >= target_hits && hot.size() >= 16U) break;
    }
    return hot;
}

[[nodiscard]] inline std::string pgo_hot_set_hash(const std::set<std::uint64_t>& hot_vas) {
    std::uint64_t h = 0xCBF29CE484222325ULL;
    for (const std::uint64_t va : hot_vas) {
        h = (h ^ va) * 0x100000001B3ULL;
    }
    char buf[24];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(h));
    return buf;
}

}  // namespace wse::recomp
