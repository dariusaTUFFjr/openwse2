// Deterministic scripted-input driver for Phase 5 R7 headless/Lavapipe soak runs
// (`docs/RECOMP_PLAN.md` "R7 — WSL x86-64 Bring-Up & Savegame Oracle").
//
// Reads a simple line-oriented script and feeds keyboard, mouse, cursor, and text events into
// `RecompInputHub` once per presented frame, waiting on frame counts or open game windows
// (`mbGameScreen::m_openWindows`). Header-only and STL-only.

#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "wse/recomp/input_hub.h"

namespace wse::recomp::shim {

enum class ScriptOp : std::uint8_t {
    wait_frames = 0,
    wait_window,
    move,
    click,
    drag,
    key_down,
    key_up,
    key_tap,
    text,
    mouse_delta,
    wheel,
    dump_frame,
    log_msg,
    stop,
};

struct ScriptStep {
    ScriptOp op = ScriptOp::wait_frames;
    int a = 0;
    int b = 0;
    int c = 0;
    int d = 0;
    int e = 0;
    std::uint8_t dik = 0U;
    std::uint32_t vk = 0U;
    std::string text{};
};

namespace script_detail {

inline std::string upper_ascii(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (char ch : s) {
        out.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(ch))));
    }
    return out;
}

inline std::string unescape_text(std::string_view raw) {
    std::string out;
    out.reserve(raw.size());
    for (std::size_t i = 0U; i < raw.size(); ++i) {
        if (raw[i] == '\\' && i + 1U < raw.size()) {
            const char nxt = raw[i + 1U];
            if (nxt == 'n') {
                out.push_back('\n');
                ++i;
                continue;
            }
            if (nxt == 'b') {
                out.push_back('\b');
                ++i;
                continue;
            }
            if (nxt == '\\') {
                out.push_back('\\');
                ++i;
                continue;
            }
        }
        out.push_back(raw[i]);
    }
    return out;
}

inline bool lookup_key_name(std::string_view token, std::uint8_t& out_dik, std::uint32_t& out_vk) noexcept {
    const std::string u = upper_ascii(token);
    if (u.empty()) return false;
    if (u.size() == 1U) {
        const char ch = u[0];
        if (ch >= 'A' && ch <= 'Z') {
            static constexpr std::uint8_t kAlphaDik[26] = {
                dik::kA, dik::kB, dik::kC, dik::kD, dik::kE, dik::kF, dik::kG, dik::kH, dik::kI,
                dik::kJ, dik::kK, dik::kL, dik::kM, dik::kN, dik::kO, dik::kP, dik::kQ, dik::kR,
                dik::kS, dik::kT, dik::kU, dik::kV, dik::kW, dik::kX, dik::kY, dik::kZ};
            out_dik = kAlphaDik[ch - 'A'];
            out_vk = static_cast<std::uint32_t>(ch);
            return true;
        }
        if (ch >= '0' && ch <= '9') {
            out_dik = (ch == '0') ? dik::k0 : static_cast<std::uint8_t>(dik::k1 + (ch - '1'));
            out_vk = static_cast<std::uint32_t>(ch);
            return true;
        }
    }
    struct NamedKey {
        std::string_view name;
        std::uint8_t dik_code;
        std::uint32_t vk_code;
    };
    static constexpr NamedKey kTable[] = {
        {"ESC", dik::kEscape, 0x1BU},       {"ESCAPE", dik::kEscape, 0x1BU},
        {"BACKSPACE", dik::kBackspace, 0x08U}, {"BS", dik::kBackspace, 0x08U},
        {"TAB", dik::kTab, 0x09U},          {"ENTER", dik::kReturn, 0x0DU},
        {"RETURN", dik::kReturn, 0x0DU},    {"SPACE", dik::kSpace, 0x20U},
        {"LCTRL", dik::kLCtrl, 0x11U},      {"CTRL", dik::kLCtrl, 0x11U},
        {"RCTRL", dik::kRCtrl, 0x11U},      {"LSHIFT", dik::kLShift, 0x10U},
        {"SHIFT", dik::kLShift, 0x10U},     {"RSHIFT", dik::kRShift, 0x10U},
        {"LALT", dik::kLAlt, 0x12U},        {"ALT", dik::kLAlt, 0x12U},
        {"RALT", dik::kRAlt, 0x12U},        {"GRAVE", dik::kGrave, 0xC0U},
        {"TILDE", dik::kGrave, 0xC0U},      {"MINUS", dik::kMinus, 0xBDU},
        {"EQUALS", dik::kEquals, 0xBBU},    {"COMMA", dik::kComma, 0xBCU},
        {"PERIOD", dik::kPeriod, 0xBEU},    {"SLASH", dik::kSlash, 0xBFU},
        {"F1", dik::kF1, 0x70U},            {"F2", dik::kF2, 0x71U},
        {"F3", dik::kF3, 0x72U},            {"F4", dik::kF4, 0x73U},
        {"F5", dik::kF5, 0x74U},            {"F6", dik::kF6, 0x75U},
        {"F7", dik::kF7, 0x76U},            {"F8", dik::kF8, 0x77U},
        {"F9", dik::kF9, 0x78U},            {"F10", dik::kF10, 0x79U},
        {"F11", dik::kF11, 0x7AU},          {"F12", dik::kF12, 0x7BU},
        {"HOME", dik::kHome, 0x24U},        {"UP", dik::kUp, 0x26U},
        {"PGUP", dik::kPgUp, 0x21U},        {"LEFT", dik::kLeft, 0x25U},
        {"RIGHT", dik::kRight, 0x27U},      {"END", dik::kEnd, 0x23U},
        {"DOWN", dik::kDown, 0x28U},        {"PGDOWN", dik::kPgDown, 0x22U},
        {"INSERT", dik::kInsert, 0x2DU},    {"DELETE", dik::kDelete, 0x2EU},
    };
    for (const auto& entry : kTable) {
        if (u == entry.name) {
            out_dik = entry.dik_code;
            out_vk = entry.vk_code;
            return true;
        }
    }
    if (u.size() > 2U && u[0] == '0' && u[1] == 'X') {
        char* endp = nullptr;
        const unsigned long v = std::strtoul(u.c_str(), &endp, 16);
        if (endp != nullptr && *endp == '\0' && v > 0UL && v <= 255UL) {
            out_dik = static_cast<std::uint8_t>(v);
            out_vk = 0U;
            return true;
        }
    }
    return false;
}

}  // namespace script_detail

class ScriptedInputPlayer {
public:
    bool load_text(std::string_view script, std::string* error_out = nullptr) {
        steps_.clear();
        reset();
        std::size_t pos = 0U;
        std::size_t line_no = 0U;
        while (pos < script.size()) {
            std::size_t end = script.find('\n', pos);
            if (end == std::string_view::npos) end = script.size();
            std::string line(script.substr(pos, end - pos));
            pos = (end < script.size()) ? end + 1U : end;
            ++line_no;
            if (!line.empty() && line.back() == '\r') line.pop_back();
            // Strip leading whitespace and full-line comments.
            std::size_t first = 0U;
            while (first < line.size() && (line[first] == ' ' || line[first] == '\t')) ++first;
            if (first >= line.size() || line[first] == '#') continue;

            std::istringstream iss(line.substr(first));
            std::string cmd;
            iss >> cmd;
            const std::string ucmd = script_detail::upper_ascii(cmd);

            ScriptStep st{};
            if (ucmd == "WAIT_FRAMES" || ucmd == "SOAK") {
                int frames = 0;
                if (!(iss >> frames) || frames < 0) {
                    return fail(error_out, line_no, "invalid frame count");
                }
                st.op = ScriptOp::wait_frames;
                st.a = frames;
            } else if (ucmd == "WAIT_WINDOW") {
                int win = -1;
                int timeout = 15000;
                if (!(iss >> win)) {
                    return fail(error_out, line_no, "missing window slot");
                }
                if (!(iss >> timeout)) timeout = 15000;
                st.op = ScriptOp::wait_window;
                st.a = win;
                st.b = std::max(1, timeout);
            } else if (ucmd == "MOVE") {
                if (!(iss >> st.a >> st.b)) return fail(error_out, line_no, "move expects <x> <y>");
                st.op = ScriptOp::move;
            } else if (ucmd == "CLICK" || ucmd == "RCLICK") {
                if (!(iss >> st.a >> st.b)) return fail(error_out, line_no, "click expects <x> <y> [btn]");
                int btn = (ucmd == "RCLICK") ? 1 : 0;
                if (ucmd == "CLICK") {
                    int parsed_btn = 0;
                    if (iss >> parsed_btn) btn = std::clamp(parsed_btn, 0, 2);
                }
                st.op = ScriptOp::click;
                st.c = btn;
            } else if (ucmd == "DRAG") {
                int frames = 10;
                if (!(iss >> st.a >> st.b >> st.c >> st.d)) {
                    return fail(error_out, line_no, "drag expects <x1> <y1> <x2> <y2> [frames]");
                }
                if (!(iss >> frames)) frames = 10;
                st.op = ScriptOp::drag;
                st.e = std::max(4, frames);
            } else if (ucmd == "KEY_DOWN" || ucmd == "KEY_UP" || ucmd == "KEY_TAP") {
                std::string kname;
                if (!(iss >> kname) || !script_detail::lookup_key_name(kname, st.dik, st.vk)) {
                    return fail(error_out, line_no, "unknown key name: " + kname);
                }
                if (ucmd == "KEY_DOWN") {
                    st.op = ScriptOp::key_down;
                } else if (ucmd == "KEY_UP") {
                    st.op = ScriptOp::key_up;
                } else {
                    int hold = 2;
                    if (!(iss >> hold)) hold = 2;
                    st.op = ScriptOp::key_tap;
                    st.a = std::max(1, hold);
                }
            } else if (ucmd == "TEXT") {
                std::string rest;
                std::getline(iss, rest);
                std::size_t s0 = 0U;
                while (s0 < rest.size() && (rest[s0] == ' ' || rest[s0] == '\t')) ++s0;
                st.op = ScriptOp::text;
                st.text = script_detail::unescape_text(std::string_view(rest).substr(s0));
            } else if (ucmd == "MOUSE_DELTA") {
                int rep = 1;
                if (!(iss >> st.a >> st.b)) return fail(error_out, line_no, "mouse_delta expects <dx> <dy> [frames]");
                if (!(iss >> rep)) rep = 1;
                st.op = ScriptOp::mouse_delta;
                st.c = std::max(1, rep);
            } else if (ucmd == "WHEEL") {
                if (!(iss >> st.a)) return fail(error_out, line_no, "wheel expects <delta>");
                st.op = ScriptOp::wheel;
            } else if (ucmd == "DUMP_FRAME") {
                if (!(iss >> st.text) || st.text.empty()) {
                    return fail(error_out, line_no, "dump_frame expects <tag>");
                }
                st.op = ScriptOp::dump_frame;
            } else if (ucmd == "LOG") {
                std::string rest;
                std::getline(iss, rest);
                std::size_t s0 = 0U;
                while (s0 < rest.size() && (rest[s0] == ' ' || rest[s0] == '\t')) ++s0;
                st.op = ScriptOp::log_msg;
                st.text = rest.substr(s0);
            } else if (ucmd == "STOP") {
                st.op = ScriptOp::stop;
            } else {
                return fail(error_out, line_no, "unknown script command: " + cmd);
            }
            steps_.push_back(std::move(st));
        }
        return true;
    }

    bool load_file(const std::string& path, std::string* error_out = nullptr) {
        std::ifstream in(path);
        if (!in) {
            if (error_out != nullptr) *error_out = "cannot open script file: " + path;
            return false;
        }
        const std::string txt((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        return load_text(txt, error_out);
    }

    void reset() noexcept {
        pc_ = 0U;
        sub_ = 0;
        finished_ = false;
        timed_out_ = false;
        timed_out_window_ = -1;
        pending_dump_tag_.clear();
    }

    [[nodiscard]] bool empty() const noexcept { return steps_.empty(); }
    [[nodiscard]] bool finished() const noexcept { return finished_; }
    [[nodiscard]] bool timed_out() const noexcept { return timed_out_; }
    [[nodiscard]] int timed_out_window() const noexcept { return timed_out_window_; }
    [[nodiscard]] bool has_pending_dump() const noexcept { return !pending_dump_tag_.empty(); }
    [[nodiscard]] std::size_t step_count() const noexcept { return steps_.size(); }
    [[nodiscard]] std::size_t pc() const noexcept { return pc_; }

    // Pops the frame-dump tag scheduled on the previous frame (so callers can read back the
    // freshly rendered Present output and write `frame_<tag>.ppm`).
    std::string take_pending_dump_tag() {
        std::string out = std::move(pending_dump_tag_);
        pending_dump_tag_.clear();
        return out;
    }

    // Executes scripted input for the upcoming frame. Zero-frame commands (`log`, met
    // `wait_window`, `dump_frame` scheduling) chain immediately within one frame up to a small cap.
    void step_frame(RecompInputHub& hub,
                    std::int32_t open_window,
                    std::string& out_log_line,
                    bool& out_stop) {
        out_log_line.clear();
        out_stop = false;
        if (finished_) {
            out_stop = true;
            return;
        }

        for (int chain = 0; chain < 32 && pc_ < steps_.size(); ++chain) {
            const ScriptStep& s = steps_[pc_];
            switch (s.op) {
            case ScriptOp::wait_frames:
                if (sub_ >= s.a) {
                    advance();
                    continue;
                }
                ++sub_;
                return;

            case ScriptOp::wait_window:
                if (open_window == s.a) {
                    advance();
                    continue;
                }
                if (sub_ >= s.b) {
                    timed_out_ = true;
                    timed_out_window_ = s.a;
                    finished_ = true;
                    out_stop = true;
                    return;
                }
                ++sub_;
                return;

            case ScriptOp::move:
                hub.inject_mouse_move_abs(s.a, s.b);
                advance();
                return;

            case ScriptOp::click:
                if (sub_ == 0) {
                    hub.inject_mouse_move_abs(s.a, s.b);
                    ++sub_;
                } else if (sub_ == 1) {
                    hub.inject_mouse_button(s.c, true);
                    ++sub_;
                } else if (sub_ == 2) {
                    ++sub_;
                } else {
                    hub.inject_mouse_button(s.c, false);
                    advance();
                }
                return;

            case ScriptOp::drag: {
                const int total = std::max(4, s.e);
                if (sub_ == 0) {
                    hub.inject_mouse_move_abs(s.a, s.b);
                    ++sub_;
                } else if (sub_ == 1) {
                    hub.inject_mouse_button(0, true);
                    ++sub_;
                } else if (sub_ + 1 < total) {
                    const float t = static_cast<float>(sub_ - 1) / static_cast<float>(total - 2);
                    const int ix = s.a + static_cast<int>((s.c - s.a) * t);
                    const int iy = s.b + static_cast<int>((s.d - s.b) * t);
                    hub.inject_mouse_move_abs(ix, iy);
                    ++sub_;
                } else {
                    hub.inject_mouse_move_abs(s.c, s.d);
                    hub.inject_mouse_button(0, false);
                    advance();
                }
                return;
            }

            case ScriptOp::key_down:
                hub.inject_key(s.dik, true, s.vk);
                advance();
                return;

            case ScriptOp::key_up:
                hub.inject_key(s.dik, false, s.vk);
                advance();
                return;

            case ScriptOp::key_tap:
                if (sub_ == 0) {
                    hub.inject_key(s.dik, true, s.vk);
                    ++sub_;
                } else if (sub_ < s.a) {
                    ++sub_;
                } else {
                    hub.inject_key(s.dik, false, s.vk);
                    advance();
                }
                return;

            case ScriptOp::text:
                hub.inject_text_utf8(s.text);
                advance();
                return;

            case ScriptOp::mouse_delta:
                hub.inject_mouse_delta(s.a, s.b);
                if (++sub_ >= s.c) {
                    advance();
                }
                return;

            case ScriptOp::wheel:
                hub.inject_mouse_wheel(s.a);
                advance();
                return;

            case ScriptOp::dump_frame:
                pending_dump_tag_ = s.text;
                advance();
                return;

            case ScriptOp::log_msg:
                if (!out_log_line.empty()) out_log_line.push_back('\n');
                out_log_line += s.text;
                advance();
                continue;

            case ScriptOp::stop:
                finished_ = true;
                out_stop = true;
                return;
            }
        }
        if (pc_ >= steps_.size()) {
            finished_ = true;
            out_stop = true;
        }
    }

private:
    static bool fail(std::string* error_out, std::size_t line_no, const std::string& msg) {
        if (error_out != nullptr) {
            *error_out = "line " + std::to_string(line_no) + ": " + msg;
        }
        return false;
    }

    void advance() noexcept {
        ++pc_;
        sub_ = 0;
    }

    std::vector<ScriptStep> steps_;
    std::size_t pc_ = 0U;
    int sub_ = 0;
    bool finished_ = false;
    bool timed_out_ = false;
    int timed_out_window_ = -1;
    std::string pending_dump_tag_;
};

}  // namespace wse::recomp::shim
