# R7 brief: the whole game in WSL, and saves that match Windows (for the agent doing R7)

Written 2026-11-25 for the R7 work the user runs with Antigravity; Claude reviews the result. Read
`AGENTS.md` first (sections 0, 1, 2 and 5 matter most here), then `STATUS.md`, then
`docs/RECOMP_PLAN.md` "R7 — WSL x86-64 Bring-Up & Savegame Oracle" (the design and its "done
when"). This file only says where things stand, what to build on, what to ask, and what went wrong
before; it does not replace those.

## Where things stand

- The phone runs the user's own WSE2 v1.1.5.6 exe translated to ARM64, reaches the main menu on
  screen, and has sound, touch, gamepad and keyboard/mouse input (R6). It has never gone past the
  main menu, on the phone or anywhere else.
- WSL Ubuntu on this PC already translates and runs the same exe for x86-64: `tools/r4_run.cpp`
  (`--work`, `--module`, `--frames`, `--jobs`, `--reuse`, `--capture`), startup to the main loop on
  the recording D3D9 device with no screen (3,000 frames in 450 s). There is no live window, no
  input and no sound in WSL yet.
- Start from branch `pw/r6` at `d158015` or later. Work on your own branch (`pw/r7`); push only to it.

## What already exists (build on it, do not rebuild it)

| Piece | Where | State |
| --- | --- | --- |
| WSL front end | `tools/r4_run.cpp` + `engine/src/recomp/r4_startup.{h,cpp}` | Translates and runs the exe and `lua5164`; logs each import's first call; `--capture` records D3D9 calls. |
| WSL build | `tools/wsl_gate.sh`, build tree `/root/wse/build-recomp` | Recompiler gate and CTest in WSL (`AGENTS.md` section 5 "WSL"). |
| D3D9 on Vulkan | `platform/vulkan/d3d9_vulkan.{h,cpp}` (`VulkanD3d9Backend`) | Draws the game on the phone; `tools/r5_replay_vulkan.cpp` replays captures on any Vulkan driver, including Lavapipe and the mock ICD. |
| Reference frames | `tools/r5_replay_d3d9.cpp` | Replays a capture on the real `d3d9.dll` on Windows: the "what it should look like" for any frame. |
| Input | `engine/include/wse/recomp/input_hub.h` (`RecompInputHub`), DirectInput/XInput in `shim_d3d9.h` / `shim_system.h`, `WM_*` in `shim_user32.h` | Built in R6 for Android; the same hub can be fed from SDL3 on Linux or from a script. |
| Sound | `engine/include/wse/recomp/shim_fmod_steam.h` (`FmodSteam`) with any `AudioDevice` | Built in R6; on Linux, `CaptureAudioDevice` (no speakers needed) or SDL3 audio. |
| Savegame reader | `engine/include/wse/game/savegame.h`, `wse/io/binary_stream.h` (Phase 2A, versions 802..1174) | The structural check the "done when" asks for. |
| Where saves go | `shim_files.h` (`SHGetFolderPathA`; the `C:\Users\Player` mount) -> the run's user `Vfs` (`<work>/user` in `r4_run`) | Documents and AppData land there, so the translated game's saves do too. |

## Rules that matter most here

1. **Do not change the translator fingerprint by accident.** Every file in `_wse_lifter_sources`
   (`CMakeLists.txt`: `engine/src/recomp/r4_startup.cpp`, `lifter_llvm.cpp`, `lift_r1*`,
   `discover.*`, `llvm_common.h`, `recomp_llvm.h`, and `engine/include/wse/recomp/{cpu_state,
   x86_decode,x86_interp,pe_image,runtime,seh}.h`) is hashed into the cache key. Editing any of
   them makes every phone retranslate the whole game (hours). Put R7 code in `tools/`, new
   headers, shims and front ends. If the translator truly needs a fix (a wrong instruction found in
   play), that is fine, but say so in the commit and in `STATUS.md`, and batch such fixes.
   `AGENTS.md` section 2 says how to compute the hash; check it before every push.
2. **Every game structure, offset or address you read must be checked against the exe and PDB.**
   Cite the function VA, the PDB type index and size (`docs/SYMBOL_MAP.json`,
   `tools/pdb_symbols.py`, capstone on the exe). Never infer a layout from what "looks like" a
   vector or a pointer: R6 read `g_gameScreen+0x0` as the window stack, and it is the window
   pointer array (`m_openWindows` is at `+0x130`); its window numbers were also wrong. `tools/
   check_citations.py --self-test --docs` must pass.
3. **WSL on this PC is small.** About 3 GB of RAM and 1 GB of swap; C: has a few GB free and WSL's
   disk image only grows. One WSL job at a time (a second build OOM-killed a translation once);
   translate with `--jobs 1`; restart an interrupted translation with `--reuse`; build at `-j2`.
   Check `df -h /mnt/c` before anything that writes a lot and stop near 2 GB free. Never build on
   Windows and in WSL at the same time.
4. **Retail code and data stay local.** Never commit the exe, PDB, game files, captures, frames,
   savegames (the user's or the game's), or anything translated from them. Saves for the oracle
   live in the user's own folders or `G:\My Drive\WSE2 Android\` (Drive for desktop, streaming;
   WSL cannot see `G:`, copy from Windows).
5. **Phone performance is still part of done.** R7 runs on the PC, but anything it adds to shared
   code (shims, input hub, D3D9 device) also runs on the phone: no new per-frame allocation, no
   new locks on the game thread, nothing that only works on x86.
6. **Always look for blind spots and tell the user** (`AGENTS.md` section 0), in plain words.

## Gates (run them; do not report success without seeing them pass)

- Windows host gate: `powershell -NoProfile -ExecutionPolicy Bypass -File engine/build_host.ps1`
  must end with `engine host tests: PASS` (24 suites, `selftest_host 24/24`). Only one full gate
  at a time on this PC.
- WSL: `wsl -e bash tools/wsl_gate.sh` (recompiler corpus `N/N exports match`, CTest all green)
  whenever you touch recompiler or shim code.
- CI on GitHub (`dariusaTUFFjr/openwse2`, push your branch): all 7 jobs green. A mixed
  `long`/`long long` `std::min`/`std::clamp` passes on Windows and fails on Linux.
- The R7 "done when" itself, run in WSL and written into `STATUS.md` with the real numbers: the
  route below with zero faults and zero imports without a shim, and the savegame oracle both ways.
- Paste the actual `PASS` lines and the run's `RESULT` line you saw. Earlier delegated runs reported
  passes that had not run (a stale build, a timeout).

## Ask the user before building (their decisions, not yours)

Interview with multiple-choice questions, recommended option first; then write the decisions into
`docs/RECOMP_PLAN.md` under R7 before writing code.

- **The route that counts as done:** main menu -> new character (or a quick start?) -> campaign
  map -> which town or village scene -> which battle (a bandit party? a real field battle?) ->
  save -> reload. How long must each part run?
- **How it is driven:** a live window you play yourself (WSLg Vulkan, SDL3 input), a scripted
  input file replayed into `RecompInputHub` (repeatable, works with no screen), or both. Scripted
  input is what makes the check repeatable.
- **Graphics in WSL:** try WSLg's Vulkan on the Intel HD 520 first, or go straight to Lavapipe
  (software Vulkan) with saved frames compared against `r5_replay_d3d9` reference frames?
- **The savegame oracle:** which saves (a fresh one from this run, one of the user's own Windows
  saves, both?), and what "loads cleanly" means: the game reaches the map with the same party,
  gold, day and position, or our `savegame.h` structural check only, or both.
- **Sound in WSL:** none (`CaptureAudioDevice`, check levels only) or real speakers through SDL3?
- **Debugging:** `gdb` in WSL on a translated crash needs the batches' symbols; how much tooling
  (a crash report with the guest call stack exists in `r4_startup.cpp`) is worth adding.

## Known traps

- The full x86-64 translation takes a long time on one thread, and one batch alone (window 38)
  needs about 2.4 GB, close to all of WSL's memory; three batches (windows 38, 39 and 30)
  dominate (`STATUS.md` section 5, "Translator update"). Keep the batch cache between runs (`--reuse`)
  and never delete `/root/wse/...` build or batch trees to "start clean" without asking.
- Gameplay will call Windows functions startup never did (character creation, battles, saving,
  file dialogs). Each one needs a shim with a test, and real Windows is the reference: compare
  against the real API on this PC (`AGENTS.md` section 5, the native oracle pattern), not against
  what you expect it to do.
- `std::thread` stacks are small; threads that run translated code must be `HostThread`s
  (`wse/recomp/host_stack.h`, 64 MiB). The phone crashed from exactly that on 2026-11-24.
- A process-wide signal (`setitimer`/`SIGALRM`) lands on whichever thread the kernel picks; use
  thread-directed timers.
- Do not switch commits under a running build (`AGENTS.md` section 5): mixed object files crash
  `selftest_host`.
- WSL commands from PowerShell go in a script file; inline `bash -c '...'` quoting breaks, and
  `tools/wsl_gate.sh` must keep LF line endings.
