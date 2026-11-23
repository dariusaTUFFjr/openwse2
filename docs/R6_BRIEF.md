# R6 brief: sound, input, touch controls (for the agent doing R6)

Written 2026-11-23 for the R6 work the user runs with Antigravity. Read `AGENTS.md` first, then
`STATUS.md`, then `docs/RECOMP_PLAN.md` "R6 — Audio, Input, Touch Controls & Desktop Mode" (the
design and its "done when") and `docs/PLATFORM_PLAN.md` "P14" (touch QoL details). This file only
says where things stand, what to build on, and what to ask; it does not replace those.

## Where things stand

- The phone runs the user's own WSE2 v1.1.5.6 exe, translated to ARM64 (Phase 5). It reaches
  Warband's main menu at 60 fps on PanVK and draws at the phone's 2412 x 1080 (Mali-G68, OPPO Reno
  11, 12 GB). Nothing can be clicked yet: there is no input and no sound.
- Start from branch `pw/r5-perf` at `bbf4531` or later. Work on your own branch; push only to it.
- The game only reaches the main menu in tests, because nothing can click "Start". R6's input is
  what lets later checks reach the campaign map and battles.

## What already exists (build on it, do not rebuild it)

| Piece | Where | State |
| --- | --- | --- |
| FMOD stand-in | `engine/include/wse/recomp/shim_fmod_steam.h` | The exe's 54 `fmodex64.dll` imports answer with valid handles and FMOD_OK; nothing plays. R6 makes these play. |
| Audio mixer | `engine/include/wse/platform/audio.h` (`WseAudioMixer`), `audio_worker.h`, `wse/asset/audio*.h` (WAV/OGG/MP3 decoders) | Built and tested in Phase 3/4 (CMake `audio_test`). The design shims FMOD onto it (`docs/RECOMP_PLAN.md` R6). |
| DirectInput | `engine/include/wse/recomp/shim_d3d9.h` (`direct_input_create`, `GetDeviceState`, `GetDeviceData`, around line 2170) | Devices exist; they report nothing pressed and no events. |
| XInput | `engine/include/wse/recomp/shim_system.h` (`XINPUT1_3` `#2`) | Answers "no controller". |
| Windows messages | `engine/include/wse/recomp/shim_user32.h` | Message queue (`PeekMessageW`), `GetCursorPos`, `ScreenToClient`, `ClipCursor`. Input as `WM_*` messages goes in here. |
| Input model | `engine/include/wse/platform/input.h` | Key names and game keys from WSE2 evidence (`mbGameKeyManager`), touch overlay layout and profiles (P7). |
| Touch overlay drawing | `engine/include/wse/platform/touch_overlay_render.h` | Procedural shapes (P11.3). Not yet drawn in the game. |
| Where the overlay draws | `platform/vulkan/d3d9_vulkan.cpp`, the present path (`run_postfx_passes_locked`, then the copy to the screen) | The overlay goes in as a final pass at full resolution, after FSR/sharpening, so it is never upscaled or blurred. |
| Menu vs 3D | The game calls `IDirect3DDevice9::ShowCursor` (recorded in `shim_d3d9.h`) | The universal signal for "menu" (cursor shown) vs "3D" (hidden); `docs/RECOMP_PLAN.md` R6 describes the hybrid detector. |
| Background pause | `platform/android/r4_jni.cpp` (`nativeSetPaused`, the per-frame callback) | Done: the game waits at its next frame while the app is in the background. Pause sound there too. |
| Game screen and its Java side | `android/.../RecompGameActivity.kt`, `R4Native.kt`, `r4_jni.cpp` (`nativeRunVulkan`) | Touch events arrive in the activity; pass them to native the way `nativeSetSurface` / `nativeSetPaused` do. |

## Rules that matter most here

1. **Never edit the translator's files:** `engine/src/recomp/r4_startup.cpp`, `lifter_llvm.cpp`,
   `lift_r1*`, `discover.*`, `llvm_common.h`, `recomp_llvm.h`, and
   `engine/include/wse/recomp/{cpu_state,x86_decode,x86_interp,pe_image,runtime,seh}.h`. Any change
   there changes the translator fingerprint and makes every phone retranslate the game (an hour or
   more). Sound and input go through the shims, which are outside it.
2. **Plain player words on screen.** No jargon in the app ("Sound", "Touch controls", "Vibrate on
   hits"), and errors say what to do.
3. **Phone performance is part of done.** Audio mixing and input must not cost frame rate or heat:
   mix on the audio thread, never on the game thread; no allocation per frame in the overlay pass.
4. **Check shims against real Windows on this PC before pushing** where it applies: the PC can run
   the real `fmodex64.dll` / DirectInput behaviour as the reference (`tools/r5_replay_d3d9.cpp` is
   the pattern for comparing against the real thing).
5. **Never commit retail files** (exe, PDB, game assets, captures, frames).
6. **Always look for blind spots and tell the user** (`AGENTS.md` section 0).

## Gates (run them; do not report success without seeing them pass)

- Windows host gate: `powershell -NoProfile -ExecutionPolicy Bypass -File engine/build_host.ps1`
  must end with `engine host tests: PASS` (24 suites, `selftest_host 24/24`). Only one full gate
  at a time on this PC (8 GB RAM); wait for at least 2 GB free before compiling.
- Linux: headers compile with clang 18 (`AGENTS.md` section 2). Windows treats `long` and
  `long long` as the same type and Linux does not: a mixed-type `std::clamp`/`std::min` passes
  locally and fails CI (it happened on 2026-11-22).
- CI on GitHub (`dariusaTUFFjr/openwse2`, push your branch): all 7 jobs green, including the
  Android APK. Kotlin and `r4_jni.cpp` are compiled only there. Install nothing on the phone
  before CI is green.
- New host tests for the FMOD shim (2D/3D playback, streaming, pause/resume) and for
  DirectInput/XInput/`WM_CHAR` injection (`docs/RECOMP_PLAN.md` R6 "Done when").

## Ask the user before building (their decisions, not yours)

- The touch layout for each scene (battle, campaign map, inventory/trade, party, dialog, menus):
  which controls, where, and how big. Show a picture or a mock-up first.
- Camera and movement feel: joystick vs drag-to-look, sensitivity, gyro aiming on or off by default.
- Combat haptics: on or off by default, and how strong.
- Sound: what to do with music and effects while the app is in the background, and the default
  volume.
- Gamepad and mouse/keyboard (DeX, external display): which comes first.
- Interview with multiple-choice questions; then write the decisions into `docs/RECOMP_PLAN.md`
  under R6 before writing code.

## Known traps

- Earlier delegated runs reported tests as passing when they had not run (a stale build, a
  timeout). Paste the actual `PASS` lines you saw.
- `std::thread` stacks on Android are about 1 MB; keep deep recursion and large stack arrays off
  worker threads.
- A process-wide signal (`setitimer`/`SIGALRM`) lands on Android's main thread, not yours: the
  phone check crashed from exactly that (fixed 2026-11-23). Use thread-directed timers.
- Retail files are read through Android's folder access (SAF) and cached in app storage
  (`caching_vfs.h`); sound files go through the same `Vfs`, never direct paths.
