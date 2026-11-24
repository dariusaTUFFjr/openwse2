package org.wse2.android.launcher

import android.view.Surface
import org.wse2.android.WseSafBridge

// Phase 5 R4 startup run and R5a live Vulkan run (docs/RECOMP_PLAN.md "R4 design", "R5 — Graphics"):
// `libwse_r0.so` (`platform/android/r4_jni.cpp`). Called only in the isolated `:recomp` process
// (`R4StartupService` or `RecompGameActivity`): the run maps the game at fixed addresses and ends
// the process when it is done.
object R4Native {
    interface HudListener {
        fun onHudUpdate(text: String)
        fun onHapticPulse(durationMs: Int, amplitude: Int) {}
        fun onUiAction(action: String) {}
    }

    val loadError: String? by lazy {
        try {
            System.loadLibrary("wse_r0")
            null
        } catch (t: Throwable) {
            t.message ?: t.javaClass.simpleName
        }
    }

    // Reads the install through `bridge` (the SAF folder) or, when `copyDir` is not empty, from
    // that directory; works in `workDir/r4`. Returns only when nothing could be started (the
    // report says why); otherwise the process ends when the run does.
    @JvmStatic
    external fun nativeRun(
        bridge: WseSafBridge?,
        copyDir: String,
        workDir: String,
        module: String,
        jobs: Int,
        frames: Long
    ): Int

    // Phase 5 R5a live Vulkan run on an Android `Surface` using the user's selected GPU driver
    // (`system` / `panvk` / `turnip`). Returns only on setup/Vulkan failure.
    @JvmStatic
    external fun nativeRunVulkan(
        bridge: WseSafBridge?,
        copyDir: String,
        workDir: String,
        module: String,
        jobs: Int,
        frames: Long,
        surface: Surface?,
        driverMode: String,
        customIcdPath: String,
        nativeLibDir: String,
        shaderCacheDir: String,
        maxTextureSize: Int,
        renderScale: Float,
        autoResolution: Boolean,
        sharpness: Float,
        fxaa: Boolean,
        hudListener: HudListener?
    ): Int

    // Updates or detaches (`surface == null`) the active Android presentation surface.
    @JvmStatic
    external fun nativeSetSurface(surface: Surface?)

    // Pauses or resumes the translated game loop at the next frame boundary (when backgrounded).
    @JvmStatic
    external fun nativeSetPaused(paused: Boolean)

    // Saves the pipeline cache, detaches the surface, and flushes the final driver + FPS report.
    @JvmStatic
    external fun nativeFinishReport()

    // Size in bytes of the copied game files under `<workDir>/r4/install-cache`.
    @JvmStatic
    external fun installCacheBytes(workDir: String): Long

    // Deletes the copied game files under `<workDir>/r4/install-cache`.
    @JvmStatic
    external fun clearInstallCache(workDir: String): Boolean

    // Size in bytes of the converted ETC2 textures under `<workDir>/r4/texcache`.
    @JvmStatic
    external fun textureCacheBytes(workDir: String): Long

    // Deletes the converted ETC2 textures under `<workDir>/r4/texcache`.
    @JvmStatic
    external fun clearTextureCache(workDir: String): Boolean

    // ---- Phase 5 R6: Sound, Input, Touch Controls, Gamepad & Desktop Mode ----

    @JvmStatic
    external fun nativeTouchEvent(pointerId: Int, action: Int, x: Float, y: Float)

    @JvmStatic
    external fun nativeKeyEvent(dikCode: Int, vkCode: Int, down: Boolean)

    @JvmStatic
    external fun nativeTextInput(utf8: String)

    @JvmStatic
    external fun nativeMouseMove(x: Int, y: Int, relative: Boolean)

    @JvmStatic
    external fun nativeMouseButton(button: Int, down: Boolean)

    @JvmStatic
    external fun nativeMouseWheel(delta: Int)

    @JvmStatic
    external fun nativeGamepadState(
        userIndex: Int,
        connected: Boolean,
        buttons: Int,
        leftTrigger: Int,
        rightTrigger: Int,
        thumbLx: Int,
        thumbLy: Int,
        thumbRx: Int,
        thumbRy: Int
    )

    @JvmStatic
    external fun nativeGyroSample(radPerSecX: Float, radPerSecY: Float, dtSec: Float, rotated270: Boolean)

    @JvmStatic
    external fun nativeSetSurfaceSize(width: Int, height: Int)

    @JvmStatic
    external fun nativeSetSafeArea(left: Float, top: Float, right: Float, bottom: Float)

    @JvmStatic
    external fun nativeSetTouchConfig(
        opacity: Float,
        buttonScale: Float,
        lookSensitivity: Float,
        aimHold: Boolean,
        gyroEnabled: Boolean,
        showTouch: Boolean
    )

    @JvmStatic
    external fun nativeSetAudioConfig(
        masterVolume: Float,
        hapticsEnabled: Boolean,
        whenHit: Boolean,
        whenDealing: Boolean,
        hapticStrength: Int
    )

    @JvmStatic
    external fun nativeIsCursorVisible(): Boolean
}
