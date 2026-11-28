package org.wse2.android.launcher

import org.json.JSONObject
import java.io.File

data class ModuleControlsPrefs(
    val touchOpacity: Float = 0.55f,
    val buttonScale: Float = 1.0f,
    val lookSensitivity: Float = 1.0f,
    val aimHold: Boolean = true,
    val gyroEnabled: Boolean = false,
    val showTouchControls: Boolean = true,
    val soundVolume: Float = 1.0f,
    val hapticsEnabled: Boolean = true,
    val hapticWhenHit: Boolean = true,
    val hapticWhenDealing: Boolean = true,
    val hapticStrength: Int = 60
)

data class ModulePicturePrefs(
    val resolutionIndex: Int = 0, // 0=Auto, 1=100%, 2=85%, 3=75%, 4=67%, 5=50%
    val sharpen: Boolean = true,
    val fxaa: Boolean = false,
    val targetFps: Int = 60
) {
    val autoResolution: Boolean get() = resolutionIndex == 0

    val fixedRenderScale: Float
        get() = when (resolutionIndex) {
            1 -> 1.0f
            2 -> 0.85f
            3 -> 0.75f
            4 -> 0.67f
            5 -> 0.50f
            else -> 1.0f
        }

    val sharpnessValue: Float get() = if (sharpen) 0.2f else 0.0f
}

/**
 * Per-module profile persistence under permanent `filesDir/wse2/user/profiles/<module>/`
 * (`docs/PLATFORM_PLAN.md` "P14.5").
 */
object ModuleProfileStore {
    private const val TOUCH_LAYOUT_FILE = "touch_layout.json"
    private const val CONTROLS_PREFS_FILE = "controls_prefs.json"
    private const val PICTURE_PREFS_FILE = "picture_prefs.json"

    fun loadTouchLayoutJson(filesDir: File, moduleName: String): String? {
        UserStorage.migrateLegacyUserDirIfNeeded(filesDir)
        val file = File(UserStorage.moduleProfileDir(filesDir, moduleName), TOUCH_LAYOUT_FILE)
        return try {
            file.takeIf { it.isFile && it.length() > 0L }?.readText(Charsets.UTF_8)
        } catch (_: Throwable) {
            null
        }
    }

    fun saveTouchLayoutJson(filesDir: File, moduleName: String, json: String): Boolean {
        val dir = UserStorage.moduleProfileDir(filesDir, moduleName)
        return writeAtomicText(File(dir, TOUCH_LAYOUT_FILE), json)
    }

    fun clearTouchLayout(filesDir: File, moduleName: String): Boolean {
        val file = File(UserStorage.moduleProfileDir(filesDir, moduleName), TOUCH_LAYOUT_FILE)
        return !file.exists() || file.delete()
    }

    fun loadControlsPrefs(
        filesDir: File,
        moduleName: String,
        fallback: ModuleControlsPrefs = ModuleControlsPrefs()
    ): ModuleControlsPrefs {
        UserStorage.migrateLegacyUserDirIfNeeded(filesDir)
        val file = File(UserStorage.moduleProfileDir(filesDir, moduleName), CONTROLS_PREFS_FILE)
        val raw = try {
            file.takeIf { it.isFile }?.readText(Charsets.UTF_8)
        } catch (_: Throwable) {
            null
        } ?: return fallback
        return try {
            val obj = JSONObject(raw)
            ModuleControlsPrefs(
                touchOpacity = obj.optDouble("touchOpacity", fallback.touchOpacity.toDouble()).toFloat().coerceIn(0.2f, 1.0f),
                buttonScale = obj.optDouble("buttonScale", fallback.buttonScale.toDouble()).toFloat().coerceIn(0.7f, 1.45f),
                lookSensitivity = obj.optDouble("lookSensitivity", fallback.lookSensitivity.toDouble()).toFloat().coerceIn(0.25f, 2.0f),
                aimHold = obj.optBoolean("aimHold", fallback.aimHold),
                gyroEnabled = obj.optBoolean("gyroEnabled", fallback.gyroEnabled),
                showTouchControls = obj.optBoolean("showTouchControls", fallback.showTouchControls),
                soundVolume = obj.optDouble("soundVolume", fallback.soundVolume.toDouble()).toFloat().coerceIn(0.0f, 1.0f),
                hapticsEnabled = obj.optBoolean("hapticsEnabled", fallback.hapticsEnabled),
                hapticWhenHit = obj.optBoolean("hapticWhenHit", fallback.hapticWhenHit),
                hapticWhenDealing = obj.optBoolean("hapticWhenDealing", fallback.hapticWhenDealing),
                hapticStrength = obj.optInt("hapticStrength", fallback.hapticStrength).coerceIn(10, 100)
            )
        } catch (_: Throwable) {
            fallback
        }
    }

    fun saveControlsPrefs(filesDir: File, moduleName: String, prefs: ModuleControlsPrefs): Boolean {
        val obj = JSONObject()
            .put("touchOpacity", prefs.touchOpacity.toDouble())
            .put("buttonScale", prefs.buttonScale.toDouble())
            .put("lookSensitivity", prefs.lookSensitivity.toDouble())
            .put("aimHold", prefs.aimHold)
            .put("gyroEnabled", prefs.gyroEnabled)
            .put("showTouchControls", prefs.showTouchControls)
            .put("soundVolume", prefs.soundVolume.toDouble())
            .put("hapticsEnabled", prefs.hapticsEnabled)
            .put("hapticWhenHit", prefs.hapticWhenHit)
            .put("hapticWhenDealing", prefs.hapticWhenDealing)
            .put("hapticStrength", prefs.hapticStrength)
        val dir = UserStorage.moduleProfileDir(filesDir, moduleName)
        return writeAtomicText(File(dir, CONTROLS_PREFS_FILE), obj.toString(2))
    }

    /** Whether the module has its own picture settings (saved from the in-game drawer). */
    fun hasPicturePrefs(filesDir: File, moduleName: String): Boolean =
        File(UserStorage.moduleProfileDir(filesDir, moduleName), PICTURE_PREFS_FILE).isFile

    fun loadPicturePrefs(
        filesDir: File,
        moduleName: String,
        fallback: ModulePicturePrefs = ModulePicturePrefs()
    ): ModulePicturePrefs {
        UserStorage.migrateLegacyUserDirIfNeeded(filesDir)
        val file = File(UserStorage.moduleProfileDir(filesDir, moduleName), PICTURE_PREFS_FILE)
        val raw = try {
            file.takeIf { it.isFile }?.readText(Charsets.UTF_8)
        } catch (_: Throwable) {
            null
        } ?: return fallback
        return try {
            val obj = JSONObject(raw)
            ModulePicturePrefs(
                resolutionIndex = obj.optInt("resolutionIndex", fallback.resolutionIndex).coerceIn(0, 5),
                sharpen = obj.optBoolean("sharpen", fallback.sharpen),
                fxaa = obj.optBoolean("fxaa", fallback.fxaa),
                targetFps = obj.optInt("targetFps", fallback.targetFps).coerceIn(30, 120)
            )
        } catch (_: Throwable) {
            fallback
        }
    }

    fun savePicturePrefs(filesDir: File, moduleName: String, prefs: ModulePicturePrefs): Boolean {
        val obj = JSONObject()
            .put("resolutionIndex", prefs.resolutionIndex.coerceIn(0, 5))
            .put("sharpen", prefs.sharpen)
            .put("fxaa", prefs.fxaa)
            .put("targetFps", prefs.targetFps.coerceIn(30, 120))
        val dir = UserStorage.moduleProfileDir(filesDir, moduleName)
        return writeAtomicText(File(dir, PICTURE_PREFS_FILE), obj.toString(2))
    }

    private fun writeAtomicText(target: File, text: String): Boolean {
        return try {
            target.parentFile?.mkdirs()
            val tmp = File(target.parentFile, "${target.name}.tmp")
            tmp.writeText(text, Charsets.UTF_8)
            if (!tmp.renameTo(target)) {
                tmp.copyTo(target, overwrite = true)
                tmp.delete()
            }
            true
        } catch (_: Throwable) {
            false
        }
    }
}
