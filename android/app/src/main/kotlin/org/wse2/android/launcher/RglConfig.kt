package org.wse2.android.launcher

import java.io.File
import java.util.Locale
import kotlin.math.abs
import kotlin.math.roundToInt

/**
 * WSE2's `rgl_config.ini` as its lines, kept verbatim. WSE2 v1.1.5.6 reads
 * `Documents/Mount&Blade Warband WSE2/rgl_config.ini` (`[Section]` headers, `key=value` lines,
 * `b`/`i`/`f`/`s`/`u`/`v` type prefixes) and never the old Warband `rgl_config.txt`.
 *
 * The file holds some 700 keys; the launcher edits a few dozen of them ([RglConfigState]) and
 * leaves every other line, its order and its line endings as WSE2 wrote them.
 */
data class RglIni(val lines: List<String> = emptyList(), val lineEnding: String = "\r\n") {

    val keyCount: Int
        get() = lines.count { line ->
            val t = line.trim()
            t.isNotEmpty() && !t.startsWith("[") && !t.startsWith(";") && !t.startsWith("#") && t.indexOf('=') > 0
        }

    fun get(section: String, key: String): String? {
        var current = ""
        for (line in lines) {
            val t = line.trim()
            val header = sectionName(t)
            if (header != null) {
                current = header
                continue
            }
            if (!current.equals(section, ignoreCase = true)) continue
            val eq = t.indexOf('=')
            if (eq <= 0) continue
            if (t.substring(0, eq).trim().equals(key, ignoreCase = true)) {
                return t.substring(eq + 1).trim()
            }
        }
        return null
    }

    // Replaces the key's line in place, or adds it at the end of its section (a new section at
    // the end of the file when there is none).
    fun with(section: String, key: String, value: String): RglIni {
        val out = ArrayList<String>(lines.size + 2)
        var current = ""
        var sectionSeen = false
        var lastLineOfSection = -1
        var replaced = false
        for (line in lines) {
            val t = line.trim()
            val header = sectionName(t)
            if (header != null) {
                current = header
                if (header.equals(section, ignoreCase = true)) {
                    sectionSeen = true
                    lastLineOfSection = out.size
                }
                out.add(line)
                continue
            }
            if (current.equals(section, ignoreCase = true)) {
                val eq = t.indexOf('=')
                if (!replaced && eq > 0 && t.substring(0, eq).trim().equals(key, ignoreCase = true)) {
                    out.add("${t.substring(0, eq).trim()}=$value")
                    replaced = true
                    lastLineOfSection = out.size - 1
                    continue
                }
                if (t.isNotEmpty()) lastLineOfSection = out.size
            }
            out.add(line)
        }
        if (!replaced) {
            if (sectionSeen) {
                out.add(lastLineOfSection + 1, "$key=$value")
            } else {
                while (out.isNotEmpty() && out.last().isBlank()) out.removeAt(out.size - 1)
                out.add("[$section]")
                out.add("$key=$value")
            }
        }
        return copy(lines = out)
    }

    fun serialize(): String {
        if (lines.isEmpty()) return ""
        return lines.joinToString(lineEnding, postfix = lineEnding)
    }

    companion object {
        private fun sectionName(trimmed: String): String? =
            if (trimmed.startsWith("[") && trimmed.endsWith("]") && trimmed.length > 2) {
                trimmed.substring(1, trimmed.length - 1).trim()
            } else {
                null
            }

        fun parse(text: String): RglIni {
            val ending = if (text.contains("\r\n")) "\r\n" else if (text.contains('\n')) "\n" else "\r\n"
            val body = text.removePrefix("\uFEFF")
            val lines = body.split("\r\n", "\n").toMutableList()
            if (lines.isNotEmpty() && lines.last().isEmpty()) lines.removeAt(lines.size - 1)
            return RglIni(lines, ending)
        }
    }
}

/**
 * The `rgl_config.ini` settings the launcher shows. Each field is one WSE2 key ([RglConfigCodec]).
 * The defaults (the values in a WSE2 v1.1.5.6 file on a Windows PC) are shown only until WSE2 has
 * written the file, and are never written for a key the player did not change. [ini] is the file
 * as read, so a save changes only the keys the player changed.
 */
data class RglConfigState(
    // Graphics
    val shadows: Boolean = false,
    val characterShadows: Boolean = false,
    val particles: Boolean = false,
    val characterDetail: Float = 0.5f,
    val floraDegradeDistance: Float = 75f,
    val useOnDemandTextures: Boolean = true,
    val useInstancing: Boolean = false,
    val gamma: Float = 2.2f,
    val maxFramerate: Int = 60,
    val forceVsync: Boolean = false,
    val showFramerate: Boolean = false,

    // Battle
    val battleSize: Float = 1.0f,
    val battleSizeMin: Int = 30,
    val battleSizeMax: Int = 150,
    val enableBlood: Boolean = true,
    val combatSpeed: Int = 2,
    val damageToPlayer: Float = 0.5f,
    val damageToFriends: Float = 1.0f,
    val reduceCombatAi: Int = 2,
    val reduceCampaignAi: Int = 2,
    val attackDirectionControl: Int = 2,
    val defendDirectionControl: Int = 0,
    val lanceControl: Int = 0,
    val horseCameraMode: Int = 2,
    val showCrosshair: Boolean = true,
    val showReticle: Boolean = true,
    val showAttackDirection: Boolean = true,
    val showDamage: Boolean = true,
    val showShotDifficulty: Boolean = true,
    val showCasualties: Boolean = true,
    val showExperienceGain: Boolean = true,

    // Audio and input
    val musicVolume: Float = 0.4f,
    val soundVolume: Float = 1.0f,
    val music: Boolean = true,
    val sound: Boolean = true,
    val disableFrequencyVariation: Boolean = false,
    val mouseSensitivity: Float = 0.5f,
    val invertMouseY: Boolean = false,
    val mouseYScale: Float = 1.5f,
    val enableGamepadVibration: Boolean = true,

    // System
    val cheatMode: Boolean = false,
    val enableEditMode: Boolean = false,

    val ini: RglIni = RglIni()
) {
    /**
     * Soldiers on the field at the chosen battle size. A module can set its own range in its
     * `module.ini` (`battle_size_min` / `battle_size_max`, for example 120 to 600); otherwise WSE2
     * uses `iBattleSizeMin` / `iBattleSizeMax` from this file.
     */
    fun battleSizeTroops(moduleRange: IntRange? = null): Int {
        val lo = moduleRange?.first ?: battleSizeMin
        val hi = (moduleRange?.last ?: battleSizeMax).coerceAtLeast(lo)
        return lo + (battleSize.coerceIn(0f, 1f) * (hi - lo)).roundToInt()
    }

    // Graphics only: a picture preset never changes the size of the armies.
    fun withQualityPreset(presetKey: String): RglConfigState = when (presetKey) {
        "low" -> copy(
            shadows = false,
            characterShadows = false,
            particles = false,
            characterDetail = 0.35f,
            floraDegradeDistance = 60f
        )
        "medium" -> copy(
            shadows = true,
            characterShadows = false,
            particles = true,
            characterDetail = 0.5f,
            floraDegradeDistance = 100f
        )
        "high" -> copy(
            shadows = true,
            characterShadows = true,
            particles = true,
            characterDetail = 1.0f,
            floraDegradeDistance = 150f
        )
        else -> this
    }
}

object RglConfigOptions {
    val COMBAT_SPEED_LABELS = listOf("Slowest", "Slower", "Normal", "Faster", "Fastest")
    val DAMAGE_VALUES = listOf(0.25f, 0.5f, 1.0f)
    val DAMAGE_LABELS = listOf("1/4", "1/2", "Full")
    val AI_LEVEL_LABELS = listOf("Good", "Average", "Poor")
    val ATTACK_CONTROL_LABELS = listOf(
        "Mouse movement",
        "Inverse mouse",
        "Relative to enemy",
        "Movement keys"
    )
    val DEFEND_CONTROL_LABELS = listOf(
        "Automatic",
        "Mouse movement",
        "Inverse mouse",
        "Movement keys"
    )
    val LANCE_CONTROL_LABELS = listOf("Automatic", "Manual (Easy)", "Manual (Hard)")
    val HORSE_CAMERA_LABELS = listOf("Never", "On horse", "Always")
    val BATTLE_SIZE_MAX_VALUES = listOf(150, 300, 500, 1000, 2000, 4000, 6000)

    fun nearestIndex(values: List<Float>, v: Float): Int =
        values.indices.minByOrNull { abs(values[it] - v) } ?: 0
}

object RglConfigCodec {
    private fun parseBool(raw: String?, default: Boolean): Boolean {
        val v = raw?.trim()?.lowercase(Locale.US) ?: return default
        return when (v) {
            "true", "1", "yes" -> true
            "false", "0", "no" -> false
            else -> default
        }
    }

    private fun parseInt(raw: String?, default: Int, min: Int, max: Int): Int {
        val v = raw?.trim()?.toIntOrNull() ?: return default
        return v.coerceIn(min, max)
    }

    private fun parseFloat(raw: String?, default: Float, min: Float, max: Float): Float {
        val v = raw?.trim()?.toFloatOrNull() ?: return default
        if (!v.isFinite()) return default
        return v.coerceIn(min, max)
    }

    // WSE2's own formats: `true`/`false`, integers, six-decimal floats.
    private fun b(v: Boolean): String = if (v) "true" else "false"
    private fun f(v: Float): String = String.format(Locale.US, "%.6f", v)

    private class Key<T>(
        val section: String,
        val key: String,
        val parse: (String?) -> T,
        val get: (RglConfigState) -> T,
        val set: (RglConfigState, T) -> RglConfigState,
        val format: (T) -> String
    ) {
        fun read(ini: RglIni, into: RglConfigState): RglConfigState = set(into, parse(ini.get(section, key)))
        fun changed(a: RglConfigState, b: RglConfigState): Boolean = get(a) != get(b)
        fun write(ini: RglIni, from: RglConfigState): RglIni = ini.with(section, key, format(get(from)))
    }

    private val D = RglConfigState()

    private fun bool(
        section: String,
        key: String,
        get: (RglConfigState) -> Boolean,
        set: (RglConfigState, Boolean) -> RglConfigState
    ) = Key(section, key, { parseBool(it, get(D)) }, get, set, ::b)

    private fun int(
        section: String,
        key: String,
        min: Int,
        max: Int,
        get: (RglConfigState) -> Int,
        set: (RglConfigState, Int) -> RglConfigState
    ) = Key(section, key, { parseInt(it, get(D), min, max) }, get, set) { it.toString() }

    private fun float(
        section: String,
        key: String,
        min: Float,
        max: Float,
        get: (RglConfigState) -> Float,
        set: (RglConfigState, Float) -> RglConfigState
    ) = Key(section, key, { parseFloat(it, get(D), min, max) }, get, set, ::f)

    // One entry per edited key. `bMusic` / `bSound` are "on" switches in the file.
    private val KEYS: List<Key<*>> = listOf(
        bool("ShadowMap", "bShadowMap", { it.shadows }) { s, v -> s.copy(shadows = v) },
        bool("FakeCharacterShadows", "bFakeCharacterShadows", { it.characterShadows }) { s, v -> s.copy(characterShadows = v) },
        bool("ParticleSystems", "bParticleSystems", { it.particles }) { s, v -> s.copy(particles = v) },
        float("Scene", "fCharacterDetail", 0f, 1f, { it.characterDetail }) { s, v -> s.copy(characterDetail = v) },
        float("Flora", "fFloraDegradeDistance", 20f, 300f, { it.floraDegradeDistance }) { s, v -> s.copy(floraDegradeDistance = v) },
        bool("Graphics", "bOnDemandTextures", { it.useOnDemandTextures }) { s, v -> s.copy(useOnDemandTextures = v) },
        bool("Graphics", "bInstancing", { it.useInstancing }) { s, v -> s.copy(useInstancing = v) },
        float("Graphics", "fGamma", 1.0f, 3.0f, { it.gamma }) { s, v -> s.copy(gamma = v) },
        int("General", "iMaxFrameRate", 0, 1000, { it.maxFramerate }) { s, v -> s.copy(maxFramerate = v) },
        bool("Graphics", "bForceVSync", { it.forceVsync }) { s, v -> s.copy(forceVsync = v) },
        bool("General", "bShowFrameRate", { it.showFramerate }) { s, v -> s.copy(showFramerate = v) },

        float("Battle", "fBattleSize", 0f, 1f, { it.battleSize }) { s, v -> s.copy(battleSize = v) },
        int("Battle", "iBattleSizeMin", 1, 100000, { it.battleSizeMin }) { s, v -> s.copy(battleSizeMin = v) },
        int("Battle", "iBattleSizeMax", 1, 100000, { it.battleSizeMax }) { s, v -> s.copy(battleSizeMax = v) },
        bool("Battle", "bBlood", { it.enableBlood }) { s, v -> s.copy(enableBlood = v) },
        int("Battle", "iCombatSpeed", 0, 4, { it.combatSpeed }) { s, v -> s.copy(combatSpeed = v) },
        float("Battle", "fDamageToSelf", 0f, 1f, { it.damageToPlayer }) { s, v -> s.copy(damageToPlayer = v) },
        float("Battle", "fDamageToFriends", 0f, 1f, { it.damageToFriends }) { s, v -> s.copy(damageToFriends = v) },
        int("BattleAi", "iReduceBattleAi", 0, 2, { it.reduceCombatAi }) { s, v -> s.copy(reduceCombatAi = v) },
        int("CampaignAi", "iReduceCampaignAi", 0, 2, { it.reduceCampaignAi }) { s, v -> s.copy(reduceCampaignAi = v) },
        int("Battle", "iAttackControl", 0, 3, { it.attackDirectionControl }) { s, v -> s.copy(attackDirectionControl = v) },
        int("Battle", "iDefendControl", 0, 3, { it.defendDirectionControl }) { s, v -> s.copy(defendDirectionControl = v) },
        int("Battle", "iLanceControl", 0, 2, { it.lanceControl }) { s, v -> s.copy(lanceControl = v) },
        int("Battle", "iHorseCameraMode", 0, 2, { it.horseCameraMode }) { s, v -> s.copy(horseCameraMode = v) },
        bool("Battle", "bShowCrosshair", { it.showCrosshair }) { s, v -> s.copy(showCrosshair = v) },
        bool("Battle", "bShowReticle", { it.showReticle }) { s, v -> s.copy(showReticle = v) },
        bool("Battle", "bShowAttackDirection", { it.showAttackDirection }) { s, v -> s.copy(showAttackDirection = v) },
        bool("Messages", "bShowDamage", { it.showDamage }) { s, v -> s.copy(showDamage = v) },
        bool("Messages", "bShowShotDifficulty", { it.showShotDifficulty }) { s, v -> s.copy(showShotDifficulty = v) },
        bool("Messages", "bShowCasualties", { it.showCasualties }) { s, v -> s.copy(showCasualties = v) },
        bool("Messages", "bShowExperienceGain", { it.showExperienceGain }) { s, v -> s.copy(showExperienceGain = v) },

        float("Sound", "fMusicVolume", 0f, 1f, { it.musicVolume }) { s, v -> s.copy(musicVolume = v) },
        float("Sound", "fSoundVolume", 0f, 1f, { it.soundVolume }) { s, v -> s.copy(soundVolume = v) },
        bool("Sound", "bMusic", { it.music }) { s, v -> s.copy(music = v) },
        bool("Sound", "bSound", { it.sound }) { s, v -> s.copy(sound = v) },
        bool("Sound", "bDisableFrequencyVariation", { it.disableFrequencyVariation }) { s, v -> s.copy(disableFrequencyVariation = v) },
        float("Input", "fMouseSensitivity", 0.05f, 3f, { it.mouseSensitivity }) { s, v -> s.copy(mouseSensitivity = v) },
        bool("Input", "bInvertMouseY", { it.invertMouseY }) { s, v -> s.copy(invertMouseY = v) },
        float("Input", "fMouseYScale", 0.25f, 4f, { it.mouseYScale }) { s, v -> s.copy(mouseYScale = v) },
        bool("Input", "bEnableGamepadVibration", { it.enableGamepadVibration }) { s, v -> s.copy(enableGamepadVibration = v) },

        bool("General", "bCheatMode", { it.cheatMode }) { s, v -> s.copy(cheatMode = v) },
        bool("General", "bEditMode", { it.enableEditMode }) { s, v -> s.copy(enableEditMode = v) }
    )

    /** `Section.key` of every key the launcher edits, in the file's own spelling. */
    val EDITED_KEYS: List<String> get() = KEYS.map { "${it.section}.${it.key}" }

    fun parse(text: String): RglConfigState = fromIni(RglIni.parse(text))

    fun fromIni(ini: RglIni): RglConfigState =
        KEYS.fold(RglConfigState(ini = ini)) { state, key -> key.read(ini, state) }

    // The file as read, with only the settings that differ from it rewritten. A key the file
    // does not have yet is added only when the player changed it, so WSE2 keeps choosing the rest.
    fun toIni(state: RglConfigState): RglIni {
        val before = fromIni(state.ini)
        var ini = state.ini
        for (key in KEYS) {
            if (key.changed(state, before)) ini = key.write(ini, state)
        }
        return ini
    }

    fun serialize(state: RglConfigState): String = toIni(state).serialize()
}

/** `battle_size_min` / `battle_size_max` from a module's `module.ini`, or null when it sets none. */
fun moduleBattleSizeRange(min: Int, max: Int): IntRange? =
    if (min > 0 && max >= min) min..max else null

// Where the R4 run's exe reads its config: the writable `user` layer maps C:\Users\Player, and
// SHGetFolderPath(CSIDL_PERSONAL) returns its Documents (engine/include/wse/recomp/shim_files.h).
//
// The launcher never writes the file just because it opened: with no file, WSE2 makes its own on
// first start, as on Windows, and the Settings screen shows the defaults it would pick. The file
// is written only when the player changes a setting. A file that exists but cannot be read is
// never overwritten: `save` first moves it to `rgl_config.ini.unreadable`, and refuses if it cannot.
internal enum class RglConfigSource { MISSING, LOADED, UNREADABLE }

internal object RglConfigStore {
    private const val RELATIVE_PATH = "wse2/user/Documents/Mount&Blade Warband WSE2/rgl_config.ini"
    private const val UNREADABLE_SUFFIX = ".unreadable"
    private const val RESET_SUFFIX = ".before_reset"

    fun configFile(filesDir: File): File = File(filesDir, RELATIVE_PATH)

    // Reads the file; writes nothing unless migrating an existing legacy `recomp/r4/user` tree.
    fun load(filesDir: File): Pair<RglConfigState, RglConfigSource> {
        UserStorage.migrateLegacyUserDirIfNeeded(filesDir)
        val file = configFile(filesDir)
        if (!file.exists()) return RglConfigState() to RglConfigSource.MISSING
        return try {
            RglConfigCodec.parse(file.readText(Charsets.UTF_8)) to RglConfigSource.LOADED
        } catch (_: Throwable) {
            RglConfigState() to RglConfigSource.UNREADABLE
        }
    }

    fun save(filesDir: File, state: RglConfigState): Boolean {
        UserStorage.migrateLegacyUserDirIfNeeded(filesDir)
        return try {
            val file = configFile(filesDir)
            file.parentFile?.mkdirs()
            if (file.exists() && !readable(file)) {
                val kept = File(file.parentFile, file.name + UNREADABLE_SUFFIX)
                if (kept.exists() && !kept.deleteRecursively()) return false
                if (!file.renameTo(kept)) return false
            }
            val tmp = File(file.parentFile, "${file.name}.tmp")
            tmp.writeText(RglConfigCodec.serialize(state), Charsets.UTF_8)
            if (!tmp.renameTo(file)) {
                tmp.copyTo(file, overwrite = true)
                tmp.delete()
            }
            true
        } catch (_: Throwable) {
            false
        }
    }

    // WSE2 writes a complete file of its own defaults when none exists, so a reset moves the file
    // aside (`rgl_config.ini.before_reset`, replacing an older one) instead of guessing defaults.
    fun reset(filesDir: File): Boolean {
        UserStorage.migrateLegacyUserDirIfNeeded(filesDir)
        return try {
            val file = configFile(filesDir)
            if (!file.exists()) return true
            val kept = File(file.parentFile, file.name + RESET_SUFFIX)
            if (kept.exists() && !kept.deleteRecursively()) return false
            file.renameTo(kept)
        } catch (_: Throwable) {
            false
        }
    }

    private fun readable(file: File): Boolean = try {
        file.isFile && file.readBytes().let { true }
    } catch (_: Throwable) {
        false
    }
}
