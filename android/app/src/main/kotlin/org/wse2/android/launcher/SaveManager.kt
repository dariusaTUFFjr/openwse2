package org.wse2.android.launcher

import android.content.Context
import android.net.Uri
import android.provider.DocumentsContract
import org.json.JSONObject
import org.wse2.android.WseSafBridge
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

data class SaveHeaderInfo(
    val valid: Boolean,
    val saveName: String = "",
    val playerName: String = "",
    val playerLevel: Int = 1,
    val day: Int = 1,
    val saveVersion: Int = 0,
    val moduleVersion: Int = 0,
    val error: String = ""
)

data class SaveBackupEntry(
    val file: File,
    val backupIndex: Int,
    val sizeBytes: Long,
    val lastModifiedMs: Long
)

data class SaveSlotEntry(
    val moduleName: String,
    val file: File,
    val fileName: String,
    val sizeBytes: Long,
    val lastModifiedMs: Long,
    val header: SaveHeaderInfo,
    val backups: List<SaveBackupEntry>
)

/**
 * Launcher Save Manager & optional Auto-Sync SAF Save Mirror (`docs/PLATFORM_PLAN.md` "P14.3").
 */
object SaveManager {
    private const val WRDR_MAGIC = 0x52445257
    private const val MIN_SAVE_VER = 802
    private const val MAX_SAVE_VER = 1174
    private const val MIRROR_PREFS = "wse2_save_mirror_sync"

    fun savegameDirs(filesDir: File, moduleName: String): List<File> {
        UserStorage.migrateLegacyUserDirIfNeeded(filesDir)
        val user = UserStorage.userDir(filesDir)
        val mod = moduleName.ifBlank { "Native" }
        return listOf(
            File(user, "Documents/Mount&Blade Warband WSE2/$mod/Savegames"),
            File(user, "Documents/Mount&Blade Warband/$mod/Savegames")
        )
    }

    fun primarySaveDir(filesDir: File, moduleName: String): File =
        savegameDirs(filesDir, moduleName).first()

    /**
     * Parses a Warband `.sav` header using native Phase 2A `savegame.h` when available,
     * falling back to pure-Kotlin binary parsing (used in Robolectric / JVM unit tests).
     */
    fun inspectSaveFile(file: File): SaveHeaderInfo {
        if (!file.isFile || file.length() < 16L) {
            return SaveHeaderInfo(valid = false, error = "File is missing or too small.")
        }
        val nativeJson = try {
            WseSafBridge.nativeInspectSavegame(file.absolutePath)
        } catch (_: Throwable) {
            null
        }
        if (!nativeJson.isNullOrBlank()) {
            try {
                val obj = JSONObject(nativeJson)
                return if (obj.optBoolean("ok", false)) {
                    SaveHeaderInfo(
                        valid = true,
                        saveName = obj.optString("name", ""),
                        playerName = obj.optString("player", ""),
                        playerLevel = obj.optInt("level", 1),
                        day = obj.optInt("day", 1),
                        saveVersion = obj.optInt("save_ver", 0),
                        moduleVersion = obj.optInt("module_ver", 0)
                    )
                } else {
                    SaveHeaderInfo(valid = false, error = obj.optString("error", "Invalid savegame."))
                }
            } catch (_: Throwable) {
            }
        }
        return parseHeaderBytes(file.readBytes())
    }

    internal fun parseHeaderBytes(bytes: ByteArray): SaveHeaderInfo {
        if (bytes.size < 20) return SaveHeaderInfo(valid = false, error = "Truncated savegame header.")
        val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val magic = bb.int
        if (magic != WRDR_MAGIC) {
            return SaveHeaderInfo(valid = false, error = "Not a Warband savegame (expected WRDR header).")
        }
        val saveVer = bb.int
        if (saveVer < MIN_SAVE_VER || saveVer > MAX_SAVE_VER) {
            return SaveHeaderInfo(valid = false, error = "Unsupported savegame version ($saveVer).")
        }
        val modVer = bb.int
        fun readRglString(): String? {
            if (bb.remaining() < 4) return null
            val len = bb.int
            if (len < 0 || len > 4096 || bb.remaining() < len) return null
            val buf = ByteArray(len)
            bb.get(buf)
            return String(buf, Charsets.UTF_8)
        }
        val saveName = readRglString() ?: return SaveHeaderInfo(valid = false, error = "Truncated save name.")
        val playerName = readRglString() ?: return SaveHeaderInfo(valid = false, error = "Truncated player name.")
        if (bb.remaining() < 8) return SaveHeaderInfo(valid = false, error = "Truncated player stats.")
        val level = bb.int
        val dateHours = bb.float
        val day = (dateHours / 24f).toInt().coerceAtLeast(0) + 1
        return SaveHeaderInfo(
            valid = true,
            saveName = saveName,
            playerName = playerName,
            playerLevel = level,
            day = day,
            saveVersion = saveVer,
            moduleVersion = modVer
        )
    }

    fun listSaves(filesDir: File, moduleName: String): List<SaveSlotEntry> {
        val mod = moduleName.ifBlank { "Native" }
        val seen = LinkedHashMap<String, SaveSlotEntry>()
        for (dir in savegameDirs(filesDir, mod)) {
            if (!dir.isDirectory) continue
            val files = dir.listFiles()?.filter { it.isFile }?.sortedBy { it.name.lowercase() } ?: continue
            for (f in files) {
                val lower = f.name.lowercase()
                if (!lower.endsWith(".sav")) continue
                if (lower == "new_game.sav" || lower == "last_savegame_backup.sav") continue
                if (seen.containsKey(lower)) continue
                val backups = (1..3).mapNotNull { idx ->
                    val bf = File(dir, "${f.name}.bak$idx")
                    if (bf.isFile) {
                        SaveBackupEntry(bf, idx, bf.length(), bf.lastModified())
                    } else {
                        null
                    }
                }
                seen[lower] = SaveSlotEntry(
                    moduleName = mod,
                    file = f,
                    fileName = f.name,
                    sizeBytes = f.length(),
                    lastModifiedMs = f.lastModified(),
                    header = inspectSaveFile(f),
                    backups = backups
                )
            }
        }
        return seen.values.sortedByDescending { it.lastModifiedMs }
    }

    fun backupBytesUsed(filesDir: File): Long {
        val user = UserStorage.userDir(filesDir)
        if (!user.isDirectory) return 0L
        return user.walkTopDown()
            .filter { it.isFile && Regex("""\.sav\.bak[1-3]$""", RegexOption.IGNORE_CASE).containsMatchIn(it.name) }
            .sumOf { it.length() }
    }

    fun clearAllBackups(filesDir: File): Int {
        val user = UserStorage.userDir(filesDir)
        if (!user.isDirectory) return 0
        var removed = 0
        for (f in user.walkTopDown()) {
            if (f.isFile && Regex("""\.sav\.bak[1-3]$""", RegexOption.IGNORE_CASE).containsMatchIn(f.name)) {
                if (f.delete()) ++removed
            }
        }
        return removed
    }

    fun writeAtomicWithBackup(target: File, bytes: ByteArray): Boolean {
        return try {
            target.parentFile?.mkdirs()
            val tmp = File(target.parentFile, "${target.name}.wse_tmp")
            tmp.writeBytes(bytes)
            if (target.isFile) {
                val b1 = File(target.parentFile, "${target.name}.bak1")
                val b2 = File(target.parentFile, "${target.name}.bak2")
                val b3 = File(target.parentFile, "${target.name}.bak3")
                if (b3.exists()) b3.delete()
                if (b2.exists()) b2.renameTo(b3)
                if (b1.exists()) b1.renameTo(b2)
                target.renameTo(b1)
            }
            if (!tmp.renameTo(target)) {
                tmp.copyTo(target, overwrite = true)
                tmp.delete()
            }
            true
        } catch (_: Throwable) {
            false
        }
    }

    fun restoreBackup(backup: File, target: File): Pair<Boolean, String> {
        if (!backup.isFile) return false to "Backup file no longer exists."
        val header = inspectSaveFile(backup)
        if (!header.valid) return false to "Backup header check failed: ${header.error}"
        val bytes = try {
            backup.readBytes()
        } catch (t: Throwable) {
            return false to "Could not read backup: ${t.message ?: t.javaClass.simpleName}"
        }
        val ok = writeAtomicWithBackup(target, bytes)
        return if (ok) true to "Restored ${target.name} from ${backup.name}." else false to "Failed to write ${target.name}."
    }

    fun exportSaveToUri(context: Context, saveFile: File, destUri: Uri): Pair<Boolean, String> {
        if (!saveFile.isFile) return false to "Save file does not exist."
        return try {
            context.contentResolver.openOutputStream(destUri, "wt")?.use { out ->
                saveFile.inputStream().use { input -> input.copyTo(out, 1 shl 16) }
            } ?: return false to "Could not open destination file."
            true to "Exported ${saveFile.name} (${saveFile.length() / 1024} KB)."
        } catch (t: Throwable) {
            false to "Export failed: ${t.message ?: t.javaClass.simpleName}"
        }
    }

    fun importSaveFromUri(
        context: Context,
        srcUri: Uri,
        filesDir: File,
        moduleName: String,
        targetSlotFileName: String = "sg00.sav"
    ): Pair<Boolean, String> {
        val bytes = try {
            context.contentResolver.openInputStream(srcUri)?.use { it.readBytes() }
                ?: return false to "Could not read selected file."
        } catch (t: Throwable) {
            return false to "Could not open file: ${t.message ?: t.javaClass.simpleName}"
        }
        val header = parseHeaderBytes(bytes)
        if (!header.valid) {
            return false to "Refused import: ${header.error}"
        }
        val safeName = targetSlotFileName.trim().ifEmpty { "sg00.sav" }.let {
            if (it.lowercase().endsWith(".sav")) it else "$it.sav"
        }
        val dest = File(primarySaveDir(filesDir, moduleName), safeName)
        return if (writeAtomicWithBackup(dest, bytes)) {
            true to "Imported ${header.playerName.ifBlank { safeName }} (Day ${header.day}) into $safeName."
        } else {
            false to "Failed to write $safeName."
        }
    }

    /**
     * Auto-Sync SAF Save Mirror: mirrors changed `.sav` files for all modules into
     * `<mirrorTree>/WSE2_Saves/<Module>/<slot>.sav`. Safe to run on a background thread.
     */
    fun syncSavesToMirror(context: Context, filesDir: File, mirrorTreeUri: Uri): Pair<Int, String> {
        val user = UserStorage.userDir(filesDir)
        val wse2Docs = File(user, "Documents/Mount&Blade Warband WSE2")
        if (!wse2Docs.isDirectory) return 0 to "No WSE2 saves yet."
        val prefs = context.getSharedPreferences(MIRROR_PREFS, Context.MODE_PRIVATE)
        val bridge = WseSafBridge(context, mirrorTreeUri)
        val rootId = bridge.rootDocumentId()
        if (rootId.isEmpty()) return 0 to "Mirror folder permission missing."

        fun ensureChildDir(parentId: String, name: String): String? {
            val kids = bridge.listChildren(parentId) ?: return null
            for (line in kids) {
                val parts = line.split('\t')
                if (parts.size >= 4 && parts[0] == "1" && parts[3].equals(name, ignoreCase = true)) {
                    return parts[2]
                }
            }
            return bridge.createDirectory(parentId, name)
        }

        val savesRootId = ensureChildDir(rootId, "WSE2_Saves") ?: return 0 to "Could not create WSE2_Saves folder."
        var synced = 0
        for (modDir in wse2Docs.listFiles()?.filter { it.isDirectory } ?: emptyList()) {
            val sgDir = File(modDir, "Savegames")
            if (!sgDir.isDirectory) continue
            val savFiles = sgDir.listFiles()?.filter {
                it.isFile && it.name.lowercase().endsWith(".sav") &&
                    !it.name.equals("new_game.sav", ignoreCase = true)
            } ?: continue
            if (savFiles.isEmpty()) continue
            val modDocId = ensureChildDir(savesRootId, modDir.name) ?: continue
            val existingDocs = HashMap<String, String>()
            for (line in bridge.listChildren(modDocId) ?: emptyArray()) {
                val parts = line.split('\t')
                if (parts.size >= 4 && parts[0] == "0") {
                    existingDocs[parts[3].lowercase()] = parts[2]
                }
            }
            for (sav in savFiles) {
                val key = "${modDir.name}/${sav.name}"
                val stamp = "${sav.length()}:${sav.lastModified()}"
                if (prefs.getString(key, null) == stamp) continue
                val existingId = existingDocs[sav.name.lowercase()] ?: ""
                val written = bridge.writeDocument(modDocId, existingId, sav.name, sav.readBytes())
                if (written != null) {
                    prefs.edit().putString(key, stamp).apply()
                    ++synced
                }
            }
        }
        return synced to if (synced > 0) "Synced $synced save(s) to mirror." else "Saves already up to date."
    }
}
