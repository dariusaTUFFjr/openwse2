package org.wse2.android.launcher

import java.io.File

/**
 * Permanent app-private user storage layout (`docs/PLATFORM_PLAN.md` "P14.3").
 *
 * Keeps saves, `rgl_config.ini`, and per-module profiles under `filesDir/wse2/user/`
 * (separate from `filesDir/recomp/r4/` caches) and migrates any existing legacy
 * `filesDir/recomp/r4/user/` tree once without overwriting newer files.
 */
object UserStorage {
    private const val MIGRATION_SENTINEL = ".migrated_v1"

    fun userDir(filesDir: File): File = File(filesDir, "wse2/user")

    fun legacyUserDir(filesDir: File): File = File(filesDir, "recomp/r4/user")

    fun profilesRoot(filesDir: File): File = File(userDir(filesDir), "profiles")

    fun pgoProfileFile(filesDir: File): File = File(userDir(filesDir), "wse_pgo.txt")

    fun crashReportFile(filesDir: File): File = File(userDir(filesDir), "crash_report.txt")

    fun moduleProfileDir(filesDir: File, moduleName: String): File {
        val safe = moduleName.trim().ifEmpty { "Native" }
            .replace(Regex("""[^A-Za-z0-9._-]"""), "_")
        return File(profilesRoot(filesDir), safe)
    }

    /**
     * Migrates files from `filesDir/recomp/r4/user/` into `filesDir/wse2/user/` once.
     *
     * Never writes anything when the legacy directory does not exist (preserves clean-room
     * "opening the launcher writes nothing" behavior), and never overwrites an existing file
     * already present in `wse2/user/`.
     */
    @Synchronized
    fun migrateLegacyUserDirIfNeeded(filesDir: File): Int {
        val legacy = legacyUserDir(filesDir)
        if (!legacy.isDirectory) return 0
        val destRoot = userDir(filesDir)
        val sentinel = File(destRoot, MIGRATION_SENTINEL)
        if (sentinel.isFile) return 0

        var copied = 0
        var allOk = true
        val legacyBase = legacy.toPath().toAbsolutePath().normalize()
        for (src in legacy.walkTopDown()) {
            if (!src.isFile) continue
            val rel = legacyBase.relativize(src.toPath().toAbsolutePath().normalize()).toString()
            val dst = File(destRoot, rel)
            if (dst.isFile) continue
            try {
                dst.parentFile?.mkdirs()
                val tmp = File(dst.parentFile, "${dst.name}.mig_tmp")
                src.inputStream().use { input ->
                    tmp.outputStream().use { output -> input.copyTo(output, 1 shl 16) }
                }
                if (tmp.length() == src.length()) {
                    if (tmp.renameTo(dst) || run { tmp.copyTo(dst, overwrite = true); tmp.delete(); true }) {
                        ++copied
                    } else {
                        tmp.delete()
                        allOk = false
                    }
                } else {
                    tmp.delete()
                    allOk = false
                }
            } catch (_: Throwable) {
                allOk = false
            }
        }
        if (allOk) {
            try {
                destRoot.mkdirs()
                sentinel.writeText("migrated=$copied\n")
            } catch (_: Throwable) {
            }
        }
        return copied
    }
}
