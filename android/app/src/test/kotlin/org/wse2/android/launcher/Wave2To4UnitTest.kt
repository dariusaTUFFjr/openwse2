package org.wse2.android.launcher

import androidx.test.ext.junit.runners.AndroidJUnit4
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import org.junit.runner.RunWith
import org.robolectric.annotation.Config
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

@RunWith(AndroidJUnit4::class)
@Config(sdk = [34])
class Wave2To4UnitTest {

    @get:Rule
    val tmp = TemporaryFolder()

    private fun makeValidSaveBytes(saveName: String, playerName: String, level: Int, day: Int): ByteArray {
        val sn = saveName.toByteArray(Charsets.UTF_8)
        val pn = playerName.toByteArray(Charsets.UTF_8)
        val bb = ByteBuffer.allocate(12 + 4 + sn.size + 4 + pn.size + 8).order(ByteOrder.LITTLE_ENDIAN)
        bb.putInt(0x52445257) // "WRDR"
        bb.putInt(1174)       // save version
        bb.putInt(1174)       // module version
        bb.putInt(sn.size)
        bb.put(sn)
        bb.putInt(pn.size)
        bb.put(pn)
        bb.putInt(level)
        bb.putFloat(((day - 1).coerceAtLeast(0) * 24).toFloat())
        return bb.array()
    }

    private fun makeZip(entries: Map<String, ByteArray>): ByteArray {
        val baos = ByteArrayOutputStream()
        ZipOutputStream(baos).use { zos ->
            for ((name, bytes) in entries) {
                zos.putNextEntry(ZipEntry(name))
                zos.write(bytes)
                zos.closeEntry()
            }
        }
        return baos.toByteArray()
    }

    @Test
    fun userStorageMigration_writesNothingWhenLegacyMissing() {
        val filesDir = tmp.newFolder("fresh_files")
        val copied = UserStorage.migrateLegacyUserDirIfNeeded(filesDir)
        assertEquals(0, copied)
        assertFalse(UserStorage.userDir(filesDir).exists())
    }

    @Test
    fun userStorageMigration_copiesOnceWithoutOverwritingExisting() {
        val filesDir = tmp.newFolder("mig_files")
        val legacySav = File(filesDir, "recomp/r4/user/Documents/Mount&Blade Warband WSE2/Native/Savegames/sg00.sav")
        legacySav.parentFile!!.mkdirs()
        legacySav.writeText("LEGACY_SAVE")

        val existingCfg = File(filesDir, "wse2/user/Documents/Mount&Blade Warband/rgl_config.txt")
        existingCfg.parentFile!!.mkdirs()
        existingCfg.writeText("NEWER_CONFIG")
        val legacyCfg = File(filesDir, "recomp/r4/user/Documents/Mount&Blade Warband/rgl_config.txt")
        legacyCfg.parentFile!!.mkdirs()
        legacyCfg.writeText("OLDER_CONFIG")

        val copied = UserStorage.migrateLegacyUserDirIfNeeded(filesDir)
        assertEquals(1, copied)
        val migratedSav = File(filesDir, "wse2/user/Documents/Mount&Blade Warband WSE2/Native/Savegames/sg00.sav")
        assertTrue(migratedSav.isFile)
        assertEquals("LEGACY_SAVE", migratedSav.readText())
        assertEquals("NEWER_CONFIG", existingCfg.readText())
        // Idempotent second call copies 0.
        assertEquals(0, UserStorage.migrateLegacyUserDirIfNeeded(filesDir))
    }

    @Test
    fun saveManager_headerParseAtomicRotationAndRestore() {
        val filesDir = tmp.newFolder("saves_files")
        val target = File(SaveManager.primarySaveDir(filesDir, "Native"), "sg00.sav")
        val v1 = makeValidSaveBytes("Slot 1", "Harlaus", 12, 5)
        val v2 = makeValidSaveBytes("Slot 1", "Harlaus", 14, 10)
        val v3 = makeValidSaveBytes("Slot 1", "Harlaus", 16, 15)
        val v4 = makeValidSaveBytes("Slot 1", "Harlaus", 18, 20)

        assertTrue(SaveManager.writeAtomicWithBackup(target, v1))
        assertTrue(SaveManager.writeAtomicWithBackup(target, v2))
        assertTrue(SaveManager.writeAtomicWithBackup(target, v3))
        assertTrue(SaveManager.writeAtomicWithBackup(target, v4))

        val listed = SaveManager.listSaves(filesDir, "Native")
        assertEquals(1, listed.size)
        val slot = listed.first()
        assertTrue(slot.header.valid)
        assertEquals("Harlaus", slot.header.playerName)
        assertEquals(18, slot.header.playerLevel)
        assertEquals(20, slot.header.day)
        assertEquals(3, slot.backups.size)
        assertTrue(SaveManager.backupBytesUsed(filesDir) > 0L)

        // Restore .bak1 (which holds v3: level 16, day 15)
        val (ok, _) = SaveManager.restoreBackup(slot.backups.first().file, slot.file)
        assertTrue(ok)
        val afterRestore = SaveManager.inspectSaveFile(slot.file)
        assertTrue(afterRestore.valid)
        assertEquals(16, afterRestore.playerLevel)
        assertEquals(15, afterRestore.day)

        val cleared = SaveManager.clearAllBackups(filesDir)
        assertEquals(3, cleared)
        assertEquals(0L, SaveManager.backupBytesUsed(filesDir))
    }

    @Test
    fun preparedZipVerifier_acceptsValidAndRefusesMissingOrTamperedManifest() {
        val soDir = tmp.newFolder("so_dir")
        val texDir = tmp.newFolder("tex_dir")

        val soBytes = "FAKE_ARM64_SO_PAYLOAD".toByteArray()
        val keyBytes = "FAKE_KEY_V1".toByteArray()
        val manifestJson = JSONObject()
            .put("format", 1)
            .put("created_utc", "2026-11-26T00:00:00Z")
            .put("translator_fingerprint", "testfp")
            .put(
                "files",
                JSONObject()
                    .put("exe_w000.so", PreparedZipVerifier.sha256Hex(soBytes))
                    .put("exe_w000.key", PreparedZipVerifier.sha256Hex(keyBytes))
            )
            .toString()

        // 1. Missing manifest.json -> refused cleanly, nothing installed.
        val noManifestZip = makeZip(mapOf("exe_w000.so" to soBytes, "exe_w000.key" to keyBytes))
        val r1 = PreparedZipVerifier.verifyAndExtract(ByteArrayInputStream(noManifestZip), soDir, texDir)
        assertFalse(r1.ok)
        assertFalse(File(soDir, "exe_w000.so").exists())

        // 2. Tampered .so payload -> refused cleanly, nothing installed.
        val tamperedZip = makeZip(
            mapOf(
                "manifest.json" to manifestJson.toByteArray(),
                "exe_w000.so" to "TAMPERED_BYTES".toByteArray(),
                "exe_w000.key" to keyBytes
            )
        )
        val r2 = PreparedZipVerifier.verifyAndExtract(ByteArrayInputStream(tamperedZip), soDir, texDir)
        assertFalse(r2.ok)
        assertTrue(r2.error.contains("SHA-256"))
        assertFalse(File(soDir, "exe_w000.so").exists())

        // 3. Valid archive -> verified and committed.
        val validZip = makeZip(
            mapOf(
                "manifest.json" to manifestJson.toByteArray(),
                "exe_w000.so" to soBytes,
                "exe_w000.key" to keyBytes
            )
        )
        val r3 = PreparedZipVerifier.verifyAndExtract(ByteArrayInputStream(validZip), soDir, texDir)
        assertTrue(r3.error, r3.ok)
        assertEquals(1, r3.partsImported)
        assertTrue(File(soDir, "exe_w000.so").isFile)
        assertTrue(File(soDir, "exe_w000.key").isFile)
    }

    @Test
    fun moduleProfileStore_roundTripsPerModuleProfiles() {
        val filesDir = tmp.newFolder("profile_files")
        assertNull(ModuleProfileStore.loadTouchLayoutJson(filesDir, "Prophesy of Pendor"))

        val layoutJson = """{"v":1,"buttons":{"attack":[0.85,0.72,0.14,1]}}"""
        assertTrue(ModuleProfileStore.saveTouchLayoutJson(filesDir, "Prophesy of Pendor", layoutJson))
        assertEquals(layoutJson, ModuleProfileStore.loadTouchLayoutJson(filesDir, "Prophesy of Pendor"))
        assertNull(ModuleProfileStore.loadTouchLayoutJson(filesDir, "Native"))

        val pic = ModulePicturePrefs(resolutionIndex = 2, sharpen = false, fxaa = true, targetFps = 90)
        assertTrue(ModuleProfileStore.savePicturePrefs(filesDir, "Prophesy of Pendor", pic))
        val loadedPic = ModuleProfileStore.loadPicturePrefs(filesDir, "Prophesy of Pendor")
        assertEquals(pic, loadedPic)

        val ctrl = ModuleControlsPrefs(touchOpacity = 0.8f, buttonScale = 1.2f, gyroEnabled = true, hapticStrength = 85)
        assertTrue(ModuleProfileStore.saveControlsPrefs(filesDir, "Prophesy of Pendor", ctrl))
        val loadedCtrl = ModuleProfileStore.loadControlsPrefs(filesDir, "Prophesy of Pendor")
        assertEquals(0.8f, loadedCtrl.touchOpacity, 0.001f)
        assertEquals(1.2f, loadedCtrl.buttonScale, 0.001f)
        assertTrue(loadedCtrl.gyroEnabled)
        assertEquals(85, loadedCtrl.hapticStrength)
        assertNotNull(UserStorage.moduleProfileDir(filesDir, "Prophesy of Pendor"))
    }

    @Test
    fun userStorage_pgoAndCrashReportPathsStayInsideUserDir() {
        val filesDir = tmp.newFolder("diag_files")
        val pgo = UserStorage.pgoProfileFile(filesDir)
        val crash = UserStorage.crashReportFile(filesDir)
        assertTrue(pgo.absolutePath.startsWith(UserStorage.userDir(filesDir).absolutePath))
        assertTrue(crash.absolutePath.startsWith(UserStorage.userDir(filesDir).absolutePath))
        assertEquals("wse_pgo.txt", pgo.name)
        assertEquals("crash_report.txt", crash.name)
    }

    @Test
    fun preparedZipVerifier_acceptsPgoHotBatchesAndProfile() {
        fun buildZip(manifestPgoHashOverride: String? = null): ByteArray {
            val soBytes = byteArrayOf(0x7F, 'E'.code.toByte(), 'L'.code.toByte(), 'F'.code.toByte())
            val keyBytes = "lifter 961e7bb2d5a391e4\n".toByteArray(Charsets.UTF_8)
            val pgoBytes = "# WSE2 PGO v1\n0x140001000 42\n".toByteArray(Charsets.UTF_8)
            val filesObj = JSONObject()
                .put("exe_hot_0.so", PreparedZipVerifier.sha256Hex(soBytes))
                .put("exe_hot_0.key", PreparedZipVerifier.sha256Hex(keyBytes))
                .put(
                    PreparedZipVerifier.PGO_PROFILE_NAME,
                    manifestPgoHashOverride ?: PreparedZipVerifier.sha256Hex(pgoBytes)
                )
            val manifestJson = JSONObject()
                .put("format", 1)
                .put("created_utc", "2026-11-27T00:00:00Z")
                .put("translator_fingerprint", "lifter 961e7bb2d5a391e4")
                .put("files", filesObj)
                .toString()
                .toByteArray(Charsets.UTF_8)
            val baos = ByteArrayOutputStream()
            ZipOutputStream(baos).use { z ->
                for ((name, bytes) in listOf(
                    PreparedZipVerifier.MANIFEST_NAME to manifestJson,
                    "exe_hot_0.so" to soBytes,
                    "exe_hot_0.key" to keyBytes,
                    PreparedZipVerifier.PGO_PROFILE_NAME to pgoBytes
                )) {
                    z.putNextEntry(ZipEntry(name))
                    z.write(bytes)
                    z.closeEntry()
                }
            }
            return baos.toByteArray()
        }

        val root = tmp.newFolder("pgo_zip_ok")
        val soDir = File(root, "so")
        val texDir = File(root, "tex")
        val pgoDest = File(root, "user/wse_pgo.txt")
        val okRes = PreparedZipVerifier.verifyAndExtract(
            zipStream = ByteArrayInputStream(buildZip()),
            soDir = soDir,
            texDir = texDir,
            pgoProfileDest = pgoDest
        )
        assertTrue(okRes.error, okRes.ok)
        assertEquals(1, okRes.partsImported)
        assertTrue(File(soDir, "exe_hot_0.so").isFile)
        assertTrue(File(soDir, "exe_hot_0.key").isFile)
        assertTrue(pgoDest.isFile)
        assertTrue(pgoDest.readText().contains("0x140001000 42"))

        val badRoot = tmp.newFolder("pgo_zip_bad")
        val badPgoDest = File(badRoot, "user/wse_pgo.txt")
        val badRes = PreparedZipVerifier.verifyAndExtract(
            zipStream = ByteArrayInputStream(buildZip("0".repeat(64))),
            soDir = File(badRoot, "so"),
            texDir = File(badRoot, "tex"),
            pgoProfileDest = badPgoDest
        )
        assertFalse(badRes.ok)
        assertFalse(badPgoDest.exists())
    }
}
