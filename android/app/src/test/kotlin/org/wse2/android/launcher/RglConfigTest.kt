package org.wse2.android.launcher

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class RglConfigTest {

    private fun tempFilesDir(): java.io.File =
        java.nio.file.Files.createTempDirectory("rgl_store").toFile().also { it.deleteOnExit() }

    // A cut-down rgl_config.ini as WSE2 v1.1.5.6 writes it: CRLF, sorted keys, typed prefixes.
    private val wse2Ini = listOf(
        "[Battle]",
        "bBlood=true",
        "fBattleSize=0.250000",
        "iBattleSizeMax=150",
        "iBattleSizeMin=30",
        "iCombatSpeed=2",
        "vStickItemToLeftHandOffset=0.000000,-0.370000,-0.040000",
        "[General]",
        "bCheatMode=false",
        "iMaxFrameRate=60",
        "sCoreResources=core_shaders,core_textures",
        "[Sound]",
        "bMusic=true",
        "fMusicVolume=0.400000"
    ).joinToString("\r\n", postfix = "\r\n")

    @Test
    fun loadingAMissingFileWritesNothing() {
        val dir = tempFilesDir()
        val (state, source) = RglConfigStore.load(dir)
        assertEquals(RglConfigSource.MISSING, source)
        assertEquals(RglConfigState(), state)
        assertFalse(RglConfigStore.configFile(dir).exists())
        dir.deleteRecursively()
    }

    @Test
    fun theFileIsTheOneWse2Reads() {
        val path = RglConfigStore.configFile(java.io.File("x")).path.replace('\\', '/')
        assertTrue(path, path.endsWith("wse2/user/Documents/Mount&Blade Warband WSE2/rgl_config.ini"))
    }

    @Test
    fun readsWse2Keys() {
        val s = RglConfigCodec.parse(wse2Ini)
        assertEquals(0.25f, s.battleSize, 1e-6f)
        assertEquals(150, s.battleSizeMax)
        assertEquals(30, s.battleSizeMin)
        assertEquals(60, s.maxFramerate)
        assertTrue(s.music)
        assertFalse(s.cheatMode)
        assertEquals(0.4f, s.musicVolume, 1e-6f)
        // Not in the file: the default WSE2 would use.
        assertEquals(RglConfigState().gamma, s.gamma, 0f)
    }

    @Test
    fun anUnchangedStateWritesTheFileBackByteForByte() {
        val s = RglConfigCodec.parse(wse2Ini)
        assertEquals(wse2Ini, RglConfigCodec.serialize(s))
    }

    @Test
    fun aChangeRewritesOnlyItsOwnLine() {
        val s = RglConfigCodec.parse(wse2Ini).copy(battleSize = 1f, cheatMode = true)
        val out = RglConfigCodec.serialize(s)
        val expected = wse2Ini
            .replace("fBattleSize=0.250000", "fBattleSize=1.000000")
            .replace("bCheatMode=false", "bCheatMode=true")
        assertEquals(expected, out)
        assertEquals(s.copy(ini = RglIni.parse(out)), RglConfigCodec.parse(out))
    }

    @Test
    fun aMissingKeyGoesIntoItsSectionAndAMissingSectionIsAdded() {
        val s = RglConfigCodec.parse(wse2Ini).copy(soundVolume = 0.5f, gamma = 2.5f)
        val ini = RglConfigCodec.toIni(s)
        assertEquals("0.500000", ini.get("Sound", "fSoundVolume"))
        assertEquals("2.500000", ini.get("Graphics", "fGamma"))
        val lines = ini.lines
        val sound = lines.indexOf("[Sound]")
        assertTrue(lines.indexOf("fSoundVolume=0.500000") > sound)
        assertEquals("[Graphics]", lines[lines.size - 2])
        // Everything WSE2 wrote is still there, in order.
        val kept = lines.filter { it != "fSoundVolume=0.500000" && it != "[Graphics]" && it != "fGamma=2.500000" }
        assertEquals(RglIni.parse(wse2Ini).lines, kept)
    }

    @Test
    fun aNewFileHoldsOnlyWhatThePlayerChanged() {
        val dir = tempFilesDir()
        assertTrue(RglConfigStore.save(dir, RglConfigState().copy(battleSizeMax = 1000)))
        val text = RglConfigStore.configFile(dir).readText()
        assertEquals("[Battle]\r\niBattleSizeMax=1000\r\n", text)
        assertEquals(1000, RglConfigStore.load(dir).first.battleSizeMax)
        dir.deleteRecursively()
    }

    @Test
    fun keysAndSectionsMatchWithoutCase() {
        val ini = RglIni.parse("[battle]\nibattlesizemax = 600\n")
        assertEquals("600", ini.get("Battle", "iBattleSizeMax"))
        assertEquals(600, RglConfigCodec.fromIni(ini).battleSizeMax)
        assertNull(ini.get("General", "iBattleSizeMax"))
        // The file's own spelling of the key is kept on a rewrite; its LF endings too.
        val out = RglConfigCodec.serialize(RglConfigCodec.fromIni(ini).copy(battleSizeMax = 900))
        assertEquals("[battle]\nibattlesizemax=900\n", out)
    }

    @Test
    fun battleSizeFollowsTheModuleRange() {
        val s = RglConfigState(battleSize = 0.5f, battleSizeMin = 30, battleSizeMax = 150)
        assertEquals(90, s.battleSizeTroops())
        assertEquals(360, s.battleSizeTroops(120..600))
        assertEquals(6000, s.copy(battleSize = 1f).battleSizeTroops(100..6000))
        assertEquals(120..600, moduleBattleSizeRange(120, 600))
        assertNull(moduleBattleSizeRange(0, 0))
        assertNull(moduleBattleSizeRange(600, 120))
    }

    @Test
    fun savingKeepsAnUnreadableFile() {
        val dir = tempFilesDir()
        val file = RglConfigStore.configFile(dir)
        // A directory where the file should be stands in for a file that cannot be read.
        java.io.File(file, "inner").mkdirs()
        assertEquals(RglConfigSource.UNREADABLE, RglConfigStore.load(dir).second)
        assertTrue(RglConfigStore.save(dir, RglConfigState(maxFramerate = 90)))
        assertTrue(java.io.File(file.parentFile, file.name + ".unreadable").isDirectory)
        assertEquals(90, RglConfigStore.load(dir).first.maxFramerate)
        dir.deleteRecursively()
    }

    @Test
    fun resetMovesTheFileAsideForWse2ToRewrite() {
        val dir = tempFilesDir()
        val file = RglConfigStore.configFile(dir)
        file.parentFile!!.mkdirs()
        file.writeText(wse2Ini)
        assertTrue(RglConfigStore.reset(dir))
        assertFalse(file.exists())
        assertEquals(wse2Ini, java.io.File(file.parentFile, file.name + ".before_reset").readText())
        assertEquals(RglConfigSource.MISSING, RglConfigStore.load(dir).second)
        assertTrue(RglConfigStore.reset(dir))
        dir.deleteRecursively()
    }

    @Test
    fun qualityPresetsChangePictureSettingsOnly() {
        val base = RglConfigCodec.parse(wse2Ini)
        val low = base.withQualityPreset("low")
        val high = base.withQualityPreset("high")
        assertFalse(low.shadows)
        assertTrue(high.shadows && high.characterShadows && high.particles)
        assertTrue(low.characterDetail < high.characterDetail)
        assertEquals(base.battleSize, high.battleSize, 0f)
        assertEquals(base.battleSizeMax, low.battleSizeMax)
    }

    @Test
    fun everyEditedKeyIsDistinct() {
        val keys = RglConfigCodec.EDITED_KEYS.map { it.lowercase() }
        assertEquals(keys.size, keys.toSet().size)
    }
}
