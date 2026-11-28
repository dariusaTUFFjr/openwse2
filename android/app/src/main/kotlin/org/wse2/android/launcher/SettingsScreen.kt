package org.wse2.android.launcher

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import org.wse2.android.launcher.theme.LocalWseColors
import org.wse2.android.launcher.theme.Wse2Theme
import org.wse2.android.launcher.theme.wseBackdrop
import org.wse2.android.launcher.ui.WseCategoryStrip
import org.wse2.android.launcher.ui.WseCategoryTab
import org.wse2.android.launcher.ui.WseDialog
import org.wse2.android.launcher.ui.WseDialogAction
import org.wse2.android.launcher.ui.WseKeyValueGrid
import org.wse2.android.launcher.ui.WseMetricItem
import org.wse2.android.launcher.ui.WseScreenHeader
import org.wse2.android.launcher.ui.WseSecondaryButton
import org.wse2.android.launcher.ui.WseStatusDot
import org.wse2.android.launcher.ui.WseStatusTone
import java.util.Locale
import kotlin.math.roundToInt

private val SettingsPlaceholderPerformance = PerformanceUiState.fromJson("{}")

enum class SettingsCategory(val label: String) {
    PERFORMANCE("Performance"),
    GRAPHICS("Graphics"),
    WORLD("Battle"),
    GAMEPLAY("Gameplay"),
    AUDIO_INPUT("Audio & Input"),
    SYSTEM("System")
}

@Composable
fun SettingsScreen(
    state: SettingsUiState,
    onQualityPresetChange: (String) -> Unit,
    onFrameRateCapChange: (Int) -> Unit,
    onPresentModeChange: (String) -> Unit,
    onRenderScaleChange: (Int) -> Unit,
    onMaxTextureSizeChange: (Int) -> Unit,
    onThermalThrottlingChange: (Boolean) -> Unit,
    onDynamicResolutionChange: (Boolean) -> Unit,
    onLodBiasScalingChange: (Boolean) -> Unit,
    onUiAspectChange: (String) -> Unit,
    onUiScaleChange: (Int) -> Unit,
    onGyroAimChange: (Boolean) -> Unit,
    onPinchZoomChange: (Boolean) -> Unit,
    onLanguageChange: (String) -> Unit,
    modifier: Modifier = Modifier,
    topContentPadding: Dp = 0.dp,
    onUpdateRglConfig: ((RglConfigState) -> RglConfigState) -> Unit = {},
    onResetRglConfig: () -> Unit = {},
    selectedModule: ModuleChoice? = null,
    initialCategory: SettingsCategory = SettingsCategory.PERFORMANCE,
    pictureResolutionIndex: Int = 0,
    pictureSharpen: Boolean = true,
    pictureFxaa: Boolean = false,
    onPictureResolutionChange: (Int) -> Unit = {},
    onPictureSharpenChange: (Boolean) -> Unit = {},
    onPictureFxaaChange: (Boolean) -> Unit = {}
) {
    val perf = state.performance ?: SettingsPlaceholderPerformance
    val rgl = state.rglConfig
    val enabled = state.isLoaded
    var selectedCategory by rememberSaveable { mutableStateOf(initialCategory) }
    var showResetDialog by remember { mutableStateOf(false) }

    val categories = remember { SettingsCategory.entries }
    val categoryTabs = remember(categories) {
        categories.map { WseCategoryTab(label = it.label) }
    }

    Box(
        modifier = modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState()),
        contentAlignment = Alignment.TopCenter
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .widthIn(max = 720.dp)
                .padding(
                    start = 16.dp,
                    end = 16.dp,
                    top = topContentPadding + 16.dp,
                    bottom = 28.dp
                ),
            verticalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            WseScreenHeader(
                title = "Settings",
                trailing = {
                    WseStatusDot(
                        text = if (state.rglConfigReady) "rgl_config.ini" else "Defaults",
                        tone = if (state.rglConfigReady) WseStatusTone.Ready else WseStatusTone.Neutral
                    )
                }
            )

            WseCategoryStrip(
                tabs = categoryTabs,
                selectedIndex = categories.indexOf(selectedCategory).coerceAtLeast(0),
                onSelect = { idx -> selectedCategory = categories[idx] }
            )

            when (selectedCategory) {
                SettingsCategory.PERFORMANCE -> PerformanceSettingsSection(
                    state = state,
                    perf = perf,
                    enabled = enabled,
                    pictureResolutionIndex = pictureResolutionIndex,
                    pictureSharpen = pictureSharpen,
                    pictureFxaa = pictureFxaa,
                    onQualityPresetChange = onQualityPresetChange,
                    onFrameRateCapChange = onFrameRateCapChange,
                    onPresentModeChange = onPresentModeChange,
                    onPictureResolutionChange = onPictureResolutionChange,
                    onPictureSharpenChange = onPictureSharpenChange,
                    onPictureFxaaChange = onPictureFxaaChange,
                    onMaxTextureSizeChange = onMaxTextureSizeChange,
                    onThermalThrottlingChange = onThermalThrottlingChange,
                    onDynamicResolutionChange = onDynamicResolutionChange,
                    onLodBiasScalingChange = onLodBiasScalingChange,
                    onUiAspectChange = onUiAspectChange,
                    onUiScaleChange = onUiScaleChange,
                    onLanguageChange = onLanguageChange
                )
                SettingsCategory.GRAPHICS -> RglGraphicsSettingsSection(
                    rgl = rgl,
                    onUpdate = onUpdateRglConfig
                )
                SettingsCategory.WORLD -> RglBattleSettingsSection(
                    rgl = rgl,
                    selectedModule = selectedModule,
                    onUpdate = onUpdateRglConfig
                )
                SettingsCategory.GAMEPLAY -> RglGameplaySettingsSection(
                    rgl = rgl,
                    onUpdate = onUpdateRglConfig
                )
                SettingsCategory.AUDIO_INPUT -> AudioInputSettingsSection(
                    state = state,
                    perf = perf,
                    rgl = rgl,
                    perfEnabled = enabled,
                    onGyroAimChange = onGyroAimChange,
                    onPinchZoomChange = onPinchZoomChange,
                    onUpdateRgl = onUpdateRglConfig
                )
                SettingsCategory.SYSTEM -> RglSystemSettingsSection(
                    state = state,
                    rgl = rgl,
                    onUpdate = onUpdateRglConfig,
                    onRequestReset = { showResetDialog = true }
                )
            }
        }
    }

    if (showResetDialog) {
        WseDialog(
            title = "Reset game settings?",
            text = "Every setting in rgl_config.ini goes back to WSE2's own default the next " +
                "time the game starts, including ones changed in the game's Options menu. " +
                "The old file is kept as rgl_config.ini.before_reset.",
            confirm = WseDialogAction("Reset") {
                showResetDialog = false
                onResetRglConfig()
            },
            dismiss = WseDialogAction("Cancel") { showResetDialog = false },
            onDismissRequest = { showResetDialog = false }
        )
    }
}

@Composable
private fun PerformanceSettingsSection(
    state: SettingsUiState,
    perf: PerformanceUiState,
    enabled: Boolean,
    pictureResolutionIndex: Int,
    pictureSharpen: Boolean,
    pictureFxaa: Boolean,
    onQualityPresetChange: (String) -> Unit,
    onFrameRateCapChange: (Int) -> Unit,
    onPresentModeChange: (String) -> Unit,
    onPictureResolutionChange: (Int) -> Unit,
    onPictureSharpenChange: (Boolean) -> Unit,
    onPictureFxaaChange: (Boolean) -> Unit,
    onMaxTextureSizeChange: (Int) -> Unit,
    onThermalThrottlingChange: (Boolean) -> Unit,
    onDynamicResolutionChange: (Boolean) -> Unit,
    onLodBiasScalingChange: (Boolean) -> Unit,
    onUiAspectChange: (String) -> Unit,
    onUiScaleChange: (Int) -> Unit,
    onLanguageChange: (String) -> Unit
) {
    LauncherSectionHeader(title = "Performance presets")
    LauncherGroupedCard {
        val presetIdx = LauncherDefaults.QUALITY_PRESET_KEYS
            .indexOf(perf.qualityPresetKey)
            .coerceAtLeast(0)
        TileGridListItem(
            headline = "Quality preset",
            supporting = "Sets texture size and the game's shadow and detail settings",
            options = LauncherDefaults.QUALITY_PRESET_LABELS,
            selectedIndex = presetIdx,
            columns = 4,
            enabled = enabled,
            onSelected = { idx ->
                onQualityPresetChange(LauncherDefaults.QUALITY_PRESET_KEYS[idx])
            }
        )
        LauncherSectionDivider()

        val capIdx = state.frameRateValues
            .indexOf(perf.frameRateCapHz)
            .coerceAtLeast(0)
        val compactCapLabels = state.frameRateValues.map { hz ->
            if (hz <= 0) "Max" else "$hz"
        }
        val capReadout = if (perf.frameRateCapHz <= 0) "Uncapped" else "${perf.frameRateCapHz} FPS"
        TileGridListItem(
            headline = "Frame-rate cap",
            options = compactCapLabels,
            selectedIndex = capIdx,
            columns = compactCapLabels.size.coerceIn(1, 5),
            valueReadout = capReadout,
            enabled = enabled,
            onSelected = { idx ->
                onFrameRateCapChange(state.frameRateValues[idx])
            }
        )
        LauncherSectionDivider()

        val resolutionLabels = listOf("Auto", "100%", "85%", "75%", "67%", "50%")
        val resIdx = pictureResolutionIndex.coerceIn(0, resolutionLabels.lastIndex)
        TileGridListItem(
            headline = "Render resolution",
            supporting = if (resIdx == 0) {
                "Scales down when the phone can't keep up or gets warm, then sharpens back up"
            } else {
                "Internal 3D viewport resolution (also adjustable live in-game)"
            },
            options = resolutionLabels,
            selectedIndex = resIdx,
            columns = 3,
            valueReadout = if (resIdx == 0) "Auto" else resolutionLabels[resIdx],
            enabled = true,
            onSelected = onPictureResolutionChange
        )
        LauncherSectionDivider()

        SwitchListItem(
            headline = "Sharpen picture (FSR)",
            supporting = "Recovers crisp edges when resolution scales down",
            checked = pictureSharpen,
            enabled = true,
            onCheckedChange = onPictureSharpenChange
        )
        LauncherSectionDivider()

        SwitchListItem(
            headline = "Smooth jagged edges (FXAA)",
            checked = pictureFxaa,
            enabled = true,
            onCheckedChange = onPictureFxaaChange
        )
        LauncherSectionDivider()

        val texIdx = LauncherDefaults.TEXTURE_SIZE_VALUES
            .indexOf(perf.maxTextureSize)
            .coerceAtLeast(0)
        TileGridListItem(
            headline = "Texture size cap",
            supporting = "Clamps oversized DDS textures to save mobile GPU memory",
            options = LauncherDefaults.TEXTURE_SIZE_LABELS,
            selectedIndex = texIdx,
            columns = 3,
            enabled = enabled,
            onSelected = { idx ->
                onMaxTextureSizeChange(LauncherDefaults.TEXTURE_SIZE_VALUES[idx])
            }
        )
        LauncherSectionDivider()

        val modeIdx = state.presentModeKeys
            .indexOf(perf.presentModeKey)
            .coerceAtLeast(0)
        ExposedDropdownListItem(
            headline = "Present mode",
            options = state.presentModeLabels,
            selectedIndex = modeIdx,
            enabled = enabled,
            onSelected = { idx ->
                onPresentModeChange(state.presentModeKeys[idx])
            }
        )
    }

    LauncherSectionHeader(title = "Thermal management")
    LauncherGroupedCard {
        SwitchListItem(
            headline = "Thermal headroom throttling",
            supporting = "Adjusts resolution, LOD bias and FPS as the phone warms up",
            checked = perf.thermalThrottlingEnabled,
            enabled = enabled,
            onCheckedChange = onThermalThrottlingChange
        )
        LauncherSectionDivider()
        SwitchListItem(
            headline = "Dynamic resolution under thermal load",
            checked = perf.dynamicResolutionEnabled,
            enabled = enabled,
            onCheckedChange = onDynamicResolutionChange
        )
        LauncherSectionDivider()
        SwitchListItem(
            headline = "LOD bias scaling under thermal load",
            checked = perf.lodBiasScalingEnabled,
            enabled = enabled,
            onCheckedChange = onLodBiasScalingChange
        )
    }

    LauncherSectionHeader(title = "Display & language")
    LauncherGroupedCard {
        val aspectIdx = LauncherDefaults.UI_ASPECT_KEYS
            .indexOf(perf.uiAspectKey)
            .coerceAtLeast(0)
        SegmentedListItem(
            headline = "UI aspect",
            supporting = "Keeps 2D menus undistorted on tall phone screens",
            options = LauncherDefaults.UI_ASPECT_LABELS,
            selectedIndex = aspectIdx,
            enabled = enabled,
            showSelectedIcon = true,
            onSelected = { idx ->
                onUiAspectChange(LauncherDefaults.UI_ASPECT_KEYS[idx])
            }
        )
        LauncherSectionDivider()
        SliderListItem(
            headline = "UI text scale",
            persistedPercent = perf.uiScalePercent,
            minPercent = LauncherDefaults.UI_SCALE_MIN,
            maxPercent = LauncherDefaults.UI_SCALE_MAX,
            stepPercent = LauncherDefaults.SCALE_STEP,
            enabled = enabled,
            onCommitPercent = onUiScaleChange
        )
        LauncherSectionDivider()
        val langOptions = state.languageOptions.ifEmpty { listOf("en") }
        val langIdx = langOptions.indexOf(perf.language).coerceAtLeast(0)
        ExposedDropdownListItem(
            headline = "Game language",
            options = langOptions,
            selectedIndex = langIdx,
            enabled = enabled,
            onSelected = { idx ->
                onLanguageChange(langOptions[idx])
            },
            supporting = "Detected from the install's languages/ folder"
        )
    }
}

@Composable
private fun RglGraphicsSettingsSection(
    rgl: RglConfigState,
    onUpdate: ((RglConfigState) -> RglConfigState) -> Unit
) {
    LauncherSectionHeader(title = "Picture")
    LauncherGroupedCard {
        SwitchListItem(
            headline = "Shadows",
            supporting = "Shadows from the sun on the scene",
            checked = rgl.shadows,
            enabled = true,
            onCheckedChange = { v -> onUpdate { it.copy(shadows = v) } }
        )
        LauncherSectionDivider()
        SwitchListItem(
            headline = "Character shadows",
            checked = rgl.characterShadows,
            enabled = true,
            onCheckedChange = { v -> onUpdate { it.copy(characterShadows = v) } }
        )
        LauncherSectionDivider()
        SwitchListItem(
            headline = "Particle effects",
            supporting = "Smoke, dust, sparks and weather",
            checked = rgl.particles,
            enabled = true,
            onCheckedChange = { v -> onUpdate { it.copy(particles = v) } }
        )
        LauncherSectionDivider()
        FloatSliderListItem(
            headline = "Character detail",
            persistedValue = rgl.characterDetail,
            minValue = 0f,
            maxValue = 1f,
            step = 0.05f,
            enabled = true,
            formatValue = { "${(it * 100f).roundToInt()}%" },
            onCommitValue = { v -> onUpdate { it.copy(characterDetail = v) } }
        )
        LauncherSectionDivider()
        FloatSliderListItem(
            headline = "Tree detail distance",
            supporting = "Beyond this, trees and plants are drawn simpler",
            persistedValue = rgl.floraDegradeDistance,
            minValue = 20f,
            maxValue = 300f,
            step = 5f,
            enabled = true,
            formatValue = { "${it.roundToInt()} m" },
            onCommitValue = { v -> onUpdate { it.copy(floraDegradeDistance = v) } }
        )
        LauncherSectionDivider()
        FloatSliderListItem(
            headline = "Gamma",
            persistedValue = rgl.gamma,
            minValue = 1.0f,
            maxValue = 3.0f,
            step = 0.05f,
            enabled = true,
            formatValue = { String.format(Locale.US, "%.2f", it) },
            onCommitValue = { v -> onUpdate { it.copy(gamma = v) } }
        )
        LauncherSectionDivider()
        SwitchListItem(
            headline = "Load textures as needed",
            supporting = "Uses less memory; a texture can show blurry for a moment",
            checked = rgl.useOnDemandTextures,
            enabled = true,
            onCheckedChange = { v -> onUpdate { it.copy(useOnDemandTextures = v) } }
        )
        LauncherSectionDivider()
        SwitchListItem(
            headline = "Instancing",
            supporting = "Draws repeated props and plants together",
            checked = rgl.useInstancing,
            enabled = true,
            onCheckedChange = { v -> onUpdate { it.copy(useInstancing = v) } }
        )
    }

    LauncherSectionHeader(title = "Frame rate")
    LauncherGroupedCard {
        IntSliderListItem(
            headline = "Game frame-rate limit",
            supporting = "Follows the frame-rate cap under Performance",
            persistedValue = rgl.maxFramerate.coerceIn(30, 240),
            minValue = 30,
            maxValue = 240,
            step = 10,
            enabled = true,
            formatValue = { "$it FPS" },
            onCommitValue = { v -> onUpdate { it.copy(maxFramerate = v) } }
        )
        LauncherSectionDivider()
        SwitchListItem(
            headline = "Vertical sync",
            checked = rgl.forceVsync,
            enabled = true,
            onCheckedChange = { v -> onUpdate { it.copy(forceVsync = v) } }
        )
        LauncherSectionDivider()
        SwitchListItem(
            headline = "Show the game's frame rate",
            checked = rgl.showFramerate,
            enabled = true,
            onCheckedChange = { v -> onUpdate { it.copy(showFramerate = v) } }
        )
    }
}

@Composable
private fun RglBattleSettingsSection(
    rgl: RglConfigState,
    selectedModule: ModuleChoice?,
    onUpdate: ((RglConfigState) -> RglConfigState) -> Unit
) {
    val moduleRange = selectedModule?.battleSizeRange
    val moduleName = selectedModule?.displayName?.ifBlank { null } ?: selectedModule?.folderName
    LauncherSectionHeader(title = "Battle size")
    LauncherGroupedCard {
        FloatSliderListItem(
            headline = "Battle size",
            supporting = if (moduleRange != null && moduleName != null) {
                "$moduleName allows ${moduleRange.first} to ${moduleRange.last} soldiers"
            } else {
                "Soldiers on the field at once"
            },
            persistedValue = rgl.battleSize,
            minValue = 0f,
            maxValue = 1f,
            step = 0.01f,
            enabled = true,
            formatValue = { v -> "${rgl.copy(battleSize = v).battleSizeTroops(moduleRange)} soldiers" },
            onCommitValue = { v -> onUpdate { it.copy(battleSize = v) } }
        )
        LauncherSectionDivider()
        // A value set in the file or the game itself (say 600) is shown as its own choice.
        val maxValues = (RglConfigOptions.BATTLE_SIZE_MAX_VALUES + rgl.battleSizeMax).distinct().sorted()
        val maxIdx = maxValues.indexOf(rgl.battleSizeMax)
        TileGridListItem(
            headline = "Largest battle",
            supporting = if (moduleRange != null) {
                "Used by modules that set no range of their own. Large battles are heavy on a phone."
            } else {
                "Large battles are heavy on a phone."
            },
            options = maxValues.map { it.toString() },
            selectedIndex = maxIdx,
            columns = 4,
            valueReadout = "${rgl.battleSizeMax} soldiers",
            enabled = true,
            onSelected = { idx ->
                val chosen = maxValues[idx]
                onUpdate { it.copy(battleSizeMax = chosen, battleSizeMin = it.battleSizeMin.coerceAtMost(chosen)) }
            }
        )
    }

    LauncherSectionHeader(title = "Battle & difficulty")
    LauncherGroupedCard {
        SwitchListItem(
            headline = "Blood",
            checked = rgl.enableBlood,
            enabled = true,
            onCheckedChange = { v -> onUpdate { it.copy(enableBlood = v) } }
        )
        LauncherSectionDivider()
        TileGridListItem(
            headline = "Combat speed",
            options = RglConfigOptions.COMBAT_SPEED_LABELS,
            selectedIndex = rgl.combatSpeed.coerceIn(0, 4),
            columns = 3,
            enabled = true,
            onSelected = { idx -> onUpdate { it.copy(combatSpeed = idx) } }
        )
        LauncherSectionDivider()
        TileGridListItem(
            headline = "Damage to you",
            options = RglConfigOptions.DAMAGE_LABELS,
            selectedIndex = RglConfigOptions.nearestIndex(RglConfigOptions.DAMAGE_VALUES, rgl.damageToPlayer),
            columns = 3,
            enabled = true,
            onSelected = { idx -> onUpdate { it.copy(damageToPlayer = RglConfigOptions.DAMAGE_VALUES[idx]) } }
        )
        LauncherSectionDivider()
        TileGridListItem(
            headline = "Damage to your troops",
            options = RglConfigOptions.DAMAGE_LABELS,
            selectedIndex = RglConfigOptions.nearestIndex(RglConfigOptions.DAMAGE_VALUES, rgl.damageToFriends),
            columns = 3,
            enabled = true,
            onSelected = { idx -> onUpdate { it.copy(damageToFriends = RglConfigOptions.DAMAGE_VALUES[idx]) } }
        )
        LauncherSectionDivider()
        TileGridListItem(
            headline = "Battle AI",
            options = RglConfigOptions.AI_LEVEL_LABELS,
            selectedIndex = rgl.reduceCombatAi.coerceIn(0, 2),
            columns = 3,
            enabled = true,
            onSelected = { idx -> onUpdate { it.copy(reduceCombatAi = idx) } }
        )
        LauncherSectionDivider()
        TileGridListItem(
            headline = "Campaign AI",
            options = RglConfigOptions.AI_LEVEL_LABELS,
            selectedIndex = rgl.reduceCampaignAi.coerceIn(0, 2),
            columns = 3,
            enabled = true,
            onSelected = { idx -> onUpdate { it.copy(reduceCampaignAi = idx) } }
        )
    }
}

@Composable
private fun RglGameplaySettingsSection(
    rgl: RglConfigState,
    onUpdate: ((RglConfigState) -> RglConfigState) -> Unit
) {
    LauncherSectionHeader(title = "Combat controls")
    LauncherGroupedCard {
        ExposedDropdownListItem(
            headline = "Attack direction",
            options = RglConfigOptions.ATTACK_CONTROL_LABELS,
            selectedIndex = rgl.attackDirectionControl.coerceIn(0, 3),
            enabled = true,
            onSelected = { idx -> onUpdate { it.copy(attackDirectionControl = idx) } }
        )
        LauncherSectionDivider()
        ExposedDropdownListItem(
            headline = "Block direction",
            options = RglConfigOptions.DEFEND_CONTROL_LABELS,
            selectedIndex = rgl.defendDirectionControl.coerceIn(0, 3),
            enabled = true,
            onSelected = { idx -> onUpdate { it.copy(defendDirectionControl = idx) } }
        )
        LauncherSectionDivider()
        TileGridListItem(
            headline = "Lance control",
            options = RglConfigOptions.LANCE_CONTROL_LABELS,
            selectedIndex = rgl.lanceControl.coerceIn(0, 2),
            columns = 3,
            enabled = true,
            onSelected = { idx -> onUpdate { it.copy(lanceControl = idx) } }
        )
        LauncherSectionDivider()
        TileGridListItem(
            headline = "Turn camera with horse",
            options = RglConfigOptions.HORSE_CAMERA_LABELS,
            selectedIndex = rgl.horseCameraMode.coerceIn(0, 2),
            columns = 3,
            enabled = true,
            onSelected = { idx -> onUpdate { it.copy(horseCameraMode = idx) } }
        )
    }

    LauncherSectionHeader(title = "Screen & messages")
    LauncherGroupedCard {
        SwitchListItem(
            headline = "Crosshair",
            checked = rgl.showCrosshair,
            enabled = true,
            onCheckedChange = { v -> onUpdate { it.copy(showCrosshair = v) } }
        )
        LauncherSectionDivider()
        SwitchListItem(
            headline = "Aiming reticle",
            checked = rgl.showReticle,
            enabled = true,
            onCheckedChange = { v -> onUpdate { it.copy(showReticle = v) } }
        )
        LauncherSectionDivider()
        SwitchListItem(
            headline = "Attack direction arrow",
            checked = rgl.showAttackDirection,
            enabled = true,
            onCheckedChange = { v -> onUpdate { it.copy(showAttackDirection = v) } }
        )
        LauncherSectionDivider()
        SwitchListItem(
            headline = "Damage messages",
            checked = rgl.showDamage,
            enabled = true,
            onCheckedChange = { v -> onUpdate { it.copy(showDamage = v) } }
        )
        LauncherSectionDivider()
        SwitchListItem(
            headline = "Shot difficulty messages",
            checked = rgl.showShotDifficulty,
            enabled = true,
            onCheckedChange = { v -> onUpdate { it.copy(showShotDifficulty = v) } }
        )
        LauncherSectionDivider()
        SwitchListItem(
            headline = "Casualty messages",
            checked = rgl.showCasualties,
            enabled = true,
            onCheckedChange = { v -> onUpdate { it.copy(showCasualties = v) } }
        )
        LauncherSectionDivider()
        SwitchListItem(
            headline = "Experience messages",
            checked = rgl.showExperienceGain,
            enabled = true,
            onCheckedChange = { v -> onUpdate { it.copy(showExperienceGain = v) } }
        )
    }
}

@Composable
private fun AudioInputSettingsSection(
    state: SettingsUiState,
    perf: PerformanceUiState,
    rgl: RglConfigState,
    perfEnabled: Boolean,
    onGyroAimChange: (Boolean) -> Unit,
    onPinchZoomChange: (Boolean) -> Unit,
    onUpdateRgl: ((RglConfigState) -> RglConfigState) -> Unit
) {
    LauncherSectionHeader(title = "Audio")
    LauncherGroupedCard {
        SwitchListItem(
            headline = "Music",
            checked = rgl.music,
            enabled = true,
            onCheckedChange = { v -> onUpdateRgl { it.copy(music = v) } }
        )
        LauncherSectionDivider()
        FloatSliderListItem(
            headline = "Music volume",
            persistedValue = rgl.musicVolume,
            minValue = 0f,
            maxValue = 1f,
            step = 0.05f,
            enabled = rgl.music,
            formatValue = { "${(it * 100f).roundToInt()}%" },
            onCommitValue = { v -> onUpdateRgl { it.copy(musicVolume = v) } }
        )
        LauncherSectionDivider()
        SwitchListItem(
            headline = "Sound effects",
            checked = rgl.sound,
            enabled = true,
            onCheckedChange = { v -> onUpdateRgl { it.copy(sound = v) } }
        )
        LauncherSectionDivider()
        FloatSliderListItem(
            headline = "Sound effects volume",
            persistedValue = rgl.soundVolume,
            minValue = 0f,
            maxValue = 1f,
            step = 0.05f,
            enabled = rgl.sound,
            formatValue = { "${(it * 100f).roundToInt()}%" },
            onCommitValue = { v -> onUpdateRgl { it.copy(soundVolume = v) } }
        )
        LauncherSectionDivider()
        SwitchListItem(
            headline = "Same pitch for every sound",
            supporting = "Turns off the small pitch changes between repeats",
            checked = rgl.disableFrequencyVariation,
            enabled = true,
            onCheckedChange = { v -> onUpdateRgl { it.copy(disableFrequencyVariation = v) } }
        )
    }

    LauncherSectionHeader(title = "Touch, sensors & controls")
    LauncherGroupedCard {
        SwitchListItem(
            headline = "Opt-in gyro aiming (Mission)",
            supporting = if (state.hasPhoneGyro) {
                "Tilt device or controller for fine ranged aiming"
            } else {
                "No phone gyroscope detected; controller gyro only"
            },
            checked = perf.gyroAimEnabled,
            enabled = perfEnabled,
            onCheckedChange = onGyroAimChange
        )
        LauncherSectionDivider()
        SwitchListItem(
            headline = "Pinch-to-zoom (Campaign map)",
            checked = perf.pinchZoomEnabled,
            enabled = perfEnabled,
            onCheckedChange = onPinchZoomChange
        )
        LauncherSectionDivider()
        SwitchListItem(
            headline = "Gamepad vibration",
            checked = rgl.enableGamepadVibration,
            enabled = true,
            onCheckedChange = { v -> onUpdateRgl { it.copy(enableGamepadVibration = v) } }
        )
        LauncherSectionDivider()
        FloatSliderListItem(
            headline = "Look sensitivity",
            persistedValue = rgl.mouseSensitivity,
            minValue = 0.05f,
            maxValue = 3.0f,
            step = 0.05f,
            enabled = true,
            formatValue = { String.format(Locale.US, "%.2f", it) },
            onCommitValue = { v -> onUpdateRgl { it.copy(mouseSensitivity = v) } }
        )
        LauncherSectionDivider()
        FloatSliderListItem(
            headline = "Up-down look speed",
            persistedValue = rgl.mouseYScale,
            minValue = 0.25f,
            maxValue = 4.0f,
            step = 0.05f,
            enabled = true,
            formatValue = { String.format(Locale.US, "%.2fx", it) },
            onCommitValue = { v -> onUpdateRgl { it.copy(mouseYScale = v) } }
        )
        LauncherSectionDivider()
        SwitchListItem(
            headline = "Invert up-down look",
            checked = rgl.invertMouseY,
            enabled = true,
            onCheckedChange = { v -> onUpdateRgl { it.copy(invertMouseY = v) } }
        )
    }
}

@Composable
private fun RglSystemSettingsSection(
    state: SettingsUiState,
    rgl: RglConfigState,
    onUpdate: ((RglConfigState) -> RglConfigState) -> Unit,
    onRequestReset: () -> Unit
) {
    val colors = LocalWseColors.current
    LauncherSectionHeader(title = "Cheats & editing")
    LauncherGroupedCard {
        SwitchListItem(
            headline = "Cheats",
            supporting = "Turns on Warband's cheat keys and the cheats menu",
            checked = rgl.cheatMode,
            enabled = true,
            onCheckedChange = { v -> onUpdate { it.copy(cheatMode = v) } }
        )
        LauncherSectionDivider()
        SwitchListItem(
            headline = "Edit mode",
            supporting = "Scene editing tools",
            checked = rgl.enableEditMode,
            enabled = true,
            onCheckedChange = { v -> onUpdate { it.copy(enableEditMode = v) } }
        )
    }

    LauncherSectionHeader(title = "Config file")
    LauncherGroupedCard {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            WseKeyValueGrid(
                items = listOf(
                    WseMetricItem(
                        label = "Config status",
                        value = if (state.rglConfigReady) "Ready" else "Made on first start",
                        accentValue = state.rglConfigReady
                    ),
                    WseMetricItem(
                        label = "Settings",
                        value = if (rgl.ini.keyCount > 0) "${rgl.ini.keyCount} in file" else "None yet"
                    )
                ),
                columns = 2
            )
            Text(
                text = state.rglConfigPath,
                style = MaterialTheme.typography.bodySmall.copy(fontFamily = FontFamily.Monospace),
                color = colors.onSurfaceVariant
            )
            WseSecondaryButton(
                text = "Reset game settings",
                onClick = onRequestReset,
                modifier = Modifier.fillMaxWidth()
            )
        }
    }
}

private fun previewSettingsState(): SettingsUiState = SettingsUiState(
    performance = PerformanceUiState(
        qualityPresetKey = "medium",
        frameRateCapHz = 60,
        presentModeKey = "fifo",
        renderScalePercent = 85,
        maxTextureSize = 1024,
        thermalThrottlingEnabled = true,
        dynamicResolutionEnabled = true,
        lodBiasScalingEnabled = true,
        uiAspectKey = "4:3",
        uiScalePercent = 100,
        gyroAimEnabled = false,
        pinchZoomEnabled = true,
        language = "en"
    ),
    isLoaded = true,
    languageOptions = listOf("en", "de", "fr", "pl"),
    hasPhoneGyro = true,
    rglConfig = RglConfigState(),
    rglConfigReady = true
)

@Preview(name = "Settings Performance Dark", showBackground = true, widthDp = 412, heightDp = 1280)
@Composable
private fun PreviewSettingsDark() {
    Wse2Theme(darkTheme = true, dynamicColor = false) {
        Box(modifier = Modifier.wseBackdrop(LocalWseColors.current.accent)) {
            SettingsScreen(
                state = previewSettingsState(),
                onQualityPresetChange = {},
                onFrameRateCapChange = {},
                onPresentModeChange = {},
                onRenderScaleChange = {},
                onMaxTextureSizeChange = {},
                onThermalThrottlingChange = {},
                onDynamicResolutionChange = {},
                onLodBiasScalingChange = {},
                onUiAspectChange = {},
                onUiScaleChange = {},
                onGyroAimChange = {},
                onPinchZoomChange = {},
                onLanguageChange = {}
            )
        }
    }
}

@Preview(name = "Settings Graphics rgl_config", showBackground = true, widthDp = 412, heightDp = 1280)
@Composable
private fun PreviewSettingsRglGraphics() {
    Wse2Theme(darkTheme = true, dynamicColor = false) {
        Box(modifier = Modifier.wseBackdrop(LocalWseColors.current.accent)) {
            SettingsScreen(
                state = previewSettingsState(),
                onQualityPresetChange = {},
                onFrameRateCapChange = {},
                onPresentModeChange = {},
                onRenderScaleChange = {},
                onMaxTextureSizeChange = {},
                onThermalThrottlingChange = {},
                onDynamicResolutionChange = {},
                onLodBiasScalingChange = {},
                onUiAspectChange = {},
                onUiScaleChange = {},
                onGyroAimChange = {},
                onPinchZoomChange = {},
                onLanguageChange = {},
                initialCategory = SettingsCategory.GRAPHICS
            )
        }
    }
}
