package org.wse2.android.launcher

import android.app.Application
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.CircularProgressIndicator
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
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import org.wse2.android.R
import org.wse2.android.launcher.theme.LocalWseColors
import org.wse2.android.launcher.theme.Wse2Theme
import org.wse2.android.launcher.theme.wseBackdrop
import org.wse2.android.launcher.ui.WseDialog
import org.wse2.android.launcher.ui.WseDialogAction
import org.wse2.android.launcher.ui.WseListRow
import org.wse2.android.launcher.ui.WseScreenHeader

@Composable
fun AdvancedScreen(
    state: AdvancedUiState,
    onVerifyScripts: () -> Unit,
    onSelectDriver: (Int) -> Unit,
    onImportDriver: () -> Unit,
    onAllowMockFallbackChange: (Boolean) -> Unit,
    onProbeVulkan: () -> Unit,
    onShowFpsOverlayChange: (Boolean) -> Unit,
    onLaunchSelfTest: () -> Unit,
    modifier: Modifier = Modifier,
    topContentPadding: Dp = 0.dp,
    r0State: R0UiState = R0UiState(),
    onRunR0Spikes: () -> Unit = {},
    onDismissR0Report: () -> Unit = {},
    r4State: R4UiState = R4UiState(),
    onRunR4Startup: () -> Unit = {},
    onLaunchRecompVulkan: () -> Unit = {},
    onR4UseCopyChange: (Boolean) -> Unit = {},
    onClearInstallCache: () -> Unit = {},
    onClearTextureCache: () -> Unit = {},
    onClearSoCache: () -> Unit = {},
    onClearShaderCache: () -> Unit = {},
    onClearSaveBackups: () -> Unit = {},
    onOpenSaveManager: () -> Unit = {},
    onCustomDriverSharedBuffersChange: (Boolean) -> Unit = {},
    onStrictTsoChange: (Boolean) -> Unit = {},
    onRecordPgoProfileChange: (Boolean) -> Unit = {},
    onClearPgoProfile: () -> Unit = {},
    onClearCrashReport: () -> Unit = {},
    onDismissR4Report: () -> Unit = {},
    onPictureResolutionChange: (Int) -> Unit = {},
    onPictureSharpenChange: (Boolean) -> Unit = {},
    onPictureFxaaChange: (Boolean) -> Unit = {}
) {
    val colors = LocalWseColors.current
    val treeUri = state.installFolder.treeUri
    val selectedModule = state.selectedModule
    val verifyEnabled = selectedModule?.hasScriptsTxt == true &&
        treeUri != null &&
        !state.scanState.inProgress
    var showLicencesDialog by remember { mutableStateOf(false) }
    var showCrashReportDialog by remember { mutableStateOf(false) }
    var confirmClear by remember { mutableStateOf<ConfirmClear?>(null) }
    var showDevTools by rememberSaveable { mutableStateOf(false) }
    val devToolsVisible = showDevTools || r0State.running || r4State.running

    val clearCacheEnabled = !r4State.running && !r0State.running
    val soMb = r4State.soCacheBytes / (1024L * 1024L)
    val texCacheMb = r4State.textureCacheBytes / (1024L * 1024L)
    val shaderMb = r4State.shaderCacheBytes / (1024L * 1024L)
    val cacheMb = r4State.installCacheBytes / (1024L * 1024L)
    val backupsMb = r4State.saveBackupsBytes / (1024L * 1024L)
    val totalMb = soMb + texCacheMb + shaderMb + cacheMb + backupsMb

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
                    bottom = 24.dp
                ),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            WseScreenHeader(title = "Advanced")

            LauncherSectionHeader(title = "GPU Driver")
            LauncherGroupedCard {
                ExposedDropdownListItem(
                    headline = "Vulkan driver",
                    options = state.driverState.options,
                    selectedIndex = state.driverState.selectedIndex,
                    enabled = true,
                    onSelected = onSelectDriver,
                    supporting = state.driverState.statusText
                )
                LauncherSectionDivider()

                WseListRow(
                    title = "Import driver package",
                    supporting = "Load a custom Adrenotools .zip package",
                    trailing = {
                        Text(
                            text = "Import",
                            style = MaterialTheme.typography.labelLarge,
                            color = colors.primary
                        )
                    },
                    showDivider = true,
                    onClick = onImportDriver
                )

                WseListRow(
                    title = "Probe Vulkan driver",
                    supporting = "Inspect physical devices and extensions",
                    trailing = {
                        if (state.driverState.isProbeInProgress) {
                            CircularProgressIndicator(
                                color = colors.accent,
                                modifier = Modifier.size(18.dp),
                                strokeWidth = 2.dp
                            )
                        } else {
                            Text(
                                text = "Probe",
                                style = MaterialTheme.typography.labelLarge,
                                color = colors.primary
                            )
                        }
                    },
                    enabled = !state.driverState.isProbeInProgress,
                    showDivider = false,
                    onClick = onProbeVulkan
                )
            }

            LauncherSectionHeader(
                title = "Saves & Storage",
                badge = if (totalMb > 0L) "$totalMb MB" else null
            )
            LauncherGroupedCard {
                WseListRow(
                    title = "Manage saved games",
                    supporting = "Inspect, import/export .sav files, restore backups, and configure auto-sync mirror.",
                    trailing = {
                        Text(
                            text = "Open",
                            style = MaterialTheme.typography.labelLarge,
                            color = colors.primary
                        )
                    },
                    showDivider = true,
                    onClick = onOpenSaveManager
                )
                WseListRow(
                    title = "Translated game code",
                    supporting = "$soMb MB of prepared arm64 code batches. Freeing this requires translating again or re-importing a PC zip.",
                    trailing = {
                        Text(
                            text = "Free up",
                            style = MaterialTheme.typography.labelLarge,
                            color = if (clearCacheEnabled) colors.primary else colors.onSurfaceVariant.copy(alpha = 0.45f)
                        )
                    },
                    showDivider = true,
                    enabled = clearCacheEnabled,
                    onClick = { confirmClear = ConfirmClear.TRANSLATED_CODE }
                )
                WseListRow(
                    title = "Converted textures",
                    supporting = "$texCacheMb MB. Textures converted for your phone's graphics chip. Cleared files are converted again on the next load.",
                    trailing = {
                        Text(
                            text = "Free up",
                            style = MaterialTheme.typography.labelLarge,
                            color = if (clearCacheEnabled) colors.primary else colors.onSurfaceVariant.copy(alpha = 0.45f)
                        )
                    },
                    showDivider = true,
                    enabled = clearCacheEnabled,
                    onClick = onClearTextureCache
                )
                WseListRow(
                    title = "Vulkan shader cache",
                    supporting = "$shaderMb MB of compiled graphics shaders. Rebuilds automatically if cleared.",
                    trailing = {
                        Text(
                            text = "Free up",
                            style = MaterialTheme.typography.labelLarge,
                            color = if (clearCacheEnabled) colors.primary else colors.onSurfaceVariant.copy(alpha = 0.45f)
                        )
                    },
                    showDivider = true,
                    enabled = clearCacheEnabled,
                    onClick = onClearShaderCache
                )
                WseListRow(
                    title = "Copied game files",
                    supporting = "$cacheMb MB kept in the app so the game loads faster. Free this space if you need it; the game copies files again as it reads them.",
                    trailing = {
                        Text(
                            text = "Free up",
                            style = MaterialTheme.typography.labelLarge,
                            color = if (clearCacheEnabled) colors.primary else colors.onSurfaceVariant.copy(alpha = 0.45f)
                        )
                    },
                    showDivider = true,
                    enabled = clearCacheEnabled,
                    onClick = onClearInstallCache
                )
                WseListRow(
                    title = "Savegame rolling backups",
                    supporting = "$backupsMb MB of automatic .bak1-.bak3 save backups. Live saves are untouched.",
                    trailing = {
                        Text(
                            text = "Free up",
                            style = MaterialTheme.typography.labelLarge,
                            color = if (clearCacheEnabled) colors.primary else colors.onSurfaceVariant.copy(alpha = 0.45f)
                        )
                    },
                    showDivider = false,
                    enabled = clearCacheEnabled,
                    onClick = { confirmClear = ConfirmClear.SAVE_BACKUPS }
                )
            }

            LauncherSectionHeader(title = "Runtime")
            LauncherGroupedCard {
                SwitchListItem(
                    headline = "Show FPS overlay",
                    supporting = "Shows the frame-rate pill at the top of the screen while playing",
                    checked = state.showFpsOverlay,
                    enabled = true,
                    onCheckedChange = onShowFpsOverlayChange
                )
            }

            state.crashReportText?.let { crashText ->
                LauncherSectionHeader(title = "Diagnostics")
                LauncherGroupedCard {
                    WseListRow(
                        title = "Last crash report",
                        supporting = crashText.lineSequence().firstOrNull { it.isNotBlank() } ?: "Crash report available",
                        trailing = {
                            Text(
                                text = "View",
                                style = MaterialTheme.typography.labelLarge,
                                color = colors.primary
                            )
                        },
                        showDivider = true,
                        onClick = { showCrashReportDialog = true }
                    )
                    WseListRow(
                        title = "Clear crash report",
                        supporting = "Removes the saved crash_report.txt diagnostic log.",
                        enabled = !r4State.running,
                        trailing = {
                            Text(
                                text = "Clear",
                                style = MaterialTheme.typography.labelLarge,
                                color = colors.primary
                            )
                        },
                        showDivider = false,
                        onClick = onClearCrashReport
                    )
                }
            }

            LauncherSectionHeader(title = "Developer tools")
            LauncherGroupedCard {
                WseListRow(
                    title = if (devToolsVisible) "Hide developer tools" else "Show developer tools",
                    supporting = "Bring-up tests, SAF info, custom driver flags, TSO/PGO, and status log",
                    trailing = {
                        Text(
                            text = if (devToolsVisible) "Hide" else "Show",
                            style = MaterialTheme.typography.labelLarge,
                            color = colors.primary
                        )
                    },
                    showDivider = devToolsVisible,
                    onClick = { showDevTools = !showDevTools }
                )

                if (devToolsVisible) {
                    WseListRow(
                        title = "Tree URI",
                        supporting = treeUri?.toString() ?: "No Warband folder selected."
                    )
                    WseListRow(
                        title = "SAF tree index",
                        supporting = state.scanState.indexStatsText
                    )
                    WseListRow(
                        title = "Verify scripts.txt",
                        supporting = "Checks module script mapping via SafVfs",
                        trailing = {
                            Text(
                                text = "Run",
                                style = MaterialTheme.typography.labelLarge,
                                color = if (verifyEnabled) {
                                    colors.primary
                                } else {
                                    colors.onSurfaceVariant.copy(alpha = 0.45f)
                                }
                            )
                        },
                        enabled = verifyEnabled,
                        showDivider = true,
                        onClick = onVerifyScripts
                    )
                    SwitchListItem(
                        headline = "Allow mock ICD fallback",
                        supporting = "Testing only",
                        checked = state.driverState.allowMockFallback,
                        enabled = true,
                        onCheckedChange = onAllowMockFallbackChange
                    )
                    LauncherSectionDivider()
                    SwitchListItem(
                        headline = "Custom driver zero-copy buffers",
                        supporting = "Experimental AHardwareBuffer bridge for PanVK/Turnip (off uses reliable CPU upload)",
                        checked = state.customDriverSharedBuffers,
                        enabled = true,
                        onCheckedChange = onCustomDriverSharedBuffersChange
                    )
                    LauncherSectionDivider()
                    WseListRow(
                        title = "SDL3 self-test",
                        supporting = "Launch native surface and input check",
                        trailing = {
                            Text(
                                text = "Launch",
                                style = MaterialTheme.typography.labelLarge,
                                color = colors.primary
                            )
                        },
                        showDivider = true,
                        onClick = onLaunchSelfTest
                    )
                    WseListRow(
                        title = "Phone check",
                        supporting = if (r0State.running) {
                            "Checking. This can take a few minutes."
                        } else {
                            "Checks that this phone can run game code translated from your " +
                                "mb_warband_wse2_x64.exe. Takes a few minutes."
                        },
                        trailing = if (r0State.running) {
                            {
                                CircularProgressIndicator(
                                    color = colors.accent,
                                    modifier = Modifier.size(20.dp),
                                    strokeWidth = 2.dp
                                )
                            }
                        } else {
                            null
                        },
                        showDivider = true,
                        enabled = !r0State.running && !r4State.running,
                        onClick = onRunR0Spikes
                    )
                    WseListRow(
                        title = "Game test",
                        supporting = if (r4State.running) {
                            r4State.status.ifEmpty { "Working…" } +
                                "\nYou can leave the app; it keeps working in the background. To stop it, " +
                                "use the notification."
                        } else {
                            "Prepares the game for your phone (needed only once; it takes a few hours " +
                                "and progress is saved), then starts Native briefly to check that it " +
                                "works. Nothing is shown on screen yet."
                        },
                        trailing = if (r4State.running) {
                            {
                                CircularProgressIndicator(
                                    color = colors.accent,
                                    modifier = Modifier.size(20.dp),
                                    strokeWidth = 2.dp
                                )
                            }
                        } else {
                            null
                        },
                        showDivider = true,
                        enabled = !r4State.running && !r0State.running,
                        onClick = onRunR4Startup
                    )
                    val liveVulkanEnabled = !r4State.running && !r0State.running && treeUri != null
                    val activeDriverLabel = state.driverState.options.getOrElse(state.driverState.selectedIndex) {
                        "Vulkan"
                    }
                    WseListRow(
                        title = "Play on screen (test)",
                        supporting = "Starts the game on screen with $activeDriverLabel, using the game files " +
                            "already prepared. Saves a report with the frame rate when you exit.",
                        trailing = {
                            Text(
                                text = "Launch",
                                style = MaterialTheme.typography.labelLarge,
                                color = if (liveVulkanEnabled) {
                                    colors.primary
                                } else {
                                    colors.onSurfaceVariant.copy(alpha = 0.45f)
                                }
                            )
                        },
                        showDivider = true,
                        enabled = liveVulkanEnabled,
                        onClick = onLaunchRecompVulkan
                    )
                    val saveContext = LocalContext.current
                    val saveReport = rememberLauncherForActivityResult(
                        ActivityResultContracts.CreateDocument("text/plain")
                    ) { uri ->
                        if (uri != null) {
                            val runner = R4StartupRunner.get(saveContext.applicationContext as Application)
                            val text = runner.reportToSave()
                            val saved = text != null && runner.saveReport(uri, text)
                            Toast.makeText(
                                saveContext,
                                if (saved) "Report saved." else "There is no report to save yet.",
                                Toast.LENGTH_SHORT
                            ).show()
                        }
                    }
                    WseListRow(
                        title = "Save the game test report",
                        supporting = "Saves what the game test has written so far as a text file, even " +
                            "while it is running.",
                        showDivider = true,
                        onClick = { saveReport.launch("wse2-game-test-report.txt") }
                    )
                    val runnerForPrev = remember(saveContext) {
                        R4StartupRunner.get(saveContext.applicationContext as Application)
                    }
                    val hasPrevCrashReport = remember(r4State) { runnerForPrev.hasPreviousCrashReport() }
                    val savePrevReport = rememberLauncherForActivityResult(
                        ActivityResultContracts.CreateDocument("text/plain")
                    ) { uri ->
                        if (uri != null) {
                            val text = runnerForPrev.previousCrashReportToSave()
                            val saved = text != null && runnerForPrev.saveReport(uri, text)
                            Toast.makeText(
                                saveContext,
                                if (saved) "Previous report saved." else "There is no previous report to save.",
                                Toast.LENGTH_SHORT
                            ).show()
                        }
                    }
                    if (hasPrevCrashReport) {
                        WseListRow(
                            title = "Save the previous report",
                            supporting = "Saves the report from the run before this one as a text file.",
                            showDivider = true,
                            onClick = { savePrevReport.launch("wse2-previous-report.txt") }
                        )
                    }
                    SwitchListItem(
                        headline = "Work from a copy of the game folder",
                        supporting = "Copies the game folder (without other modules) into the app " +
                            "first. Needs that much free space; turning this off deletes the copy.",
                        checked = r4State.useCopy,
                        enabled = !r4State.running,
                        onCheckedChange = onR4UseCopyChange
                    )
                    LauncherSectionDivider()
                    SwitchListItem(
                        headline = "Strict memory ordering (TSO)",
                        supporting = "Slower safe fallback for mods with lock-free thread races (re-translates on change).",
                        checked = state.strictTso,
                        enabled = !r4State.running,
                        onCheckedChange = onStrictTsoChange
                    )
                    LauncherSectionDivider()
                    SwitchListItem(
                        headline = "Record performance profile (PGO)",
                        supporting = "Samples hot game loops during play (${state.pgoProfileSummary}) to speed up future runs.",
                        checked = state.recordPgoProfile,
                        enabled = !r4State.running,
                        onCheckedChange = onRecordPgoProfileChange
                    )
                    if (state.pgoProfileSummary != "None") {
                        LauncherSectionDivider()
                        WseListRow(
                            title = "Clear performance profile",
                            supporting = state.pgoProfileSummary,
                            trailing = {
                                Text(
                                    text = "Clear",
                                    style = MaterialTheme.typography.labelLarge,
                                    color = colors.primary
                                )
                            },
                            showDivider = true,
                            enabled = !r4State.running,
                            onClick = onClearPgoProfile
                        )
                    }
                    LauncherSectionDivider()
                    Text(
                        text = state.statusLog,
                        style = MaterialTheme.typography.bodySmall.copy(
                            fontFamily = FontFamily.Monospace
                        ),
                        color = colors.onSurfaceVariant,
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(16.dp)
                    )
                }
            }

            LauncherSectionHeader(title = "About")
            LauncherGroupedCard {
                WseListRow(
                    title = "Licences",
                    supporting = "SIL Open Font License 1.1 (Cinzel, Manrope)",
                    showDivider = false,
                    onClick = { showLicencesDialog = true }
                )
            }
        }
    }

    r0State.report?.let { report ->
        val context = LocalContext.current
        WseDialog(
            title = "Phone check",
            text = report,
            confirm = WseDialogAction("Copy") {
                val clipboard =
                    context.getSystemService(Context.CLIPBOARD_SERVICE) as? ClipboardManager
                clipboard?.setPrimaryClip(ClipData.newPlainText("WSE2 phone check", report))
                onDismissR0Report()
            },
            dismiss = WseDialogAction("Share") {
                val send = Intent(Intent.ACTION_SEND).apply {
                    type = "text/plain"
                    putExtra(Intent.EXTRA_SUBJECT, "WSE2 phone check")
                    putExtra(Intent.EXTRA_TEXT, report)
                }
                context.startActivity(Intent.createChooser(send, "Share the phone check report"))
                onDismissR0Report()
            },
            onDismissRequest = onDismissR0Report
        )
    }

    r4State.report?.let { report ->
        val context = LocalContext.current
        WseDialog(
            title = "Game test",
            text = report,
            confirm = WseDialogAction("Copy") {
                val clipboard =
                    context.getSystemService(Context.CLIPBOARD_SERVICE) as? ClipboardManager
                clipboard?.setPrimaryClip(ClipData.newPlainText("WSE2 game test", report))
                onDismissR4Report()
            },
            dismiss = WseDialogAction("Share") {
                val send = Intent(Intent.ACTION_SEND).apply {
                    type = "text/plain"
                    putExtra(Intent.EXTRA_SUBJECT, "WSE2 game test")
                    putExtra(Intent.EXTRA_TEXT, report)
                }
                context.startActivity(Intent.createChooser(send, "Share the game test report"))
                onDismissR4Report()
            },
            onDismissRequest = onDismissR4Report
        )
    }

    if (showLicencesDialog) {
        val licenceText = rememberFontLicencesText()
        WseDialog(
            title = "Licences",
            text = licenceText,
            confirm = WseDialogAction("Close") { showLicencesDialog = false },
            onDismissRequest = { showLicencesDialog = false }
        )
    }

    if (showCrashReportDialog && state.crashReportText != null) {
        val context = LocalContext.current
        WseDialog(
            title = "Last crash report",
            text = state.crashReportText,
            confirm = WseDialogAction("Copy") {
                val cm = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                cm.setPrimaryClip(ClipData.newPlainText("WSE2 crash report", state.crashReportText))
                Toast.makeText(context, "Crash report copied.", Toast.LENGTH_SHORT).show()
                showCrashReportDialog = false
            },
            dismiss = WseDialogAction("Close") { showCrashReportDialog = false },
            onDismissRequest = { showCrashReportDialog = false }
        )
    }

    when (confirmClear) {
        ConfirmClear.TRANSLATED_CODE -> WseDialog(
            title = "Free up translated game code?",
            text = "The game's code was translated for your phone, which can take a long time. " +
                "If you free this space, it is translated again the next time you play " +
                "(or imported again from a zip prepared on your PC).",
            confirm = WseDialogAction("Free up") {
                confirmClear = null
                onClearSoCache()
            },
            dismiss = WseDialogAction("Keep") { confirmClear = null },
            onDismissRequest = { confirmClear = null }
        )
        ConfirmClear.SAVE_BACKUPS -> WseDialog(
            title = "Delete all save backups?",
            text = "This permanently deletes the older copies of every save, including saves you " +
                "deleted in the Save Manager. Your current saves stay.",
            confirm = WseDialogAction("Delete backups") {
                confirmClear = null
                onClearSaveBackups()
            },
            dismiss = WseDialogAction("Keep") { confirmClear = null },
            onDismissRequest = { confirmClear = null }
        )
        null -> Unit
    }
}

private enum class ConfirmClear { TRANSLATED_CODE, SAVE_BACKUPS }

@Composable
private fun rememberFontLicencesText(): String {
    val context = LocalContext.current
    return remember(context) {
        val cinzel = readRawLicence(context, R.raw.ofl_cinzel)
        val manrope = readRawLicence(context, R.raw.ofl_manrope)
        buildString {
            append("CINZEL\n\n")
            append(cinzel)
            append("\n\n-----------------------------------------------------------\n\n")
            append("MANROPE\n\n")
            append(manrope)
        }
    }
}

private fun readRawLicence(context: Context, resId: Int): String = try {
    context.resources.openRawResource(resId).bufferedReader(Charsets.UTF_8).use {
        it.readText().trim()
    }
} catch (_: Throwable) {
    "SIL Open Font License 1.1"
}

private fun previewAdvancedState(): AdvancedUiState = AdvancedUiState(
    installFolder = InstallFolderState(displayName = "Mount & Blade - Warband"),
    scanState = ScanState(indexStatsText = "SafVfs index: 4210 files, 184 dirs in 380 ms"),
    selectedModule = ModuleChoice(
        folderName = "Native",
        modulePath = "Modules/Native",
        displayName = "Native",
        hasScriptsTxt = true,
        loadResourceCount = 221,
        loadModResourceCount = 0
    ),
    driverState = DriverUiState(
        selectedIndex = 0,
        statusText = "Uses the device's built-in Vulkan driver.",
        allowMockFallback = false,
        isProbeInProgress = false
    ),
    showFpsOverlay = true,
    statusLog = "OK: 1755 scripts, 131080 statements, 65/65 mapped (cached index, parse 42 ms)"
)

@Preview(name = "Advanced Dark", showBackground = true, widthDp = 412, heightDp = 1120)
@Composable
private fun PreviewAdvancedDark() {
    Wse2Theme(darkTheme = true, dynamicColor = false) {
        Box(modifier = Modifier.wseBackdrop(LocalWseColors.current.accent)) {
            AdvancedScreen(
                state = previewAdvancedState(),
                onVerifyScripts = {},
                onSelectDriver = {},
                onImportDriver = {},
                onAllowMockFallbackChange = {},
                onProbeVulkan = {},
                onShowFpsOverlayChange = {},
                onLaunchSelfTest = {}
            )
        }
    }
}

@Preview(name = "Advanced Light", showBackground = true, widthDp = 412, heightDp = 1120)
@Composable
private fun PreviewAdvancedLight() {
    Wse2Theme(darkTheme = false, dynamicColor = false) {
        Box(modifier = Modifier.wseBackdrop(LocalWseColors.current.accent)) {
            AdvancedScreen(
                state = previewAdvancedState(),
                onVerifyScripts = {},
                onSelectDriver = {},
                onImportDriver = {},
                onAllowMockFallbackChange = {},
                onProbeVulkan = {},
                onShowFpsOverlayChange = {},
                onLaunchSelfTest = {}
            )
        }
    }
}
