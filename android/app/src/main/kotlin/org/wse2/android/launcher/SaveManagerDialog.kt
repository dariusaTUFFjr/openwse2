package org.wse2.android.launcher

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import org.wse2.android.launcher.theme.LocalWseColors
import org.wse2.android.launcher.ui.WseContentDialog
import org.wse2.android.launcher.ui.WseDialogAction
import org.wse2.android.launcher.ui.WseListRow
import org.wse2.android.launcher.ui.WseTextButton
import java.text.DateFormat
import java.util.Date

@Composable
fun SaveManagerDialog(
    moduleName: String,
    saves: List<SaveSlotEntry>,
    mirrorFolderLabel: String?,
    statusMessage: String?,
    onImportSave: () -> Unit,
    onExportSave: (SaveSlotEntry) -> Unit,
    onRestoreBackup: (SaveSlotEntry, SaveBackupEntry) -> Unit,
    onDeleteSave: (SaveSlotEntry) -> Unit,
    onPickMirrorFolder: () -> Unit,
    onClearMirrorFolder: () -> Unit,
    onSyncMirrorNow: () -> Unit,
    onDismiss: () -> Unit
) {
    val colors = LocalWseColors.current
    val dateFmt = DateFormat.getDateTimeInstance(DateFormat.SHORT, DateFormat.SHORT)

    WseContentDialog(
        title = "Save Manager ($moduleName)",
        confirm = WseDialogAction("Import .sav", onClick = onImportSave),
        dismiss = WseDialogAction("Close", onClick = onDismiss),
        onDismissRequest = onDismiss,
        content = {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(max = 420.dp)
                    .verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                if (!statusMessage.isNullOrBlank()) {
                    Text(
                        text = statusMessage,
                        style = MaterialTheme.typography.bodySmall,
                        color = colors.accent
                    )
                }

                Text(
                    text = "Auto-sync SAF save mirror",
                    style = MaterialTheme.typography.labelLarge,
                    color = colors.primary
                )
                Text(
                    text = if (mirrorFolderLabel.isNullOrBlank()) {
                        "Not configured. Pick a folder (Syncthing, Drive, SD card) to mirror saves automatically."
                    } else {
                        "Mirroring to: $mirrorFolderLabel"
                    },
                    style = MaterialTheme.typography.bodySmall,
                    color = colors.onSurfaceVariant
                )
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    WseTextButton(
                        text = if (mirrorFolderLabel.isNullOrBlank()) "Pick mirror" else "Change",
                        onClick = onPickMirrorFolder
                    )
                    if (!mirrorFolderLabel.isNullOrBlank()) {
                        WseTextButton(text = "Sync now", onClick = onSyncMirrorNow)
                        WseTextButton(text = "Turn off", onClick = onClearMirrorFolder)
                    }
                }

                LauncherSectionDivider()

                if (saves.isEmpty()) {
                    Text(
                        text = "No saved games found for $moduleName yet. Saves written in-game appear here " +
                            "with automatic rolling backups.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = colors.onSurfaceVariant,
                        modifier = Modifier.padding(vertical = 8.dp)
                    )
                } else {
                    for (slot in saves) {
                        val headline = if (slot.header.valid) {
                            val who = slot.header.playerName.ifBlank { slot.header.saveName.ifBlank { slot.fileName } }
                            "$who · Day ${slot.header.day} (Lv ${slot.header.playerLevel})"
                        } else {
                            slot.fileName
                        }
                        val sub = "${slot.fileName} · ${slot.sizeBytes / 1024} KB · " +
                            dateFmt.format(Date(slot.lastModifiedMs)) +
                            if (slot.backups.isNotEmpty()) " · ${slot.backups.size} backup(s)" else ""
                        WseListRow(
                            title = headline,
                            supporting = sub,
                            showDivider = false
                        )
                        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                            WseTextButton(text = "Export", onClick = { onExportSave(slot) })
                            slot.backups.firstOrNull()?.let { latest ->
                                WseTextButton(
                                    text = "Restore backup",
                                    onClick = { onRestoreBackup(slot, latest) }
                                )
                            }
                            WseTextButton(text = "Delete", onClick = { onDeleteSave(slot) })
                        }
                        LauncherSectionDivider()
                    }
                }
            }
        }
    )
}
