package org.wse2.android.launcher

import android.app.Application
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.Message
import android.os.Messenger
import android.os.RemoteException
import android.provider.DocumentsContract
import org.wse2.android.R4ForegroundService
import org.wse2.android.R4StartupService
import java.io.File
import java.util.concurrent.CopyOnWriteArraySet

// Phase 5 R4 startup run on the phone (docs/RECOMP_PLAN.md "R4 design", "Phone run"): optionally
// copies the install into app storage, then runs R4StartupService in its own ":recomp" process,
// which translates the user's exe and lua5164.dll (cached in filesDir/recomp/r4/so), starts the exe
// on the recording D3D9 device and ends after FRAME_LIMIT frames or the first fault. While it runs,
// `status` follows filesDir/recomp/r4/progress.txt and the report's last line; when the process
// ends, the report (filesDir/recomp/r4/report.txt) is shown.
//
// One runner per launcher process (`get`), not per screen: the run goes on in the background, with
// the screen off, or after the app is swiped from Recents, kept alive by R4ForegroundService (its
// notification shows the status and has Stop). Screens subscribe with `addListener`. Every ending
// of the worker (frame limit, fault, crash, being killed) reaches this process as the binding
// dying, so the result notification is always posted here.
//
// Resuming by itself: when Android ends the worker before the run has a RESULT line (in practice
// for memory while translating on every core), the runner binds it again a few seconds later with
// half the threads (kept for later runs); finished batches are cached, so only the batches in
// flight are lost. It gives up after MAX_STALLS deaths in a row that cached no new batch. When the
// launcher process itself was killed mid-run, R4ForegroundService (sticky) or the next app start
// calls `resumeIfInterrupted`, which starts the run again from the saved folder and counts the
// kill like a worker death (halving the threads, and giving up the same way), from preferences:
// otherwise a few huge batches that run the phone out of memory on every core restart from zero
// forever (seen on the phone at 77/80, 2026-11-21).
class R4StartupRunner private constructor(private val app: Application) {
    private val main = Handler(Looper.getMainLooper())
    @Volatile
    private var state = R4UiState()
    private val listeners = CopyOnWriteArraySet<(R4UiState) -> Unit>()
    private var connection: ServiceConnection? = null
    private var finished = false
    @Volatile
    private var stopRequested = false
    private val prefs = app.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
    private var runTree: Uri? = null
    private var runCopy = ""
    private var jobs = 1
    private var soAtBind = 0
    private var stalls = 0
    private var resumes = 0
    @Volatile
    private var copying = false
    @Volatile
    private var r5Preparing = false

    init {
        UserStorage.migrateLegacyUserDirIfNeeded(app.filesDir)
        // "Work from a copy" is remembered across app starts, not only across interrupted runs.
        state = state.copy(useCopy = prefs.getBoolean(PREF_USE_COPY, false))
        refreshAllStorageSizes()
    }

    fun addListener(listener: (R4UiState) -> Unit) {
        listeners.add(listener)
        val now = state
        main.post { listener(now) }
    }

    fun removeListener(listener: (R4UiState) -> Unit) {
        listeners.remove(listener)
    }

    fun isRunning(): Boolean = state.running

    // The launcher process was killed while a run was going on: start it again (cached batches are
    // reused). Called by R4ForegroundService's sticky restart and when the launcher opens.
    fun resumeIfInterrupted() {
        if (state.running || !prefs.getBoolean(PREF_WANTED, false)) return
        val uri = prefs.getString(PREF_TREE, null) ?: return
        update(state.copy(useCopy = prefs.getBoolean(PREF_USE_COPY, state.useCopy)))
        start(Uri.parse(uri), resumed = true)
    }

    // The notification's Stop: ends the worker process (translated batches stay cached).
    fun stop() {
        if (!state.running || stopRequested) return
        stopRequested = true
        // While copying, the copy thread sees the flag and finishes; otherwise end the worker now
        // (or the resume waiting to bind it again).
        if (!copying) main.post { finish(STOPPED) }
    }

    // The result notification was tapped (or RecompGameActivity returned): show the last report,
    // even if this process was restarted since the run (the report is in app storage).
    fun showLastReport(force: Boolean = false) {
        refreshAllStorageSizes()
        if (state.running || (!force && state.report != null)) return
        val file = File(r4Dir, "report.txt")
        if (!file.isFile) return
        update(state.copy(report = reportText(null)))
    }

    fun refreshAllStorageSizes() {
        Thread({
            val installBytes = readInstallCacheBytes()
            val texBytes = readTextureCacheBytes()
            val soBytes = dirSizeBytes(File(r4Dir, "so"))
            val shaderBytes = dirSizeBytes(File(app.cacheDir, "mesa_shader_cache"))
            val backupBytes = SaveManager.backupBytesUsed(app.filesDir)
            synchronized(this) {
                if (state.installCacheBytes != installBytes ||
                    state.textureCacheBytes != texBytes ||
                    state.soCacheBytes != soBytes ||
                    state.shaderCacheBytes != shaderBytes ||
                    state.saveBackupsBytes != backupBytes
                ) {
                    update(
                        state.copy(
                            installCacheBytes = installBytes,
                            textureCacheBytes = texBytes,
                            soCacheBytes = soBytes,
                            shaderCacheBytes = shaderBytes,
                            saveBackupsBytes = backupBytes
                        )
                    )
                }
            }
        }, "wse-r4-storage-sizes").start()
    }

    private fun dirSizeBytes(dir: File): Long {
        if (!dir.isDirectory) return 0L
        return dir.walkTopDown().filter { it.isFile }.sumOf { it.length() }
    }

    fun clearSoCache() {
        if (state.running) return
        Thread({
            File(r4Dir, "so").listFiles()?.forEach { it.deleteRecursively() }
            refreshAllStorageSizes()
        }, "wse-r4-so-clear").start()
    }

    fun clearShaderCache() {
        if (state.running) return
        Thread({
            File(app.cacheDir, "mesa_shader_cache").listFiles()?.forEach { it.deleteRecursively() }
            refreshAllStorageSizes()
        }, "wse-r4-shader-clear").start()
    }

    fun clearSaveBackups() {
        if (state.running) return
        Thread({
            SaveManager.clearAllBackups(app.filesDir)
            refreshAllStorageSizes()
        }, "wse-r4-backups-clear").start()
    }

    /**
     * Checks whether the Warband folder contains an unconfirmed `wse2-prepared*.zip` prepared on
     * a PC (`docs/PLATFORM_PLAN.md` "P14.6"). Returns `null` if none exists or if the user already
     * imported or declined that exact zip (name/size/lastModified stamp).
     */
    fun findPendingPreparedZip(treeUri: Uri): PendingPreparedZip? {
        return try {
            val root = DocumentsContract.getTreeDocumentId(treeUri)
            val children = DocumentsContract.buildChildDocumentsUriUsingTree(treeUri, root)
            val projection = arrayOf(
                DocumentsContract.Document.COLUMN_DOCUMENT_ID,
                DocumentsContract.Document.COLUMN_DISPLAY_NAME,
                DocumentsContract.Document.COLUMN_SIZE,
                DocumentsContract.Document.COLUMN_LAST_MODIFIED
            )
            var pending: PendingPreparedZip? = null
            app.contentResolver.query(children, projection, null, null, null)?.use { c ->
                while (c.moveToNext()) {
                    val name = c.getString(1) ?: continue
                    if (PREPARED_ZIP.matches(name)) {
                        val size = if (c.isNull(2)) -1L else c.getLong(2)
                        val mtime = if (c.isNull(3)) -1L else c.getLong(3)
                        val stamp = "$name/$size/$mtime"
                        val alreadyImported = prefs.getString(PREF_IMPORTED, null) == stamp
                        val alreadyDeclined = prefs.getString(PREF_DECLINED_ZIP, null) == stamp
                        if (!alreadyImported && !alreadyDeclined) {
                            pending = PendingPreparedZip(
                                documentId = c.getString(0),
                                fileName = name,
                                sizeBytes = size,
                                lastModifiedMs = mtime,
                                stamp = stamp
                            )
                            break
                        }
                    }
                }
            }
            pending
        } catch (_: Throwable) {
            null
        }
    }

    fun declinePreparedZip(zip: PendingPreparedZip) {
        prefs.edit().putString(PREF_DECLINED_ZIP, zip.stamp).apply()
        importNote("Skipped ${zip.fileName} (user declined).")
    }

    fun importConfirmedPreparedZip(
        treeUri: Uri,
        zip: PendingPreparedZip,
        onProgress: (String) -> Unit,
        onDone: (PreparedZipImportResult) -> Unit
    ) {
        Thread({
            r4Dir.mkdirs()
            val soDir = File(r4Dir, "so").apply { mkdirs() }
            val texDir = File(r4Dir, "texcache").apply { mkdirs() }
            val doc = DocumentsContract.buildDocumentUriUsingTree(treeUri, zip.documentId)
            val result = try {
                app.contentResolver.openInputStream(doc)?.use { input ->
                    main.post { onProgress("Verifying SHA-256 hashes in ${zip.fileName}…") }
                    PreparedZipVerifier.verifyAndExtract(
                        zipStream = input,
                        soDir = soDir,
                        texDir = texDir,
                        isCancelled = { stopRequested },
                        onProgress = { parts, tex ->
                            main.post {
                                onProgress("Verified & imported $parts code parts, $tex textures…")
                            }
                        }
                    )
                } ?: PreparedZipImportResult(ok = false, error = "Could not open ${zip.fileName}.")
            } catch (t: Throwable) {
                PreparedZipImportResult(ok = false, error = t.message ?: t.javaClass.simpleName)
            }
            if (result.ok) {
                prefs.edit().putString(PREF_IMPORTED, zip.stamp).remove(PREF_DECLINED_ZIP).apply()
                importNote("Verified and imported ${result.partsImported} parts and ${result.texturesImported} textures from ${zip.fileName}.")
                refreshAllStorageSizes()
            } else {
                importNote("PC zip verification failed (${zip.fileName}): ${result.error}")
            }
            main.post { onDone(result) }
        }, "wse-r4-zip-verify").start()
    }

    fun refreshInstallCacheSize() {
        Thread({
            val bytes = readInstallCacheBytes()
            updateInstallCacheBytes(bytes)
        }, "wse-r4-cache-size").start()
    }

    fun clearInstallCache() {
        if (state.running) return
        Thread({
            if (R4Native.loadError == null) {
                try {
                    R4Native.clearInstallCache(workDir.absolutePath)
                } catch (_: Throwable) {
                }
            }
            val bytes = readInstallCacheBytes()
            updateInstallCacheBytes(bytes)
        }, "wse-r4-cache-clear").start()
    }

    private fun readInstallCacheBytes(): Long {
        if (R4Native.loadError != null) return 0L
        return try {
            R4Native.installCacheBytes(workDir.absolutePath).coerceAtLeast(0L)
        } catch (_: Throwable) {
            0L
        }
    }

    @Synchronized
    private fun updateInstallCacheBytes(bytes: Long) {
        if (state.installCacheBytes != bytes) {
            update(state.copy(installCacheBytes = bytes))
        }
    }

    fun refreshTextureCacheSize() {
        Thread({
            val bytes = readTextureCacheBytes()
            updateTextureCacheBytes(bytes)
        }, "wse-r4-texcache-size").start()
    }

    fun clearTextureCache() {
        if (state.running) return
        Thread({
            if (R4Native.loadError == null) {
                try {
                    R4Native.clearTextureCache(workDir.absolutePath)
                } catch (_: Throwable) {
                }
            }
            val bytes = readTextureCacheBytes()
            updateTextureCacheBytes(bytes)
        }, "wse-r4-texcache-clear").start()
    }

    private fun readTextureCacheBytes(): Long {
        if (R4Native.loadError != null) return 0L
        return try {
            R4Native.textureCacheBytes(workDir.absolutePath).coerceAtLeast(0L)
        } catch (_: Throwable) {
            0L
        }
    }

    @Synchronized
    private fun updateTextureCacheBytes(bytes: Long) {
        if (state.textureCacheBytes != bytes) {
            update(state.copy(textureCacheBytes = bytes))
        }
    }

    // Prepares for a live R5a/R8 Vulkan run: rotates the previous report, computes the memory-safe
    // thread count, and calls `onReady` on the main thread with (copyDir, workDir, jobs).
    // PC-prepared zips are now confirmed and SHA-256 verified via `findPendingPreparedZip` +
    // `importConfirmedPreparedZip` before launching (P14.6).
    fun prepareForLiveVulkan(
        treeUri: Uri,
        onReady: (copyDir: String, workDir: String, jobs: Int) -> Unit
    ) {
        if (state.running || r5Preparing) return
        r5Preparing = true
        stopRequested = false
        UserStorage.migrateLegacyUserDirIfNeeded(app.filesDir)
        r4Dir.mkdirs()
        val previous = File(r4Dir, "report.txt")
        if (previous.isFile && previous.length() > 0) {
            val kept = File(r4Dir, PREVIOUS_REPORT)
            kept.delete()
            if (!previous.renameTo(kept)) previous.delete()
        } else {
            previous.delete()
        }
        File(r4Dir, "progress.txt").delete()
        val cores = Runtime.getRuntime().availableProcessors().coerceAtLeast(1)
        val memory = android.app.ActivityManager.MemoryInfo()
        (app.getSystemService(Context.ACTIVITY_SERVICE) as android.app.ActivityManager).getMemoryInfo(memory)
        val fit = (memory.availMem / BYTES_PER_THREAD).toInt().coerceIn(1, cores)
        val resolvedJobs = prefs.getInt(PREF_JOBS, fit).coerceIn(1, fit)
        val resolvedCopy = if (state.useCopy && File(copyDir, "mb_warband_wse2_x64.exe").isFile) {
            copyDir.absolutePath
        } else {
            ""
        }
        main.post {
            r5Preparing = false
            onReady(resolvedCopy, workDir.absolutePath, resolvedJobs)
        }
    }

    // The report to save as a file (Advanced > "Save the game test report"), during a run too:
    // the finished report, else what the running test has written so far under its status.
    // Null when there is no report yet.
    // What the last run's zip import did, for the saved report (one line, replaced each run).
    private fun importNote(text: String) {
        try {
            File(r4Dir, IMPORT_NOTE).writeText(text + "\n")
        } catch (_: Throwable) {
        }
    }

    fun reportToSave(): String? {
        val earlier = try {
            File(r4Dir, PREVIOUS_REPORT).takeIf { it.isFile }?.readText()
        } catch (_: Throwable) {
            null
        }
        val imported = try {
            File(r4Dir, IMPORT_NOTE).takeIf { it.isFile }?.readText()?.trim()
        } catch (_: Throwable) {
            null
        }
        val tail = (if (imported.isNullOrEmpty()) "" else "\n\nPC zip: $imported") +
            (if (earlier.isNullOrBlank()) "" else "\n\nThe attempt before this one:\n$earlier")
        state.report?.let { return it + tail }
        val file = File(r4Dir, "report.txt")
        val text = try {
            file.takeIf { it.isFile }?.readText()
        } catch (_: Throwable) {
            null
        }
        if (text == null) return if (tail.isEmpty()) null else tail.trimStart()
        val headline = if (state.running) {
            "The game test is still running. " + state.status.ifEmpty { "Working…" }
        } else {
            reportText(null).substringBefore("\n\n")
        }
        return "$headline\n\nDetails for the developers:\n$text$tail"
    }

    // Writes `text` to the document the player picked (ACTION_CREATE_DOCUMENT). False on failure.
    fun saveReport(uri: Uri, text: String): Boolean = try {
        app.contentResolver.openOutputStream(uri, "wt")?.use { it.write(text.toByteArray(Charsets.UTF_8)) } != null
    } catch (_: Throwable) {
        false
    }

    // The report of the run before this one (rotated to PREVIOUS_REPORT when a run starts).
    fun hasPreviousCrashReport(): Boolean {
        val f = File(r4Dir, PREVIOUS_REPORT)
        return f.isFile && f.length() > 0
    }

    fun previousCrashReportToSave(): String? {
        val f = File(r4Dir, PREVIOUS_REPORT)
        return try {
            f.takeIf { it.isFile && it.length() > 0 }?.readText()
        } catch (_: Throwable) {
            null
        }
    }

    private val workDir: File get() = File(app.filesDir, "recomp")
    private val r4Dir: File get() = File(workDir, "r4")
    private val copyDir: File get() = File(r4Dir, "install")

    private val poll = object : Runnable {
        override fun run() {
            if (!state.running) return
            val status = statusLine()
            update(state.copy(status = status))
            R4ForegroundService.showProgress(app, status)
            main.postDelayed(this, 1000L)
        }
    }

    // Turning the copy off removes it: it is a whole game folder (1.5-4 GB) in app storage.
    fun setUseCopy(useCopy: Boolean) {
        if (state.running) return
        update(state.copy(useCopy = useCopy))
        prefs.edit().putBoolean(PREF_USE_COPY, useCopy).apply()
        if (!useCopy && copyDir.exists()) {
            Thread({
                val freed = copyDir.walkBottomUp().filter { it.isFile }.sumOf { it.length() }
                val removed = copyDir.deleteRecursively()
                main.post {
                    if (!state.running) {
                        update(state.copy(status = if (removed) {
                            "Removed the copied game files (${freed / (1024 * 1024)} MB freed)."
                        } else {
                            "Some copied game files could not be removed. Clearing the app's storage removes them."
                        }))
                    }
                }
            }, "wse-r4-remove-copy").start()
        }
    }

    fun dismissReport() = update(state.copy(report = null))

    fun start(treeUri: Uri?) = start(treeUri, resumed = false)

    private fun start(treeUri: Uri?, resumed: Boolean) {
        if (state.running) return
        if (treeUri == null) {
            update(state.copy(report = "Choose your Warband folder first."))
            return
        }
        R4Native.loadError?.let {
            update(state.copy(report = "The game test is not available: part of the app did not load.\n\n(libwse_r0.so: $it)"))
            return
        }
        r4Dir.mkdirs()
        // The last attempt's report is kept (one back) so it can still be saved: a resume after
        // an app update or a kill starts a new report before the player could save the old one.
        val previous = File(r4Dir, "report.txt")
        if (previous.isFile && previous.length() > 0) {
            val kept = File(r4Dir, PREVIOUS_REPORT)
            kept.delete()
            if (!previous.renameTo(kept)) previous.delete()
        } else {
            previous.delete()
        }
        File(r4Dir, "progress.txt").delete()
        finished = false
        stopRequested = false
        runTree = treeUri
        if (resumed) {
            // Android killed the whole app mid-run (the launcher too, not only the worker), which
            // `workerEnded` never sees: count it the same way, from what the last bind recorded.
            val before = prefs.getInt(PREF_SO_AT_BIND, -1)
            stalls = if (before < 0 || cachedBatches() > before) 0 else prefs.getInt(PREF_STALLS, 0) + 1
            resumes = prefs.getInt(PREF_RESUMES, 0) + 1
        } else {
            stalls = 0
            resumes = 0
        }
        val cores = Runtime.getRuntime().availableProcessors().coerceAtLeast(1)
        // One thread per core, but no more than the free memory holds: on the PC (tools/r4_run, 2
        // threads, 2026-11-20) translating this exe took ~1.8 GB, ~0.9 GB a thread. More threads
        // than memory make Android swap until the run crawls, or kill the worker.
        val memory = android.app.ActivityManager.MemoryInfo()
        (app.getSystemService(Context.ACTIVITY_SERVICE) as android.app.ActivityManager).getMemoryInfo(memory)
        val fit = (memory.availMem / BYTES_PER_THREAD).toInt().coerceIn(1, cores)
        jobs = prefs.getInt(PREF_JOBS, fit).coerceIn(1, fit)
        if (resumed) {
            // As in `workerEnded`: half the threads, down to 1 once a death brought no new batch.
            val floor = if (stalls > 0) 1 else 2
            if (jobs > floor) jobs = (jobs / 2).coerceAtLeast(floor)
        }
        prefs.edit().putBoolean(PREF_WANTED, true).putString(PREF_TREE, treeUri.toString())
            .putBoolean(PREF_USE_COPY, state.useCopy).putInt(PREF_STALLS, stalls)
            .putInt(PREF_RESUMES, resumes).apply()
        if (resumed) prefs.edit().putInt(PREF_JOBS, jobs).apply()
        if (stalls > MAX_STALLS) {
            finish(
                "The phone kept running out of memory and closing the app, so the game test stopped " +
                    "after $stalls tries in a row. Everything finished so far is saved: tap the test " +
                    "again to continue."
            )
            return
        }
        R4ForegroundService.clearResult(app)
        update(state.copy(running = true, status = "Getting ready…", report = null))
        try {
            R4ForegroundService.start(app)
        } catch (_: Throwable) {
            // A sticky restart is already the foreground service; elsewhere the run still goes on.
        }
        val useCopy = state.useCopy
        copying = true
        Thread({
            // A zip prepared on the PC (tools/prepare_for_phone.ps1) in the Warband folder is
            // imported first; failing to read it only means translating here as before.
            val imported = try {
                importPrepared(treeUri)
            } catch (t: Throwable) {
                importNote("Importing the zip prepared on a PC failed: ${t.message ?: t.javaClass.simpleName}")
                0
            }
            val error = if (!useCopy || stopRequested) {
                null
            } else {
                try {
                    copyInstall(treeUri)
                    null
                } catch (t: Throwable) {
                    t.message ?: t.javaClass.simpleName
                }
            }
            main.post {
                copying = false
                if (imported > 0) {
                    update(state.copy(status = "Imported $imported prepared parts from your PC. Starting…"))
                }
                if (stopRequested) {
                    finish(STOPPED)
                } else if (error != null) {
                    finish("Could not copy the game folder: $error")
                } else {
                    bind(treeUri, if (useCopy) copyDir.absolutePath else "")
                }
            }
        }, "wse-r4-copy").start()
    }

    // Background R4 game test: only imports a PC zip if it passes full SHA-256 `manifest.json`
    // verification (P14.6). Unverified zips are refused and logged cleanly.
    private fun importPrepared(treeUri: Uri): Int {
        val pending = findPendingPreparedZip(treeUri) ?: run {
            importNote("No new wse2-prepared*.zip to import.")
            return 0
        }
        val soDir = File(r4Dir, "so").apply { mkdirs() }
        val texDir = File(r4Dir, "texcache").apply { mkdirs() }
        update(state.copy(status = "Verifying SHA-256 hashes in ${pending.fileName}…"))
        val doc = DocumentsContract.buildDocumentUriUsingTree(treeUri, pending.documentId)
        val res = app.contentResolver.openInputStream(doc)?.use { input ->
            PreparedZipVerifier.verifyAndExtract(
                zipStream = input,
                soDir = soDir,
                texDir = texDir,
                isCancelled = { stopRequested },
                onProgress = { parts, _ ->
                    if (parts > 0) update(state.copy(status = "Verified & imported $parts parts from ${pending.fileName}…"))
                }
            )
        } ?: PreparedZipImportResult(ok = false, error = "${pending.fileName} could not be opened.")
        if (!res.ok) {
            importNote("Refused ${pending.fileName}: ${res.error}")
            return 0
        }
        prefs.edit().putString(PREF_IMPORTED, pending.stamp).remove(PREF_DECLINED_ZIP).apply()
        importNote("Verified and imported ${res.partsImported} parts and ${res.texturesImported} converted textures from ${pending.fileName}.")
        return res.partsImported
    }

    private fun bind(treeUri: Uri, copy: String) {
        runCopy = copy
        soAtBind = cachedBatches()
        prefs.edit().putInt(PREF_SO_AT_BIND, soAtBind).apply()
        val args = Bundle().apply {
            putString(R4StartupService.KEY_TREE_URI, treeUri.toString())
            putString(R4StartupService.KEY_COPY_DIR, copy)
            putString(R4StartupService.KEY_WORK_DIR, workDir.absolutePath)
            putString(R4StartupService.KEY_MODULE, MODULE)
            // Every core the free memory holds (the user's choice, 2026-11-21: the phone does
            // nothing else meanwhile; see `start`), halved each time Android ends the worker
            // (`workerEnded`). The thread count is not part of the batch cache key.
            putInt(R4StartupService.KEY_JOBS, jobs)
            putLong(R4StartupService.KEY_FRAMES, FRAME_LIMIT)
        }
        val replies = Messenger(object : Handler(Looper.getMainLooper()) {
            override fun handleMessage(msg: Message) {
                if (msg.what == R4StartupService.MSG_RESULT) {
                    val code = msg.data.getInt(R4StartupService.KEY_RESULT)
                    finish("The game test could not start (error code $code).")
                } else {
                    super.handleMessage(msg)
                }
            }
        })
        val conn = object : ServiceConnection {
            override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
                val request = Message.obtain(null, R4StartupService.MSG_RUN).apply {
                    data = args
                    replyTo = replies
                }
                try {
                    Messenger(service).send(request)
                } catch (_: RemoteException) {
                    finish("The game test could not start: its background process closed right away.")
                }
            }

            // The run ends its process: frame limit, fault, or a crash in translated code.
            override fun onServiceDisconnected(name: ComponentName?) = workerEnded()

            override fun onBindingDied(name: ComponentName?) = workerEnded()
        }
        connection = conn
        // BIND_IMPORTANT | BIND_ABOVE_CLIENT: the worker gets the launcher's priority (top app while
        // it is on screen, foreground service otherwise) and is the last of the two to be killed.
        val flags = Context.BIND_AUTO_CREATE or Context.BIND_IMPORTANT or Context.BIND_ABOVE_CLIENT
        if (!app.bindService(Intent(app, R4StartupService::class.java), conn, flags)) {
            finish("The game test could not start its background process.")
            return
        }
        main.post(poll)
    }

    // The worker process ended. With a RESULT line the run is over; without one Android ended it
    // (or it crashed): bind again with half the threads unless it keeps dying without progress.
    private fun workerEnded() {
        if (finished || connection == null) return
        val tree = runTree
        if (stopRequested || tree == null || resultLine() != null) {
            finish(null)
            return
        }
        val cached = cachedBatches()
        stalls = if (cached > soAtBind) 0 else stalls + 1
        if (stalls > MAX_STALLS) {
            finish(
                "The phone kept running out of memory, so the game test stopped after $stalls tries " +
                    "in a row. Everything finished so far is saved: tap the test again to continue."
            )
            return
        }
        val floor = if (stalls > 0) 1 else 2
        if (jobs > floor) jobs = (jobs / 2).coerceAtLeast(floor)
        ++resumes
        prefs.edit().putInt(PREF_JOBS, jobs).putInt(PREF_STALLS, stalls).putInt(PREF_RESUMES, resumes).apply()
        unbindQuietly()
        main.removeCallbacks(poll)
        val status = "The phone ran low on memory and paused the work. Continuing in a moment with " +
            "fewer processor cores; nothing finished is lost."
        update(state.copy(status = status))
        R4ForegroundService.showProgress(app, status)
        main.postDelayed({ if (!finished && !stopRequested) bind(tree, runCopy) }, RESUME_DELAY_MS)
    }

    private fun unbindQuietly() {
        connection?.let {
            try {
                app.unbindService(it)
            } catch (_: Throwable) {
            }
        }
        connection = null
    }

    private fun cachedBatches(): Int =
        File(r4Dir, "so").listFiles { f -> f.name.endsWith(".so") }?.size ?: 0

    private fun resultLine(): String? = try {
        File(r4Dir, "report.txt").readLines().lastOrNull { it.startsWith("RESULT ") }
    } catch (_: Throwable) {
        null
    }

    private fun finish(error: String?) {
        if (finished) return
        finished = true
        prefs.edit().putBoolean(PREF_WANTED, false).remove(PREF_SO_AT_BIND).remove(PREF_STALLS)
            .remove(PREF_RESUMES).apply()
        unbindQuietly()
        main.removeCallbacks(poll)
        val report = reportText(error)
        update(state.copy(running = false, status = "", report = report))
        R4ForegroundService.finish(app, report.substringBefore("\n\n"))
        refreshAllStorageSizes()
    }

    // The headline (the given error, or a plain-language diagnostic from the report) and the report.
    private fun reportText(error: String?): String {
        val text = try {
            File(r4Dir, "report.txt").readText()
        } catch (_: Throwable) {
            ""
        }
        val result = text.lineSequence().lastOrNull { it.startsWith("RESULT ") }?.removePrefix("RESULT ")
        val lower = text.lowercase()
        val headline = when {
            error != null -> error
            result == null -> "The game ended unexpectedly (Android may have closed it for memory). " +
                "Everything translated so far is saved."
            result.startsWith("pass-r5a") -> "Game session finished cleanly ($result)."
            result.startsWith("stopped-r5a") -> "Game closed before the first frame appeared."
            result.startsWith("vulkan-error") -> "Your phone's Vulkan graphics driver could not start. " +
                "Try switching back to System Vulkan under Advanced -> GPU Driver."
            lower.contains("pdb") && (lower.contains("missing") || lower.contains("mismatch") || lower.contains("not found")) ->
                "Missing or mismatched mb_warband_wse2_x64.pdb. Copy the matching .pdb file from your " +
                    "WSE2 PC installation into the Warband folder and try again."
            lower.contains("missing_import") -> {
                val impLine = text.lineSequence().firstOrNull { it.contains("missing_import", ignoreCase = true) }?.trim()
                "This version of WSE2 called a Windows function that is not yet supported" +
                    (if (impLine != null) " ($impLine)." else ".") + " Please share this report."
            }
            result.startsWith("pass") -> "The game test passed."
            else -> {
                val faultLine = text.lineSequence().firstOrNull {
                    it.contains("FAULT", ignoreCase = true) || it.contains("SIGSEGV", ignoreCase = true)
                }?.trim()
                "The game stopped with an error" +
                    (if (faultLine != null) " ($faultLine)." else ".") + " Please share this report."
            }
        }
        return "$headline\n\nDetails for the developers:\n$text"
    }

    private fun statusLine(): String {
        val progress = try {
            File(r4Dir, "progress.txt").readText().trim().split(' ')
        } catch (_: Throwable) {
            emptyList()
        }
        // Plain words for the player (2026-11-21, the user's request): the report's raw lines stay in
        // the final report, under "Details for the developers".
        // progress.txt counts this attempt's batches; the report's "<tag>: ...; N batches (...), T to
        // translate" line adds the cached ones, so the status counts the whole image.
        val translating = if (progress.size == 3 && progress[1] != progress[2]) {
            val done = progress[1].toIntOrNull() ?: 0
            val todo = progress[2].toIntOrNull() ?: 0
            val summary = try {
                File(r4Dir, "report.txt").readLines()
                    .lastOrNull { it.startsWith(progress[0] + ": ") && it.contains(" to translate") }
            } catch (_: Throwable) {
                null
            }
            val total = summary?.let { BATCHES.find(it)?.groupValues?.get(1)?.toIntOrNull() } ?: todo
            val kept = (total - todo).coerceAtLeast(0)
            val what = if (progress[0] == "exe") "the game" else "the game's script engine"
            "Preparing $what for your phone: ${kept + done} of $total parts done. This is needed " +
                "only once; finished parts are saved."
        } else if (progress.size == 3) {
            "Starting a short test of the game (nothing is shown on screen)…"
        } else {
            "Getting ready…"
        }
        val again = when (resumes) {
            0 -> ""
            1 -> "\nRestarted once after the phone ran low on memory."
            else -> "\nRestarted $resumes times after the phone ran low on memory."
        }
        return translating + again
    }

    // The copy thread and the main thread both update `state`.
    @Synchronized
    private fun update(next: R4UiState) {
        state = next
        main.post { for (l in listeners) l(next) }
    }

    // Everything under the folder except the other modules in Modules/: root files, CommonRes,
    // Data, Textures, languages, ... and Modules/<MODULE>. A file already copied with the same size
    // is kept, so a second run copies only what changed.
    private fun copyInstall(treeUri: Uri) {
        copyDir.mkdirs()
        var files = 0
        var bytes = 0L
        fun walk(parentId: String, dest: File, depth: Int) {
            val children = DocumentsContract.buildChildDocumentsUriUsingTree(treeUri, parentId)
            val projection = arrayOf(
                DocumentsContract.Document.COLUMN_DOCUMENT_ID,
                DocumentsContract.Document.COLUMN_DISPLAY_NAME,
                DocumentsContract.Document.COLUMN_MIME_TYPE,
                DocumentsContract.Document.COLUMN_SIZE
            )
            val entries = ArrayList<Array<Any?>>()
            app.contentResolver.query(children, projection, null, null, null)?.use { c ->
                while (c.moveToNext()) {
                    entries.add(arrayOf(c.getString(0), c.getString(1), c.getString(2), if (c.isNull(3)) -1L else c.getLong(3)))
                }
            }
            for (e in entries) {
                if (stopRequested) return
                val id = e[0] as String? ?: continue
                val name = e[1] as String? ?: continue
                val isDir = e[2] == DocumentsContract.Document.MIME_TYPE_DIR
                val size = e[3] as Long
                if (isDir) {
                    // Inside Modules/, only the module the run starts.
                    if (depth == 1 && dest.name.equals("Modules", ignoreCase = true) && !name.equals(MODULE, ignoreCase = true)) {
                        continue
                    }
                    val sub = File(dest, name)
                    sub.mkdirs()
                    walk(id, sub, depth + 1)
                    continue
                }
                val out = File(dest, name)
                if (size >= 0 && out.isFile && out.length() == size) continue
                // Stop before the storage runs full (the phone needs room for itself and for the
                // translated game too) rather than fail halfway with a cryptic error.
                if (size > 0 && copyDir.usableSpace - size < COPY_RESERVE_BYTES) {
                    throw IllegalStateException(
                        "there is not enough free space on the phone. Free up some space, or turn off \"Work from a copy of the game folder\""
                    )
                }
                val doc = DocumentsContract.buildDocumentUriUsingTree(treeUri, id)
                app.contentResolver.openInputStream(doc)?.use { input ->
                    out.outputStream().use { output -> bytes += input.copyTo(output, 1 shl 16) }
                } ?: throw IllegalStateException("cannot open $name")
                ++files
                if (files % 50 == 0) {
                    update(state.copy(status = "Copying the game folder: $files files, ${bytes / (1024 * 1024)} MB"))
                }
            }
        }
        walk(DocumentsContract.getTreeDocumentId(treeUri), copyDir, 0)
        update(state.copy(status = "Copied the game folder ($files files, ${bytes / (1024 * 1024)} MB). Starting…"))
    }

    companion object {
        const val MODULE = "Native"
        const val FRAME_LIMIT = 1000L
        private const val STOPPED = "You stopped the game test. Everything finished so far is saved, and the next " +
            "run continues from there."
        private const val PREFS = "r4_startup"
        private const val PREF_WANTED = "running"
        private const val PREF_TREE = "tree_uri"
        private const val PREF_USE_COPY = "use_copy"
        private const val COPY_RESERVE_BYTES = 1024L * 1024L * 1024L
        private const val PREF_JOBS = "jobs"
        private const val PREVIOUS_REPORT = "report.previous.txt"
        private const val IMPORT_NOTE = "import.txt"
        private const val PREF_IMPORTED = "imported_zip"  // the last PC zip imported (name/size/time)
        private const val PREF_DECLINED_ZIP = "declined_zip"
        private val PREPARED_ZIP = Regex("""(?i)wse2-prepared.*\.zip""")
        private val PREPARED_PART = Regex("""(exe|lua5164)_w\d+\.(so|key)""")
        private val PREPARED_TEXTURE = Regex("""[0-9a-f]{16}\.etc2""")
        // Kept across launcher restarts, so a run Android keeps killing whole still slows down and
        // gives up: the cached batch count when the worker was last bound, the deaths in a row
        // without a new batch, and the resumes so far.
        private const val PREF_SO_AT_BIND = "so_at_bind"
        private const val PREF_STALLS = "stalls"
        private const val PREF_RESUMES = "resumes"
        private const val MAX_STALLS = 3
        private const val RESUME_DELAY_MS = 3000L
        private const val BYTES_PER_THREAD = 1200L * 1024 * 1024
        private val BATCHES = Regex("""(\d+) batches \(""")

        @Volatile
        private var instance: R4StartupRunner? = null

        fun get(app: Application): R4StartupRunner =
            instance ?: synchronized(this) { instance ?: R4StartupRunner(app).also { instance = it } }
    }
}
