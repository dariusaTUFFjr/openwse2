package org.wse2.android

import android.app.Activity
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Process
import android.util.TypedValue
import android.view.Gravity
import android.view.Surface
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.View
import android.view.WindowInsets
import android.view.WindowInsetsController
import android.view.WindowManager
import android.widget.FrameLayout
import android.widget.TextView
import android.window.OnBackInvokedDispatcher
import org.wse2.android.launcher.R4Native
import java.util.concurrent.atomic.AtomicBoolean

// Phase 5 R5a live Vulkan presentation window (`docs/RECOMP_PLAN.md` "R5 — Graphics").
// Runs in its own isolated `:recomp` process so translated fixed-address game mappings and
// experimental GPU drivers (System Vulkan / PanVK / Turnip) never affect the launcher process.
class RecompGameActivity : Activity(), SurfaceHolder.Callback {

    private lateinit var surfaceView: SurfaceView
    private lateinit var hudPill: TextView
    private lateinit var exitButton: TextView

    private val started = AtomicBoolean(false)
    private val finishing = AtomicBoolean(false)
    private var hudDimmed = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        window.decorView.setBackgroundColor(Color.BLACK)

        val root = FrameLayout(this).apply {
            setBackgroundColor(Color.BLACK)
        }

        surfaceView = SurfaceView(this).apply {
            holder.addCallback(this@RecompGameActivity)
        }
        root.addView(
            surfaceView,
            FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
            )
        )

        val padH = dp(12)
        val padV = dp(6)
        val margin = dp(12)

        hudPill = TextView(this).apply {
            text = "Starting the game…"
            setTextColor(Color.WHITE)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 12f)
            typeface = Typeface.MONOSPACE
            setPadding(padH, padV, padH, padV)
            background = pillDrawable(0xB314161A.toInt())
            setOnClickListener {
                hudDimmed = !hudDimmed
                alpha = if (hudDimmed) 0.25f else 1.0f
            }
        }
        root.addView(
            hudPill,
            FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.WRAP_CONTENT,
                FrameLayout.LayoutParams.WRAP_CONTENT,
                Gravity.TOP or Gravity.START
            ).apply {
                setMargins(margin, margin, margin, margin)
            }
        )

        exitButton = TextView(this).apply {
            text = "Exit and save report"
            setTextColor(Color.WHITE)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 12f)
            typeface = Typeface.DEFAULT_BOLD
            setPadding(padH, padV, padH, padV)
            background = pillDrawable(0xCC7A2222.toInt())
            setOnClickListener { exitAndFinish() }
        }
        root.addView(
            exitButton,
            FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.WRAP_CONTENT,
                FrameLayout.LayoutParams.WRAP_CONTENT,
                Gravity.TOP or Gravity.END
            ).apply {
                setMargins(margin, margin, margin, margin)
            }
        )

        setContentView(root)
        applyImmersiveFullscreen()

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            onBackInvokedDispatcher.registerOnBackInvokedCallback(
                OnBackInvokedDispatcher.PRIORITY_DEFAULT
            ) {
                exitAndFinish()
            }
        }
    }

    override fun onPause() {
        super.onPause()
        if (started.get() && R4Native.loadError == null) {
            R4Native.nativeSetPaused(true)
        }
    }

    override fun onResume() {
        super.onResume()
        applyImmersiveFullscreen()
        if (started.get() && R4Native.loadError == null) {
            R4Native.nativeSetPaused(false)
        }
    }

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        if (hasFocus) applyImmersiveFullscreen()
    }

    @Deprecated("Deprecated in Java")
    override fun onBackPressed() {
        exitAndFinish()
    }

    override fun surfaceCreated(holder: SurfaceHolder) {
        val surface = holder.surface
        if (started.compareAndSet(false, true)) {
            startGameThread(surface)
        } else if (R4Native.loadError == null) {
            R4Native.nativeSetSurface(surface)
        }
    }

    override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
        if (started.get() && R4Native.loadError == null) {
            R4Native.nativeSetSurface(holder.surface)
        }
    }

    override fun surfaceDestroyed(holder: SurfaceHolder) {
        if (started.get() && R4Native.loadError == null) {
            R4Native.nativeSetSurface(null)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        if (R4Native.loadError == null) {
            try {
                R4Native.nativeFinishReport()
            } catch (_: Throwable) {
            }
        }
        // End the isolated `:recomp` process so the next run starts with clean fixed-address maps.
        Process.killProcess(Process.myPid())
    }

    private fun startGameThread(initialSurface: Surface) {
        val loadErr = R4Native.loadError
        if (loadErr != null) {
            updateHud("libwse_r0.so load failed: $loadErr")
            exitAndFinish()
            return
        }
        val treeUri = intent.getStringExtra(EXTRA_TREE_URI).orEmpty()
        val copyDir = intent.getStringExtra(EXTRA_COPY_DIR).orEmpty()
        val workDir = intent.getStringExtra(EXTRA_WORK_DIR).orEmpty()
        val module = intent.getStringExtra(EXTRA_MODULE).orEmpty().ifEmpty { "Native" }
        val jobs = intent.getIntExtra(EXTRA_JOBS, 1).coerceAtLeast(1)
        val driverMode = intent.getStringExtra(EXTRA_DRIVER_MODE).orEmpty().ifEmpty { "system" }
        val customIcdPath = intent.getStringExtra(EXTRA_CUSTOM_ICD_PATH).orEmpty()
        val nativeLibDir = intent.getStringExtra(EXTRA_NATIVE_LIB_DIR).orEmpty()
        val shaderCacheDir = intent.getStringExtra(EXTRA_SHADER_CACHE_DIR).orEmpty()
        val maxTextureSize = intent.getIntExtra(EXTRA_MAX_TEXTURE_SIZE, 0).coerceAtLeast(0)
        val renderScale = intent.getFloatExtra(EXTRA_RENDER_SCALE, 1.0f).coerceIn(0.5f, 1.0f)
        val autoResolution = intent.getBooleanExtra(EXTRA_AUTO_RESOLUTION, true)
        val sharpness = intent.getFloatExtra(EXTRA_SHARPNESS, 0.2f).coerceAtLeast(0.0f)
        val fxaa = intent.getBooleanExtra(EXTRA_FXAA, false)

        val hudListener = object : R4Native.HudListener {
            override fun onHudUpdate(text: String) {
                updateHud(text)
            }
        }

        Thread({
            raisePriority()
            val bridge = if (copyDir.isEmpty() && treeUri.isNotEmpty()) {
                WseSafBridge(applicationContext, Uri.parse(treeUri))
            } else {
                null
            }
            val rc = try {
                R4Native.nativeRunVulkan(
                    bridge,
                    copyDir,
                    workDir,
                    module,
                    jobs,
                    0L, // continuous live run until user taps Exit / Back
                    initialSurface,
                    driverMode,
                    customIcdPath,
                    nativeLibDir,
                    shaderCacheDir,
                    maxTextureSize,
                    renderScale,
                    autoResolution,
                    sharpness,
                    fxaa,
                    hudListener
                )
            } catch (t: Throwable) {
                updateHud("The game stopped with an error: ${t.message ?: t.javaClass.simpleName}")
                1
            }
            // Returns only on setup/Vulkan failure: return cleanly so the launcher shows report.txt.
            runOnUiThread {
                updateHud("The game stopped (code $rc)")
                exitAndFinish()
            }
        }, "wse-r5a-vulkan").start()
    }

    private fun exitAndFinish() {
        if (!finishing.compareAndSet(false, true)) return
        if (R4Native.loadError == null) {
            try {
                R4Native.nativeFinishReport()
            } catch (_: Throwable) {
            }
        }
        setResult(RESULT_OK)
        finish()
    }

    private fun updateHud(text: String) {
        runOnUiThread {
            if (!isFinishing && !isDestroyed) {
                hudPill.text = text
            }
        }
    }

    private fun raisePriority() {
        for (p in intArrayOf(
            Process.THREAD_PRIORITY_URGENT_DISPLAY,
            Process.THREAD_PRIORITY_DISPLAY,
            Process.THREAD_PRIORITY_FOREGROUND
        )) {
            try {
                Process.setThreadPriority(p)
                return
            } catch (_: Throwable) {
            }
        }
    }

    @Suppress("DEPRECATION")
    private fun applyImmersiveFullscreen() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            window.setDecorFitsSystemWindows(false)
            window.insetsController?.let { controller ->
                controller.hide(WindowInsets.Type.statusBars() or WindowInsets.Type.navigationBars())
                controller.systemBarsBehavior =
                    WindowInsetsController.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
            }
        } else {
            window.decorView.systemUiVisibility = (
                View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY or
                    View.SYSTEM_UI_FLAG_FULLSCREEN or
                    View.SYSTEM_UI_FLAG_HIDE_NAVIGATION or
                    View.SYSTEM_UI_FLAG_LAYOUT_STABLE or
                    View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION or
                    View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                )
        }
    }

    private fun dp(value: Int): Int =
        TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, value.toFloat(), resources.displayMetrics).toInt()

    private fun pillDrawable(colorArgb: Int): GradientDrawable = GradientDrawable().apply {
        shape = GradientDrawable.RECTANGLE
        cornerRadius = dp(14).toFloat()
        setColor(colorArgb)
    }

    companion object {
        const val EXTRA_TREE_URI = "wse2.r5a.TREE_URI"
        const val EXTRA_COPY_DIR = "wse2.r5a.COPY_DIR"
        const val EXTRA_WORK_DIR = "wse2.r5a.WORK_DIR"
        const val EXTRA_MODULE = "wse2.r5a.MODULE"
        const val EXTRA_JOBS = "wse2.r5a.JOBS"
        const val EXTRA_DRIVER_MODE = "wse2.r5a.DRIVER_MODE"
        const val EXTRA_CUSTOM_ICD_PATH = "wse2.r5a.CUSTOM_ICD_PATH"
        const val EXTRA_NATIVE_LIB_DIR = "wse2.r5a.NATIVE_LIB_DIR"
        const val EXTRA_SHADER_CACHE_DIR = "wse2.r5a.SHADER_CACHE_DIR"
        const val EXTRA_MAX_TEXTURE_SIZE = "wse2.r5a.MAX_TEXTURE_SIZE"
        const val EXTRA_RENDER_SCALE = "wse2.r5b.RENDER_SCALE"
        const val EXTRA_AUTO_RESOLUTION = "wse2.r5b.AUTO_RESOLUTION"
        const val EXTRA_SHARPNESS = "wse2.r5b.SHARPNESS"
        const val EXTRA_FXAA = "wse2.r5b.FXAA"
    }
}
