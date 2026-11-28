package org.wse2.android

import android.app.Activity
import android.content.ClipboardManager
import android.content.Context
import android.graphics.Color
import android.graphics.Rect
import android.graphics.Typeface
import android.graphics.drawable.ClipDrawable
import android.graphics.drawable.GradientDrawable
import android.graphics.drawable.LayerDrawable
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.text.Editable
import android.text.InputType
import android.text.TextWatcher
import android.util.TypedValue
import android.view.Gravity
import android.view.KeyEvent
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.view.ViewTreeObserver
import android.view.WindowInsets
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputMethodManager
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.HorizontalScrollView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.SeekBar
import android.widget.TextView
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Minimalistic Iron & Gold in-game overlays for `:recomp` (`RecompGameActivity`).
 * Pure Android Views with zero Compose initialization to keep `:recomp` memory lean.
 */
internal class InGameStyle(private val ctx: Context) {
    val bgCard = 0xEC131519.toInt()
    val bgSurface = 0xFF1A1D24.toInt()
    val bgChip = 0xFF222630.toInt()
    val bgGoldTint = 0x2EC8A45D.toInt()
    val gold = 0xFFC8A45D.toInt()
    val goldText = 0xFFE8D4A6.toInt()
    val onGold = 0xFF141518.toInt()
    val textPrimary = 0xFFF2ECE2.toInt()
    val textMuted = 0xFF9C968B.toInt()
    val border = 0xFF2E333E.toInt()
    val dangerBg = 0x2EE05A5A.toInt()
    val dangerBorder = 0x7AE05A5A.toInt()
    val dangerText = 0xFFF09090.toInt()

    val cinzel: Typeface = loadFont(R.font.cinzel, Typeface.SERIF)
    val manrope: Typeface = loadFont(R.font.manrope_medium, Typeface.SANS_SERIF)
    val manropeBold: Typeface = loadFont(R.font.manrope_semibold, Typeface.DEFAULT_BOLD)

    private fun loadFont(resId: Int, fallback: Typeface): Typeface = try {
        ctx.resources.getFont(resId)
    } catch (_: Throwable) {
        fallback
    }

    fun dp(v: Int): Int =
        TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v.toFloat(), ctx.resources.displayMetrics).toInt()

    fun rounded(fill: Int, radiusDp: Int, strokeColor: Int = 0, strokeDp: Int = 1): GradientDrawable =
        GradientDrawable().apply {
            setColor(fill)
            cornerRadius = dp(radiusDp).toFloat()
            if (strokeColor != 0) {
                setStroke(dp(strokeDp).coerceAtLeast(1), strokeColor)
            }
        }

    fun label(
        text: String,
        sp: Float = 13f,
        color: Int = textPrimary,
        font: Typeface = manrope
    ): TextView = TextView(ctx).apply {
        this.text = text
        setTextSize(TypedValue.COMPLEX_UNIT_SP, sp)
        setTextColor(color)
        typeface = font
    }

    fun pillButton(
        text: String,
        primary: Boolean = false,
        danger: Boolean = false,
        sp: Float = 12f,
        padH: Int = 12,
        padV: Int = 8,
        onClick: () -> Unit
    ): TextView = TextView(ctx).apply {
        this.text = text
        setTextSize(TypedValue.COMPLEX_UNIT_SP, sp)
        typeface = manropeBold
        gravity = Gravity.CENTER
        setPadding(dp(padH), dp(padV), dp(padH), dp(padV))
        when {
            primary -> {
                setTextColor(onGold)
                background = rounded(gold, 10)
            }
            danger -> {
                setTextColor(dangerText)
                background = rounded(dangerBg, 10, dangerBorder)
            }
            else -> {
                setTextColor(textPrimary)
                background = rounded(bgChip, 10, border)
            }
        }
        setOnClickListener { onClick() }
    }

    fun switchRow(
        title: String,
        subtitle: String? = null,
        initial: Boolean,
        onToggle: (Boolean) -> Unit
    ): View {
        var checked = initial
        val row = LinearLayout(ctx).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, dp(7), 0, dp(7))
        }
        val textCol = LinearLayout(ctx).apply {
            orientation = LinearLayout.VERTICAL
        }
        textCol.addView(label(title, 13f, textPrimary, manropeBold))
        if (!subtitle.isNullOrBlank()) {
            textCol.addView(label(subtitle, 11f, textMuted, manrope))
        }
        row.addView(
            textCol,
            LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply {
                marginEnd = dp(10)
            }
        )

        val trackW = dp(40)
        val trackH = dp(22)
        val thumbSz = dp(16)
        val thumbPad = dp(3)
        val track = FrameLayout(ctx)
        val thumb = View(ctx)
        track.addView(thumb, FrameLayout.LayoutParams(thumbSz, thumbSz, Gravity.CENTER_VERTICAL))

        fun render() {
            track.background = rounded(if (checked) gold else bgChip, 12, if (checked) gold else border)
            thumb.background = rounded(if (checked) onGold else textMuted, 8)
            val lp = thumb.layoutParams as FrameLayout.LayoutParams
            lp.marginStart = if (checked) trackW - thumbSz - thumbPad else thumbPad
            thumb.layoutParams = lp
        }
        render()
        row.addView(track, LinearLayout.LayoutParams(trackW, trackH))
        row.setOnClickListener {
            checked = !checked
            render()
            onToggle(checked)
        }
        return row
    }

    fun sliderRow(
        title: String,
        minVal: Int,
        maxVal: Int,
        initialVal: Int,
        suffix: String = "%",
        onTrackingChanged: ((Boolean) -> Unit)? = null,
        onChange: (Int) -> Unit
    ): View {
        val cur = initialVal.coerceIn(minVal, maxVal)
        val col = LinearLayout(ctx).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(0, dp(6), 0, dp(4))
        }
        val header = LinearLayout(ctx).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        val titleTv = label(title, 13f, textPrimary, manropeBold)
        val valTv = label("$cur$suffix", 12f, goldText, manropeBold)
        header.addView(titleTv, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        header.addView(valTv)
        col.addView(header)

        val bgTrack = rounded(bgChip, 4)
        val progTrack = ClipDrawable(rounded(gold, 4), Gravity.START, ClipDrawable.HORIZONTAL)
        val progressLayer = LayerDrawable(arrayOf(bgTrack, progTrack)).apply {
            setId(0, android.R.id.background)
            setId(1, android.R.id.progress)
        }
        val thumbDrawable = GradientDrawable().apply {
            shape = GradientDrawable.OVAL
            setSize(dp(15), dp(15))
            setColor(gold)
            setStroke(dp(2), bgCard)
        }

        val seek = SeekBar(ctx).apply {
            max = (maxVal - minVal).coerceAtLeast(1)
            progress = (cur - minVal).coerceIn(0, max)
            progressDrawable = progressLayer
            thumb = thumbDrawable
            splitTrack = false
            setPadding(dp(8), dp(8), dp(8), dp(4))
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(sb: SeekBar?, p: Int, fromUser: Boolean) {
                    val actual = minVal + p
                    valTv.text = "$actual$suffix"
                    onChange(actual)
                }

                override fun onStartTrackingTouch(sb: SeekBar?) {
                    onTrackingChanged?.invoke(true)
                }

                override fun onStopTrackingTouch(sb: SeekBar?) {
                    onTrackingChanged?.invoke(false)
                }
            })
        }
        col.addView(
            seek,
            LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dp(32))
        )
        return col
    }

    fun chipGroup(
        title: String,
        options: List<String>,
        selectedIndex: Int,
        valueReadout: ((Int) -> String)? = null,
        onSelected: (Int) -> Unit
    ): View {
        var cur = selectedIndex.coerceIn(0, (options.size - 1).coerceAtLeast(0))
        val col = LinearLayout(ctx).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(0, dp(6), 0, dp(6))
        }
        val header = LinearLayout(ctx).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, 0, 0, dp(6))
        }
        header.addView(
            label(title, 13f, textPrimary, manropeBold),
            LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        )
        val readoutTv = if (valueReadout != null) {
            label(valueReadout(cur), 12f, goldText, manropeBold).also { header.addView(it) }
        } else {
            null
        }
        col.addView(header)

        val row = LinearLayout(ctx).apply {
            orientation = LinearLayout.HORIZONTAL
        }
        val views = ArrayList<TextView>(options.size)
        fun refresh() {
            readoutTv?.text = valueReadout?.invoke(cur).orEmpty()
            views.forEachIndexed { idx, tv ->
                val sel = idx == cur
                tv.setTextColor(if (sel) goldText else textMuted)
                tv.typeface = if (sel) manropeBold else manrope
                tv.background = rounded(
                    fill = if (sel) bgGoldTint else bgSurface,
                    radiusDp = 8,
                    strokeColor = if (sel) gold else border
                )
            }
        }
        options.forEachIndexed { idx, opt ->
            val tv = TextView(ctx).apply {
                text = opt
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 11.5f)
                gravity = Gravity.CENTER
                maxLines = 1
                setPadding(dp(4), dp(8), dp(4), dp(8))
                setOnClickListener {
                    if (cur != idx) {
                        cur = idx
                        refresh()
                        onSelected(idx)
                    }
                }
            }
            views.add(tv)
            row.addView(
                tv,
                LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply {
                    if (idx > 0) marginStart = dp(5)
                }
            )
        }
        refresh()
        col.addView(row)
        return col
    }

    fun divider(): View = View(ctx).apply {
        setBackgroundColor(border)
        layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 1).apply {
            topMargin = dp(4)
            bottomMargin = dp(4)
        }
    }
}

/**
 * Hydrogen-inspired minimalistic right-docked Quick Options glass card.
 */
internal class QuickOptionsOverlay(
    private val activity: Activity,
    private val parent: FrameLayout,
    private val moduleName: String,
    private val telemetryText: String,
    private val callbacks: Callbacks
) {
    interface Callbacks {
        var pictureResIndex: Int
        var pictureTargetFps: Int
        var pictureSharpen: Boolean
        var pictureFxaa: Boolean
        var showTouchControls: Boolean
        var aimHold: Boolean
        var gyroEnabled: Boolean
        var touchOpacity: Float
        var buttonScale: Float
        var lookSensitivity: Float
        var soundVolume: Float
        var hapticsEnabled: Boolean
        var hapticWhenHit: Boolean
        var hapticWhenDealing: Boolean
        var hapticStrength: Int
        var showFpsOverlay: Boolean

        fun onGraphicsChanged()
        fun onControlsChanged()
        fun onFpsOverlayChanged(show: Boolean)
        fun onOpenKeyboard()
        fun onEnterLayoutEdit()
        fun onSendEscape()
        fun onSendQuickSave()
        fun onExitGame()
        fun onDismissed()
    }

    private val s = InGameStyle(activity)
    private val handler = Handler(Looper.getMainLooper())
    private var rootView: FrameLayout? = null
    private var confirmExitPending = false
    private var resetExitRunnable: Runnable? = null

    val isShowing: Boolean get() = rootView != null

    fun show() {
        if (rootView != null) return
        val scrim = FrameLayout(activity).apply {
            setBackgroundColor(0x66000000)
            isClickable = true
            isFocusable = true
            setOnClickListener { dismiss() }
        }

        val screenW = activity.resources.displayMetrics.widthPixels
        val cardW = min(s.dp(400), (screenW * 0.44f).roundToInt()).coerceAtLeast(s.dp(300))

        val card = LinearLayout(activity).apply {
            orientation = LinearLayout.VERTICAL
            background = s.rounded(s.bgCard, 16, s.border)
            setPadding(s.dp(16), s.dp(14), s.dp(16), s.dp(12))
            isClickable = true // Prevent tap-through to scrim
        }

        fun setGhosting(ghost: Boolean) {
            card.animate().alpha(if (ghost) 0.16f else 1.0f).setDuration(110L).start()
            scrim.setBackgroundColor(if (ghost) Color.TRANSPARENT else 0x66000000)
        }

        // 1. Header
        val header = LinearLayout(activity).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        val titleCol = LinearLayout(activity).apply {
            orientation = LinearLayout.VERTICAL
        }
        titleCol.addView(s.label(moduleName.uppercase(), 16f, s.textPrimary, s.cinzel))
        if (telemetryText.isNotBlank()) {
            titleCol.addView(s.label(telemetryText, 11f, s.textMuted, s.manrope))
        }
        header.addView(titleCol, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        val closeBtn = s.pillButton("✕", sp = 13f, padH = 10, padV = 6) { dismiss() }
        header.addView(closeBtn)
        card.addView(header)

        // 2. Top 4 minimal action tiles: Resume, Keyboard, Edit HUD, Exit (2-tap confirm)
        val actionRow = LinearLayout(activity).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding(0, s.dp(10), 0, s.dp(10))
        }
        val resumeBtn = s.pillButton("Resume", primary = true, sp = 12f, padH = 6, padV = 9) { dismiss() }
        val kbdBtn = s.pillButton("Keyboard", sp = 12f, padH = 6, padV = 9) {
            dismiss(notifyDismissed = false)
            callbacks.onOpenKeyboard()
        }
        val hudBtn = s.pillButton("Edit HUD", sp = 12f, padH = 6, padV = 9) {
            dismiss(notifyDismissed = false)
            callbacks.onEnterLayoutEdit()
        }
        lateinit var exitBtn: TextView
        exitBtn = s.pillButton("Exit", danger = true, sp = 12f, padH = 6, padV = 9) {
            if (confirmExitPending) {
                resetExitRunnable?.let { handler.removeCallbacks(it) }
                resetExitRunnable = null
                confirmExitPending = false
                dismiss(notifyDismissed = false)
                callbacks.onExitGame()
            } else {
                confirmExitPending = true
                exitBtn.text = "Confirm?"
                resetExitRunnable?.let { handler.removeCallbacks(it) }
                val r = Runnable {
                    confirmExitPending = false
                    exitBtn.text = "Exit"
                }
                resetExitRunnable = r
                handler.postDelayed(r, 3000L)
            }
        }
        listOf(resumeBtn, kbdBtn, hudBtn, exitBtn).forEachIndexed { idx, btn ->
            actionRow.addView(
                btn,
                LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply {
                    if (idx > 0) marginStart = s.dp(6)
                }
            )
        }
        card.addView(actionRow)

        // 3. Minimal 4-tab bar: Picture, Controls, Audio, Session
        val tabNames = listOf("Picture", "Controls", "Audio", "Session")
        val tabBar = LinearLayout(activity).apply {
            orientation = LinearLayout.HORIZONTAL
            background = s.rounded(s.bgSurface, 10, s.border)
            setPadding(s.dp(3), s.dp(3), s.dp(3), s.dp(3))
        }
        val tabViews = ArrayList<TextView>(tabNames.size)
        val contentHost = FrameLayout(activity)

        fun buildPictureTab(): View = LinearLayout(activity).apply {
            orientation = LinearLayout.VERTICAL
            val resLabels = listOf("Auto", "100%", "85%", "75%", "67%", "50%")
            addView(
                s.chipGroup(
                    title = "Render resolution",
                    options = resLabels,
                    selectedIndex = callbacks.pictureResIndex,
                    valueReadout = { idx -> if (idx == 0) "Auto" else resLabels[idx] }
                ) { idx ->
                    callbacks.pictureResIndex = idx
                    callbacks.onGraphicsChanged()
                }
            )
            val fpsValues = intArrayOf(30, 60, 90, 120)
            val fpsIdx = fpsValues.indexOf(callbacks.pictureTargetFps).let { if (it < 0) 1 else it }
            addView(
                s.chipGroup(
                    title = "Target FPS",
                    options = listOf("30", "60", "90", "120"),
                    selectedIndex = fpsIdx,
                    valueReadout = { idx -> "${fpsValues[idx]} FPS" }
                ) { idx ->
                    callbacks.pictureTargetFps = fpsValues[idx]
                    callbacks.onGraphicsChanged()
                }
            )
            addView(s.divider())
            addView(
                s.switchRow(
                    title = "Sharpen picture (FSR)",
                    subtitle = "Crispens scaled 3D image",
                    initial = callbacks.pictureSharpen
                ) {
                    callbacks.pictureSharpen = it
                    callbacks.onGraphicsChanged()
                }
            )
            addView(
                s.switchRow(
                    title = "Smooth jagged edges (FXAA)",
                    subtitle = "Lightweight post-process AA",
                    initial = callbacks.pictureFxaa
                ) {
                    callbacks.pictureFxaa = it
                    callbacks.onGraphicsChanged()
                }
            )
        }

        fun buildControlsTab(): View = LinearLayout(activity).apply {
            orientation = LinearLayout.VERTICAL
            addView(
                s.switchRow("Show touch controls", initial = callbacks.showTouchControls) {
                    callbacks.showTouchControls = it
                    callbacks.onControlsChanged()
                }
            )
            addView(
                s.switchRow("Hold to zoom", initial = callbacks.aimHold) {
                    callbacks.aimHold = it
                    callbacks.onControlsChanged()
                }
            )
            addView(
                s.switchRow("Gyro aiming while zooming", initial = callbacks.gyroEnabled) {
                    callbacks.gyroEnabled = it
                    callbacks.onControlsChanged()
                }
            )
            addView(s.divider())
            addView(
                s.sliderRow(
                    title = "Touch opacity",
                    minVal = 20,
                    maxVal = 100,
                    initialVal = (callbacks.touchOpacity * 100f).roundToInt(),
                    onTrackingChanged = ::setGhosting
                ) {
                    callbacks.touchOpacity = (it / 100f).coerceIn(0.2f, 1.0f)
                    callbacks.onControlsChanged()
                }
            )
            addView(
                s.sliderRow(
                    title = "Button size",
                    minVal = 70,
                    maxVal = 145,
                    initialVal = (callbacks.buttonScale * 100f).roundToInt(),
                    onTrackingChanged = ::setGhosting
                ) {
                    callbacks.buttonScale = (it / 100f).coerceIn(0.7f, 1.45f)
                    callbacks.onControlsChanged()
                }
            )
            addView(
                s.sliderRow(
                    title = "Look speed",
                    minVal = 25,
                    maxVal = 200,
                    initialVal = (callbacks.lookSensitivity * 100f).roundToInt()
                ) {
                    callbacks.lookSensitivity = (it / 100f).coerceIn(0.25f, 2.0f)
                    callbacks.onControlsChanged()
                }
            )
        }

        fun buildAudioTab(): View = LinearLayout(activity).apply {
            orientation = LinearLayout.VERTICAL
            addView(
                s.sliderRow(
                    title = "Sound volume",
                    minVal = 0,
                    maxVal = 100,
                    initialVal = (callbacks.soundVolume * 100f).roundToInt()
                ) {
                    callbacks.soundVolume = (it / 100f).coerceIn(0.0f, 1.0f)
                    callbacks.onControlsChanged()
                }
            )
            addView(s.divider())
            addView(
                s.switchRow("Vibrate on hits", initial = callbacks.hapticsEnabled) {
                    callbacks.hapticsEnabled = it
                    callbacks.onControlsChanged()
                }
            )
            addView(
                s.switchRow("When you are hit", initial = callbacks.hapticWhenHit) {
                    callbacks.hapticWhenHit = it
                    callbacks.onControlsChanged()
                }
            )
            addView(
                s.switchRow("When landing a hit", initial = callbacks.hapticWhenDealing) {
                    callbacks.hapticWhenDealing = it
                    callbacks.onControlsChanged()
                }
            )
            addView(
                s.sliderRow(
                    title = "Vibration strength",
                    minVal = 10,
                    maxVal = 100,
                    initialVal = callbacks.hapticStrength
                ) {
                    callbacks.hapticStrength = it.coerceIn(10, 100)
                    callbacks.onControlsChanged()
                }
            )
        }

        fun buildSessionTab(): View = LinearLayout(activity).apply {
            orientation = LinearLayout.VERTICAL
            val shortcutRow = LinearLayout(activity).apply {
                orientation = LinearLayout.HORIZONTAL
                setPadding(0, s.dp(6), 0, s.dp(8))
            }
            val escBtn = s.pillButton("Game Menu (ESC)", sp = 12f, padH = 8, padV = 10) {
                dismiss(notifyDismissed = false)
                callbacks.onSendEscape()
            }
            val f12Btn = s.pillButton("Quick Save (F12)", sp = 12f, padH = 8, padV = 10) {
                dismiss(notifyDismissed = false)
                callbacks.onSendQuickSave()
            }
            shortcutRow.addView(escBtn, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
            shortcutRow.addView(
                f12Btn,
                LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply {
                    marginStart = s.dp(8)
                }
            )
            addView(shortcutRow)
            addView(s.divider())
            addView(
                s.switchRow(
                    title = "Show FPS pill",
                    subtitle = "Top-center performance indicator",
                    initial = callbacks.showFpsOverlay
                ) {
                    callbacks.showFpsOverlay = it
                    callbacks.onFpsOverlayChanged(it)
                }
            )
        }

        fun selectTab(idx: Int) {
            tabViews.forEachIndexed { i, tv ->
                val sel = i == idx
                tv.setTextColor(if (sel) s.goldText else s.textMuted)
                tv.typeface = if (sel) s.manropeBold else s.manrope
                tv.background = if (sel) s.rounded(s.bgGoldTint, 8, s.gold) else null
            }
            contentHost.removeAllViews()
            val page = when (idx) {
                0 -> buildPictureTab()
                1 -> buildControlsTab()
                2 -> buildAudioTab()
                else -> buildSessionTab()
            }
            contentHost.addView(page)
        }

        tabNames.forEachIndexed { idx, name ->
            val tv = TextView(activity).apply {
                text = name
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 12f)
                gravity = Gravity.CENTER
                setPadding(s.dp(4), s.dp(7), s.dp(4), s.dp(7))
                setOnClickListener { selectTab(idx) }
            }
            tabViews.add(tv)
            tabBar.addView(tv, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        }
        card.addView(tabBar)

        val scroll = ScrollView(activity).apply {
            isVerticalScrollBarEnabled = false
            addView(
                contentHost,
                ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
            )
        }
        card.addView(
            scroll,
            LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f).apply {
                topMargin = s.dp(8)
            }
        )

        card.addView(
            s.label("Changes apply live · Saved to $moduleName", 11f, s.textMuted, s.manrope).apply {
                gravity = Gravity.CENTER
                setPadding(0, s.dp(6), 0, 0)
            }
        )

        selectTab(0)

        scrim.addView(
            card,
            FrameLayout.LayoutParams(cardW, FrameLayout.LayoutParams.MATCH_PARENT, Gravity.END).apply {
                setMargins(s.dp(12), s.dp(12), s.dp(12), s.dp(12))
            }
        )
        parent.addView(
            scrim,
            FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT)
        )
        rootView = scrim
    }

    fun dismiss(notifyDismissed: Boolean = true) {
        handler.removeCallbacksAndMessages(null)
        confirmExitPending = false
        val v = rootView ?: return
        rootView = null
        parent.removeView(v)
        if (notifyDismissed) {
            callbacks.onDismissed()
        }
    }
}

/**
 * Neon-inspired IME-docked floating keyboard bar + PC key rail.
 * Sits directly above the system keyboard so WSE2's top console & center dialogs stay unobstructed.
 */
internal class InGameKeyboardOverlay(
    private val activity: Activity,
    private val parent: FrameLayout,
    private val onText: (String) -> Unit,
    private val onKeyPulse: (dik: Int, vk: Int, ctrl: Boolean, alt: Boolean) -> Unit,
    private val onDismissed: () -> Unit
) {
    private val s = InGameStyle(activity)
    private val handler = Handler(Looper.getMainLooper())
    private var rootView: FrameLayout? = null
    private var editText: EditText? = null
    private var globalLayoutListener: ViewTreeObserver.OnGlobalLayoutListener? = null
    private var liveMode = true
    private var stickyCtrl = false
    private var stickyAlt = false
    private var ignoreWatcher = false
    private var lastLiveText = ""

    val isShowing: Boolean get() = rootView != null

    fun show() {
        if (rootView != null) return
        liveMode = true
        stickyCtrl = false
        stickyAlt = false
        lastLiveText = ""

        val overlay = FrameLayout(activity).apply {
            setBackgroundColor(Color.TRANSPARENT)
            isClickable = true
            setOnClickListener { dismiss() }
        }

        val dock = LinearLayout(activity).apply {
            orientation = LinearLayout.VERTICAL
            background = GradientDrawable().apply {
                setColor(s.bgCard)
                setStroke(s.dp(1), s.border)
            }
            setPadding(s.dp(10), s.dp(6), s.dp(10), s.dp(6))
            isClickable = true
        }

        // Optional inline F1-F12 row (toggled from key rail)
        val fnRow = LinearLayout(activity).apply {
            orientation = LinearLayout.HORIZONTAL
            visibility = View.GONE
            setPadding(0, 0, 0, s.dp(5))
        }
        val fDiks = intArrayOf(0x3B, 0x3C, 0x3D, 0x3E, 0x3F, 0x40, 0x41, 0x42, 0x43, 0x44, 0x57, 0x58)
        for (i in 0 until 12) {
            val fNum = i + 1
            val btn = s.pillButton("F$fNum", sp = 11f, padH = 4, padV = 5) {
                sendKey(fDiks[i], 0x70 + i)
            }
            fnRow.addView(
                btn,
                LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply {
                    if (i > 0) marginStart = s.dp(4)
                }
            )
        }
        dock.addView(fnRow)

        // Row 1: Compact PC Key Rail
        val rail = LinearLayout(activity).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, 0, 0, s.dp(6))
        }
        lateinit var ctrlBtn: TextView
        lateinit var altBtn: TextView

        fun refreshModifiers() {
            ctrlBtn.setTextColor(if (stickyCtrl) s.goldText else s.textPrimary)
            ctrlBtn.background = s.rounded(
                if (stickyCtrl) s.bgGoldTint else s.bgChip, 8, if (stickyCtrl) s.gold else s.border
            )
            altBtn.setTextColor(if (stickyAlt) s.goldText else s.textPrimary)
            altBtn.background = s.rounded(
                if (stickyAlt) s.bgGoldTint else s.bgChip, 8, if (stickyAlt) s.gold else s.border
            )
        }

        fun addRailKey(label: String, onTap: () -> Unit): TextView {
            val tv = s.pillButton(label, sp = 11.5f, padH = 9, padV = 5, onClick = onTap)
            rail.addView(
                tv,
                LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply {
                    marginEnd = s.dp(5)
                }
            )
            return tv
        }

        addRailKey("Esc") { sendKey(0x01, 0x1B) }
        addRailKey("Tab") { sendKey(0x0F, 0x09) }
        addRailKey("~") { sendKey(0x29, 0xC0) }
        addRailKey("↑") { sendKey(0xC8, 0x26) }
        addRailKey("↓") { sendKey(0xD0, 0x28) }
        addRailKey("←") { sendKey(0xCB, 0x25) }
        addRailKey("→") { sendKey(0xCD, 0x27) }
        ctrlBtn = addRailKey("Ctrl") {
            stickyCtrl = !stickyCtrl
            refreshModifiers()
        }
        altBtn = addRailKey("Alt") {
            stickyAlt = !stickyAlt
            refreshModifiers()
        }
        addRailKey("F1-12") {
            fnRow.visibility = if (fnRow.visibility == View.VISIBLE) View.GONE else View.VISIBLE
        }
        addRailKey("Paste") {
            val cm = activity.getSystemService(Context.CLIPBOARD_SERVICE) as? ClipboardManager
            val clip = cm?.primaryClip?.getItemAt(0)?.coerceToText(activity)?.toString().orEmpty()
            if (clip.isNotEmpty()) {
                if (liveMode) {
                    onText(clip)
                } else {
                    editText?.append(clip)
                }
            }
        }
        val railScroll = HorizontalScrollView(activity).apply {
            isHorizontalScrollBarEnabled = false
            addView(rail)
        }
        dock.addView(railScroll)

        // Row 2: Live/Draft toggle + Input + Hold-to-repeat Backspace + Enter + Close
        val inputRow = LinearLayout(activity).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }

        lateinit var modeBtn: TextView
        val input = EditText(activity).apply {
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS
            imeOptions = EditorInfo.IME_FLAG_NO_EXTRACT_UI or
                EditorInfo.IME_FLAG_NO_FULLSCREEN or
                EditorInfo.IME_ACTION_SEND
            hint = "Streaming live keystrokes…"
            setHintTextColor(s.textMuted)
            setTextColor(s.textPrimary)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 13.5f)
            typeface = s.manrope
            isSingleLine = true
            background = s.rounded(s.bgSurface, 10, s.border)
            setPadding(s.dp(12), s.dp(8), s.dp(12), s.dp(8))
        }
        editText = input

        modeBtn = s.pillButton("Live", sp = 11.5f, padH = 10, padV = 8) {
            liveMode = !liveMode
            modeBtn.text = if (liveMode) "Live" else "Draft"
            modeBtn.setTextColor(if (liveMode) s.goldText else s.textPrimary)
            modeBtn.background = s.rounded(
                if (liveMode) s.bgGoldTint else s.bgChip, 10, if (liveMode) s.gold else s.border
            )
            input.hint = if (liveMode) "Streaming live keystrokes…" else "Type text, then tap Enter…"
            if (liveMode) {
                ignoreWatcher = true
                input.setText("")
                lastLiveText = ""
                ignoreWatcher = false
            }
        }.apply {
            setTextColor(s.goldText)
            background = s.rounded(s.bgGoldTint, 10, s.gold)
        }

        input.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(cs: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(cs: CharSequence?, start: Int, before: Int, count: Int) {}
            override fun afterTextChanged(ed: Editable?) {
                if (ignoreWatcher || !liveMode) return
                val cur = ed?.toString().orEmpty()
                // Find longest common prefix with lastLiveText to diff live edits cleanly.
                var common = 0
                val maxCommon = min(cur.length, lastLiveText.length)
                while (common < maxCommon && cur[common] == lastLiveText[common]) {
                    ++common
                }
                val deleted = lastLiveText.length - common
                val added = cur.substring(common)
                repeat(deleted) { onText("\b") }
                if (added.isNotEmpty()) {
                    if (stickyCtrl || stickyAlt) {
                        val ch = added.last().uppercaseChar()
                        val dik = asciiToDik(ch)
                        val vk = ch.code
                        if (dik != 0) {
                            sendKey(dik, vk)
                            refreshModifiers()
                        } else {
                            onText(added)
                        }
                    } else {
                        onText(added)
                    }
                }
                lastLiveText = cur
                // Keep live field short so IME state stays responsive.
                if (cur.length > 48) {
                    ignoreWatcher = true
                    input.setText("")
                    lastLiveText = ""
                    ignoreWatcher = false
                }
            }
        })

        fun commitEnter() {
            if (liveMode) {
                onText("\n")
                ignoreWatcher = true
                input.setText("")
                lastLiveText = ""
                ignoreWatcher = false
            } else {
                val txt = input.text?.toString().orEmpty()
                if (txt.isNotEmpty()) {
                    onText(txt)
                }
                onText("\n")
                input.setText("")
            }
        }

        input.setOnEditorActionListener { _, actionId, event ->
            if (actionId == EditorInfo.IME_ACTION_SEND ||
                actionId == EditorInfo.IME_ACTION_DONE ||
                (event?.keyCode == KeyEvent.KEYCODE_ENTER && event.action == KeyEvent.ACTION_DOWN)
            ) {
                commitEnter()
                true
            } else {
                false
            }
        }

        // Hold-to-repeat Backspace button (never closes the bar!)
        val bsRepeat = object : Runnable {
            override fun run() {
                onText("\b")
                handler.postDelayed(this, 65L)
            }
        }
        val bsBtn = s.pillButton("⌫", sp = 14f, padH = 12, padV = 7) {}.apply {
            setOnTouchListener { v, ev ->
                when (ev.actionMasked) {
                    MotionEvent.ACTION_DOWN -> {
                        v.isPressed = true
                        onText("\b")
                        handler.removeCallbacks(bsRepeat)
                        handler.postDelayed(bsRepeat, 320L)
                        true
                    }
                    MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                        v.isPressed = false
                        handler.removeCallbacks(bsRepeat)
                        true
                    }
                    else -> false
                }
            }
        }

        val enterBtn = s.pillButton("↵ Enter", primary = true, sp = 12f, padH = 12, padV = 8) {
            commitEnter()
        }
        val closeBtn = s.pillButton("✕", sp = 13f, padH = 10, padV = 8) { dismiss() }

        inputRow.addView(modeBtn, LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply {
            marginEnd = s.dp(6)
        })
        inputRow.addView(input, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply {
            marginEnd = s.dp(6)
        })
        inputRow.addView(bsBtn, LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply {
            marginEnd = s.dp(6)
        })
        inputRow.addView(enterBtn, LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT).apply {
            marginEnd = s.dp(6)
        })
        inputRow.addView(closeBtn)
        dock.addView(inputRow)

        val dockLp = FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.MATCH_PARENT,
            FrameLayout.LayoutParams.WRAP_CONTENT,
            Gravity.BOTTOM
        )
        overlay.addView(dock, dockLp)

        // Track IME height cleanly so the bar hugs the top of the software keyboard.
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            overlay.setOnApplyWindowInsetsListener { _, insets ->
                val imeBottom = insets.getInsets(WindowInsets.Type.ime()).bottom
                dockLp.bottomMargin = imeBottom
                dock.layoutParams = dockLp
                insets
            }
        } else {
            val rect = Rect()
            val listener = ViewTreeObserver.OnGlobalLayoutListener {
                parent.getWindowVisibleDisplayFrame(rect)
                val imeBottom = (parent.height - rect.bottom).coerceAtLeast(0)
                if (dockLp.bottomMargin != imeBottom) {
                    dockLp.bottomMargin = imeBottom
                    dock.layoutParams = dockLp
                }
            }
            globalLayoutListener = listener
            parent.viewTreeObserver.addOnGlobalLayoutListener(listener)
        }

        parent.addView(
            overlay,
            FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT)
        )
        rootView = overlay

        input.requestFocus()
        input.post {
            val imm = activity.getSystemService(Context.INPUT_METHOD_SERVICE) as? InputMethodManager
            imm?.showSoftInput(input, InputMethodManager.SHOW_IMPLICIT)
        }
    }

    private fun sendKey(dik: Int, vk: Int) {
        val c = stickyCtrl
        val a = stickyAlt
        stickyCtrl = false
        stickyAlt = false
        onKeyPulse(dik, vk, c, a)
    }

    private fun asciiToDik(ch: Char): Int = when (ch) {
        'A' -> 0x1E; 'B' -> 0x30; 'C' -> 0x2E; 'D' -> 0x20; 'E' -> 0x12; 'F' -> 0x21
        'G' -> 0x22; 'H' -> 0x23; 'I' -> 0x17; 'J' -> 0x24; 'K' -> 0x25; 'L' -> 0x26
        'M' -> 0x32; 'N' -> 0x31; 'O' -> 0x18; 'P' -> 0x19; 'Q' -> 0x10; 'R' -> 0x13
        'S' -> 0x1F; 'T' -> 0x14; 'U' -> 0x16; 'V' -> 0x2F; 'W' -> 0x11; 'X' -> 0x2D
        'Y' -> 0x15; 'Z' -> 0x2C
        ' ' -> 0x39
        else -> 0
    }

    fun dismiss() {
        handler.removeCallbacksAndMessages(null)
        globalLayoutListener?.let { parent.viewTreeObserver.removeOnGlobalLayoutListener(it) }
        globalLayoutListener = null
        editText?.let { et ->
            val imm = activity.getSystemService(Context.INPUT_METHOD_SERVICE) as? InputMethodManager
            imm?.hideSoftInputFromWindow(et.windowToken, 0)
        }
        editText = null
        val v = rootView ?: return
        rootView = null
        parent.removeView(v)
        onDismissed()
    }
}
