package org.wse2.android

import android.app.Activity
import android.app.AlertDialog
import android.content.Context
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Process
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.text.InputType
import android.util.TypedValue
import android.view.Gravity
import android.view.InputDevice
import android.view.KeyEvent
import android.view.MotionEvent
import android.view.Surface
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.View
import android.view.WindowInsets
import android.view.WindowInsetsController
import android.view.WindowManager
import android.view.inputmethod.InputMethodManager
import android.widget.CheckBox
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.SeekBar
import android.widget.TextView
import android.window.OnBackInvokedDispatcher
import org.wse2.android.launcher.R4Native
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.roundToInt

// Phase 5 R5a/R6 live Vulkan presentation & input window (`docs/RECOMP_PLAN.md` "R5", "R6").
// Runs in its own isolated `:recomp` process so translated fixed-address game mappings and
// experimental GPU drivers never affect the launcher process.
class RecompGameActivity : Activity(), SurfaceHolder.Callback, SensorEventListener {

    private lateinit var surfaceView: SurfaceView
    private lateinit var hudPill: TextView
    private lateinit var exitButton: TextView

    private val started = AtomicBoolean(false)
    private val finishing = AtomicBoolean(false)
    private var hudDimmed = false

    // Live R6 controls & sound preferences (plain player words in the quick drawer).
    private var touchOpacity = 0.55f
    private var buttonScale = 1.0f
    private var lookSensitivity = 1.0f
    private var aimHold = true
    private var gyroEnabled = false
    private var showTouchControls = true
    private var soundVolume = 1.0f
    private var hapticsEnabled = true
    private var hapticWhenHit = true
    private var hapticWhenDealing = true
    private var hapticStrength = 60

    // Gamepad state snapshot for XInput 1.3.
    private var padButtons = 0
    private var padLt = 0
    private var padRt = 0
    private var padLx = 0
    private var padLy = 0
    private var padRx = 0
    private var padRy = 0
    private var padConnected = false

    // Optional gyroscope aiming sensor.
    private var sensorManager: SensorManager? = null
    private var gyroSensor: Sensor? = null
    private var lastGyroNs = 0L

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        window.decorView.setBackgroundColor(Color.BLACK)

        sensorManager = getSystemService(Context.SENSOR_SERVICE) as? SensorManager
        gyroSensor = sensorManager?.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

        val root = FrameLayout(this).apply {
            setBackgroundColor(Color.BLACK)
            isFocusable = true
            isFocusableInTouchMode = true
        }

        surfaceView = SurfaceView(this).apply {
            holder.addCallback(this@RecompGameActivity)
            isFocusable = true
            isFocusableInTouchMode = true
            setOnTouchListener { _, event -> handleTouchEvent(event) }
            setOnGenericMotionListener { _, event -> handleGenericMotionEvent(event) }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                setOnCapturedPointerListener { _, event -> handleCapturedPointerEvent(event) }
            }
        }
        root.addView(
            surfaceView,
            FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
            )
        )

        root.setOnApplyWindowInsetsListener { v, insets ->
            val w = v.width.coerceAtLeast(1).toFloat()
            val h = v.height.coerceAtLeast(1).toFloat()
            val cutout = insets.displayCutout
            if (cutout != null && R4Native.loadError == null) {
                R4Native.nativeSetSafeArea(
                    cutout.safeInsetLeft / w,
                    cutout.safeInsetTop / h,
                    cutout.safeInsetRight / w,
                    cutout.safeInsetBottom / h
                )
            }
            insets
        }

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
                Gravity.TOP or Gravity.CENTER_HORIZONTAL
            ).apply {
                setMargins(margin, margin, margin, margin)
            }
        )

        exitButton = TextView(this).apply {
            text = "Menu"
            setTextColor(Color.WHITE)
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 12f)
            typeface = Typeface.DEFAULT_BOLD
            setPadding(padH, padV, padH, padV)
            background = pillDrawable(0xCC28221A.toInt())
            setOnClickListener { showQuickDrawer() }
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
                showQuickDrawer()
            }
        }
    }

    override fun onPause() {
        super.onPause()
        unregisterGyro()
        if (started.get() && R4Native.loadError == null) {
            R4Native.nativeSetPaused(true)
        }
    }

    override fun onResume() {
        super.onResume()
        applyImmersiveFullscreen()
        registerGyroIfEnabled()
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
        showQuickDrawer()
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
        unregisterGyro()
        if (R4Native.loadError == null) {
            try {
                R4Native.nativeFinishReport()
            } catch (_: Throwable) {
            }
        }
        // End the isolated `:recomp` process so the next run starts with clean fixed-address maps.
        Process.killProcess(Process.myPid())
    }

    // ---- Multi-touch, Mouse, Keyboard & Gamepad Input ----

    private fun handleTouchEvent(event: MotionEvent): Boolean {
        if (!started.get() || R4Native.loadError != null) return false
        val src = event.source
        if ((src and InputDevice.SOURCE_MOUSE) == InputDevice.SOURCE_MOUSE &&
            event.getToolType(0) == MotionEvent.TOOL_TYPE_MOUSE
        ) {
            handleMouseMotion(event)
            return true
        }
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN, MotionEvent.ACTION_POINTER_DOWN -> {
                val idx = event.actionIndex
                R4Native.nativeTouchEvent(event.getPointerId(idx), 0, event.getX(idx), event.getY(idx))
            }
            MotionEvent.ACTION_MOVE -> {
                for (i in 0 until event.pointerCount) {
                    R4Native.nativeTouchEvent(event.getPointerId(i), 2, event.getX(i), event.getY(i))
                }
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_POINTER_UP -> {
                val idx = event.actionIndex
                R4Native.nativeTouchEvent(event.getPointerId(idx), 1, event.getX(idx), event.getY(idx))
            }
            MotionEvent.ACTION_CANCEL -> {
                for (i in 0 until event.pointerCount) {
                    R4Native.nativeTouchEvent(event.getPointerId(i), 3, event.getX(i), event.getY(i))
                }
            }
        }
        return true
    }

    private fun handleMouseMotion(event: MotionEvent) {
        if (R4Native.loadError != null) return
        val cursorVisible = R4Native.nativeIsCursorVisible()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            if (!cursorVisible && !surfaceView.hasPointerCapture()) {
                surfaceView.requestPointerCapture()
            } else if (cursorVisible && surfaceView.hasPointerCapture()) {
                surfaceView.releasePointerCapture()
            }
        }
        R4Native.nativeMouseMove(event.x.roundToInt(), event.y.roundToInt(), false)
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN, MotionEvent.ACTION_BUTTON_PRESS -> {
                val b = mapMouseButton(event.actionButton)
                R4Native.nativeMouseButton(b, true)
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_BUTTON_RELEASE -> {
                val b = mapMouseButton(event.actionButton)
                R4Native.nativeMouseButton(b, false)
            }
        }
    }

    private fun handleCapturedPointerEvent(event: MotionEvent): Boolean {
        if (!started.get() || R4Native.loadError != null) return false
        if (R4Native.nativeIsCursorVisible() && Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            surfaceView.releasePointerCapture()
            return false
        }
        val dx = event.x.roundToInt()
        val dy = event.y.roundToInt()
        if (dx != 0 || dy != 0) {
            R4Native.nativeMouseMove(dx, dy, true)
        }
        when (event.actionMasked) {
            MotionEvent.ACTION_BUTTON_PRESS, MotionEvent.ACTION_DOWN ->
                R4Native.nativeMouseButton(mapMouseButton(event.actionButton), true)
            MotionEvent.ACTION_BUTTON_RELEASE, MotionEvent.ACTION_UP ->
                R4Native.nativeMouseButton(mapMouseButton(event.actionButton), false)
            MotionEvent.ACTION_SCROLL -> {
                val v = event.getAxisValue(MotionEvent.AXIS_VSCROLL)
                if (v != 0f) R4Native.nativeMouseWheel((v * 120f).roundToInt())
            }
        }
        return true
    }

    private fun handleGenericMotionEvent(event: MotionEvent): Boolean {
        if (!started.get() || R4Native.loadError != null) return false
        val src = event.source
        if ((src and InputDevice.SOURCE_JOYSTICK) == InputDevice.SOURCE_JOYSTICK ||
            (src and InputDevice.SOURCE_GAMEPAD) == InputDevice.SOURCE_GAMEPAD
        ) {
            padConnected = true
            padLx = (event.getAxisValue(MotionEvent.AXIS_X) * 32767f).roundToInt().coerceIn(-32768, 32767)
            padLy = (-event.getAxisValue(MotionEvent.AXIS_Y) * 32767f).roundToInt().coerceIn(-32768, 32767)
            val rx = event.getAxisValue(MotionEvent.AXIS_Z)
            val ry = event.getAxisValue(MotionEvent.AXIS_RZ)
            padRx = (rx * 32767f).roundToInt().coerceIn(-32768, 32767)
            padRy = (-ry * 32767f).roundToInt().coerceIn(-32768, 32767)
            val lt = maxOf(event.getAxisValue(MotionEvent.AXIS_LTRIGGER), event.getAxisValue(MotionEvent.AXIS_BRAKE))
            val rt = maxOf(event.getAxisValue(MotionEvent.AXIS_RTRIGGER), event.getAxisValue(MotionEvent.AXIS_GAS))
            padLt = (lt * 255f).roundToInt().coerceIn(0, 255)
            padRt = (rt * 255f).roundToInt().coerceIn(0, 255)
            val hx = event.getAxisValue(MotionEvent.AXIS_HAT_X)
            val hy = event.getAxisValue(MotionEvent.AXIS_HAT_Y)
            padButtons = (padButtons and 0x000F.inv()) or
                (if (hy < -0.5f) 0x0001 else 0) or
                (if (hy > 0.5f) 0x0002 else 0) or
                (if (hx < -0.5f) 0x0004 else 0) or
                (if (hx > 0.5f) 0x0008 else 0)
            pushGamepadState()
            return true
        }
        if ((src and InputDevice.SOURCE_MOUSE) == InputDevice.SOURCE_MOUSE) {
            when (event.actionMasked) {
                MotionEvent.ACTION_HOVER_MOVE ->
                    R4Native.nativeMouseMove(event.x.roundToInt(), event.y.roundToInt(), false)
                MotionEvent.ACTION_SCROLL -> {
                    val v = event.getAxisValue(MotionEvent.AXIS_VSCROLL)
                    if (v != 0f) R4Native.nativeMouseWheel((v * 120f).roundToInt())
                }
            }
            return true
        }
        return false
    }

    override fun dispatchKeyEvent(event: KeyEvent): Boolean {
        if (!started.get() || R4Native.loadError != null) return super.dispatchKeyEvent(event)
        val down = event.action == KeyEvent.ACTION_DOWN
        if (event.action != KeyEvent.ACTION_DOWN && event.action != KeyEvent.ACTION_UP) {
            return super.dispatchKeyEvent(event)
        }
        val padBit = mapGamepadButton(event.keyCode)
        if (padBit != 0 && ((event.source and InputDevice.SOURCE_GAMEPAD) == InputDevice.SOURCE_GAMEPAD ||
                (event.source and InputDevice.SOURCE_JOYSTICK) == InputDevice.SOURCE_JOYSTICK)
        ) {
            padConnected = true
            padButtons = if (down) (padButtons or padBit) else (padButtons and padBit.inv())
            pushGamepadState()
            return true
        }
        val dik = mapAndroidKeyToDik(event.keyCode)
        if (dik != 0) {
            R4Native.nativeKeyEvent(dik, event.keyCode, down)
            if (down && event.unicodeChar != 0 && !event.isCtrlPressed) {
                R4Native.nativeTextInput(event.unicodeChar.toChar().toString())
            }
            return true
        }
        return super.dispatchKeyEvent(event)
    }

    private fun pushGamepadState() {
        R4Native.nativeGamepadState(0, padConnected, padButtons, padLt, padRt, padLx, padLy, padRx, padRy)
    }

    private fun mapMouseButton(btn: Int): Int = when (btn) {
        MotionEvent.BUTTON_SECONDARY -> 1
        MotionEvent.BUTTON_TERTIARY -> 2
        else -> 0
    }

    private fun mapGamepadButton(keyCode: Int): Int = when (keyCode) {
        KeyEvent.KEYCODE_DPAD_UP -> 0x0001
        KeyEvent.KEYCODE_DPAD_DOWN -> 0x0002
        KeyEvent.KEYCODE_DPAD_LEFT -> 0x0004
        KeyEvent.KEYCODE_DPAD_RIGHT -> 0x0008
        KeyEvent.KEYCODE_BUTTON_START -> 0x0010
        KeyEvent.KEYCODE_BUTTON_SELECT -> 0x0020
        KeyEvent.KEYCODE_BUTTON_THUMBL -> 0x0040
        KeyEvent.KEYCODE_BUTTON_THUMBR -> 0x0080
        KeyEvent.KEYCODE_BUTTON_L1 -> 0x0100
        KeyEvent.KEYCODE_BUTTON_R1 -> 0x0200
        KeyEvent.KEYCODE_BUTTON_A -> 0x1000
        KeyEvent.KEYCODE_BUTTON_B -> 0x2000
        KeyEvent.KEYCODE_BUTTON_X -> 0x4000
        KeyEvent.KEYCODE_BUTTON_Y -> 0x8000
        else -> 0
    }

    private fun mapAndroidKeyToDik(keyCode: Int): Int = when (keyCode) {
        KeyEvent.KEYCODE_ESCAPE -> 0x01
        KeyEvent.KEYCODE_1 -> 0x02; KeyEvent.KEYCODE_2 -> 0x03; KeyEvent.KEYCODE_3 -> 0x04
        KeyEvent.KEYCODE_4 -> 0x05; KeyEvent.KEYCODE_5 -> 0x06; KeyEvent.KEYCODE_6 -> 0x07
        KeyEvent.KEYCODE_7 -> 0x08; KeyEvent.KEYCODE_8 -> 0x09; KeyEvent.KEYCODE_9 -> 0x0A
        KeyEvent.KEYCODE_0 -> 0x0B; KeyEvent.KEYCODE_DEL -> 0x0E; KeyEvent.KEYCODE_TAB -> 0x0F
        KeyEvent.KEYCODE_Q -> 0x10; KeyEvent.KEYCODE_W -> 0x11; KeyEvent.KEYCODE_E -> 0x12
        KeyEvent.KEYCODE_R -> 0x13; KeyEvent.KEYCODE_T -> 0x14; KeyEvent.KEYCODE_Y -> 0x15
        KeyEvent.KEYCODE_U -> 0x16; KeyEvent.KEYCODE_I -> 0x17; KeyEvent.KEYCODE_O -> 0x18
        KeyEvent.KEYCODE_P -> 0x19; KeyEvent.KEYCODE_ENTER -> 0x1C
        KeyEvent.KEYCODE_CTRL_LEFT -> 0x1D; KeyEvent.KEYCODE_A -> 0x1E; KeyEvent.KEYCODE_S -> 0x1F
        KeyEvent.KEYCODE_D -> 0x20; KeyEvent.KEYCODE_F -> 0x21; KeyEvent.KEYCODE_G -> 0x22
        KeyEvent.KEYCODE_H -> 0x23; KeyEvent.KEYCODE_J -> 0x24; KeyEvent.KEYCODE_K -> 0x25
        KeyEvent.KEYCODE_L -> 0x26; KeyEvent.KEYCODE_SHIFT_LEFT -> 0x2A
        KeyEvent.KEYCODE_Z -> 0x2C; KeyEvent.KEYCODE_X -> 0x2D; KeyEvent.KEYCODE_C -> 0x2E
        KeyEvent.KEYCODE_V -> 0x2F; KeyEvent.KEYCODE_B -> 0x30; KeyEvent.KEYCODE_N -> 0x31
        KeyEvent.KEYCODE_M -> 0x32; KeyEvent.KEYCODE_SPACE -> 0x39
        KeyEvent.KEYCODE_F1 -> 0x3B; KeyEvent.KEYCODE_F2 -> 0x3C; KeyEvent.KEYCODE_F3 -> 0x3D
        KeyEvent.KEYCODE_F4 -> 0x3E; KeyEvent.KEYCODE_F5 -> 0x3F; KeyEvent.KEYCODE_F12 -> 0x58
        KeyEvent.KEYCODE_DPAD_UP -> 0xC8; KeyEvent.KEYCODE_DPAD_LEFT -> 0xCB
        KeyEvent.KEYCODE_DPAD_RIGHT -> 0xCD; KeyEvent.KEYCODE_DPAD_DOWN -> 0xD0
        else -> 0
    }

    // ---- Opt-in Gyro Aiming ----

    private fun registerGyroIfEnabled() {
        val sm = sensorManager ?: return
        val gs = gyroSensor ?: return
        if (gyroEnabled) {
            lastGyroNs = 0L
            sm.registerListener(this, gs, SensorManager.SENSOR_DELAY_GAME)
        }
    }

    private fun unregisterGyro() {
        sensorManager?.unregisterListener(this)
        lastGyroNs = 0L
    }

    override fun onSensorChanged(event: SensorEvent) {
        if (!gyroEnabled || !started.get() || R4Native.loadError != null) return
        if (event.sensor.type != Sensor.TYPE_GYROSCOPE) return
        val nowNs = event.timestamp
        if (lastGyroNs != 0L && nowNs > lastGyroNs) {
            val dt = (nowNs - lastGyroNs) * 1e-9f
            R4Native.nativeGyroSample(event.values[0], event.values[1], dt)
        }
        lastGyroNs = nowNs
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    // ---- Combat Haptics (VibrationEffect) ----

    @Suppress("DEPRECATION")
    private fun triggerHapticPulse(durationMs: Int, amplitude: Int) {
        if (!hapticsEnabled || durationMs <= 0 || amplitude <= 0) return
        try {
            val vib = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                (getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as? VibratorManager)?.defaultVibrator
            } else {
                getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator
            } ?: return
            if (!vib.hasVibrator()) return
            val clampedAmp = amplitude.coerceIn(1, 255)
            val clampedMs = durationMs.coerceIn(8, 120).toLong()
            vib.vibrate(VibrationEffect.createOneShot(clampedMs, clampedAmp))
        } catch (_: Throwable) {
        }
    }

    // ---- Soft Keyboard & Plain-Language Quick Settings Drawer ----

    private fun showSoftKeyboardDialog() {
        runOnUiThread {
            if (isFinishing || isDestroyed) return@runOnUiThread
            val input = EditText(this).apply {
                inputType = InputType.TYPE_CLASS_TEXT
                hint = "Type text for the game"
            }
            val dlg = AlertDialog.Builder(this)
                .setTitle("Keyboard")
                .setView(input)
                .setPositiveButton("Send") { _, _ ->
                    val txt = input.text?.toString().orEmpty()
                    if (txt.isNotEmpty() && R4Native.loadError == null) {
                        R4Native.nativeTextInput(txt)
                    }
                    applyImmersiveFullscreen()
                }
                .setNegativeButton("Cancel") { _, _ -> applyImmersiveFullscreen() }
                .create()
            dlg.show()
            input.requestFocus()
            input.post {
                val imm = getSystemService(Context.INPUT_METHOD_SERVICE) as? InputMethodManager
                imm?.showSoftInput(input, InputMethodManager.SHOW_IMPLICIT)
            }
        }
    }

    private fun showQuickDrawer() {
        runOnUiThread {
            if (isFinishing || isDestroyed) return@runOnUiThread
            val container = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                setPadding(dp(18), dp(12), dp(18), dp(12))
            }

            fun addCheck(label: String, checked: Boolean, onChange: (Boolean) -> Unit) {
                container.addView(CheckBox(this).apply {
                    text = label
                    isChecked = checked
                    setOnCheckedChangeListener { _, isChecked ->
                        onChange(isChecked)
                        pushLiveConfig()
                    }
                })
            }

            fun addSlider(label: String, max: Int, value: Int, onChange: (Int) -> Unit) {
                val tv = TextView(this).apply { text = "$label: $value%" }
                container.addView(tv)
                container.addView(SeekBar(this).apply {
                    this.max = max
                    progress = value.coerceIn(0, max)
                    setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                        override fun onProgressChanged(sb: SeekBar?, p: Int, fromUser: Boolean) {
                            tv.text = "$label: $p%"
                            onChange(p)
                            pushLiveConfig()
                        }
                        override fun onStartTrackingTouch(sb: SeekBar?) {}
                        override fun onStopTrackingTouch(sb: SeekBar?) {}
                    })
                })
            }

            addCheck("Show touch controls", showTouchControls) { showTouchControls = it }
            addCheck("Hold to zoom", aimHold) { aimHold = it }
            addCheck("Gyro aiming while zooming", gyroEnabled) {
                gyroEnabled = it
                if (it) registerGyroIfEnabled() else unregisterGyro()
            }
            addSlider("Touch opacity", 100, (touchOpacity * 100f).roundToInt()) {
                touchOpacity = (it / 100f).coerceIn(0.2f, 1.0f)
            }
            addSlider("Look speed", 200, (lookSensitivity * 100f).roundToInt()) {
                lookSensitivity = (it / 100f).coerceIn(0.25f, 2.0f)
            }
            addSlider("Sound volume", 100, (soundVolume * 100f).roundToInt()) {
                soundVolume = (it / 100f).coerceIn(0.0f, 1.0f)
            }
            addCheck("Vibrate on hits", hapticsEnabled) { hapticsEnabled = it }
            addCheck("Vibrate when you are hit", hapticWhenHit) { hapticWhenHit = it }
            addCheck("Vibrate when you land a hit", hapticWhenDealing) { hapticWhenDealing = it }
            addSlider("Vibration strength", 100, hapticStrength) {
                hapticStrength = it.coerceIn(10, 100)
            }

            val scroll = ScrollView(this).apply { addView(container) }
            AlertDialog.Builder(this)
                .setTitle("Sound & Touch Controls")
                .setView(scroll)
                .setNeutralButton("Open keyboard") { _, _ -> showSoftKeyboardDialog() }
                .setNegativeButton("Exit and save report") { _, _ -> exitAndFinish() }
                .setPositiveButton("Resume") { _, _ -> applyImmersiveFullscreen() }
                .setOnCancelListener { applyImmersiveFullscreen() }
                .show()
        }
    }

    private fun pushLiveConfig() {
        if (R4Native.loadError != null) return
        R4Native.nativeSetTouchConfig(
            touchOpacity, buttonScale, lookSensitivity, aimHold, gyroEnabled, showTouchControls
        )
        R4Native.nativeSetAudioConfig(
            soundVolume, hapticsEnabled, hapticWhenHit, hapticWhenDealing, hapticStrength
        )
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

        pushLiveConfig()

        val hudListener = object : R4Native.HudListener {
            override fun onHudUpdate(text: String) {
                updateHud(text)
            }
            override fun onHapticPulse(durationMs: Int, amplitude: Int) {
                triggerHapticPulse(durationMs, amplitude)
            }
            override fun onUiAction(action: String) {
                when (action) {
                    "pause_drawer" -> showQuickDrawer()
                    "soft_keyboard" -> showSoftKeyboardDialog()
                }
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

    private fun pillDrawable(color: Int): GradientDrawable = GradientDrawable().apply {
        setColor(color)
        cornerRadius = dp(16).toFloat()
    }

    companion object {
        const val EXTRA_TREE_URI = "wse.recomp.tree_uri"
        const val EXTRA_COPY_DIR = "wse.recomp.copy_dir"
        const val EXTRA_WORK_DIR = "wse.recomp.work_dir"
        const val EXTRA_MODULE = "wse.recomp.module"
        const val EXTRA_JOBS = "wse.recomp.jobs"
        const val EXTRA_DRIVER_MODE = "wse.recomp.driver_mode"
        const val EXTRA_CUSTOM_ICD_PATH = "wse.recomp.custom_icd_path"
        const val EXTRA_NATIVE_LIB_DIR = "wse.recomp.native_lib_dir"
        const val EXTRA_SHADER_CACHE_DIR = "wse.recomp.shader_cache_dir"
        const val EXTRA_MAX_TEXTURE_SIZE = "wse.recomp.max_texture_size"
        const val EXTRA_RENDER_SCALE = "wse.recomp.render_scale"
        const val EXTRA_AUTO_RESOLUTION = "wse.recomp.auto_resolution"
        const val EXTRA_SHARPNESS = "wse.recomp.sharpness"
        const val EXTRA_FXAA = "wse.recomp.fxaa"
    }
}
