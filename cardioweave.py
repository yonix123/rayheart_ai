#!/usr/bin/env python3
"""
CardioWeave — Unified ECG Recorder + MI Detection
==================================================
Hardware : 2x AD8232 + ADS1115 + Raspberry Pi
Leads    : V1-V4 (chip 1), V3-V5 (chip 2)
AI       : Random Forest MI detector, inference every 60s
Alerts   : PyQtGraph display + WebSocket to phone browser

WIRING QUICK REFERENCE:
  ADS1115 SDA  → GPIO2  (Pin 3)
  ADS1115 SCL  → GPIO3  (Pin 5)
  ADS1115 VDD  → 3.3V   (Pin 1)
  ADS1115 GND  → GND    (Pin 6)
  ADS1115 ADDR → GND    (sets address 0x48)
  ADS1115 A0   → AD8232 chip 1 OUTPUT  (V1-V4)
  ADS1115 A1   → AD8232 chip 2 OUTPUT  (V3-V5)
  AD8232 LO+ chip1 → GPIO17 (Pin 11)
  AD8232 LO- chip1 → GPIO27 (Pin 13)
  AD8232 LO+ chip2 → GPIO22 (Pin 15)
  AD8232 LO- chip2 → GPIO23 (Pin 16)

INSTALL ON PI:
  pip install adafruit-circuitpython-ads1x15 RPi.GPIO
  pip install numpy scipy scikit-learn pyqtgraph PyQt5
  pip install websockets pyEDFlib

RUN:
  python cardioweave.py

PHONE BROWSER:
  Open http://<pi-ip-address>:8765 on any device on same WiFi
"""

import sys, os, time, threading, asyncio, datetime, json, pickle
import numpy as np
from collections import deque
from scipy.signal import butter, sosfilt, sosfilt_zi, iirnotch

# ── Display ────────────────────────────────────────────────────────────────
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui

# ── Storage ────────────────────────────────────────────────────────────────
import pyedflib

# ── Hardware (comment out when testing on Mac with mock mode) ──────────────
try:
    import board, busio, RPi.GPIO as GPIO
    import adafruit_ads1x15.ads1115 as ADS
    from adafruit_ads1x15.analog_in import AnalogIn
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False
    print("[WARN] Hardware libs not found — running in MOCK mode")

# ── WebSocket ──────────────────────────────────────────────────────────────
import websockets

# =============================================================================
#  CONFIGURATION
# =============================================================================

SAMPLE_RATE      = 500          # Hz
DISPLAY_WINDOW   = 5            # seconds shown on screen
DISPLAY_FPS      = 30           # screen refresh rate
MAINS_HZ         = 50           # notch filter frequency
INFERENCE_EVERY  = 60           # seconds between AI inference runs
ALERT_THRESHOLD  = 0.55        # from threshold optimizer
MODEL_DIR        = "/Users/rauanakendirbayeva/Desktop/rayheart_ai/models/"         # same folder as this script
WS_PORT          = 8765         # WebSocket port for phone browser
RECORDINGS_DIR   = os.path.expanduser("~/ecg_recordings")

LO_PINS = {"p1": 17, "n1": 27, "p2": 22, "n2": 23}
LEAD_NAMES  = ["chip1 (V1-V4)", "chip2 (V3-V5)"]
LEAD_COLORS = ["#00ff88", "#00ccff"]


# =============================================================================
#  SIGNAL PROCESSING
# =============================================================================

def make_filters(fs):
    sos_bp = butter(4, [0.5, 40.0], btype="bandpass", fs=fs, output="sos")
    b, a   = iirnotch(MAINS_HZ, Q=30, fs=fs)
    sos_n  = np.array([[b[0], b[1], b[2], 1.0, a[1], a[2]]])
    return sos_bp, sos_n


# =============================================================================
#  FEATURE EXTRACTION (must match training exactly)
# =============================================================================

def extract_features(chip1_sig, chip2_sig, fs):
    features = []
    for sig in [chip1_sig, chip2_sig]:
        features.extend([
            np.mean(sig), np.std(sig), np.min(sig), np.max(sig),
            np.median(sig), np.percentile(sig, 25), np.percentile(sig, 75),
            np.max(sig) - np.min(sig),
            np.mean(np.abs(sig - np.mean(sig))),
        ])
        fft_v  = np.abs(np.fft.rfft(sig))
        freqs  = np.fft.rfftfreq(len(sig), d=1.0 / fs)
        qrs_b  = fft_v[(freqs >= 5)  & (freqs <= 40)]
        st_b   = fft_v[(freqs >= 0.5) & (freqs <= 5)]
        ns_b   = fft_v[freqs > 40]
        features.extend([
            np.sum(qrs_b**2) if len(qrs_b) > 0 else 0,
            np.sum(st_b**2)  if len(st_b)  > 0 else 0,
            np.sum(ns_b**2)  if len(ns_b)  > 0 else 0,
            float(np.argmax(fft_v)),
        ])
        ms = fs / 1000.0
        pk = int(np.argmax(np.abs(sig)))
        ss = min(pk + int(60  * ms), len(sig) - 1)
        se = min(pk + int(120 * ms), len(sig))
        st = sig[ss:se]
        features.extend([
            np.mean(st) if len(st) > 0 else 0,
            np.std(st)  if len(st) > 0 else 0,
        ])
        features.extend([
            np.sum(sig**2) / len(sig),
            np.sum(np.diff(sig)**2) / len(sig),
            np.sum(np.abs(np.diff(sig))),
        ])
    return np.array(features, dtype=np.float64)


# =============================================================================
#  MI PREDICTOR
# =============================================================================

class MIPredictor:
    def __init__(self):
        self.model, self.scaler = self._load()
        print(f"[AI] Model ready. Threshold: {ALERT_THRESHOLD}")

    def _load(self):
        files = sorted([
            f for f in os.listdir(MODEL_DIR)
            if f.startswith("mi_model_") and f.endswith(".pkl")
        ])
        if not files:
            raise FileNotFoundError("No model found. Run cardioweave_train_complete.py first.")
        ts = files[-1].replace("mi_model_", "").replace(".pkl", "")
        with open(os.path.join(MODEL_DIR, f"mi_model_{ts}.pkl"), "rb") as f:
            model = pickle.load(f)
        with open(os.path.join(MODEL_DIR, f"scaler_{ts}.pkl"), "rb") as f:
            scaler = pickle.load(f)
        print(f"[AI] Loaded: mi_model_{ts}.pkl")
        return model, scaler

    def predict(self, chip1_buf, chip2_buf):
        if len(chip1_buf) < 100:
            return {"probability": 0.0, "alert": False, "message": "Not enough data"}
        c1 = np.array(chip1_buf, dtype=np.float64)
        c2 = np.array(chip2_buf, dtype=np.float64)
        feats = extract_features(c1, c2, SAMPLE_RATE).reshape(1, -1)
        feats_scaled = self.scaler.transform(feats)
        prob  = float(self.model.predict_proba(feats_scaled)[0, 1])
        alert = prob >= ALERT_THRESHOLD
        msg   = (f"MI SUSPECTED — {prob:.1%} — seek help immediately"
                 if alert else f"Normal — {prob:.1%}")
        return {"probability": prob, "alert": alert, "message": msg}


# =============================================================================
#  HARDWARE INTERFACE
# =============================================================================

class ECGHardware:
    def __init__(self):
        if not HARDWARE_AVAILABLE:
            raise RuntimeError("Hardware not available")
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.ads = ADS.ADS1115(self.i2c)
        self.ads.data_rate = 860
        self.ads.gain = ADS.Gain.ONE
        self.ch0 = AnalogIn(self.ads, ADS.P0)
        self.ch1 = AnalogIn(self.ads, ADS.P1)
        GPIO.setmode(GPIO.BCM)
        for pin in LO_PINS.values():
            GPIO.setup(pin, GPIO.IN)
        print("[HW] ADS1115 ready")

    def read(self):
        lo = any(GPIO.input(p) for p in LO_PINS.values())
        if lo:
            return None, True
        return (self.ch0.voltage * 1000.0,
                self.ch1.voltage * 1000.0), False

    def cleanup(self):
        GPIO.cleanup()


class MockHardware:
    """Fake hardware for testing on Mac — generates synthetic ECG."""
    def __init__(self):
        self._t = 0.0
        print("[MOCK] Using synthetic ECG data")

    def read(self):
        self._t += 1.0 / SAMPLE_RATE
        ecg  = 0.1 * np.sin(2 * np.pi * 1.2 * self._t)
        if self._t % 0.8 < 0.01:
            ecg += 1.5
        ecg += np.random.normal(0, 0.02)
        return (ecg, ecg * 0.85 + np.random.normal(0, 0.02)), False

    def cleanup(self):
        pass


# =============================================================================
#  ACQUISITION THREAD
# =============================================================================

class AcquisitionThread(threading.Thread):
    def __init__(self, hw):
        super().__init__(daemon=True)
        self.hw      = hw
        self.running = False
        self.lo_flag = False

        win = DISPLAY_WINDOW * SAMPLE_RATE
        self.disp_c1 = deque([0.0] * win, maxlen=win)
        self.disp_c2 = deque([0.0] * win, maxlen=win)

        # Inference buffers — 60 seconds
        inf_win = INFERENCE_EVERY * SAMPLE_RATE
        self.inf_c1 = deque(maxlen=inf_win)
        self.inf_c2 = deque(maxlen=inf_win)

        # Recording buffers
        self.rec_c1     = []
        self.rec_c2     = []
        self.recording  = False
        self.rec_start  = None

        # Filters
        sos_bp, sos_n = make_filters(SAMPLE_RATE)
        self.sos_bp, self.sos_n = sos_bp, sos_n
        self.zi_bp = [sosfilt_zi(sos_bp) * 0 for _ in range(2)]
        self.zi_n  = [sosfilt_zi(sos_n)  * 0 for _ in range(2)]

        self._interval = 1.0 / SAMPLE_RATE
        self.lock = threading.Lock()

    def run(self):
        self.running = True
        # Warm up filters
        for _ in range(200):
            self._process(0.0, 0.0)

        while self.running:
            t0 = time.perf_counter()
            raw, lo = self.hw.read()
            if raw is None or lo:
                self.lo_flag = True
                with self.lock:
                    self.disp_c1.append(0.0)
                    self.disp_c2.append(0.0)
            else:
                self.lo_flag = False
                self._process(raw[0], raw[1])
            elapsed = time.perf_counter() - t0
            time.sleep(max(0, self._interval - elapsed))

    def _process(self, v1, v2):
        s1, self.zi_bp[0] = sosfilt(self.sos_bp, [v1], zi=self.zi_bp[0])
        s1, self.zi_n[0]  = sosfilt(self.sos_n,  s1,   zi=self.zi_n[0])
        s2, self.zi_bp[1] = sosfilt(self.sos_bp, [v2], zi=self.zi_bp[1])
        s2, self.zi_n[1]  = sosfilt(self.sos_n,  s2,   zi=self.zi_n[1])
        f1, f2 = float(s1[0]), float(s2[0])
        with self.lock:
            self.disp_c1.append(f1)
            self.disp_c2.append(f2)
            self.inf_c1.append(f1)
            self.inf_c2.append(f2)
            if self.recording:
                self.rec_c1.append(f1)
                self.rec_c2.append(f2)

    def get_display(self):
        with self.lock:
            return list(self.disp_c1), list(self.disp_c2)

    def get_inference_window(self):
        with self.lock:
            return list(self.inf_c1), list(self.inf_c2)

    def start_recording(self):
        with self.lock:
            self.rec_c1, self.rec_c2 = [], []
            self.rec_start = datetime.datetime.now()
            self.recording = True

    def stop_recording(self):
        with self.lock:
            self.recording = False
            return list(self.rec_c1), list(self.rec_c2), self.rec_start

    def stop(self):
        self.running = False


# =============================================================================
#  EDF STORAGE
# =============================================================================

def save_edf(c1, c2, start_time):
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    ts   = start_time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RECORDINGS_DIR, f"ecg_{ts}.edf")
    f    = pyedflib.EdfWriter(path, 2, file_type=pyedflib.FILETYPE_EDFPLUS)
    f.setStartdatetime(start_time)
    for i, (name, data) in enumerate(zip(["chip1_V1V4", "chip2_V3V5"], [c1, c2])):
        sig = np.array(data)
        mx  = float(np.max(np.abs(sig)) + 0.1) if len(sig) else 5.0
        f.setSignalHeader(i, {
            "label": name, "dimension": "mV",
            "sample_frequency": SAMPLE_RATE,
            "physical_max": mx, "physical_min": -mx,
            "digital_max": 32767, "digital_min": -32768,
            "transducer": "AD8232+ADS1115",
            "prefilter": "HP:0.5Hz LP:40Hz N:50Hz",
        })
    f.writeSamples([np.array(c1, dtype=np.float64),
                    np.array(c2, dtype=np.float64)])
    f.close()
    print(f"[EDF] Saved: {path}")
    return path


# =============================================================================
#  WEBSOCKET SERVER  (runs in background asyncio loop)
# =============================================================================

class WSServer:
    def __init__(self):
        self.clients  = set()
        self.loop     = asyncio.new_event_loop()
        self._thread  = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._serve())

    async def _serve(self):
        async with websockets.serve(self._handler, "0.0.0.0", WS_PORT):
            print(f"[WS] Server running on port {WS_PORT}")
            await asyncio.Future()  # run forever

    async def _handler(self, ws):
        self.clients.add(ws)
        # Send the phone browser the HTML dashboard on connect
        try:
            await ws.send(json.dumps({"type": "init", "threshold": ALERT_THRESHOLD}))
            await ws.wait_closed()
        finally:
            self.clients.discard(ws)

    def broadcast(self, payload: dict):
        """Send a message to all connected phone browsers."""
        if not self.clients:
            return
        msg = json.dumps(payload)
        asyncio.run_coroutine_threadsafe(self._broadcast(msg), self.loop)

    async def _broadcast(self, msg):
        dead = set()
        for ws in self.clients:
            try:
                await ws.send(msg)
            except Exception:
                dead.add(ws)
        self.clients -= dead


# =============================================================================
#  INFERENCE THREAD  (runs every 60 seconds)
# =============================================================================

class InferenceThread(threading.Thread):
    def __init__(self, acq: AcquisitionThread,
                 predictor: MIPredictor, ws: WSServer):
        super().__init__(daemon=True)
        self.acq       = acq
        self.predictor = predictor
        self.ws        = ws
        self.running   = False
        self.last_result = None
        self.lock      = threading.Lock()

    def run(self):
        self.running = True
        while self.running:
            time.sleep(INFERENCE_EVERY)
            c1, c2 = self.acq.get_inference_window()
            result  = self.predictor.predict(c1, c2)
            with self.lock:
                self.last_result = result
            print(f"[AI] {result['message']}")
            # Push to phone
            self.ws.broadcast({
                "type":        "inference",
                "probability": round(result["probability"], 4),
                "alert":       result["alert"],
                "message":     result["message"],
                "timestamp":   datetime.datetime.now().isoformat(),
            })

    def get_result(self):
        with self.lock:
            return self.last_result

    def stop(self):
        self.running = False


# =============================================================================
#  DISPLAY
# =============================================================================

class ECGDisplay:
    def __init__(self, acq: AcquisitionThread,
                 inference: InferenceThread, ws: WSServer):
        self.acq       = acq
        self.inference = inference
        self.ws        = ws
        self.is_recording = False

        self.app = QtWidgets.QApplication(sys.argv)
        self.app.setStyle("Fusion")
        p = QtGui.QPalette()
        p.setColor(QtGui.QPalette.Window, QtGui.QColor(10, 14, 14))
        p.setColor(QtGui.QPalette.WindowText, QtGui.QColor(200, 220, 220))
        self.app.setPalette(p)

        self.win = QtWidgets.QMainWindow()
        self.win.setWindowTitle("CardioWeave — Live ECG")
        self.win.resize(1280, 700)

        central = QtWidgets.QWidget()
        self.win.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # ── Status bar ────────────────────────────────────────────────────
        top = QtWidgets.QHBoxLayout()

        self.lbl_status = QtWidgets.QLabel("● LIVE")
        self.lbl_status.setStyleSheet(
            "color:#00ff88; font-size:13px; font-weight:bold;")

        self.lbl_lo = QtWidgets.QLabel("")
        self.lbl_lo.setStyleSheet(
            "color:#ff4444; font-size:12px; font-weight:bold;")

        self.lbl_ai = QtWidgets.QLabel("AI: waiting for first window...")
        self.lbl_ai.setStyleSheet(
            "color:#888888; font-size:13px;")

        self.lbl_next = QtWidgets.QLabel("")
        self.lbl_next.setStyleSheet(
            "color:#888888; font-size:11px;")

        top.addWidget(self.lbl_status)
        top.addStretch()
        top.addWidget(self.lbl_lo)
        top.addStretch()
        top.addWidget(self.lbl_ai)
        top.addSpacing(16)
        top.addWidget(self.lbl_next)
        layout.addLayout(top)

        # ── Alert banner (hidden by default) ──────────────────────────────
        self.alert_banner = QtWidgets.QLabel("")
        self.alert_banner.setAlignment(QtCore.Qt.AlignCenter)
        self.alert_banner.setStyleSheet(
            "background:#3a0000; color:#ff4444; font-size:16px; "
            "font-weight:bold; padding:8px; border-radius:4px;")
        self.alert_banner.hide()
        layout.addWidget(self.alert_banner)

        # ── ECG plots ─────────────────────────────────────────────────────
        self.gw = pg.GraphicsLayoutWidget()
        self.gw.setBackground("#0a0e0e")
        layout.addWidget(self.gw, stretch=1)

        x = np.arange(DISPLAY_WINDOW * SAMPLE_RATE)
        self.curves = []
        for i, (name, color) in enumerate(zip(LEAD_NAMES, LEAD_COLORS)):
            p = self.gw.addPlot(row=i, col=0)
            p.setYRange(-2.0, 2.0)
            p.setXRange(0, DISPLAY_WINDOW * SAMPLE_RATE)
            p.hideAxis("bottom")
            p.setLabel("left", name, color=color, size="10pt")
            p.getAxis("left").setWidth(80)
            p.showGrid(x=True, y=True, alpha=0.12)
            p.addLine(y=0, pen=pg.mkPen("#334444", width=0.5))
            curve = p.plot(x, np.zeros(len(x)),
                           pen=pg.mkPen(color, width=1.2))
            self.curves.append(curve)

        # ── Probability bar ───────────────────────────────────────────────
        prob_row = QtWidgets.QHBoxLayout()
        prob_row.addWidget(QtWidgets.QLabel("MI probability:"))
        self.prob_bar = QtWidgets.QProgressBar()
        self.prob_bar.setRange(0, 100)
        self.prob_bar.setValue(0)
        self.prob_bar.setFixedHeight(20)
        self.prob_bar.setStyleSheet(
            "QProgressBar { border:1px solid #334; border-radius:3px; }"
            "QProgressBar::chunk { background:#00ff88; border-radius:3px; }"
        )
        self.lbl_prob = QtWidgets.QLabel("0.0%")
        self.lbl_prob.setStyleSheet("color:#00ff88; font-weight:bold; min-width:50px;")
        prob_row.addWidget(self.prob_bar, stretch=1)
        prob_row.addWidget(self.lbl_prob)
        layout.addLayout(prob_row)

        # ── Buttons ───────────────────────────────────────────────────────
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_rec = QtWidgets.QPushButton("⏺  START RECORDING")
        self.btn_rec.setFixedHeight(40)
        self.btn_rec.setStyleSheet(
            "QPushButton{background:#1a3a1a;color:#00ff88;"
            "border:1px solid #00ff88;border-radius:4px;"
            "font-size:13px;font-weight:bold;}"
            "QPushButton:hover{background:#1f4f1f;}")
        self.btn_rec.clicked.connect(self.toggle_recording)

        self.lbl_saved = QtWidgets.QLabel("")
        self.lbl_saved.setStyleSheet("color:#888; font-size:10px;")

        btn_quit = QtWidgets.QPushButton("✕  QUIT")
        btn_quit.setFixedHeight(40)
        btn_quit.setFixedWidth(100)
        btn_quit.setStyleSheet(
            "QPushButton{background:#2a0a0a;color:#ff4444;"
            "border:1px solid #ff4444;border-radius:4px;font-size:13px;}"
            "QPushButton:hover{background:#3a0f0f;}")
        btn_quit.clicked.connect(self.quit)

        btn_row.addWidget(self.btn_rec)
        btn_row.addWidget(self.lbl_saved)
        btn_row.addStretch()
        btn_row.addWidget(btn_quit)
        layout.addLayout(btn_row)

        # ── Timers ────────────────────────────────────────────────────────
        self.disp_timer = QtCore.QTimer()
        self.disp_timer.timeout.connect(self.update_display)
        self.disp_timer.start(1000 // DISPLAY_FPS)

        self.ai_timer = QtCore.QTimer()
        self.ai_timer.timeout.connect(self.update_ai_panel)
        self.ai_timer.start(2000)   # check AI result every 2s

        self._next_inference = time.time() + INFERENCE_EVERY
        self.rec_start_wall = None

    def update_display(self):
        c1, c2 = self.acq.get_display()
        self.curves[0].setData(np.array(c1))
        self.curves[1].setData(np.array(c2))

        if self.acq.lo_flag:
            self.lbl_lo.setText("⚠  ELECTRODE OFF")
            self.lbl_status.setText("● LEAD OFF")
            self.lbl_status.setStyleSheet(
                "color:#ff4444;font-size:13px;font-weight:bold;")
        else:
            self.lbl_lo.setText("")
            if self.is_recording:
                elapsed = time.time() - self.rec_start_wall
                m, s = divmod(int(elapsed), 60)
                self.lbl_status.setText(f"● REC  {m:02d}:{s:02d}")
                self.lbl_status.setStyleSheet(
                    "color:#ff4444;font-size:13px;font-weight:bold;")
            else:
                self.lbl_status.setText("● LIVE")
                self.lbl_status.setStyleSheet(
                    "color:#00ff88;font-size:13px;font-weight:bold;")

        secs_left = max(0, int(self._next_inference - time.time()))
        self.lbl_next.setText(f"next AI check in {secs_left}s")

    def update_ai_panel(self):
        result = self.inference.get_result()
        if result is None:
            return

        prob    = result["probability"]
        alert   = result["alert"]
        pct     = int(prob * 100)

        self.prob_bar.setValue(pct)
        self.lbl_prob.setText(f"{prob:.1%}")

        if alert:
            self.prob_bar.setStyleSheet(
                "QProgressBar{border:1px solid #334;border-radius:3px;}"
                "QProgressBar::chunk{background:#ff4444;border-radius:3px;}")
            self.lbl_prob.setStyleSheet("color:#ff4444;font-weight:bold;min-width:50px;")
            self.lbl_ai.setText(f"AI ALERT — {prob:.1%} MI probability")
            self.lbl_ai.setStyleSheet("color:#ff4444;font-size:13px;font-weight:bold;")
            self.alert_banner.setText(
                f"MI SUSPECTED — {prob:.1%}   |   Seek medical attention immediately")
            self.alert_banner.show()
        else:
            self.prob_bar.setStyleSheet(
                "QProgressBar{border:1px solid #334;border-radius:3px;}"
                "QProgressBar::chunk{background:#00ff88;border-radius:3px;}")
            self.lbl_prob.setStyleSheet("color:#00ff88;font-weight:bold;min-width:50px;")
            self.lbl_ai.setText(f"AI: Normal — {prob:.1%}")
            self.lbl_ai.setStyleSheet("color:#00ff88;font-size:13px;")
            self.alert_banner.hide()

        self._next_inference = time.time() + INFERENCE_EVERY

    def toggle_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.rec_start_wall = time.time()
            self.acq.start_recording()
            self.btn_rec.setText("⏹  STOP & SAVE")
            self.btn_rec.setStyleSheet(
                "QPushButton{background:#3a0a0a;color:#ff4444;"
                "border:1px solid #ff4444;border-radius:4px;"
                "font-size:13px;font-weight:bold;}")
        else:
            self.is_recording = False
            c1, c2, start = self.acq.stop_recording()
            self.btn_rec.setText("⏺  START RECORDING")
            self.btn_rec.setStyleSheet(
                "QPushButton{background:#1a3a1a;color:#00ff88;"
                "border:1px solid #00ff88;border-radius:4px;"
                "font-size:13px;font-weight:bold;}")
            def _save():
                path = save_edf(c1, c2, start)
                self.lbl_saved.setText(f"Saved: {os.path.basename(path)}")
            threading.Thread(target=_save, daemon=True).start()

    def quit(self):
        self.disp_timer.stop()
        self.ai_timer.stop()
        self.acq.stop()
        self.inference.stop()
        self.app.quit()

    def run(self):
        self.win.show()
        sys.exit(self.app.exec_())


# =============================================================================
#  PHONE BROWSER DASHBOARD
#  Served as a static HTML page over a simple HTTP server on port 8766
# =============================================================================

PHONE_HTML = """<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CardioWeave</title>
<style>
  body{margin:0;background:#0a0e0e;color:#cce;font-family:monospace;
       display:flex;flex-direction:column;align-items:center;padding:20px;}
  h2{color:#00ff88;margin:0 0 16px;}
  #status{font-size:14px;color:#888;margin-bottom:20px;}
  #prob-box{width:90%;max-width:400px;background:#111;border-radius:8px;padding:20px;
            text-align:center;border:1px solid #334;}
  #prob-num{font-size:48px;font-weight:bold;color:#00ff88;margin:10px 0;}
  #prob-bar-wrap{background:#1a1a1a;border-radius:4px;height:16px;margin:10px 0;}
  #prob-bar{height:16px;border-radius:4px;background:#00ff88;width:0%;
            transition:width 0.5s,background 0.5s;}
  #message{margin-top:16px;font-size:14px;color:#aaa;min-height:40px;}
  #alert-box{display:none;margin-top:20px;padding:16px;border-radius:8px;
             background:#3a0000;border:2px solid #ff4444;
             color:#ff4444;font-size:16px;font-weight:bold;text-align:center;
             width:90%;max-width:400px;}
  #ts{margin-top:12px;font-size:11px;color:#555;}
  #threshold{font-size:12px;color:#666;margin-top:8px;}
</style>
</head>
<body>
<h2>CardioWeave</h2>
<div id="status">Connecting...</div>
<div id="prob-box">
  <div>MI Probability</div>
  <div id="prob-num">--%</div>
  <div id="prob-bar-wrap"><div id="prob-bar"></div></div>
  <div id="threshold"></div>
  <div id="message">Waiting for first inference window...</div>
  <div id="ts"></div>
</div>
<div id="alert-box">MI SUSPECTED — Seek medical attention immediately</div>

<script>
const WS_URL = `ws://${location.hostname}:8765`;
let ws;

function connect() {
  ws = new WebSocket(WS_URL);
  ws.onopen  = () => document.getElementById('status').textContent = 'Connected';
  ws.onclose = () => { document.getElementById('status').textContent = 'Reconnecting...';
                        setTimeout(connect, 2000); };
  ws.onerror = () => ws.close();
  ws.onmessage = e => {
    const d = JSON.parse(e.data);
    if (d.type === 'init') {
      document.getElementById('threshold').textContent =
        `Alert threshold: ${(d.threshold*100).toFixed(1)}%`;
    }
    if (d.type === 'inference') {
      const pct = Math.round(d.probability * 100);
      document.getElementById('prob-num').textContent  = pct + '%';
      document.getElementById('prob-bar').style.width  = pct + '%';
      document.getElementById('prob-bar').style.background =
        d.alert ? '#ff4444' : '#00ff88';
      document.getElementById('prob-num').style.color =
        d.alert ? '#ff4444' : '#00ff88';
      document.getElementById('message').textContent   = d.message;
      document.getElementById('ts').textContent        =
        'Last update: ' + new Date(d.timestamp).toLocaleTimeString();
      document.getElementById('alert-box').style.display =
        d.alert ? 'block' : 'none';
      if (d.alert) {
        try { new Audio('data:audio/wav;base64,UklGRl9vT19XQVZFZm10IBAAAA').play(); }
        catch(e) {}
      }
    }
  };
}
connect();
</script>
</body>
</html>"""


def serve_phone_dashboard():
    """Serve the phone HTML dashboard on port 8766."""
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(PHONE_HTML.encode())
        def log_message(self, *args):
            pass  # silence HTTP logs

    server = HTTPServer(("0.0.0.0", 8766), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print("[HTTP] Phone dashboard on port 8766")


# =============================================================================
#  MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("  CardioWeave — ECG Recorder + MI Detection")
    print(f"  Sample rate    : {SAMPLE_RATE} SPS")
    print(f"  Inference every: {INFERENCE_EVERY}s")
    print(f"  Alert threshold: {ALERT_THRESHOLD}")
    print(f"  Recordings     : {RECORDINGS_DIR}")
    print("=" * 60)

    # Hardware
    if HARDWARE_AVAILABLE:
        try:
            hw = ECGHardware()
        except Exception as e:
            print(f"[WARN] Hardware init failed ({e}) — switching to mock")
            hw = MockHardware()
    else:
        hw = MockHardware()

    # AI model
    try:
        predictor = MIPredictor()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # WebSocket server
    ws = WSServer()
    ws.start()

    # Phone dashboard HTTP server
    serve_phone_dashboard()
    print(f"[INFO] Open on phone: http://<pi-ip>:8766")

    # Acquisition thread
    acq = AcquisitionThread(hw)
    acq.start()

    # Inference thread
    inf = InferenceThread(acq, predictor, ws)
    inf.start()

    # Display (blocks until quit)
    try:
        display = ECGDisplay(acq, inf, ws)
        display.run()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted")
    finally:
        acq.stop()
        inf.stop()
        hw.cleanup()
        print("[INFO] Shutdown complete")


if __name__ == "__main__":
    main()
