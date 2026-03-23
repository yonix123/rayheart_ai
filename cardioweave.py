#!/usr/bin/env python3
"""
CardioWeave — ECG + SpO2 + Heart Rate
======================================
Hardware : 2x AD8232 + ADS1115 + MAX30102 + Raspberry Pi
Displays : ECG waveform, HR, SpO2, MI probability
Alerts   : Matplotlib + WebSocket to phone browser

INSTALL:
  pip install adafruit-circuitpython-ads1x15 RPi.GPIO
  pip install numpy scipy scikit-learn smbus2
  pip install matplotlib websockets pyedflib
"""

import sys, os, time, threading, asyncio, datetime, json, pickle
import numpy as np
from collections import deque
from scipy.signal import butter, sosfilt, sosfilt_zi, iirnotch

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

import pyedflib

try:
    import board, busio, RPi.GPIO as GPIO
    import adafruit_ads1x15.ads1115 as ADS
    from adafruit_ads1x15.analog_in import AnalogIn
    import smbus2
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False
    print("[WARN] Hardware libs not found — running in MOCK mode")

import websockets

# =============================================================================
#  CONFIGURATION
# =============================================================================

SAMPLE_RATE     = 500
DISPLAY_WINDOW  = 5
DISPLAY_FPS     = 20
MAINS_HZ        = 50
INFERENCE_EVERY = 60
ALERT_THRESHOLD = 0.55
MODEL_DIR       = "./models/"
WS_PORT         = 8765
RECORDINGS_DIR  = os.path.expanduser("~/ecg_recordings")

LO_PINS     = {"p1": 17, "n1": 27, "p2": 22, "n2": 23}
LEAD_NAMES  = ["chip1 (V1-V4)", "chip2 (V3-V5)"]
LEAD_COLORS = ["#00ff88", "#00ccff"]

MAX30102_ADDR     = 0x57
REG_INTR_ENABLE_1 = 0x02
REG_FIFO_WR_PTR   = 0x04
REG_FIFO_RD_PTR   = 0x06
REG_FIFO_DATA     = 0x07
REG_MODE_CONFIG   = 0x09
REG_SPO2_CONFIG   = 0x0A
REG_LED1_PA       = 0x0C
REG_LED2_PA       = 0x0D


# =============================================================================
#  SIGNAL PROCESSING
# =============================================================================

def make_filters(fs):
    sos_bp = butter(4, [0.5, 40.0], btype="bandpass", fs=fs, output="sos")
    b, a   = iirnotch(MAINS_HZ, Q=30, fs=fs)
    sos_n  = np.array([[b[0], b[1], b[2], 1.0, a[1], a[2]]])
    return sos_bp, sos_n


# =============================================================================
#  FEATURE EXTRACTION
# =============================================================================

def extract_features(chip1_sig, chip2_sig, fs=100):
    features = []
    for sig in [chip1_sig, chip2_sig]:
        features.extend([
            np.mean(sig), np.std(sig), np.min(sig), np.max(sig),
            np.median(sig), np.percentile(sig, 25), np.percentile(sig, 75),
            np.max(sig) - np.min(sig),
            np.mean(np.abs(sig - np.mean(sig))),
        ])
        fft_v = np.abs(np.fft.rfft(sig))
        freqs = np.fft.rfftfreq(len(sig), d=1.0 / 100)
        qrs_b = fft_v[(freqs >= 5)  & (freqs <= 40)]
        st_b  = fft_v[(freqs >= 0.5) & (freqs <= 5)]
        ns_b  = fft_v[freqs > 40]
        features.extend([
            np.sum(qrs_b**2), np.sum(st_b**2), np.sum(ns_b**2),
            np.argmax(fft_v),
        ])
        pk = int(np.argmax(np.abs(sig)))
        ss = min(pk + 6,  len(sig) - 1)
        se = min(pk + 12, len(sig))
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
            raise FileNotFoundError("No model found.")
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
        from scipy.signal import resample as sp_resample
        c1 = np.array(chip1_buf, dtype=np.float64)
        c2 = np.array(chip2_buf, dtype=np.float64)
        target       = int(len(c1) * 100 / SAMPLE_RATE)
        c1           = sp_resample(c1, target)
        c2           = sp_resample(c2, target)
        feats        = extract_features(c1, c2).reshape(1, -1)
        feats_scaled = self.scaler.transform(feats)
        prob         = float(self.model.predict_proba(feats_scaled)[0, 1])
        alert        = prob >= ALERT_THRESHOLD
        msg          = (f"MI SUSPECTED — {prob:.1%} — seek help immediately"
                        if alert else f"Normal — {prob:.1%}")
        return {"probability": prob, "alert": alert, "message": msg}


# =============================================================================
#  MAX30102 DRIVER
# =============================================================================

class MAX30102:
    def __init__(self, bus_num=1):
        self.bus = smbus2.SMBus(bus_num)
        self._init_sensor()
        self._ir_buf  = deque(maxlen=200)
        self._red_buf = deque(maxlen=200)
        self.hr   = 0
        self.spo2 = 0
        print("[MAX] MAX30102 initialized")

    def _write(self, reg, val):
        self.bus.write_byte_data(MAX30102_ADDR, reg, val)

    def _init_sensor(self):
        self._write(REG_MODE_CONFIG,   0x40)
        time.sleep(0.1)
        self._write(REG_INTR_ENABLE_1, 0xC0)
        self._write(REG_FIFO_WR_PTR,   0x00)
        self._write(REG_FIFO_RD_PTR,   0x00)
        self._write(REG_MODE_CONFIG,   0x03)
        self._write(REG_SPO2_CONFIG,   0x27)
        self._write(REG_LED1_PA,       0x24)
        self._write(REG_LED2_PA,       0x24)

    def read_fifo(self):
        try:
            raw = self.bus.read_i2c_block_data(MAX30102_ADDR, REG_FIFO_DATA, 6)
            red = (raw[0] << 16 | raw[1] << 8 | raw[2]) & 0x3FFFF
            ir  = (raw[3] << 16 | raw[4] << 8 | raw[5]) & 0x3FFFF
            return red, ir
        except Exception:
            return None

    def update(self):
        sample = self.read_fifo()
        if sample is None:
            return
        red, ir = sample
        self._ir_buf.append(ir)
        self._red_buf.append(red)
        if len(self._ir_buf) < 100:
            return
        ir_arr  = np.array(self._ir_buf,  dtype=np.float64)
        red_arr = np.array(self._red_buf, dtype=np.float64)
        ir_norm = ir_arr - np.mean(ir_arr)
        peaks   = self._find_peaks(ir_norm)
        if len(peaks) >= 2:
            avg_interval = np.mean(np.diff(peaks))
            bpm = int(60.0 / (avg_interval / 100.0))
            self.hr = max(30, min(250, bpm))
        ir_ac  = np.std(ir_arr)
        red_ac = np.std(red_arr)
        ir_dc  = np.mean(ir_arr)
        red_dc = np.mean(red_arr)
        if ir_ac > 0 and ir_dc > 0 and red_dc > 0:
            R = (red_ac / red_dc) / (ir_ac / ir_dc)
            spo2 = 110.0 - 25.0 * R
            self.spo2 = int(max(80, min(100, spo2)))

    def _find_peaks(self, sig, min_distance=30):
        peaks = []
        for i in range(1, len(sig) - 1):
            if (sig[i] > sig[i-1] and sig[i] > sig[i+1]
                    and sig[i] > 0.3 * np.max(sig)):
                if not peaks or (i - peaks[-1]) >= min_distance:
                    peaks.append(i)
        return peaks

    def cleanup(self):
        try:
            self._write(REG_MODE_CONFIG, 0x80)
            self.bus.close()
        except Exception:
            pass


class MockMAX30102:
    def __init__(self):
        self._t   = 0.0
        self.hr   = 72
        self.spo2 = 98
        print("[MOCK] Using synthetic HR/SpO2 data")

    def update(self):
        self._t  += 0.1
        self.hr   = int(72 + 3 * np.sin(self._t * 0.1))
        self.spo2 = int(max(95, min(100, 98 + np.random.randint(-1, 2))))

    def cleanup(self):
        pass


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
        self.ads.gain = 1
        self.ch0 = AnalogIn(self.ads, 2)
        self.ch1 = AnalogIn(self.ads, 3)
        GPIO.setmode(GPIO.BCM)
        for pin in LO_PINS.values():
            GPIO.setup(pin, GPIO.IN)
        print("[HW] ADS1115 ready")

    def read(self):
        lo = any(GPIO.input(p) for p in LO_PINS.values())
        if lo:
            return None, True
        v1 = self.ch0.voltage * 1000.0
        v2 = self.ch1.voltage * 1000.0
        return (v1, v2), False

    def cleanup(self):
        GPIO.cleanup()


class MockHardware:
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
#  MAX30102 THREAD
# =============================================================================

class MAX30102Thread(threading.Thread):
    def __init__(self, sensor):
        super().__init__(daemon=True)
        self.sensor  = sensor
        self.running = False
        self.lock    = threading.Lock()

    def run(self):
        self.running = True
        while self.running:
            self.sensor.update()
            time.sleep(0.01)

    def get_vitals(self):
        with self.lock:
            return self.sensor.hr, self.sensor.spo2

    def stop(self):
        self.running = False


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

        inf_win = INFERENCE_EVERY * SAMPLE_RATE
        self.inf_c1 = deque(maxlen=inf_win)
        self.inf_c2 = deque(maxlen=inf_win)

        self.rec_c1    = []
        self.rec_c2    = []
        self.recording = False
        self.rec_start = None

        sos_bp, sos_n = make_filters(SAMPLE_RATE)
        self.sos_bp, self.sos_n = sos_bp, sos_n
        self.zi_bp = [sosfilt_zi(sos_bp) * 0 for _ in range(2)]
        self.zi_n  = [sosfilt_zi(sos_n)  * 0 for _ in range(2)]

        self._interval = 1.0 / SAMPLE_RATE
        self.lock = threading.Lock()

    def run(self):
        self.running = True
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
        v1 = v1 - 550.0
        v2 = v2 - 500.0
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
#  WEBSOCKET SERVER
# =============================================================================

class WSServer:
    def __init__(self):
        self.clients = set()
        self.loop    = asyncio.new_event_loop()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._serve())

    async def _serve(self):
        async with websockets.serve(self._handler, "0.0.0.0", WS_PORT):
            print(f"[WS] Server on port {WS_PORT}")
            await asyncio.Future()

    async def _handler(self, ws):
        self.clients.add(ws)
        try:
            await ws.send(json.dumps({"type": "init", "threshold": ALERT_THRESHOLD}))
            await ws.wait_closed()
        finally:
            self.clients.discard(ws)

    def broadcast(self, payload: dict):
        if not self.clients:
            return
        msg = json.dumps(payload)
        asyncio.run_coroutine_threadsafe(self._bcast(msg), self.loop)

    async def _bcast(self, msg):
        dead = set()
        for ws in self.clients:
            try:
                await ws.send(msg)
            except Exception:
                dead.add(ws)
        self.clients -= dead


# =============================================================================
#  INFERENCE THREAD
# =============================================================================

class InferenceThread(threading.Thread):
    def __init__(self, acq, predictor, ws, max_thread):
        super().__init__(daemon=True)
        self.acq         = acq
        self.predictor   = predictor
        self.ws          = ws
        self.max_thread  = max_thread
        self.running     = False
        self.last_result = None
        self.lock        = threading.Lock()

    def run(self):
        self.running = True
        while self.running:
            time.sleep(INFERENCE_EVERY)
            c1, c2   = self.acq.get_inference_window()
            result   = self.predictor.predict(c1, c2)
            hr, spo2 = self.max_thread.get_vitals()
            with self.lock:
                self.last_result = {**result, "hr": hr, "spo2": spo2}
            print(f"[AI] {result['message']}  HR:{hr}bpm  SpO2:{spo2}%")
            self.ws.broadcast({
                "type":        "inference",
                "probability": round(result["probability"], 4),
                "alert":       result["alert"],
                "message":     result["message"],
                "hr":          hr,
                "spo2":        spo2,
                "timestamp":   datetime.datetime.now().isoformat(),
            })

    def get_result(self):
        with self.lock:
            return self.last_result

    def stop(self):
        self.running = False


# =============================================================================
#  VITALS BROADCAST THREAD — sends HR/SpO2 to phone every 2 seconds
# =============================================================================

class VitalsBroadcastThread(threading.Thread):
    def __init__(self, max_thread, ws):
        super().__init__(daemon=True)
        self.max_thread = max_thread
        self.ws         = ws
        self.running    = False

    def run(self):
        self.running = True
        while self.running:
            time.sleep(2)
            hr, spo2 = self.max_thread.get_vitals()
            self.ws.broadcast({
                "type":      "vitals",
                "hr":        hr,
                "spo2":      spo2,
                "timestamp": datetime.datetime.now().isoformat(),
            })

    def stop(self):
        self.running = False


# =============================================================================
#  DISPLAY
# =============================================================================

class ECGDisplay:
    def __init__(self, acq, inference, ws, max_thread):
        self.acq          = acq
        self.inference    = inference
        self.ws           = ws
        self.max_thread   = max_thread
        self.is_recording = False
        self._next_inf    = time.time() + INFERENCE_EVERY
        self._rec_wall    = 0.0

        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(13, 8), facecolor="#0a0e0e")
        self.fig.canvas.manager.set_window_title("CardioWeave")

        gs = GridSpec(4, 2, figure=self.fig,
                      height_ratios=[2, 2, 1, 1],
                      hspace=0.45, wspace=0.3,
                      top=0.92, bottom=0.07,
                      left=0.08, right=0.97)

        n = DISPLAY_WINDOW * SAMPLE_RATE
        x = np.arange(n) / SAMPLE_RATE

        self.axes  = []
        self.lines = []
        colors = ["#00ff88", "#00ccff"]
        for i in range(2):
            ax = self.fig.add_subplot(gs[i, :])
            ax.set_facecolor("#0a0e0e")
            ax.set_xlim(0, DISPLAY_WINDOW)
            ax.set_ylim(-50, 50)
            ax.set_ylabel(LEAD_NAMES[i], color=colors[i], fontsize=9)
            ax.tick_params(colors="#445555", labelsize=7)
            ax.grid(True, color="#1a2a2a", linewidth=0.5)
            ax.axhline(0, color="#334444", linewidth=0.5)
            for sp in ax.spines.values():
                sp.set_color("#223333")
            line, = ax.plot(x, np.zeros(n), color=colors[i],
                            linewidth=0.9, antialiased=True)
            self.axes.append(ax)
            self.lines.append(line)

        # HR panel
        ax_hr = self.fig.add_subplot(gs[2, 0])
        ax_hr.set_facecolor("#0a0e0e")
        ax_hr.axis("off")
        self.txt_hr = ax_hr.text(0.5, 0.5, "HR\n--",
                                  color="#ff6644", fontsize=22,
                                  fontweight="bold", ha="center",
                                  va="center", transform=ax_hr.transAxes)

        # SpO2 panel
        ax_spo2 = self.fig.add_subplot(gs[2, 1])
        ax_spo2.set_facecolor("#0a0e0e")
        ax_spo2.axis("off")
        self.txt_spo2 = ax_spo2.text(0.5, 0.5, "SpO2\n--%",
                                      color="#00aaff", fontsize=22,
                                      fontweight="bold", ha="center",
                                      va="center", transform=ax_spo2.transAxes)

        # Status / AI panel
        ax_info = self.fig.add_subplot(gs[3, :])
        ax_info.set_facecolor("#0a0e0e")
        ax_info.axis("off")

        self.txt_status = ax_info.text(0.01, 0.75, "● LIVE",
                                        color="#00ff88", fontsize=12,
                                        fontweight="bold",
                                        transform=ax_info.transAxes)
        self.txt_lo = ax_info.text(0.18, 0.75, "", color="#ff4444",
                                    fontsize=11, fontweight="bold",
                                    transform=ax_info.transAxes)
        self.txt_ai = ax_info.text(0.01, 0.20,
                                    "AI: waiting for first window...",
                                    color="#888888", fontsize=11,
                                    transform=ax_info.transAxes)
        self.txt_next  = ax_info.text(0.60, 0.20, "", color="#555555",
                                       fontsize=9, transform=ax_info.transAxes)
        self.txt_rec   = ax_info.text(0.75, 0.75, "", color="#ff4444",
                                       fontsize=10, fontweight="bold",
                                       transform=ax_info.transAxes)
        self.txt_saved = ax_info.text(0.60, 0.50, "", color="#556666",
                                       fontsize=9, transform=ax_info.transAxes)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.text(0.5, 0.01,
                      "Press  R = start/stop recording   Q = quit",
                      ha="center", color="#445555", fontsize=9)
        self.fig.suptitle("CardioWeave — ECG + SpO2 Monitor",
                          color="#00ff88", fontsize=13, fontweight="bold")

    def _on_key(self, event):
        if event.key in ("r", "R"):
            self._toggle_recording()
        elif event.key in ("q", "Q"):
            self._quit()

    def _toggle_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self._rec_wall = time.time()
            self.acq.start_recording()
            self.txt_rec.set_text("● REC")
        else:
            self.is_recording = False
            c1, c2, start = self.acq.stop_recording()
            self.txt_rec.set_text("")
            def _save():
                path = save_edf(c1, c2, start)
                self.txt_saved.set_text(f"Saved: {os.path.basename(path)}")
            threading.Thread(target=_save, daemon=True).start()

    def _quit(self):
        self.acq.stop()
        self.inference.stop()
        plt.close("all")

    def _animate(self, _frame):
        c1, c2 = self.acq.get_display()
        self.lines[0].set_ydata(c1)
        self.lines[1].set_ydata(c2)

        hr, spo2 = self.max_thread.get_vitals()
        self.txt_hr.set_text(f"HR\n{hr} bpm")
        self.txt_spo2.set_text(f"SpO2\n{spo2}%")
        self.txt_spo2.set_color("#ff4444" if spo2 < 95 else "#00aaff")

        if self.acq.lo_flag:
            self.txt_lo.set_text("⚠  ELECTRODE OFF")
            self.txt_status.set_text("● LEAD OFF")
            self.txt_status.set_color("#ff4444")
        else:
            self.txt_lo.set_text("")
            if self.is_recording:
                elapsed = time.time() - self._rec_wall
                m, s = divmod(int(elapsed), 60)
                self.txt_status.set_text(f"● REC  {m:02d}:{s:02d}")
                self.txt_status.set_color("#ff4444")
            else:
                self.txt_status.set_text("● LIVE")
                self.txt_status.set_color("#00ff88")

        secs = max(0, int(self._next_inf - time.time()))
        self.txt_next.set_text(f"next AI check in {secs}s")

        result = self.inference.get_result()
        if result:
            prob  = result["probability"]
            alert = result["alert"]
            if alert:
                self.txt_ai.set_text(f"⚠ MI ALERT — {prob:.1%} probability")
                self.txt_ai.set_color("#ff4444")
            else:
                self.txt_ai.set_text(f"AI: Normal — {prob:.1%}")
                self.txt_ai.set_color("#00ff88")
            self._next_inf = time.time() + INFERENCE_EVERY

        return self.lines + [self.txt_status, self.txt_lo, self.txt_ai,
                              self.txt_next, self.txt_rec, self.txt_saved,
                              self.txt_hr, self.txt_spo2]

    def run(self):
        self._anim = animation.FuncAnimation(
            self.fig, self._animate,
            interval=1000 // DISPLAY_FPS,
            blit=True, cache_frame_data=False)
        plt.show()


# =============================================================================
#  PHONE BROWSER DASHBOARD
# =============================================================================

PHONE_HTML = """<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>RayHeart</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #ffffff;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    max-width: 430px;
    margin: 0 auto;
  }
  .status-bar {
    height: 44px;
    background: #fff;
    display: flex;
    align-items: center;
    padding: 0 20px;
    justify-content: space-between;
    font-size: 12px;
    color: #333;
    border-bottom: 0.5px solid #eee;
  }
  .header {
    padding: 16px 20px 12px;
    background: #fff;
    border-bottom: 0.5px solid #f0f0f0;
  }
  .header h1 {
    font-size: 26px;
    font-weight: 700;
    color: #F57C00;
    letter-spacing: -0.5px;
  }
  .header p {
    font-size: 13px;
    color: #999;
    margin-top: 2px;
  }
  .content {
    flex: 1;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 14px;
    background: #f7f7f7;
  }
  .card {
    background: #fff;
    border-radius: 16px;
    padding: 18px;
    border: 0.5px solid #ececec;
  }
  .card-label {
    font-size: 11px;
    font-weight: 600;
    color: #bbb;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 10px;
  }
  .status-row {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #4CAF50;
  }
  .dot.alert { background: #F44336; }
  .status-text {
    font-size: 15px;
    font-weight: 500;
    color: #222;
  }
  .prob-big {
    font-size: 56px;
    font-weight: 700;
    color: #222;
    line-height: 1;
    margin: 8px 0 4px;
  }
  .prob-big.alert { color: #F44336; }
  .prob-big.normal { color: #4CAF50; }
  .prob-sub {
    font-size: 13px;
    color: #999;
  }
  .bar-wrap {
    background: #f0f0f0;
    border-radius: 8px;
    height: 8px;
    margin-top: 14px;
    overflow: hidden;
  }
  .bar-fill {
    height: 8px;
    border-radius: 8px;
    background: #4CAF50;
    width: 0%;
    transition: width 0.6s ease, background 0.4s;
  }
  .bar-fill.alert { background: #F44336; }
  .message-box {
    background: #fff8f0;
    border: 1px solid #FFE0B2;
    border-radius: 12px;
    padding: 14px 16px;
    display: none;
  }
  .message-box.visible { display: block; }
  .message-box p {
    font-size: 14px;
    color: #E65100;
    font-weight: 500;
    line-height: 1.5;
  }
  .info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 0;
    border-bottom: 0.5px solid #f5f5f5;
  }
  .info-row:last-child { border-bottom: none; }
  .info-label { font-size: 13px; color: #999; }
  .info-val { font-size: 13px; font-weight: 500; color: #333; }
  .conn-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
    background: #E8F5E9;
    color: #2E7D32;
  }
  .conn-badge.disconnected {
    background: #ffeee8;
    color: #c0392b;
  }
  .footer {
    padding: 16px 20px 32px;
    background: #f7f7f7;
    text-align: center;
  }
  .footer p {
    font-size: 11px;
    color: #ccc;
  }
  .footer span { color: #F57C00; font-weight: 600; }
</style>
</head>
<body>

<div class="status-bar">
  <span id="clock">--:--</span>
  <span>RayHeart</span>
  <span>&#9679;</span>
</div>

<div class="header">
  <h1>RayHeart</h1>
  <p>Live cardiac monitoring</p>
</div>

<div class="content">

  <div class="card">
    <div class="card-label">Connection</div>
    <div class="status-row">
      <div class="dot" id="conn-dot"></div>
      <span class="status-text" id="conn-text">Connecting...</span>
      <span class="conn-badge disconnected" id="conn-badge" style="margin-left:auto;">Offline</span>
    </div>
  </div>

  <div class="card">
    <div class="card-label">MI Probability</div>
    <div class="prob-big" id="prob-num">--%</div>
    <div class="prob-sub" id="prob-sub">Waiting for first reading...</div>
    <div class="bar-wrap">
      <div class="bar-fill" id="prob-bar"></div>
    </div>
  </div>

  <div class="message-box" id="alert-box">
    <p>MI SUSPECTED — Seek medical attention immediately.</p>
  </div>

  <div class="card">
    <div class="card-label">Details</div>
    <div class="info-row">
      <span class="info-label">Threshold</span>
      <span class="info-val" id="threshold-val">--</span>
    </div>
    <div class="info-row">
      <span class="info-label">Last update</span>
      <span class="info-val" id="last-update">--</span>
    </div>
    <div class="info-row">
      <span class="info-label">Status</span>
      <span class="info-val" id="status-val">--</span>
    </div>
  </div>

</div>

<div class="footer">
  <p>Powered by <span>RayHeart</span> &bull; CardioWeave AI</p>
</div>

<script>
function updateClock() {
  const now = new Date();
  document.getElementById("clock").textContent =
    now.getHours().toString().padStart(2,"0") + ":" +
    now.getMinutes().toString().padStart(2,"0");
}
updateClock();
setInterval(updateClock, 10000);

const WS_URL = "ws://" + location.hostname + ":8765";
let ws;

function setConnected(on) {
  const dot = document.getElementById("conn-dot");
  const text = document.getElementById("conn-text");
  const badge = document.getElementById("conn-badge");
  if (on) {
    dot.className = "dot";
    text.textContent = "Connected";
    badge.className = "conn-badge";
    badge.textContent = "Live";
  } else {
    dot.className = "dot alert";
    text.textContent = "Disconnected";
    badge.className = "conn-badge disconnected";
    badge.textContent = "Offline";
  }
}

function connect() {
  ws = new WebSocket(WS_URL);
  ws.onopen = () => setConnected(true);
  ws.onclose = () => { setConnected(false); setTimeout(connect, 2000); };
  ws.onerror = () => ws.close();
  ws.onmessage = e => {
    const d = JSON.parse(e.data);
    if (d.type === "init") {
      document.getElementById("threshold-val").textContent =
        Math.round(d.threshold * 100) + "%";
    }
    if (d.type === "inference") {
      const pct = Math.round(d.probability * 100);
      const probEl = document.getElementById("prob-num");
      const barEl = document.getElementById("prob-bar");
      const alertBox = document.getElementById("alert-box");
      const subEl = document.getElementById("prob-sub");
      const statusEl = document.getElementById("status-val");

      probEl.textContent = pct + "%";
      barEl.style.width = pct + "%";

      if (d.alert) {
        probEl.className = "prob-big alert";
        barEl.className = "bar-fill alert";
        alertBox.className = "message-box visible";
        subEl.textContent = "Elevated risk detected";
        statusEl.textContent = "ALERT";
        statusEl.style.color = "#F44336";
      } else {
        probEl.className = "prob-big normal";
        barEl.className = "bar-fill";
        alertBox.className = "message-box";
        subEl.textContent = d.message;
        statusEl.textContent = "Normal";
        statusEl.style.color = "#4CAF50";
      }

      const ts = new Date(d.timestamp);
      document.getElementById("last-update").textContent =
        ts.getHours().toString().padStart(2,"0") + ":" +
        ts.getMinutes().toString().padStart(2,"0") + ":" +
        ts.getSeconds().toString().padStart(2,"0");
    }
  };
}
connect();
</script>
</body>
</html>"""


def serve_phone_dashboard():
    from http.server import HTTPServer, BaseHTTPRequestHandler
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(PHONE_HTML.encode())
        def log_message(self, *args):
            pass
    server = HTTPServer(("0.0.0.0", 8766), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print("[HTTP] Phone dashboard on port 8766")


# =============================================================================
#  MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("  CardioWeave — ECG + SpO2 + HR + MI Detection")
    print(f"  Sample rate    : {SAMPLE_RATE} SPS")
    print(f"  Inference every: {INFERENCE_EVERY}s")
    print(f"  Alert threshold: {ALERT_THRESHOLD}")
    print(f"  Recordings     : {RECORDINGS_DIR}")
    print("  Controls       : R = record/stop   Q = quit")
    print("=" * 60)

    if HARDWARE_AVAILABLE:
        try:
            hw = ECGHardware()
        except Exception as e:
            print(f"[WARN] ECG hardware failed ({e}) — mock mode")
            hw = MockHardware()
    else:
        hw = MockHardware()

    if HARDWARE_AVAILABLE:
        try:
            max_sensor = MAX30102()
        except Exception as e:
            print(f"[WARN] MAX30102 failed ({e}) — mock vitals")
            max_sensor = MockMAX30102()
    else:
        max_sensor = MockMAX30102()

    try:
        predictor = MIPredictor()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    ws = WSServer()
    serve_phone_dashboard()
    print(f"[INFO] Phone dashboard: http://<pi-ip>:8766")

    acq          = AcquisitionThread(hw)
    max_thread   = MAX30102Thread(max_sensor)
    vitals_bcast = VitalsBroadcastThread(max_thread, ws)
    inf          = InferenceThread(acq, predictor, ws, max_thread)

    acq.start()
    max_thread.start()
    vitals_bcast.start()
    inf.start()

    try:
        display = ECGDisplay(acq, inf, ws, max_thread)
        display.run()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted")
    finally:
        acq.stop()
        inf.stop()
        vitals_bcast.stop()
        max_thread.stop()
        hw.cleanup()
        max_sensor.cleanup()
        print("[INFO] Shutdown complete")


if __name__ == "__main__":
    main()
