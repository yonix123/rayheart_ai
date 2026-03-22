#!/usr/bin/env python3
"""
CardioWeave — Unified ECG Recorder + MI Detection
==================================================
Hardware : 1x AD8232 (CJMCU-8232) + ADS1115 + Raspberry Pi
Leads    : V1-V4 (chip 1 on A2), chip 2 mirrored
AI       : Random Forest MI detector, inference every 60s
Alerts   : Matplotlib display + WebSocket to phone browser
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
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False
    print("[WARN] Hardware libs not found — running in MOCK mode")

import websockets

# =============================================================================
#  CONFIGURATION
# =============================================================================

SAMPLE_RATE      = 500
DISPLAY_WINDOW   = 5
DISPLAY_FPS      = 20
MAINS_HZ         = 50
INFERENCE_EVERY  = 60
ALERT_THRESHOLD  = 0.55
MODEL_DIR        = "./models/"
WS_PORT          = 8765
RECORDINGS_DIR   = os.path.expanduser("~/ecg_recordings")

LO_PINS     = {"p1": 17, "n1": 27, "p2": 22, "n2": 23}
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
        fft_v  = np.abs(np.fft.rfft(sig))
        freqs  = np.fft.rfftfreq(len(sig), d=1.0 / 100)
        qrs_b  = fft_v[(freqs >= 5)  & (freqs <= 40)]
        st_b   = fft_v[(freqs >= 0.5) & (freqs <= 5)]
        ns_b   = fft_v[freqs > 40]
        features.extend([
            np.sum(qrs_b**2),
            np.sum(st_b**2),
            np.sum(ns_b**2),
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
        target = int(len(c1) * 100 / SAMPLE_RATE)
        c1 = sp_resample(c1, target)
        c2 = sp_resample(c2, target)
        feats        = extract_features(c1, c2).reshape(1, -1)
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
        self.ads.gain =           # highest gain: ±0.256V — needed for ECG signal
        self.ch0 = AnalogIn(self.ads, 2)   # A2 — chip 1
        self.ch1 = AnalogIn(self.ads, 3)   # A3 — chip 2 (or mirror of chip 1)
        GPIO.setmode(GPIO.BCM)
        for pin in LO_PINS.values():
            GPIO.setup(pin, GPIO.IN)
        print("[HW] ADS1115 ready (gain=16)")

    def read(self):
        lo = any(GPIO.input(p) for p in LO_PINS.values())
        if lo:
            return None, True
        v1 = self.ch0.voltage * 1000.0   # convert to mV
        v2 = self.ch1.voltage * 1000.0   # mirror chip1 if chip2 not connected
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
        v1 = v1 - 550.0   # remove ~0.55V DC offset (in mV: 550mV)
        v2 = v2 - 500.0   # remove ~0.50V DC offset
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
    def __init__(self, acq, predictor, ws):
        super().__init__(daemon=True)
        self.acq         = acq
        self.predictor   = predictor
        self.ws          = ws
        self.running     = False
        self.last_result = None
        self.lock        = threading.Lock()

    def run(self):
        self.running = True
        while self.running:
            time.sleep(INFERENCE_EVERY)
            c1, c2  = self.acq.get_inference_window()
            result  = self.predictor.predict(c1, c2)
            with self.lock:
                self.last_result = result
            print(f"[AI] {result['message']}")
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
    def __init__(self, acq, inference, ws):
        self.acq          = acq
        self.inference    = inference
        self.ws           = ws
        self.is_recording = False
        self._next_inf    = time.time() + INFERENCE_EVERY
        self._rec_wall    = 0.0

        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(13, 7), facecolor="#0a0e0e")
        self.fig.canvas.manager.set_window_title("CardioWeave — Live ECG")

        gs = GridSpec(3, 1, figure=self.fig,
                      height_ratios=[2, 2, 1],
                      hspace=0.35,
                      top=0.92, bottom=0.08, left=0.08, right=0.97)

        n = DISPLAY_WINDOW * SAMPLE_RATE
        x = np.arange(n) / SAMPLE_RATE

        self.axes  = []
        self.lines = []
        colors = ["#00ff88", "#00ccff"]
        for i in range(2):
            ax = self.fig.add_subplot(gs[i])
            ax.set_facecolor("#0a0e0e")
            ax.set_xlim(0, DISPLAY_WINDOW)
            ax.set_ylim(-50, 50)          # ±10 mV range for real ECG with gain 16
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

        ax_info = self.fig.add_subplot(gs[2])
        ax_info.set_facecolor("#0a0e0e")
        ax_info.axis("off")

        self.txt_status = ax_info.text(0.01, 0.75, "● LIVE", color="#00ff88",
                                        fontsize=12, fontweight="bold",
                                        transform=ax_info.transAxes)
        self.txt_lo     = ax_info.text(0.20, 0.75, "", color="#ff4444",
                                        fontsize=11, fontweight="bold",
                                        transform=ax_info.transAxes)
        self.txt_ai     = ax_info.text(0.01, 0.35, "AI: waiting for first window...",
                                        color="#888888", fontsize=11,
                                        transform=ax_info.transAxes)
        self.txt_next   = ax_info.text(0.01, 0.05, "", color="#555555",
                                        fontsize=9, transform=ax_info.transAxes)
        self.txt_rec    = ax_info.text(0.70, 0.75, "", color="#ff4444",
                                        fontsize=10, fontweight="bold",
                                        transform=ax_info.transAxes)
        self.txt_saved  = ax_info.text(0.70, 0.35, "", color="#556666",
                                        fontsize=9, transform=ax_info.transAxes)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.text(0.5, 0.01, "Press  R = start/stop recording   Q = quit",
                      ha="center", color="#445555", fontsize=9)
        self.fig.suptitle("CardioWeave — ECG Monitor",
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

        return self.lines + [self.txt_status, self.txt_lo,
                              self.txt_ai, self.txt_next,
                              self.txt_rec, self.txt_saved]

    def run(self):
        interval_ms = 1000 // DISPLAY_FPS
        self._anim = animation.FuncAnimation(
            self.fig, self._animate,
            interval=interval_ms, blit=True, cache_frame_data=False)
        plt.show()


# =============================================================================
#  PHONE BROWSER DASHBOARD
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
  #prob-box{width:90%;max-width:400px;background:#111;border-radius:8px;
            padding:20px;text-align:center;border:1px solid #334;}
  #prob-num{font-size:48px;font-weight:bold;color:#00ff88;margin:10px 0;}
  #prob-bar-wrap{background:#1a1a1a;border-radius:4px;height:16px;margin:10px 0;}
  #prob-bar{height:16px;border-radius:4px;background:#00ff88;width:0%;
            transition:width 0.5s,background 0.5s;}
  #message{margin-top:16px;font-size:14px;color:#aaa;min-height:40px;}
  #alert-box{display:none;margin-top:20px;padding:16px;border-radius:8px;
             background:#3a0000;border:2px solid #ff4444;
             color:#ff4444;font-size:16px;font-weight:bold;
             text-align:center;width:90%;max-width:400px;}
  #ts{margin-top:12px;font-size:11px;color:#555;}
</style>
</head>
<body>
<h2>CardioWeave</h2>
<div id="status">Connecting...</div>
<div id="prob-box">
  <div>MI Probability</div>
  <div id="prob-num">--%</div>
  <div id="prob-bar-wrap"><div id="prob-bar"></div></div>
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
    if (d.type === 'inference') {
      const pct = Math.round(d.probability * 100);
      document.getElementById('prob-num').textContent  = pct + '%';
      document.getElementById('prob-bar').style.width  = pct + '%';
      document.getElementById('prob-bar').style.background = d.alert ? '#ff4444' : '#00ff88';
      document.getElementById('prob-num').style.color  = d.alert ? '#ff4444' : '#00ff88';
      document.getElementById('message').textContent   = d.message;
      document.getElementById('ts').textContent = 'Last: ' + new Date(d.timestamp).toLocaleTimeString();
      document.getElementById('alert-box').style.display = d.alert ? 'block' : 'none';
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
    print("  CardioWeave — ECG Recorder + MI Detection")
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
            print(f"[WARN] Hardware init failed ({e}) — mock mode")
            hw = MockHardware()
    else:
        hw = MockHardware()

    try:
        predictor = MIPredictor()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    ws = WSServer()
    serve_phone_dashboard()
    print(f"[INFO] Open on phone: http://<pi-ip>:8766")

    acq = AcquisitionThread(hw)
    acq.start()

    inf = InferenceThread(acq, predictor, ws)
    inf.start()

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
