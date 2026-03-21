"""
CardioWeave - Real-Time MI Prediction
File: predict-2.py

Loads the trained model and runs inference on live ECG windows.
Designed to run on Raspberry Pi alongside ecg_recorder.py.
"""

import numpy as np
import pickle
import os
from scipy.signal import butter, filtfilt, iirnotch, resample

# ============================================================================
# CONFIGURATION
# ============================================================================

SAMPLING_RATE   = 500       # Hz — your ADS1115 rate
ALERT_THRESHOLD = 0.425     # From threshold optimizer
WINDOW_SECONDS  = 10        # Seconds of ECG per inference call
MODEL_DIR       = "./models/"

# ============================================================================
# SIGNAL PREPROCESSING — matches preprocess_ecg_signals() in training
# ============================================================================

def preprocess_signal(sig, fs=100):
    nyquist = fs / 2
    low  = 0.5 / nyquist
    high = 40.0 / nyquist
    b_bp, a_bp = butter(4, [low, high], btype='band')
    b_n, a_n   = iirnotch(50.0 / nyquist, Q=30)
    filtered = filtfilt(b_bp, a_bp, sig)
    filtered = filtfilt(b_n,  a_n,  filtered)
    return filtered

# ============================================================================
# FEATURE EXTRACTION — exactly matches extract_ecg_features() in training
# ============================================================================

def extract_features_from_two_signals(chip1_signal, chip2_signal):
    """
    Extract 36 features from 2 signals.
    MUST match extract_ecg_features() in cardioweave_train_complete.py exactly.
    Assumes signals are already at 100 Hz (resampled before calling).
    """
    sample_features = []

    for sig in [chip1_signal, chip2_signal]:

        # === Statistical features (9) ===
        sample_features.extend([
            np.mean(sig),
            np.std(sig),
            np.min(sig),
            np.max(sig),
            np.median(sig),
            np.percentile(sig, 25),
            np.percentile(sig, 75),
            np.max(sig) - np.min(sig),
            np.mean(np.abs(sig - np.mean(sig))),
        ])

        # === Frequency domain features (4) ===
        fft_vals = np.abs(np.fft.rfft(sig))
        freqs    = np.fft.rfftfreq(len(sig), d=1.0 / 100)  # hardcoded 100 Hz

        qrs_band   = fft_vals[(freqs >= 5)  & (freqs <= 40)]
        st_band    = fft_vals[(freqs >= 0.5) & (freqs <= 5)]
        noise_band = fft_vals[freqs > 40]

        sample_features.extend([
            np.sum(qrs_band**2),    # always computed, same as training
            np.sum(st_band**2),
            np.sum(noise_band**2),
            np.argmax(fft_vals),    # int, matches training (no float() cast)
        ])

        # === ST segment proxy (2) ===
        # Hardcoded to 100 Hz: 60ms = 6 samples, 120ms = 12 samples
        qrs_peak = np.argmax(np.abs(sig))
        st_start = min(qrs_peak + 6,  len(sig) - 1)
        st_end   = min(qrs_peak + 12, len(sig))
        st_seg   = sig[st_start:st_end]

        sample_features.extend([
            np.mean(st_seg) if len(st_seg) > 0 else 0,
            np.std(st_seg)  if len(st_seg) > 0 else 0,
        ])

        # === Signal energy and shape (3) ===
        sample_features.extend([
            np.sum(sig**2) / len(sig),
            np.sum(np.diff(sig)**2) / len(sig),
            np.sum(np.abs(np.diff(sig))),
        ])

    return np.array(sample_features, dtype=np.float64)  # shape: (36,)

# ============================================================================
# PREDICTOR CLASS
# ============================================================================

class MIPredictor:

    def __init__(self, model_dir=MODEL_DIR, threshold=ALERT_THRESHOLD):
        self.threshold = threshold
        self.model, self.scaler = self._load_latest_model(model_dir)
        print(f"[AI] Model loaded. Alert threshold: {self.threshold}")

    def _load_latest_model(self, model_dir):
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        model_files = sorted([
            f for f in os.listdir(model_dir)
            if f.startswith("mi_model_") and f.endswith(".pkl")
        ])
        if not model_files:
            raise FileNotFoundError("No trained model found in ./models/")

        latest    = model_files[-1]
        timestamp = latest.replace("mi_model_", "").replace(".pkl", "")

        with open(os.path.join(model_dir, f"mi_model_{timestamp}.pkl"), "rb") as f:
            model = pickle.load(f)
        with open(os.path.join(model_dir, f"scaler_{timestamp}.pkl"), "rb") as f:
            scaler = pickle.load(f)

        print(f"[AI] Loaded: {latest}")
        return model, scaler

    def predict(self, chip1_buffer, chip2_buffer):
        chip1 = np.array(chip1_buffer, dtype=np.float64)
        chip2 = np.array(chip2_buffer, dtype=np.float64)

        if len(chip1) < 100 or len(chip2) < 100:
            return self._insufficient_data()

        # Step 1: preprocess at original sample rate
        chip1_clean = preprocess_signal(chip1, fs=SAMPLING_RATE)
        chip2_clean = preprocess_signal(chip2, fs=SAMPLING_RATE)

        # Step 2: resample to 100 Hz to match training
        target_samples = int(len(chip1_clean) * 100 / SAMPLING_RATE)
        chip1_clean = resample(chip1_clean, target_samples)
        chip2_clean = resample(chip2_clean, target_samples)

        # Step 3: extract features (hardcoded 100 Hz inside, matches training)
        features = extract_features_from_two_signals(
            chip1_clean, chip2_clean
        ).reshape(1, -1)

        # Step 4: scale and predict
        features_scaled = self.scaler.transform(features)
        probability     = float(self.model.predict_proba(features_scaled)[0, 1])

        alert      = probability >= self.threshold
        confidence = self._confidence_label(probability)

        if alert:
            message = (f"MI SUSPECTED — probability {probability:.1%}. "
                       f"Seek medical attention immediately.")
        else:
            message = f"Normal — probability {probability:.1%}"

        return {
            "probability": probability,
            "alert":       alert,
            "confidence":  confidence,
            "message":     message,
        }

    def _confidence_label(self, prob):
        if prob < 0.25:   return "low"
        elif prob < 0.60: return "medium"
        else:             return "high"

    def _insufficient_data(self):
        return {
            "probability": 0.0,
            "alert":       False,
            "confidence":  "low",
            "message":     "Insufficient data — need at least 100 samples",
        }

# ============================================================================
# INTEGRATION SNIPPET FOR ecg_recorder.py
# ============================================================================

INTEGRATION_EXAMPLE = """
# ── Add to ecg_recorder.py ──────────────────────────────────────────────────
from predict import MIPredictor
from collections import deque

predictor     = MIPredictor()
WINDOW        = 500 * 10   # 10 seconds at 500 SPS
chip1_history = deque(maxlen=WINDOW)
chip2_history = deque(maxlen=WINDOW)
inference_counter = 0

# Inside your acquisition loop, after reading ADS1115:
chip1_history.append(lead_I_mv)
chip2_history.append(lead_II_mv)
inference_counter += 1

if inference_counter >= WINDOW:
    result = predictor.predict(list(chip1_history), list(chip2_history))
    print(result["message"])
    if result["alert"]:
        pass  # Trigger your alert — sound, screen, WebSocket, etc.
    inference_counter = 0
# ────────────────────────────────────────────────────────────────────────────
"""

# ============================================================================
# STANDALONE TEST
# ============================================================================

def generate_fake_ecg(n_samples, fs, mi=False):
    t   = np.linspace(0, n_samples / fs, n_samples)
    ecg = 0.1 * np.sin(2 * np.pi * 1.2 * t)
    for spike_t in np.arange(0, n_samples / fs, 0.8):
        idx = int(spike_t * fs)
        if idx < n_samples:
            ecg[idx] += 1.5
    if mi:
        ecg += 0.3
    ecg += np.random.normal(0, 0.02, n_samples)
    return ecg

def main():
    print("\n" + "=" * 60)
    print("  CardioWeave — Predict Script Test")
    print("=" * 60)

    predictor = MIPredictor()
    n = WINDOW_SECONDS * SAMPLING_RATE

    print("\n[TEST 1] Normal ECG")
    result = predictor.predict(
        generate_fake_ecg(n, SAMPLING_RATE, mi=False),
        generate_fake_ecg(n, SAMPLING_RATE, mi=False)
    )
    print(f"  Probability : {result['probability']:.3f}")
    print(f"  Alert       : {result['alert']}")
    print(f"  Message     : {result['message']}")

    print("\n[TEST 2] MI ECG (ST elevated)")
    result_mi = predictor.predict(
        generate_fake_ecg(n, SAMPLING_RATE, mi=True),
        generate_fake_ecg(n, SAMPLING_RATE, mi=True)
    )
    print(f"  Probability : {result_mi['probability']:.3f}")
    print(f"  Alert       : {result_mi['alert']}")
    print(f"  Message     : {result_mi['message']}")

    print("\n[TEST 3] Insufficient data")
    result_short = predictor.predict([0.1, 0.2], [0.1, 0.2])
    print(f"  Message     : {result_short['message']}")

    print("\n" + "=" * 60)
    print("  Integration snippet for ecg_recorder.py:")
    print("=" * 60)
    print(INTEGRATION_EXAMPLE)

if __name__ == "__main__":
    main()