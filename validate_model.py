"""
CardioWeave — Real Validation Test
Pulls actual labeled ECGs from PTB-XL and tests the model against them.
Run from your rayheart_ai project folder:
    python validate_model.py
"""

import os
import numpy as np
import pandas as pd
import wfdb
import ast
from scipy.signal import butter, filtfilt, iirnotch

import importlib.util
spec = importlib.util.spec_from_file_location("predict", "./predict-2.py")
predict_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(predict_module)
MIPredictor = predict_module.MIPredictor
extract_features = predict_module.extract_features_from_two_signals

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH  = "./"
N_SAMPLES  = 50
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_100hz(sig):
    """Filter signal that is already at 100 Hz — no resampling needed."""
    nyq = 50.0
    b, a   = butter(4, [0.5 / nyq, 40.0 / nyq], btype='band')
    bn, an = iirnotch(50.0 / nyq, Q=30)
    return filtfilt(bn, an, filtfilt(b, a, sig))


def predict_ptbxl(predictor, v1, v3):
    """
    Run inference on a PTB-XL record (already 100 Hz).
    Skips the resampling step inside predict-2.py's predict() method.
    """
    chip1 = np.array(v1, dtype=np.float64)
    chip2 = np.array(v3, dtype=np.float64)

    if len(chip1) < 100:
        return {"alert": False}

    chip1 = preprocess_100hz(chip1)
    chip2 = preprocess_100hz(chip2)

    features = extract_features(chip1, chip2).reshape(1, -1)
    features_scaled = predictor.scaler.transform(features)
    prob = float(predictor.model.predict_proba(features_scaled)[0, 1])

    return {
        "probability": prob,
        "alert": prob >= predictor.threshold,
    }


def load_ptbxl_labels(data_path):
    csv_path = os.path.join(data_path, "ptbxl_database.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find ptbxl_database.csv in {data_path}")

    df = pd.read_csv(csv_path)
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

    scp_df = pd.read_csv(os.path.join(data_path, "scp_statements.csv"), index_col=0)
    mi_codes   = set(scp_df[scp_df["diagnostic_class"] == "MI"].index.tolist())
    norm_codes = {"NORM"}

    normals = df[df["scp_codes"].apply(lambda c: any(k in norm_codes for k in c))]["filename_lr"].tolist()
    mis     = df[df["scp_codes"].apply(lambda c: any(k in mi_codes   for k in c))]["filename_lr"].tolist()

    return normals, mis


def load_ecg(data_path, filename):
    try:
        record  = wfdb.rdrecord(os.path.join(data_path, filename))
        signals = record.p_signal  # (1000, 12)
        v1 = signals[:, 6].tolist()
        v3 = signals[:, 8].tolist()
        return v1, v3
    except Exception:
        return None, None


def run_validation():
    print("=" * 60)
    print("  CardioWeave — Real PTB-XL Validation")
    print("=" * 60)

    predictor = MIPredictor()
    print(f"[AI] Threshold: {predictor.threshold}\n")

    normals, mis = load_ptbxl_labels(DATA_PATH)
    print(f"  Found {len(normals)} normal, {len(mis)} MI records")
    print(f"  Testing {N_SAMPLES} of each\n")

    # ── NORMALS ───────────────────────────────────────────────────────────────
    print("Testing NORMAL ECGs...")
    normal_results, tested = [], 0
    for fname in normals:
        if tested >= N_SAMPLES:
            break
        v1, v3 = load_ecg(DATA_PATH, fname)
        if v1 is None:
            continue
        result = predict_ptbxl(predictor, v1, v3)
        normal_results.append(result["alert"])
        tested += 1

    fp      = sum(normal_results)
    fp_rate = fp / len(normal_results) * 100 if normal_results else 0
    print(f"  Tested: {len(normal_results)}")
    print(f"  False positives: {fp} ({fp_rate:.1f}%)")

    # ── MIs ───────────────────────────────────────────────────────────────────
    print("\nTesting MI ECGs...")
    mi_results, tested = [], 0
    for fname in mis:
        if tested >= N_SAMPLES:
            break
        v1, v3 = load_ecg(DATA_PATH, fname)
        if v1 is None:
            continue
        result = predict_ptbxl(predictor, v1, v3)
        mi_results.append(result["alert"])
        tested += 1

    tp          = sum(mi_results)
    sensitivity = tp / len(mi_results) * 100 if mi_results else 0
    print(f"  Tested: {len(mi_results)}")
    print(f"  Correctly detected: {tp} ({sensitivity:.1f}%)")

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Sensitivity (MI caught)  : {sensitivity:.1f}%  (target: ≥75%)")
    print(f"  False positive rate      : {fp_rate:.1f}%   (target: ≤15%)")

    if sensitivity >= 75 and fp_rate <= 15:
        print("\n  ✅ Model meets clinical MVP targets.")
    elif sensitivity >= 75:
        print("\n  ⚠️  Sensitivity OK but too many false alarms.")
        print("     Try increasing threshold in predict-2.py (e.g. 0.50)")
    elif fp_rate <= 15:
        print("\n  ⚠️  Low false alarms but missing too many real MIs.")
        print("     Try lowering threshold in predict-2.py (e.g. 0.40)")
    else:
        print("\n  ❌ Model needs retuning.")
    print("=" * 60)


if __name__ == "__main__":
    run_validation()