"""
CardioWeave - PTB-XL Heart Attack Detection ML Pipeline
File: src/train_model.py

This script trains a machine learning model to detect Myocardial Infarction (MI)
from ECG data using the PTB-XL dataset.
"""

import numpy as np
import pandas as pd
import wfdb
import os
import pickle
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from scipy import signal
import ast

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration parameters for the pipeline"""
    # Paths - ADJUST THESE TO YOUR SETUP
    DATA_PATH = './'
    MODEL_SAVE_PATH = './models/'
    
    # Data parameters
    SAMPLING_RATE = 100  # Use 100 Hz (faster) or 500 Hz (higher quality)
    USE_FULL_DATASET = True  # Set True to use all 21,000+ records
    SUBSET_SIZE = 2000  # Number of samples if USE_FULL_DATASET=False
    
    # Model parameters
    N_ESTIMATORS = 100
    MAX_DEPTH = 20
    MIN_SAMPLES_SPLIT = 10
    RANDOM_STATE = 42

# ============================================================================
# STEP 1: DATA LOADING
# ============================================================================

def verify_data_exists(data_path):
    """Verify all required files exist"""
    print("Verifying data files...")
    
    required_files = [
        'ptbxl_database.csv',
        'scp_statements.csv',
        'records100/'
    ]
    
    missing = []
    for file in required_files:
        if not os.path.exists(data_path + file):
            missing.append(file)
    
    if missing:
        print(f"❌ ERROR: Missing required files: {missing}")
        print(f"\nPlease ensure PTB-XL is extracted to: {data_path}")
        return False
    
    print("✅ All required files found!")
    return True

def load_ptbxl_metadata(data_path):
    """Load PTB-XL database metadata"""
    print("\nLoading PTB-XL metadata...")
    
    # Load main database
    Y = pd.read_csv(data_path + 'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    
    # Load diagnostic codes
    agg_df = pd.read_csv(data_path + 'scp_statements.csv', index_col=0)
    
    print(f"✅ Loaded {len(Y)} ECG records")
    return Y, agg_df

def load_raw_ecg_data(df, sampling_rate, path):
    """
    Load raw ECG waveforms from files
    
    Args:
        df: DataFrame with ECG metadata
        sampling_rate: 100 or 500 Hz
        path: Path to data folder
    Returns:
        numpy array of ECG signals (n_samples, n_leads, signal_length)
    """
    print(f"Loading {len(df)} ECG waveforms at {sampling_rate} Hz...")
    print("This may take a few minutes...")
    
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path + f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path + f) for f in df.filename_hr]
    
    # Reshape from (samples, time_points, leads) to (samples, leads, time_points)
    data = np.array([signal[0] for signal in data])
    data = np.transpose(data, (0, 2, 1))
    print(f"✅ Loaded ECG data shape: {data.shape}")
    return data

# ============================================================================
# STEP 2: DATA FILTERING AND LABELING
# ============================================================================

def create_mi_labels(Y, agg_df):
    """
    Create binary labels for MI detection
    
    Returns:
        DataFrame with MI_label column (1=MI, 0=Normal)
    """
    print("\nCreating MI labels...")
    
    # Aggregate diagnostic codes to superclass
    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))
    
    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)
    
    # Filter for NORM (normal) and MI (myocardial infarction) cases only
    Y_filtered = Y[Y.diagnostic_superclass.apply(
        lambda x: 'NORM' in x or 'MI' in x
    )].copy()
    
    # Create binary label: 1 for MI, 0 for NORM
    Y_filtered['MI_label'] = Y_filtered.diagnostic_superclass.apply(
        lambda x: 1 if 'MI' in x else 0
    )
    
    # Display class distribution
    n_normal = sum(Y_filtered.MI_label == 0)
    n_mi = sum(Y_filtered.MI_label == 1)
    
    print(f"\n📊 Dataset Composition:")
    print(f"   Normal cases: {n_normal} ({n_normal/len(Y_filtered)*100:.1f}%)")
    print(f"   MI cases: {n_mi} ({n_mi/len(Y_filtered)*100:.1f}%)")
    print(f"   Total: {len(Y_filtered)}")
    
    return Y_filtered

# ============================================================================
# STEP 3: SIGNAL PREPROCESSING
# ============================================================================

def preprocess_ecg_signals(ecg_data, sampling_rate=100):
    """
    Apply signal preprocessing filters to ECG data
    
    Filters applied:
    1. Bandpass filter (0.5-40 Hz) - removes baseline wander and high-freq noise
    2. Notch filter (50 Hz) - removes powerline interference
    
    Args:
        ecg_data: Raw ECG signals (n_samples, n_leads, signal_length)
        sampling_rate: Sampling frequency in Hz
    Returns:
        Filtered ECG signals with same shape
    """
    print("\n🔧 Preprocessing ECG signals...")
    
    # Design bandpass filter
    nyquist = sampling_rate / 2
    low_cutoff = 0.5 / nyquist
    high_cutoff = 40 / nyquist
    b_band, a_band = signal.butter(4, [low_cutoff, high_cutoff], btype='band')
    
    # Design notch filter at 50 Hz (Europe) - change to 60 for US
    notch_freq = 50.0 / nyquist
    quality_factor = 30.0
    b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor)
    
    # Apply filters to all samples and leads
    filtered_data = np.zeros_like(ecg_data)
    
    for i in range(ecg_data.shape[0]):
        if i % 100 == 0:
            print(f"   Processing sample {i}/{ecg_data.shape[0]}...", end='\r')
        
        for lead in range(ecg_data.shape[1]):
            # Apply bandpass filter
            filtered_signal = signal.filtfilt(b_band, a_band, ecg_data[i, lead, :])
            # Apply notch filter
            filtered_signal = signal.filtfilt(b_notch, a_notch, filtered_signal)
            filtered_data[i, lead, :] = filtered_signal
    
    print(f"   Processing sample {ecg_data.shape[0]}/{ecg_data.shape[0]}... Done!")
    print("✅ Preprocessing complete")
    
    return filtered_data

# ============================================================================
# STEP 4: FEATURE EXTRACTION
# ============================================================================

def extract_ecg_features(ecg_data, sampling_rate=100):
    """
    Extract features from 2 differential signals only.
    Chip 1: V1 (index 0) — proxy for V1-V4 differential
    Chip 2: V3 (index 2) — proxy for V3-V5 differential
    """
    print("\n📈 Extracting features from ECG signals...")

    n_samples = ecg_data.shape[0]
    features_list = []

    for i in range(n_samples):
        if i % 100 == 0:
            print(f"   Extracting features: {i}/{n_samples}...", end='\r')

        sample_features = []

        # ── Two signals only ──────────────────────────────
        chip1 = ecg_data[i, 0, :]   # V1 → proxy for V1−V4
        chip2 = ecg_data[i, 2, :]   # V3 → proxy for V3−V5

        for sig in [chip1, chip2]:

            # === Statistical features ===
            sample_features.extend([
                np.mean(sig),
                np.std(sig),
                np.min(sig),
                np.max(sig),
                np.median(sig),
                np.percentile(sig, 25),
                np.percentile(sig, 75),
                np.max(sig) - np.min(sig),          # amplitude range
                np.mean(np.abs(sig - np.mean(sig))), # mean absolute deviation
            ])

            # === Frequency domain features ===
            fft_vals = np.abs(np.fft.rfft(sig))
            freqs    = np.fft.rfftfreq(len(sig), d=1.0/sampling_rate)

            # Power in clinical ECG bands
            qrs_band  = fft_vals[(freqs >= 5)  & (freqs <= 40)]
            st_band   = fft_vals[(freqs >= 0.5) & (freqs <= 5)]
            noise_band = fft_vals[freqs > 40]

            sample_features.extend([
                np.sum(qrs_band**2),    # QRS band power
                np.sum(st_band**2),     # ST/T wave band power
                np.sum(noise_band**2),  # noise power
                np.argmax(fft_vals),    # dominant frequency index
            ])

            # === ST segment proxy ===
            # ST segment is roughly 60-120ms after QRS peak
            # At 100Hz that's samples 6-12 after the max
            qrs_peak = np.argmax(np.abs(sig))
            st_start = min(qrs_peak + 6,  len(sig) - 1)
            st_end   = min(qrs_peak + 12, len(sig))
            st_segment = sig[st_start:st_end]

            sample_features.extend([
                np.mean(st_segment) if len(st_segment) > 0 else 0,  # ST elevation proxy
                np.std(st_segment)  if len(st_segment) > 0 else 0,  # ST variability
            ])

            # === Signal energy and shape ===
            sample_features.extend([
                np.sum(sig**2) / len(sig),           # signal energy
                np.sum(np.diff(sig)**2) / len(sig),  # signal roughness
                np.sum(np.abs(np.diff(sig))),         # total variation
            ])

        features_list.append(sample_features)

    print(f"   Extracting features: {n_samples}/{n_samples}... Done!")

    feature_matrix = np.array(features_list)
    print(f"✅ Feature extraction complete. Shape: {feature_matrix.shape}")

    return feature_matrix

# ============================================================================
# STEP 5: MODEL TRAINING
# ============================================================================

def split_train_test_official(features, labels, Y_subset):
    """
    Split data using official PTB-XL train/test folds
    
    PTB-XL folds:
    - Folds 1-8: Training
    - Fold 9: Validation (optional)
    - Fold 10: Test (for final evaluation)
    """
    print("\n📊 Splitting data using official PTB-XL folds...")
    
    train_idx = Y_subset.strat_fold <= 8
    test_idx = Y_subset.strat_fold == 10
    
    X_train = features[train_idx]
    X_test = features[test_idx]
    y_train = labels[train_idx]
    y_test = labels[test_idx]
    
    print(f"   Training set: {len(X_train)} samples (folds 1-8)")
    print(f"   Test set: {len(X_test)} samples (fold 10)")
    print(f"   Train MI rate: {sum(y_train)/len(y_train)*100:.1f}%")
    print(f"   Test MI rate: {sum(y_test)/len(y_test)*100:.1f}%")
    
    return X_train, X_test, y_train, y_test

def train_random_forest(X_train, y_train, config):
    """
    Train Random Forest classifier
    
    Args:
        X_train: Training features
        y_train: Training labels
        config: Configuration object
    Returns:
        Trained model and scaler
    """
    print("\n🤖 Training Random Forest model...")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Initialize model
    model = RandomForestClassifier(
        n_estimators=config.N_ESTIMATORS,
        max_depth=config.MAX_DEPTH,
        min_samples_split=config.MIN_SAMPLES_SPLIT,
        random_state=config.RANDOM_STATE,
        class_weight='balanced',  # Handle class imbalance
        n_jobs=-1,  # Use all CPU cores
        verbose=1
    )
    
    # Train
    model.fit(X_train_scaled, y_train)
    
    print("✅ Model training complete!")
    
    return model, scaler

def evaluate_model(model, scaler, X_test, y_test):
    """
    Evaluate model performance on test set
    
    Metrics:
    - Classification report (precision, recall, F1)
    - Confusion matrix
    - ROC-AUC score
    """
    print("\n" + "="*70)
    print("📊 MODEL EVALUATION RESULTS")
    print("="*70)
    
    # Scale test data
    X_test_scaled = scaler.transform(X_test)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Normal', 'MI'],
                                digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("📊 Confusion Matrix:")
    print("                 Predicted")
    print("                Normal    MI")
    print(f"Actual Normal   {cm[0,0]:6d}  {cm[0,1]:6d}")
    print(f"       MI       {cm[1,0]:6d}  {cm[1,1]:6d}")
    
    # ROC-AUC
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\n🎯 ROC-AUC Score: {roc_auc:.4f}")
    
    # Additional metrics
    sensitivity = cm[1,1] / (cm[1,1] + cm[1,0])  # True positive rate
    specificity = cm[0,0] / (cm[0,0] + cm[0,1])  # True negative rate
    
    print(f"\n📈 Additional Metrics:")
    print(f"   Sensitivity (Recall for MI): {sensitivity:.4f}")
    print(f"   Specificity: {specificity:.4f}")
    print(f"   False Alarm Rate: {1-specificity:.4f}")
    
    print("="*70)
    
    return y_pred, y_pred_proba

def plot_feature_importance(model, top_n=15):
    """Plot top N most important features"""
    print("\n📊 Generating feature importance plot...")
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(top_n), importances[indices])
    plt.yticks(range(top_n), [f"Feature {i}" for i in indices])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Most Important Features')
    plt.tight_layout()
    
    # Save plot
    os.makedirs('./models/', exist_ok=True)
    plt.savefig('./models/feature_importance.png', dpi=300, bbox_inches='tight')
    print("✅ Saved feature importance plot to: ./models/feature_importance.png")
    plt.close()

def save_model(model, scaler, config):
    """Save trained model and scaler to disk"""
    print("\n💾 Saving model...")
    
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_path = f"{config.MODEL_SAVE_PATH}mi_model_{timestamp}.pkl"
    scaler_path = f"{config.MODEL_SAVE_PATH}scaler_{timestamp}.pkl"
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"✅ Model saved to: {model_path}")
    print(f"✅ Scaler saved to: {scaler_path}")
    
    return model_path, scaler_path

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """
    Complete ML pipeline for MI detection
    """
    print("\n" + "="*70)
    print("🫀 CARDIOWEAVE - MI DETECTION MODEL TRAINING PIPELINE")
    print("="*70)
    
    # Load configuration
    config = Config()
    
    # Step 1: Verify data exists
    if not verify_data_exists(config.DATA_PATH):
        return
    
    # Step 2: Load metadata
    Y, agg_df = load_ptbxl_metadata(config.DATA_PATH)
    
    # Step 3: Create MI labels
    Y_filtered = create_mi_labels(Y, agg_df)
    
    # Step 4: Select subset if needed
    if not config.USE_FULL_DATASET:
        print(f"\n⚠️  Using subset of {config.SUBSET_SIZE} samples for faster training")
        print("    Set USE_FULL_DATASET=True in Config for full dataset")
        Y_subset = Y_filtered.head(config.SUBSET_SIZE)
    else:
        Y_subset = Y_filtered
    
    # Step 5: Load raw ECG data
    raw_ecg = load_raw_ecg_data(Y_subset, config.SAMPLING_RATE, config.DATA_PATH)
    labels = Y_subset.MI_label.values
    
    # Step 6: Preprocess signals
    processed_ecg = preprocess_ecg_signals(raw_ecg, config.SAMPLING_RATE)
    
    # Step 7: Extract features
    features = extract_ecg_features(processed_ecg, config.SAMPLING_RATE)
    
    # Step 8: Split using official folds
    X_train, X_test, y_train, y_test = split_train_test_official(
        features, labels, Y_subset
    )
    
    # Step 9: Train model
    model, scaler = train_random_forest(X_train, y_train, config)
    
    # Step 10: Evaluate
    y_pred, y_pred_proba = evaluate_model(model, scaler, X_test, y_test)
    
    # Step 11: Feature importance
    plot_feature_importance(model)
    
    # Step 12: Save model
    model_path, scaler_path = save_model(model, scaler, config)
    
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE!")
    print("="*70)
    print("\n📝 Next Steps:")
    print("1. Review model performance metrics above")
    print("2. Run optimize_threshold.py to find optimal decision threshold")
    print("3. Use predict.py for real-time prediction with your textile sensors")
    print("\n🫀 CardioWeave - Saving lives through smart textiles")
    print("="*70 + "\n")
    
    return model, scaler, features, labels

if __name__ == "__main__":
    # Run the complete pipeline
    model, scaler, features, labels = main()
