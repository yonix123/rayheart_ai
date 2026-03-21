"""
CardioWeave - Threshold Optimization Script
File: src/optimize_threshold.py

This script finds the optimal decision threshold to maximize sensitivity
while maintaining acceptable specificity.
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Must match your train_model.py configuration"""
    DATA_PATH = './'
    MODEL_SAVE_PATH = './models/'
    SAMPLING_RATE = 100
    USE_FULL_DATASET = True

# ============================================================================
# LOAD MODEL AND DATA
# ============================================================================

def find_latest_model(model_path):
    """Find the most recently saved model"""
    model_files = [f for f in os.listdir(model_path) if f.startswith('mi_model_')]
    if not model_files:
        raise FileNotFoundError("No model files found in ./models/")
    
    latest_model = sorted(model_files)[-1]
    timestamp = latest_model.replace('mi_model_', '').replace('.pkl', '')
    
    return f"{model_path}{latest_model}", f"{model_path}scaler_{timestamp}.pkl"

def load_model_and_scaler():
    """Load the trained model and scaler"""
    print("🔍 Finding latest model...")
    
    model_path, scaler_path = find_latest_model(Config.MODEL_SAVE_PATH)
    
    print(f"   Model: {model_path}")
    print(f"   Scaler: {scaler_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    print("✅ Model and scaler loaded successfully!")
    return model, scaler

# ============================================================================
# RECREATE TEST SET
# ============================================================================

def recreate_test_data():
    """
    Recreate the exact test set used during training.
    """
    print("\n📊 Recreating test dataset...")
    print("⚠️  This will take a few minutes as we need to reload and preprocess data")
    
    # Import functions from cardioweave_train_complete
    import sys
    sys.path.append('./src')
    from cardioweave_train_complete import (
        load_ptbxl_metadata,
        create_mi_labels,
        load_raw_ecg_data,
        preprocess_ecg_signals,
        extract_ecg_features
    )
    
    # Load and process data (same as training)
    Y, agg_df = load_ptbxl_metadata(Config.DATA_PATH)
    Y_filtered = create_mi_labels(Y, agg_df)
    
    if Config.USE_FULL_DATASET:
        Y_subset = Y_filtered
    else:
        Y_subset = Y_filtered.head(2000)
    
    print("   Loading ECG waveforms...")
    raw_ecg = load_raw_ecg_data(Y_subset, Config.SAMPLING_RATE, Config.DATA_PATH)
    labels = Y_subset.MI_label.values
    
    # Ensure correct shape (samples, leads, time_points)
    if raw_ecg.ndim == 3 and raw_ecg.shape[1] != 12:
        import numpy as np
        raw_ecg = np.transpose(raw_ecg, (0, 2, 1))
    
    print("   Preprocessing signals...")
    processed_ecg = preprocess_ecg_signals(raw_ecg, Config.SAMPLING_RATE)
    
    print("   Extracting features...")
    features = extract_ecg_features(processed_ecg, Config.SAMPLING_RATE)
    
    # Split using official folds (same as training)
    test_idx = Y_subset.strat_fold == 10
    
    X_test = features[test_idx]
    y_test = labels[test_idx]
    
    print(f"✅ Test set recreated: {len(X_test)} samples")
    return X_test, y_test

# ============================================================================
# THRESHOLD OPTIMIZATION
# ============================================================================

def calculate_metrics(y_true, y_pred):
    """Calculate sensitivity and specificity"""
    cm = confusion_matrix(y_true, y_pred)
    
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return sensitivity, specificity

def optimize_threshold(model, scaler, X_test, y_test):
    """
    Find optimal threshold by testing different values
    """
    print("\n🔧 Optimizing decision threshold...")
    
    # Scale test data
    X_test_scaled = scaler.transform(X_test)
    
    # Get probability predictions
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Test different thresholds
    thresholds = np.arange(0.1, 0.9, 0.025)
    results = []
    
    print("\n" + "="*80)
    print("Threshold | Sensitivity | Specificity | F1-Score | TP  | FN  | FP  | TN ")
    print("="*80)
    
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        
        sensitivity, specificity = calculate_metrics(y_test, y_pred)
        
        # Calculate F1 score
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        results.append({
            'threshold': thresh,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'f1_score': f1,
            'tp': tp,
            'fn': fn,
            'fp': fp,
            'tn': tn
        })
        
        print(f"  {thresh:.3f}   |    {sensitivity:.4f}    |    {specificity:.4f}    |  {f1:.4f}  | {tp:3d} | {fn:3d} | {fp:3d} | {tn:3d}")
    
    print("="*80)
    
    return pd.DataFrame(results), y_pred_proba

def find_best_thresholds(results_df):
    """
    Identify best thresholds for different criteria
    """
    print("\n" + "="*80)
    print("🎯 RECOMMENDED THRESHOLDS")
    print("="*80)
    
    # 1. Maximize F1 score (balance)
    best_f1_idx = results_df['f1_score'].idxmax()
    best_f1 = results_df.iloc[best_f1_idx]
    
    print("\n1️⃣  Best Overall Balance (Highest F1-Score):")
    print(f"   Threshold: {best_f1['threshold']:.3f}")
    print(f"   Sensitivity: {best_f1['sensitivity']:.2%}")
    print(f"   Specificity: {best_f1['specificity']:.2%}")
    print(f"   F1-Score: {best_f1['f1_score']:.4f}")
    
    # 2. Target 75% sensitivity
    target_75_sens = results_df[results_df['sensitivity'] >= 0.75]
    if len(target_75_sens) > 0:
        best_75_idx = target_75_sens['specificity'].idxmax()
        best_75 = target_75_sens.loc[best_75_idx]
        
        print("\n2️⃣  Target 75%+ Sensitivity (Medical Device Standard):")
        print(f"   Threshold: {best_75['threshold']:.3f}")
        print(f"   Sensitivity: {best_75['sensitivity']:.2%}")
        print(f"   Specificity: {best_75['specificity']:.2%}")
        print(f"   F1-Score: {best_75['f1_score']:.4f}")
    else:
        # Find closest to 75%
        results_df['sens_diff'] = abs(results_df['sensitivity'] - 0.75)
        closest_idx = results_df['sens_diff'].idxmin()
        closest = results_df.iloc[closest_idx]
        
        print("\n2️⃣  Closest to 75% Sensitivity:")
        print(f"   Threshold: {closest['threshold']:.3f}")
        print(f"   Sensitivity: {closest['sensitivity']:.2%}")
        print(f"   Specificity: {closest['specificity']:.2%}")
        print(f"   F1-Score: {closest['f1_score']:.4f}")
    
    # 3. High sensitivity (catch most MIs)
    high_sens_target = results_df[results_df['sensitivity'] >= 0.80]
    if len(high_sens_target) > 0:
        best_high_sens_idx = high_sens_target['specificity'].idxmax()
        best_high_sens = high_sens_target.loc[best_high_sens_idx]
        
        print("\n3️⃣  High Sensitivity Mode (Catch Most Heart Attacks):")
        print(f"   Threshold: {best_high_sens['threshold']:.3f}")
        print(f"   Sensitivity: {best_high_sens['sensitivity']:.2%}")
        print(f"   Specificity: {best_high_sens['specificity']:.2%}")
        print(f"   F1-Score: {best_high_sens['f1_score']:.4f}")
        print(f"   ⚠️  Warning: Higher false alarm rate")
    
    # 4. High specificity (minimize false alarms)
    best_spec_idx = results_df['specificity'].idxmax()
    best_spec = results_df.iloc[best_spec_idx]
    
    print("\n4️⃣  Low False Alarm Mode (Highest Specificity):")
    print(f"   Threshold: {best_spec['threshold']:.3f}")
    print(f"   Sensitivity: {best_spec['sensitivity']:.2%}")
    print(f"   Specificity: {best_spec['specificity']:.2%}")
    print(f"   F1-Score: {best_spec['f1_score']:.4f}")
    print(f"   ⚠️  Warning: Will miss more heart attacks")
    
    print("="*80)
    
    return best_f1, results_df

def plot_threshold_curves(results_df, y_test, y_pred_proba):
    """
    Create visualization of threshold effects
    """
    print("\n📊 Generating threshold analysis plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Sensitivity vs Specificity
    ax1 = axes[0, 0]
    ax1.plot(results_df['threshold'], results_df['sensitivity'], 'b-', label='Sensitivity', linewidth=2)
    ax1.plot(results_df['threshold'], results_df['specificity'], 'r-', label='Specificity', linewidth=2)
    ax1.axhline(y=0.75, color='g', linestyle='--', label='75% Target')
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Rate')
    ax1.set_title('Sensitivity vs Specificity by Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: F1 Score
    ax2 = axes[0, 1]
    ax2.plot(results_df['threshold'], results_df['f1_score'], 'g-', linewidth=2)
    best_f1_idx = results_df['f1_score'].idxmax()
    ax2.axvline(x=results_df.iloc[best_f1_idx]['threshold'], color='r', linestyle='--', label='Best F1')
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('F1 Score by Threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: ROC Curve
    ax3 = axes[1, 0]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    ax3.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
    ax3.plot([0, 1], [0, 1], 'r--', label='Random')
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('ROC Curve')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Confusion Matrix Components
    ax4 = axes[1, 1]
    ax4.plot(results_df['threshold'], results_df['tp'], label='True Positives (MI detected)', linewidth=2)
    ax4.plot(results_df['threshold'], results_df['fn'], label='False Negatives (MI missed)', linewidth=2)
    ax4.plot(results_df['threshold'], results_df['fp'], label='False Positives (False alarms)', linewidth=2)
    ax4.set_xlabel('Threshold')
    ax4.set_ylabel('Count')
    ax4.set_title('Prediction Outcomes by Threshold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('./models/', exist_ok=True)
    plt.savefig('./models/threshold_optimization.png', dpi=300, bbox_inches='tight')
    print("✅ Saved plot to: ./models/threshold_optimization.png")
    
    plt.show()

# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Complete threshold optimization pipeline
    """
    print("\n" + "="*80)
    print("🫀 CARDIOWEAVE - THRESHOLD OPTIMIZATION")
    print("="*80)
    
    # Step 1: Load model
    model, scaler = load_model_and_scaler()
    
    # Step 2: Recreate test data
    X_test, y_test = recreate_test_data()
    
    # Step 3: Optimize threshold
    results_df, y_pred_proba = optimize_threshold(model, scaler, X_test, y_test)
    
    # Step 4: Find best thresholds
    best_threshold, results_df = find_best_thresholds(results_df)
    
    # Step 5: Visualize
    plot_threshold_curves(results_df, y_test, y_pred_proba)
    
    # Step 6: Save results
    results_df.to_csv('./models/threshold_analysis.csv', index=False)
    print("\n✅ Saved threshold analysis to: ./models/threshold_analysis.csv")
    
    print("\n" + "="*80)
    print("🎯 RECOMMENDATION FOR YOUR WEARABLE:")
    print("="*80)
    print(f"\nUse threshold: {best_threshold['threshold']:.3f}")
    print("\nImplementation in your device:")
    print("```python")
    print(f"ALERT_THRESHOLD = {best_threshold['threshold']:.3f}")
    print("if mi_probability >= ALERT_THRESHOLD:")
    print("    send_alert_to_user()")
    print("```")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
