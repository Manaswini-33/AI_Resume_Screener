import os
import json
import joblib

def evaluate_saved_model():
    """
    Evaluates the saved XGBoost ranking model and outputs detailed metrics report.
    """
    meta_path = "models/model_metadata.json"
    if not os.path.exists(meta_path):
        print("❌ Model metadata not found. Please run train.py first.")
        return

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    print("==================================================")
    print("      HireSense Model Evaluation Summary          ")
    print("==================================================")
    print(f"Model Type:           {meta.get('model_type')}")
    print(f"Training Samples:     {meta.get('training_samples')}")
    print(f"Test Samples:         {meta.get('test_samples')}")
    print("\n--- Evaluation Metrics ---")
    for k, v in meta.get("metrics", {}).items():
        print(f"  {k:<18}: {v:.4f}")

    print("\n--- Top Feature Importances ---")
    fi = meta.get("feature_importances", {})
    sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)
    for feat, val in sorted_fi[:6]:
        print(f"  {feat:<26}: {val:.4f}")

    print("\n--- Confusion Matrix ---")
    print(meta.get("confusion_matrix"))

if __name__ == "__main__":
    evaluate_saved_model()
