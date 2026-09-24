import os
import json
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def train_ranking_models():
    """
    Trains supervised ML models (XGBoost primary, Random Forest benchmark) on candidate-job multi-factor features.
    Saves trained model and detailed evaluation metadata.
    """
    print("Generating synthetic candidate feature dataset for model training...")
    np.random.seed(42)
    n_samples = 800

    # Multi-factor feature distributions
    sem_sim = np.random.uniform(0.35, 0.95, n_samples)
    exp_sim = sem_sim * np.random.uniform(0.65, 1.0, n_samples)
    proj_sim = sem_sim * np.random.uniform(0.60, 1.0, n_samples)
    skill_ratio = np.random.uniform(0.20, 1.0, n_samples)
    req_match = skill_ratio * np.random.uniform(0.70, 1.0, n_samples)
    pref_match = np.random.uniform(0.10, 0.90, n_samples)
    evidence_str = np.random.uniform(0.25, 1.0, n_samples)
    evidence_cov = evidence_str * np.random.uniform(0.50, 1.0, n_samples)
    tfidf_sim = sem_sim * np.random.uniform(0.50, 0.95, n_samples)
    uniq_ratio = np.random.uniform(0.40, 0.85, n_samples)
    rep_ratio = 1.0 - uniq_ratio
    kw_density = np.random.uniform(0.05, 0.40, n_samples)
    ext_quality = np.random.uniform(0.70, 1.0, n_samples)

    # Multi-factor label formula representing holistic candidate relevance
    score = (
        0.25 * sem_sim +
        0.20 * req_match +
        0.15 * exp_sim +
        0.15 * proj_sim +
        0.15 * evidence_str +
        0.10 * pref_match -
        0.15 * (rep_ratio > 0.55).astype(int) -
        0.10 * (kw_density > 0.35).astype(int)
    )
    labels = (score > 0.52).astype(int)

    df = pd.DataFrame({
        "semantic_similarity": sem_sim,
        "experience_semantic_match": exp_sim,
        "project_semantic_match": proj_sim,
        "skill_match_ratio": skill_ratio,
        "required_skill_match": req_match,
        "preferred_skill_match": pref_match,
        "evidence_strength": evidence_str,
        "evidence_coverage": evidence_cov,
        "tfidf_similarity": tfidf_sim,
        "unique_word_ratio": uniq_ratio,
        "repetition_ratio": rep_ratio,
        "keyword_density": kw_density,
        "extraction_quality": ext_quality,
        "label": labels
    })

    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Primary Model: XGBoost
    xgb_model = XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)

    y_pred_xgb = xgb_model.predict(X_test)
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

    # Benchmark Model: Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    metrics_xgb = {
        "Accuracy": float(accuracy_score(y_test, y_pred_xgb)),
        "Precision": float(precision_score(y_test, y_pred_xgb)),
        "Recall": float(recall_score(y_test, y_pred_xgb)),
        "F1": float(f1_score(y_test, y_pred_xgb)),
        "ROC-AUC": float(roc_auc_score(y_test, y_prob_xgb))
    }

    metrics_rf = {
        "Accuracy": float(accuracy_score(y_test, y_pred_rf)),
        "Precision": float(precision_score(y_test, y_pred_rf)),
        "Recall": float(recall_score(y_test, y_pred_rf)),
        "F1": float(f1_score(y_test, y_pred_rf))
    }

    cm = confusion_matrix(y_test, y_pred_xgb).tolist()

    os.makedirs("models", exist_ok=True)
    joblib.dump(xgb_model, "models/ranking_model.pkl")

    metadata = {
        "model_type": "XGBoost Classifier",
        "benchmark_model": "Random Forest Classifier",
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "features": list(X.columns),
        "feature_importances": {f: float(imp) for f, imp in zip(X.columns, xgb_model.feature_importances_)},
        "metrics": metrics_xgb,
        "benchmark_metrics": metrics_rf,
        "confusion_matrix": cm,
        "data_note": "Dataset constructed with 13 multi-factor signals (semantic, skill match, evidence level, density penalties)."
    }

    with open("models/model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("Model Training Complete!")
    print("Primary XGBoost Metrics:", json.dumps(metrics_xgb, indent=2))
    return xgb_model, metadata

if __name__ == "__main__":
    train_ranking_models()
