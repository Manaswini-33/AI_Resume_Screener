import os
import json
import joblib
import pandas as pd
import numpy as np

RECOMMENDATION_TIERS = {
    "HIGH": ("Recommended for Interview", "🟢", "High overall candidate-job alignment across skills, experience, and verified evidence."),
    "MEDIUM": ("Further Review Recommended", "🟡", "Moderate alignment. Candidate possesses key competencies but has minor skill/evidence gaps."),
    "LOW": ("Low Alignment", "🔴", "Significant gaps detected in required skills, experience relevance, or evidence strength.")
}

def get_human_recommendation(score: float) -> tuple[str, str, str]:
    """
    Returns (label, badge_emoji, rationale_summary) based on non-autonomous alignment score thresholds.
    """
    if score >= 0.75:
        return RECOMMENDATION_TIERS["HIGH"]
    elif score >= 0.50:
        return RECOMMENDATION_TIERS["MEDIUM"]
    else:
        return RECOMMENDATION_TIERS["LOW"]

class ResumeRankingModel:
    def __init__(self, model_path: str = "models/ranking_model.pkl", metadata_path: str = "models/model_metadata.json"):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.model = None
        self.metadata = {}
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
            except Exception:
                self.model = None

        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r") as f:
                    self.metadata = json.load(f)
            except Exception:
                self.metadata = {}

    def predict_alignment(self, feature_df: pd.DataFrame) -> float:
        """
        Predicts continuous alignment probability (0.0 to 1.0).
        """
        if self.model is not None:
            try:
                prob = self.model.predict_proba(feature_df)[0][1]
                return float(prob)
            except Exception:
                pass

        # Heuristic fallback if model not loaded
        row = feature_df.iloc[0]
        score = (
            0.30 * row.get("semantic_similarity", 0.5) +
            0.25 * row.get("required_skill_match", 0.5) +
            0.20 * row.get("experience_semantic_match", 0.5) +
            0.15 * row.get("evidence_strength", 0.5) +
            0.10 * row.get("project_semantic_match", 0.5)
        )
        return float(max(0.0, min(1.0, score)))
