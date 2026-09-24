import pandas as pd
import numpy as np

FEATURE_DISPLAY_NAMES = {
    "semantic_similarity": "Overall Semantic Alignment",
    "experience_semantic_match": "Experience Relevance",
    "project_semantic_match": "Project Relevance",
    "skill_match_ratio": "Overall Skill Match",
    "required_skill_match": "Required Skills Match",
    "preferred_skill_match": "Preferred Skills Match",
    "evidence_strength": "Skill Evidence Strength",
    "evidence_coverage": "Evidence Coverage",
    "tfidf_similarity": "Keyword Match (TF-IDF)",
    "unique_word_ratio": "Vocabulary Diversity",
    "repetition_ratio": "Repetition Penalty Index",
    "keyword_density": "Keyword Density",
    "extraction_quality": "Resume Quality Index"
}

class ExplanationEngine:
    def __init__(self, model=None):
        self.model = model

    def generate_feature_contributions(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes feature-level contribution metrics for UI visualization.
        """
        row = feature_df.iloc[0].to_dict()
        contributions = []

        if self.model is not None and hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            feature_names = feature_df.columns.tolist()

            for feat, imp in zip(feature_names, importances):
                val = float(row.get(feat, 0.0))
                # Normalized contribution score
                contrib_score = (val - 0.5) * float(imp) * 10.0
                contributions.append({
                    "feature_key": feat,
                    "feature_name": FEATURE_DISPLAY_NAMES.get(feat, feat),
                    "value": round(val, 4),
                    "model_importance": round(float(imp), 4),
                    "contribution_score": round(contrib_score, 4),
                    "impact": "Positive Boost" if contrib_score >= 0 else "Area for Review"
                })
        else:
            for feat, val in row.items():
                contributions.append({
                    "feature_key": feat,
                    "feature_name": FEATURE_DISPLAY_NAMES.get(feat, feat),
                    "value": round(float(val), 4),
                    "model_importance": 0.1,
                    "contribution_score": round((float(val) - 0.5) * 2.0, 4),
                    "impact": "Positive Boost" if float(val) >= 0.5 else "Area for Review"
                })

        df_contrib = pd.DataFrame(contributions)
        return df_contrib.sort_values(by="model_importance", ascending=False)

    def generate_human_readable_explanation(self, feature_df: pd.DataFrame, matched_req: list, missing_req: list, evidence_summary: dict) -> dict:
        """
        Generates intuitive bullet points for HR / recruiters.
        """
        row = feature_df.iloc[0].to_dict()

        positives = []
        cautions = []

        # Required skill alignment
        req_ratio = row.get("required_skill_match", 0.0)
        if req_ratio >= 0.8:
            positives.append(f"✓ **Strong Required Skills Alignment**: {len(matched_req)} required skill(s) detected with high overlap.")
        elif req_ratio >= 0.5:
            positives.append(f"✓ **Moderate Required Skills Match**: {len(matched_req)} required skill(s) matched.")
        else:
            cautions.append(f"⚠ **Required Skill Gaps**: {len(missing_req)} required skill(s) missing ({', '.join(missing_req[:3])}).")

        # Project & Experience Semantic relevance
        exp_sim = row.get("experience_semantic_match", 0.0)
        if exp_sim >= 0.70:
            positives.append("✓ **High Experience Relevance**: Work history aligns strongly with job responsibilities.")
        elif exp_sim < 0.45:
            cautions.append("⚠ **Low Experience Overlap**: Previous roles show limited contextual overlap with target role.")

        proj_sim = row.get("project_semantic_match", 0.0)
        if proj_sim >= 0.70:
            positives.append("✓ **Strong Project Evidence**: Candidate exhibits relevant hands-on technical project experience.")

        # Evidence strength
        ev_strength = row.get("evidence_strength", 0.0)
        if ev_strength >= 0.65:
            positives.append("✓ **High Evidence Level**: Skills are substantiated by action verbs and quantifiable metrics.")
        elif ev_strength < 0.40:
            cautions.append("⚠ **Weak Evidence Coverage**: Claimed skills have limited supporting action context or metrics.")

        # Keyword stuffing flag
        rep_ratio = row.get("repetition_ratio", 0.0)
        kw_density = row.get("keyword_density", 0.0)
        if rep_ratio > 0.55 or kw_density > 0.35:
            cautions.append("⚠ **Keyword Repetition Detected**: High word repetition ratio detected. Evaluated with evidence penalty.")

        return {
            "strengths": positives if positives else ["Basic document requirements met."],
            "areas_for_review": cautions if cautions else ["No critical alignment flags detected."]
        }
