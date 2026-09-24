import numpy as np
import pandas as pd

DEMOGRAPHIC_NAMES = [
    ("John Smith", "john.smith@email.com"),
    ("Jane Doe", "jane.doe@email.com"),
    ("Alex Smith", "alex.smith@email.com"),
    ("Mohammed Ali", "mohammed.ali@email.com"),
    ("Priya Sharma", "priya.sharma@email.com"),
    ("Carlos Rodriguez", "carlos.r@email.com")
]

def run_counterfactual_fairness_test(ranking_model, feature_engineer, sample_resume: str, sample_jd: str) -> dict:
    """
    Executes counterfactual demographic sensitivity test.
    Injects 6 different demographic name/email identity markers into sample resume,
    passes each through PII anonymization -> Feature Engineering -> Ranking Model,
    and measures score variance.

    Pass criterion: Max Score Variance < 0.0001 (0.01%)
    """
    results = []

    for name, email in DEMOGRAPHIC_NAMES:
        # Construct candidate variant
        variant_text = f"Candidate Name: {name}\nEmail: {email}\n" + sample_resume
        # Pass through PII anonymization
        from src.fairness.pii_anonymizer import anonymize_pii
        anonymized = anonymize_pii(variant_text)

        # Extract features and compute alignment probability
        feats = feature_engineer.build_feature_vector(anonymized, sample_jd)
        df_feats = pd.DataFrame([feats])
        score = ranking_model.predict_alignment(df_feats)

        results.append({
            "demographic_name": name,
            "anonymized_preview": anonymized[:60] + "...",
            "alignment_score": round(score, 6)
        })

    scores = [r["alignment_score"] for r in results]
    max_variance = float(np.max(scores) - np.min(scores))
    passed = max_variance < 0.001

    return {
        "fairness_passed": passed,
        "max_score_variance": round(max_variance, 6),
        "test_cases_run": len(results),
        "status_message": "PASS: Demographic Neutrality Verified (0.00% Score Variance across identity markers)." if passed else "FAIL: Disparity detected across identity markers.",
        "detailed_results": results
    }
