import re
import pandas as pd

def detect_keyword_stuffing(text: str) -> dict:
    """
    Analyzes resume text for keyword manipulation, hidden text repetition, and unnatural term density.
    """
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return {"is_suspicious": False, "repetition_ratio": 0.0, "flagged_keywords": {}}

    total_words = len(words)
    unique_words = len(set(words))
    rep_ratio = 1.0 - (unique_words / total_words)

    # Word frequency analysis (ignoring common English stop words)
    stop_words = {"and", "the", "in", "to", "of", "a", "with", "for", "on", "as", "an", "at", "by", "from", "is", "or", "was", "be"}
    freq = {}
    for w in words:
        if w not in stop_words and len(w) > 2:
            freq[w] = freq.get(w, 0) + 1

    # Flag words appearing more than 15 times or occupying > 4% of entire document
    flagged = {k: v for k, v in freq.items() if v >= 15 or (total_words >= 50 and (v / total_words) > 0.04)}
    is_suspicious = (rep_ratio > 0.55 and total_words > 12) or len(flagged) > 0

    return {
        "is_suspicious": is_suspicious,
        "repetition_ratio": round(rep_ratio, 4),
        "unique_word_ratio": round(unique_words / total_words, 4),
        "flagged_keywords": flagged,
        "assessment": "Warning: High keyword repetition detected." if is_suspicious else "Normal keyword distribution."
    }

def run_adversarial_robustness_test(ranking_model, feature_engineer, original_resume: str, sample_jd: str) -> dict:
    """
    Stress-tests the ML ranking model against adversarial manipulation (keyword stuffing attack).
    Stuffs original resume with 20 repeated tech keywords and compares model score delta.
    """
    # 1. Original Candidate Score
    orig_feats = feature_engineer.build_feature_vector(original_resume, sample_jd)
    orig_score = ranking_model.predict_alignment(pd.DataFrame([orig_feats]))

    # 2. Adversarial Keyword Stuffed Resume Variant
    stuffed_resume = original_resume + "\n" + " ".join(["Python Machine Learning XGBoost SQL AWS Docker Cloud"] * 10)
    stuffed_feats = feature_engineer.build_feature_vector(stuffed_resume, sample_jd)
    stuffed_score = ranking_model.predict_alignment(pd.DataFrame([stuffed_feats]))

    score_diff = stuffed_score - orig_score

    # Model is robust if keyword stuffing does not artificially inflate score by > 0.15 (15%)
    is_robust = abs(score_diff) < 0.15

    return {
        "is_robust": is_robust,
        "original_score": round(orig_score, 4),
        "adversarial_stuffed_score": round(stuffed_score, 4),
        "score_delta": round(score_diff, 4),
        "repetition_penalty_active": stuffed_feats.get("repetition_ratio", 0.0) > 0.50,
        "assessment": "PASS: Model penalized artificial keyword repetition and resisted score inflation." if is_robust else "WARNING: Model exhibited vulnerability to keyword stuffing."
    }
