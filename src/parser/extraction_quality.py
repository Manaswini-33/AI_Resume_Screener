import re

def compute_extraction_quality(text: str, parsing_method: str) -> dict:
    """
    Computes an empirical extraction quality score (0-100%) based on:
    - Character and word count
    - Alphabetic character ratio
    - Suspicious symbol ratio
    - OCR usage penalty

    Returns:
      dict with quality_score, word_count, alphabetic_ratio, suspicious_ratio, parsing_method, and warning.
    """
    if not text or not text.strip():
        return {
            "quality_score": 0,
            "word_count": 0,
            "alphabetic_ratio": 0.0,
            "suspicious_ratio": 1.0,
            "parsing_method": parsing_method,
            "warning": "Warning: Resume text extraction produced no readable content. Please verify document formatting or re-upload."
        }

    char_count = len(text)
    words = re.findall(r'\w+', text)
    word_count = len(words)

    if word_count == 0:
        return {
            "quality_score": 0,
            "word_count": 0,
            "alphabetic_ratio": 0.0,
            "suspicious_ratio": 1.0,
            "parsing_method": parsing_method,
            "warning": "Warning: Document contains no recognizable words."
        }

    alpha_chars = sum(1 for c in text if c.isalpha())
    alpha_ratio = alpha_chars / char_count if char_count > 0 else 0.0

    suspicious_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    suspicious_ratio = suspicious_chars / char_count if char_count > 0 else 0.0

    # Base score start at 100
    score = 100.0

    if alpha_ratio < 0.65:
        score -= 25.0
    elif alpha_ratio < 0.50:
        score -= 40.0

    if suspicious_ratio > 0.15:
        score -= 20.0
    if suspicious_ratio > 0.30:
        score -= 30.0

    if word_count < 60:
        score -= 30.0
    elif word_count < 120:
        score -= 15.0

    if "OCR" in parsing_method:
        score -= 10.0

    final_score = max(0, min(100, int(score)))

    warning = None
    if final_score < 60:
        warning = "Warning: Resume extraction quality is low (<60%). Some key information may not have been detected correctly due to formatting or document noise."

    return {
        "quality_score": final_score,
        "word_count": word_count,
        "char_count": char_count,
        "alphabetic_ratio": round(alpha_ratio, 2),
        "suspicious_ratio": round(suspicious_ratio, 2),
        "parsing_method": parsing_method,
        "warning": warning
    }
