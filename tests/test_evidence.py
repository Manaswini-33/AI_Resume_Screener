from src.parser.section_detector import detect_sections
from src.nlp.evidence_extractor import verify_skill_evidence

def test_evidence_verification():
    text = """
    PROJECTS
    Built a fraud detection model using Python and XGBoost that improved accuracy by 25%.
    """
    sections = detect_sections(text)
    ev = verify_skill_evidence("Python", sections)

    assert ev["evidence_level"] == "VERY_STRONG"
    assert ev["has_action"] is True
    assert ev["has_metric"] is True
