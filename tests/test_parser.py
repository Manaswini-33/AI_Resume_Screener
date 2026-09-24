from src.parser.section_detector import detect_sections
from src.parser.extraction_quality import compute_extraction_quality

def test_section_detection():
    sample_text = """
    John Doe
    john@example.com

    SUMMARY
    Experienced Machine Learning Engineer with 3+ years in AI.

    SKILLS
    Python, SQL, PyTorch, Docker, AWS

    EXPERIENCE
    Senior ML Engineer - Acme Corp
    - Developed house price prediction model using Python and XGBoost.
    - Improved model inference speed by 35%.

    EDUCATION
    B.Tech in Computer Science
    """
    sections = detect_sections(sample_text)
    assert "SKILLS" in sections
    assert "EXPERIENCE" in sections
    assert "EDUCATION" in sections
    assert "Python" in sections["SKILLS"]

def test_extraction_quality():
    good_text = """
    Senior Machine Learning Engineer with extensive experience designing, training, and deploying scalable artificial intelligence models.
    Proficient in Python, SQL, PyTorch, Scikit-Learn, and AWS cloud infrastructure. Built end-to-end data processing pipelines,
    optimized SQL database queries, and implemented containerized deployments using Docker and Kubernetes. Demonstrated strong problem-solving
    skills, team leadership, and track record of delivering high-impact machine learning systems for enterprise applications.
    """
    res = compute_extraction_quality(good_text, "PyMuPDF")
    assert res["quality_score"] >= 70
    assert res["warning"] is None

    empty_res = compute_extraction_quality("", "PyMuPDF")
    assert empty_res["quality_score"] == 0
    assert empty_res["warning"] is not None
