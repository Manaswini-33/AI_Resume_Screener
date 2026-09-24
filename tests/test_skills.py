from src.nlp.skill_normalizer import SkillNormalizer
from src.nlp.skill_extractor import SkillExtractor
from src.nlp.jd_parser import parse_job_description

def test_skill_normalization():
    norm = SkillNormalizer()
    assert norm.normalize_skill("py") == "Python"
    assert norm.normalize_skill("ml") == "Machine Learning"
    assert norm.normalize_skill("k8s") == "Kubernetes"

def test_skill_extraction():
    extractor = SkillExtractor()
    text = "Proficient in Python3, ML models, PostgreSQL databases, and AWS deployment."
    skills = extractor.extract_skills(text)
    assert "Python" in skills
    assert "Machine Learning" in skills
    assert "SQL" in skills
    assert "AWS" in skills

def test_jd_parsing():
    jd = """
    Job Title: ML Engineer
    Requirements: Must have Python, SQL, and XGBoost.
    Preferred: AWS and Docker experience is a plus.
    Experience: 3+ years required.
    """
    info = parse_job_description(jd)
    assert "Python" in info["required_skills"]
    assert "AWS" in info["preferred_skills"]
    assert info["target_years_experience"] == 3
