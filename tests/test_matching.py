from src.matching.feature_engineering import FeatureEngineer

def test_feature_engineering():
    fe = FeatureEngineer()
    resume = "Experienced ML Engineer specializing in Python, SQL, and XGBoost. Developed prediction models."
    jd = "Looking for ML Engineer with Python and SQL experience."

    vector = fe.build_feature_vector(resume, jd)
    assert "semantic_similarity" in vector
    assert "required_skill_match" in vector
    assert vector["semantic_similarity"] >= 0.0
    assert vector["required_skill_match"] > 0.0
