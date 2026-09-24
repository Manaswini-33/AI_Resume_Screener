import pandas as pd

def analyze_skill_gaps(candidate_skills: list[str], required_skills: list[str], preferred_skills: list[str], evidence_records: list[dict]) -> dict:
    """
    Detailed skill-gap breakdown distinguishing matched vs missing required/preferred skills
    and flagging skills with WEAK evidence.
    """
    cand_set = set([s.lower() for s in candidate_skills])
    req_set = set([s.lower() for s in required_skills])
    pref_set = set([s.lower() for s in preferred_skills])

    matched_req = [s for s in required_skills if s.lower() in cand_set]
    missing_req = [s for s in required_skills if s.lower() not in cand_set]

    matched_pref = [s for s in preferred_skills if s.lower() in cand_set]
    missing_pref = [s for s in preferred_skills if s.lower() not in cand_set]

    weak_evidence_skills = []
    for rec in evidence_records:
        if rec.get("evidence_level") == "WEAK":
            weak_evidence_skills.append(rec.get("skill"))

    req_match_pct = round(len(matched_req) / len(required_skills) * 100, 1) if required_skills else 100.0

    return {
        "matched_required_skills": matched_req,
        "missing_required_skills": missing_req,
        "matched_preferred_skills": matched_pref,
        "missing_preferred_skills": missing_pref,
        "weak_evidence_skills": weak_evidence_skills,
        "required_match_percentage": req_match_pct
    }

def simulate_what_if_analysis(current_features: dict, ranking_model, missing_skills_to_add: list[str], total_required_skills: int) -> dict:
    """
    Counterfactual 'What-If' Simulation:
    Simulates alignment score change if candidate adds targeted missing skills with STRONG project evidence.
    """
    if not current_features:
        return {"simulated_score": 0.0, "score_delta": 0.0}

    modified = current_features.copy()

    # Simulate improvement in required skill match & evidence coverage
    added_count = len(missing_skills_to_add)
    if total_required_skills > 0:
        current_req_ratio = modified.get("required_skill_match", 0.5)
        new_req_ratio = min(1.0, current_req_ratio + (added_count / total_required_skills))
        modified["required_skill_match"] = new_req_ratio

    # Boost evidence strength and semantic match slightly
    modified["evidence_strength"] = min(1.0, modified.get("evidence_strength", 0.5) + 0.15)
    modified["evidence_coverage"] = min(1.0, modified.get("evidence_coverage", 0.5) + 0.20)
    modified["skill_match_ratio"] = min(1.0, modified.get("skill_match_ratio", 0.5) + 0.15)

    df_mod = pd.DataFrame([modified])
    simulated_score = ranking_model.predict_alignment(df_mod)

    df_orig = pd.DataFrame([current_features])
    original_score = ranking_model.predict_alignment(df_orig)

    score_delta = simulated_score - original_score

    return {
        "original_score": round(original_score, 4),
        "simulated_score": round(simulated_score, 4),
        "score_delta": round(score_delta, 4),
        "delta_percentage": f"+{round(score_delta * 100, 1)}%",
        "simulated_skills": missing_skills_to_add,
        "disclaimer": "Note: Simulation shows potential model score change assuming demonstrated project evidence. Does not guarantee selection."
    }
