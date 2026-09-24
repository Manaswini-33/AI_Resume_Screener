import re

ACTION_VERBS = [
    "developed", "built", "designed", "implemented", "engineered", "improved",
    "optimized", "created", "led", "architected", "deployed", "scaled", "automated",
    "trained", "fine-tuned", "analyzed", "reduced", "increased", "maintained"
]

METRIC_PATTERNS = [
    r'\d+%',                          # Percentages e.g. 25%
    r'\$\d+(?:,\d{3})*(?:\.\d+)?',     # Financial e.g. $50,000
    r'\b\d+\s*(?:years?|yrs?|months?)\b', # Time duration e.g. 3 years
    r'\b\d+x\b',                      # Multipliers e.g. 10x
    r'\b\d+\s*(?:GB|TB|MB|users|clients|models|datasets|queries)\b' # Scale indicators
]

def verify_skill_evidence(skill: str, sections_dict: dict) -> dict:
    """
    Evaluates contextual evidence strength for a detected skill across resume sections.

    Levels:
      - VERY_STRONG: Skill in PROJECTS/EXPERIENCE + Action Verb + Quantifiable Metric
      - STRONG: Skill in PROJECTS/EXPERIENCE + Action Verb OR Metric
      - MODERATE: Skill mentioned in PROJECTS/EXPERIENCE without action/metric
      - WEAK: Skill only listed in SKILLS or GENERAL section without context

    Returns:
      dict with skill, evidence_level, section, contexts, has_action, has_metric, numerical_weight
    """
    skill_lower = skill.lower()
    skill_pattern = r'\b' + re.escape(skill_lower) + r'\b'

    found_section = "SKILLS"
    best_level = "WEAK"
    matching_sentences = []
    has_action = False
    has_metric = False

    # Priority search through sections: PROJECTS / EXPERIENCE first, then SKILLS / GENERAL
    priority_order = ["PROJECTS", "EXPERIENCE", "SUMMARY", "SKILLS", "GENERAL"]

    for sec_name in priority_order:
        sec_text = sections_dict.get(sec_name, "")
        if not sec_text:
            continue

        sentences = re.split(r'[.\n•▪➢]', sec_text)
        for sent in sentences:
            sent_clean = sent.strip()
            if re.search(skill_pattern, sent_clean, flags=re.IGNORECASE):
                matching_sentences.append(sent_clean)
                found_section = sec_name

                # Check action verbs
                if any(re.search(r'\b' + re.escape(verb) + r'\b', sent_clean, re.IGNORECASE) for verb in ACTION_VERBS):
                    has_action = True

                # Check metrics
                if any(re.search(pattern, sent_clean, re.IGNORECASE) for pattern in METRIC_PATTERNS):
                    has_metric = True

    # Determine evidence level
    if found_section in ["PROJECTS", "EXPERIENCE"]:
        if has_action and has_metric:
            best_level = "VERY_STRONG"
        elif has_action or has_metric:
            best_level = "STRONG"
        else:
            best_level = "MODERATE"
    else:
        best_level = "WEAK"

    # Numerical weight mapping for ML feature engineering
    weight_map = {
        "WEAK": 0.25,
        "MODERATE": 0.50,
        "STRONG": 0.75,
        "VERY_STRONG": 1.00
    }

    return {
        "skill": skill,
        "evidence_level": best_level,
        "section": found_section,
        "contexts": matching_sentences[:2], # top 2 snippets
        "has_action": has_action,
        "has_metric": has_metric,
        "numerical_weight": weight_map[best_level]
    }

def analyze_all_skills_evidence(skills: list[str], sections_dict: dict) -> list[dict]:
    """
    Computes evidence analysis for every skill in candidate profile.
    """
    results = []
    for s in skills:
        results.append(verify_skill_evidence(s, sections_dict))
    return results
