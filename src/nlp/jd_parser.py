import re
from src.nlp.skill_extractor import SkillExtractor

REQUIRED_KEYWORDS = [
    "must have", "required", "requirements", "essential", "minimum qualifications",
    "mandatory", "prerequisites", "key requirements"
]

PREFERRED_KEYWORDS = [
    "preferred", "nice to have", "plus", "desirable", "bonus", "good to have",
    "optional", "secondary qualifications"
]

def parse_job_description(jd_text: str) -> dict:
    """
    Parses raw Job Description text into structured profile:
    - role_title
    - required_skills
    - preferred_skills
    - target_years_experience
    - required_education
    """
    if not jd_text or not jd_text.strip():
        return {
            "role_title": "General Technical Role",
            "required_skills": [],
            "preferred_skills": [],
            "target_years_experience": 0,
            "required_education": ["Bachelor's Degree"]
        }

    lines = jd_text.split('\n')
    required_text_lines = []
    preferred_text_lines = []
    current_mode = "REQUIRED"

    for line in lines:
        line_lower = line.lower()
        if any(kw in line_lower for kw in PREFERRED_KEYWORDS):
            current_mode = "PREFERRED"
        elif any(kw in line_lower for kw in REQUIRED_KEYWORDS):
            current_mode = "REQUIRED"

        if current_mode == "REQUIRED":
            required_text_lines.append(line)
        else:
            preferred_text_lines.append(line)

    req_text = "\n".join(required_text_lines)
    pref_text = "\n".join(preferred_text_lines)

    extractor = SkillExtractor()
    all_skills = extractor.extract_skills(jd_text)

    # Distinguish required vs preferred skills
    req_skills = extractor.extract_skills(req_text)
    pref_skills = extractor.extract_skills(pref_text)

    # If preferred skills empty, assign remaining all_skills
    if not pref_skills and req_skills:
        pref_skills = [s for s in all_skills if s not in req_skills]
    elif not req_skills:
        req_skills = all_skills
        pref_skills = []

    # Extract target years of experience
    yoe_match = re.search(r'(\d+)\+?\s*(?:-\s*(\d+))?\s*(?:years|yrs)\b', jd_text, re.IGNORECASE)
    target_yoe = int(yoe_match.group(1)) if yoe_match else 2

    # Role Title Detection
    role_match = re.search(r'(?:Role|Title|Position|Job Title)[:\s]*([^\n]+)', jd_text, re.IGNORECASE)
    role_title = role_match.group(1).strip() if role_match else "Software & Data Professional"

    return {
        "role_title": role_title,
        "required_skills": req_skills,
        "preferred_skills": pref_skills,
        "all_jd_skills": all_skills,
        "target_years_experience": target_yoe
    }
