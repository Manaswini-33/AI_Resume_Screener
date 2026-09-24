import re

SECTION_HEADERS = {
    "SUMMARY": ["summary", "profile", "professional summary", "about me", "objective", "career objective"],
    "SKILLS": ["skills", "technical skills", "core competencies", "technologies", "expertise", "technical expertise", "skills & tools"],
    "EXPERIENCE": ["experience", "work experience", "professional experience", "employment history", "work history", "career history"],
    "PROJECTS": ["projects", "key projects", "academic projects", "personal projects", "technical projects"],
    "EDUCATION": ["education", "academic background", "academic qualifications", "education & credentials"],
    "CERTIFICATIONS": ["certifications", "licenses", "courses", "certifications & licenses", "professional certifications"],
    "ACHIEVEMENTS": ["achievements", "awards", "honors", "awards & achievements"]
}

def detect_sections(text: str) -> dict:
    """
    Segments raw resume text into normalized section blocks:
    SUMMARY, SKILLS, EXPERIENCE, PROJECTS, EDUCATION, CERTIFICATIONS, ACHIEVEMENTS, GENERAL.

    Returns dict mapping section keys to extracted text strings.
    """
    lines = text.split('\n')
    current_section = "GENERAL"
    sections = {key: [] for key in SECTION_HEADERS.keys()}
    sections["GENERAL"] = []

    for line in lines:
        clean_line = line.strip().lower()
        # Remove trailing colons, dashes or punctuation for header matching
        clean_header_candidate = re.sub(r'[:\-–—]+$', '', clean_line).strip()

        matched_header = None
        for sec_key, aliases in SECTION_HEADERS.items():
            if clean_header_candidate in aliases or any(clean_line.startswith(alias + ":") for alias in aliases):
                matched_header = sec_key
                break

        if matched_header:
            current_section = matched_header
        else:
            sections[current_section].append(line)

    result = {}
    for k, v in sections.items():
        joined_text = "\n".join(v).strip()
        if joined_text:
            result[k] = joined_text

    return result
