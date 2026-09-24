import json
import os
import re
from src.nlp.skill_normalizer import SkillNormalizer

class SkillExtractor:
    def __init__(self, dictionary_path: str = "data/skills_dictionary.json", ontology_path: str = "data/skill_ontology.json"):
        self.normalizer = SkillNormalizer(ontology_path=ontology_path)
        self.dictionary_skills = []

        if os.path.exists(dictionary_path):
            try:
                with open(dictionary_path, "r", encoding="utf-8") as f:
                    self.dictionary_skills = json.load(f)
            except Exception:
                pass

    def extract_skills(self, text: str) -> list[str]:
        """
        Extracts all unique canonical skills from text.
        """
        if not text:
            return []

        extracted_canonicals = set()

        # 1. Extract via Ontology
        ontology_matches = self.normalizer.extract_canonical_skills(text)
        for item in ontology_matches:
            extracted_canonicals.add(item["canonical"])

        # 2. Extract via Dictionary lookup
        text_lower = text.lower()
        for skill in self.dictionary_skills:
            pattern = r'\b' + re.escape(skill.lower()) + r'\b'
            if re.search(pattern, text_lower):
                canonical = self.normalizer.normalize_skill(skill)
                extracted_canonicals.add(canonical)

        return sorted(list(extracted_canonicals))
