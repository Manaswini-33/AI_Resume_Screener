import json
import os
import re
import logging

logger = logging.getLogger(__name__)

class SkillNormalizer:
    def __init__(self, ontology_path: str = "data/skill_ontology.json"):
        self.alias_map = {}
        self.canonical_details = {}

        if os.path.exists(ontology_path):
            try:
                with open(ontology_path, "r", encoding="utf-8") as f:
                    self.ontology = json.load(f)

                for key, data in self.ontology.items():
                    canonical = data.get("canonical", key.title())
                    self.canonical_details[canonical] = data
                    self.alias_map[key.lower()] = canonical
                    self.alias_map[canonical.lower()] = canonical
                    for alias in data.get("aliases", []):
                        self.alias_map[alias.lower()] = canonical
            except Exception as e:
                logger.error(f"Failed to load skill ontology from {ontology_path}: {e}")
        else:
            logger.warning(f"Ontology file not found at {ontology_path}. Using fallback normalization.")

    def normalize_skill(self, raw_skill: str) -> str:
        """
        Maps a raw skill term or alias to its canonical skill name.
        """
        if not raw_skill:
            return ""
        clean = raw_skill.strip().lower()
        return self.alias_map.get(clean, raw_skill.strip().title())

    def get_skill_category(self, canonical_skill: str) -> str:
        """
        Returns category of canonical skill (e.g. 'AI & Data Science', 'Databases & Storage').
        """
        details = self.canonical_details.get(canonical_skill, {})
        return details.get("category", "Technical Skills")

    def extract_canonical_skills(self, text: str) -> list[dict]:
        """
        Scans input text against the ontology alias map and returns detected canonical skills.
        """
        if not text:
            return []

        text_lower = text.lower()
        detected = []
        seen_canonicals = set()

        for term, canonical in self.alias_map.items():
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, text_lower):
                if canonical not in seen_canonicals:
                    seen_canonicals.add(canonical)
                    detected.append({
                        "matched_term": term,
                        "canonical": canonical,
                        "category": self.get_skill_category(canonical)
                    })

        return detected
