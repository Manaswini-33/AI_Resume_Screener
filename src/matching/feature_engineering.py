import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.matching.embeddings import EmbeddingEngine
from src.parser.section_detector import detect_sections
from src.nlp.skill_extractor import SkillExtractor
from src.nlp.jd_parser import parse_job_description
from src.nlp.evidence_extractor import analyze_all_skills_evidence

class FeatureEngineer:
    def __init__(self):
        self.embedder = EmbeddingEngine()
        self.skill_extractor = SkillExtractor()

    def build_feature_vector(self, resume_text: str, jd_text: str, extraction_quality: float = 100.0) -> dict:
        """
        Calculates multi-factor candidate-JD alignment signals for supervised ML ranking.

        Returns feature dictionary with 13 quantitative signals.
        """
        sections = detect_sections(resume_text)
        jd_info = parse_job_description(jd_text)

        # 1. Semantic Similarity Signals
        doc_sem_sim = self.embedder.compute_similarity(resume_text, jd_text)
        exp_sem_sim = self.embedder.compute_similarity(sections.get("EXPERIENCE", resume_text), jd_text)
        proj_sem_sim = self.embedder.compute_similarity(sections.get("PROJECTS", resume_text), jd_text)

        # 2. TF-IDF Similarity Signal
        try:
            vec = TfidfVectorizer(stop_words='english', max_features=1000)
            tfidf_mat = vec.fit_transform([resume_text, jd_text])
            tfidf_sim = float(cosine_similarity(tfidf_mat[0:1], tfidf_mat[1:2])[0][0])
        except Exception:
            tfidf_sim = doc_sem_sim

        # 3. Skill & Requirement Alignment Signals
        cand_skills = self.skill_extractor.extract_skills(resume_text)
        req_skills = jd_info.get("required_skills", [])
        pref_skills = jd_info.get("preferred_skills", [])
        all_jd_skills = jd_info.get("all_jd_skills", [])

        cand_set = set([s.lower() for s in cand_skills])
        req_set = set([s.lower() for s in req_skills])
        pref_set = set([s.lower() for s in pref_skills])
        jd_set = set([s.lower() for s in all_jd_skills])

        req_matched = len(cand_set.intersection(req_set))
        req_match_ratio = req_matched / len(req_set) if req_set else 1.0

        pref_matched = len(cand_set.intersection(pref_set))
        pref_match_ratio = pref_matched / len(pref_set) if pref_set else 0.5

        overall_matched = len(cand_set.intersection(jd_set))
        skill_match_ratio = overall_matched / len(jd_set) if jd_set else 0.5

        # 4. Evidence Strength Signals
        evidence_records = analyze_all_skills_evidence(cand_skills, sections)
        if evidence_records:
            weights = [rec["numerical_weight"] for rec in evidence_records]
            avg_evidence_strength = float(np.mean(weights))
            strong_count = sum(1 for rec in evidence_records if rec["evidence_level"] in ["STRONG", "VERY_STRONG"])
            evidence_coverage = float(strong_count / len(evidence_records))
        else:
            avg_evidence_strength = 0.25
            evidence_coverage = 0.0

        # 5. Text Structure & Keyword Density Signals
        words = re.findall(r'\b\w+\b', resume_text.lower())
        total_words = len(words) if words else 1
        unique_words = len(set(words))
        uniq_ratio = unique_words / total_words
        rep_ratio = 1.0 - uniq_ratio

        jd_words = set(re.findall(r'\b\w+\b', jd_text.lower()))
        jd_matches = sum(1 for w in words if w in jd_words)
        kw_density = jd_matches / total_words

        return {
            "semantic_similarity": round(float(doc_sem_sim), 4),
            "experience_semantic_match": round(float(exp_sem_sim), 4),
            "project_semantic_match": round(float(proj_sem_sim), 4),
            "skill_match_ratio": round(float(skill_match_ratio), 4),
            "required_skill_match": round(float(req_match_ratio), 4),
            "preferred_skill_match": round(float(pref_match_ratio), 4),
            "evidence_strength": round(float(avg_evidence_strength), 4),
            "evidence_coverage": round(float(evidence_coverage), 4),
            "tfidf_similarity": round(float(tfidf_sim), 4),
            "unique_word_ratio": round(float(uniq_ratio), 4),
            "repetition_ratio": round(float(rep_ratio), 4),
            "keyword_density": round(float(kw_density), 4),
            "extraction_quality": round(float(extraction_quality / 100.0), 4)
        }
