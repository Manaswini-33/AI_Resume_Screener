# HireSense — Explainable, Evidence-Based & Fair AI Resume Intelligence System

**HireSense** is a state-of-the-art, human-in-the-loop AI resume intelligence and candidate alignment platform designed for transparent, evidence-backed recruitment decision support.

---

## 🌟 Key Architectural Pillars

- **Intelligent Multi-Stage PDF Parser**: PyMuPDF primary parser with `pdfplumber` and `Tesseract OCR` fallbacks.
- **Extraction Quality Evaluator**: Computes empirical document extraction quality (0-100%) and emits warnings for noisy scans.
- **100% PII Anonymization**: Redacts names, emails, phone numbers, addresses, and dates of birth (`[NAME]`, `[EMAIL]`, `[PHONE]`, `[ADDRESS]`, `[YEAR]`) prior to feature extraction.
- **Canonical Skill Ontology & Normalization**: Maps skill aliases ("py", "k8s", "ml") to canonical entities ("Python", "Kubernetes", "Machine Learning") using `data/skill_ontology.json`.
- **Evidence-Based Skill Verification Engine**: Classifies skill evidence into `VERY_STRONG`, `STRONG`, `MODERATE`, and `WEAK` based on surrounding action verbs and quantitative metrics.
- **Adversarial Keyword Stuffing Defense**: Penalizes abnormal term repetition and un-evidenced keyword spamming.
- **13 Multi-Factor Feature Vector**: Combines document semantic similarity, section-level relevance, skill match ratio, evidence coverage, TF-IDF n-grams, and extraction quality.
- **Supervised XGBoost Ranking Model**: Trained classifier evaluating holistic candidate fit.
- **SHAP & Human-Readable Explainability**: Quantifies feature contributions and generates intuitive HR rationale.
- **Skill Gap & What-If Counterfactual Simulator**: Identifies missing skills and projects score changes if skills are acquired with project evidence.
- **Demographic Neutrality Audit**: Live counterfactual test suite verifying zero score variance across demographic identity markers.
- **Ethical Human-in-the-Loop Decision Support**: Categorizes candidates into transparent recommendation tiers (`Recommended for Interview`, `Further Review Recommended`, `Low Alignment`) without autonomous rejection.

---

## 📁 Project Structure

```
AI_Resume_Screener/
│
├── app.py                      # Multi-tab Interactive Streamlit UI
├── train.py                    # XGBoost & Random Forest Model Training Script
├── evaluate.py                 # Standalone Model Evaluation Script
├── UNIQUE_PROJECT_POINTS.txt   # Comparative Differentiation Analysis & Viva Guide
├── requirements.txt            # Project Dependencies
├── README.md                   # Project Documentation
├── .gitignore                  # Git Ignore Rules
├── .env.example                # Environment Variable Template
│
├── src/
│   ├── parser/                 # PDF Parsing, OCR & Extraction Quality
│   │   ├── pdf_parser.py
│   │   ├── section_detector.py
│   │   └── extraction_quality.py
│   ├── nlp/                    # Skill Extraction, Normalization & Evidence
│   │   ├── skill_extractor.py
│   │   ├── skill_normalizer.py
│   │   ├── evidence_extractor.py
│   │   └── jd_parser.py
│   ├── matching/               # Embeddings & 13 Multi-Factor Signals
│   │   ├── embeddings.py
│   │   └── feature_engineering.py
│   ├── ranking/                # Supervised Ranking Model & Decision Tiers
│   │   └── ranking_model.py
│   ├── explainability/         # SHAP Feature Contribution & Rationale
│   │   └── explanation_engine.py
│   ├── fairness/               # PII Redaction & Demographic Audit
│   │   ├── pii_anonymizer.py
│   │   ├── fairness_audit.py
│   │   └── bias_tests.py
│   ├── skill_gap/              # Gap Analysis & What-If Simulator
│   │   └── gap_analyzer.py
│   ├── robustness/             # Keyword Stuffing & Adversarial Testing
│   │   └── robustness_tests.py
│   └── utils/                  # Text Cleaning Helpers
│       └── text_utils.py
│
├── data/                       # Skill Ontology & Dictionary
│   ├── skill_ontology.json
│   └── skills_dictionary.json
│
├── models/                     # Saved ML Models & Metadata
│   ├── ranking_model.pkl
│   └── model_metadata.json
│
└── tests/                      # Pytest Automated Test Suite
    ├── test_parser.py
    ├── test_skills.py
    ├── test_matching.py
    ├── test_evidence.py
    ├── test_fairness.py
    └── test_robustness.py
```

---

## 🚀 Quick Start Guide

### 1. Installation

```bash
# Clone repository
git clone https://github.com/Manaswini-33/AI_Resume_Screener.git
cd AI_Resume_Screener

# Install requirements
pip install -r requirements.txt
```

### 2. Train Ranking Model

```bash
python train.py
```

### 3. Run Streamlit Application

```bash
streamlit run app.py
```

### 4. Run Automated Test Suite

```bash
pytest tests/
```

---

## 📜 License & Governance

Designed for ethical AI recruitment support. Strictly complies with privacy guidelines via 100% PII anonymization and human-in-the-loop oversight.
