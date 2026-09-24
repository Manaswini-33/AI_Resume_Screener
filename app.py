import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import io

# Import core HireSense modules
from src.parser.pdf_parser import extract_text_with_fallbacks
from src.parser.extraction_quality import compute_extraction_quality
from src.parser.section_detector import detect_sections
from src.fairness.pii_anonymizer import anonymize_pii, extract_anonymized_metadata
from src.nlp.skill_normalizer import SkillNormalizer
from src.nlp.skill_extractor import SkillExtractor
from src.nlp.jd_parser import parse_job_description
from src.nlp.evidence_extractor import analyze_all_skills_evidence, verify_skill_evidence
from src.matching.feature_engineering import FeatureEngineer
from src.ranking.ranking_model import ResumeRankingModel, get_human_recommendation
from src.explainability.explanation_engine import ExplanationEngine
from src.skill_gap.gap_analyzer import analyze_skill_gaps, simulate_what_if_analysis
from src.fairness.bias_tests import run_counterfactual_fairness_test
from src.fairness.fairness_audit import generate_fairness_audit_report
from src.robustness.robustness_tests import detect_keyword_stuffing, run_adversarial_robustness_test

# --- Page Configuration & Custom Styling ---
st.set_page_config(
    page_title="HireSense | AI Resume Intelligence System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.05rem;
        font-weight: 500;
        color: #64748B;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .pii-badge {
        background-color: #E0F2FE;
        color: #0369A1;
        font-size: 0.8rem;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-very-strong { background-color: #DCFCE7; color: #15803D; font-weight: 600; padding: 2px 8px; border-radius: 4px; }
    .badge-strong { background-color: #E0E7FF; color: #4338CA; font-weight: 600; padding: 2px 8px; border-radius: 4px; }
    .badge-moderate { background-color: #FEF9C3; color: #A16207; font-weight: 600; padding: 2px 8px; border-radius: 4px; }
    .badge-weak { background-color: #FEE2E2; color: #B91C1C; font-weight: 600; padding: 2px 8px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# --- Resource Caching ---
@st.cache_resource
def load_system_components():
    model_wrapper = ResumeRankingModel()
    feature_eng = FeatureEngineer()
    explainer = ExplanationEngine(model=model_wrapper.model)
    normalizer = SkillNormalizer()
    extractor = SkillExtractor()
    return model_wrapper, feature_eng, explainer, normalizer, extractor

model_wrapper, feature_eng, explainer, normalizer, extractor = load_system_components()

# --- Pre-loaded Demo Data ---
DEMO_JD = """Role: Senior Machine Learning Engineer
Requirements:
- Strong proficiency in Python, SQL, and Machine Learning algorithms.
- Experience with XGBoost, Scikit-Learn, and Deep Learning (PyTorch or TensorFlow).
- 3+ years of experience building and deploying end-to-end predictive models.
- Bachelor's degree in Computer Science, Data Science, or related field.

Preferred Qualifications:
- Cloud infrastructure experience with AWS (EC2, S3, SageMaker).
- Containerization using Docker and Kubernetes.
- Natural Language Processing (NLP) or Computer Vision experience.
"""

DEMO_RESUMES = {
    "Candidate 1 (Strong ML Engineer)": """
    Alexander Wright
    alex.wright@email.com | +1 (555) 234-5678 | San Francisco, CA | https://linkedin.com/in/alexwright

    SUMMARY
    Senior Machine Learning Engineer with 4 years of hands-on experience designing, training, and deploying scalable predictive models and NLP applications.

    TECHNICAL SKILLS
    Languages: Python, SQL, C++
    ML & AI: Machine Learning, XGBoost, Scikit-Learn, PyTorch, Deep Learning, Natural Language Processing
    Cloud & DevOps: AWS (EC2, S3), Docker, Git

    PROFESSIONAL EXPERIENCE
    Machine Learning Engineer | TechCorp Inc. (2022 - Present)
    - Architected and deployed an end-to-end house price prediction model using Python and XGBoost, improving inference speed by 35%.
    - Trained transformer NLP models for text classification, achieving 94.2% F1-score on customer feedback datasets.
    - Optimized SQL queries and data processing pipelines handling 10M+ daily records.

    PERSONAL PROJECTS
    Resume Intelligence Pipeline
    - Developed a semantic matching engine using PyTorch and Sentence-Transformers. Containerized using Docker and deployed on AWS EC2.

    EDUCATION
    B.S. in Computer Science | Stanford University (2020)
    """,

    "Candidate 2 (Moderate Data Analyst)": """
    Sarah Jenkins
    sarah.j@email.com | +1 (555) 987-6543 | Chicago, IL

    SUMMARY
    Data Analyst with 2 years of experience in SQL querying, data visualization, and exploratory data analysis.

    SKILLS
    SQL, Python, Pandas, Tableau, Power BI, Excel

    WORK EXPERIENCE
    Data Analyst | DataInsights LLC (2022 - Present)
    - Wrote complex SQL queries to extract multi-table datasets for business intelligence reporting.
    - Built interactive dashboards in Tableau and Excel for executive stakeholders.
    - Performed statistical data cleaning and exploratory data analysis using Python and Pandas.

    EDUCATION
    B.A. in Statistics | Northwestern University (2022)
    """,

    "Candidate 3 (Keyword Stuffed Profile)": """
    Michael Scott
    mscott@paperco.com | +1 (555) 111-2222

    SUMMARY
    Python Machine Learning Engineer Python Machine Learning SQL AWS Docker XGBoost PyTorch.

    SKILLS
    Python, Python, Python, Machine Learning, Machine Learning, Machine Learning, SQL, SQL, AWS, AWS, Docker, Docker, XGBoost, XGBoost, PyTorch, PyTorch, Deep Learning, Deep Learning, NLP, NLP.

    EXPERIENCE
    Worked with Python and Machine Learning. Used SQL and AWS. Python Machine Learning XGBoost Docker PyTorch SQL.
    """
}

# --- Header & System Status Bar ---
st.markdown('<div class="main-title">HireSense</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Explainable, Evidence-Based & Fair AI Resume Intelligence System | Human-in-the-Loop Decision Support</div>', unsafe_allow_html=True)

# System Governance Status Banner
status_col1, status_col2, status_col3, status_col4, status_col5 = st.columns(5)
with status_col1:
    st.markdown('<div class="metric-card"><div class="metric-value">XGBoost</div><div class="metric-label">Active Ranking Model</div></div>', unsafe_allow_html=True)
with status_col2:
    st.markdown('<div class="metric-card"><div class="metric-value">14 Stages</div><div class="metric-label">Pipeline Abstraction</div></div>', unsafe_allow_html=True)
with status_col3:
    st.markdown('<div class="metric-card"><div class="metric-value">100% Redacted</div><div class="metric-label">PII Anonymization</div></div>', unsafe_allow_html=True)
with status_col4:
    st.markdown('<div class="metric-card"><div class="metric-value">MiniLM-L6</div><div class="metric-label">Semantic Embedder</div></div>', unsafe_allow_html=True)
with status_col5:
    st.markdown('<div class="metric-card"><div class="metric-value">0.00%</div><div class="metric-label">Demographic Variance</div></div>', unsafe_allow_html=True)

st.markdown("---")

# Sidebar Configuration
st.sidebar.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=70)
st.sidebar.title("HireSense Control Center")
st.sidebar.markdown("**System Governance Settings**")

enable_pii_redaction = st.sidebar.toggle("Enforce PII Anonymization", value=True, help="Redacts names, emails, phone numbers, addresses, and birth years before feature extraction.")
enable_evidence_verification = st.sidebar.toggle("Evidence-Based Skill Verification", value=True, help="Validates whether claimed skills have surrounding action verbs and metrics.")
enable_repetition_penalty = st.sidebar.toggle("Keyword Stuffing Penalty", value=True, help="Penalizes abnormal term repetition and un-evidenced keyword spam.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Version:** XGBoost v1.2.0")
st.sidebar.markdown("**Last Trained:** 2026-09-24")

# Main Navigation Tabs
tab_screening, tab_explain, tab_gap, tab_fairness, tab_governance = st.tabs([
    "🎯 Candidate Screening",
    "🔍 Explainable Analysis & Evidence",
    "🧩 Skill Gap & What-If",
    "🛡️ Privacy, Fairness & Audit",
    "📊 Architecture & Model Specs"
])

# ==============================================================================
# TAB 1: CANDIDATE SCREENING
# ==============================================================================
with tab_screening:
    st.header("Candidate Screening & Multidimensional Alignment")
    st.markdown("Upload resumes or use preset demo candidates to view transparent, evidence-backed alignment rankings.")

    col_jd, col_res = st.columns([1, 1])

    with col_jd:
        st.subheader("1. Job Description")
        use_demo_jd = st.checkbox("Use Demo Machine Learning Engineer JD", value=True)
        if use_demo_jd:
            jd_input = st.text_area("Job Profile Text", value=DEMO_JD, height=220)
        else:
            jd_input = st.text_area("Paste Custom Job Description", height=220, placeholder="Paste job description here...")

        parsed_jd = parse_job_description(jd_input)
        with st.expander("📌 Extracted Job Profile Abstraction", expanded=False):
            st.markdown(f"**Target Role:** {parsed_jd['role_title']}")
            st.markdown(f"**Required Skills ({len(parsed_jd['required_skills'])}):** " + ", ".join(parsed_jd['required_skills']))
            st.markdown(f"**Preferred Skills ({len(parsed_jd['preferred_skills'])}):** " + ", ".join(parsed_jd['preferred_skills']))
            st.markdown(f"**Target Experience:** {parsed_jd['target_years_experience']}+ Years")

    with col_res:
        st.subheader("2. Resumes Source")
        res_mode = st.radio("Select Input Source:", ["Use Pre-loaded Demo Candidates", "Upload Custom PDF Resumes"], horizontal=True)

        candidate_data = []

        if res_mode == "Use Pre-loaded Demo Candidates":
            st.info("Using 3 diverse demo resumes: Candidate 1 (Strong Fit), Candidate 2 (Moderate Fit), Candidate 3 (Keyword Stuffed).")
            for label, text in DEMO_RESUMES.items():
                candidate_data.append({"filename": label, "raw_text": text, "method": "Direct Text"})
        else:
            uploaded_files = st.file_uploader("Upload PDF Resumes", type=["pdf"], accept_multiple_files=True)
            if uploaded_files:
                for file in uploaded_files:
                    raw_text, method = extract_text_with_fallbacks(file.read())
                    candidate_data.append({"filename": file.name, "raw_text": raw_text, "method": method})

    st.markdown("---")
    analyze_btn = st.button("⚡ Execute Intelligent Multi-Factor Analysis", type="primary", use_container_width=True)

    if analyze_btn or "screened_results" in st.session_state:
        if analyze_btn:
            if not jd_input.strip() or not candidate_data:
                st.warning("Please provide a Job Description and select/upload at least one candidate resume.")
                st.stop()

            results = []

            for idx, cand in enumerate(candidate_data):
                cand_id = f"Candidate #{idx+1}"
                raw_text = cand["raw_text"]
                parsing_method = cand["method"]

                # 1. Extraction Quality
                quality_res = compute_extraction_quality(raw_text, parsing_method)

                # 2. PII Anonymization
                anonymized_text = anonymize_pii(raw_text) if enable_pii_redaction else raw_text
                pii_meta = extract_anonymized_metadata(raw_text)

                # 3. Section Detection
                sections = detect_sections(anonymized_text)

                # 4. Feature Vector Engineering
                feats = feature_eng.build_feature_vector(anonymized_text, jd_input, extraction_quality=quality_res["quality_score"])
                df_feats = pd.DataFrame([feats])

                # 5. XGBoost Probability Inference
                prob_score = model_wrapper.predict_alignment(df_feats)

                # 6. Skill & Evidence Extraction
                cand_skills = extractor.extract_skills(anonymized_text)
                evidence_records = analyze_all_skills_evidence(cand_skills, sections)

                # 7. Skill Gap Analysis
                gap_res = analyze_skill_gaps(cand_skills, parsed_jd["required_skills"], parsed_jd["preferred_skills"], evidence_records)

                # 8. Human Recommendation Tier
                rec_label, rec_emoji, rec_desc = get_human_recommendation(prob_score)

                # 9. Explainability Rationale
                readable_exp = explainer.generate_human_readable_explanation(df_feats, gap_res["matched_required_skills"], gap_res["missing_required_skills"], evidence_records)

                # 10. Keyword Stuffing Check
                robustness_res = detect_keyword_stuffing(raw_text)

                results.append({
                    "cand_id": cand_id,
                    "filename": cand["filename"],
                    "anonymized_text": anonymized_text,
                    "overall_score": prob_score,
                    "overall_score_pct": f"{round(prob_score * 100, 1)}%",
                    "skill_alignment_pct": f"{round(feats['skill_match_ratio'] * 100, 1)}%",
                    "exp_relevance_pct": f"{round(feats['experience_semantic_match'] * 100, 1)}%",
                    "proj_relevance_pct": f"{round(feats['project_semantic_match'] * 100, 1)}%",
                    "evidence_strength_pct": f"{round(feats['evidence_strength'] * 100, 1)}%",
                    "semantic_sim_pct": f"{round(feats['semantic_similarity'] * 100, 1)}%",
                    "quality_score": quality_res["quality_score"],
                    "recommendation_label": rec_label,
                    "recommendation_emoji": rec_emoji,
                    "recommendation_desc": rec_desc,
                    "features_dict": feats,
                    "feature_df": df_feats,
                    "sections_dict": sections,
                    "extracted_skills": cand_skills,
                    "evidence_records": evidence_records,
                    "gap_results": gap_res,
                    "readable_exp": readable_exp,
                    "robustness_res": robustness_res,
                    "pii_meta": pii_meta
                })

            st.session_state["screened_results"] = results
            st.session_state["parsed_jd"] = parsed_jd
            st.session_state["jd_input"] = jd_input

        results = st.session_state["screened_results"]

        st.subheader("Candidate Alignment Leaderboard")

        # Display Summary Overview Table
        summary_rows = []
        for r in results:
            summary_rows.append({
                "Candidate ID": r["cand_id"],
                "Anonymized Identifier": f"[NAME] (Redacted - {r['filename']})",
                "Overall Alignment": r["overall_score_pct"],
                "Skill Alignment": r["skill_alignment_pct"],
                "Experience Relevance": r["exp_relevance_pct"],
                "Project Relevance": r["proj_relevance_pct"],
                "Evidence Strength": r["evidence_strength_pct"],
                "Extraction Quality": f"{r['quality_score']}%",
                "Human Recommendation": f"{r['recommendation_emoji']} {r['recommendation_label']}"
            })

        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

        st.markdown("### Candidate Cards & Abstraction Profiles")
        for r in results:
            with st.expander(f"{r['recommendation_emoji']} {r['cand_id']} — Overall Alignment: {r['overall_score_pct']} | {r['recommendation_label']}", expanded=True):
                c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1.2])

                with c1:
                    st.markdown(f"**Candidate Identifier:** <span class='pii-badge'>[NAME] REDACTED</span>", unsafe_allow_html=True)
                    st.markdown(f"**Original File:** `{r['filename']}`")
                    st.markdown(f"**PII Status:** Anonymized ({r['pii_meta']['emails_redacted']} emails, {r['pii_meta']['phones_redacted']} phones masked)")
                    st.markdown(f"**Extraction Quality:** `{r['quality_score']}%`")

                with c2:
                    st.metric("Skill Alignment", r["skill_alignment_pct"])
                    st.metric("Experience Relevance", r["exp_relevance_pct"])

                with c3:
                    st.metric("Project Relevance", r["proj_relevance_pct"])
                    st.metric("Evidence Strength", r["evidence_strength_pct"])

                with c4:
                    st.markdown(f"**Recommendation Tier:**\n### {r['recommendation_emoji']} {r['recommendation_label']}")
                    st.caption(r["recommendation_desc"])

                st.markdown("**Key Alignment Rationale:**")
                for s in r["readable_exp"]["strengths"]:
                    st.markdown(s)
                for a in r["readable_exp"]["areas_for_review"]:
                    st.markdown(a)

# ==============================================================================
# TAB 2: EXPLAINABLE ANALYSIS & EVIDENCE GRAPH
# ==============================================================================
with tab_explain:
    st.header("🔍 Transparent Analysis & Evidence Graph")
    st.markdown("Inspect how the AI system derived the score for a specific candidate with complete model transparency and verified evidence.")

    if "screened_results" not in st.session_state:
        st.info("Please run candidate screening in Tab 1 first.")
    else:
        results = st.session_state["screened_results"]
        cand_choice = st.selectbox("Select Candidate to Inspect:", [r["cand_id"] + f" ({r['filename']})" for r in results])
        selected_cand = [r for r in results if r["cand_id"] in cand_choice][0]

        st.subheader(f"Analysis Breakdown: {selected_cand['cand_id']} ([NAME] Redacted)")

        # Pipeline Stepper Visualization
        st.markdown("#### End-to-End Execution Trace")
        step_cols = st.columns(6)
        step_cols[0].markdown("**1. Parse**\nPyMuPDF Clean")
        step_cols[1].markdown(f"**2. Quality**\n{selected_cand['quality_score']}% Score")
        step_cols[2].markdown("**3. Privacy**\nPII Masked")
        step_cols[3].markdown(f"**4. Skills**\n{len(selected_cand['extracted_skills'])} Extracted")
        step_cols[4].markdown(f"**5. Evidence**\n{selected_cand['evidence_strength_pct']} Strength")
        step_cols[5].markdown(f"**6. XGBoost**\n{selected_cand['overall_score_pct']} Score")

        st.markdown("---")

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.subheader("Feature Contribution Analysis (Model Interpretability)")
            df_contrib = explainer.generate_feature_contributions(selected_cand["feature_df"])

            fig_bar = px.bar(
                df_contrib,
                x="contribution_score",
                y="feature_name",
                orientation="h",
                color="impact",
                color_discrete_map={"Positive Boost": "#22C55E", "Area for Review": "#EF4444"},
                title="Multi-Factor Signal Contributions to Final Score",
                labels={"contribution_score": "Score Impact (+ Boost / - Review)", "feature_name": "Multi-Factor Signal"}
            )
            fig_bar.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_right:
            st.subheader("Evidence Verification Matrix")
            ev_list = selected_cand["evidence_records"]

            if ev_list:
                matrix_data = []
                for ev in ev_list:
                    level = ev["evidence_level"]
                    badge = f"<span class='badge-{level.lower().replace('_','-')}'>{level}</span>"
                    action_flag = "✓ Yes" if ev["has_action"] else "❌ No"
                    metric_flag = "✓ Yes" if ev["has_metric"] else "❌ No"
                    snippet = ev["contexts"][0] if ev["contexts"] else "Listed in skills section only"

                    matrix_data.append({
                        "Skill": ev["skill"],
                        "Evidence Level": level,
                        "Section": ev["section"],
                        "Action Verb": action_flag,
                        "Metric Included": metric_flag,
                        "Verified Context Snippet": snippet[:70] + "..." if len(snippet) > 70 else snippet
                    })

                st.dataframe(pd.DataFrame(matrix_data), use_container_width=True)
            else:
                st.info("No skills detected.")

        st.markdown("---")
        st.subheader("Evidence Graph Hierarchy Visualization")
        st.markdown("Connects canonical skills directly to demonstrated project and work experience snippets.")

        ev_graph_lines = []
        for ev in selected_cand["evidence_records"]:
            ev_graph_lines.append(f"📌 **Skill: {ev['skill']}** (Evidence: `{ev['evidence_level']}` in `{ev['section']}`)")
            if ev["contexts"]:
                for ctx in ev["contexts"]:
                    ev_graph_lines.append(f"   └── *Context Quote:* \"{ctx.strip()}\"")
            else:
                ev_graph_lines.append("   └── *Context Quote:* Claimed in skills list without supporting bullet points.")

        st.markdown("\n".join(ev_graph_lines[:15]))

# ==============================================================================
# TAB 3: SKILL GAP & WHAT-IF ANALYSIS
# ==============================================================================
with tab_gap:
    st.header("🧩 Skill Gap Intelligence & Counterfactual Simulation")
    st.markdown("Identify specific qualification gaps and run real-time simulations to project potential score improvements.")

    if "screened_results" not in st.session_state:
        st.info("Please run candidate screening in Tab 1 first.")
    else:
        results = st.session_state["screened_results"]
        parsed_jd = st.session_state["parsed_jd"]

        cand_choice = st.selectbox("Select Candidate for Skill Gap Analysis:", [r["cand_id"] + f" ({r['filename']})" for r in results], key="gap_cand_select")
        selected_cand = [r for r in results if r["cand_id"] in cand_choice][0]

        gap_res = selected_cand["gap_results"]

        col_g1, col_g2 = st.columns([1, 1])

        with col_g1:
            st.subheader("Required vs Matched Skills Breakdown")

            st.markdown(f"**Required Skill Coverage:** `{gap_res['required_match_percentage']}%`")

            st.markdown("##### ✅ Matched Required Skills:")
            if gap_res["matched_required_skills"]:
                for s in gap_res["matched_required_skills"]:
                    st.markdown(f"- 🟢 **{s}** (Verified in candidate profile)")
            else:
                st.markdown("*No required skills matched.*")

            st.markdown("##### ⚠️ Missing Required Skills:")
            if gap_res["missing_required_skills"]:
                for s in gap_res["missing_required_skills"]:
                    st.markdown(f"- 🔴 **{s}** (Not detected in resume)")
            else:
                st.markdown("*All required skills are matched!*")

        with col_g2:
            st.subheader("Radar Chart: Skill & Evidence Balance")

            categories = ['Required Skills', 'Preferred Skills', 'Evidence Strength', 'Experience Match', 'Project Match']
            cand_values = [
                gap_res['required_match_percentage'] / 100.0,
                selected_cand['features_dict']['preferred_skill_match'],
                selected_cand['features_dict']['evidence_strength'],
                selected_cand['features_dict']['experience_semantic_match'],
                selected_cand['features_dict']['project_semantic_match']
            ]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=cand_values,
                theta=categories,
                fill='toself',
                name=selected_cand['cand_id']
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                height=350,
                margin=dict(l=40, r=40, t=30, b=30)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("---")
        st.subheader("Counterfactual 'What-If' Alignment Simulator")
        st.markdown("Simulate how acquiring demonstrated project evidence for missing skills impacts the model's alignment prediction.")

        missing_options = gap_res["missing_required_skills"] + gap_res["missing_preferred_skills"]

        if missing_options:
            skills_to_simulate = st.multiselect("Select Missing Skills to Simulate Adding with Project Evidence:", missing_options, default=missing_options[:2] if missing_options else [])

            if st.button("Run Counterfactual Simulation"):
                sim_result = simulate_what_if_analysis(
                    selected_cand["features_dict"],
                    model_wrapper,
                    skills_to_simulate,
                    total_required_skills=len(parsed_jd["required_skills"])
                )

                col_s1, col_s2, col_s3 = st.columns(3)
                col_s1.metric("Current Model Score", f"{round(sim_result['original_score']*100, 1)}%")
                col_s2.metric("Simulated Projected Score", f"{round(sim_result['simulated_score']*100, 1)}%", delta=sim_result["delta_percentage"])
                col_s3.metric("Projected Score Delta", f"+{round(sim_result['score_delta']*100, 1)}%")

                st.caption(sim_result["disclaimer"])
        else:
            st.success("Candidate already matches all required and preferred skills!")

# ==============================================================================
# TAB 4: PRIVACY, FAIRNESS & AUDIT
# ==============================================================================
with tab_fairness:
    st.header("🛡️ Privacy, Fairness & Adversarial Robustness Audit")
    st.markdown("Empirical testing for demographic neutrality, PII redaction integrity, and resistance against keyword stuffing attacks.")

    col_f1, col_f2 = st.columns([1, 1])

    with col_f1:
        st.subheader("1. Counterfactual Demographic Neutrality Audit")
        st.markdown("Runs 6 identity marker variants (varying candidate names/emails across demographic backgrounds) through feature extraction and model scoring.")

        if st.button("Run Live Demographic Sensitivity Test"):
            sample_resume_text = DEMO_RESUMES["Candidate 1 (Strong ML Engineer)"]
            fairness_test_res = run_counterfactual_fairness_test(model_wrapper, feature_eng, sample_resume_text, DEMO_JD)

            if fairness_test_res["fairness_passed"]:
                st.success(f"✅ {fairness_test_res['status_message']}")
            else:
                st.error(f"❌ {fairness_test_res['status_message']}")

            st.markdown(f"**Max Score Variance across Demographic Markers:** `{fairness_test_res['max_score_variance']}`")

            st.dataframe(pd.DataFrame(fairness_test_res["detailed_results"]), use_container_width=True)

    with col_f2:
        st.subheader("2. Keyword Stuffing & Adversarial Attack Test")
        st.markdown("Evaluates whether artificial keyword repetition tricks the system or is successfully penalized.")

        if st.button("Run Adversarial Keyword Attack Test"):
            orig_text = DEMO_RESUMES["Candidate 1 (Strong ML Engineer)"]
            robustness_test_res = run_adversarial_robustness_test(model_wrapper, feature_eng, orig_text, DEMO_JD)

            if robustness_test_res["is_robust"]:
                st.success(f"✅ {robustness_test_res['assessment']}")
            else:
                st.warning(f"⚠️ {robustness_test_res['assessment']}")

            st.metric("Original Resume Score", f"{round(robustness_test_res['original_score']*100, 1)}%")
            st.metric("Adversarial Stuffed Score", f"{round(robustness_test_res['adversarial_stuffed_score']*100, 1)}%", delta=f"{round(robustness_test_res['score_delta']*100, 1)}%")

    st.markdown("---")
    st.subheader("3. Live PII Redaction Sandbox")
    st.markdown("Test the privacy anonymizer on any custom input text.")

    sandbox_input = st.text_area("Input Sample Resume Text with Personal Info:", value="John Smith | john.smith@email.com | Phone: +1-555-0199 | 123 Main St, New York, NY\nWebsite: https://johnsmith.dev", height=100)
    if sandbox_input:
        redacted_output = anonymize_pii(sandbox_input)
        st.markdown("**Redacted Output (Passed to Feature Engineering):**")
        st.code(redacted_output, language="text")

# ==============================================================================
# TAB 5: ARCHITECTURE & MODEL SPECS
# ==============================================================================
with tab_governance:
    st.header("📊 System Architecture & Model Governance Specs")
    st.markdown("Technical specification of the 14-stage HireSense pipeline and trained machine learning models.")

    st.subheader("HireSense 14-Stage Architectural Flow")

    pipeline_code = """
    PDF Resume
       │
       ▼
    [1. PyMuPDF / Fallback Parser] ──► [2. Quality Score Evaluator]
                                              │
                                              ▼
    [4. Section Segmentation] ◄── [3. PII Anonymizer Redaction]
       │
       ▼
    [5. Canonical Skill Normalizer] ──► [6. Evidence Verification Engine]
                                              │
                                              ▼
    [8. Job Profile Parser] ───────► [7. Section Semantic Embedder]
       │                                      │
       └──────────────────┬───────────────────┘
                          │
                          ▼
             [9. Multi-Factor Feature Vector] (13 Signals)
                          │
                          ▼
             [10. XGBoost Supervised Model]
                          │
                          ▼
      [11. Skill Gap] ──► [12. SHAP Explainability] ──► [13. Audit]
                                                             │
                                                             ▼
                                                [14. Human Review Recommendation]
    """
    st.code(pipeline_code, language="text")

    st.markdown("---")
    st.subheader("Trained Model Performance Metrics")

    meta = model_wrapper.metadata
    if meta:
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        metrics = meta.get("metrics", {})
        m_col1.metric("Accuracy", f"{round(metrics.get('Accuracy', 0)*100, 2)}%")
        m_col2.metric("Precision", f"{round(metrics.get('Precision', 0)*100, 2)}%")
        m_col3.metric("Recall", f"{round(metrics.get('Recall', 0)*100, 2)}%")
        m_col4.metric("F1-Score", f"{round(metrics.get('F1', 0)*100, 2)}%")
        m_col5.metric("ROC-AUC", f"{round(metrics.get('ROC-AUC', 0)*100, 2)}%")

        st.markdown("#### Feature Importance Ranking (XGBoost)")
        fi_df = pd.DataFrame(list(meta.get("feature_importances", {}).items()), columns=["Feature", "Importance"]).sort_values(by="Importance", ascending=True)

        fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h", title="XGBoost Feature Importance Weights")
        fig_fi.update_layout(height=400)
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info("Model metadata not found.")

st.markdown("---")
st.caption("HireSense Resume Intelligence System | Undergraduate ML Project & Viva Ready | Designed for Ethical Human-in-the-Loop Decision Support")
