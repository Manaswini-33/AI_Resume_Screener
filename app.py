import streamlit as st
import pandas as pd
import os
from src.parser import ResumeParser
from src.matcher import ResumeMatcher

# 1. Page Configuration
st.set_page_config(
    page_title="AI Talent Intel - Resume Screener",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Professional UI Styling (Slate & Sky Theme)
st.markdown("""
    <style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #F8FAFC;
    }
    
    /* Main Title Styling */
    .main-title {
        color: #1E293B;
        font-weight: 700;
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: #64748B;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }

    /* Professional Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1E293B;
        color: white;
    }
    
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
        color: #38BDF8;
    }

    /* Card Styling */
    .stMetric {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #E2E8F0;
    }

    /* Button Styling */
    .stButton>button {
        background-color: #3B82F6;
        color: white;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #2563EB;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    /* Custom Side Headings (as requested) */
    h3 {
        font-size: 26px !important;
        font-weight: 600 !important;
        color: #1E293B !important;
        border-left: 5px solid #3B82F6;
        padding-left: 15px;
        margin-top: 30px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Application Logic
def main():
    # Header Area
    st.markdown('<h1 class="main-title">Talent Intel AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Next-generation semantic resume ranking engine.</p>', unsafe_allow_html=True)

    # Sidebar Controls
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3850/3850285.png", width=80)
        st.title("Admin Console")
        st.divider()
        threshold = st.slider("Match Threshold (%)", 0, 100, 75)
        st.info("Profiles above this threshold will be marked as 'High Priority'.")
        
    # Main Dashboard Columns
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 📝 Job Requirement")
        jd_input = st.text_area("Paste the Job Description (JD) here:", height=300, placeholder="Identify core competencies and skills required...")

    with col2:
        st.markdown("### 📂 Applicant Pool")
        uploaded_files = st.file_uploader("Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)
        st.warning("All PII (Names, Emails, Phones) is automatically scrubbed for unbiased evaluation.")

    if st.button("Run AI Intelligence Analysis") and jd_input and uploaded_files:
        parser = ResumeParser()
        matcher = ResumeMatcher()
        
        results = []
        
        with st.status("Analyzing applicant pool...", expanded=True) as status:
            for file in uploaded_files:
                # Runtime directory management
                temp_path = os.path.join("data/resumes", file.name)
                with open(temp_path, "wb") as f:
                    f.write(file.getbuffer())
                
                # Logic Execution
                text = parser.extract_text_from_pdf(temp_path)
                score = matcher.calculate_similarity(text, jd_input)
                
                os.remove(temp_path) # Cleanup
                
                results.append({
                    "Candidate": file.name.replace(".pdf", ""),
                    "Score": score,
                    "Status": "🌟 Top Talent" if score >= threshold else "🔍 Shortlisted" if score >= 50 else "⚪ Reviewed"
                })
            status.update(label="Analysis Complete!", state="complete", expanded=False)

        # 4. Display Results
        st.divider()
        st.markdown("### 📊 Ranking Leaderboard")
        
        # Performance Summary Cards
        df = pd.DataFrame(results).sort_values(by="Score", ascending=False)
        
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Total Applicants", len(df))
        kpi2.metric("High Match Rate", f"{len(df[df['Score'] >= threshold])}")
        kpi3.metric("Avg. Match Score", f"{df['Score'].mean():.1f}%")

        # Results Table
        st.table(df.style.format({"Score": "{:.1f}%"}).background_gradient(cmap="Blues", subset=["Score"]))

if __name__ == "__main__":
    main()