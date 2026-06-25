import streamlit as st
import pandas as pd
import os
from src.parser import ResumeParser
from src.matcher import ResumeMatcher

st.set_page_config(page_title="AI Resume Screener", layout="wide")
st.title("🎯 AI Resume Screener & Ranking System")

st.sidebar.header("Configuration")
threshold = st.sidebar.slider("Minimum Match Threshold (%)", 0, 100, 50)

# Paste Job Description
jd_input = st.text_area("Paste the Job Description (JD) here:", height=200)

# Upload Resumes
uploaded_files = st.file_uploader("Upload Candidate Resumes (PDF format)", type=["pdf"], accept_multiple_files=True)

if st.button("Analyze and Rank Resumes") and jd_input and uploaded_files:
    parser = ResumeParser()
    matcher = ResumeMatcher()
    
    results = []
    
    with st.spinner("Processing documents..."):
        for file in uploaded_files:
            # Temporary save file to process inside runtime data folder
            temp_path = os.path.join("data/resumes", file.name)
            with open(temp_path, "wb") as f:
                f.write(file.getbuffer())
            
            # Process & Match
            extracted_text = parser.extract_text_from_pdf(temp_path)
            match_score = matcher.calculate_similarity(extracted_text, jd_input)
            
            # Clean up temp file
            os.remove(temp_path)
            
            results.append({
                "Candidate Name": file.name.replace(".pdf", ""),
                "Match Score (%)": match_score,
                "Status": "Shortlisted" if match_score >= threshold else "Reviewed"
            })
            
    # Display Results as a sorted Dataframe
    df = pd.DataFrame(results).sort_values(by="Match Score (%)", ascending=False)
    st.subheader("📊 Candidate Ranking Dashboard")
    st.dataframe(df.style.background_gradient(cmap="Blues", subset=["Match Score (%)"]))
