# 🎯 Enterprise AI Resume Screener & Ranking System

An enterprise-grade, privacy-aware AI tool designed to parse unstructured resume documents, mitigate human and algorithmic bias, and rank applicants contextually against specific Job Descriptions (JDs) using Natural Language Processing (NLP) and Semantic Embeddings.

---

## 📌 1. Comprehensive Problem Statements (PS)

### a.) Layout and Formatting Fragility
* **The Failure Mechanism:** Legacy Applicant Tracking Systems (ATS) rely on primitive serialization modules (such as `PyPDF2`) that flatten PDF documents by stripping text sequentially along horizontal coordinate lines from left to right.
* **The Business Impact:** Modern candidate resumes heavily utilize structural enhancements including dual-column layouts, sidebars for technical skills, tables for employment history, and stylized headers. When processed by standard horizontal extraction, the text across parallel columns merges into a corrupted, nonsensical string (e.g., mixing a company name on the left with a personal hobby listed on the right). This breaks the relational data structure, making it impossible for basic parsing algorithms to understand context, which results in the accidental rejection of highly qualified candidates.

### b.) The "Keyword Stuffing" Exploit
* **The Failure Mechanism:** Traditional screening tools evaluate resumes through exact-string matching and keyword frequency counting (TF-IDF methodologies). They act as basic text search engines rather than intelligent systems.
* **The Business Impact:** This creates a severe security and quality vulnerability. Candidates can easily reverse-engineer the system by copying the entire text of a job description and pasting it into their resume footer 50 times in a microscopic, white font. While invisible to human recruiters, the legacy parser detects a "perfect match" and pushes the fraudulent profile to the top of the pile. Conversely, genuine professionals who describe their achievements using sophisticated vocabulary or industry synonyms are penalized and filtered out simply because their word choice does not exactly match the recruiter's specific search terms.

### c.) Algorithmic Bias Loop
* **The Failure Mechanism:** Machine Learning models trained on historically unvetted human resource datasets tend to codify, internalize, and amplify existing human biases. The model treats non-merit factors as highly predictive features.
* **The Business Impact:** If historical corporate data shows a higher rate of historical promotion for male candidates from specific geographic postal codes or elite universities, standard classification models learn to score incoming applications based on these proxy variables. The system creates a self-fulfilling prophecy or "black box feedback loop" that systematically downgrades candidates based on gender markers, ethnic naming conventions, non-traditional career paths, or localized educational backgrounds. This stifles workplace diversity and leaves the enterprise vulnerable to compliance risks.

---

## 🎯 2. Detailed Project Objectives

* **Architect Structural Layout-Aware Parsing:** To design and deploy an advanced document-processing layer that bypasses string-flattening limits. This layer must achieve 100% text-hierarchy reading order integrity by using visual element coordinate tracking, treating every resume as a graphical canvas rather than an unformatted text document.
* **Establish High-Dimensional Semantic Evaluation:** To move completely away from token-frequency metrics and implement contextual semantic analysis. The platform must map resumes and job requirements into a unified vector space, ensuring the system intelligently connects identical abstract concepts (such as correlating "distributed infrastructure" with "cloud microservices") regardless of specific variations in phrasing.
* **Enforce Blind Screening Protocols:** To programmatically guarantee data neutrality by embedding an automated, deterministic sanitization pipeline at the edge of the system. This gatekeeper mechanism must completely scrub all identifying data fields before the text hits the inference engine, ensuring that candidate evaluations are strictly based on skills and experience.
* **Deliver a High-Performance HR Operations Dashboard:** To construct an interactive, low-latency web portal engineered specifically for recruitment professionals. The frontend must abstract the underlying complex NLP models into a clean layout, providing non-technical personnel with granular slider controls to adjust match thresholds and view color-coded applicant leaderboards in real time.

---

## 💡 3. Deep Dive Technical Solution & Pipeline Architecture

The platform architecture is built as a highly decoupled, three-tier NLP engineering pipeline. It transforms raw, unvetted PDF files into clean mathematical structures for precise comparison:

[ RAW CANDIDATE RESUMES ]
                               │
                               ▼
      ┌──────────────────────────────────────────────────┐
      │     1. INGESTION & LAYOUT ISOLATION LAYER       │
      │  - PDF Bounding-box analysis via pdfplumber      │
      │  - Multi-column vertical sorting heuristics      │
      └──────────────────────────────────────────────────┘
                               │
                               ▼
      ┌──────────────────────────────────────────────────┐
      │       2. DETERMINISTIC PII SCRUBBING LAYER       │
      │  - Regex processing removes emails & phones      │
      │  - Anonymization & cleaning of personal data     │
      └──────────────────────────────────────────────────┘
                               │
                               ▼
[ JOB DESCRIPTION ] ────► [ SENTENCE-TRANSFORMERS ]
                               │
                               ▼
      ┌──────────────────────────────────────────────────┐
      │      3. SEMANTIC VECTOR MAPPING LAYER            │
      │  - Text converted to dense vector embeddings     │
      │  - Mathematical Cosine Similarity comparison     │
      └──────────────────────────────────────────────────┘
                               │
                               ▼
                 [ STREAMLIT RANKING DASHBOARD ]

### 1. Ingestion & Layout Isolation Layer
Instead of treating incoming PDFs as lines of text, this layer uses structural coordinate evaluation via `pdfplumber`. It isolates the layout boundaries of the document by mapping words to exact pixel coordinates on the page grid. The system runs grouping heuristics to detect distinct columns and sidebars. It processes each text column vertically and isolated before stitching the pieces together, ensuring multi-column layouts are parsed exactly how a human reads them.

### 2. Deterministic PII Scrubbing Layer
Before the parsed resume is sent to the machine learning model, it passes through an absolute data sanitization gateway. The application uses optimized Regular Expressions (Regex) to target and erase explicit biographical markers, including standard and non-standard email structures, international or multi-spaced telephone formats, and portfolio hyperlinks. This text erasure step creates a sanitized profile where the model only evaluates professional skill vectors, completely removing demographic variables from the equation.

### 3. Semantic Vector Mapping Layer
The sanitized text and the baseline job description are passed to the `all-MiniLM-L6-v2` transformer network. This network maps the textual sentences into a dense 384-dimensional vector space where words with similar contextual meanings sit close together. 

The system computes the exact directional angle between the Candidate Resume Vector ($\vec{A}$) and the Job Description Vector ($\vec{B}$) using Cosine Similarity. This outputs an exact match percentage from 0% to 100% that measures contextual alignment rather than basic word counts, rendering keyword stuffing completely useless.

---

## 🛠️ 4. Detailed Technology Stack & Justification

### 🐍 Core Language
* **Technology:** `Python 3.10+`
* **Justification:** Serves as the structural runtime backbone for the entire pipeline. Python is the industry standard for production-grade AI and data science, offering robust native libraries for advanced text processing.

### 🖨️ Document Parser
* **Technology:** `pdfplumber`
* **Justification:** Operates at the ingestion layer to extract raw text elements from PDF files. Unlike primitive parsers like `PyPDF2` that only extract characters sequentially, `pdfplumber` exposes coordinate geometry, allowing the application to accurately isolate complex multi-column layouts.

### 🧠 Linguistic Preprocessing
* **Technology:** `SpaCy (en_core_web_sm)`
* **Justification:** Provides structural language tokenization and token parsing for text data normalization. It allows the system to clean and prepare text fields with minimal latency compared to older libraries.

### 🤖 Deep Learning Architecture
* **Technology:** `Sentence-Transformers`
* **Justification:** Powers the core AI engine by generating high-dimensional semantic vector embeddings. Implements the pretrained `all-MiniLM-L6-v2` model, delivering top-tier semantic accuracy while maintaining a fast execution speed on standard CPUs.

### 📐 Mathematical Engine
* **Technology:** `Scikit-Learn`
* **Justification:** Executes the vector calculations used to determine final candidate match rankings. Supplies optimized, pre-compiled functions for fast vector distance and cosine similarity processing.

### 📊 Data Orchestration
* **Technology:** `Pandas`
* **Justification:** Manages backend tabular data structuring, filtering, sorting, and grading criteria, handling dataframes that feed directly into the presentation UI.

### 🖥️ UI Application Layer
* **Technology:** `Streamlit`
* **Justification:** Renders the web frontend, client-side data uploading inputs, and visualization graphs. It spins up an interactive frontend dashboard directly from Python, completely bypassing the need for a complex JavaScript or React setup.

---

## 📊 5. Comprehensive Key Metrics & Project Deliverables

* **100% Zero-Bias Compliance:** Complete deletion of personal data fields (emails, phone numbers, portfolio links) across all incoming files before vector encoding, guaranteeing a blind screening process.
* **True Semantic Correlation:** Successful text matching based on conceptual meaning rather than exact words. The model automatically correlates related terms like "AWS / Cloud Infrastructure" or "Deep Learning / Neural Networks" without requiring recruiters to manually build keyword lists.
* **Real-Time Leaderboard Rendering:** A dynamic frontend interface that displays applicant profiles in a color-coded, descending rank sequence based on semantic fit, processing uploaded batches within seconds.

---

## 🚀 Installation & Local Execution

### Step 1: Clone the Repository
```bash
git clone [https://github.com/your_github_username/ai-resume-screener.git](https://github.com/your_github_username/ai-resume-screener.git)
cd ai-resume-screener
```
###Step 2: Install Dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

###Step 3: Run the Application
streamlit run app.py
