#Enterprise AI Resume Screener & Ranking System
📌 1. Problem Statements (PS)
Traditional corporate recruitment workflows rely heavily on legacy Applicant Tracking Systems (ATS) that use keyword matching. This approach introduces three critical systemic flaws:

a.) Layout and Formatting Fragility: Standard text extraction engines process layouts strictly left-to-right. When analyzing modern multi-column resumes, tables, or complex sidebars, unrelated sections are merged into an incoherent text block, corrupting the data hierarchy.

b.) The "Keyword Stuffing" Exploit: Because legacy scanners search for exact character strings, candidates can easily manipulate the system. Artificially repeating a key tool or phrase multiple times (even in hidden white fonts) forces the profile to rank higher than superior candidates with diverse vocabulary.

c.) Algorithmic Bias Loop: Machine learning systems trained on historic hiring data naturally inherit and scale human biases regarding gender, ethnicity, geographic regions, and institutional names. This systematically filters out high-potential, non-traditional candidates.

🎯 2. Project Objectives
Architect Structural Parsing: Implement a structural, layout-aware PDF reading layer capable of isolating and extracting multi-column resume formatting with 100% reading order integrity.

Establish Semantic Evaluation: Replace token frequency counting with dense vector semantic analysis to understand the contextual meaning of an applicant's experience, matching synonyms and conceptually identical phrases.

Enforce Blind Screening Protocols: Build an automated data scrubbing pipeline that completely sanitizes sensitive personal data prior to model inference, ensuring completely unbiased scoring.

Deliver an Interactive HR Interface: Design a lightweight, high-performance web dashboard allowing non-technical recruitment managers to dynamically upload resumes and adjust matching thresholds on demand.

💡 3. The Technical Solution
The system is engineered as a modular, three-tier NLP pipeline that processes unstructured data into actionable insights:

Ingestion & Layout Isolation Layer: Utilizes visual bounding-box analysis to segment PDF zones vertically and horizontally. This ensures text blocks are parsed in the exact sequence humans read them.

Deterministic PII Scrubbing: Runs specialized regular expression (Regex) passes over the raw extracted text. This instantly strips out email strings, complex global telephone digits, and hyperlinks, creating an anonymous, compliant text profile.

Semantic Vector Mapping Layer: Feeds the cleaned text and target job description into a deep learning Transformer network. The network maps both texts into a shared high-dimensional vector space. The final match score is computed using the mathematical cosine distance between the two vectors:
​
 
🛠️ 4. Technology Stack
🐍 Core Language
Technology: Python 3.10+
Justification: The industry standard for scalable AI/ML pipelines, offering robust native libraries for advanced text processing.

🖨️ Document Parser
Technology: pdfplumber
Justification: Outperforms standard libraries like PyPDF2 by providing precise visual element inspection and coordinate-based tracking to handle complex document layouts.

🧠 Linguistic Preprocessing
Technology: SpaCy (en_core_web_sm)
Justification: An enterprise-grade NLP library used to quickly parse, tokenize, and break down complex sentence syntax.

🤖 Deep Learning Models
Technology: Sentence-Transformers
Justification: Implements the pretrained all-MiniLM-L6-v2 model, allowing for high-speed, highly accurate semantic encoding of text blocks.

📐 Mathematical Engine
Technology: Scikit-Learn
Justification: Supplies optimized, pre-compiled functions for fast vector distance calculations and cosine similarity processing.

📊 Data Orchestration
Technology: Pandas
Justification: Efficiently handles backend data framing, structured filtering, sorting, and structural grading criteria.

🖥️ UI Application Layer
Technology: Streamlit
Justification: Spins up an interactive, responsive frontend dashboard directly from Python, completely bypassing the need for a complex JavaScript or React setup.

📊 5. Key Metrics & Deliverables
Zero-Bias Compliance: 100% of PII data (emails, links, phone numbers) is successfully scrubbed prior to semantic matching.
Contextual Matching: Accurately correlates synonyms (e.g., matching "Deep Learning" to "Neural Networks") without requiring manual keyword configuration.
Instant Sorting Dashboard: Renders a visually color-coded, descending candidate leaderboard within seconds of text ingestion.
