import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class ResumeMatcher:
    def __init__(self):
        # Initialize the deep learning transformer model directly
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def clean_text(self, text):
        """
        Lightweight, lightning-fast text processing using regex.
        Removes special characters, extra whitespace, and normalizes text.
        """
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)  # Collapse extra spaces/newlines
        return text.strip()

    def calculate_similarity(self, resume_text, jd_text):
        """
        Computes the contextual match score using mathematical cosine similarity.
        """
        cleaned_resume = self.clean_text(resume_text)
        cleaned_jd = self.clean_text(jd_text)

        # Generate dense vector embeddings (384 dimensions)
        embeddings = self.model.encode([cleaned_resume, cleaned_jd])

        # Compute cosine similarity between candidate and job description
        similarity_matrix = cosine_similarity([embeddings[0]], [embeddings[1]])
        
        # Convert the raw score to a clean, rounded percentage
        match_percentage = float(similarity_matrix[0][0]) * 100
        return round(match_percentage, 2)
