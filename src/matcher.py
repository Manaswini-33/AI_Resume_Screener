from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

class ResumeMatcher:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # Load a lightweight, powerful transformer model
        self.model = SentenceTransformer(model_name)
        # Load SpaCy linguistic layer
        self.nlp = spacy.load("en_core_web_sm")

    def calculate_similarity(self, resume_text, job_description):
        """
        SOLUTION 2: Fixes Keyword Stuffing
        Computes semantic similarity mapping using Vector Space Cosine Distance.
        """
        embeddings = self.model.encode([resume_text, job_description])
        resume_vector = embeddings[0].reshape(1, -1)
        jd_vector = embeddings[1].reshape(1, -1)
        
        score = cosine_similarity(resume_vector, jd_vector)[0][0]
        return round(float(score) * 100, 2) # Return as percentage
