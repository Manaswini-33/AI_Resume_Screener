import numpy as np
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class EmbeddingEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_transformer: bool = False):
        self.model_name = model_name
        self.st_model = None
        if use_transformer:
            self._init_transformer()

    def _init_transformer(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.st_model = SentenceTransformer(self.model_name)
        except Exception as e:
            logger.warning(f"Could not load SentenceTransformer ({e}). Using TF-IDF fallback.")
            self.st_model = None

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Computes cosine similarity (0.0 to 1.0) between text1 and text2.
        Uses SentenceTransformer if loaded, otherwise fast TF-IDF cosine fallback.
        """
        if not text1 or not text2 or not text1.strip() or not text2.strip():
            return 0.0

        if self.st_model is not None:
            try:
                embeddings = self.st_model.encode([text1, text2], convert_to_numpy=True)
                sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                return float(max(0.0, min(1.0, sim)))
            except Exception:
                pass

        # Fast TF-IDF Cosine Similarity Fallback
        try:
            vec = TfidfVectorizer(stop_words='english')
            tfidf_mat = vec.fit_transform([text1, text2])
            sim = cosine_similarity(tfidf_mat[0:1], tfidf_mat[1:2])[0][0]
            return float(max(0.0, min(1.0, sim)))
        except Exception:
            return 0.0
