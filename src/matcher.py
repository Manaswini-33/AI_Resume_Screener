import spacy
import os

class ResumeMatcher:
    def __init__(self):
        # Dynamically download en_core_web_sm if it's not present in the environment
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("SpaCy model not found. Downloading dynamically...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
