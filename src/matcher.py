import spacy
from spacy.cli import download

class ResumeMatcher:
    def __init__(self):
        try:
            # Try to load the model normally
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # If not found, use SpaCy's official downloader package to grab it safely
            print("Model 'en_core_web_sm' not found. Downloading via SpaCy CLI...")
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
