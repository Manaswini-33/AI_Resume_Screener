import spacy
# Import the model directly as a python package
import en_core_web_sm

class ResumeMatcher:
    def __init__(self):
        # Load the directly imported model layout
        self.nlp = en_core_web_sm.load()
