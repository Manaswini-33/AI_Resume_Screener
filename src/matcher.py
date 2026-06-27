import os
import sys
import spacy

class ResumeMatcher:
    def __init__(self):
        # Define a local path inside your project directory that you have permissions to write to
        local_target_dir = os.path.join(os.getcwd(), "local_models")
        
        # Add the local directory to the Python system path so it can see installed modules
        if local_target_dir not in sys.path:
            sys.path.append(local_target_dir)

        try:
            # Try loading the model from the system or local path
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Model 'en_core_web_sm' not found. Installing into local directory...")
            
            # Programmatically trigger pip to install the model inside your local writable directory
            import subprocess
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.0/en_core_web_sm-3.7.0.tar.gz",
                "--target", local_target_dir
            ])
            
            # Load the model directly from the specific local directory package
            self.nlp = spacy.load("en_core_web_sm")
