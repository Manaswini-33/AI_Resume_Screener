import pdfplumber
import re

class ResumeParser:
    def __init__(self):
        pass

    def extract_text_from_pdf(self, pdf_path):
        """
        SOLUTION 1: Fixes Layout & Formatting Fragility
        Uses pdfplumber with structural layout=True to properly parse multi-column text.
        """
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text(layout=True)
                if page_text:
                    text += page_text + "\n"
        return self._clean_text(text)

    def _clean_text(self, text):
        """
        SOLUTION 3: Mitigates Bias Loop by Stripping PII
        Removes emails, phone numbers, website links, and extra structural noise.
        """
        # Remove Emails
        text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '', text)
        
        # Remove Phone Numbers (various formats)
        text = re.sub(r'\+?\d{1,4}?[-.\s]?\(?\d{1,3}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}', '', text)
        
        # Remove URLs/Hyperlinks
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Collapse excessive whitespace layout noise
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
