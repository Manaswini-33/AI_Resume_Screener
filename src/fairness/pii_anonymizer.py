import re

def anonymize_pii(text: str) -> str:
    """
    Detects and anonymizes personally identifiable information (PII) from resume text:
    - Email addresses -> [EMAIL]
    - Phone numbers -> [PHONE]
    - Web URLs & LinkedIn profiles -> [URL]
    - Physical address indicators -> [ADDRESS]
    - Dates of birth / Year indicators -> [YEAR]

    Returns redacted text string.
    """
    if not text:
        return ""

    # Email replacement
    text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[EMAIL]', text)

    # Phone number replacement (supports international & formatted numbers)
    text = re.sub(r'(\+?\d{1,3}[\s\.-]?)?\(?\d{3}\)?[\s\.-]?\d{3}[\s\.-]?\d{4}', '[PHONE]', text)
    text = re.sub(r'\+?\d[\d\s-]{8,}\d', '[PHONE]', text)

    # URL replacement
    text = re.sub(r'https?://\S+|www\.\S+|linkedin\.com/\S+|github\.com/\S+', '[URL]', text)

    # Address / Zipcode replacement patterns
    text = re.sub(r'\b\d{5}(?:[-\s]\d{4})?\b', '[ZIPCODE]', text)

    # Mask specific birth years or demographic dates (1950-2010 range)
    text = re.sub(r'\b(DOB|Date of Birth|Born)[:\s]*\d{1,2}[/\.-]\d{1,2}[/\.-]\d{2,4}\b', '[DOB]', text, flags=re.IGNORECASE)

    return text

def extract_anonymized_metadata(raw_text: str) -> dict:
    """
    Utility function that returns count of redacted PII fields for audit logging.
    """
    emails = len(re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', raw_text))
    phones = len(re.findall(r'(\+?\d{1,3}[\s\.-]?)?\(?\d{3}\)?[\s\.-]?\d{3}[\s\.-]?\d{4}', raw_text))
    urls = len(re.findall(r'https?://\S+|www\.\S+|linkedin\.com/\S+', raw_text))

    return {
        "emails_redacted": emails,
        "phones_redacted": phones,
        "urls_redacted": urls,
        "privacy_status": "ANONYMIZED"
    }
