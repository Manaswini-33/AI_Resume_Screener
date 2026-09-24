import re

def clean_text(text: str) -> str:
    """
    Cleans raw text by normalizing whitespace, removing non-printable characters,
    and standardizing newlines.
    """
    if not text:
        return ""
    # Remove non-printable chars
    text = "".join(char for char in text if char.isprintable() or char in ['\n', '\t', '\r'])
    # Replace multiple horizontal spaces with single space
    text = re.sub(r'[ \t]+', ' ', text)
    # Replace 3+ consecutive newlines with 2 newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def normalize_whitespace(text: str) -> str:
    """Normalizes all whitespace characters into single spaces."""
    return re.sub(r'\s+', ' ', text).strip()
