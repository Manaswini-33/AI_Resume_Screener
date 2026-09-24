from src.fairness.pii_anonymizer import anonymize_pii

def test_pii_anonymization():
    raw = "John Doe, email: john@example.com, phone: +1-555-0199, website: https://johndoe.com"
    clean = anonymize_pii(raw)

    assert "[EMAIL]" in clean
    assert "[PHONE]" in clean
    assert "[URL]" in clean
    assert "john@example.com" not in clean
