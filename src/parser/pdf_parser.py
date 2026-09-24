import io
import logging

logger = logging.getLogger(__name__)

def extract_text_with_fallbacks(pdf_source) -> tuple[str, str]:
    """
    Extracts text from PDF source (file path or bytes).
    Pipeline:
      1. PyMuPDF (fitz)
      2. pdfplumber fallback (if PyMuPDF produces < 50 chars)
      3. OCR fallback (pytesseract / PIL image processing if text remains < 50 chars)

    Returns:
      (extracted_text, method_used)
    """
    text = ""
    method = "PyMuPDF"

    # Step 1: PyMuPDF (fitz)
    try:
        import fitz
        if isinstance(pdf_source, bytes):
            doc = fitz.open(stream=pdf_source, filetype="pdf")
        else:
            doc = fitz.open(pdf_source)

        pages_text = []
        for page in doc:
            pages_text.append(page.get_text())
        text = "\n".join(pages_text).strip()
    except Exception as e:
        logger.warning(f"PyMuPDF parsing failed or unavailable: {e}")
        text = ""

    # Step 2: pdfplumber fallback
    if len(text.strip()) < 50:
        method = "pdfplumber"
        try:
            import pdfplumber
            if isinstance(pdf_source, bytes):
                with pdfplumber.open(io.BytesIO(pdf_source)) as pdf:
                    pages_text = [p.extract_text() or "" for p in pdf.pages]
            else:
                with pdfplumber.open(pdf_source) as pdf:
                    pages_text = [p.extract_text() or "" for p in pdf.pages]
            text = "\n".join(pages_text).strip()
        except Exception as e:
            logger.warning(f"pdfplumber parsing failed: {e}")

    # Step 3: OCR fallback
    if len(text.strip()) < 50:
        method = "OCR (Tesseract)"
        try:
            import pytesseract
            from PIL import Image
            import fitz

            if isinstance(pdf_source, bytes):
                doc = fitz.open(stream=pdf_source, filetype="pdf")
            else:
                doc = fitz.open(pdf_source)

            ocr_pages = []
            for page in doc:
                pix = page.get_pixmap()
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                ocr_text = pytesseract.image_to_string(img)
                ocr_pages.append(ocr_text)
            text = "\n".join(ocr_pages).strip()
        except Exception as e:
            logger.warning(f"OCR fallback failed: {e}")
            if len(text.strip()) < 10:
                method = "Failed / Insufficient Text"

    return text.strip(), method
