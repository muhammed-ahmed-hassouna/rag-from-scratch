from pypdf import PdfReader

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts all text content from a PDF file path.
    """
    import re
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
            
    # Clean dirty PDF characters
    text = re.sub(r'\x00', '', text) # Remove null bytes
    # Optional: could normalize whitespace here, but newlines are useful for chunking
    return text
