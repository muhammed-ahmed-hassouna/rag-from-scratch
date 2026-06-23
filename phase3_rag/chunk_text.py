from pypdf import PdfReader

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Split text into chunks of ~chunk_size characters.
    overlap means consecutive chunks share some text — this prevents
    cutting a sentence in half and losing context at the boundaries.
    """
    chunks = []
    start = 0
    while start < len(text):
        # start = 0 and chunk size = 500 mean's take characters from index 0 to index 500
        end = start + chunk_size
        """
        Slice The String Python slicing syntax: text[start:end] mean's
        Example: text = "Hello World" ===> text[0:5] returns ==> Hello
        """
        chunk = text[start:end]
        chunks.append(chunk)
        #  Move Forward First Itirate start = 0 + 500 - 50 ====>  450
        start += chunk_size - overlap
    return chunks

# Test it
text = extract_text_from_pdf("../pdfs/MohammedHassouneh_BackEnd_cv.pdf")
chunks = chunk_text(text)

print(f"Total characters: {len(text)}")
print(f"Total chunks: {len(chunks)}")
print(f"\nFirst chunk:\n{chunks[0]}")
print(f"\nSecond chunk (notice overlap with first):\n{chunks[1]}")