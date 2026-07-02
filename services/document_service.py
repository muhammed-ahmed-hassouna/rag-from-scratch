import os
from utils.pdf_loader import extract_text_from_pdf
from utils.text_chunker import chunk_text
from services.embedding_service import get_embeddings
from db.chroma_client import get_collection

def process_pdf(pdf_path: str, collection_name: str = "knowledge_base") -> int:
    """
    Reads a PDF, extracts text, chunks it, generates embeddings, and stores them in ChromaDB.
    Returns the number of chunks successfully processed and stored.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")

    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        return 0

    chunks = chunk_text(text, chunk_size=500, overlap=50)
    if not chunks:
        return 0

    embeddings = get_embeddings(chunks)
    collection = get_collection(collection_name)

    # Generate unique IDs using filename and index to avoid collisions
    filename = os.path.basename(pdf_path)
    safe_filename = "".join([c if c.isalnum() or c in "._-" else "_" for c in filename])
    
    ids = [f"{safe_filename}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename} for _ in chunks]

    # Delete any existing chunks for this document first to prevent
    # orphaned chunks if the chunking configuration (size, overlap) has changed.
    collection.delete(where={"source": filename})

    collection.upsert(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )

    return len(chunks)
