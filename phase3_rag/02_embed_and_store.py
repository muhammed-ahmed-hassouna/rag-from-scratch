import chromadb
from sentence_transformers import SentenceTransformer
from chunk_text import extract_text_from_pdf, chunk_text  # reuse from file 01

def build_knowledge_base(pdf_path, collection_name="knowledge_base"):
    # Load PDF and chunk it
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)

    # Load embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")

    # Create a PERSISTENT ChromaDB (saves to disk)
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(collection_name)

    # Embed all chunks
    print(f"Embedding {len(chunks)} chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True).tolist()

    # Store in ChromaDB
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )

    print(f"Stored {len(chunks)} chunks in ChromaDB.")
    return collection

build_knowledge_base("../pdfs/MohammedHassouneh_BackEnd_cv.pdf")