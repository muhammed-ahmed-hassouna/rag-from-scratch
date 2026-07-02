import chromadb
import os

CHROMA_DB_PATH = os.path.join(os.getcwd(), "chroma_db")

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

def get_collection(name: str = "knowledge_base"):
    """
    Get or create a ChromaDB collection by name.
    """
    return client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
