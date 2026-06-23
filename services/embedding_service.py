from sentence_transformers import SentenceTransformer

# Load embedding model once globally
model = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generates a list of vector representations (embeddings) for a list of texts.
    """
    if not texts:
        return []
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()

def get_embedding(text: str) -> list[float]:
    """
    Generates a single vector representation (embedding) for a text.
    """
    return get_embeddings([text])[0]

