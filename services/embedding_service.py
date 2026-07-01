import os
from dotenv import load_dotenv

load_dotenv()

# Lazy-load embedding model to avoid loading PyTorch in uvicorn's reloader process
_model = None

def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        from huggingface_hub import login

        # Authenticate with HuggingFace for higher rate limits and faster downloads
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            login(token=hf_token, add_to_git_credential=False)

        _model = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")
    return _model

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generates a list of vector representations (embeddings) for a list of texts.
    """
    if not texts:
        return []
    embeddings = _get_model().encode(texts, show_progress_bar=False)
    return embeddings.tolist()

def get_embedding(text: str) -> list[float]:
    """
    Generates a single vector representation (embedding) for a text.
    """
    return get_embeddings([text])[0]

