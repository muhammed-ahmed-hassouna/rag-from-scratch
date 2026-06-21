from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

documents = [
    "FastAPI is a Python framework for building REST APIs.",
    "Docker containers package your app with all its dependencies.",
    "RAG stands for Retrieval Augmented Generation.",
    "ChromaDB is a vector database for storing embeddings.",
    "Python lists are ordered, mutable collections.",
    "LangChain is a framework for building LLM-powered apps.",
    "Git is a version control system used by developers.",
    "PostgreSQL is a relational database management system.",
]
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

doc_embeddings = model.encode(documents)

def search(query, top_k=3):
    query_embed = model.encode(query)

    scores = []
    for i, doc_embed in enumerate(doc_embeddings):
        score = cosine_similarity(query_embed, doc_embed)
        scores.append((score, documents[i]))
        
    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[:top_k]

results = search("How do I build a web API in Python?")

for score, doc in results:
    print(f"{score:.4f} | {doc}")