import chromadb
from sentence_transformers import SentenceTransformer
import hashlib

model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device="cpu"
)
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(name="my_docs")


def generate_id(text):
    return hashlib.md5(text.encode()).hexdigest()
def add_documents(docs):
    embeddings = model.encode(
    docs,
    batch_size=64,
    show_progress_bar=True
    ) 

    ids = [generate_id(doc) for doc in docs]
    collection.add(documents=docs, embeddings=embeddings, ids=ids)
    print(f"Added {len(docs)} documents.")

def search(query, top_k=3):
    query_embedding = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results['documents'][0]

# First run: add documents
add_documents([
    "FastAPI is a Python framework for building REST APIs.",
    "Docker containers package your app with all its dependencies.",
    "RAG stands for Retrieval Augmented Generation.",
])

# Any run: search
results = search("What is used for building APIs?")
for r in results:
    print(r)