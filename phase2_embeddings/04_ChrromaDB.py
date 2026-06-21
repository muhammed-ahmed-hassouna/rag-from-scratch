import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device="cpu"
)
# Create an in-memory ChromaDB (resets when script ends)
client = chromadb.Client()

# Create a collection (like a table in SQL)
collection = client.create_collection(name="my_docs")

documents = [
    "FastAPI is a Python framework for building REST APIs.",
    "Docker containers package your app with all its dependencies.",
    "RAG stands for Retrieval Augmented Generation.",
    "ChromaDB is a vector database for storing embeddings.",
    "LangChain is a framework for building LLM-powered apps.",
]

# Embed them
embeddings = model.encode(
    documents,
    batch_size=64,
    show_progress_bar=True
) # ChromaDB needs a list, not numpy array

# Add to collection
collection.add(
    documents=documents,                              # the actual text
    embeddings=embeddings,                            # the vectors
    ids=["doc_0", "doc_1", "doc_2", "doc_3", "doc_4"]  # unique IDs
)

# Query
results = collection.query(
    query_embeddings=[model.encode("How do I create an API?").tolist()],
    n_results=2
)

print("Query: How do I create an API?")
print("\nTop results:")
for doc in results['documents'][0]:
    print(f"  - {doc}")