import chromadb
from sentence_transformers import SentenceTransformer

def search_knowledge_base(question, collection_name="knowledge_base", top_k=3):
    model = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(collection_name)

    # Embed the question
    question_embedding = model.encode(question).tolist()

    # Find the  top_k most similar chunks
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )

    return results['documents'][0]  # list of chunk strings

# Test it
question = "what is the project the person have ?"
relevant_chunks = search_knowledge_base(question)

for i, chunk in enumerate(relevant_chunks):
    print(f"--- Chunk {i+1} ---")
    print(chunk)
    print()