from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
import os

load_dotenv()

llm_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("knowledge_base")

def answer_question(question):
    # Step 1: Embed the question
    question_embedding = embedding_model.encode(question).tolist()

    # Step 2: Find relevant chunks
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3
    )
    context_chunks = results['documents'][0]
    context = "\n\n".join(context_chunks)

    # Step 3: Build the prompt
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""

    # Step 4: Send to LLM
    response = llm_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    return response.choices[0].message.content

# Run it
print("RAG System Ready. Type 'quit' to exit.\n")
while True:
    question = input("Your question: ")
    if question.lower() == "quit":
        break
    answer = answer_question(question)
    print(f"\nAnswer: {answer}\n")
    print("-" * 50 + "\n")