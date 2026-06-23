import os
from openai import OpenAI
from google import genai as GenAI
from dotenv import load_dotenv
from services.embedding_service import get_embedding
from db.chroma_client import get_collection

load_dotenv()

def retrieve_context(query: str, collection_name: str = "knowledge_base", n_results: int = 3) -> tuple[str, list[str]]:
    """
    Finds the most relevant chunks in ChromaDB for a given query.
    Returns a tuple of (joined_context_string, list_of_raw_chunks).
    """
    collection = get_collection(collection_name)
    query_vector = get_embedding(query)
    
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results
    )
    
    if not results or not results.get('documents') or not results['documents'][0]:
        return "", []
        
    chunks = results['documents'][0]
    context = "\n\n".join(chunks)
    return context, chunks

def generate_answer_with_llm(question: str, context: str) -> str:
    """
    Queries the available LLM (Groq Llama) using the retrieved context to answer the question.
    """
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""

    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=groq_key
            )
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating answer with Groq Llama: {str(e)}"

    return "Error: No LLM API key configured. Please set GOOGLE_API_KEY or GROQ_API_KEY in your .env file."
