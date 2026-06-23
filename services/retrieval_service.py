import os
from openai import OpenAI
import json
from pydantic import BaseModel, Field
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

class AIAnswer(BaseModel):
    answer: str = Field(description="The final answer to the user's question. If the answer is not in the context, say 'I don't have enough information to answer that.'")
    confidence_score: int = Field(description="Your confidence score from 1 to 100 on whether the answer is fully supported by the provided context.")

def generate_answer_with_llm(question: str, context: str, mode: str = "strict") -> dict:
    """
    Queries the available LLM (Groq Llama) using the retrieved context to answer the question,
    and returns a structured dictionary parsed via Pydantic.
    """
    if mode == "hybrid":
        instruction = "Use the provided context to help answer the user's question if it is relevant. If the context doesn't contain the answer, you can use your general knowledge to answer."
    else:
        instruction = "Answer the question using ONLY the context provided below. If the answer is not in the context, say 'I don't have enough information to answer that.'"

    prompt = f"""You are a helpful assistant. {instruction}

You MUST return your response as a valid JSON object with EXACTLY these two keys:
1. "answer": The final answer to the user's question.
2. "confidence_score": An integer from 1 to 100 representing your confidence that the answer is accurate.

Context:
{context}

Question: {question}"""

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
                temperature=0.1,
                max_tokens=1024,
                response_format={"type": "json_object"}
            )
            
            # Pydantic validates the JSON and converts it securely to a dictionary
            raw_json = response.choices[0].message.content
            structured_data = AIAnswer.model_validate_json(raw_json)
            return structured_data.model_dump()
            
        except Exception as e:
            return {"answer": f"Error generating answer: {str(e)}", "confidence_score": 0}

    return {"answer": "Error: No LLM API key configured.", "confidence_score": 0}
