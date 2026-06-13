from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key= os.getenv('GROQ_API_KEY')
)

SYSTEM_PROMPT = """
You are a senior Python developer. Answer every question with a short code example. Never write more than 5 lines.
"""

question = input("Ask Groq (Llama 3.3): ")

response = client.chat.completions.create(
  model="llama-3.3-70b-versatile",
    temperature=0.2,  # Lower temperature makes the model strictly follow instructions
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},  
        {"role": "user", "content": question}
    ]
    
)

print(response.choices[0].message.content)
