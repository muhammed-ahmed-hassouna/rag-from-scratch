import json
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key= os.getenv('GROQ_API_KEY')
)

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "system",
            "content": "You are a data extractor. Always respond with valid JSON only. No explanation, no markdown."
        },
        {
            "role": "user",
            "content": "Extract the name, age, and city from: 'My name is Omar, I am 28 years old and I live in Amman.'"
        }
    ],
    temperature=0
)

raw = response.choices[0].message.content
parsed = json.loads(raw)
print(parsed)
print(f"Name: {parsed['name']}")