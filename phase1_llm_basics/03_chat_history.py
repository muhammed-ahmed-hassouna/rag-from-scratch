from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key= os.getenv('GROQ_API_KEY')
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."}
]

while True:
    user_input = input("You: ")
    if user_input == "quit":
        break

    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )

    assistant_reply = response.choices[0].message.content
    messages.append({"role": "assistant", "content": assistant_reply})

    print(f"AI: {assistant_reply}\n")
    
"""
How the Memory Works here
The Setup: You start with just the system instruction in the list.

Turn 1 (User): You type something. Your message is appended. The API sees: [System, User 1].

Turn 1 (AI): The model replies. That reply is appended. The list becomes: [System, User 1, Assistant 1].

Turn 2 (User): You ask a follow-up. Your new message is appended. The API now receives: [System, User 1, Assistant 1, User 2].

Because the LLM receives the entire history every time you make a new request, it can remember things you said five turns ago.
"""
