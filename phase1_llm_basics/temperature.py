from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key= os.getenv('GROQ_API_KEY')
)

SYSTEM_PROMPT = """
You are a helpfull assistant.
"""

question = "Give me a creative name for an AI startup"

for temp in [0.0, 0.7, 1.5]:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": question}],
        temperature=temp
    )
    print(f"Temperature {temp}: {response.choices[0].message.content}\n")

"""
Run it 3 times. Notice: temperature=0 gives the same answer every time.
temperature=1.5 gets weird. For RAG systems you'll use low temperature (0 to 0.3) because you want factual,
consistent answers.
"""

"""
When an LLM generates text, it doesn't just pick "the best" next word. Instead,
it calculates a list of raw scores (called logits) for thousands of possible next words, 
which are converted into percentages.
"""

