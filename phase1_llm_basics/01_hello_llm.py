from google import genai as GenAI
from dotenv import load_dotenv
import os
from pprint import pprint

load_dotenv()

# Creates a connection to Gemini
client = GenAI.Client(
    api_key= os.getenv('GOOGLE_API_KEY')
)

question = input("Ask Gemini : ")

# Send's a prompt, every AI has its own style to send prompt 
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=question
)

# Multiple Methods to see what happened actually behind
# print(dir(response))
# pprint(response.model_dump())
# print(response.usage_metadata)
# print(response.candidates)

print(response.text)
