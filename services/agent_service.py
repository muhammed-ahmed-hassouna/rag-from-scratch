import os
import json
import datetime
from openai import OpenAI
from dotenv import load_dotenv
from services.retrieval_service import retrieve_context

load_dotenv()


# Cnstant Variables 
DEFAULT_MAX_ITERATIONS = 5


# --- Map of Tool Names to Python Functions ---
AVAILABLE_TOOLS = {
    "retrieve_documents": retrieve_documents,
    "get_current_time": get_current_time,
    "search_internet_mock": search_internet_mock
}

# --- OpenAI/Groq Tool Definitions (JSON Schemas) ---
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_documents",
            "description": "Use this tool to search the local document database for context. Call this when the user asks questions about uploaded documents, PDFs, or specific knowledge that has been ingested.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A specific search query containing key terms to query the document index."
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of relevant text chunks to retrieve.",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Use this tool to get the current date and time. Call this when the user asks 'what time is it?', 'what is today's date?', or asks for current real-time system clock info.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_internet_mock",
            "description": "Use this tool to perform a mock internet search. Call this when the user asks about real-time, external, or general knowledge questions (like weather, capitals, news) that are NOT related to the uploaded document PDFs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up on the internet."
                    }
                },
                "required": ["query"]
            }
        }
    }
]

SYSTEM_PROMPT = """
You are a helpful assistant.

Use tools whenever necessary.

Never invent tool results.
"""

MODEL_NAME = "llama-3.3-70b-versatile"

# --- Tool Implementations ---
def retrieve_documents(query: str, n_results: int = 3) -> str:
    """
    Search the ChromaDB knowledge base for relevant context matching the query.
    """
    print(f"[Agent Tool] retrieve_documents called with query='{query}', n_results={n_results}")
    try:
        context_str, _, confidence = retrieve_context(query, n_results=n_results)
        if not context_str.strip():
            return "No matching documents found in the database."
        return f"Retrieved Context (Confidence: {confidence}%):\n{context_str}"
    except Exception as e:
        return f"Error retrieving documents: {str(e)}"

def get_current_time() -> str:
    """
    Get the current system date and time.
    """
    print("[Agent Tool] get_current_time called")
    now = datetime.datetime.now()
    return f"The current system date and time is: {now.strftime('%Y-%m-%d %H:%M:%S')}"

def search_internet_mock(query: str) -> str:
    """
    Simulates searching the internet for public information or weather.
    """
    print(f"[Agent Tool] search_internet_mock called with query='{query}'")
    query_lower = query.lower()
    
    # Simple hardcoded mock answers for demonstrations
    if "weather" in query_lower:
        if "tokyo" in query_lower:
            return "Mock Web Search Result: The weather in Tokyo is currently 22°C and partly cloudy."
        elif "london" in query_lower:
            return "Mock Web Search Result: The weather in London is 15°C with light rain."
        else:
            return f"Mock Web Search Result: The weather in the requested location is 20°C and sunny. (Query: {query})"
    elif "capital" in query_lower:
        if "france" in query_lower:
            return "Mock Web Search Result: The capital of France is Paris."
        elif "japan" in query_lower:
            return "Mock Web Search Result: The capital of Japan is Tokyo."
            
    return f"Mock Web Search Result: No specific internet results found for '{query}', but search completed successfully."

# Agent Run Loop
"""
START
   │
   ▼
call_llm()
   │
   ▼
Has tool?
 ┌───────┴────────┐
 │                │
No               Yes
 │                │
 ▼                ▼
 END        execute_tools()
                 │
                 ▼
           append_results()
                 │
                 ▼
             call_llm()
"""
def create_client() -> OpenAI:
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing.")

    return OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

def initialize_messages(question: str):

    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": question
        }
    ]

def call_llm(client, messages):

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=TOOL_SCHEMAS,
        tool_choice="auto",
        temperature=0.2,
    )

    return response.choices[0].message

    
def execute_tool_call(tool_call) -> dict:

    function_name = tool_call.function.name

    try:
        arguments = json.loads(
            tool_call.function.arguments or "{}"
        )

    except json.JSONDecodeError:
        arguments = {}

    tool = AVAILABLE_TOOLS.get(function_name)

    if tool is None:

        result = f"Tool '{function_name}' not found."

    else:

        try:
            result = tool(**arguments)

        except Exception as exc:
            result = f"Tool execution failed: {exc}"

    return {
        "tool": function_name,
        "arguments": arguments,
        "result": result
    }

def append_tool_result(messages, tool_call, result):

    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_call.function.name,
            "content": str(result)
        }
    )

def run_agent_loop(user_question: str) -> dict:
    """
    Executes an agent loop until the model returns a final answer
    or the maximum number of iterations is reached.
    """

    client = create_client()
    messages = initialize_messages(user_question)

    steps = []

    for iteration in range(DEFAULT_MAX_ITERATIONS):

        try:
            assistant_message = call_llm(client, messages)

        except Exception as e:
            return {
                "answer": f"LLM Error: {e}",
                "steps": steps
            }

            if assistant_message is None:
                return {
                    "answer": f"LLM Error: {exc}",
                    "steps": steps
                }

        # Save assistant response
        messages.append(assistant_message)

        # Final Answer
        if not assistant_message.tool_calls:

            return {
                "answer": assistant_message.content,
                "steps": steps
            }

        # Execute Requested Tools
        for tool_call in assistant_message.tool_calls:

            step = execute_tool_call(tool_call)

            steps.append(step)

            append_tool_result(
                messages=messages,
                tool_call=tool_call,
                result=step["result"]
            )

    return {
        "answer": "Maximum iterations reached.",
        "steps": steps
    }