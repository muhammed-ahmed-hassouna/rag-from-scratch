from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from services.agent_service import run_agent_loop

router = APIRouter()

class AgentQuestionRequest(BaseModel):
    question: str = Field(..., description="The question to ask the agentic RAG system.")

class AgentToolStep(BaseModel):
    tool: str = Field(..., description="The name of the tool called by the agent.")
    arguments: dict = Field(..., description="The arguments passed to the tool.")
    result: str = Field(..., description="The result/output returned by the tool.")

class AgentQuestionResponse(BaseModel):
    answer: str = Field(..., description="The final answered response synthesized by the agent.")
    steps: list[AgentToolStep] = Field(default=[], description="The list of tool-calling steps executed during the agent loop.")

@router.post("/agent/ask", response_model=AgentQuestionResponse)
def ask_agent(request: AgentQuestionRequest):
    """
    Endpoint to ask a question via the Agentic RAG flow.
    The agent dynamically decides which tools to call (e.g. ChromaDB document retrieval, current time, or web search).
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question text cannot be empty.")

    try:
        result = run_agent_loop(request.question)
        return AgentQuestionResponse(
            answer=result["answer"],
            steps=result["steps"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
