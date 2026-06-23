from fastapi import FastAPI
from routers import upload, ask

app = FastAPI(
    title="RAG from Scratch API",
    description="A modular FastAPI service for Retrieval-Augmented Generation.",
    version="2.0.0"
)

# Include API routers
app.include_router(upload.router, tags=["Ingestion"])
app.include_router(ask.router, tags=["Retrieval"])

@app.get("/")
def read_root():
    """
    Health-check and metadata root endpoint.
    """
    return {
        "status": "healthy",
        "project": "RAG from Scratch",
        "framework": "FastAPI",
        "phase": 2
    }
