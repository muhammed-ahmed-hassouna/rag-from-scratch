# RAG from Scratch - My Learning Journey

This is my personal workspace for learning, building, and experimenting with Retrieval-Augmented Generation (RAG) systems from the ground up. I am using this repository to log my progress, store practical exercises, and document technical insights.

## Phase 1: LLM Basics (Completed)

During the first phase of my learning, I focused on the core concepts of interacting with Large Language Models (LLMs). Rather than relying on high-level frameworks, I worked directly with client SDKs to understand how these systems operate under the hood.

I learned how to establish connections to models like Gemini and Llama via API calls and inspected raw response metadata to monitor token usage. I experimented with system instructions to guide model behavior, define roles, and enforce constraints. Because API endpoints are stateless, I implemented conversation memory by manually preserving state and passing the history of user and assistant responses back to the model with each new request. To bridge the gap between unstructured text and application logic, I structured outputs by prompting models to return JSON, which I parsed into native dictionaries. Finally, I investigated token sampling by comparing different temperature settings and implementing a manual calculator to see how logits are converted to token probabilities using the softmax function.

## Phase 2: FastAPI and Modular RAG Architecture

For the second phase, I am transitioning from isolated scripts to a modular web application. The goal is to build a structured RAG API using FastAPI. To support this, the repository is being reorganized to separate concerns across routers, services, database clients, and utility functions.

The directory structure for this phase is organized as follows:

rag-from-scratch/
├── main.py
├── routers/
│   ├── ask.py
│   └── upload.py
├── services/
│   ├── document_service.py
│   ├── embedding_service.py
│   └── retrieval_service.py
├── db/
│   └── chroma_client.py
├── utils/
│   ├── pdf_loader.py
│   └── text_chunker.py
├── .env
└── requirements.txt

In this architecture, main.py serves as the web server entry point. The routers folder contains ask.py to handle user questions and upload.py to process document uploads. The services folder contains document_service.py for document ingestion workflows, embedding_service.py for generating vector representations, and retrieval_service.py for searching database context. The database configurations are housed in db/chroma_client.py to manage vector storage connections. The utils folder holds helper modules pdf_loader.py and text_chunker.py to extract and split document content.
