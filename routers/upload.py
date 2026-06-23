import os
import shutil
from fastapi import APIRouter, File, UploadFile, HTTPException
from services.document_service import process_pdf

router = APIRouter()

TEMP_DIR = os.path.join(os.getcwd(), "temp_uploads")
os.makedirs(TEMP_DIR, exist_ok=True)

@router.post("/upload")
def upload_document(file: UploadFile = File(...)):
    """
    Endpoint to upload a PDF document. The document is chunked, embedded, and stored in ChromaDB.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    temp_file_path = os.path.join(TEMP_DIR, file.filename)

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        chunks_count = process_pdf(temp_file_path)

        return {
            "status": "success",
            "message": f"Processed and indexed {file.filename} successfully.",
            "filename": file.filename,
            "chunks_count": chunks_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
