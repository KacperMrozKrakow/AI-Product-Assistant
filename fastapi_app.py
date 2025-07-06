# fastapi_app.py
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from rag_pipeline import create_vectorstore
from loader import load_documents
from pathlib import Path

app = FastAPI()

# Pozwól na CORS (zwłaszcza do testów lokalnych)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # na produkcję doprecyzuj
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("data/docs/")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".pdf", ".md")):
        raise HTTPException(status_code=400, detail="Tylko pliki PDF i MD są wspierane.")
    save_path = UPLOAD_DIR / file.filename
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)
    # Przebuduj vectorstore po zapisie nowego pliku
    docs = load_documents(str(UPLOAD_DIR))
    create_vectorstore(docs)
    return {"message": f"Plik '{file.filename}' przesłany i vectorstore przebudowany."}