import os
import fitz  # PyMuPDF
import markdown
from pathlib import Path
from langchain.schema import Document

def load_documents(folder_path: str) -> list[Document]:
    documents = []
    for file in Path(folder_path).glob("*"):
        if file.suffix.lower() == ".pdf":
            doc = fitz.open(file)
            for i, page in enumerate(doc):
                text = page.get_text()
                metadata = {
                    "filename": file.name,
                    "page": i
                }
                documents.append(Document(page_content=text, metadata=metadata))
        elif file.suffix.lower() == ".md":
            with open(file, encoding="utf-8") as f:
                md = f.read()
                html = markdown.markdown(md)
                metadata = {
                    "filename": file.name
                }
                documents.append(Document(page_content=html, metadata=metadata))
    return documents
