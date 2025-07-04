from loader import load_documents
from rag_pipeline import create_vectorstore

if __name__ == "__main__":
    docs = load_documents("data/docs/")
    print(f"Liczba załadowanych dokumentów: {len(docs)}")
    if docs:
        create_vectorstore(docs)
        print("Wektorowa baza została utworzona i zapisana w folderze 'vectorstore'.")
    else:
        print("Brak dokumentów do przetworzenia w folderze data/docs/")