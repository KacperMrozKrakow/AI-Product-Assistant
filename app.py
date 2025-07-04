import streamlit as st
from dotenv import load_dotenv
import os
from pathlib import Path
from rag_pipeline import load_vectorstore, build_qa_chain
from loader import load_documents
from difflib import SequenceMatcher

# Załaduj zmienne środowiskowe
load_dotenv()
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Konfiguracja strony
st.set_page_config(page_title="📚 AI Documentation Chatbot")

st.title("📚 AI Expert Chatbot")
st.write("Zadaj pytanie na podstawie dokumentów...")

# Jeśli nie ma bazy wektorowej, stwórz ją
if not Path("vectorstore/index.faiss").exists():
    with st.spinner("Tworzę bazę wiedzy..."):
        docs = load_documents("data/docs/")
        from rag_pipeline import create_vectorstore
        create_vectorstore(docs)

# Załaduj bazę
db = load_vectorstore()

# Bezpieczne przypisanie tokena
if not token:
    token = ""

# Inicjalizacja łańcucha QA
qa_chain = build_qa_chain(db)

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

# Interfejs użytkownika
query = st.text_input("Zadaj pytanie:")

if query:
    with st.spinner("Szukam odpowiedzi..."):
        result = qa_chain(query)

        # Wyświetl odpowiedź
        st.markdown("### 💡 Odpowiedź:")
        st.markdown(result["result"])

        source_docs = result.get("source_documents", [])

        if source_docs:
            # Sortuj wg podobieństwa do odpowiedzi
            sorted_docs = sorted(
                source_docs,
                key=lambda d: similarity(d.page_content, result["result"]),
                reverse=True
            )

            # Pokaż najlepsze źródło
            top_doc = sorted_docs[0]
            filename = top_doc.metadata.get("filename", "Nieznany plik")
            page = top_doc.metadata.get("page", None)
            source_info = f"{filename}"
            if page is not None:
                source_info += f", strona {page + 1}"

            snippet = top_doc.page_content[:300].strip().replace("\n", " ")
            st.markdown("### 📄 Najtrafniejsze źródło:")
            st.markdown(f"**Źródło:** `{source_info}`")
            st.markdown(f"> {snippet}...")

            # Checkbox do pokazywania wszystkich źródeł
            show_all = st.checkbox("Pokaż wszystkie cytowane źródła")

            if show_all:
                st.markdown("### 📄 Wszystkie cytowane źródła:")
                for i, doc in enumerate(sorted_docs):
                    filename = doc.metadata.get("filename", "Nieznany plik")
                    page = doc.metadata.get("page", None)
                    source_info = f"{filename}"
                    if page is not None:
                        source_info += f", strona {page + 1}"

                    snippet = doc.page_content[:300].strip().replace("\n", " ")
                    st.markdown(f"**{i + 1}. Źródło:** `{source_info}`")
                    st.markdown(f"> {snippet}...")
