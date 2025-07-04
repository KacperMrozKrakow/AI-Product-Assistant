import streamlit as st
from dotenv import load_dotenv
import os
from pathlib import Path
from rag_pipeline import load_vectorstore, build_qa_chain
from loader import load_documents
from difflib import SequenceMatcher

load_dotenv()
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

st.set_page_config(page_title="AI Product Assistant", layout="wide", initial_sidebar_state="collapsed")

dark_mode_css = """
<style>
    .main {
        background-color: #121212;
        color: #E0E0E0;
    }
    #MainMenu, footer, header {
        visibility: hidden;
    }
    .user-msg, .bot-msg {
        padding: 12px 18px;
        border-radius: 18px;
        margin: 10px 0;
        max-width: 75%;
        font-size: 15px;
        clear: both;
    }
    .user-msg {
        background-color: #2a2a2a;
        color: #eee;
        margin-left: auto;
        text-align: right;
    }
    .bot-msg {
        background-color: #333333;
        color: #ddd;
        margin-right: auto;
        text-align: left;
    }
    .stTextInput>div>div>input {
        background-color: #222 !important;
        color: #eee !important;
        border-radius: 8px;
        border: none;
        padding: 10px;
        font-size: 16px;
    }
    .source-box {
        background-color: #1c1c1c;
        border-radius: 10px;
        padding: 10px;
        margin-top: 10px;
        font-size: 13px;
    }
</style>
"""
st.markdown(dark_mode_css, unsafe_allow_html=True)

st.title("AI Product Assistant")
st.markdown("""
Ten inteligentny asystent odpowiada na pytania w oparciu o dokumenty (np. PDF-y z ofertami, instrukcjami, katalogami).  
Możesz go wykorzystać np. jako wsparcie klienta — wystarczy załadować dokumenty, a użytkownik może zadawać pytania w języku naturalnym.  
Przykład: *"Czy produkt X obsługuje integrację z systemem Y?"*

⏳ **Poczekaj kilka sekund, aż aplikacja się załaduje...**
""")

# Inicjalizacja historii i inputu
if "history" not in st.session_state:
    st.session_state.history = []

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# Jeśli brak bazy — buduj
if not Path("vectorstore/index.faiss").exists():
    with st.spinner("Tworzę bazę wiedzy..."):
        docs = load_documents("data/docs/")
        from rag_pipeline import create_vectorstore
        create_vectorstore(docs)

db = load_vectorstore()
qa_chain = build_qa_chain(db)

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def ask_question(query):
    st.session_state.history.append({"role": "user", "content": query})
    result = qa_chain(query)
    st.session_state.history.append({
        "role": "bot",
        "content": result["result"],
        "sources": result.get("source_documents", [])
    })

# Wyświetlanie historii
for msg in st.session_state.history:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg">{msg["content"]}</div>', unsafe_allow_html=True)

        sources = msg.get("sources", [])
        if sources:
            with st.expander("📄 Pokaż źródła użyte do odpowiedzi"):
                sorted_docs = sorted(
                    sources,
                    key=lambda d: similarity(d.page_content, msg["content"]),
                    reverse=True
                )
                for i, doc in enumerate(sorted_docs):
                    filename = doc.metadata.get("filename", "Nieznany plik")
                    page = doc.metadata.get("page", None)
                    source_info = f"{filename}"
                    if page is not None:
                        source_info += f", strona {page + 1}"
                    snippet = doc.page_content[:300].strip().replace("\n", " ")
                    st.markdown(f'<div class="source-box">{i+1}. `{source_info}` - {snippet}...</div>', unsafe_allow_html=True)

# Input na samym dole
query = st.text_input("Zadaj pytanie:", value=st.session_state.input_text, key="input_text")

if query:
    ask_question(query)
    st.session_state.input_text = ""
    st.experimental_rerun()