import streamlit as st
from dotenv import load_dotenv
import os
from pathlib import Path
from rag_pipeline import load_vectorstore, build_qa_chain
from loader import load_documents
from difflib import SequenceMatcher

# Załaduj token
load_dotenv()
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

st.set_page_config(page_title="LLM Doc Chatbot", layout="wide", initial_sidebar_state="collapsed")

# Styl CSS — dark mode + bąbelki
dark_mode_css = """
<style>
    .main {
        background-color: #121212;
        color: #E0E0E0;
    }
    #MainMenu, footer, header {
        visibility: hidden;
    }
    .user-msg {
        background-color: #2a2a2a;
        color: #eee;
        padding: 12px 18px;
        border-radius: 18px 18px 0 18px;
        margin: 8px 0 8px 20%;
        max-width: 70%;
        text-align: right;
        float: right;
        clear: both;
        font-size: 15px;
    }
    .bot-msg {
        background-color: #333333;
        color: #ddd;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 0;
        margin: 8px 20% 8px 0;
        max-width: 70%;
        text-align: left;
        float: left;
        clear: both;
        font-size: 15px;
    }
    .clearfix::after {
        content: "";
        clear: both;
        display: table;
    }
    .stTextInput>div>div>input {
        background-color: #222 !important;
        color: #eee !important;
        border-radius: 8px;
        border: none;
        padding: 10px;
        font-size: 16px;
    }
</style>
"""
st.markdown(dark_mode_css, unsafe_allow_html=True)

# Nagłówek
st.title("LLM Doc Chatbot")
st.write("Zadaj pytanie na podstawie dokumentów")

# Inicjalizacja historii
if "history" not in st.session_state:
    st.session_state.history = []

# Inicjalizacja bazy wiedzy
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
    answer = result["result"]
    st.session_state.history.append({
        "role": "bot",
        "content": answer,
        "sources": result.get("source_documents", [])
    })

# Obsługa inputu z resetem
def handle_input():
    query = st.session_state.input
    if query:
        ask_question(query)
        st.session_state.input = ""
        st.rerun()

# Pole tekstowe
st.text_input("Zadaj pytanie...", key="input", on_change=handle_input)

# Wyświetlanie historii jako bąbelki
for msg in st.session_state.history:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg clearfix">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg clearfix">{msg["content"]}</div>', unsafe_allow_html=True)
        sources = msg.get("sources", [])
        if sources:
            st.markdown("**Źródła:**")
            for i, doc in enumerate(sources):
                filename = doc.metadata.get("filename", "Nieznany plik")
                page = doc.metadata.get("page", None)
                source_info = f"{filename}"
                if page is not None:
                    source_info += f", strona {page + 1}"
                snippet = doc.page_content[:300].strip().replace("\n", " ")
                st.markdown(f"{i+1}. `{source_info}` - {snippet}...")
