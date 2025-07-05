import streamlit as st
from dotenv import load_dotenv
import os
from pathlib import Path
from rag_pipeline import load_vectorstore, build_qa_chain
from loader import load_documents
from difflib import SequenceMatcher

# Load HuggingFace token
load_dotenv()
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

st.set_page_config(page_title="AI Product Assistant", layout="wide", initial_sidebar_state="collapsed")

# Dark mode + bubbles CSS
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

# Title and description
st.title("AI Product Assistant")
st.markdown("""
Ten inteligentny asystent odpowiada na pytania w oparciu o dokumenty (np. PDF-y z ofertami, instrukcjami, katalogami).  
Możesz go wykorzystać np. jako wsparcie klienta — wystarczy załadować dokumenty, a użytkownik może zadawać pytania w języku naturalnym.  
Przykład: *"Czy produkt X obsługuje integrację z systemem Y?"*

⏳ **Poczekaj kilka sekund, aż aplikacja się załaduje...**
""")

# Initialize chat history and loading flag
if "history" not in st.session_state:
    st.session_state.history = []

if "loading" not in st.session_state:
    st.session_state.loading = False

# Build knowledge base if missing
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

# Callback to handle input submit and reset loading state
def handle_input():
    query = st.session_state.input
    if query:
        st.session_state.loading = True
        ask_question(query)
        st.session_state.input = ""
        st.session_state.loading = False

# Display chat history first
for msg in st.session_state.history:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg clearfix">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg clearfix">{msg["content"]}</div>', unsafe_allow_html=True)

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
                    st.markdown(f'<div class="source-box">{i+1}. {source_info} - {snippet}...</div>', unsafe_allow_html=True)

# Show spinner during loading
if st.session_state.loading:
    st.markdown(
        """
        <div style="display:flex; justify-content:center; margin: 10px;">
            <div style="
                border: 4px solid rgba(255, 255, 255, 0.15);
                border-top: 4px solid #09d3ac;
                border-radius: 50%;
                width: 30px;
                height: 30px;
                animation: spin 1s linear infinite;">
            </div>
        </div>
        <style>
        @keyframes spin {
          0% { transform: rotate(0deg);}
          100% { transform: rotate(360deg);}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Input at the bottom
st.text_input("Zadaj pytanie...", key="input", on_change=handle_input)
