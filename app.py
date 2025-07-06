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

# Dark mode + chat bubbles CSS with improvements
dark_mode_css = """
<style>
    html, body, .block-container, main, .appview-container, .main {
        background-color: #0a0f1f !important;
        background-image: linear-gradient(135deg, #0a0f1f, #1e2238, #0a0f1f);
        background-repeat: no-repeat;
        background-size: cover;
        background-attachment: fixed;
        color: #E0E0E0 !important;
        min-height: 100vh !important;
        height: 100% !important;
    }
    html {
        overflow-x: hidden;
    }
    .css-18e3th9, .css-1dp5vir {
        background-color: transparent !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
    }
    #MainMenu, footer, header {
        visibility: hidden;
    }
    h1 {
        text-shadow: 1px 1px 3px rgba(0,0,0,0.7);
    }
    .user-msg {
        background-color: #1c1c2c;
        color: #eee;
        padding: 12px 18px;
        border-radius: 18px 18px 0 18px;
        margin: 8px 0 8px 12%;
        max-width: 75%;
        text-align: right;
        float: right;
        clear: both;
        font-size: 15px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.6);
    }
    .bot-msg {
        background-color: #252540;
        color: #ddd;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 0;
        margin: 8px 12% 8px 0;
        max-width: 75%;
        text-align: left;
        float: left;
        clear: both;
        font-size: 15px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.6);
    }
    .clearfix::after {
        content: "";
        clear: both;
        display: table;
    }
    .stTextInput>div>div>input {
        background-color: #222 !important;
        color: #fff !important;
        border-radius: 8px;
        border: none;
        padding: 10px;
        font-size: 16px;
    }
    .stTextInput>div>div>input::placeholder {
        color: #fff !important;
        opacity: 1 !important;
    }
    .source-box {
        background-color: #1a1a2b;
        border-radius: 10px;
        padding: 10px;
        margin-top: 10px;
        font-size: 13px;
    }
</style>
"""
st.markdown(dark_mode_css, unsafe_allow_html=True)

# ---- UI: LEFT + RIGHT columns ----
col1, col2 = st.columns([2, 1])

with col1:
    st.title("AI Product Assistant")
    st.markdown("""
**AI Product Assistant – chatbot oparty na Retrieval-Augmented Generation (RAG)**  
Zaawansowany asystent AI wspierający klientów w podejmowaniu decyzji zakupowych poprzez połączenie dużych modeli językowych (LLM) z niestandardową bazą wektorową. System został wytrenowany specjalnie na podstawie parametrów technicznych i specyfikacji smartfonów Samsung wymienionych po prawej stronie, co umożliwia precyzyjne i kontekstowe odpowiedzi z podaniem źródeł.

Chatbot wykorzystuje Pythona i LangChain do orkiestracji procesu RAG, FAISS do efektywnego wyszukiwania podobieństw oraz dostrojony model LLaMA 3.1 hostowany na Hugging Face Hub. Przetwarza obszerne katalogi produktów i techniczne pliki PDF, aby rozumieć pytania w języku naturalnym i na bieżąco dostarczać relewantne informacje.

Projekt demonstruje praktyczne umiejętności inżynierii GenAI, w tym embedding dokumentów, wyszukiwanie wektorowe, integrację LLM oraz przyjazny interfejs użytkownika, co czyni go idealnym rozwiązaniem dla e-commerce i działów obsługi klienta.

**Przykładowe pytania:**  
- *Który telefon ma najwięcej pamięci RAM?*  
- *Jakie są różnice między Galaxy S25 Ultra a S24 FE?*  
- *Czy Galaxy Z Flip 6 obsługuje Dual SIM?*  

⏳ **Proszę chwilę poczekać, aż aplikacja się załaduje...**  
""")

with col2:
    st.markdown("### 🗂️ Modele w bazie wiedzy:")
    st.markdown("""
- **Galaxy A56**  
- **Galaxy S24 / S24 FE / S24+ / S24 Ultra**  
- **Galaxy S25 / S25 Edge / S25+**  
- **Galaxy Z Flip 6**  
- **Galaxy Z Fold 5 / Z Fold 6**  
- **Galaxy XCover 6**
    """)

# ---- Chatbot logic ----
if "history" not in st.session_state:
    st.session_state.history = []

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

def handle_input():
    query = st.session_state.input
    if query:
        ask_question(query)
        st.session_state.input = ""

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

st.text_input(label="", key="input", on_change=handle_input, placeholder="Zadaj pytanie...")

# Footer - Made by Kacper Mróz
st.markdown(
    """
    <div style="text-align:center; font-size:12px; color:#888; margin-top: 40px;">
        Made by <strong>Kacper Mróz</strong>, 2025
    </div>
    """,
    unsafe_allow_html=True
)
