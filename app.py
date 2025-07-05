import streamlit as st
from dotenv import load_dotenv
import os
from pathlib import Path
from rag_pipeline import load_vectorstore, build_qa_chain
from loader import load_documents
from difflib import SequenceMatcher
import threading

# Load token
load_dotenv()
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

st.set_page_config(page_title="AI Product Assistant", layout="wide", initial_sidebar_state="collapsed")

# CSS: gradient, dark/light mode, avatars, spinner
css = """
<style>
    body, .main {
        background: linear-gradient(135deg, #1f1f1f 0%, #121212 100%);
        color: #E0E0E0;
        transition: background 0.3s ease, color 0.3s ease;
    }
    body.light, .main.light {
        background: linear-gradient(135deg, #e0e0e0 0%, #f9f9f9 100%);
        color: #222222;
    }
    #MainMenu, footer, header { visibility: hidden; }

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
        position: relative;
        padding-left: 40px;
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
        position: relative;
        padding-left: 40px;
    }
    body.light .user-msg {
        background-color: #d0d0d0;
        color: #111;
    }
    body.light .bot-msg {
        background-color: #e5e5e5;
        color: #111;
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
        transition: background-color 0.3s ease, color 0.3s ease;
    }
    body.light .stTextInput>div>div>input {
        background-color: #f0f0f0 !important;
        color: #222 !important;
    }
    .source-box {
        background-color: #1c1c1c;
        border-radius: 10px;
        padding: 10px;
        margin-top: 10px;
        font-size: 13px;
    }
    body.light .source-box {
        background-color: #e0e0e0;
        color: #222;
    }

    .user-msg::before { content: "🧑"; position: absolute; left: 10px; top: 12px; font-size: 22px; }
    .bot-msg::before { content: "🤖"; position: absolute; left: 10px; top: 12px; font-size: 22px; }

    .spinner {
        border: 4px solid rgba(255, 255, 255, 0.15);
        border-top: 4px solid #09d3ac;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        animation: spin 1s linear infinite;
        margin: 10px auto;
    }
    body.light .spinner {
        border: 4px solid rgba(0, 0, 0, 0.1);
        border-top: 4px solid #0a9484;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    .toggle-container {
        position: fixed;
        top: 10px;
        right: 15px;
        z-index: 9999;
        user-select: none;
        font-family: sans-serif;
        display: flex;
        align-items: center;
    }
    .toggle-label { margin-right: 8px; font-weight: 600; }
    .toggle-switch {
        position: relative;
        display: inline-block;
        width: 50px;
        height: 24px;
    }
    .toggle-switch input {
        opacity: 0;
        width: 0;
        height: 0;
    }
    .slider {
        position: absolute;
        cursor: pointer;
        background-color: #ccc;
        border-radius: 34px;
        top: 0; left: 0; right: 0; bottom: 0;
        transition: .4s;
    }
    .slider:before {
        position: absolute;
        content: "";
        height: 18px;
        width: 18px;
        left: 3px;
        bottom: 3px;
        background-color: white;
        border-radius: 50%;
        transition: .4s;
    }
    input:checked + .slider {
        background-color: #09d3ac;
    }
    input:checked + .slider:before {
        transform: translateX(26px);
    }
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# Dark mode toggle UI
def dark_mode_toggle():
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = True

    toggle_html = f"""
    <div class="toggle-container">
        <span class="toggle-label">Dark Mode</span>
        <label class="toggle-switch">
            <input type="checkbox" id="dark-mode-checkbox" {'checked' if st.session_state.dark_mode else ''}>
            <span class="slider"></span>
        </label>
    </div>
    <script>
        const checkbox = window.parent.document.getElementById('dark-mode-checkbox');
        function setDarkMode(dark) {{
            if(dark) {{
                document.body.classList.remove('light');
            }} else {{
                document.body.classList.add('light');
            }}
        }}
        checkbox.addEventListener('change', e => {{
            setDarkMode(e.target.checked);
            window.parent.streamlitDarkMode = e.target.checked;
        }});
        // Initial
        setDarkMode({str(st.session_state.dark_mode).lower()});
    </script>
    """
    st.components.v1.html(toggle_html, height=40)

dark_mode_toggle()

# Welcome text
st.markdown("""
# AI Product Assistant
Ten inteligentny asystent odpowiada na pytania w oparciu o dokumenty (np. PDF-y z ofertami, instrukcjami, katalogami).  
Możesz go wykorzystać np. jako wsparcie klienta — wystarczy załadować dokumenty, a użytkownik może zadawać pytania w języku naturalnym.  
Przykład: *"Czy produkt X obsługuje integrację z systemem Y?"*

⏳ **Poczekaj kilka sekund, aż aplikacja się załaduje...**
""")

# Initialize states
if "history" not in st.session_state:
    st.session_state.history = []
if "loading" not in st.session_state:
    st.session_state.loading = False

# Load or build knowledge base
if not Path("vectorstore/index.faiss").exists():
    with st.spinner("Tworzę bazę wiedzy..."):
        docs = load_documents("data/docs/")
        from rag_pipeline import create_vectorstore
        create_vectorstore(docs)

db = load_vectorstore()
qa_chain = build_qa_chain(db)

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def play_notification():
    st.markdown("""
    <audio autoplay hidden>
      <source src="https://actions.google.com/sounds/v1/cartoon/clang_and_wobble.ogg" type="audio/ogg">
    </audio>
    """, unsafe_allow_html=True)

def answer_question(query):
    st.session_state.loading = True
    # Show spinner immediately
    st.experimental_rerun()  # <-- this line removed to avoid error, replaced by a threaded approach below

# Threaded approach to avoid blocking UI and rerun:
def ask_and_store_answer(query):
    result = qa_chain(query)
    answer = result["result"]
    st.session_state.history.append({"role": "user", "content": query})
    st.session_state.history.append({
        "role": "bot",
        "content": answer,
        "sources": result.get("source_documents", [])
    })
    st.session_state.loading = False
    play_notification()

def handle_input():
    query = st.session_state.input
    if query and not st.session_state.loading:
        st.session_state.loading = True
        # Start thread so UI does not block (simulate async)
        thread = threading.Thread(target=ask_and_store_answer, args=(query,))
        thread.start()
        st.session_state.input = ""

# Show spinner if loading
if st.session_state.loading:
    st.markdown('<div class="spinner"></div>', unsafe_allow_html=True)

# Display chat history
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
                    st.markdown(f'<div class="source-box">{i+1}. `{source_info}` - {snippet}...</div>', unsafe_allow_html=True)

# Input at bottom
st.text_input("Zadaj pytanie...", key="input", on_change=handle_input)
