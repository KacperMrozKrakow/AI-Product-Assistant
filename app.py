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

st.set_page_config(
    page_title="AI Product Assistant", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# Dark & Light mode CSS + avatars + gradient background + spinner animation
dark_light_css = """
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
    /* Light mode overrides */
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

    /* Avatars */
    .user-msg::before {
        content: "🧑";
        position: absolute;
        left: 10px;
        top: 12px;
        font-size: 22px;
    }
    .bot-msg::before {
        content: "🤖";
        position: absolute;
        left: 10px;
        top: 12px;
        font-size: 22px;
    }

    /* Spinner */
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

    /* Toggle button */
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
    .toggle-label {
        margin-right: 8px;
        font-weight: 600;
    }
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
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
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

st.markdown(dark_light_css, unsafe_allow_html=True)

# Light/dark mode toggle UI
def mode_toggle():
    # default dark mode on first load
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = True

    toggle_container = st.empty()
    with toggle_container.container():
        checked = st.checkbox("Dark Mode", value=st.session_state.dark_mode, key="dark_mode_checkbox")
        st.session_state.dark_mode = checked

    # Add a small JS snippet to switch body class based on checkbox
    js_code = f"""
    <script>
    const darkMode = {str(st.session_state.dark_mode).lower()};
    if(darkMode){{
        document.body.classList.add('light') === false && document.body.classList.remove('light');
        document.body.classList.remove('light');
    }} else {{
        document.body.classList.add('light');
    }}
    </script>
    """
    st.components.v1.html(js_code, height=0, width=0)

mode_toggle()

# Notification sound using JS Audio API
def play_notification_sound():
    audio_html = """
    <audio autoplay hidden>
      <source src="https://actions.google.com/sounds/v1/cartoon/clang_and_wobble.ogg" type="audio/ogg">
      Your browser does not support the audio element.
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

# Initialize chat history & loading state
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
    st.session_state.loading = True
    st.experimental_rerun()  # Show spinner

def process_question(query):
    # Real answering function
    result = qa_chain(query)
    answer = result["result"]
    st.session_state.history.append({"role": "user", "content": query})
    st.session_state.history.append({
        "role": "bot",
        "content": answer,
        "sources": result.get("source_documents", [])
    })
    st.session_state.loading = False
    play_notification_sound()

def handle_input():
    query = st.session_state.input
    if query:
        ask_question(query)
        process_question(query)
        st.session_state.input = ""

# Input and spinner display
if st.session_state.get("loading", False):
    st.markdown('<div class="spinner"></div>', unsafe_allow_html=True)

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
                    st.markdown(f'<div class="source-box">{i+1}. `{source_info}` - {snippet}...</div>', unsafe_allow_html=True)

# Input at the bottom
st.text_input("Zadaj pytanie...", key="input", on_change=handle_input)
