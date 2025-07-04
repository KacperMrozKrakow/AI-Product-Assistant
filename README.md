---
title: AI Product Assistant
colorFrom: indigo
colorTo: gray
sdk: docker
sdk_version: "1.0"
app_file: app.py
pinned: false
---

#  LLM Doc Chatbot – Your AI Assistant for Document Understanding

**LLM Doc Chatbot** is an intelligent assistant designed to answer questions based on your custom documents – like manuals, product sheets, internal wikis or policies.

This is a fully-featured Retrieval-Augmented Generation (RAG) application combining powerful LLMs with local vector search and an elegant chat interface.

---
##  Technologies Used

- **LangChain** – for document splitting, embeddings, and QA pipelines.
- **FAISS** – efficient vector database for semantic search over document chunks.
- **Hugging Face Inference API** – access to LLaMA 3.1–8B Instruct model via hosted API.
- **Custom `HFInferenceLLM`** – wrapper class connecting Hugging Face to LangChain seamlessly.
- **Streamlit** – interactive and minimalist web UI with chat-style UX and dark mode.
- **Dotenv** – secure API key management using environment variables.

---

## Key Features

- Upload PDFs and plain-text docs to create a searchable knowledge base.
- Ask questions in natural language and receive relevant answers with:
  - **Citations** (file name and page)
  - **Expandable sources** within each response
- Clean, scrollable conversation with bubble-style layout.
- Fully responsive dark theme for desktop and mobile.
- Built-in loading indicators and UX safeguards.

---

## Tech Stack

| Technology         | Purpose                                                    |
|--------------------|------------------------------------------------------------|
| Python             | Core programming language                                  |
| LangChain          | LLM orchestration and retriever pipeline                   |
| FAISS              | Fast vector-based search engine                            |
| Hugging Face Hub   | Hosting and inference for open-source LLMs                 |
| Streamlit          | Web-based UI framework                                     |
| dotenv             | Secret and config management                               |

---

## How to Run Locally

1. Clone the repository and add a `.env` file with your Hugging Face token:

    ```bash
    HUGGINGFACEHUB_API_TOKEN=hf_...
    ```

2. Run locally with Streamlit:

    ```bash
    streamlit run app.py
    ```

---

## Deploy on Hugging Face Spaces

This app is compatible with `streamlit` or `docker` SDKs. To deploy:
- Choose a Space type (Streamlit recommended)
- Upload your code and `.env` (via secrets)
- Done 🎉

---

## Use Case Example

This chatbot is ideal for:

> Letting your clients interact with documentation such as product catalogs, internal processes, or legal policies – through intelligent search and chat.

---

## Project Scope

This app serves as a **portfolio-level GenAI Engineering project**, showcasing my skills in:

- RAG architecture
- LLM API integration
- Vector databases
- UI/UX with Streamlit
- Real-world deployment (HF Spaces)

---

*Main logic resides in `rag_pipeline.py`, where documents are chunked, embedded, stored in FAISS, and queried using the LLaMA model.*
