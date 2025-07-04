from dotenv import load_dotenv
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from huggingface_hub import InferenceClient
from typing import Optional, List
from pydantic import PrivateAttr, Field

# Załaduj zmienne środowiskowe z pliku `.env`
load_dotenv()
print("Token z env:", os.getenv("HUGGINGFACEHUB_API_TOKEN"))

class HFInferenceLLM(LLM):
    model_name: str = Field(...)
    token: Optional[str] = Field(default=None)
    temperature: float = 0.5
    max_new_tokens: int = 512

    _client: InferenceClient = PrivateAttr()

    def __init__(self, **data):
        if not data.get("token"):
            data["token"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        super().__init__(**data)
        self._client = InferenceClient(token=self.token)

    @property
    def _llm_type(self) -> str:
        return "huggingface_inference"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        messages = [
            {"role": "user", "content": prompt}
        ]
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            stop=stop,
        )
        generated_text = response.choices[0].message["content"]
        return generated_text


def create_vectorstore(documents, persist_path="vectorstore/"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)  # <-- to jest kluczowa zmiana
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(chunks, embedding=embedder)
    vectordb.save_local(persist_path)
    return vectordb


def load_vectorstore(persist_path="vectorstore/"):
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(persist_path, embedder, allow_dangerous_deserialization=True)


def build_qa_chain(vectordb):
    llm = HFInferenceLLM(
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        temperature=0.5,
        max_new_tokens=512
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
    )
