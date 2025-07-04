---
title: LLM Doc Chatbot
sdk: streamlit
app_file: app.py
---

# LLM Doc Chatbot

**LLM Doc Chatbot** to nowoczesna aplikacja typu Retrieval-Augmented Generation (RAG) do zadawania pytań na podstawie dokumentów.  
Projekt integruje następujące technologie i komponenty:

- **LangChain**: zarządzanie łańcuchem logiki, splitowanie i wyszukiwanie dokumentów.
- **FAISS**: szybka i skalowalna baza wektorowa do wyszukiwania podobieństwa tekstu.
- **Hugging Face Inference API**: korzystanie z potężnych modeli LLaMA 3.1-8B w trybie instrukcyjnym.
- **Custom LLM Wrapper (`HFInferenceLLM`)**: klasa integrująca API Hugging Face z LangChain.
- **Streamlit**: szybki, nowoczesny interfejs webowy z ciemnym trybem i chatbąbelkami.
- **Dotenv**: zarządzanie sekretami i kluczami API w pliku `.env`.

## Funkcjonalności:

- Budowa bazy wiedzy na podstawie PDF i dokumentów tekstowych.
- Wyszukiwanie odpowiedzi z dokładnym cytowaniem źródeł (plik, strona).
- Intuicyjny interfejs czatu z historią pytań i odpowiedzi.
- Tryb ciemny i nowoczesny, minimalistyczny design.

---

Zapraszam do testowania i rozwoju!  
Projekt idealny jako pokaz umiejętności dla ról GenAI Engineer, ML Engineer, czy AI Researcher.

---

## Technologie

| Technologia           | Opis                                                       |
|----------------------|------------------------------------------------------------|
| Python               | Główny język programowania                                  |
| LangChain            | Framework do łączenia modeli LLM z danymi i pipeline'ami   |
| FAISS                | Biblioteka do wyszukiwania podobieństw wektorowych         |
| Hugging Face Hub     | Hosting modeli i API do inference                           |
| Streamlit            | Framework do szybkiego tworzenia web UI                    |
| dotenv               | Łatwe zarządzanie sekretami i konfiguracją                  |

---

## Jak uruchomić

1. Skonfiguruj plik `.env` z kluczem `HUGGINGFACEHUB_API_TOKEN`.  
2. Uruchom `streamlit run app.py` lokalnie.  
3. Lub wrzuć na Hugging Face Spaces z odpowiednią konfiguracją.

---

*Plik `rag_pipeline.py` zawiera logikę tworzenia i ładowania bazy FAISS oraz wywoływania modelu LLaMA.*
