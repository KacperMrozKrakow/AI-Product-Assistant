FROM python:3.10-slim

# Dodaj użytkownika bez praw root (opcja bezpieczeństwa)
RUN useradd -m -u 1000 user
USER user

ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Kopiujemy wymagania i instalujemy pakiety
COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Kopiujemy resztę aplikacji
COPY --chown=user . /app

# Eksponujemy port 8501, bo taki używa Streamlit
EXPOSE 8501

# Uruchom Streamlit w trybie headless na porcie 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.headless=true"]