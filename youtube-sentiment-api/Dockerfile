# Base image légère
FROM python:3.10-slim

# Evite les caches python + logs inutiles
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 1) Copier requirements d'abord pour profiter du cache Docker
COPY requirements.txt .

# 2) Installer dépendances
RUN pip install --no-cache-dir -r requirements.txt

# 3) Copier uniquement le nécessaire
COPY app_api.py .
COPY models/ models/

# Port HF Spaces
EXPOSE 7860

# Lancer l'API sur 0.0.0.0:7860
CMD ["uvicorn", "app_api:app", "--host", "0.0.0.0", "--port", "7860"]
