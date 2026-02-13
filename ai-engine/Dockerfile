FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Créer le dossier models s'il n'existe pas
RUN mkdir -p models
CMD ["python", "brain.py"]