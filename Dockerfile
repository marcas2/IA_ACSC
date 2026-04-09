FROM python:3.11-slim

# Dependencias del sistema para librosa/scipy
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Crear directorios necesarios
RUN mkdir -p models data/reports

EXPOSE 5000

CMD ["python", "main.py"]
