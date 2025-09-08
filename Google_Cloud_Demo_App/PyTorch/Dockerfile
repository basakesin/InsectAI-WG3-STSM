# Python image'i kullanarak başlıyoruz
FROM python:3.10-slim

# Gerekli sistem paketlerini yükleyelim
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Çalışma dizini oluştur
WORKDIR /app

# Gereksinimlerimizi yükleyelim
COPY requirements.txt requirements.txt
RUN python -m pip install -r requirements.txt

# Uygulama dosyalarını kopyalıyoruz
COPY . .

# Port açıyoruz
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]
