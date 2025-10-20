FROM python:3.11-slim

WORKDIR /app

# Tools needed for some packages (scikit-learn etc)
RUN apt-get update && \
    apt-get install -y build-essential gcc g++ gfortran && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --upgrade pip --no-cache-dir
RUN pip install --no-cache-dir -r requirements.txt

# Copy pre-downloaded NLTK data into image and set lookup path
COPY nltk_data /usr/local/share/nltk_data
ENV NLTK_DATA=/usr/local/share/nltk_data

# Copy app code (and your .pkl model files)
COPY . .

EXPOSE 8000

# Adjust module name if your main file is different (main:app means main.py has `app = FastAPI()`)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
