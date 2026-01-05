FROM python:3.10-slim

# Set working directory
WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY api.py .
COPY inference.py .
COPY mlruns ./mlruns

EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
