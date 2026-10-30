# Prediction server for Vertex AI. The PyPI torch wheel bundles the CUDA 12.4 runtime,
# so the image runs on Vertex AI's GPU machines (e.g. NVIDIA L4) without a CUDA base image.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY server.py .

# Vertex AI sets AIP_HTTP_PORT; 8080 is the default.
CMD ["sh", "-c", "exec uvicorn server:app --host 0.0.0.0 --port ${AIP_HTTP_PORT:-8080}"]
