# Dockerfile for Hugging Face Spaces (Docker SDK) deployment.
# Builds the policy-RAG FastAPI app with hybrid retrieval + reranker + Qwen2.5-0.5B.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf-cache \
    TRANSFORMERS_CACHE=/app/.hf-cache \
    SENTENCE_TRANSFORMERS_HOME=/app/.hf-cache

WORKDIR /app

# System deps for pypdf / faiss / torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps (CPU torch)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt && \
    pip install transformers>=4.45.0

# Pre-download retrieval / reranker / generator models so the Space starts fast
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" && \
    python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')" && \
    python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct'); \
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')"

# App code
COPY . .

# Hugging Face Spaces routes traffic to port 7860
ENV PORT=7860
EXPOSE 7860

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "7860"]
