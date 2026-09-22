# Simple Dockerfile for the Zepto Support Assistant FastAPI app.
# Runs in MOCK_LLM mode by default - no API key needed.

FROM python:3.11-slim

WORKDIR /app

# install only what this module needs (smaller image than the whole
# project's requirements.txt, which also covers data_pipeline/analytics)
COPY support_assistant_requirements.txt .
RUN pip install --no-cache-dir -r support_assistant_requirements.txt

# copy the app code, the policy docs, and the already-built chroma_store
# (if chroma_store isn't present, run ingest.py once before building)
COPY docs/ ./docs/
COPY main.py .
COPY ingest.py .
COPY chroma_store/ ./chroma_store/

# MOCK_LLM=1 is the default graded mode - no LLM API key required
ENV MOCK_LLM=1

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]