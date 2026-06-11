# docagent — CPU image serving the web API + chat UI.
#
#   docker build -t docagent .
#   # mount a prebuilt Chroma store + bm25s index at /data, or ingest into the volume:
#   docker run -p 8000:8000 -v $PWD/chroma_db:/data/chroma -e OPENAI_API_KEY=sk-... docagent
#
# The image carries no knowledge base; point CHROMA_PATH (default /data/chroma) at
# a mounted volume you ingested into, e.g.:
#   docker run --rm -v $PWD/papers:/papers -v $PWD/chroma_db:/data/chroma docagent \
#     python -m docagent.ingest --path /papers --reset
FROM python:3.11-slim

WORKDIR /app

# Install CPU-only torch first so the heavy CUDA wheels are never pulled in.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Dependency layer (cached unless pyproject changes).
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

# Embedding + reranker (and optional NLI) models download on first request; set a
# cache dir so a mounted volume can persist them across container restarts.
ENV CHROMA_PATH=/data/chroma \
    HF_HOME=/data/hf \
    LOG_LEVEL=INFO

EXPOSE 8000
CMD ["uvicorn", "docagent.web:app", "--host", "0.0.0.0", "--port", "8000"]
