FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        gcc \
        libxml2-dev \
        libxslt1-dev \
        tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY . .
RUN sed -i 's/\r$//' docker/entrypoint.sh \
    && chmod +x docker/entrypoint.sh \
    && mkdir -p data/raw data/pdfs data/extracted data/logs data/reports data/kg

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["python", "run.py", "--help"]
