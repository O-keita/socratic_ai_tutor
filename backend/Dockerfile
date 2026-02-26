FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Runtime directories (overridden by volume mounts in docker-compose)
RUN mkdir -p /app/models /app/data /app/content

EXPOSE 81

# PORT defaults to 80; docker-compose sets it explicitly via environment
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-81} --workers 1"]
