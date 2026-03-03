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

# Save config.json to a location outside the /app/data volume mount.
# The entrypoint script copies it back on every container start, ensuring
# code-level config changes always take effect after a rebuild.
RUN cp /app/data/config.json /app/_image_config.json

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 8880

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8880} --workers 1"]
