# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV MODEL_PATH=/app/models/socratic-q4_k_m.gguf

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create the models directory if it doesn't exist (though it should be copied or mounted)
RUN mkdir -p /app/models

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
# We use uvicorn with 0.0.0.0 to allow external connections on the Droplet
CMD ["uvicorn", "main.py:app", "--host", "0.0.0.0", "--port", "8000"]
