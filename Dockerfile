# Dockerfile optimized for Hugging Face Spaces with uv and virtual environment
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# Set work directory
WORKDIR /app

# Prepare persistent data mount used by Spaces (/data)
RUN mkdir -p /data && chown -R root:root /data

# Persist Hugging Face caches under /data
ENV HF_HOME=/data/.huggingface

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv to /usr/local/bin
RUN curl -LsSf https://astral.sh/uv/install.sh | sh -s -- --prefix /usr/local --quiet

# Copy requirements
COPY requirements.txt .

# Create virtual environment and install Python dependencies with uv
RUN uv venv /app/venv && \
    uv pip install --python /app/venv/bin/python -r requirements.txt

# Copy project files
COPY . .

# Ensure /data is writable by runtime user (will be adjusted later)
RUN chmod -R 0777 /data || true

# Create a non-root user and fix ownership (including venv)
RUN useradd -m -u 1000 user && \
    chown -R user:user /app
USER user

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start the application (activating venv)
CMD ["sh", "-c", ". /app/venv/bin/activate && exec uvicorn app.main:app --host 0.0.0.0 --port 7860 --workers 1"]