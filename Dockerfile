# Dockerfile optimized for Hugging Face Spaces with uv and virtual environment
FROM python:3.12-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860
ENV PATH="/app/venv/bin:$PATH"

# Install system dependencies as root
RUN apt-get update && apt-get install -y \
    curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 user

# Set working directory
WORKDIR /app

# Prepare persistent data mount with correct permissions for HF Spaces
RUN mkdir -p /data && chmod -R 777 /data
ENV HF_HOME=/data/.huggingface

# Copy requirements first (for better build cache)
COPY pyproject.toml .

# Copy uv binaries from official image (pinned version; already in /bin PATH)
COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /uvx /bin/

# Create virtual environment at /app/venv and install dependencies with uv
ENV VIRTUAL_ENV=/app/venv
RUN uv venv $VIRTUAL_ENV
RUN uv sync

# Copy project files
COPY . .

# Change ownership to non-root user after all files are copied
RUN chown -R user:user /app

# Switch to non-root user
USER user

# Expose port
EXPOSE 7860

# Health check (assumes your app has a /health endpoint; adjust if needed)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start the application using venv-installed uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]