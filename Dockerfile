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


# Install uv to /usr/local/bin


RUN curl -LsSf https://astral.sh/uv/install.sh | sh -s --prefix /usr/local --quiet

# COPY requirements.txt .

RUN uv venv /app/venv && \



# uv pip install --python /app/venv/bin/python -r requirements.txt
# Copy project files

COPY . .


# Create a non-root user and fix ownership (including venv)


RUN useradd -m -u 1000 user && \


chown -R user:user /app

USER user


# Prepare persistent data mount with correct permissions for HF Spaces
RUN mkdir -p /data && chmod -R 777 /data
ENV HF_HOME=/data/.huggingface

# Copy requirements first (for better build cache)
COPY pyproject.toml .

# Copy uv binaries from official image (pinned version; already in /bin PATH)

RUN uv sync

# Copy project files
COPY . .



EXPOSE 7860

# Health check (assumes your app has a /health endpoint; adjust if needed)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start the application using venv-installed uvicorn
CMD ["sh", "-c", ". /app/venv/bin/activate && exec uvicorn app.main:app --host 0.0.0.0 --port 7860 --workers 1"]

