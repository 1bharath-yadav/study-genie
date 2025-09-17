FROM python:3.12-slim

# Environment
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860
ENV PATH="/app/venv/bin:$PATH"

WORKDIR /app

# Prepare persistent data mount used by Spaces (/data)
RUN mkdir -p /data && chown -R root:root /data
ENV HF_HOME=/data/.huggingface

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (no --prefix, we move binary ourselves)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv ~/.cargo/bin/uv /usr/local/bin/uv

# Copy requirements and install inside virtualenv
COPY requirements.txt .

RUN uv venv /app/venv && \
    uv pip install -r requirements.txt

# Copy project files
COPY . .

# Ensure /data writable
RUN chmod -R 0777 /data || true

# Create non-root user
RUN useradd -m -u 1000 user && \
    chown -R user:user /app
USER user

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start app (uvicorn inside venv)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
