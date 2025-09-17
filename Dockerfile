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

# Set working directory and create necessary directories
WORKDIR /app

# Prepare persistent data mount with correct permissions
RUN mkdir -p /data && chmod -R 777 /data
ENV HF_HOME=/data/.huggingface

# Copy requirements first (for better build cache)
COPY pyproject.toml .

# Install uv (stable prebuilt binary) into /usr/local/bin
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies with uv
RUN uv sync
   

# Copy project files
COPY . .

# Change ownership to user after all files are copied
RUN chown -R user:user /app

# Switch to non-root user
USER user

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start the application using venv-installed uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
