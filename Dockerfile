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
    
RUN mkdir -p /data && chmod -R 777 /data


# Create a non-root user with UID 1000
RUN useradd -m -u 1000 user

# Switch to the non-root user
USER user

# Set working directory for the app
WORKDIR /home/user/app

# Copy application files and set ownership to the non-root user
COPY --chown=user . /home/user/app

# Download and run uv installer as the non-root user
RUN curl -fsSL https://astral.sh/uv/install.sh -o /home/user/uv-installer.sh \
    && chmod +x /home/user/uv-installer.sh \
    && sh /home/user/uv-installer.sh \
    && rm /home/user/uv-installer.sh

# Environment setup - make sure uv is in the PATH
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Install Python dependencies with uv (creates .venv owned by user)
RUN uv sync --no-cache-dir

# Prepare persistent data mount with correct permissions for HF Spaces
ENV HF_HOME=/data/.huggingface

# Expose port
EXPOSE 7860

# Health check (assumes your app has a /health endpoint; adjust if needed)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start the application using venv-installed uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]