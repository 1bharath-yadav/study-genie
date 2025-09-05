# Use an official Python image as the base
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install uv
RUN pip install --upgrade pip && pip install uv

# Copy dependency files and install dependencies with uv
COPY pyproject.toml uv.lock ./
RUN uv sync 

# Copy project
COPY . .

# Expose port (Hugging Face Spaces uses port 7860)
EXPOSE 7860

# Start the backend (adjust the command as per your backend entrypoint)
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
