FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and model artifacts
COPY main.py .
COPY models/ ./models/

# Hugging Face Spaces requires the app to listen on port 7860
# and expects the container to run as a non-root user (UID 1000).
RUN useradd -m -u 1000 appuser
USER appuser

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
