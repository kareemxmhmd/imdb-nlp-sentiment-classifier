FROM python:3.11-slim

WORKDIR /app

# Install system dependencies if any needed for NLP/ML (e.g. build-essential)
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directories
RUN mkdir -p /app/src /app/models /app/data

# Copy source code and models
COPY src/ /app/src/
COPY models/ /app/models/

# Set Python path
ENV PYTHONPATH=/app

# Expose API port
EXPOSE 8000

# Command to run the API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
