# Dockerfile, Container image for the delay prediction API
# Designed for deployment to Hugging Face Spaces (Docker SDK)

FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install Python dependencies first
COPY api/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API application code and model artifact
COPY api/ ./api/

# Expose port 7860 (Hugging Face Spaces convention)
EXPOSE 7860

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
