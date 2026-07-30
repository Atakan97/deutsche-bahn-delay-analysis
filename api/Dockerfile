# Dockerfile, Container image for the delay prediction API
# Designed for deployment to Render (Docker Web Service)

FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install Python dependencies first
COPY api/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API application code and model artifact
COPY api/ ./api/

# Expose port 8000
EXPOSE 8000

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
