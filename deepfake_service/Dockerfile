# Use Python 3.9 slim as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy only requirements file first for better caching
COPY requirements.txt requirements.txt

# Upgrade pip and install dependencies with increased timeout
RUN pip install --upgrade pip \
    && pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Expose the port for FastAPI
EXPOSE 8002

# Command to run FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
