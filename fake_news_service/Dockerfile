# Use Python 3.9 slim as base image
FROM python:3.9-slim

# Install git and build dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set environment variable for NLTK data
ENV NLTK_DATA="/usr/local/nltk_data"

# Set working directory
WORKDIR /app

# Copy only requirements file first for better caching
COPY requirements.txt requirements.txt

# Upgrade pip and install dependencies with increased timeout
RUN pip install --upgrade pip \
    && pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Ensure NLTK stopwords are downloaded properly
RUN python -c "import nltk; nltk.data.path.append('/usr/local/nltk_data'); nltk.download('stopwords', download_dir='/usr/local/nltk_data')"

# Copy the rest of the application files
COPY . .

# Expose the port for FastAPI
EXPOSE 8001

# Command to run FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
