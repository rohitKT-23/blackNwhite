# Base image
FROM python:3.9-alpine


# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    cmake \
    build-essential \
    python3-dev \
    g++ \
    && rm -rf /var/lib/apt/lists/* /tmp/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port
EXPOSE 8003

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
