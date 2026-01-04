FROM python:3.14-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential=12.9 \
        curl=7.88.1-1+deb12u1 && \
    rm -rf /var/lib/apt/lists/*

# Copy dependencies and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training code
COPY train.py .

# Default command
CMD ["python", "train.py"]
