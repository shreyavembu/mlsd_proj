# 1. Use an official lightweight Python runtime as a parent image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install system dependencies required for clean runtime setups
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy your dependency tracker file and install libraries
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Copy your custom deep learning source code modules into the container
COPY src/ ./src/
COPY train.py .

# 6. Set environment variables to handle secure background configurations
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV KMP_DUPLICATE_LIB_OK=True

# 7. Set the default command to trigger your training pipeline when the container boots
CMD ["python3", "train.py"]