# Base Image: Python 3.9 Slim (Lightweight)
FROM python:3.9-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies required for OpenCV and MediaPipe
# - LibGL/LibEGL for OpenCV image processing
# - Sound libraries for MediaPipe audio (even if not used)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to cache dependencies
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Download MediaPipe task file (if not present) during build
# This ensures the container has the model ready
RUN python -c "import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task', 'hand_landmarker.task')"

# Command to run the application
CMD ["python", "main.py"]
