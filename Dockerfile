# Use an official Python runtime as a parent image
FROM --platform=linux/amd64 python:3.12

# Set the working directory in the container
WORKDIR /app

# Copy only requirements first to leverage Docker caching
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install Qt and build dependencies
RUN apt-get update && apt-get install -y \
    qt6-base-dev \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Now copy the rest of the application files
COPY . /app

# Ensure trained_model directory is available
RUN mkdir -p /app/trained_model && cp -r /app/synth/trained_model /app/

# Create directories for temporary file storage
RUN mkdir -p /tmp/uploads /tmp/synth_output /tmp/synth_normalization

# Create visualization directories
RUN mkdir -p /app/visualization_output \
    /app/visualization/Deeplayout/build \
    /app/visualization/Deeplayout/build/texture

# Build the C++ visualization tool
RUN cd /app/visualization/Deeplayout/build && \
    cmake .. && \
    make

# Copy texture files
RUN cp -r /app/visualization/Deeplayout/texture/*.jpg /app/visualization/Deeplayout/build/texture/

# Expose the port Flask runs on
EXPOSE 8080

# Set environment variables
ENV FLASK_APP=/app/synth/main.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=8080
ENV FLASK_ENV=production
# Add timeout configuration for gunicorn if used
ENV GUNICORN_TIMEOUT=900

# Set additional environment variables for visualization
ENV VISUALIZATION_FOLDER=/app/visualization_output
ENV DEEPLAYOUT_PATH=/app/visualization/Deeplayout/build/Deeplayout
ENV TEXTURE_SOURCE=/app/visualization/Deeplayout/texture
ENV TEXTURE_DEST=/app/visualization/Deeplayout/build/texture

# Ensure the entrypoint runs as expected
ENTRYPOINT ["python", "/app/synth/main.py"]