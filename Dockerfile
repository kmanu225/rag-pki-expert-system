# Use a minimal Python image
FROM python:3.13.2-slim

# Set the working directory
WORKDIR /usr/src/pki-expert

# Install basic dependencies (optional but useful for builds)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy requirements file first (for caching)
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose Gradio's default port
EXPOSE 7860

# Ensure Gradio is accessible from outside the container
ENV GRADIO_SERVER_NAME=0.0.0.0

# Run the app
CMD ["python", "src/main.py"]
