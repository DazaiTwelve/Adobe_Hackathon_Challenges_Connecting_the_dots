# Filename: Dockerfile

# Use a specific, lightweight Python base image with the required platform
FROM --platform=linux/amd64 python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file
COPY ./requirements.txt .

# Install the Python dependency
RUN pip install --no-cache-dir -r requirements.txt

# Copy your Python script into the container
COPY ./run_round1a.py .

# Set the command to run when the container starts.
# This will automatically process PDFs from /app/input to /app/output
CMD ["python", "run_round1a.py"]
