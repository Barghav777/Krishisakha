# Dockerfile

# 1. Use an official Python runtime as a parent image
FROM python:3.9-slim

# 2. Set the working directory in the container
WORKDIR /app

# 3. Install system dependencies required for your project
# - ffmpeg is for pydub (audio processing)
# - build-essential is for compiling some Python packages
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy the requirements file and install Python packages
# This step is done separately to leverage Docker's layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your application's source code, models, and data into the container
COPY . .

# 6. Expose the port the app runs on
EXPOSE 5000

# 7. Define the command to run your application using a production server (Gunicorn)
#    - We use a single worker and increase the timeout to handle long-running model processing.
#    - 0.0.0.0 makes the server accessible from outside the container.
CMD ["gunicorn", "--workers", "1", "--timeout", "300", "--bind", "0.0.0.0:5000", "app:app"]