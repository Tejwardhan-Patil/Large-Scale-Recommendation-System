# Use a base image with Python and necessary tools
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the application port
EXPOSE 8000

# Set environment variables
ENV DATA_SOURCE_URL=https://website.com/data
ENV OUTPUT_PATH=/data/output

# Define the entry point for the container
ENTRYPOINT ["python", "ingest_data.py"]