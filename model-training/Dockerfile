# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY src/ /app/src/
COPY configs/ /app/configs/

# Install additional dependencies for Scala and Java components
RUN apt-get update && \
    apt-get install -y openjdk-11-jdk scala && \
    rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME environment variable
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# Copy and install Maven and Scala configurations
COPY pom.xml /app/
COPY build.sbt /app/

# Build the Scala and Java components
RUN sbt compile && mvn install

# Set the entry point for the training process
CMD ["python", "src/pipelines/training_pipeline.py"]