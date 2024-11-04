# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Expose the port the app runs on
EXPOSE 8080

# Define environment variables
ENV RECOMMENDER_CONFIG_PATH=/app/configs/algorithm_config.yaml

# Run the recommendation engine script when the container launches
CMD ["python", "./src/algorithms/cf_algorithm.py"]