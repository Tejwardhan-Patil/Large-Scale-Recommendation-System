# Use an official Scala image as the base image
FROM openjdk:11

# Set working directory
WORKDIR /app

# Copy the Scala build files and install dependencies
COPY model-serving/build.sbt ./
RUN sbt update

# Copy the source code
COPY model-serving/src ./src

# Install necessary Python dependencies
RUN apt-get update && apt-get install -y python3 python3-pip
COPY model-serving/requirements.txt ./
RUN pip3 install -r requirements.txt

# Expose the port for the API
EXPOSE 8080

# Copy configurations
COPY model-serving/configs ./configs

# Copy the Kubernetes manifests if needed for deployment
COPY model-serving/Kubernetes ./Kubernetes

# Command to run the application
CMD ["sbt", "run"]