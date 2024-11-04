# Use a base image with Kafka and Zookeeper pre-installed
FROM wurstmeister/kafka:latest

# Set environment variables for Kafka and Zookeeper
ENV KAFKA_BROKER_ID=1
ENV KAFKA_LOG_DIRS=/kafka/kafka-logs
ENV KAFKA_ZOOKEEPER_CONNECT=zookeeper:2181
ENV KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092
ENV KAFKA_LISTENERS=PLAINTEXT://0.0.0.0:9092

# Expose Kafka and Zookeeper ports
EXPOSE 9092
EXPOSE 2181

# Create directories for Kafka logs and configuration
RUN mkdir -p /kafka/kafka-logs /kafka/config /kafka/logs

# Copy custom Kafka and Zookeeper configurations into the container
COPY configs/event_bus_config.yaml /kafka/config/event_bus_config.yaml
COPY configs/server.properties /kafka/config/server.properties
COPY configs/zookeeper.properties /kafka/config/zookeeper.properties
COPY event-schema/event_schema.json /kafka/config/event_schema.json

# Install dependencies for monitoring, logging, or additional services
RUN apt-get update && apt-get install -y \
    curl \
    jq \
    net-tools \
    dnsutils \
    && apt-get clean

# Start Zookeeper and Kafka services when the container starts
CMD ["/bin/bash", "-c", "start-kafka.sh"]

# Healthcheck to ensure Kafka is running
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:9092 || exit 1