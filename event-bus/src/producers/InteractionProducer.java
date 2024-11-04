package producers;

import org.apache.kafka.clients.producer.*;
import org.apache.kafka.common.serialization.StringSerializer;
import java.util.Properties;
import java.util.concurrent.ExecutionException;
import java.time.LocalDateTime;
import org.json.JSONObject;
import java.util.Timer;
import java.util.TimerTask;

// Kafka producer for user interactions
public class InteractionProducer {

    private final KafkaProducer<String, String> producer;
    private final String topic;
    private final Timer timer;

    // Constructor to initialize Kafka producer
    public InteractionProducer(String brokers, String topic) {
        this.topic = topic;
        this.producer = createKafkaProducer(brokers);
        this.timer = new Timer(true);
        startPeriodicHealthCheck();
    }

    // Method to create and configure Kafka producer
    private KafkaProducer<String, String> createKafkaProducer(String brokers) {
        Properties props = new Properties();
        props.put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, brokers);
        props.put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG, StringSerializer.class.getName());
        props.put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG, StringSerializer.class.getName());
        props.put(ProducerConfig.ACKS_CONFIG, "all"); // Ensure message delivery
        props.put(ProducerConfig.RETRIES_CONFIG, 3); // Retry in case of failure
        props.put(ProducerConfig.LINGER_MS_CONFIG, 1); // Minimize latency
        props.put(ProducerConfig.ENABLE_IDEMPOTENCE_CONFIG, "true"); // Ensure exactly-once delivery
        props.put(ProducerConfig.BUFFER_MEMORY_CONFIG, 33554432); // 32MB buffer size
        props.put(ProducerConfig.COMPRESSION_TYPE_CONFIG, "gzip"); // Compress messages to save bandwidth

        return new KafkaProducer<>(props);
    }

    // Method to produce user interaction events
    public void produceInteractionEvent(String userId, String interactionType, String itemId) {
        // Create the event payload as JSON
        JSONObject interactionEvent = new JSONObject();
        interactionEvent.put("user_id", userId);
        interactionEvent.put("interaction_type", interactionType);
        interactionEvent.put("item_id", itemId);
        interactionEvent.put("timestamp", LocalDateTime.now().toString());

        // Send the event to Kafka
        ProducerRecord<String, String> record = new ProducerRecord<>(topic, userId, interactionEvent.toString());

        // Synchronously send the message to Kafka and handle potential exceptions
        try {
            producer.send(record).get(); // Wait for the send to complete
            logEvent("Sent event: " + interactionEvent.toString());
        } catch (InterruptedException | ExecutionException e) {
            logEvent("Failed to send event: " + e.getMessage(), "ERROR");
        }
    }

    // Periodic health check task for the producer
    private void startPeriodicHealthCheck() {
        timer.scheduleAtFixedRate(new TimerTask() {
            @Override
            public void run() {
                logEvent("Performing periodic health check for Kafka producer");
                try {
                    // Check producer's status (record send to validate connection)
                    producer.send(new ProducerRecord<>(topic, "health-check", "ping")).get();
                    logEvent("Kafka producer health check passed.");
                } catch (Exception e) {
                    logEvent("Kafka producer health check failed: " + e.getMessage(), "ERROR");
                }
            }
        }, 0, 60000); // Every 60 seconds
    }

    // Close the producer when done, also stopping the health check timer
    public void close() {
        logEvent("Closing Kafka producer...");
        timer.cancel();
        producer.close();
        logEvent("Kafka producer closed successfully.");
    }

    // Helper method to log events
    private void logEvent(String message) {
        logEvent(message, "INFO");
    }

    // Overloaded helper method to log events with severity
    private void logEvent(String message, String severity) {
        System.out.println("[" + severity + "] [" + LocalDateTime.now() + "] - " + message);
    }

    // Method to retry sending events in case of failure
    public void produceWithRetry(String userId, String interactionType, String itemId, int maxRetries) {
        int attempt = 0;
        boolean success = false;

        while (attempt < maxRetries && !success) {
            try {
                produceInteractionEvent(userId, interactionType, itemId);
                success = true;
            } catch (Exception e) {
                attempt++;
                logEvent("Retry " + attempt + " for sending event failed: " + e.getMessage(), "WARN");
                if (attempt == maxRetries) {
                    logEvent("Max retries reached. Event sending failed permanently.", "ERROR");
                }
            }
        }
    }

    // Main method for testing the producer
    public static void main(String[] args) {
        String brokers = "localhost:9092";
        String topic = "user-interactions";

        // Initialize the interaction producer
        InteractionProducer interactionProducer = new InteractionProducer(brokers, topic);

        // Produce some interaction events with retry logic
        interactionProducer.produceWithRetry("user123", "click", "item456", 3);
        interactionProducer.produceWithRetry("user789", "view", "item123", 3);
        interactionProducer.produceWithRetry("user456", "like", "item789", 3);

        // Close the producer
        interactionProducer.close();
    }
}