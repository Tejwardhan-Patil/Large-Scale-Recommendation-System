package monitoring;

import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;
import java.util.Map;
import java.util.HashMap;
import java.util.concurrent.atomic.AtomicBoolean;

public class PerformanceMonitor {

    private static final Logger logger = Logger.getLogger(PerformanceMonitor.class.getName());
    private ScheduledExecutorService scheduler;
    private long requestCount = 0;
    private long errorCount = 0;
    private long startTime;
    private Map<String, Long> latencyRecords;
    private AtomicBoolean isMonitoringActive;

    public PerformanceMonitor() {
        this.scheduler = Executors.newScheduledThreadPool(1);
        this.latencyRecords = new HashMap<>();
        this.startTime = System.currentTimeMillis();
        this.isMonitoringActive = new AtomicBoolean(false);
    }

    // Start monitoring periodically
    public void startMonitoring() {
        if (isMonitoringActive.get()) {
            logger.warning("Monitoring is already active.");
            return;
        }
        logger.info("Starting performance monitoring...");
        isMonitoringActive.set(true);
        scheduler.scheduleAtFixedRate(this::reportMetrics, 0, 1, TimeUnit.MINUTES);
    }

    // Stop the monitoring process
    public void stopMonitoring() {
        if (!isMonitoringActive.get()) {
            logger.warning("Monitoring is not active.");
            return;
        }
        logger.info("Stopping performance monitoring...");
        scheduler.shutdown();
        isMonitoringActive.set(false);
    }

    // Record the latency of an inference request
    public void recordLatency(String endpoint, long latency) {
        if (!isMonitoringActive.get()) {
            logger.warning("Cannot record latency. Monitoring is not active.");
            return;
        }
        latencyRecords.put(endpoint, latencyRecords.getOrDefault(endpoint, 0L) + latency);
        requestCount++;
        logger.info("Latency recorded for endpoint " + endpoint + ": " + latency + " ms");
    }

    // Record an error occurrence
    public void recordError() {
        if (!isMonitoringActive.get()) {
            logger.warning("Cannot record error. Monitoring is not active.");
            return;
        }
        errorCount++;
        logger.info("Error recorded. Total errors so far: " + errorCount);
    }

    // Report the performance metrics
    private void reportMetrics() {
        if (!isMonitoringActive.get()) {
            logger.warning("Cannot report metrics. Monitoring is not active.");
            return;
        }
        long uptime = (System.currentTimeMillis() - startTime) / 1000;
        double errorRate = requestCount > 0 ? (double) errorCount / requestCount * 100 : 0;
        
        logger.info("Performance Metrics Report:");
        logger.info("Uptime: " + uptime + " seconds");
        logger.info("Total Requests: " + requestCount);
        logger.info("Total Errors: " + errorCount);
        logger.info("Error Rate: " + String.format("%.2f", errorRate) + "%");

        if (requestCount == 0) {
            logger.warning("No requests have been processed yet.");
        } else {
            for (Map.Entry<String, Long> entry : latencyRecords.entrySet()) {
                long averageLatency = entry.getValue() / requestCount;
                logger.info("Average Latency for " + entry.getKey() + ": " + averageLatency + " ms");
            }
        }

        logAdvancedMetrics();
    }

    // Log additional advanced metrics for deeper insights
    private void logAdvancedMetrics() {
        logger.info("Advanced Metrics:");
        logger.info("Memory Usage: " + getMemoryUsage() + " MB");
        logger.info("CPU Load: " + getCPULoad() + "%");
        logger.info("Thread Count: " + getThreadCount());
    }

    // Get memory usage in megabytes
    private long getMemoryUsage() {
        Runtime runtime = Runtime.getRuntime();
        long memoryUsed = (runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024);
        return memoryUsed;
    }

    // Simulated CPU load
    private double getCPULoad() {
        return Math.random() * 100;
    }

    // Get the current thread count
    private int getThreadCount() {
        return Thread.activeCount();
    }

    // Reset the counters
    public void resetMetrics() {
        if (!isMonitoringActive.get()) {
            logger.warning("Cannot reset metrics. Monitoring is not active.");
            return;
        }
        logger.info("Resetting performance metrics...");
        requestCount = 0;
        errorCount = 0;
        latencyRecords.clear();
        logger.info("Performance metrics have been reset.");
    }

    // Record custom metrics, useful for tracking specific events
    public void recordCustomMetric(String metricName, long value) {
        if (!isMonitoringActive.get()) {
            logger.warning("Cannot record custom metric. Monitoring is not active.");
            return;
        }
        logger.info("Custom Metric [" + metricName + "] recorded with value: " + value);
        // Custom metric handling logic can be added here
    }

    // Generate a summary report at any point in time
    public void generateSummaryReport() {
        logger.info("Generating summary report...");
        reportMetrics();
    }

    // Simulate traffic and errors to test monitoring capabilities
    public void simulateTraffic() {
        logger.info("Simulating traffic for testing...");
        for (int i = 0; i < 100; i++) {
            recordLatency("/predict", (long) (Math.random() * 200));
            if (i % 10 == 0) {
                recordError();
            }
        }
    }

    // A utility method to print the status of monitoring
    public void printMonitoringStatus() {
        if (isMonitoringActive.get()) {
            logger.info("Monitoring is currently active.");
        } else {
            logger.info("Monitoring is currently inactive.");
        }
    }

    // Simulated traffic
    public void simulateTrafficForEndpoints() {
        logger.info("Simulating traffic for multiple endpoints...");
        for (int i = 0; i < 50; i++) {
            recordLatency("/predict", (long) (Math.random() * 200));
            recordLatency("/recommend", (long) (Math.random() * 150));
            if (i % 7 == 0) {
                recordError();
            }
        }
    }

    public static void main(String[] args) {
        PerformanceMonitor monitor = new PerformanceMonitor();
        monitor.startMonitoring();

        // Simulate some activity
        monitor.recordLatency("/predict", 120);
        monitor.recordLatency("/predict", 110);
        monitor.recordError();

        // Simulate traffic for testing
        monitor.simulateTraffic();
        monitor.simulateTrafficForEndpoints();

        // Generate a summary report
        monitor.generateSummaryReport();

        // Check the monitoring status
        monitor.printMonitoringStatus();

        // Stop monitoring after some time
        try {
            Thread.sleep(5000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        monitor.stopMonitoring();
    }
}