package ensembles;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;

import com.google.gson.JsonObject;


public class Ensemble {

    // URLs for different recommendation algorithms
    private static final String CF_URL = "http://localhost:5001/cf_recommend";
    private static final String MF_URL = "http://localhost:5002/mf_recommend";
    private static final String DL_URL = "http://localhost:5003/dl_recommend";

    // Logger for logging purposes
    private static final Logger logger = Logger.getLogger(ensemble.class.getName());

    /**
     * Main method to test the ensemble system.
     */
    public static void main(String[] args) {
        try {
            Ensemble ensemble = new Ensemble();
            String userId = "123";
            List<String> recommendations = ensemble.getRecommendations(userId);
            
            // Print out the final recommendations
            recommendations.forEach(System.out::println);
        } catch (IOException e) {
            logger.log(Level.SEVERE, "Error fetching recommendations", e);
        }
    }

    /**
     * Method to get recommendations from all available algorithms
     *
     * @param userId The user ID for which recommendations are being fetched
     * @return A list of combined recommendations from all algorithms
     * @throws IOException If there is a problem with the HTTP request
     */
    public List<String> getRecommendations(String userId) throws IOException {
        logger.info("Fetching recommendations for user: " + userId);

        // Fetch recommendations from each microservice
        String cfRecommendations = fetchRecommendations(CF_URL, userId);
        String mfRecommendations = fetchRecommendations(MF_URL, userId);
        String dlRecommendations = fetchRecommendations(DL_URL, userId);

        logger.info("Collaborative Filtering Recommendations: " + cfRecommendations);
        logger.info("Matrix Factorization Recommendations: " + mfRecommendations);
        logger.info("Deep Learning Recommendations: " + dlRecommendations);

        // Combine the recommendations from all algorithms
        List<String> combinedRecommendations = combineRecommendations(cfRecommendations, mfRecommendations, dlRecommendations);
        logger.info("Final combined recommendations: " + combinedRecommendations);

        return combinedRecommendations;
    }

    /**
     * Method to send an HTTP POST request to a microservice and retrieve recommendations
     *
     * @param url    The URL of the microservice
     * @param userId The user ID for which the recommendations are requested
     * @return The recommendations as a string from the service
     * @throws IOException If an issue occurs during the HTTP request
     */
    private String fetchRecommendations(String url, String userId) throws IOException {
        logger.info("Sending request to " + url + " for user: " + userId);

        // Create the HTTP client for executing the request
        try (CloseableHttpClient client = HttpClients.createDefault()) {
            // Prepare the POST request with user_id in JSON format
            HttpPost request = new HttpPost(url);
            JsonObject json = new JsonObject();
            json.addProperty("user_id", userId);

            StringEntity entity = new StringEntity(json.toString());
            request.setEntity(entity);
            request.setHeader("Content-Type", "application/json");

            // Execute the request and retrieve the response
            try (CloseableHttpResponse response = client.execute(request)) {
                String result = EntityUtils.toString(response.getEntity());
                logger.info("Response from " + url + ": " + result);
                return result;
            } catch (IOException e) {
                logger.log(Level.SEVERE, "Error communicating with " + url, e);
                throw e;
            }
        }
    }

    /**
     * Method to combine the recommendations from different algorithms
     *
     * @param cfRecommendations The result from collaborative filtering
     * @param mfRecommendations The result from matrix factorization
     * @param dlRecommendations The result from deep learning model
     * @return A combined list of recommendations
     */
    private List<String> combineRecommendations(String cf, String mf, String dl) {
        logger.info("Combining recommendations from all algorithms");

        List<String> combined = new ArrayList<>();
        
        // Add collaborative filtering recommendations
        combined.add("Collaborative Filtering Recommendations: " + cf);

        // Add matrix factorization recommendations
        combined.add("Matrix Factorization Recommendations: " + mf);

        // Add deep learning recommendations
        combined.add("Deep Learning Recommendations: " + dl);

        logger.info("Combined recommendations: " + combined);
        return combined;
    }

    /**
     * A utility method for logging and handling errors uniformly
     *
     * @param message The message to log
     * @param exception The exception to log if available
     */
    private void logError(String message, Exception exception) {
        logger.log(Level.SEVERE, message, exception);
    }

    /**
     * Adds custom headers to the HTTP request
     *
     * @param request The HTTP POST request object
     */
    private void addCustomHeaders(HttpPost request) {
        request.setHeader("Authorization", "api-key");
        logger.info("Custom headers added to the request");
    }
    
    /**
     * Handles scenarios where a microservice is unavailable
     *
     * @param serviceName The name of the microservice
     */
    private void handleServiceUnavailable(String serviceName) {
        logger.warning("Service unavailable: " + serviceName);
        // Handle fallback logic here
    }
    
    /**
     * Method to fetch recommendations with retry logic in case of temporary failures.
     *
     * @param url The URL of the microservice
     * @param userId The user ID for which recommendations are requested
     * @param retries Number of retries in case of failure
     * @return The recommendations from the service or a fallback in case of failure
     */
    private String fetchWithRetry(String url, String userId, int retries) {
        int attempt = 0;
        while (attempt <= retries) {
            try {
                return fetchRecommendations(url, userId);  // Call the original fetch method
            } catch (IOException e) {
                attempt++;
                logger.warning("Attempt " + attempt + " failed for " + url + ". Retrying...");
                if (attempt > retries) {
                    handleServiceUnavailable(url);  // Handle service unavailable scenario after retries
                    return getFallbackRecommendations(url);  // Provide fallback recommendations if retries exhausted
                }
                try {
                    Thread.sleep(1000 * attempt);  // Exponential backoff between retries
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                }
            }
        }
        return getFallbackRecommendations(url);
    }

    /**
     * Provides fallback recommendations in case a service is unavailable.
     *
     * @param url The URL of the service that failed
     * @return A fallback recommendation message
     */
    private String getFallbackRecommendations(String url) {
        logger.warning("Providing fallback recommendations for service: " + url);
        return "Service " + url + " unavailable. Showing default recommendations.";
    }

    /**
     * Handles combining recommendations with a priority system.
     * Gives higher weight to certain algorithms if needed.
     *
     * @param cfRecommendations The result from collaborative filtering
     * @param mfRecommendations The result from matrix factorization
     * @param dlRecommendations The result from deep learning model
     * @return A prioritized list of combined recommendations
     */
    private List<String> combineRecommendationsWithPriority(String cf, String mf, String dl) {
        logger.info("Combining recommendations using a priority system");

        List<String> prioritized = new ArrayList<>();
        
        // Prioritize collaborative filtering over others
        if (!cf.isEmpty()) {
            prioritized.add("Priority: Collaborative Filtering Recommendations: " + cf);
        }

        if (!mf.isEmpty()) {
            prioritized.add("Secondary: Matrix Factorization Recommendations: " + mf);
        }

        if (!dl.isEmpty()) {
            prioritized.add("Tertiary: Deep Learning Recommendations: " + dl);
        }

        return prioritized;
    }

    /**
     * Handles scenarios where a service returns invalid or empty data.
     *
     * @param recommendations The recommendations received from a service
     * @param serviceName The name of the service that provided the data
     * @return A boolean indicating whether the data is valid
     */
    private boolean validateRecommendations(String recommendations, String serviceName) {
        if (recommendations == null || recommendations.trim().isEmpty()) {
            logger.warning("Invalid or empty recommendations received from: " + serviceName);
            return false;
        }
        logger.info("Valid recommendations received from: " + serviceName);
        return true;
    }

    /**
     * Combines recommendations with error handling for missing data.
     *
     * @param cfRecommendations The result from collaborative filtering
     * @param mfRecommendations The result from matrix factorization
     * @param dlRecommendations The result from deep learning model
     * @return A combined list of recommendations with error checks
     */
    private List<String> safeCombineRecommendations(String cf, String mf, String dl) {
        logger.info("Safely combining recommendations with validation");

        List<String> combined = new ArrayList<>();

        // Check if each service returned valid data and handle accordingly
        if (validateRecommendations(cf, "Collaborative Filtering")) {
            combined.add("Collaborative Filtering Recommendations: " + cf);
        } else {
            combined.add("Collaborative Filtering service failed. Using fallback.");
        }

        if (validateRecommendations(mf, "Matrix Factorization")) {
            combined.add("Matrix Factorization Recommendations: " + mf);
        } else {
            combined.add("Matrix Factorization service failed. Using fallback.");
        }

        if (validateRecommendations(dl, "Deep Learning")) {
            combined.add("Deep Learning Recommendations: " + dl);
        } else {
            combined.add("Deep Learning service failed. Using fallback.");
        }

        return combined;
    }

    // Main method for testing the retry mechanism and error handling.
    /* 
    public static void main(String[] args) {
        try {
            ensemble ensemble = new ensemble();
            String userId = "123";

            // Fetch recommendations with retry logic
            String cfRecommendations = ensemble.fetchWithRetry(CF_URL, userId, 3);
            String mfRecommendations = ensemble.fetchWithRetry(MF_URL, userId, 3);
            String dlRecommendations = ensemble.fetchWithRetry(DL_URL, userId, 3);

            // Combine the recommendations with error handling
            List<String> recommendations = ensemble.safeCombineRecommendations(cfRecommendations, mfRecommendations, dlRecommendations);
            recommendations.forEach(System.out::println);
        } catch (Exception e) {
            logger.log(Level.SEVERE, "An error occurred while fetching recommendations", e);
        }
    }*/

    /**
     * Simulates the logging of additional metadata along with recommendations.
     * This can help track and monitor the performance of each recommendation.
     *
     * @param recommendations The combined list of recommendations
     * @param metadata Additional metadata such as timestamp or request ID
     */
    private void logRecommendationsWithMetadata(List<String> recommendations, String metadata) {
        logger.info("Logging recommendations with metadata: " + metadata);
        recommendations.forEach(recommendation -> logger.info(metadata + ": " + recommendation));
    }

    /**
     * Handles monitoring the service's response time and performance.
     * This can be extended to track average response times over a period.
     */
    private void monitorServicePerformance(String serviceName, long startTime, long endTime) {
        long duration = endTime - startTime;
        logger.info("Service " + serviceName + " took " + duration + " ms to respond.");
        // Logic to track performance metrics can be added here
    }
    
    /**
     * Method to handle caching recommendations for faster retrieval.
     *
     * @param userId The user ID for whom recommendations were fetched
     * @param recommendations The recommendations to be cached
     */
    private void cacheRecommendations(String userId, List<String> recommendations) {
        // Logging cache behavior
        logger.info("Caching recommendations for user: " + userId);
        // Implement caching logic
    }

    /**
     * Method to retrieve cached recommendations if available.
     *
     * @param userId The user ID for whom recommendations are requested
     * @return Cached recommendations if available, null otherwise
     */
    private List<String> getCachedRecommendations(String userId) {
        // Simulate cache retrieval
        logger.info("Checking cache for recommendations for user: " + userId);
        // This would retrieve data from a cache
        return null;  // Return null to indicate no cache hit
    }

        /**
     * Tracks the success and failure rates of services to monitor their reliability over time.
     *
     * @param serviceName The name of the service being tracked
     * @param success Whether the service call was successful
     */
    private void trackServiceReliability(String serviceName, boolean success) {
        if (success) {
            logger.info("Service " + serviceName + " call was successful.");
            // Implement logic to track success (increment success counter)
        } else {
            logger.warning("Service " + serviceName + " call failed.");
            // Implement logic to track failure (increment failure counter)
        }
    }

    /**
     * Advanced method to evaluate and rank combined recommendations based on algorithm confidence.
     * Each algorithm returns confidence levels with its recommendations.
     *
     * @param cfRecommendations The result from collaborative filtering
     * @param mfRecommendations The result from matrix factorization
     * @param dlRecommendations The result from deep learning
     * @return Ranked list of combined recommendations based on algorithm confidence
     */
    private List<String> rankRecommendationsByConfidence(Recommendation cfRecommendations, Recommendation mfRecommendations, Recommendation dlRecommendations) {
        logger.info("Ranking recommendations based on confidence levels.");

        // Extract confidence levels
        int cfConfidence = cfRecommendations.getConfidence();
        int mfConfidence = mfRecommendations.getConfidence();
        int dlConfidence = dlRecommendations.getConfidence();

        // Use a list to hold all recommendations
        List<Recommendation> allRecommendations = Arrays.asList(cfRecommendations, mfRecommendations, dlRecommendations);

        // Sort recommendations by confidence level in descending order
        allRecommendations.sort((r1, r2) -> Integer.compare(r2.getConfidence(), r1.getConfidence()));

        // Prepare the ranked list output
        List<String> rankedRecommendations = new ArrayList<>();
        for (Recommendation rec : allRecommendations) {
            rankedRecommendations.add(rec.getType() + " Recommendations (Confidence: " + rec.getConfidence() + "%): " + rec.getDetails());
        }

        return rankedRecommendations;
    }

    /**
     * Logs the overall system health and uptime statistics.
     */
    private void logSystemHealth() {
        // Logic for system health check
        boolean allServicesUp = checkAllServicesHealth(); 

        if (allServicesUp) {
            logger.info("All services are operational.");
        } else {
            logger.warning("One or more services are down.");
        }

        // Monitor uptime statistics
        long uptime = System.currentTimeMillis() / 1000;  // Uptime in seconds
        logger.info("System uptime: " + uptime + " seconds.");
    }

    /**
     * Enhanced version of the method to handle service outages gracefully.
     * Tries to balance load among working services.
     *
     * @param userId The user ID for which recommendations are requested
     * @return Combined recommendations from the working services
     */
    public List<String> getRecommendationsWithOutageHandling(String userId) {
        logger.info("Fetching recommendations with service outage handling.");

        // Fetch recommendations from each service, using fallback
        String cfRecommendations = fetchWithRetry(CF_URL, userId, 2);
        String mfRecommendations = fetchWithRetry(MF_URL, userId, 2);
        String dlRecommendations = fetchWithRetry(DL_URL, userId, 2);

        // Rank and combine recommendations based on the confidence and availability of the services
        List<String> combinedRecommendations = rankRecommendationsByConfidence(cfRecommendations, mfRecommendations, dlRecommendations);
        
        logRecommendationsWithMetadata(combinedRecommendations, "Outage Handling Mode");

        return combinedRecommendations;
    }

    /**
     * Simulates alerting in case of service degradation or failure.
     *
     * @param serviceName The name of the service that is experiencing issues
     */
    private void alertServiceDegradation(String serviceName) {
        logger.severe("Service degradation detected: " + serviceName);
        // Simulated logic to trigger alerts (send an email or trigger an alerting system)
        try {
            boolean alertSent = sendAlert("Service degradation detected in: " + serviceName);
            if (alertSent) {
                logger.info("Alert successfully sent for service: " + serviceName);
            } else {
                logger.warning("Failed to send alert for service: " + serviceName);
            }
        } catch (Exception e) {
            logger.severe("Exception while sending alert: " + e.getMessage());
        }
    }

    /**
     * Handles predictive scaling of services based on current and predicted load.
     * This method implements a simple scaling mechanism.
     */
    private void handlePredictiveScaling() {
        logger.info("Predictive scaling is being handled based on current load.");

        double currentLoad = getCurrentLoad(); // Fetch load metrics
        double threshold = 0.75;

        if (currentLoad > threshold) {
            logger.info("Current load (" + currentLoad + ") exceeds threshold (" + threshold + "). Scaling up the services to handle increased load.");
            // Code to initiate scaling, like interacting with a cloud provider's API
            boolean scalingSuccess = scaleUpServices();
            if (scalingSuccess) {
                logger.info("Successfully scaled up the services.");
            } else {
                logger.warning("Failed to scale up the services.");
            }
        } else {
            logger.info("Current load (" + currentLoad + ") is below threshold (" + threshold + "). No scaling necessary at this time.");
        }
    }

    /**
     * Simulates sending an alert to an alerting system.
     *
     * @param message The alert message to be sent.
     * @return True if the alert was successfully sent, otherwise false.
     */
    private boolean sendAlert(String message) {
        // Logic to simulate sending an alert (email, SMS, or external alerting system)
        logger.info("Sending alert: " + message);
        // Simulate the success or failure of the alert being sent
        return new Random().nextBoolean();
    }

    /**
     * Simulates fetching the current load metric for the service.
     *
     * @return A double value representing the current load (0.0 to 1.0).
     */
    private double getCurrentLoad() {
        // Logic to fetch current load metric, such as CPU or memory
        double load = new Random().nextDouble();
        logger.info("Current load fetched: " + load);
        return load;
    }

    /**
     * Simulates scaling up the services by interacting with an infrastructure API.
     *
     * @return True if scaling was successful, otherwise false.
     */
    private boolean scaleUpServices() {
        // Logic to simulate interaction with an infrastructure API to scale up services
        logger.info("Interacting with infrastructure to scale up the service.");
        // Simulate the success or failure of scaling operation
        return new Random().nextBoolean();
    }

    /**
     * Performs load balancing between services to distribute requests evenly.
     */
    private void performLoadBalancing() {
        logger.info("Performing load balancing between services.");
        // Simulate load balancing logic
        boolean balanceSuccess = true;

        if (balanceSuccess) {
            logger.info("Load balancing was successful.");
        } else {
            logger.warning("Load balancing encountered an issue.");
        }
    }

    /**
     * Handles model versioning, ensuring that the appropriate model version is used when combining results.
     *
     * @param modelVersion The version of the model to use for combining results
     * @return Combined recommendations using the specified model version
     */
    public List<String> getRecommendationsWithVersioning(String userId, String modelVersion) {
        logger.info("Fetching recommendations using model version: " + modelVersion);

        // Fetch recommendations from each service, using model version where applicable
        String cfRecommendations = fetchWithRetry(CF_URL + "?version=" + modelVersion, userId, 3);
        String mfRecommendations = fetchWithRetry(MF_URL + "?version=" + modelVersion, userId, 3);
        String dlRecommendations = fetchWithRetry(DL_URL + "?version=" + modelVersion, userId, 3);

        // Combine recommendations as usual
        List<String> combinedRecommendations = safeCombineRecommendations(cfRecommendations, mfRecommendations, dlRecommendations);

        // Log recommendations with the model version metadata
        logRecommendationsWithMetadata(combinedRecommendations, "Model Version: " + modelVersion);

        return combinedRecommendations;
    }

    /**
     * Handles model rollback in case a newer model version shows degraded performance.
     * This method simulates rolling back to a previous model version.
     *
     * @param userId The user ID for which recommendations are requested
     * @param rollbackToVersion The version of the model to roll back to
     * @return Combined recommendations using the rolled-back model version
     */
    public List<String> handleModelRollback(String userId, String rollbackToVersion) {
        logger.warning("Rolling back to model version: " + rollbackToVersion);

        // Fetch recommendations using the rolled-back model version
        return getRecommendationsWithVersioning(userId, rollbackToVersion);
    }

    /**
     * Simulates A/B testing between two model versions, evaluating which performs better.
     * This method compares the results of both versions and returns the winner.
     *
     * @param userId The user ID for which recommendations are requested
     * @param versionA Model version A
     * @param versionB Model version B
     * @return The combined recommendations from the better-performing model
     */
    public List<String> performABTesting(String userId, String versionA, String versionB) {
        logger.info("Performing A/B testing between model versions: " + versionA + " and " + versionB);

        // Fetch recommendations for both versions
        List<String> recommendationsA = getRecommendationsWithVersioning(userId, versionA);
        List<String> recommendationsB = getRecommendationsWithVersioning(userId, versionB);

        boolean versionAWins = true;  // Simulated A/B test result

        if (versionAWins) {
            logger.info("Model version " + versionA + " outperforms version " + versionB);
            return recommendationsA;
        } else {
            logger.info("Model version " + versionB + " outperforms version " + versionA);
            return recommendationsB;
        }
    }
}