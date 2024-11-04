package transformers;

import java.util.*;
import java.util.stream.Collectors;

public class FeatureEngineering {

    // Perform transformations
    public static Map<String, Object> transformFeatures(Map<String, Object> rawFeatures) {
        Map<String, Object> transformedFeatures = new HashMap<>();

        // Scaling numerical features
        if (rawFeatures.containsKey("age")) {
            transformedFeatures.put("age_scaled", scaleFeature((Integer) rawFeatures.get("age")));
        }

        // One-hot encoding categorical features
        if (rawFeatures.containsKey("gender")) {
            transformedFeatures.putAll(oneHotEncode("gender", (String) rawFeatures.get("gender")));
        }

        // Additional feature engineering steps
        if (rawFeatures.containsKey("interaction_count")) {
            transformedFeatures.put("interaction_count_log", logTransform((Integer) rawFeatures.get("interaction_count")));
        }

        return transformedFeatures;
    }

    // Scales a numerical feature between 0 and 1
    private static double scaleFeature(int value) {
        int min = 0;
        int max = 100;
        return (double) (value - min) / (max - min);
    }

    // One-hot encode a categorical feature
    private static Map<String, Integer> oneHotEncode(String featureName, String value) {
        Map<String, Integer> encoded = new HashMap<>();
        List<String> categories = Arrays.asList("male", "female", "other");
        
        for (String category : categories) {
            String encodedKey = featureName + "_" + category;
            encoded.put(encodedKey, category.equals(value) ? 1 : 0);
        }

        return encoded;
    }

    // Log transformation for skewed features
    private static double logTransform(int value) {
        return Math.log(value + 1); // Adding 1 to avoid log(0)
    }

    // Handling missing values
    public static Map<String, Object> handleMissingValues(Map<String, Object> rawFeatures) {
        Map<String, Object> filledFeatures = new HashMap<>(rawFeatures);

        if (!rawFeatures.containsKey("age")) {
            filledFeatures.put("age", 0); // Default value for missing age
        }

        return filledFeatures;
    }

    // Feature extraction for interaction features
    public static List<Double> extractInteractionFeatures(Map<String, Object> rawFeatures) {
        List<Double> interactionFeatures = new ArrayList<>();

        if (rawFeatures.containsKey("clicks") && rawFeatures.containsKey("views")) {
            int clicks = (Integer) rawFeatures.get("clicks");
            int views = (Integer) rawFeatures.get("views");
            interactionFeatures.add(calculateCTR(clicks, views));
        }

        return interactionFeatures;
    }

    // Calculate Click-Through Rate (CTR)
    private static double calculateCTR(int clicks, int views) {
        if (views == 0) return 0;
        return (double) clicks / views;
    }

    // Normalizing the interaction features based on min-max normalization
    public static List<Double> normalizeInteractionFeatures(List<Double> interactionFeatures) {
        double min = Collections.min(interactionFeatures);
        double max = Collections.max(interactionFeatures);
        
        List<Double> normalized = new ArrayList<>();
        for (double feature : interactionFeatures) {
            normalized.add((feature - min) / (max - min));
        }
        return normalized;
    }

    // Extract and transform textual features (like user reviews or descriptions)
    public static Map<String, Integer> extractTextFeatures(String text) {
        Map<String, Integer> textFeatures = new HashMap<>();
        
        // Count number of words in the text
        int wordCount = text.split("\\s+").length;
        textFeatures.put("word_count", wordCount);

        // Calculate length of the text
        textFeatures.put("text_length", text.length());

        return textFeatures;
    }

    // Tokenization of text for NLP-related tasks
    public static List<String> tokenizeText(String text) {
        // Basic tokenization using space as delimiter
        return Arrays.asList(text.split("\\s+"));
    }

    // Create N-grams from a tokenized text
    public static List<String> generateNgrams(List<String> tokens, int n) {
        List<String> ngrams = new ArrayList<>();
        for (int i = 0; i < tokens.size() - n + 1; i++) {
            ngrams.add(String.join(" ", tokens.subList(i, i + n)));
        }
        return ngrams;
    }

    // Part of speech tagging
    public static Map<String, String> posTagging(List<String> tokens) {
        Map<String, String> posTags = new HashMap<>();
        
        // Simulating POS tagging
        for (String token : tokens) {
            posTags.put(token, "NOUN"); // All tokens are labeled as NOUN
        }
        
        return posTags;
    }

    // Extract numeric features from textual data
    public static Map<String, Double> extractNumericTextFeatures(String text) {
        Map<String, Double> numericTextFeatures = new HashMap<>();

        // Calculate the ratio of alphabetic characters to total characters
        long alphaCount = text.chars().filter(Character::isLetter).count();
        numericTextFeatures.put("alpha_ratio", (double) alphaCount / text.length());

        // Calculate the ratio of digits to total characters
        long digitCount = text.chars().filter(Character::isDigit).count();
        numericTextFeatures.put("digit_ratio", (double) digitCount / text.length());

        return numericTextFeatures;
    }

    // Binarize categorical features (transform into 0/1)
    public static Map<String, Integer> binarizeCategoricalFeature(String featureName, String value, List<String> categories) {
        Map<String, Integer> binaryEncoded = new HashMap<>();

        for (String category : categories) {
            binaryEncoded.put(featureName + "_" + category, category.equals(value) ? 1 : 0);
        }

        return binaryEncoded;
    }

    // Feature for binary classification
    public static int binaryClassificationFeature(String featureValue) {
        // Simulate a binary classification
        return featureValue.equalsIgnoreCase("positive") ? 1 : 0;
    }

    // Creating a polynomial feature from two existing features
    public static double createPolynomialFeature(double feature1, double feature2, int degree) {
        return Math.pow(feature1 * feature2, degree);
    }

    // Cross features between two categorical features
    public static String createCrossFeature(String feature1, String feature2) {
        return feature1 + "_" + feature2;
    }

    // Generate interaction features from multiple raw features
    public static List<Double> generateInteractionFeatures(Map<String, Object> rawFeatures, List<String> featureNames) {
        List<Double> interactionFeatures = new ArrayList<>();

        // Multiply values of all numerical features to create interaction terms
        double interactionTerm = 1.0;
        for (String featureName : featureNames) {
            if (rawFeatures.containsKey(featureName)) {
                interactionTerm *= (Double) rawFeatures.get(featureName);
            }
        }
        interactionFeatures.add(interactionTerm);

        return interactionFeatures;
    }

    // Error handling for missing features
    public static void validateFeature(Map<String, Object> rawFeatures, String featureName) throws FeatureNotFoundException {
        if (!rawFeatures.containsKey(featureName)) {
            throw new FeatureNotFoundException("Feature: " + featureName + " is missing.");
        }
    }

    // Exception for feature validation
    public static class FeatureNotFoundException extends Exception {
        public FeatureNotFoundException(String errorMessage) {
            super(errorMessage);
        }
    }

    public static void main(String[] args) {
        // Usage
        Map<String, Object> rawFeatures = new HashMap<>();
        rawFeatures.put("age", 25);
        rawFeatures.put("gender", "female");
        rawFeatures.put("interaction_count", 150);
        rawFeatures.put("clicks", 10);
        rawFeatures.put("views", 100);

        Map<String, Object> transformed = transformFeatures(rawFeatures);
        System.out.println("Transformed Features: " + transformed);

        List<Double> interactionFeatures = extractInteractionFeatures(rawFeatures);
        System.out.println("Interaction Features: " + interactionFeatures);

        List<String> tokens = tokenizeText("The quick brown fox jumps over the lazy dog.");
        System.out.println("Tokens: " + tokens);

        List<String> bigrams = generateNgrams(tokens, 2);
        System.out.println("Bigrams: " + bigrams);
    }

    // Generate polynomial features for a list of numerical features
    public static Map<String, Double> generatePolynomialFeatures(Map<String, Double> numericalFeatures, int degree) {
        Map<String, Double> polynomialFeatures = new HashMap<>();

        for (Map.Entry<String, Double> entry : numericalFeatures.entrySet()) {
            String featureName = entry.getKey();
            Double featureValue = entry.getValue();

            // Generate polynomial feature for the given degree
            for (int d = 1; d <= degree; d++) {
                polynomialFeatures.put(featureName + "_poly_" + d, Math.pow(featureValue, d));
            }
        }

        return polynomialFeatures;
    }

    // Normalization of numerical features using Z-score standardization
    public static Map<String, Double> zScoreNormalization(Map<String, Double> numericalFeatures) {
        Map<String, Double> normalizedFeatures = new HashMap<>();
        double mean = numericalFeatures.values().stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        double stdDev = Math.sqrt(numericalFeatures.values().stream().mapToDouble(val -> Math.pow(val - mean, 2)).average().orElse(0.0));

        for (Map.Entry<String, Double> entry : numericalFeatures.entrySet()) {
            String featureName = entry.getKey();
            Double featureValue = entry.getValue();
            normalizedFeatures.put(featureName, (featureValue - mean) / stdDev);
        }

        return normalizedFeatures;
    }

    // Binarize numerical features using a threshold value
    public static Map<String, Integer> binarizeNumericalFeatures(Map<String, Double> numericalFeatures, double threshold) {
        Map<String, Integer> binaryFeatures = new HashMap<>();

        for (Map.Entry<String, Double> entry : numericalFeatures.entrySet()) {
            String featureName = entry.getKey();
            Double featureValue = entry.getValue();
            binaryFeatures.put(featureName, featureValue > threshold ? 1 : 0);
        }

        return binaryFeatures;
    }

    // Feature selection by variance threshold
    public static Map<String, Double> selectFeaturesByVariance(Map<String, Double> numericalFeatures, double varianceThreshold) {
        Map<String, Double> selectedFeatures = new HashMap<>();
        double mean = numericalFeatures.values().stream().mapToDouble(Double::doubleValue).average().orElse(0.0);

        for (Map.Entry<String, Double> entry : numericalFeatures.entrySet()) {
            String featureName = entry.getKey();
            Double featureValue = entry.getValue();
            double variance = Math.pow(featureValue - mean, 2);

            // Select features with variance above the threshold
            if (variance >= varianceThreshold) {
                selectedFeatures.put(featureName, featureValue);
            }
        }

        return selectedFeatures;
    }

    // Interaction between multiple features using pairwise multiplication
    public static Map<String, Double> createPairwiseInteractionFeatures(Map<String, Double> numericalFeatures) {
        Map<String, Double> interactionFeatures = new HashMap<>();
        List<String> featureNames = new ArrayList<>(numericalFeatures.keySet());

        for (int i = 0; i < featureNames.size(); i++) {
            for (int j = i + 1; j < featureNames.size(); j++) {
                String feature1 = featureNames.get(i);
                String feature2 = featureNames.get(j);
                double interactionValue = numericalFeatures.get(feature1) * numericalFeatures.get(feature2);

                interactionFeatures.put(feature1 + "_x_" + feature2, interactionValue);
            }
        }

        return interactionFeatures;
    }

    // Scaling numerical features using min-max scaling
    public static Map<String, Double> minMaxScaling(Map<String, Double> numericalFeatures) {
        Map<String, Double> scaledFeatures = new HashMap<>();
        double min = Collections.min(numericalFeatures.values());
        double max = Collections.max(numericalFeatures.values());

        for (Map.Entry<String, Double> entry : numericalFeatures.entrySet()) {
            String featureName = entry.getKey();
            Double featureValue = entry.getValue();
            scaledFeatures.put(featureName, (featureValue - min) / (max - min));
        }

        return scaledFeatures;
    }

    // Impute missing numerical features using the mean of available features
    public static Map<String, Double> imputeMissingValuesWithMean(Map<String, Double> numericalFeatures) {
        Map<String, Double> imputedFeatures = new HashMap<>();
        double mean = numericalFeatures.values().stream().mapToDouble(Double::doubleValue).average().orElse(0.0);

        for (Map.Entry<String, Double> entry : numericalFeatures.entrySet()) {
            String featureName = entry.getKey();
            Double featureValue = entry.getValue();
            imputedFeatures.put(featureName, featureValue != null ? featureValue : mean);
        }

        return imputedFeatures;
    }

    // Detect and handle outliers using Z-score method
    public static Map<String, Double> handleOutliers(Map<String, Double> numericalFeatures, double zThreshold) {
        Map<String, Double> cleanFeatures = new HashMap<>();
        double mean = numericalFeatures.values().stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        double stdDev = Math.sqrt(numericalFeatures.values().stream().mapToDouble(val -> Math.pow(val - mean, 2)).average().orElse(0.0));

        for (Map.Entry<String, Double> entry : numericalFeatures.entrySet()) {
            String featureName = entry.getKey();
            Double featureValue = entry.getValue();
            double zScore = (featureValue - mean) / stdDev;

            // Mark as outlier if Z-score exceeds threshold
            if (Math.abs(zScore) < zThreshold) {
                cleanFeatures.put(featureName, featureValue);
            } else {
                cleanFeatures.put(featureName, mean);
            }
        }

        return cleanFeatures;
    }

    // Generate binary features for a list of categorical features
    public static Map<String, Integer> generateBinaryFeatures(Map<String, String> categoricalFeatures, List<String> uniqueValues) {
        Map<String, Integer> binaryFeatures = new HashMap<>();

        for (Map.Entry<String, String> entry : categoricalFeatures.entrySet()) {
            String featureName = entry.getKey();
            String featureValue = entry.getValue();
            for (String value : uniqueValues) {
                binaryFeatures.put(featureName + "_" + value, featureValue.equals(value) ? 1 : 0);
            }
        }

        return binaryFeatures;
    }

    // Function to combine multiple feature sets into one
    public static Map<String, Object> combineFeatureSets(List<Map<String, Object>> featureSets) {
        Map<String, Object> combinedFeatures = new HashMap<>();

        for (Map<String, Object> featureSet : featureSets) {
            combinedFeatures.putAll(featureSet);
        }

        return combinedFeatures;
    }

    // Interaction between numerical and categorical features
    public static Map<String, Double> interactionBetweenNumericalAndCategorical(Map<String, Double> numericalFeatures, Map<String, String> categoricalFeatures) {
        Map<String, Double> interactionFeatures = new HashMap<>();

        for (Map.Entry<String, Double> numFeature : numericalFeatures.entrySet()) {
            for (Map.Entry<String, String> catFeature : categoricalFeatures.entrySet()) {
                String interactionKey = numFeature.getKey() + "_x_" + catFeature.getValue();
                interactionFeatures.put(interactionKey, numFeature.getValue() * catFeature.getValue().length());
            }
        }

        return interactionFeatures;
    }

    // Generating lagged features from a time series
    public static Map<String, Double> generateLaggedFeatures(List<Double> timeSeries, int lag) {
        Map<String, Double> laggedFeatures = new HashMap<>();

        for (int i = lag; i < timeSeries.size(); i++) {
            laggedFeatures.put("lag_" + lag + "_t_" + i, timeSeries.get(i - lag));
        }

        return laggedFeatures;
    }

    // Impute missing values for categorical features using the most frequent category
    public static Map<String, String> imputeMissingCategorical(Map<String, String> categoricalFeatures) {
        Map<String, String> imputedFeatures = new HashMap<>();
        Map<String, Long> frequencyMap = new HashMap<>();

        // Count frequencies of each category
        for (String value : categoricalFeatures.values()) {
            frequencyMap.put(value, frequencyMap.getOrDefault(value, 0L) + 1);
        }

        // Find the most frequent category
        String mostFrequentCategory = frequencyMap.entrySet().stream()
                .max(Map.Entry.comparingByValue()).map(Map.Entry::getKey).orElse(null);

        for (Map.Entry<String, String> entry : categoricalFeatures.entrySet()) {
            String featureName = entry.getKey();
            String featureValue = entry.getValue();
            imputedFeatures.put(featureName, featureValue != null ? featureValue : mostFrequentCategory);
        }

        return imputedFeatures;
    }

    // Feature scaling by standard deviation normalization
    public static Map<String, Double> standardDeviationNormalization(Map<String, Double> numericalFeatures) {
        Map<String, Double> normalizedFeatures = new HashMap<>();
        double mean = numericalFeatures.values().stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        double stdDev = Math.sqrt(numericalFeatures.values().stream().mapToDouble(val -> Math.pow(val - mean, 2)).average().orElse(0.0));

        for (Map.Entry<String, Double> entry : numericalFeatures.entrySet()) {
            String featureName = entry.getKey();
            Double featureValue = entry.getValue();
            normalizedFeatures.put(featureName, (featureValue - mean) / stdDev);
        }

        return normalizedFeatures;
    }

    // Feature encoding using frequency encoding
    public static Map<String, Double> frequencyEncode(Map<String, String> categoricalFeatures) {
        Map<String, Double> encodedFeatures = new HashMap<>();
        Map<String, Long> frequencyMap = new HashMap<>();

        // Calculate frequency of each category
        for (String value : categoricalFeatures.values()) {
            frequencyMap.put(value, frequencyMap.getOrDefault(value, 0L) + 1);
        }

        for (Map.Entry<String, String> entry : categoricalFeatures.entrySet()) {
            String featureName = entry.getKey();
            String featureValue = entry.getValue();
            encodedFeatures.put(featureName, (double) frequencyMap.getOrDefault(featureValue, 0L));
        }

        return encodedFeatures;
    }


    // Calculate the rolling mean for time series data
    public static Map<String, Double> calculateRollingMean(List<Double> timeSeries, int windowSize) {
        Map<String, Double> rollingMeanFeatures = new HashMap<>();

        for (int i = windowSize; i < timeSeries.size(); i++) {
            double sum = 0.0;
            for (int j = i - windowSize; j < i; j++) {
                sum += timeSeries.get(j);
            }
            double rollingMean = sum / windowSize;
            rollingMeanFeatures.put("rolling_mean_t_" + i, rollingMean);
        }

        return rollingMeanFeatures;
    }

    // Extract domain-specific features (for e-commerce data)
    public static Map<String, Double> extractEcommerceFeatures(Map<String, Object> transactionData) {
        Map<String, Double> ecommerceFeatures = new HashMap<>();

        // Calculate average basket size
        if (transactionData.containsKey("items_purchased") && transactionData.containsKey("total_spent")) {
            int itemsPurchased = (Integer) transactionData.get("items_purchased");
            double totalSpent = (Double) transactionData.get("total_spent");

            double avgBasketSize = totalSpent / itemsPurchased;
            ecommerceFeatures.put("avg_basket_size", avgBasketSize);
        }

        // Calculate discount percentage
        if (transactionData.containsKey("original_price") && transactionData.containsKey("discounted_price")) {
            double originalPrice = (Double) transactionData.get("original_price");
            double discountedPrice = (Double) transactionData.get("discounted_price");

            double discountPercentage = ((originalPrice - discountedPrice) / originalPrice) * 100;
            ecommerceFeatures.put("discount_percentage", discountPercentage);
        }

        return ecommerceFeatures;
    }

    // Create lag features for time series analysis
    public static Map<String, Double> createLagFeatures(List<Double> timeSeries, int lag) {
        Map<String, Double> lagFeatures = new HashMap<>();

        for (int i = lag; i < timeSeries.size(); i++) {
            lagFeatures.put("lag_t_" + i, timeSeries.get(i - lag));
        }

        return lagFeatures;
    }

    // Detect seasonality in time series data
    public static Map<String, Double> detectSeasonality(List<Double> timeSeries, int period) {
        Map<String, Double> seasonalityFeatures = new HashMap<>();
        double periodMean = 0.0;

        for (int i = 0; i < timeSeries.size(); i++) {
            if (i % period == 0 && i > 0) {
                periodMean = timeSeries.subList(i - period, i).stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
                seasonalityFeatures.put("seasonality_period_t_" + i, periodMean);
            }
        }

        return seasonalityFeatures;
    }

    // Extract sentiment score from text
    public static double extractSentiment(String text) {
        // Simulating a sentiment analysis process
        String[] positiveWords = {"good", "great", "excellent", "positive"};
        String[] negativeWords = {"bad", "poor", "negative", "terrible"};

        int sentimentScore = 0;

        for (String word : text.toLowerCase().split("\\s+")) {
            if (Arrays.asList(positiveWords).contains(word)) {
                sentimentScore++;
            } else if (Arrays.asList(negativeWords).contains(word)) {
                sentimentScore--;
            }
        }

        return sentimentScore / (double) text.split("\\s+").length;
    }

    // Time-based feature extraction (hour of the day, day of the week, etc.)
    public static Map<String, Integer> extractTimeFeatures(Date timestamp) {
        Map<String, Integer> timeFeatures = new HashMap<>();
        Calendar calendar = Calendar.getInstance();
        calendar.setTime(timestamp);

        // Extract hour of the day
        timeFeatures.put("hour_of_day", calendar.get(Calendar.HOUR_OF_DAY));

        // Extract day of the week
        timeFeatures.put("day_of_week", calendar.get(Calendar.DAY_OF_WEEK));

        // Extract day of the month
        timeFeatures.put("day_of_month", calendar.get(Calendar.DAY_OF_MONTH));

        return timeFeatures;
    }

    // Feature hashing for high-cardinality categorical features
    public static Map<String, Integer> featureHashing(List<String> categoricalFeatures, int numBuckets) {
        Map<String, Integer> hashedFeatures = new HashMap<>();

        for (String feature : categoricalFeatures) {
            int hashValue = feature.hashCode();
            int bucket = Math.abs(hashValue % numBuckets);
            hashedFeatures.put(feature, bucket);
        }

        return hashedFeatures;
    }

    // Detect trends in time series data using slope of linear regression
    public static double detectTrend(List<Double> timeSeries) {
        int n = timeSeries.size();
        double sumX = 0.0, sumY = 0.0, sumXY = 0.0, sumXX = 0.0;

        for (int i = 0; i < n; i++) {
            double x = i;
            double y = timeSeries.get(i);
            sumX += x;
            sumY += y;
            sumXY += x * y;
            sumXX += x * x;
        }

        // Calculate slope (trend)
        return (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    }

    // Calculate moving average for time series data
    public static Map<String, Double> calculateMovingAverage(List<Double> timeSeries, int windowSize) {
        Map<String, Double> movingAverageFeatures = new HashMap<>();

        for (int i = windowSize; i < timeSeries.size(); i++) {
            double sum = 0.0;
            for (int j = i - windowSize; j < i; j++) {
                sum += timeSeries.get(j);
            }
            double movingAverage = sum / windowSize;
            movingAverageFeatures.put("moving_avg_t_" + i, movingAverage);
        }

        return movingAverageFeatures;
    }

    // Detect anomalies in time series data using a simple threshold-based method
    public static Map<String, Boolean> detectAnomalies(List<Double> timeSeries, double threshold) {
        Map<String, Boolean> anomalyFeatures = new HashMap<>();

        for (int i = 0; i < timeSeries.size(); i++) {
            anomalyFeatures.put("anomaly_t_" + i, timeSeries.get(i) > threshold);
        }

        return anomalyFeatures;
    }

    // Extract domain-specific features for financial transactions
    public static Map<String, Double> extractFinancialFeatures(Map<String, Object> transactionData) {
        Map<String, Double> financialFeatures = new HashMap<>();

        // Calculate transaction amount to income ratio
        if (transactionData.containsKey("transaction_amount") && transactionData.containsKey("monthly_income")) {
            double transactionAmount = (Double) transactionData.get("transaction_amount");
            double monthlyIncome = (Double) transactionData.get("monthly_income");

            double incomeRatio = transactionAmount / monthlyIncome;
            financialFeatures.put("transaction_income_ratio", incomeRatio);
        }

        // Detect large transactions
        if (transactionData.containsKey("transaction_amount")) {
            double transactionAmount = (Double) transactionData.get("transaction_amount");

            financialFeatures.put("is_large_transaction", transactionAmount > 1000 ? 1.0 : 0.0);
        }

        return financialFeatures;
    }

    /* Main function to demonstrate feature engineering usage
    public static void main(String[] args) {
        // Time series data
        List<Double> timeSeries = Arrays.asList(100.0, 110.0, 120.0, 130.0, 125.0, 115.0, 105.0);

        // Extract time-based features
        Map<String, Double> movingAvgFeatures = calculateMovingAverage(timeSeries, 3);
        System.out.println("Moving Average Features: " + movingAvgFeatures);

        // Detect trend in the time series
        double trend = detectTrend(timeSeries);
        System.out.println("Time Series Trend: " + trend);

        // Detect anomalies
        Map<String, Boolean> anomalies = detectAnomalies(timeSeries, 120.0);
        System.out.println("Anomalies: " + anomalies);
    }*/

}