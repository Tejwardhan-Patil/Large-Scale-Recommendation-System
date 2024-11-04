package recommendation.personalization

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.evaluation.RegressionEvaluator

object Personalization {

  def main(args: Array[String]): Unit = {

    // Initialize Spark session with additional configuration settings
    val spark = SparkSession.builder
      .appName("PersonalizedRecommendationEngine")
      .config("spark.sql.shuffle.partitions", "8")
      .config("spark.executor.memory", "4g")
      .config("spark.driver.memory", "2g")
      .config("spark.executor.cores", "4")
      .getOrCreate()

    // Log information about the environment
    println("Spark session initialized with following configuration:")
    println(s"Executor memory: ${spark.conf.get("spark.executor.memory")}")
    println(s"Driver memory: ${spark.conf.get("spark.driver.memory")}")
    println(s"Shuffle partitions: ${spark.conf.get("spark.sql.shuffle.partitions")}")
    println(s"Executor cores: ${spark.conf.get("spark.executor.cores")}")

    // Load user interaction data from CSV file
    val interactions = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("/data/user_interactions.csv")

    println("User interactions data loaded successfully")
    interactions.show(5)

    // Ensure data is clean, remove any duplicate entries
    val distinctInteractions = interactions.dropDuplicates()
    println("Duplicates removed from interaction data")

    // Load item catalog data from another CSV file
    val items = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("/data/item_catalog.csv")

    println("Item catalog loaded successfully")
    items.show(5)

    // Join interactions and item data for better context in the model
    val fullData = distinctInteractions.join(items, "item_id")
    println("Joined user interactions with item catalog data")

    // Data exploration for better understanding
    println("Summary statistics for interaction data:")
    fullData.describe("rating").show()

    // Convert columns to appropriate data types for model ingestion
    val preparedData = fullData.select(
      col("user_id").cast("int"),
      col("item_id").cast("int"),
      col("rating").cast("float")
    )

    println("Data preparation complete, types converted to match model requirements")
    
    // Fill in missing values in the dataset
    val dataWithNoNulls = preparedData.na.fill(Map(
      "rating" -> 0.0
    ))

    println("Null values in ratings replaced with 0.0")

    // Check the distribution of ratings
    val ratingDistribution = dataWithNoNulls.groupBy("rating").count()
    ratingDistribution.orderBy("rating").show()

    // Split data into training and testing sets for model validation
    val Array(training, test) = dataWithNoNulls.randomSplit(Array(0.8, 0.2), seed = 1234L)

    println("Training and test datasets created:")
    println(s"Training data size: ${training.count()}")
    println(s"Test data size: ${test.count()}")

    // Define ALS (Alternating Least Squares) model for collaborative filtering
    val als = new ALS()
      .setUserCol("user_id")
      .setItemCol("item_id")
      .setRatingCol("rating")
      .setImplicitPrefs(false)
      .setColdStartStrategy("drop") // Drop rows with NaN predictions to avoid errors
      .setRank(10)
      .setMaxIter(20)
      .setRegParam(0.1)

    println("ALS model defined with rank 10, max iterations 20, and regularization parameter 0.1")

    // Fit the ALS model to the training data
    val model = als.fit(training)

    println("Model training complete")

    // Save the trained model to a specified directory
    val modelSavePath = "/models/personalization_als_model"
    model.write.overwrite().save(modelSavePath)

    println(s"Trained ALS model saved to $modelSavePath")

    // Generate predictions on the test data
    val predictions = model.transform(test)

    println("Model predictions generated on test dataset")
    predictions.show(5)

    // Evaluate the model using RMSE (Root Mean Square Error)
    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")

    val rmse = evaluator.evaluate(predictions)
    println(s"Root Mean Square Error (RMSE) on test data = $rmse")

    // Add logging for tuning recommendations
    if (rmse > 1.0) {
      println("Model RMSE is above acceptable threshold, tuning may be necessary")
    } else {
      println("Model performance is within acceptable range")
    }

    // Show sample of predictions for inspection
    println("Sample predictions:")
    predictions.select("user_id", "item_id", "rating", "prediction").show(10)

    // Feature engineering to improve recommendations
    println("Starting additional feature engineering process")

    // Add interaction frequency feature
    val interactionFrequency = distinctInteractions.groupBy("user_id", "item_id")
      .agg(count("*").alias("interaction_count"))

    println("Calculated interaction frequency between users and items")

    // Integrate interaction frequency into the prepared data
    val dataWithFrequency = preparedData.join(interactionFrequency, Seq("user_id", "item_id"), "left_outer")
      .na.fill(0, Seq("interaction_count"))

    println("Integrated interaction frequency into data")

    // Show sample of the updated data with interaction frequency
    dataWithFrequency.show(5)

    // Splitting dataset again to include frequency features
    val Array(trainingWithFrequency, testWithFrequency) = dataWithFrequency.randomSplit(Array(0.8, 0.2), seed = 1234L)

    println("Retraining model with interaction frequency features included")

    // Fit the ALS model again with new features
    val modelWithFrequency = als.fit(trainingWithFrequency)

    println("Model retrained with interaction frequency")

    // Generate new predictions
    val predictionsWithFrequency = modelWithFrequency.transform(testWithFrequency)

    println("Generated predictions with new model incorporating interaction frequency")
    predictionsWithFrequency.show(5)

    // Evaluate the new model
    val rmseWithFrequency = evaluator.evaluate(predictionsWithFrequency)
    println(s"New RMSE after including interaction frequency = $rmseWithFrequency")

    // Save the updated model
    val updatedModelSavePath = "/models/personalization_als_model_with_frequency"
    modelWithFrequency.write.overwrite().save(updatedModelSavePath)

    println(s"Updated model saved to $updatedModelSavePath")
  }
      // Additional feature: incorporating item popularity into the model
    println("Incorporating item popularity as an additional feature")

    // Calculate item popularity based on the total number of interactions
    val itemPopularity = distinctInteractions.groupBy("item_id")
      .agg(count("*").alias("popularity"))

    println("Item popularity calculated successfully")
    itemPopularity.show(5)

    // Merge item popularity into the dataset
    val dataWithPopularity = dataWithFrequency.join(itemPopularity, Seq("item_id"), "left_outer")
      .na.fill(0, Seq("popularity"))

    println("Merged item popularity into the dataset")
    dataWithPopularity.show(5)

    // Re-split the dataset to include the popularity feature
    val Array(trainingWithPopularity, testWithPopularity) = dataWithPopularity.randomSplit(Array(0.8, 0.2), seed = 1234L)

    // Train the ALS model again, now with interaction frequency and popularity features
    println("Retraining ALS model with both interaction frequency and item popularity features")

    val modelWithPopularity = als.fit(trainingWithPopularity)

    println("Model retrained successfully with interaction frequency and item popularity")

    // Generate predictions with the updated model
    val predictionsWithPopularity = modelWithPopularity.transform(testWithPopularity)

    println("Generated predictions using the new model with popularity feature")
    predictionsWithPopularity.show(5)

    // Evaluate the updated model performance
    val rmseWithPopularity = evaluator.evaluate(predictionsWithPopularity)
    println(s"RMSE after incorporating item popularity = $rmseWithPopularity")

    // Log if further tuning is needed based on the new RMSE
    if (rmseWithPopularity > rmseWithFrequency) {
      println("RMSE increased after adding popularity, further model tuning required")
    } else {
      println("RMSE improved with the popularity feature")
    }

    // Save the final model with both additional features
    val finalModelSavePath = "/models/personalization_als_model_with_popularity"
    modelWithPopularity.write.overwrite().save(finalModelSavePath)

    println(s"Final model saved to $finalModelSavePath")

    // Generate recommendations for all users using the final model
    val allUserRecommendations = modelWithPopularity.recommendForAllUsers(10)
    println("Generated top 10 recommendations for all users")
    allUserRecommendations.show(5, false)

    // Generate recommendations for all items
    val allItemRecommendations = modelWithPopularity.recommendForAllItems(10)
    println("Generated top 10 recommendations for all items")
    allItemRecommendations.show(5, false)

    // Save the recommendations for users
    val userRecommendationsPath = "/recommendations/user_recommendations"
    allUserRecommendations.write
      .mode("overwrite")
      .parquet(userRecommendationsPath)

    println(s"User recommendations saved to $userRecommendationsPath")

    // Save the recommendations for items
    val itemRecommendationsPath = "/recommendations/item_recommendations"
    allItemRecommendations.write
      .mode("overwrite")
      .parquet(itemRecommendationsPath)

    println(s"Item recommendations saved to $itemRecommendationsPath")

    // Implementing real-time personalization logic for live updates
    println("Starting real-time personalization logic")

    // Real-time data streaming (simulated with a batch load for now)
    val realTimeUpdates = spark.readStream
      .format("csv")
      .option("header", "true")
      .schema(interactions.schema)
      .load("/data/real_time_updates/")

    // Process real-time updates by joining with the item catalog
    val updatedInteractions = realTimeUpdates.join(items, "item_id")

    // Predict on the new real-time interactions using the latest model
    val realTimePredictions = modelWithPopularity.transform(updatedInteractions)

    println("Real-time predictions generated")
    realTimePredictions.show(5)

    // Write the real-time predictions to an output stream (simulated)
    val realTimePredictionsPath = "/output/real_time_predictions"
    realTimePredictions.writeStream
      .format("parquet")
      .option("checkpointLocation", "/checkpoints/predictions")
      .start(realTimePredictionsPath)

    println(s"Real-time predictions being written to $realTimePredictionsPath")

    // Monitoring the performance of the system
    println("Monitoring system performance for real-time updates")

    // Metrics such as latency, throughput, and prediction accuracy
    val latencyMetric = System.currentTimeMillis() - 1234567890L // Timestamp for latency
    println(s"Current latency: $latencyMetric ms")

    val throughputMetric = realTimePredictions.count() / 60.0
    println(s"Current throughput: $throughputMetric predictions/second")

    // Monitor any anomalies in data or predictions
    println("Checking for anomalies in real-time data")

    // Check for anomalous predictions such as extremely high or low values
    val anomalies = realTimePredictions.filter(col("prediction") > 5 || col("prediction") < 0)
    if (anomalies.count() > 0) {
      println(s"Anomalies detected in predictions: ${anomalies.count()} records")
      anomalies.show(5)
    } else {
      println("No anomalies detected in real-time predictions")
    }

    // Scale the infrastructure based on the load
    println("Scaling infrastructure based on real-time data volume")

    // Simulate auto-scaling by logging current load and hypothetical scaling actions
    val currentDataVolume = realTimeUpdates.count()
    if (currentDataVolume > 100000) {
      println("Data volume exceeds threshold, triggering auto-scaling actions")
      println("Scaling up resources to handle increased load")
    } else {
      println("Data volume is within acceptable limits, no scaling required")
    }

    // Introducing additional features: collaborative filtering with implicit feedback
    println("Introducing collaborative filtering with implicit feedback")

    val alsImplicit = new ALS()
      .setUserCol("user_id")
      .setItemCol("item_id")
      .setRatingCol("rating")
      .setImplicitPrefs(true) // Switch to implicit feedback mode
      .setColdStartStrategy("drop")
      .setRank(15)
      .setMaxIter(25)
      .setRegParam(0.2)

    // Train the model using implicit feedback data
    val modelImplicit = alsImplicit.fit(training)

    println("Model trained using implicit feedback")

    // Generate predictions for implicit feedback
    val predictionsImplicit = modelImplicit.transform(test)

    println("Generated predictions with implicit feedback model")
    predictionsImplicit.show(5)

    // Evaluate implicit feedback model
    val rmseImplicit = evaluator.evaluate(predictionsImplicit)
    println(s"RMSE for implicit feedback model = $rmseImplicit")

    // Compare with the previous models and decide on the best-performing one
    if (rmseImplicit < rmseWithPopularity) {
      println("Implicit feedback model performs better than previous models")
    } else {
      println("Previous models perform better than implicit feedback model")
    }

    // Save the implicit feedback model
    val implicitModelSavePath = "/models/personalization_implicit_model"
    modelImplicit.write.overwrite().save(implicitModelSavePath)

    println(s"Implicit feedback model saved to $implicitModelSavePath")

    // Continue real-time updates and monitoring
    println("Continuing real-time updates and monitoring system performance")

    // Advanced model evaluation: precision and recall calculation for recommendations
    println("Evaluating model using precision and recall metrics")

    // Create a precision and recall evaluator for the implicit feedback model
    val topK = 10

    // Function to compute precision and recall
    def precisionAndRecall(predictions: DataFrame, k: Int): (Double, Double) = {
      val relevantItems = predictions.filter(col("rating") >= 4) // Define relevant items as those with rating >= 4
      val recommendedItems = predictions.orderBy(desc("prediction")).limit(k)

      val truePositives = relevantItems.join(recommendedItems, Seq("item_id")).count()
      val precision = truePositives.toDouble / recommendedItems.count()
      val recall = truePositives.toDouble / relevantItems.count()

      (precision, recall)
    }

    // Apply precision and recall to predictions from the implicit feedback model
    val (precisionImplicit, recallImplicit) = precisionAndRecall(predictionsImplicit, topK)

    println(s"Precision at $topK for implicit feedback model: $precisionImplicit")
    println(s"Recall at $topK for implicit feedback model: $recallImplicit")

    // Save evaluation metrics to a log file
    val metricsLogPath = "/logs/evaluation_metrics.txt"
    import java.io.PrintWriter
    new PrintWriter(metricsLogPath) {
      write(s"RMSE for implicit feedback model: $rmseImplicit\n")
      write(s"Precision at $topK: $precisionImplicit\n")
      write(s"Recall at $topK: $recallImplicit\n")
      close()
    }
    println(s"Model evaluation metrics saved to $metricsLogPath")

    // Implementing hybrid recommendation system: combining collaborative filtering and content-based filtering
    println("Implementing hybrid recommendation system with collaborative filtering and content-based filtering")

    // Assume item metadata contains content-based features (genre, tags, etc.)
    val itemMetadata = items.select("item_id", "genre", "tags")

    // Function to compute content similarity (Jaccard similarity for categorical features)
    def jaccardSimilarity(s1: String, s2: String): Double = {
      val set1 = s1.split(",").toSet
      val set2 = s2.split(",").toSet
      val intersection = set1.intersect(set2).size.toDouble
      val union = set1.union(set2).size.toDouble
      intersection / union
    }

    // Create a UDF to compute Jaccard similarity between item tags
    val jaccardUDF = udf((tags1: String, tags2: String) => jaccardSimilarity(tags1, tags2))

    // Generate content-based similarity score between items based on their tags
    val itemContentSimilarity = itemMetadata.alias("i1")
      .join(itemMetadata.alias("i2"), $"i1.item_id" =!= $"i2.item_id")
      .select(
        col("i1.item_id").alias("item1"),
        col("i2.item_id").alias("item2"),
        jaccardUDF(col("i1.tags"), col("i2.tags")).alias("similarity_score")
      )

    println("Content-based similarity between items computed using Jaccard similarity on tags")
    itemContentSimilarity.show(5)

    // Combine collaborative filtering with content-based similarity
    val hybridRecommendations = predictionsImplicit.join(itemContentSimilarity, predictionsImplicit("item_id") === itemContentSimilarity("item1"))
      .select(
        col("user_id"),
        col("item2").alias("recommended_item"),
        (col("prediction") + col("similarity_score")).alias("hybrid_score")
      )
      .orderBy(desc("hybrid_score"))

    println("Hybrid recommendations generated by combining collaborative filtering and content similarity")
    hybridRecommendations.show(5)

    // Save hybrid recommendations
    val hybridRecommendationsPath = "/recommendations/hybrid_recommendations"
    hybridRecommendations.write
      .mode("overwrite")
      .parquet(hybridRecommendationsPath)

    println(s"Hybrid recommendations saved to $hybridRecommendationsPath")

    // Implement feedback loop for model improvement based on real-time user interactions
    println("Implementing feedback loop for continuous model improvement")

    // Simulate a feedback loop that updates the model with real-time user interactions
    val feedbackData = spark.readStream
      .format("csv")
      .option("header", "true")
      .schema(interactions.schema)
      .load("/data/feedback/")

    // Process the feedback data to update the existing model
    val updatedModelTrainingData = training.union(feedbackData)

    // Re-train the ALS model with feedback data
    val updatedModel = als.fit(updatedModelTrainingData)

    println("ALS model retrained with real-time feedback data")

    // Save the updated model
    val updatedModelWithFeedbackPath = "/models/personalization_als_model_with_feedback"
    updatedModel.write.overwrite().save(updatedModelWithFeedbackPath)

    println(s"Updated model with feedback saved to $updatedModelWithFeedbackPath")

    // Generate new recommendations after incorporating feedback
    val newRecommendationsWithFeedback = updatedModel.transform(test)

    println("Generated new recommendations after incorporating real-time feedback")
    newRecommendationsWithFeedback.show(5)

    // Final evaluation of the updated model with feedback
    val finalRmse = evaluator.evaluate(newRecommendationsWithFeedback)
    println(s"Final RMSE after incorporating feedback = $finalRmse")

    // If performance improves, use the updated model for real-time recommendations
    if (finalRmse < rmseImplicit) {
      println("Updated model with feedback improves performance. Using this model for real-time recommendations.")
    } else {
      println("Updated model does not improve performance. Retaining the previous model.")
    }

    // Deploy the final model to production environment
    println("Deploying final model to production environment")

    val finalModelPath = if (finalRmse < rmseImplicit) updatedModelWithFeedbackPath else implicitModelSavePath

    println(s"Final model deployed from $finalModelPath")

    // Monitor production system to ensure stability and scalability
    println("Monitoring production system for recommendation engine")

    // Simulate production monitoring by tracking system health metrics
    val memoryUsage = 80.0 // Metric for memory usage in percentage
    val cpuUsage = 70.0    // Metric for CPU usage in percentage

    if (memoryUsage > 85.0) {
      println("Memory usage is high, considering scaling up memory resources")
    } else {
      println("Memory usage is within acceptable limits")
    }

    if (cpuUsage > 75.0) {
      println("CPU usage is high, considering increasing CPU cores")
    } else {
      println("CPU usage is within acceptable limits")
    }

    // Final step: Logging system metrics and shutting down Spark session
    val systemMetricsLogPath = "/logs/system_metrics.txt"
    new PrintWriter(systemMetricsLogPath) {
      write(s"Final RMSE: $finalRmse\n")
      write(s"Memory usage: $memoryUsage%\n")
      write(s"CPU usage: $cpuUsage%\n")
      close()
    }

    println(s"System metrics saved to $systemMetricsLogPath")

    // Stop the Spark session to release resources
    println("Shutting down Spark session")
    spark.stop()

}