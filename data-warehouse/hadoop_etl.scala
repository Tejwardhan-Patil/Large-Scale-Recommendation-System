import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.hadoop.fs._
import org.apache.log4j.{Level, Logger}
import java.time.LocalDateTime

// Logging configuration
Logger.getLogger("org").setLevel(Level.WARN)

object HadoopETL {
  
  // Initialize Spark session
  val spark = SparkSession.builder()
    .appName("HadoopETL")
    .config("spark.master", "yarn")
    .config("spark.sql.shuffle.partitions", "200")
    .getOrCreate()

  // Function to log messages with timestamps
  def logMessage(message: String): Unit = {
    val timestamp = LocalDateTime.now
    println(s"[$timestamp] - $message")
  }

  // Extract step: Load data from HDFS, S3, or another source
  def extract(sourcePath: String): DataFrame = {
    logMessage(s"Extracting data from $sourcePath")
    try {
      val data = spark.read.format("parquet").load(sourcePath)
      logMessage(s"Successfully loaded data from $sourcePath")
      data
    } catch {
      case e: Exception =>
        logMessage(s"Failed to load data from $sourcePath: ${e.getMessage}")
        throw e
    }
  }

  // Transform step: Apply necessary data transformations
  def transform(data: DataFrame): DataFrame = {
    logMessage("Starting data transformation")

    // Transformation: Filter, Group By, Aggregations, etc.
    val transformedData = data
      .filter(col("is_active") === true)
      .withColumn("processed_date", current_date())
      .groupBy("category")
      .agg(
        count("item_id").as("item_count"),
        avg("price").as("average_price")
      )
      .withColumnRenamed("category", "item_category")

    logMessage("Data transformation completed")
    transformedData
  }

  // Load step: Save the transformed data to HDFS or another target
  def load(data: DataFrame, targetPath: String): Unit = {
    logMessage(s"Loading data to $targetPath")
    try {
      data.write.mode("overwrite").parquet(targetPath)
      logMessage(s"Data successfully loaded to $targetPath")
    } catch {
      case e: Exception =>
        logMessage(s"Failed to load data to $targetPath: ${e.getMessage}")
        throw e
    }
  }

  // Validate step: Check if target directory is written correctly
  def validate(targetPath: String): Boolean = {
    logMessage(s"Validating data in $targetPath")
    try {
      val fs = FileSystem.get(spark.sparkContext.hadoopConfiguration)
      val path = new Path(targetPath)
      val isValid = fs.exists(path) && fs.getContentSummary(path).getLength > 0

      if (isValid) {
        logMessage(s"Data validation successful for $targetPath")
      } else {
        logMessage(s"Data validation failed for $targetPath")
      }

      isValid
    } catch {
      case e: Exception =>
        logMessage(s"Validation failed for $targetPath: ${e.getMessage}")
        false
    }
  }

  // Function to clean the target directory before loading new data
  def cleanTarget(targetPath: String): Unit = {
    logMessage(s"Cleaning target directory: $targetPath")
    try {
      val fs = FileSystem.get(spark.sparkContext.hadoopConfiguration)
      val path = new Path(targetPath)
      if (fs.exists(path)) {
        fs.delete(path, true)
        logMessage(s"Target directory $targetPath cleaned successfully")
      } else {
        logMessage(s"Target directory $targetPath does not exist, no need to clean")
      }
    } catch {
      case e: Exception =>
        logMessage(s"Failed to clean target directory $targetPath: ${e.getMessage}")
        throw e
    }
  }

  // Run the ETL process
  def runETL(sourcePath: String, targetPath: String): Unit = {
    logMessage("Starting ETL process")

    // Step 1: Clean the target directory
    cleanTarget(targetPath)

    // Step 2: Extract
    val extractedData = extract(sourcePath)

    // Step 3: Transform
    val transformedData = transform(extractedData)

    // Step 4: Load
    load(transformedData, targetPath)

    // Step 5: Validate
    val isValid = validate(targetPath)
    if (!isValid) {
      logMessage(s"ETL process failed during validation for $targetPath")
      throw new Exception("ETL validation failed")
    }

    logMessage("ETL process completed successfully")
  }

  // Retry mechanism for fault-tolerant ETL execution
  def retryETL(sourcePath: String, targetPath: String, maxRetries: Int): Unit = {
    var retries = 0
    var success = false
    while (retries < maxRetries && !success) {
      try {
        runETL(sourcePath, targetPath)
        success = true
      } catch {
        case e: Exception =>
          retries += 1
          logMessage(s"ETL failed on attempt $retries: ${e.getMessage}")
          if (retries == maxRetries) {
            logMessage(s"Max retries reached. ETL failed for $sourcePath to $targetPath")
            throw e
          } else {
            logMessage(s"Retrying ETL process for $sourcePath to $targetPath (attempt $retries of $maxRetries)")
          }
      }
    }
  }

  // Main method to start the ETL process
  def main(args: Array[String]): Unit = {
    if (args.length != 3) {
      logMessage("Usage: HadoopETL <sourcePath> <targetPath> <maxRetries>")
      sys.exit(1)
    }

    val sourcePath = args(0)
    val targetPath = args(1)
    val maxRetries = args(2).toInt

    try {
      retryETL(sourcePath, targetPath, maxRetries)
    } catch {
      case e: Exception =>
        logMessage(s"ETL process failed: ${e.getMessage}")
        sys.exit(1)
    }
  }

  // Function to monitor ETL job performance
  def monitorPerformance(startTime: Long): Unit = {
    val endTime = System.currentTimeMillis()
    val duration = (endTime - startTime) / 1000 // Convert to seconds
    logMessage(s"ETL job completed in $duration seconds")
  }

  // Function to perform advanced data transformations
  def advancedTransform(data: DataFrame): DataFrame = {
    logMessage("Starting advanced data transformations")

    // Advanced transformation: Pivot, Windowing, and Complex Calculations
    val windowSpec = org.apache.spark.sql.expressions.Window.partitionBy("item_category").orderBy(desc("price"))

    val transformedData = data
      .withColumn("rank", row_number().over(windowSpec))
      .filter(col("rank") <= 10) // Get top 10 highest-priced items per category
      .withColumn("price_category",
        when(col("price") > 1000, "Premium")
          .when(col("price").between(500, 1000), "Mid-range")
          .otherwise("Economy")
      )
      .groupBy("price_category")
      .agg(
        count("item_id").as("total_items"),
        avg("price").as("avg_price"),
        max("price").as("max_price"),
        min("price").as("min_price")
      )

    logMessage("Advanced transformations completed")
    transformedData
  }

  // Function to handle errors and exceptions during ETL
  def handleETLError(e: Exception, sourcePath: String, targetPath: String, attempt: Int): Unit = {
    logMessage(s"ETL process encountered an error: ${e.getMessage}")
    logMessage(s"Attempting cleanup after failure (attempt $attempt)")
    
    try {
      cleanTarget(targetPath)
    } catch {
      case cleanupError: Exception =>
        logMessage(s"Cleanup failed: ${cleanupError.getMessage}")
    }

    if (attempt >= 3) {
      logMessage("Max retries reached. ETL process cannot recover from the error.")
      sys.exit(1)
    } else {
      logMessage(s"Retrying ETL after error (attempt $attempt)")
    }
  }

  // Function to generate summary reports after ETL
  def generateReport(transformedData: DataFrame, reportPath: String): Unit = {
    logMessage(s"Generating ETL summary report at $reportPath")
    
    try {
      transformedData.write.mode("overwrite").csv(reportPath)
      logMessage(s"Report successfully generated at $reportPath")
    } catch {
      case e: Exception =>
        logMessage(s"Failed to generate report: ${e.getMessage}")
        throw e
    }
  }

  // Function to extract, transform, and load with advanced transformation
  def runAdvancedETL(sourcePath: String, targetPath: String, reportPath: String): Unit = {
    logMessage("Starting Advanced ETL process")

    val startTime = System.currentTimeMillis()

    // Step 1: Clean target directory
    cleanTarget(targetPath)

    // Step 2: Extract
    val extractedData = extract(sourcePath)

    // Step 3: Basic Transformation
    val transformedData = transform(extractedData)

    // Step 4: Advanced Transformation
    val advancedData = advancedTransform(transformedData)

    // Step 5: Load advanced transformed data
    load(advancedData, targetPath)

    // Step 6: Generate summary report
    generateReport(advancedData, reportPath)

    // Step 7: Validate
    val isValid = validate(targetPath)
    if (!isValid) {
      logMessage(s"ETL process failed during validation for $targetPath")
      throw new Exception("ETL validation failed")
    }

    // Monitor job performance
    monitorPerformance(startTime)

    logMessage("Advanced ETL process completed successfully")
  }

  // Function to run multiple ETL jobs concurrently
  def runConcurrentETL(jobs: Seq[(String, String, String)]): Unit = {
    logMessage(s"Starting concurrent ETL jobs")

    val jobFutures = jobs.map { case (sourcePath, targetPath, reportPath) =>
      scala.concurrent.Future {
        try {
          runAdvancedETL(sourcePath, targetPath, reportPath)
        } catch {
          case e: Exception =>
            logMessage(s"ETL job failed for $sourcePath to $targetPath: ${e.getMessage}")
        }
      }(scala.concurrent.ExecutionContext.global)
    }

    import scala.concurrent.Await
    import scala.concurrent.duration._

    // Wait for all ETL jobs to complete with a timeout
    Await.result(scala.concurrent.Future.sequence(jobFutures), 60.minutes)

    logMessage("All concurrent ETL jobs completed successfully")
  }

  // Additional Data Transformations - Handle Multiple Data Sources
  def mergeMultipleSources(sourcePaths: Seq[String]): DataFrame = {
    logMessage("Merging data from multiple sources")

    val dataFrames = sourcePaths.map { path =>
      try {
        extract(path)
      } catch {
        case e: Exception =>
          logMessage(s"Failed to extract data from $path: ${e.getMessage}")
          spark.emptyDataFrame
      }
    }

    val mergedData = dataFrames.reduce((df1, df2) => df1.unionByName(df2))

    logMessage("Merging of data from multiple sources completed")
    mergedData
  }

  // Function to handle schema evolution during ETL
  def handleSchemaEvolution(data: DataFrame, expectedSchema: Seq[String]): DataFrame = {
    logMessage("Handling schema evolution")

    // Get columns in the current data
    val currentSchema = data.columns.toSeq

    // Find missing columns
    val missingColumns = expectedSchema.diff(currentSchema)

    // Add missing columns with null values
    val dataWithFullSchema = missingColumns.foldLeft(data) { (df, colName) =>
      df.withColumn(colName, lit(null))
    }

    logMessage("Schema evolution handled successfully")
    dataWithFullSchema
  }

  // Function to clean data (remove duplicates, handle nulls, etc.)
  def cleanData(data: DataFrame): DataFrame = {
    logMessage("Starting data cleaning")

    val cleanedData = data
      .dropDuplicates("item_id") // Remove duplicates based on item_id
      .na.fill("unknown", Seq("category")) // Fill null categories with 'unknown'
      .na.fill(0, Seq("price", "stock_quantity")) // Fill missing numeric values with 0

    logMessage("Data cleaning completed successfully")
    cleanedData
  }

  // Run ETL with Schema Evolution and Data Cleaning
  def runETLWithSchemaAndCleaning(sourcePath: String, targetPath: String, expectedSchema: Seq[String]): Unit = {
    logMessage("Starting ETL process with schema evolution and data cleaning")

    // Step 1: Clean target directory
    cleanTarget(targetPath)

    // Step 2: Extract
    val extractedData = extract(sourcePath)

    // Step 3: Handle Schema Evolution
    val dataWithFullSchema = handleSchemaEvolution(extractedData, expectedSchema)

    // Step 4: Clean Data
    val cleanedData = cleanData(dataWithFullSchema)

    // Step 5: Basic Transformation
    val transformedData = transform(cleanedData)

    // Step 6: Load transformed data
    load(transformedData, targetPath)

    // Step 7: Validate
    val isValid = validate(targetPath)
    if (!isValid) {
      logMessage(s"ETL process failed during validation for $targetPath")
      throw new Exception("ETL validation failed")
    }

    logMessage("ETL with schema evolution and data cleaning completed successfully")
  }

  // Function to integrate with external logging systems (ELK, Datadog)
  def externalLogging(message: String, severity: String = "INFO"): Unit = {
    logMessage(s"Sending log to external system: $message with severity: $severity")
    // Integration with an external logging service
    // REST API call to send logs to an ELK stack or Datadog
    try {
      val response = scala.io.Source.fromURL(s"http://logging-system/api/log?message=$message&severity=$severity")
      response.close()
      logMessage(s"Successfully logged to external system: $message")
    } catch {
      case e: Exception =>
        logMessage(s"Failed to log to external system: ${e.getMessage}")
    }
  }

  // Function to send notifications (Slack, Email) after ETL success or failure
  def sendNotification(message: String, notificationType: String = "slack"): Unit = {
    logMessage(s"Sending $notificationType notification: $message")
    // Send a notification using a webhook or email service
    notificationType match {
      case "slack" =>
        // Send Slack notification via webhook
        val webhookUrl = "https://hooks.slack.com/services/webhook/url"
        try {
          val payload = s"""{"text": "$message"}"""
          val connection = new java.net.URL(webhookUrl).openConnection().asInstanceOf[java.net.HttpURLConnection]
          connection.setRequestMethod("POST")
          connection.setDoOutput(true)
          val wr = new java.io.OutputStreamWriter(connection.getOutputStream)
          wr.write(payload)
          wr.flush()
          wr.close()
          connection.getResponseCode match {
            case 200 => logMessage(s"Slack notification sent successfully: $message")
            case _   => logMessage(s"Failed to send Slack notification")
          }
        } catch {
          case e: Exception =>
            logMessage(s"Error sending Slack notification: ${e.getMessage}")
        }
      case "email" =>
        // Send email notification (this can be extended with an email API)
        logMessage(s"Sending email: $message")
      case _ =>
        logMessage(s"Unknown notification type: $notificationType")
    }
  }

  // Function to check system resource utilization (CPU, Memory)
  def checkSystemResources(): Unit = {
    logMessage("Checking system resources")
    val runtime = Runtime.getRuntime
    val availableProcessors = runtime.availableProcessors()
    val freeMemory = runtime.freeMemory() / (1024 * 1024) // Convert to MB
    val maxMemory = runtime.maxMemory() / (1024 * 1024) // Convert to MB
    val usedMemory = (runtime.totalMemory() - runtime.freeMemory()) / (1024 * 1024) // Convert to MB

    logMessage(s"System Resources - Processors: $availableProcessors, Free Memory: $freeMemory MB, Used Memory: $usedMemory MB, Max Memory: $maxMemory MB")

    // If resources are low, log a warning and notify the team
    if (freeMemory < 500) {
      val warningMessage = "Warning: Low system memory detected."
      logMessage(warningMessage, "WARN")
      externalLogging(warningMessage, "WARN")
      sendNotification(warningMessage, "slack")
    }
  }

  // Advanced validation function (comparing row counts, data consistency checks)
  def advancedValidation(sourceData: DataFrame, targetData: DataFrame): Boolean = {
    logMessage("Starting advanced validation")

    // Validate row counts between source and target
    val sourceCount = sourceData.count()
    val targetCount = targetData.count()
    logMessage(s"Source row count: $sourceCount, Target row count: $targetCount")

    if (sourceCount != targetCount) {
      logMessage("Row count mismatch detected during validation")
      sendNotification("Row count mismatch detected in ETL process", "slack")
      return false
    }

    // Data consistency check
    val invalidRecords = targetData.filter(col("price") < 0)
    if (invalidRecords.count() > 0) {
      logMessage(s"Data consistency issue found: Negative prices detected")
      sendNotification("Data consistency issue in ETL process: Negative prices detected", "slack")
      return false
    }

    logMessage("Advanced validation passed successfully")
    true
  }

  // Function to load incremental data (appending only new or updated records)
  def incrementalLoad(data: DataFrame, targetPath: String, lastUpdated: String): Unit = {
    logMessage("Starting incremental load")

    // Filter only new or updated records based on the 'updated_at' timestamp
    val incrementalData = data.filter(col("updated_at") > lastUpdated)

    if (incrementalData.count() == 0) {
      logMessage("No new or updated records found for incremental load")
    } else {
      logMessage(s"Loading ${incrementalData.count()} new or updated records to $targetPath")
      incrementalData.write.mode("append").parquet(targetPath)
      logMessage("Incremental load completed successfully")
    }
  }

  // Function to archive old data (move older files to an archive location)
  def archiveOldData(targetPath: String, archivePath: String, retentionDays: Int): Unit = {
    logMessage(s"Archiving data older than $retentionDays days from $targetPath to $archivePath")

    try {
      val fs = FileSystem.get(spark.sparkContext.hadoopConfiguration)
      val path = new Path(targetPath)
      if (fs.exists(path)) {
        val cutoffDate = System.currentTimeMillis() - (retentionDays * 24 * 60 * 60 * 1000L)

        val files = fs.listStatus(path).filter(_.getModificationTime < cutoffDate)
        files.foreach { file =>
          val srcPath = file.getPath
          val destPath = new Path(archivePath + "/" + srcPath.getName)
          fs.rename(srcPath, destPath)
          logMessage(s"Archived file: $srcPath to $destPath")
        }
      } else {
        logMessage(s"No data found at $targetPath to archive")
      }
    } catch {
      case e: Exception =>
        logMessage(s"Failed to archive old data: ${e.getMessage}")
        throw e
    }
  }

  // Function to handle complex joins in ETL
  def complexJoin(data1: DataFrame, data2: DataFrame, joinColumns: Seq[String]): DataFrame = {
    logMessage("Performing complex join between datasets")

    try {
      val joinedData = data1.join(data2, joinColumns, "inner")
      logMessage("Complex join completed successfully")
      joinedData
    } catch {
      case e: Exception =>
        logMessage(s"Failed to perform join: ${e.getMessage}")
        throw e
    }
  }

  // Function to perform post-load optimization (compacting small files)
  def optimizeData(targetPath: String): Unit = {
    logMessage(s"Starting post-load optimization for $targetPath")

    try {
      val data = spark.read.parquet(targetPath)
      data.repartition(1).write.mode("overwrite").parquet(targetPath)
      logMessage(s"Optimization completed for $targetPath (compacted small files)")
    } catch {
      case e: Exception =>
        logMessage(s"Failed to optimize data: ${e.getMessage}")
        throw e
    }
  }

  // ETL function that supports incremental loading, validation, and post-load optimization
  def runFullETL(sourcePath: String, targetPath: String, archivePath: String, lastUpdated: String, retentionDays: Int): Unit = {
    logMessage("Starting full ETL process with incremental load and optimization")

    val startTime = System.currentTimeMillis()

    // Step 1: Extract
    val extractedData = extract(sourcePath)

    // Step 2: Incremental Load
    incrementalLoad(extractedData, targetPath, lastUpdated)

    // Step 3: Archive Old Data
    archiveOldData(targetPath, archivePath, retentionDays)

    // Step 4: Post-load Optimization
    optimizeData(targetPath)

    // Monitor performance
    monitorPerformance(startTime)

    logMessage("Full ETL process completed successfully")
  }

  // Main method to execute full ETL process
  def main(args: Array[String]): Unit = {
    if (args.length != 5) {
      logMessage("Usage: HadoopETL <sourcePath> <targetPath> <archivePath> <lastUpdated> <retentionDays>")
      sys.exit(1)
    }

    val sourcePath = args(0)
    val targetPath = args(1)
    val archivePath = args(2)
    val lastUpdated = args(3)
    val retentionDays = args(4).toInt

    try {
      runFullETL(sourcePath, targetPath, archivePath, lastUpdated, retentionDays)
    } catch {
      case e: Exception =>
        logMessage(s"ETL process failed: ${e.getMessage}")
        sys.exit(1)
    }
  }
}