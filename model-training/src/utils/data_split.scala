package utils

import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType

object DataSplit {

  // Function to split data into training, validation, and test sets
  def splitData(df: DataFrame, trainRatio: Double, validationRatio: Double): (DataFrame, DataFrame, DataFrame) = {
    // Ensure the sum of ratios is 1.0
    require(trainRatio + validationRatio < 1.0, "Train and validation ratios must sum to less than 1.0")

    // Calculate test ratio
    val testRatio = 1.0 - trainRatio - validationRatio

    // Split the data
    val Array(trainData, validationData, testData) = df.randomSplit(Array(trainRatio, validationRatio, testRatio))

    (trainData, validationData, testData)
  }

  // Function to read data from a CSV file
  def loadData(spark: SparkSession, filePath: String, schema: StructType, delimiter: String = ","): DataFrame = {
    spark.read
      .option("header", "true")
      .option("delimiter", delimiter)
      .schema(schema)
      .csv(filePath)
  }

  // Function to write DataFrame to storage
  def saveData(df: DataFrame, path: String, format: String = "parquet"): Unit = {
    df.write.format(format).save(path)
  }

  def main(args: Array[String]): Unit = {
    // Initialize Spark session
    val spark = SparkSession.builder()
      .appName("Data Split Utility")
      .getOrCreate()

    // Load data
    val schema = new StructType()
      .add("id", "integer")
      .add("feature1", "double")
      .add("feature2", "double")
      .add("label", "double")

    val dataPath = "s3a://website.com/data/training_data.csv"
    val data = loadData(spark, dataPath, schema)

    // Split data into train, validation, and test sets
    val (trainData, validationData, testData) = splitData(data, 0.7, 0.2)

    // Save the datasets
    saveData(trainData, "s3a://website.com/data/train/")
    saveData(validationData, "s3a://website.com/data/validation/")
    saveData(testData, "s3a://website.com/data/test/")

    // Stop the Spark session
    spark.stop()
  }
}