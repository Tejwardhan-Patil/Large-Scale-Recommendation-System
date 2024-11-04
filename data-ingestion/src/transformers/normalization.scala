package transformers

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window
import org.slf4j.{Logger, LoggerFactory}

// Normalization object handling different types of normalization for large-scale datasets
object Normalization {

  val logger: Logger = LoggerFactory.getLogger(this.getClass)

  // Min-Max Normalization: Scales data between 0 and 1
  def minMaxNormalization(df: DataFrame, column: String): DataFrame = {
    logger.info(s"Applying Min-Max Normalization to column: $column")

    val minVal = df.agg(min(col(column))).first().getDouble(0)
    val maxVal = df.agg(max(col(column))).first().getDouble(0)

    if (minVal == maxVal) {
      logger.warn(s"Column $column has the same min and max values. Returning the original dataframe.")
      return df
    }

    df.withColumn(column, (col(column) - minVal) / (maxVal - minVal))
  }

  // Z-Score Normalization: Standardizes data to have mean 0 and standard deviation 1
  def zScoreNormalization(df: DataFrame, column: String): DataFrame = {
    logger.info(s"Applying Z-Score Normalization to column: $column")

    val meanVal = df.agg(mean(col(column))).first().getDouble(0)
    val stdDevVal = df.agg(stddev(col(column))).first().getDouble(0)

    if (stdDevVal == 0) {
      logger.warn(s"Column $column has a standard deviation of 0. Returning the original dataframe.")
      return df
    }

    df.withColumn(column, (col(column) - meanVal) / stdDevVal)
  }

  // Robust Scaler: Scales data based on the interquartile range (IQR)
  def robustScaler(df: DataFrame, column: String): DataFrame = {
    logger.info(s"Applying Robust Scaler to column: $column")

    val quantiles = df.stat.approxQuantile(column, Array(0.25, 0.75), 0.0)
    val q1 = quantiles(0)
    val q3 = quantiles(1)
    val iqr = q3 - q1

    if (iqr == 0) {
      logger.warn(s"Column $column has IQR of 0. Returning the original dataframe.")
      return df
    }

    df.withColumn(column, (col(column) - q1) / iqr)
  }

  // Log Normalization: Transforms data by taking the log of each value
  def logNormalization(df: DataFrame, column: String): DataFrame = {
    logger.info(s"Applying Log Normalization to column: $column")

    df.withColumn(column, log(col(column) + lit(1))) // Adding 1 to avoid log(0)
  }

  // L2 Normalization: Normalizes data by the L2 norm (Euclidean distance)
  def l2Normalization(df: DataFrame, column: String): DataFrame = {
    logger.info(s"Applying L2 Normalization to column: $column")

    val squaredSum = df.agg(sum(pow(col(column), 2))).first().getDouble(0)
    val l2Norm = math.sqrt(squaredSum)

    if (l2Norm == 0) {
      logger.warn(s"Column $column has an L2 norm of 0. Returning the original dataframe.")
      return df
    }

    df.withColumn(column, col(column) / l2Norm)
  }

  // Utility function to check if a column exists in a DataFrame
  def validateColumnExists(df: DataFrame, column: String): Boolean = {
    val columnExists = df.columns.contains(column)
    if (!columnExists) {
      logger.error(s"Column $column does not exist in the DataFrame.")
    }
    columnExists
  }

  // Dynamic function to normalize multiple columns based on a specified method
  def normalizeData(df: DataFrame, method: String, columns: Seq[String])(implicit spark: SparkSession): DataFrame = {
    columns.foldLeft(df) { (accDf, column) =>
      if (validateColumnExists(accDf, column)) {
        method.toLowerCase match {
          case "minmax"   => minMaxNormalization(accDf, column)
          case "zscore"   => zScoreNormalization(accDf, column)
          case "robust"   => robustScaler(accDf, column)
          case "log"      => logNormalization(accDf, column)
          case "l2"       => l2Normalization(accDf, column)
          case _          => 
            logger.error(s"Unknown normalization method: $method")
            throw new IllegalArgumentException(s"Unknown normalization method: $method")
        }
      } else accDf
    }
  }

  // Method to apply normalization on a grouped DataFrame (e.g., by category)
  def normalizeGroupedData(df: DataFrame, method: String, columns: Seq[String], groupByCol: String)(implicit spark: SparkSession): DataFrame = {
    logger.info(s"Applying $method normalization for grouped data by $groupByCol")
    val window = Window.partitionBy(groupByCol)
    columns.foldLeft(df) { (accDf, column) =>
      if (validateColumnExists(accDf, column)) {
        method.toLowerCase match {
          case "minmax"   => accDf.withColumn(column, (col(column) - min(col(column)).over(window)) / (max(col(column)).over(window) - min(col(column)).over(window)))
          case "zscore"   => accDf.withColumn(column, (col(column) - mean(col(column)).over(window)) / stddev(col(column)).over(window))
          case "robust"   => 
            val q1 = accDf.stat.approxQuantile(column, Array(0.25), 0.0).head
            val q3 = accDf.stat.approxQuantile(column, Array(0.75), 0.0).head
            accDf.withColumn(column, (col(column) - lit(q1)) / (lit(q3) - lit(q1)))
          case _          => 
            logger.error(s"Unknown normalization method: $method for grouped data.")
            throw new IllegalArgumentException(s"Unknown normalization method: $method")
        }
      } else accDf
    }
  }

  // Method to handle missing values before normalization
  def handleMissingValues(df: DataFrame, strategy: String, columns: Seq[String]): DataFrame = {
    logger.info(s"Handling missing values using strategy: $strategy")

    strategy.toLowerCase match {
      case "mean"    => columns.foldLeft(df) { (accDf, column) => accDf.withColumn(column, when(col(column).isNull, mean(col(column)).over(Window.partitionBy())).otherwise(col(column))) }
      case "median"  => columns.foldLeft(df) { (accDf, column) => 
                        val medianVal = accDf.stat.approxQuantile(column, Array(0.5), 0.0).head
                        accDf.withColumn(column, when(col(column).isNull, medianVal).otherwise(col(column)))
                      }
      case "drop"    => df.na.drop(columns)
      case _         => 
        logger.error(s"Unknown missing value handling strategy: $strategy")
        throw new IllegalArgumentException(s"Unknown missing value handling strategy: $strategy")
    }
  }

  // Entry point for normalizing data with missing value handling
  def processNormalization(df: DataFrame, normalizationMethod: String, missingValueStrategy: String, columns: Seq[String])(implicit spark: SparkSession): DataFrame = {
    logger.info("Starting data normalization process")

    val dfHandledMissing = handleMissingValues(df, missingValueStrategy, columns)
    normalizeData(dfHandledMissing, normalizationMethod, columns)
  }

  // For performance monitoring, can log the execution time of a normalization method
  def measureExecutionTime[T](block: => T): T = {
    val startTime = System.currentTimeMillis()
    val result = block
    val endTime = System.currentTimeMillis()
    logger.info(s"Execution time: ${endTime - startTime} ms")
    result
  }

}