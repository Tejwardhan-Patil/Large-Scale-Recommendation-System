package consumers

import org.apache.kafka.clients.consumer.{KafkaConsumer, ConsumerConfig, ConsumerRecords}
import org.apache.kafka.common.serialization.StringDeserializer
import java.util.{Collections, Properties}
import scala.collection.JavaConverters._
import scala.concurrent.Future
import scala.concurrent.ExecutionContext.Implicits.global
import org.json.JSONObject
import java.sql.{Connection, DriverManager, PreparedStatement}
import java.time.LocalDateTime
import scala.util.{Failure, Success, Try}

// Consumer class to handle feedback events
object FeedbackConsumer {

  // Define Kafka consumer properties
  def createConsumer(brokers: String, groupId: String): KafkaConsumer[String, String] = {
    val props = new Properties()
    props.put(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, brokers)
    props.put(ConsumerConfig.GROUP_ID_CONFIG, groupId)
    props.put(ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG, classOf[StringDeserializer].getName)
    props.put(ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG, classOf[StringDeserializer].getName)
    props.put(ConsumerConfig.AUTO_OFFSET_RESET_CONFIG, "earliest") // Start from the earliest message
    props.put(ConsumerConfig.ENABLE_AUTO_COMMIT_CONFIG, "true")    // Automatically commit offsets

    new KafkaConsumer[String, String](props)
  }

  // Method to process the feedback event from JSON
  def processFeedbackEvent(event: String): Unit = {
    val feedback = new JSONObject(event)
    val userId = feedback.getString("user_id")
    val recommendationId = feedback.getString("recommendation_id")
    val feedbackText = feedback.getString("feedback_text")
    val feedbackDate = feedback.getString("feedback_date")

    println(s"Processing feedback from user $userId for recommendation $recommendationId")

    // Retry saving feedback to the database with up to 3 retries
    retry(3) {
      saveFeedbackToDatabase(userId, recommendationId, feedbackText, feedbackDate)
    } match {
      case Success(_) => println(s"Successfully saved feedback for user $userId")
      case Failure(e) => println(s"Failed to save feedback after retries: ${e.getMessage}")
    }

    // Send notification
    sendNotification(userId, recommendationId)
  }

  // Method to save feedback to a database
  def saveFeedbackToDatabase(userId: String, recommendationId: String, feedbackText: String, feedbackDate: String): Unit = {
    val url = "jdbc:postgresql://localhost:5432/feedback_db"
    val username = "dbuser"
    val password = "dbpassword"
    var connection: Connection = null
    var preparedStatement: PreparedStatement = null

    try {
      connection = DriverManager.getConnection(url, username, password)
      val insertSQL = "INSERT INTO feedback (user_id, recommendation_id, feedback_text, feedback_date) VALUES (?, ?, ?, ?)"
      preparedStatement = connection.prepareStatement(insertSQL)
      preparedStatement.setString(1, userId)
      preparedStatement.setString(2, recommendationId)
      preparedStatement.setString(3, feedbackText)
      preparedStatement.setString(4, feedbackDate)
      preparedStatement.executeUpdate()
    } catch {
      case e: Exception => throw new RuntimeException(s"Database error: ${e.getMessage}")
    } finally {
      if (preparedStatement != null) preparedStatement.close()
      if (connection != null) connection.close()
    }
  }

  // Method to consume and process feedback events
  def consumeFeedbackEvents(brokers: String, topic: String, groupId: String): Unit = {
    val consumer = createConsumer(brokers, groupId)
    consumer.subscribe(Collections.singletonList(topic))

    println(s"Subscribed to topic: $topic")

    // Infinite loop to continuously listen for feedback events
    while (true) {
      val records: ConsumerRecords[String, String] = consumer.poll(1000) // Poll every second
      for (record <- records.asScala) {
        println(s"Consumed feedback event: ${record.value()}")
        processFeedbackEvent(record.value())
      }
    }
  }

  // Retry logic to handle failures with a limited number of retries
  def retry[T](n: Int)(fn: => T): Try[T] = {
    Try(fn) match {
      case Success(result) => Success(result)
      case Failure(e) if n > 1 =>
        println(s"Retrying after failure: ${e.getMessage}")
        retry(n - 1)(fn)
      case Failure(e) => Failure(e)
    }
  }

  // Function to send notification about feedback (Slack or Email)
  def sendNotification(userId: String, recommendationId: String): Unit = {
    Future {
      val webhookUrl = "https://hooks.slack.com/services/webhook/url"
      val message = s"New feedback received from user $userId for recommendation $recommendationId"

      Try {
        val payload = s"""{"text": "$message"}"""
        val connection = new java.net.URL(webhookUrl).openConnection().asInstanceOf[java.net.HttpURLConnection]
        connection.setRequestMethod("POST")
        connection.setDoOutput(true)
        val writer = new java.io.OutputStreamWriter(connection.getOutputStream)
        writer.write(payload)
        writer.flush()
        writer.close()
        connection.getResponseCode match {
          case 200 => println(s"Notification sent successfully for feedback on recommendation $recommendationId")
          case _   => println(s"Failed to send notification")
        }
      } match {
        case Success(_) => // Success case already handled
        case Failure(e) => println(s"Notification error: ${e.getMessage}")
      }
    }
  }

  // Health check method to log the consumer status every minute
  def startHealthCheck(): Unit = {
    val scheduler = new java.util.Timer()
    scheduler.scheduleAtFixedRate(new java.util.TimerTask {
      override def run(): Unit = {
        println(s"Health Check: Feedback consumer is running at ${LocalDateTime.now()}")
      }
    }, 0, 60000) // Every 60 seconds
  }

  // Main method to start the feedback consumer
  def main(args: Array[String]): Unit = {
    if (args.length != 3) {
      println("Usage: FeedbackConsumer <brokers> <topic> <groupId>")
      sys.exit(1)
    }

    val brokers = args(0)
    val topic = args(1)
    val groupId = args(2)

    println(s"Starting feedback consumer on topic: $topic with group ID: $groupId")

    // Start health check monitoring
    startHealthCheck()

    // Start consuming feedback events
    consumeFeedbackEvents(brokers, topic, groupId)
  }
}