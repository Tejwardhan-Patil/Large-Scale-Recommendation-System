package api

import akka.actor.{ActorSystem, Props}
import akka.http.scaladsl.Http
import akka.http.scaladsl.model._
import akka.http.scaladsl.server.Directives._
import akka.http.scaladsl.server.Route
import akka.stream.ActorMaterializer
import scala.concurrent.{ExecutionContextExecutor, Future}
import spray.json._
import akka.util.Timeout
import scala.concurrent.duration._
import akka.pattern.ask
import scala.util.{Failure, Success}
import akka.http.scaladsl.marshallers.sprayjson.SprayJsonSupport._
import akka.http.scaladsl.model.StatusCodes._
import akka.http.scaladsl.server.ExceptionHandler

import inference.Predictor

// JSON support for request and response handling
case class InferenceRequest(data: Seq[Double])
case class InferenceResponse(prediction: Seq[Double])

object JsonSupport extends DefaultJsonProtocol {
  implicit val inferenceRequestFormat = jsonFormat1(InferenceRequest)
  implicit val inferenceResponseFormat = jsonFormat1(InferenceResponse)
}

object InferenceAPI extends App with JsonSupport {

  // Actor system initialization
  implicit val system: ActorSystem = ActorSystem("inference-api")
  implicit val materializer: ActorMaterializer = ActorMaterializer()
  implicit val executionContext: ExecutionContextExecutor = system.dispatcher
  implicit val timeout: Timeout = Timeout(5.seconds)

  val predictor = system.actorOf(Props[PredictorActor], "predictorActor")

  // Error handling for the API
  implicit def myExceptionHandler: ExceptionHandler =
    ExceptionHandler {
      case e: NoSuchElementException =>
        complete(NotFound -> s"Error: ${e.getMessage}")
      case e: IllegalArgumentException =>
        complete(BadRequest -> s"Invalid input: ${e.getMessage}")
      case e: Exception =>
        complete(InternalServerError -> s"An error occurred: ${e.getMessage}")
    }

  // Predictor actor handles the prediction asynchronously
  class PredictorActor extends akka.actor.Actor {
    val predictor = new Predictor()

    def receive: Receive = {
      case data: Seq[Double] =>
        sender() ! predictor.predict(data)
    }
  }

  // Route for prediction handling
  def route: Route =
    handleExceptions(myExceptionHandler) {
      path("predict") {
        post {
          entity(as[InferenceRequest]) { request =>
            val predictionFuture: Future[Seq[Double]] = (predictor ? request.data).mapTo[Seq[Double]]

            onComplete(predictionFuture) {
              case Success(prediction) =>
                complete(InferenceResponse(prediction))
              case Failure(ex) =>
                complete(InternalServerError -> s"Prediction failed: ${ex.getMessage}")
            }
          }
        }
      } ~ path("health") {
        get {
          complete(HttpEntity(ContentTypes.`text/plain(UTF-8)`, "API is running"))
        }
      } ~ path("metrics") {
        get {
          complete(HttpEntity(ContentTypes.`text/plain(UTF-8)`, "No metrics available yet"))
        }
      }
    }

  // Start the HTTP server
  Http().newServerAt("0.0.0.0", 8080).bind(route)

  // Graceful shutdown handling
  sys.addShutdownHook {
    system.terminate()
  }

  println("Inference API server started at http://0.0.0.0:8080/")
}

// JSON structures to handle complex responses or inputs
case class DetailedInferenceRequest(features: Seq[Double], metadata: Map[String, String])
case class DetailedInferenceResponse(prediction: Seq[Double], details: Map[String, String])

object ExtendedJsonSupport extends DefaultJsonProtocol {
  implicit val detailedInferenceRequestFormat = jsonFormat2(DetailedInferenceRequest)
  implicit val detailedInferenceResponseFormat = jsonFormat2(DetailedInferenceResponse)
}

// Enhanced API with additional endpoints and detailed request/response handling
object EnhancedInferenceAPI extends App with ExtendedJsonSupport {

  // Another route to handle more complex input and provide more detailed responses
  def extendedRoute: Route =
    handleExceptions(myExceptionHandler) {
      path("predict-detailed") {
        post {
          entity(as[DetailedInferenceRequest]) { request =>
            val predictionFuture: Future[Seq[Double]] = (predictor ? request.features).mapTo[Seq[Double]]
            val metadata = request.metadata

            onComplete(predictionFuture) {
              case Success(prediction) =>
                val responseDetails = Map(
                  "model_version" -> "1.0",
                  "request_id" -> java.util.UUID.randomUUID().toString,
                  "timestamp" -> java.time.Instant.now().toString
                ) ++ metadata

                complete(DetailedInferenceResponse(prediction, responseDetails))
              case Failure(ex) =>
                complete(InternalServerError -> s"Prediction failed: ${ex.getMessage}")
            }
          }
        }
      } ~ path("status") {
        get {
          complete(HttpEntity(ContentTypes.`application/json`, """{"status": "running"}"""))
        }
      }
    }

  // Extending the original API to include the new routes
  def combinedRoute: Route =
    route ~ extendedRoute

  // Starting the enhanced API server with additional routes
  Http().newServerAt("0.0.0.0", 8081).bind(combinedRoute)

  println("Enhanced Inference API server started at http://0.0.0.0:8081/")
}

// Performance monitoring actor to track API performance and latency
class PerformanceMonitorActor extends akka.actor.Actor {
  var totalRequests: Int = 0
  var successfulRequests: Int = 0
  var failedRequests: Int = 0
  var totalLatency: Long = 0

  def receive: Receive = {
    case "request_received" =>
      totalRequests += 1
    case ("request_successful", latency: Long) =>
      successfulRequests += 1
      totalLatency += latency
    case "request_failed" =>
      failedRequests += 1
  }

  def getMetrics: Map[String, String] = {
    val averageLatency = if (successfulRequests > 0) totalLatency / successfulRequests else 0
    Map(
      "total_requests" -> totalRequests.toString,
      "successful_requests" -> successfulRequests.toString,
      "failed_requests" -> failedRequests.toString,
      "average_latency_ms" -> averageLatency.toString
    )
  }
}

object MetricsAPI extends App {

  // Creating the performance monitor actor
  val monitorActor = system.actorOf(Props[PerformanceMonitorActor], "monitorActor")

  def metricsRoute: Route =
    path("metrics") {
      get {
        val metricsFuture = (monitorActor ? "get_metrics").mapTo[Map[String, String]]

        onComplete(metricsFuture) {
          case Success(metrics) =>
            complete(HttpEntity(ContentTypes.`application/json`, metrics.toJson.toString))
          case Failure(ex) =>
            complete(InternalServerError -> s"Failed to retrieve metrics: ${ex.getMessage}")
        }
      }
    }

  // Start the metrics server
  Http().newServerAt("0.0.0.0", 8082).bind(metricsRoute)

  println("Metrics API server started at http://0.0.0.0:8082/")
}

// Helper functions to track latency for performance monitoring
object LatencyTracker {
  def trackLatency[T](action: => Future[T])(onComplete: Long => Unit): Future[T] = {
    val startTime = System.currentTimeMillis()
    val result = action

    result.onComplete { _ =>
      val latency = System.currentTimeMillis() - startTime
      onComplete(latency)
    }

    result
  }
}

// Adding latency tracking to the prediction endpoint
def routeWithLatencyTracking: Route =
  handleExceptions(myExceptionHandler) {
    path("predict") {
      post {
        entity(as[InferenceRequest]) { request =>
          val predictionFuture: Future[Seq[Double]] = LatencyTracker.trackLatency {
            (predictor ? request.data).mapTo[Seq[Double]]
          } { latency =>
            monitorActor ! ("request_successful", latency)
          }

          onComplete(predictionFuture) {
            case Success(prediction) =>
              complete(InferenceResponse(prediction))
            case Failure(ex) =>
              monitorActor ! "request_failed"
              complete(InternalServerError -> s"Prediction failed: ${ex.getMessage}")
          }
        }
      }
    } ~ path("health") {
      get {
        complete(HttpEntity(ContentTypes.`text/plain(UTF-8)`, "API is running"))
      }
    }
  }

// Route with latency tracking for all endpoints
def combinedRouteWithLatencyTracking: Route =
  handleExceptions(myExceptionHandler) {
    path("predict") {
      post {
        entity(as[InferenceRequest]) { request =>
          val predictionFuture: Future[Seq[Double]] = LatencyTracker.trackLatency {
            (predictor ? request.data).mapTo[Seq[Double]]
          } { latency =>
            monitorActor ! ("request_successful", latency)
          }

          onComplete(predictionFuture) {
            case Success(prediction) =>
              complete(InferenceResponse(prediction))
            case Failure(ex) =>
              monitorActor ! "request_failed"
              complete(InternalServerError -> s"Prediction failed: ${ex.getMessage}")
          }
        }
      }
    } ~ path("predict-detailed") {
      post {
        entity(as[DetailedInferenceRequest]) { request =>
          val predictionFuture: Future[Seq[Double]] = LatencyTracker.trackLatency {
            (predictor ? request.features).mapTo[Seq[Double]]
          } { latency =>
            monitorActor ! ("request_successful", latency)
          }

          val metadata = request.metadata

          onComplete(predictionFuture) {
            case Success(prediction) =>
              val responseDetails = Map(
                "model_version" -> "1.0",
                "request_id" -> java.util.UUID.randomUUID().toString,
                "timestamp" -> java.time.Instant.now().toString
              ) ++ metadata

              complete(DetailedInferenceResponse(prediction, responseDetails))
            case Failure(ex) =>
              monitorActor ! "request_failed"
              complete(InternalServerError -> s"Prediction failed: ${ex.getMessage}")
          }
        }
      }
    } ~ path("metrics") {
      get {
        val metricsFuture = (monitorActor ? "get_metrics").mapTo[Map[String, String]]

        onComplete(metricsFuture) {
          case Success(metrics) =>
            complete(HttpEntity(ContentTypes.`application/json`, metrics.toJson.toString))
          case Failure(ex) =>
            complete(InternalServerError -> s"Failed to retrieve metrics: ${ex.getMessage}")
        }
      }
    } ~ path("health") {
      get {
        complete(HttpEntity(ContentTypes.`text/plain(UTF-8)`, "API is running"))
      }
    }
  }

// Start the server for combined routes with latency tracking
Http().newServerAt("0.0.0.0", 8080).bind(combinedRouteWithLatencyTracking)

// Graceful shutdown handling
sys.addShutdownHook {
  system.terminate()
}

// Additional utility to log prediction requests and responses
object RequestLogger {
  def logRequest(request: InferenceRequest): Unit = {
    println(s"Received prediction request: ${request.data.mkString(", ")}")
  }

  def logResponse(response: InferenceResponse): Unit = {
    println(s"Returning prediction response: ${response.prediction.mkString(", ")}")
  }

  def logDetailedRequest(request: DetailedInferenceRequest): Unit = {
    println(s"Received detailed prediction request with metadata: ${request.metadata}")
  }

  def logDetailedResponse(response: DetailedInferenceResponse): Unit = {
    println(s"Returning detailed prediction response: ${response.prediction.mkString(", ")}, details: ${response.details}")
  }
}

// Wrapping the route with request and response logging
def routeWithLogging: Route =
  handleExceptions(myExceptionHandler) {
    path("predict") {
      post {
        entity(as[InferenceRequest]) { request =>
          RequestLogger.logRequest(request)
          val predictionFuture: Future[Seq[Double]] = LatencyTracker.trackLatency {
            (predictor ? request.data).mapTo[Seq[Double]]
          } { latency =>
            monitorActor ! ("request_successful", latency)
          }

          onComplete(predictionFuture) {
            case Success(prediction) =>
              val response = InferenceResponse(prediction)
              RequestLogger.logResponse(response)
              complete(response)
            case Failure(ex) =>
              monitorActor ! "request_failed"
              complete(InternalServerError -> s"Prediction failed: ${ex.getMessage}")
          }
        }
      }
    } ~ path("predict-detailed") {
      post {
        entity(as[DetailedInferenceRequest]) { request =>
          RequestLogger.logDetailedRequest(request)
          val predictionFuture: Future[Seq[Double]] = LatencyTracker.trackLatency {
            (predictor ? request.features).mapTo[Seq[Double]]
          } { latency =>
            monitorActor ! ("request_successful", latency)
          }

          onComplete(predictionFuture) {
            case Success(prediction) =>
              val responseDetails = Map(
                "model_version" -> "1.0",
                "request_id" -> java.util.UUID.randomUUID().toString,
                "timestamp" -> java.time.Instant.now().toString
              ) ++ request.metadata

              val response = DetailedInferenceResponse(prediction, responseDetails)
              RequestLogger.logDetailedResponse(response)
              complete(response)
            case Failure(ex) =>
              monitorActor ! "request_failed"
              complete(InternalServerError -> s"Prediction failed: ${ex.getMessage}")
          }
        }
      }
    } ~ path("metrics") {
      get {
        val metricsFuture = (monitorActor ? "get_metrics").mapTo[Map[String, String]]

        onComplete(metricsFuture) {
          case Success(metrics) =>
            complete(HttpEntity(ContentTypes.`application/json`, metrics.toJson.toString))
          case Failure(ex) =>
            complete(InternalServerError -> s"Failed to retrieve metrics: ${ex.getMessage}")
        }
      }
    } ~ path("health") {
      get {
        complete(HttpEntity(ContentTypes.`text/plain(UTF-8)`, "API is running"))
      }
    }
  }

// Restart logic for handling system failures
object SystemRestartHandler {
  def restartSystem(): Unit = {
    println("Restarting system due to failure")
    system.terminate()
    system.whenTerminated.onComplete { _ =>
      println("System has terminated, restarting...")
      val newSystem = ActorSystem("inference-api")
      val newMaterializer = ActorMaterializer()(newSystem)
      Http()(newSystem).newServerAt("0.0.0.0", 8080).bind(routeWithLogging)
    }
  }
}

// Implementing system restart logic in case of critical failure
system.scheduler.scheduleOnce(1.hour) {
  SystemRestartHandler.restartSystem()
}

println("Server running with logging and performance monitoring enabled")