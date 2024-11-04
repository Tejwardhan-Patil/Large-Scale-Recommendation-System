package scaling

import scala.concurrent.duration._
import akka.actor.{ Actor, ActorSystem, Props }
import akka.pattern.ask
import akka.util.Timeout
import scala.concurrent.{ ExecutionContext, Future }
import scala.sys.process._
import scala.language.postfixOps
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

// AutoScaler class to manage scaling based on resource usage
class AutoScaler extends Actor {
  
  implicit val ec: ExecutionContext = context.system.dispatcher
  implicit val timeout: Timeout = Timeout(5.seconds)

  // Thresholds for scaling
  val cpuThreshold: Double = 0.75
  val memoryThreshold: Double = 0.80
  val checkInterval: FiniteDuration = 10.seconds

  // Replica settings for scaling actions
  val minReplicas: Int = 1
  val maxReplicas: Int = 10

  // Current number of replicas
  var currentReplicas: Int = 1

  // Scaling cooldown settings to avoid rapid scaling
  val scaleCooldown: FiniteDuration = 60.seconds
  var lastScaleTime: Option[LocalDateTime] = None

  // Commands to scale the system
  def scaleUp(): Unit = {
    if (canScale()) {
      if (currentReplicas < maxReplicas) {
        currentReplicas += 1
        val scaleUpCmd = s"kubectl scale deployment model-serving --replicas=$currentReplicas"
        scaleUpCmd.!
        logAction(s"Scaling up to $currentReplicas replicas")
        lastScaleTime = Some(LocalDateTime.now())
      }
    }
  }

  def scaleDown(): Unit = {
    if (canScale()) {
      if (currentReplicas > minReplicas) {
        currentReplicas -= 1
        val scaleDownCmd = s"kubectl scale deployment model-serving --replicas=$currentReplicas"
        scaleDownCmd.!
        logAction(s"Scaling down to $currentReplicas replicas")
        lastScaleTime = Some(LocalDateTime.now())
      }
    }
  }

  // Check if the system can scale (cooldown period between scaling actions)
  def canScale(): Boolean = {
    lastScaleTime match {
      case Some(lastTime) =>
        val now = LocalDateTime.now()
        val diff = java.time.Duration.between(lastTime, now).toSeconds
        diff >= scaleCooldown.toSeconds
      case None => true
    }
  }

  // Log scaling actions with timestamps
  def logAction(action: String): Unit = {
    val formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")
    val now = LocalDateTime.now().format(formatter)
    println(s"[$now] $action")
  }

  // Method to check system resource usage
  def checkResourceUsage(): Future[(Double, Double)] = Future {
    val cpuUsage = "top -bn1 | grep 'Cpu(s)' | sed 's/.*, *\\([0-9.]*\\)%* id.*/\\1/' | awk '{print 100 - $1}'".!!.toDouble / 100
    val memoryUsage = "free | grep Mem | awk '{print $3/$2}'".!!.toDouble
    logResourceUsage(cpuUsage, memoryUsage)
    (cpuUsage, memoryUsage)
  }

  // Log resource usage with timestamps
  def logResourceUsage(cpu: Double, memory: Double): Unit = {
    val formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")
    val now = LocalDateTime.now().format(formatter)
    println(s"[$now] CPU Usage: ${(cpu * 100).formatted("%.2f")}% | Memory Usage: ${(memory * 100).formatted("%.2f")}%")
  }

  // Method to scale based on resource usage
  def handleScaling(cpuUsage: Double, memoryUsage: Double): Unit = {
    if (cpuUsage > cpuThreshold || memoryUsage > memoryThreshold) {
      scaleUp()
    } else if (cpuUsage < cpuThreshold && memoryUsage < memoryThreshold) {
      scaleDown()
    } else {
      logAction("No scaling action required")
    }
  }

  // Actor receive block for handling messages
  override def receive: Receive = {
    case "check" =>
      checkResourceUsage().map {
        case (cpu, memory) => handleScaling(cpu, memory)
      }
  }
}

// Companion object for running the AutoScaler
object AutoScalerApp extends App {
  
  // Initialize the Actor system and AutoScaler
  val system = ActorSystem("AutoScalerSystem")
  val autoScaler = system.actorOf(Props[AutoScaler], "autoScaler")

  // Schedule periodic resource usage checks
  system.scheduler.scheduleAtFixedRate(0.seconds, 10.seconds)(() => autoScaler ! "check")

  // Define shutdown behavior
  sys.addShutdownHook {
    println("Shutting down the AutoScaler system...")
    system.terminate()
  }
}

// Additional helper class for future scaling strategies
class AdvancedAutoScaler extends AutoScaler {

  // Override scaleUp to implement advanced scaling strategy
  override def scaleUp(): Unit = {
    if (canScale()) {
      if (currentReplicas < maxReplicas) {
        // Increase the replicas more aggressively based on usage
        val additionalReplicas = calculateAdditionalReplicas()
        currentReplicas = Math.min(currentReplicas + additionalReplicas, maxReplicas)
        val scaleUpCmd = s"kubectl scale deployment model-serving --replicas=$currentReplicas"
        scaleUpCmd.!
        logAction(s"Advanced scaling up to $currentReplicas replicas")
        lastScaleTime = Some(LocalDateTime.now())
      }
    }
  }

  // Calculate additional replicas based on advanced logic
  def calculateAdditionalReplicas(cpuUsage: Double, requestLoad: Int): Int = {
    val cpuThreshold = 0.75   // Threshold for CPU usage (75%)
    val loadThreshold = 1000  // Threshold for request load

    // Logic to scale based on CPU and request load
    if (cpuUsage > cpuThreshold && requestLoad > loadThreshold) {
      3  // Add 3 replicas if both thresholds are exceeded
    } else if (cpuUsage > cpuThreshold || requestLoad > loadThreshold) {
      2  // Add 2 replicas if either threshold is exceeded
    } else {
      1  // Add 1 replica as a minimum scaling
    }
  }
}

// Class for system health check and alerting
class SystemHealthChecker {

  // Check the overall health of the system
  def checkSystemHealth(): Unit = {
    val systemHealthCmd = "kubectl get pods"
    val result = systemHealthCmd.!!
    println(s"System Health:\n$result")
  }

  // Trigger alerts if necessary
  def triggerAlerts(): Unit = {
    val errorLogCmd = "kubectl logs --tail=10"
    val logs = errorLogCmd.!!
    if (logs.contains("error")) {
      sendAlert(s"System error detected:\n$logs")
    }
  }

  // Send alert to monitoring system
  def sendAlert(message: String): Unit = {
    println(s"ALERT: $message")
    // Implementation of sending the alert (e.g., via API)
  }
}

// Periodically run health checks
object HealthCheckerApp extends App {

  val healthChecker = new SystemHealthChecker

  // Schedule periodic health checks every minute
  val system = ActorSystem("HealthCheckerSystem")
  system.scheduler.scheduleAtFixedRate(0.seconds, 60.seconds)(() => healthChecker.checkSystemHealth())

  // Define shutdown behavior
  sys.addShutdownHook {
    println("Shutting down the HealthChecker system...")
    system.terminate()
  }
}