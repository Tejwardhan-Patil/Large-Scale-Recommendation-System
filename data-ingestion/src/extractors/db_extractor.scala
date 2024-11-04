import java.sql.{Connection, DriverManager, ResultSet, SQLException}
import scala.concurrent.{ExecutionContext, Future}
import scala.util.{Failure, Success, Try}
import java.util.logging.{Level, Logger}

// Database Extractor Class
class DBExtractor(dbUrl: String, dbUser: String, dbPassword: String) {

  private val logger: Logger = Logger.getLogger(classOf[DBExtractor].getName)

  // Initialize JDBC Connection
  def getConnection: Option[Connection] = {
    Try(DriverManager.getConnection(dbUrl, dbUser, dbPassword)) match {
      case Success(connection) => Some(connection)
      case Failure(exception) =>
        logger.log(Level.SEVERE, "Failed to connect to the database", exception)
        None
    }
  }

  // Method to fetch data using a query
  def fetchData(query: String, limit: Int = 1000, offset: Int = 0)(implicit ec: ExecutionContext): Future[List[Map[String, Any]]] = Future {
    val connectionOpt = getConnection
    connectionOpt match {
      case Some(connection) =>
        val paginatedQuery = s"$query LIMIT $limit OFFSET $offset"
        val statement = connection.createStatement()
        val resultSet = statement.executeQuery(paginatedQuery)

        val metadata = resultSet.getMetaData
        val columnCount = metadata.getColumnCount
        var results: List[Map[String, Any]] = List()

        while (resultSet.next()) {
          var row: Map[String, Any] = Map()
          for (i <- 1 to columnCount) {
            row += (metadata.getColumnName(i) -> resultSet.getObject(i))
          }
          results = results :+ row
        }

        resultSet.close()
        statement.close()
        connection.close()

        logger.log(Level.INFO, s"Fetched ${results.size} records")
        results
      case None =>
        logger.log(Level.SEVERE, "No connection available")
        List.empty
    }
  }

  // Method to handle paginated fetch and process large datasets
  def fetchPaginatedData(query: String, batchSize: Int = 1000)(implicit ec: ExecutionContext): Future[Unit] = Future {
    var offset = 0
    var continueFetching = true

    while (continueFetching) {
      val data = fetchData(query, batchSize, offset)
      data.onComplete {
        case Success(result) =>
          processData(result)
          if (result.isEmpty) continueFetching = false
        case Failure(exception) =>
          logger.log(Level.SEVERE, "Failed to fetch paginated data", exception)
          continueFetching = false
      }
      offset += batchSize
    }
  }

  // Method to fetch data concurrently from multiple queries
  def fetchMultipleData(queries: List[String])(implicit ec: ExecutionContext): Future[List[List[Map[String, Any]]]] = {
    logger.log(Level.INFO, s"Starting concurrent fetch for ${queries.size} queries")
    Future.sequence(queries.map(fetchData(_)))
  }

  // Method to create connection pool (optional extension for larger systems)
  def createConnectionPool(poolSize: Int): List[Option[Connection]] = {
    (1 to poolSize).map(_ => getConnection).toList
  }

  // A helper method to process the extracted data (transformation, saving to storage)
  def processData(data: List[Map[String, Any]]): Unit = {
    logger.log(Level.INFO, s"Processing ${data.size} records")
    data.foreach(record => logger.log(Level.FINE, s"Record: $record"))
  }

  // Method to retry fetching data in case of failure (for fault tolerance)
  def retryFetch(query: String, retries: Int = 3)(implicit ec: ExecutionContext): Future[List[Map[String, Any]]] = {
    def attempt(remainingRetries: Int): Future[List[Map[String, Any]]] = {
      fetchData(query).recoverWith {
        case ex: SQLException if remainingRetries > 0 =>
          logger.log(Level.WARNING, s"Retrying query due to failure: $ex, retries left: $remainingRetries")
          attempt(remainingRetries - 1)
        case ex =>
          logger.log(Level.SEVERE, "Failed to fetch data after retries", ex)
          Future.failed(ex)
      }
    }
    attempt(retries)
  }

  // Method to handle database-specific queries (supports different SQL dialects)
  def executeQueryBasedOnDBType(query: String, dbType: String)(implicit ec: ExecutionContext): Future[List[Map[String, Any]]] = {
    val formattedQuery = dbType.toLowerCase match {
      case "mysql" | "postgresql" => query
      case "oracle" => query.replace("LIMIT", "ROWNUM")
      case _ =>
        logger.log(Level.SEVERE, s"Unsupported DB Type: $dbType")
        throw new UnsupportedOperationException(s"DB Type $dbType not supported")
    }
    fetchData(formattedQuery)
  }

  // A method to handle transactions (for batch insert/update)
  def runTransaction(queries: List[String])(implicit ec: ExecutionContext): Future[Unit] = Future {
    val connectionOpt = getConnection
    connectionOpt match {
      case Some(connection) =>
        try {
          connection.setAutoCommit(false)
          val statement = connection.createStatement()
          queries.foreach(query => statement.executeUpdate(query))
          connection.commit()
          logger.log(Level.INFO, "Transaction completed successfully")
        } catch {
          case ex: SQLException =>
            connection.rollback()
            logger.log(Level.SEVERE, "Transaction failed, rolling back", ex)
        } finally {
          connection.close()
        }
      case None => logger.log(Level.SEVERE, "No connection available for transaction")
    }
  }
}

// Usage
object DBExtractorApp extends App {

  // Define ExecutionContext for concurrency
  implicit val ec: ExecutionContext = ExecutionContext.global

  // Database configuration
  val dbUrl = "jdbc:mysql://localhost:3306/mydb"
  val dbUser = "root"
  val dbPassword = "password"

  val extractor = new DBExtractor(dbUrl, dbUser, dbPassword)

  // Query to extract data
  val query = "SELECT * FROM users LIMIT 100"

  // Fetch and process data
  val dataFuture = extractor.fetchData(query)
  dataFuture.onComplete {
    case Success(data) => extractor.processData(data)
    case Failure(exception) => println(s"Failed to fetch data: ${exception.getMessage}")
  }

  // Fetching data concurrently from multiple queries
  val queries = List(
    "SELECT * FROM users WHERE age > 25",
    "SELECT * FROM orders WHERE amount > 1000"
  )

  val multipleDataFuture = extractor.fetchMultipleData(queries)
  multipleDataFuture.onComplete {
    case Success(results) => results.foreach(extractor.processData)
    case Failure(exception) => println(s"Failed to fetch data: ${exception.getMessage}")
  }

  // Handling paginated data
  extractor.fetchPaginatedData(query)

  // Running a transaction
  val transactionQueries = List(
    "UPDATE users SET status = 'active' WHERE id = 1",
    "INSERT INTO logs (message) VALUES ('User 1 activated')"
  )
  extractor.runTransaction(transactionQueries)

  // Keep the main thread alive to wait for futures
  Thread.sleep(10000)
}