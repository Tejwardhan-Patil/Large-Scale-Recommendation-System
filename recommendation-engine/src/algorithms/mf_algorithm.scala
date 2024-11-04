package recommendationengine.algorithms

import breeze.linalg.{DenseMatrix, DenseVector}
import scala.util.Random

class MFAlgorithm(rank: Int, iterations: Int, learningRate: Double, regularization: Double) {
  
  // Initialization of matrices
  def initializeMatrices(numUsers: Int, numItems: Int): (DenseMatrix[Double], DenseMatrix[Double]) = {
    val userMatrix = DenseMatrix.rand(numUsers, rank, Random.nextDouble)
    val itemMatrix = DenseMatrix.rand(numItems, rank, Random.nextDouble)
    (userMatrix, itemMatrix)
  }

  // Train the Matrix Factorization model
  def train(
    ratings: DenseMatrix[Double], 
    numUsers: Int, 
    numItems: Int
  ): (DenseMatrix[Double], DenseMatrix[Double]) = {
    
    var (userMatrix, itemMatrix) = initializeMatrices(numUsers, numItems)
    
    for (iter <- 0 until iterations) {
      for (i <- 0 until numUsers) {
        for (j <- 0 until numItems) {
          val rating = ratings(i, j)
          if (rating > 0) {
            val prediction = predictSingle(userMatrix(i, ::).t, itemMatrix(j, ::).t)
            val error = rating - prediction
            updateParameters(i, j, error, userMatrix, itemMatrix)
          }
        }
      }
    }
    
    (userMatrix, itemMatrix)
  }

  // Predict a single user-item pair
  def predictSingle(user: DenseVector[Double], item: DenseVector[Double]): Double = {
    user.dot(item)
  }

  // Update user and item matrices with stochastic gradient descent
  def updateParameters(
    userIndex: Int, 
    itemIndex: Int, 
    error: Double, 
    userMatrix: DenseMatrix[Double], 
    itemMatrix: DenseMatrix[Double]
  ): Unit = {
    
    for (k <- 0 until rank) {
      val userFactor = userMatrix(userIndex, k)
      val itemFactor = itemMatrix(itemIndex, k)
      
      userMatrix(userIndex, k) += learningRate * (2 * error * itemFactor - regularization * userFactor)
      itemMatrix(itemIndex, k) += learningRate * (2 * error * userFactor - regularization * itemFactor)
    }
  }

  // Make predictions for all user-item pairs
  def predict(userMatrix: DenseMatrix[Double], itemMatrix: DenseMatrix[Double]): DenseMatrix[Double] = {
    userMatrix * itemMatrix.t
  }

  // Compute root mean squared error (RMSE) for the model's performance
  def computeRMSE(
    ratings: DenseMatrix[Double], 
    predictions: DenseMatrix[Double]
  ): Double = {
    
    var sumSquaredError = 0.0
    var count = 0
    
    for (i <- 0 until ratings.rows) {
      for (j <- 0 until ratings.cols) {
        if (ratings(i, j) > 0) {
          val error = ratings(i, j) - predictions(i, j)
          sumSquaredError += error * error
          count += 1
        }
      }
    }
    
    math.sqrt(sumSquaredError / count)
  }

  // Method to evaluate the model
  def evaluateModel(
    ratings: DenseMatrix[Double], 
    numUsers: Int, 
    numItems: Int
  ): Unit = {
    
    val (userMatrix, itemMatrix) = train(ratings, numUsers, numItems)
    val predictions = predict(userMatrix, itemMatrix)
    val rmse = computeRMSE(ratings, predictions)
    
    println(s"RMSE: $rmse")
  }
  
  // Additional utility methods for future expansion

  // Generate a random ratings matrix (for testing purposes)
  def generateRandomRatings(numUsers: Int, numItems: Int): DenseMatrix[Double] = {
    val randomRatings = DenseMatrix.zeros[Double](numUsers, numItems)
    
    for (i <- 0 until numUsers) {
      for (j <- 0 until numItems) {
        randomRatings(i, j) = Random.nextInt(5).toDouble
      }
    }
    
    randomRatings
  }

  // Apply regularization to user and item matrices
  def applyRegularization(userMatrix: DenseMatrix[Double], itemMatrix: DenseMatrix[Double]): Unit = {
    for (i <- 0 until userMatrix.rows) {
      for (k <- 0 until rank) {
        userMatrix(i, k) -= regularization * userMatrix(i, k)
      }
    }

    for (j <- 0 until itemMatrix.rows) {
      for (k <- 0 until rank) {
        itemMatrix(j, k) -= regularization * itemMatrix(j, k)
      }
    }
  }

  // Train using alternating least squares (ALS)
  def trainALS(
    ratings: DenseMatrix[Double], 
    numUsers: Int, 
    numItems: Int
  ): (DenseMatrix[Double], DenseMatrix[Double]) = {
    
    var (userMatrix, itemMatrix) = initializeMatrices(numUsers, numItems)

    for (iter <- 0 until iterations) {
      for (i <- 0 until numUsers) {
        val userRatings = ratings(i, ::).t
        val itemPredictions = itemMatrix * itemMatrix.t * userMatrix(i, ::).t
        userMatrix(i, ::) := itemPredictions.t
      }

      for (j <- 0 until numItems) {
        val itemRatings = ratings(::, j)
        val userPredictions = userMatrix * userMatrix.t * itemMatrix(j, ::).t
        itemMatrix(j, ::) := userPredictions.t
      }

      applyRegularization(userMatrix, itemMatrix)
    }

    (userMatrix, itemMatrix)
  }
}

  // Method to compute the loss function for gradient descent
  def computeLoss(
    ratings: DenseMatrix[Double],
    userMatrix: DenseMatrix[Double],
    itemMatrix: DenseMatrix[Double]
  ): Double = {
    var loss = 0.0
    for (i <- 0 until ratings.rows) {
      for (j <- 0 until ratings.cols) {
        if (ratings(i, j) > 0) {
          val prediction = predictSingle(userMatrix(i, ::).t, itemMatrix(j, ::).t)
          val error = ratings(i, j) - prediction
          loss += error * error
        }
      }
    }
    loss
  }

  // A function to compute the gradient for user and item matrices
  def computeGradients(
    ratings: DenseMatrix[Double],
    userMatrix: DenseMatrix[Double],
    itemMatrix: DenseMatrix[Double]
  ): (DenseMatrix[Double], DenseMatrix[Double]) = {

    val userGradient = DenseMatrix.zeros[Double](userMatrix.rows, userMatrix.cols)
    val itemGradient = DenseMatrix.zeros[Double](itemMatrix.rows, itemMatrix.cols)

    for (i <- 0 until ratings.rows) {
      for (j <- 0 until ratings.cols) {
        if (ratings(i, j) > 0) {
          val prediction = predictSingle(userMatrix(i, ::).t, itemMatrix(j, ::).t)
          val error = ratings(i, j) - prediction

          for (k <- 0 until rank) {
            userGradient(i, k) += -2 * error * itemMatrix(j, k)
            itemGradient(j, k) += -2 * error * userMatrix(i, k)
          }
        }
      }
    }

    (userGradient, itemGradient)
  }

  // Update the matrices based on the gradients and learning rate
  def applyGradients(
    userMatrix: DenseMatrix[Double],
    itemMatrix: DenseMatrix[Double],
    userGradient: DenseMatrix[Double],
    itemGradient: DenseMatrix[Double]
  ): Unit = {
    
    for (i <- 0 until userMatrix.rows) {
      for (k <- 0 until rank) {
        userMatrix(i, k) -= learningRate * userGradient(i, k)
      }
    }

    for (j <- 0 until itemMatrix.rows) {
      for (k <- 0 until rank) {
        itemMatrix(j, k) -= learningRate * itemGradient(j, k)
      }
    }
  }

  // Perform matrix factorization using gradient descent
  def matrixFactorizationGD(
    ratings: DenseMatrix[Double], 
    numUsers: Int, 
    numItems: Int
  ): (DenseMatrix[Double], DenseMatrix[Double]) = {
    
    var (userMatrix, itemMatrix) = initializeMatrices(numUsers, numItems)

    for (iter <- 0 until iterations) {
      val lossBefore = computeLoss(ratings, userMatrix, itemMatrix)
      
      val (userGradient, itemGradient) = computeGradients(ratings, userMatrix, itemMatrix)
      applyGradients(userMatrix, itemMatrix, userGradient, itemGradient)
      
      val lossAfter = computeLoss(ratings, userMatrix, itemMatrix)

      println(s"Iteration: $iter, Loss Before: $lossBefore, Loss After: $lossAfter")
    }

    (userMatrix, itemMatrix)
  }

  // Method for making recommendations for a specific user
  def recommendForUser(
    userId: Int, 
    userMatrix: DenseMatrix[Double], 
    itemMatrix: DenseMatrix[Double], 
    topN: Int
  ): Seq[(Int, Double)] = {
    
    val userVector = userMatrix(userId, ::).t
    val scores = (0 until itemMatrix.rows).map { j =>
      val itemVector = itemMatrix(j, ::).t
      val score = predictSingle(userVector, itemVector)
      (j, score)
    }

    scores.sortBy(-_._2).take(topN)
  }

  // Method for recommending items for all users
  def recommendForAllUsers(
    userMatrix: DenseMatrix[Double], 
    itemMatrix: DenseMatrix[Double], 
    topN: Int
  ): Map[Int, Seq[(Int, Double)]] = {
    
    (0 until userMatrix.rows).map { userId =>
      userId -> recommendForUser(userId, userMatrix, itemMatrix, topN)
    }.toMap
  }

  // A method for recommending users for a given item
  def recommendForItem(
    itemId: Int, 
    userMatrix: DenseMatrix[Double], 
    itemMatrix: DenseMatrix[Double], 
    topN: Int
  ): Seq[(Int, Double)] = {
    
    val itemVector = itemMatrix(itemId, ::).t
    val scores = (0 until userMatrix.rows).map { i =>
      val userVector = userMatrix(i, ::).t
      val score = predictSingle(userVector, itemVector)
      (i, score)
    }

    scores.sortBy(-_._2).take(topN)
  }

  // Method for recommending users for all items
  def recommendForAllItems(
    userMatrix: DenseMatrix[Double], 
    itemMatrix: DenseMatrix[Double], 
    topN: Int
  ): Map[Int, Seq[(Int, Double)]] = {
    
    (0 until itemMatrix.rows).map { itemId =>
      itemId -> recommendForItem(itemId, userMatrix, itemMatrix, topN)
    }.toMap
  }

  // Calculate the Mean Absolute Error (MAE) of the predictions
  def computeMAE(
    ratings: DenseMatrix[Double], 
    predictions: DenseMatrix[Double]
  ): Double = {
    
    var sumAbsoluteError = 0.0
    var count = 0
    
    for (i <- 0 until ratings.rows) {
      for (j <- 0 until ratings.cols) {
        if (ratings(i, j) > 0) {
          val error = math.abs(ratings(i, j) - predictions(i, j))
          sumAbsoluteError += error
          count += 1
        }
      }
    }
    
    sumAbsoluteError / count
  }

  // Save the user and item matrices to a file
  def saveModel(userMatrix: DenseMatrix[Double], itemMatrix: DenseMatrix[Double], userFile: String, itemFile: String): Unit = {
    breeze.linalg.csvwrite(new java.io.File(userFile), userMatrix)
    breeze.linalg.csvwrite(new java.io.File(itemFile), itemMatrix)
  }

  // Load user and item matrices from files
  def loadModel(userFile: String, itemFile: String): (DenseMatrix[Double], DenseMatrix[Double]) = {
    val userMatrix = breeze.linalg.csvread(new java.io.File(userFile))
    val itemMatrix = breeze.linalg.csvread(new java.io.File(itemFile))
    (userMatrix, itemMatrix)
  }

  // Normalize the ratings matrix
  def normalizeRatings(ratings: DenseMatrix[Double]): DenseMatrix[Double] = {
    val meanRatings = ratings(::, *).map(row => mean(row))
    ratings - meanRatings
  }

  // Method to denormalize the predicted ratings
  def denormalizeRatings(predictions: DenseMatrix[Double], originalRatings: DenseMatrix[Double]): DenseMatrix[Double] = {
    val meanRatings = originalRatings(::, *).map(row => mean(row))
    predictions + meanRatings
  }

  // Perform weighted matrix factorization, incorporating implicit feedback
  def weightedMatrixFactorization(
    ratings: DenseMatrix[Double], 
    confidence: DenseMatrix[Double], 
    numUsers: Int, 
    numItems: Int
  ): (DenseMatrix[Double], DenseMatrix[Double]) = {
    
    var (userMatrix, itemMatrix) = initializeMatrices(numUsers, numItems)

    for (iter <- 0 until iterations) {
      for (i <- 0 until numUsers) {
        for (j <- 0 until numItems) {
          val rating = ratings(i, j)
          val confidenceValue = confidence(i, j)
          
          if (rating > 0) {
            val prediction = predictSingle(userMatrix(i, ::).t, itemMatrix(j, ::).t)
            val error = rating - prediction
            
            for (k <- 0 until rank) {
              userMatrix(i, k) += learningRate * (confidenceValue * error * itemMatrix(j, k) - regularization * userMatrix(i, k))
              itemMatrix(j, k) += learningRate * (confidenceValue * error * userMatrix(i, k) - regularization * itemMatrix(j, k))
            }
          }
        }
      }
      println(s"Iteration: $iter completed for weighted matrix factorization.")
    }

    (userMatrix, itemMatrix)
  }

  // Apply biases to the model (user and item biases)
  def applyBiases(
    ratings: DenseMatrix[Double], 
    numUsers: Int, 
    numItems: Int
  ): (DenseVector[Double], DenseVector[Double]) = {
    
    val userBiases = DenseVector.zeros[Double](numUsers)
    val itemBiases = DenseVector.zeros[Double](numItems)
    
    for (i <- 0 until numUsers) {
      userBiases(i) = mean(ratings(i, ::))
    }
    
    for (j <- 0 until numItems) {
      itemBiases(j) = mean(ratings(::, j))
    }
    
    (userBiases, itemBiases)
  }

  // Make predictions considering biases
  def predictWithBiases(
    userMatrix: DenseMatrix[Double], 
    itemMatrix: DenseMatrix[Double], 
    userBiases: DenseVector[Double], 
    itemBiases: DenseVector[Double], 
    globalBias: Double
  ): DenseMatrix[Double] = {
    
    val predictions = DenseMatrix.zeros[Double](userMatrix.rows, itemMatrix.rows)

    for (i <- 0 until userMatrix.rows) {
      for (j <- 0 until itemMatrix.rows) {
        predictions(i, j) = globalBias + userBiases(i) + itemBiases(j) + predictSingle(userMatrix(i, ::).t, itemMatrix(j, ::).t)
      }
    }

    predictions
  }

  // Calculate the global bias (mean rating) for the dataset
  def calculateGlobalBias(ratings: DenseMatrix[Double]): Double = {
    var totalRating = 0.0
    var count = 0

    for (i <- 0 until ratings.rows) {
      for (j <- 0 until ratings.cols) {
        if (ratings(i, j) > 0) {
          totalRating += ratings(i, j)
          count += 1
        }
      }
    }
    
    totalRating / count
  }

  // A method to adjust learning rate dynamically during training
  def adjustLearningRate(iteration: Int, decayFactor: Double): Unit = {
    learningRate = learningRate / (1 + decayFactor * iteration)
  }

  // Train the model with dynamic learning rate adjustment
  def trainWithDynamicLearningRate(
    ratings: DenseMatrix[Double], 
    numUsers: Int, 
    numItems: Int, 
    decayFactor: Double
  ): (DenseMatrix[Double], DenseMatrix[Double]) = {
    
    var (userMatrix, itemMatrix) = initializeMatrices(numUsers, numItems)

    for (iter <- 0 until iterations) {
      adjustLearningRate(iter, decayFactor)

      for (i <- 0 until numUsers) {
        for (j <- 0 until numItems) {
          val rating = ratings(i, j)
          if (rating > 0) {
            val prediction = predictSingle(userMatrix(i, ::).t, itemMatrix(j, ::).t)
            val error = rating - prediction
            updateParameters(i, j, error, userMatrix, itemMatrix)
          }
        }
      }

      val currentLoss = computeLoss(ratings, userMatrix, itemMatrix)
      println(s"Iteration: $iter, Current Loss: $currentLoss, Learning Rate: $learningRate")
    }

    (userMatrix, itemMatrix)
  }

  // Generate a heatmap of the ratings matrix for analysis
  def generateHeatmap(matrix: DenseMatrix[Double]): Unit = {
    val plot = breeze.plot.Figure()
    val p = plot.subplot(0)
    p += breeze.plot.image(matrix)
    plot.saveas("heatmap.png")
  }

  // Add regularization to the gradient descent method
  def gradientDescentWithRegularization(
    ratings: DenseMatrix[Double], 
    numUsers: Int, 
    numItems: Int
  ): (DenseMatrix[Double], DenseMatrix[Double]) = {
    
    var (userMatrix, itemMatrix) = initializeMatrices(numUsers, numItems)

    for (iter <- 0 until iterations) {
      for (i <- 0 until numUsers) {
        for (j <- 0 until numItems) {
          val rating = ratings(i, j)
          if (rating > 0) {
            val prediction = predictSingle(userMatrix(i, ::).t, itemMatrix(j, ::).t)
            val error = rating - prediction
            
            for (k <- 0 until rank) {
              userMatrix(i, k) += learningRate * (error * itemMatrix(j, k) - regularization * userMatrix(i, k))
              itemMatrix(j, k) += learningRate * (error * userMatrix(i, k) - regularization * itemMatrix(j, k))
            }
          }
        }
      }
      
      val loss = computeLoss(ratings, userMatrix, itemMatrix)
      println(s"Iteration: $iter, Loss: $loss")
    }

    (userMatrix, itemMatrix)
  }

  // Tune the regularization parameter dynamically
  def tuneRegularizationParameter(
    ratings: DenseMatrix[Double], 
    numUsers: Int, 
    numItems: Int, 
    initialReg: Double, 
    finalReg: Double
  ): (DenseMatrix[Double], DenseMatrix[Double]) = {
    
    var reg = initialReg
    var (userMatrix, itemMatrix) = initializeMatrices(numUsers, numItems)

    for (iter <- 0 until iterations) {
      reg = initialReg + (finalReg - initialReg) * (iter / iterations.toDouble)

      for (i <- 0 until numUsers) {
        for (j <- 0 until numItems) {
          val rating = ratings(i, j)
          if (rating > 0) {
            val prediction = predictSingle(userMatrix(i, ::).t, itemMatrix(j, ::).t)
            val error = rating - prediction
            
            for (k <- 0 until rank) {
              userMatrix(i, k) += learningRate * (error * itemMatrix(j, k) - reg * userMatrix(i, k))
              itemMatrix(j, k) += learningRate * (error * userMatrix(i, k) - reg * itemMatrix(j, k))
            }
          }
        }
      }

      val loss = computeLoss(ratings, userMatrix, itemMatrix)
      println(s"Iteration: $iter, Loss: $loss, Regularization: $reg")
    }

    (userMatrix, itemMatrix)
  }

  // A method to visualize the predicted vs actual ratings
  def visualizePredictions(predictions: DenseMatrix[Double], actual: DenseMatrix[Double]): Unit = {
    val plot = breeze.plot.Figure()
    val p = plot.subplot(0)
    p += breeze.plot.plot(predictions.toDenseVector, actual.toDenseVector, '+')
    plot.saveas("predictions_vs_actual.png")
  }


// Akka HTTP service
import akka.actor.ActorSystem
import akka.http.scaladsl.Http
import akka.http.scaladsl.model._
import akka.http.scaladsl.server.Directives._
import akka.stream.ActorMaterializer
import scala.io.StdIn

object MFService {
  def main(args: Array[String]): Unit = {
    implicit val system = ActorSystem("matrix-factorization-system")
    implicit val materializer = ActorMaterializer()
    implicit val executionContext = system.dispatcher

    val route =
      path("mf_recommend") {
        post {
          entity(as[String]) { userId =>
            val recommendations = mfAlgorithm.recommend(userId.toInt) 
            complete(HttpEntity(ContentTypes.`application/json`, recommendations.toJson))  // Convert result to JSON
          }
        }
      }

    val bindingFuture = Http().bindAndHandle(route, "0.0.0.0", 5002)

    println(s"Server online at http://localhost:5002/\nPress RETURN to stop...")
    StdIn.readLine()
    bindingFuture
      .flatMap(_.unbind())
      .onComplete(_ => system.terminate())
  }
}