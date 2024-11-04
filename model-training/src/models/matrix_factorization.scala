package models

import breeze.linalg._
import breeze.numerics._
import scala.util.Random

class MatrixFactorization(val numUsers: Int, val numItems: Int, val numFactors: Int, val iterations: Int, val learningRate: Double, val regularization: Double) {

  // Randomly initialize user and item latent factor matrices
  val userMatrix: DenseMatrix[Double] = DenseMatrix.rand[Double](numUsers, numFactors, Random)
  val itemMatrix: DenseMatrix[Double] = DenseMatrix.rand[Double](numItems, numFactors, Random)

  // Function to predict ratings for a given user and item
  def predict(user: Int, item: Int): Double = {
    (userMatrix(user, ::) * itemMatrix(item, ::).t).t(0)
  }

  // Function to train the model using stochastic gradient descent
  def train(ratings: List[(Int, Int, Double)]): Unit = {
    for (_ <- 0 until iterations) {
      for ((user, item, rating) <- ratings) {
        val prediction = predict(user, item)
        val error = rating - prediction

        // Update user and item latent factors
        val userLatent = userMatrix(user, ::).t
        val itemLatent = itemMatrix(item, ::).t

        userMatrix(user, ::) := (userLatent + learningRate * (error * itemLatent - regularization * userLatent)).t
        itemMatrix(item, ::) := (itemLatent + learningRate * (error * userLatent - regularization * itemLatent)).t
      }
    }
  }

  // Function to compute the root mean square error (RMSE) of predictions
  def computeRMSE(ratings: List[(Int, Int, Double)]): Double = {
    val squaredErrors = for ((user, item, rating) <- ratings) yield {
      val prediction = predict(user, item)
      pow(rating - prediction, 2)
    }
    sqrt(squaredErrors.sum / squaredErrors.size)
  }

  // Function to generate recommendations for a user
  def recommend(user: Int, topN: Int): List[(Int, Double)] = {
    val itemScores = (0 until numItems).map { item =>
      (item, predict(user, item))
    }.toList
    itemScores.sortBy(-_._2).take(topN)
  }

  // Function to evaluate the model using a test set
  def evaluate(testRatings: List[(Int, Int, Double)]): Double = {
    computeRMSE(testRatings)
  }

  // Function to save the model's latent factors to a file
  def saveModel(userPath: String, itemPath: String): Unit = {
    csvwrite(new java.io.File(userPath), userMatrix)
    csvwrite(new java.io.File(itemPath), itemMatrix)
  }

  // Function to load the model's latent factors from a file
  def loadModel(userPath: String, itemPath: String): Unit = {
    val userMatrixLoaded = csvread(new java.io.File(userPath))
    val itemMatrixLoaded = csvread(new java.io.File(itemPath))
    userMatrix := userMatrixLoaded
    itemMatrix := itemMatrixLoaded
  }

  // Function to print a summary of the model's parameters
  def printSummary(): Unit = {
    println(s"Number of Users: $numUsers")
    println(s"Number of Items: $numItems")
    println(s"Number of Factors: $numFactors")
    println(s"Number of Iterations: $iterations")
    println(s"Learning Rate: $learningRate")
    println(s"Regularization: $regularization")
  }

  // Function to initialize user and item latent factors with a specific seed
  def initializeFactors(seed: Long): Unit = {
    val rand = new Random(seed)
    userMatrix := DenseMatrix.rand[Double](numUsers, numFactors, rand)
    itemMatrix := DenseMatrix.rand[Double](numItems, numFactors, rand)
  }

  // Function to compute the total loss (sum of squared errors + regularization term)
  def computeTotalLoss(ratings: List[(Int, Int, Double)]): Double = {
    val loss = ratings.map { case (user, item, rating) =>
      val prediction = predict(user, item)
      val error = rating - prediction
      pow(error, 2)
    }.sum

    val regularizationTerm = regularization * (sum(userMatrix :* userMatrix) + sum(itemMatrix :* itemMatrix))
    loss + regularizationTerm
  }

  // Function to get the top-N recommendations for all users
  def recommendAllUsers(topN: Int): Map[Int, List[(Int, Double)]] = {
    (0 until numUsers).map { user =>
      user -> recommend(user, topN)
    }.toMap
  }

object MatrixFactorization {
  def apply(numUsers: Int, numItems: Int, numFactors: Int, iterations: Int, learningRate: Double, regularization: Double): MatrixFactorization = {
    new MatrixFactorization(numUsers, numItems, numFactors, iterations, learningRate, regularization)
  }

  // Function to split data into training and testing sets
  def splitData(ratings: List[(Int, Int, Double)], trainRatio: Double): (List[(Int, Int, Double)], List[(Int, Int, Double)]) = {
    val shuffledRatings = Random.shuffle(ratings)
    val trainSize = (shuffledRatings.size * trainRatio).toInt
    val trainSet = shuffledRatings.take(trainSize)
    val testSet = shuffledRatings.drop(trainSize)
    (trainSet, testSet)
  }

  // Function to normalize ratings between 0 and 1
  def normalizeRatings(ratings: List[(Int, Int, Double)]): List[(Int, Int, Double)] = {
    val minRating = ratings.map(_._3).min
    val maxRating = ratings.map(_._3).max
    ratings.map { case (user, item, rating) =>
      (user, item, (rating - minRating) / (maxRating - minRating))
    }
  }

  // Function to denormalize ratings back to the original scale
  def denormalizeRatings(ratings: List[(Int, Int, Double)], minRating: Double, maxRating: Double): List[(Int, Int, Double)] = {
    ratings.map { case (user, item, normalizedRating) =>
      (user, item, normalizedRating * (maxRating - minRating) + minRating)
    }
  }
}

  object MatrixFactorization {
    
    // Function to load ratings from a CSV file
    def loadRatings(filePath: String): List[(Int, Int, Double)] = {
      val source = io.Source.fromFile(filePath)
      val ratings = source.getLines().drop(1).map { line =>
        val Array(user, item, rating) = line.split(",").map(_.trim)
        (user.toInt, item.toInt, rating.toDouble)
      }.toList
      source.close()
      ratings
    }

    // Function to save the predicted ratings to a CSV file
    def savePredictions(predictions: List[(Int, Int, Double)], filePath: String): Unit = {
      val file = new java.io.PrintWriter(new java.io.File(filePath))
      file.write("user,item,predicted_rating\n")
      predictions.foreach { case (user, item, predictedRating) =>
        file.write(s"$user,$item,$predictedRating\n")
      }
      file.close()
    }

    // Function to get a list of the top-N similar items to a given item
    def similarItems(item: Int, topN: Int, itemMatrix: DenseMatrix[Double]): List[(Int, Double)] = {
      val targetItemVector = itemMatrix(item, ::).t
      val similarities = (0 until itemMatrix.rows).map { otherItem =>
        val otherItemVector = itemMatrix(otherItem, ::).t
        val similarity = cosineSimilarity(targetItemVector, otherItemVector)
        (otherItem, similarity)
      }
      similarities.toList.sortBy(-_._2).take(topN)
    }

    // Function to calculate cosine similarity between two vectors
    def cosineSimilarity(vectorA: DenseVector[Double], vectorB: DenseVector[Double]): Double = {
      (vectorA dot vectorB) / (norm(vectorA) * norm(vectorB))
    }

    // Function to recommend items to a user based on similar users' preferences
    def recommendBasedOnSimilarUsers(user: Int, topN: Int, userMatrix: DenseMatrix[Double], ratings: List[(Int, Int, Double)]): List[(Int, Double)] = {
      val similarUsers = (0 until userMatrix.rows).map { otherUser =>
        val similarity = cosineSimilarity(userMatrix(user, ::).t, userMatrix(otherUser, ::).t)
        (otherUser, similarity)
      }.filter(_._1 != user).toList.sortBy(-_._2).take(topN)

      val weightedScores = ratings.groupBy(_._2).map { case (item, itemRatings) =>
        val weightedSum = itemRatings.collect {
          case (otherUser, _, rating) if similarUsers.exists(_._1 == otherUser) =>
            val similarity = similarUsers.find(_._1 == otherUser).get._2
            similarity * rating
        }.sum

        val totalWeight = similarUsers.collect {
          case (otherUser, similarity) if itemRatings.exists(_._1 == otherUser) => similarity
        }.sum

        val score = if (totalWeight > 0) weightedSum / totalWeight else 0.0
        (item, score)
      }

      weightedScores.toList.sortBy(-_._2).take(topN)
    }

    // Function to perform grid search for optimal hyperparameters
    def gridSearch(ratings: List[(Int, Int, Double)], numUsers: Int, numItems: Int, hyperParams: List[(Int, Double, Double)]): (Int, Double, Double, Double) = {
      var bestParams: (Int, Double, Double) = (0, 0.0, 0.0)
      var bestRMSE = Double.MaxValue

      for ((numFactors, learningRate, regularization) <- hyperParams) {
        val model = MatrixFactorization(numUsers, numItems, numFactors, 100, learningRate, regularization)
        val (trainRatings, testRatings) = MatrixFactorization.splitData(ratings, 0.8)
        model.train(trainRatings)
        val rmse = model.computeRMSE(testRatings)
        if (rmse < bestRMSE) {
          bestRMSE = rmse
          bestParams = (numFactors, learningRate, regularization)
        }
      }
      (bestParams._1, bestParams._2, bestParams._3, bestRMSE)
    }

    // Function to perform cross-validation to evaluate the model
    def crossValidate(ratings: List[(Int, Int, Double)], numFolds: Int, numUsers: Int, numItems: Int, numFactors: Int, learningRate: Double, regularization: Double): Double = {
      val foldSize = ratings.size / numFolds
      val shuffledRatings = Random.shuffle(ratings)
      val foldErrors = (0 until numFolds).map { fold =>
        val testFold = shuffledRatings.slice(fold * foldSize, (fold + 1) * foldSize)
        val trainFold = shuffledRatings.take(fold * foldSize) ++ shuffledRatings.drop((fold + 1) * foldSize)
        val model = MatrixFactorization(numUsers, numItems, numFactors, 100, learningRate, regularization)
        model.train(trainFold)
        model.computeRMSE(testFold)
      }
      foldErrors.sum / numFolds
    }

    // Function to calculate the precision at k for the recommendations
    def precisionAtK(user: Int, trueItems: Set[Int], recommendedItems: List[(Int, Double)], k: Int): Double = {
      val recommendedTopK = recommendedItems.take(k).map(_._1).toSet
      val intersection = recommendedTopK.intersect(trueItems).size
      intersection.toDouble / k
    }

    // Function to calculate the recall at k for the recommendations
    def recallAtK(user: Int, trueItems: Set[Int], recommendedItems: List[(Int, Double)], k: Int): Double = {
      val recommendedTopK = recommendedItems.take(k).map(_._1).toSet
      val intersection = recommendedTopK.intersect(trueItems).size
      intersection.toDouble / trueItems.size
    }

    // Function to calculate the Mean Average Precision (MAP) for a set of recommendations
    def meanAveragePrecision(users: List[Int], trueItemsMap: Map[Int, Set[Int]], recommendationMap: Map[Int, List[(Int, Double)]], k: Int): Double = {
      val avgPrecisions = users.map { user =>
        val trueItems = trueItemsMap.getOrElse(user, Set.empty[Int])
        val recommendedItems = recommendationMap.getOrElse(user, List.empty)
        val precisions = (1 to k).map { i =>
          precisionAtK(user, trueItems, recommendedItems, i)
        }
        precisions.sum / k
      }
      avgPrecisions.sum / users.size
    }

    // Function to calculate the Mean Reciprocal Rank (MRR) for a set of recommendations
    def meanReciprocalRank(users: List[Int], trueItemsMap: Map[Int, Set[Int]], recommendationMap: Map[Int, List[(Int, Double)]]): Double = {
      val reciprocalRanks = users.map { user =>
        val trueItems = trueItemsMap.getOrElse(user, Set.empty[Int])
        val recommendedItems = recommendationMap.getOrElse(user, List.empty)
        recommendedItems.zipWithIndex.collectFirst {
          case (item, rank) if trueItems.contains(item) => 1.0 / (rank + 1)
        }.getOrElse(0.0)
      }
      reciprocalRanks.sum / users.size
    }
  }
}