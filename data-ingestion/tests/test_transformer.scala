import org.scalatest.FunSuite
import scala.util.{Success, Failure}

// Import necessary classes from the project
import transformers.Normalization

class TransformerTest extends FunSuite {

  // Nrmalization tests
  test("Test normalization of a single numeric value") {
    val input = 150.0
    val result = Normalization.normalize(input, 0.0, 200.0)
    assert(result === 0.75)
  }

  test("Test normalization of a small numeric value") {
    val input = 25.0
    val result = Normalization.normalize(input, 0.0, 100.0)
    assert(result === 0.25)
  }

  test("Test normalization with a zero range") {
    val input = 100.0
    val result = Normalization.normalize(input, 100.0, 100.0)
    assert(result === 0.0)
  }

  test("Test normalization with negative values in range") {
    val input = -50.0
    val result = Normalization.normalize(input, -100.0, 0.0)
    assert(result === 0.5)
  }

  test("Test normalization at lower bound of range") {
    val input = 0.0
    val result = Normalization.normalize(input, 0.0, 100.0)
    assert(result === 0.0)
  }

  test("Test normalization at upper bound of range") {
    val input = 100.0
    val result = Normalization.normalize(input, 0.0, 100.0)
    assert(result === 1.0)
  }

  // Invalid input tests
  test("Test normalization with invalid range") {
    val input = 75.0
    val exception = intercept[IllegalArgumentException] {
      Normalization.normalize(input, 200.0, 100.0)
    }
    assert(exception.getMessage === "Invalid range: min must be less than max")
  }

  test("Test normalization with NaN input") {
    val input = Double.NaN
    val result = Normalization.normalize(input, 0.0, 100.0)
    assert(result.isNaN)
  }

  test("Test normalization with infinity input") {
    val input = Double.PositiveInfinity
    val result = Normalization.normalize(input, 0.0, 100.0)
    assert(result === 1.0)
  }

  test("Test normalization with negative infinity input") {
    val input = Double.NegativeInfinity
    val result = Normalization.normalize(input, 0.0, 100.0)
    assert(result === 0.0)
  }

  test("Test normalization with very large values") {
    val input = 1e10
    val result = Normalization.normalize(input, 0.0, 1e10)
    assert(result === 1.0)
  }

  test("Test normalization with very small values") {
    val input = 1e-10
    val result = Normalization.normalize(input, 0.0, 1e-10)
    assert(result === 1.0)
  }

  // Batch normalization tests
  test("Test batch normalization for a list of values") {
    val inputList = List(100.0, 50.0, 150.0)
    val expectedResults = List(0.5, 0.0, 1.0)
    val result = Normalization.batchNormalize(inputList, 50.0, 150.0)
    assert(result === expectedResults)
  }

  test("Test batch normalization with empty list") {
    val inputList = List.empty[Double]
    val result = Normalization.batchNormalize(inputList, 0.0, 100.0)
    assert(result.isEmpty)
  }

  test("Test batch normalization with a list of negative values") {
    val inputList = List(-10.0, -20.0, -30.0)
    val expectedResults = List(1.0, 0.5, 0.0)
    val result = Normalization.batchNormalize(inputList, -30.0, -10.0)
    assert(result === expectedResults)
  }

  test("Test batch normalization with mixed positive and negative values") {
    val inputList = List(-10.0, 0.0, 10.0)
    val expectedResults = List(0.0, 0.5, 1.0)
    val result = Normalization.batchNormalize(inputList, -10.0, 10.0)
    assert(result === expectedResults)
  }

  test("Test batch normalization with large dataset") {
    val inputList = (1 to 100000).map(_.toDouble).toList
    val result = Normalization.batchNormalize(inputList, 1.0, 100000.0)
    assert(result.head === 0.0)
    assert(result.last === 1.0)
  }

  test("Test batch normalization with all zero values") {
    val inputList = List(0.0, 0.0, 0.0)
    val result = Normalization.batchNormalize(inputList, 0.0, 100.0)
    assert(result.forall(_ == 0.0))
  }

  // Edge cases for normalization
  test("Test normalization with a single repeated value in range") {
    val inputList = List.fill(100)(50.0)
    val result = Normalization.batchNormalize(inputList, 0.0, 100.0)
    assert(result.forall(_ == 0.5))
  }

  test("Test normalization for list with varying step increments") {
    val inputList = List(10.0, 20.0, 30.0, 40.0, 50.0)
    val expectedResults = List(0.0, 0.25, 0.5, 0.75, 1.0)
    val result = Normalization.batchNormalize(inputList, 10.0, 50.0)
    assert(result === expectedResults)
  }

  test("Test normalization when min and max are very close") {
    val inputList = List(0.99, 1.0, 1.01)
    val expectedResults = List(0.0, 0.5, 1.0)
    val result = Normalization.batchNormalize(inputList, 0.99, 1.01)
    assert(result === expectedResults)
  }

  // Performance and stress tests
  test("Test normalization performance with a large dataset") {
    val largeList = (1 to 1000000).map(_.toDouble).toList
    val start = System.nanoTime()
    val result = Normalization.batchNormalize(largeList, 1.0, 1000000.0)
    val end = System.nanoTime()
    val duration = (end - start) / 1e9d
    assert(result.nonEmpty && duration < 2.0, "Batch normalization took too long")
  }

  test("Test normalization performance with extreme values") {
    val largeList = List.fill(1000000)(Double.MaxValue)
    val start = System.nanoTime()
    val result = Normalization.batchNormalize(largeList, 0.0, Double.MaxValue)
    val end = System.nanoTime()
    val duration = (end - start) / 1e9d
    assert(result.forall(_ == 1.0) && duration < 2.0, "Normalization of extreme values took too long")
  }

  // Test cases for handling edge cases gracefully
  test("Test normalization with a constant value range") {
    val input = 50.0
    val result = Normalization.normalize(input, 50.0, 50.0)
    assert(result === 0.0, "Normalization of constant range should return 0")
  }

  test("Test normalization of batch with constant values") {
    val inputList = List.fill(100)(10.0)
    val result = Normalization.batchNormalize(inputList, 10.0, 10.0)
    assert(result.forall(_ == 0.0), "Normalization of constant value batch should return 0")
  }

  // Edge case handling with NaN and infinity in batch
  test("Test batch normalization with NaN values in list") {
    val inputList = List(10.0, Double.NaN, 30.0)
    val result = Normalization.batchNormalize(inputList, 10.0, 30.0)
    assert(result.exists(_.isNaN), "Batch normalization should propagate NaN values")
  }

  test("Test batch normalization with infinite values") {
    val inputList = List(Double.PositiveInfinity, 100.0, Double.NegativeInfinity)
    val result = Normalization.batchNormalize(inputList, 0.0, 100.0)
    assert(result === List(1.0, 1.0, 0.0), "Infinite values should normalize to bounds")
  }

  test("Test batch normalization with only NaN values") {
    val inputList = List(Double.NaN, Double.NaN)
    val result = Normalization.batchNormalize(inputList, 0.0, 100.0)
    assert(result.forall(_.isNaN), "All NaN values should return NaN after normalization")
  }
}