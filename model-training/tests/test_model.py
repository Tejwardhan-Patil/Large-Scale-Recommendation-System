import unittest
from src.models.collaborative_filtering import CollaborativeFiltering
from src.models.deep_learning import DeepLearningModel
from src.evaluation.model_evaluation import ModelEvaluation
from src.utils.metrics import calculate_accuracy, calculate_precision

class TestCollaborativeFilteringModel(unittest.TestCase):
    """
    Unit tests for the Collaborative Filtering model.
    Includes tests for model training, predictions, and evaluation.
    """

    @classmethod
    def setUpClass(cls):
        """
        Initialize the model once for all tests to avoid re-initialization overhead.
        """
        cls.model = CollaborativeFiltering()

    def setUp(self):
        """
        Common setup before each test case.
        Can include any shared data preparation.
        """
        self.user_id = 123
        self.train_data = "/train/data"
        self.test_data = "/test/data"

    def test_train_model(self):
        """
        Test case to check if the model training process runs without issues.
        """
        print("Testing CollaborativeFiltering model training...")
        result = self.model.train(self.train_data)
        self.assertTrue(result, "CollaborativeFiltering model training failed")
        print("Training completed successfully.")

    def test_train_with_empty_data(self):
        """
        Test case for handling empty training data.
        """
        print("Testing CollaborativeFiltering model training with empty data...")
        with self.assertRaises(ValueError, msg="Model should raise ValueError on empty data"):
            self.model.train(None)

    def test_predict(self):
        """
        Test case to check if predictions are returned correctly for a valid user ID.
        """
        print(f"Testing CollaborativeFiltering prediction for user {self.user_id}...")
        recommendations = self.model.predict(self.user_id)
        self.assertIsInstance(recommendations, list, "Prediction result should be a list")
        self.assertGreater(len(recommendations), 0, "CollaborativeFiltering returned no recommendations")
        print(f"Prediction completed with {len(recommendations)} recommendations.")

    def test_predict_with_invalid_user(self):
        """
        Test case for handling invalid user IDs in prediction.
        """
        print("Testing CollaborativeFiltering prediction with invalid user ID...")
        invalid_user_id = -1
        with self.assertRaises(ValueError, msg="Model should raise ValueError for invalid user ID"):
            self.model.predict(invalid_user_id)

    def test_evaluate_model(self):
        """
        Test case to ensure the model evaluation provides acceptable accuracy.
        """
        print("Testing CollaborativeFiltering model evaluation...")
        evaluation = ModelEvaluation()
        accuracy = evaluation.evaluate(self.model, self.test_data)
        self.assertGreaterEqual(accuracy, 0.7, "CollaborativeFiltering model accuracy is below 0.7")
        print(f"Model evaluation completed with accuracy: {accuracy}")

    def test_evaluate_with_empty_test_data(self):
        """
        Test case for handling empty test data during evaluation.
        """
        print("Testing CollaborativeFiltering evaluation with empty test data...")
        evaluation = ModelEvaluation()
        with self.assertRaises(ValueError, msg="Evaluation should raise ValueError for empty test data"):
            evaluation.evaluate(self.model, None)


class TestDeepLearningModel(unittest.TestCase):
    """
    Unit tests for the Deep Learning model.
    Includes tests for model training, predictions, and metrics calculations.
    """

    @classmethod
    def setUpClass(cls):
        """
        Initialize the deep learning model once for all tests.
        """
        cls.model = DeepLearningModel()

    def setUp(self):
        """
        Common setup before each test case.
        """
        self.user_id = 456
        self.train_data = "/train/data"
        self.test_data = "/test/data"

    def test_train_model(self):
        """
        Test case to check if the deep learning model trains correctly.
        """
        print("Testing DeepLearning model training...")
        result = self.model.train(self.train_data)
        self.assertTrue(result, "DeepLearning model training failed")
        print("Training completed successfully.")

    def test_train_with_invalid_data(self):
        """
        Test case for handling invalid data during training.
        """
        print("Testing DeepLearning model training with invalid data...")
        invalid_data = []
        with self.assertRaises(ValueError, msg="Training should raise ValueError for invalid data"):
            self.model.train(invalid_data)

    def test_predict(self):
        """
        Test case to check predictions for a valid user ID.
        """
        print(f"Testing DeepLearning prediction for user {self.user_id}...")
        recommendations = self.model.predict(self.user_id)
        self.assertIsInstance(recommendations, list, "Prediction result should be a list")
        self.assertGreater(len(recommendations), 0, "DeepLearning returned no recommendations")
        print(f"Prediction completed with {len(recommendations)} recommendations.")

    def test_predict_with_invalid_user(self):
        """
        Test case for handling invalid user IDs in deep learning prediction.
        """
        print("Testing DeepLearning prediction with invalid user ID...")
        invalid_user_id = None
        with self.assertRaises(ValueError, msg="Model should raise ValueError for invalid user ID"):
            self.model.predict(invalid_user_id)

    def test_model_metrics(self):
        """
        Test case to ensure metrics calculation works as expected for predictions.
        """
        print("Testing DeepLearning model metrics calculation...")
        predictions = [1, 0, 1, 1]
        ground_truth = [1, 0, 0, 1]
        accuracy = calculate_accuracy(predictions, ground_truth)
        precision = calculate_precision(predictions, ground_truth)

        self.assertGreaterEqual(accuracy, 0.8, "Accuracy is below 0.8")
        self.assertGreaterEqual(precision, 0.75, "Precision is below 0.75")
        print(f"Metrics calculation completed. Accuracy: {accuracy}, Precision: {precision}")

    def test_model_metrics_with_empty_data(self):
        """
        Test case for handling empty data during metrics calculation.
        """
        print("Testing DeepLearning model metrics with empty data...")
        predictions = []
        ground_truth = []
        with self.assertRaises(ValueError, msg="Metrics calculation should raise ValueError for empty data"):
            calculate_accuracy(predictions, ground_truth)


if __name__ == '__main__':
    unittest.main()