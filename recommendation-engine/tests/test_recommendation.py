import unittest
from recommendation_engine.src.algorithms.cf_algorithm import CollaborativeFiltering
from recommendation_engine.src.algorithms.dl_algorithm import DeepLearningRecommender
from recommendation_engine.src.algorithms.mf_algorithm import MatrixFactorization

class TestRecommendationEngine(unittest.TestCase):

    def setUp(self):
        # Initialize models
        self.cf_model = CollaborativeFiltering()
        self.dl_model = DeepLearningRecommender()
        self.mf_model = MatrixFactorization()

        # Sample data for testing recommendations
        self.user_data_valid = {'user_id': 1, 'preferences': [10, 20, 30]}
        self.item_data_valid = {'item_id': 101, 'features': [5.0, 3.2, 1.5]}

        # Edge cases for invalid data
        self.user_data_invalid = {'user_id': None, 'preferences': []}
        self.item_data_invalid = {'item_id': None, 'features': []}

        # Edge case: large dataset
        self.large_user_data = {'user_id': 1000, 'preferences': list(range(1000))}
        self.large_item_data = {'item_id': 500, 'features': [float(i) for i in range(100)]}

        # Edge case: minimal input
        self.minimal_user_data = {'user_id': 1, 'preferences': [1]}
        self.minimal_item_data = {'item_id': 1, 'features': [1.0]}

        # Setup to mimic cold-start problem
        self.new_user_data = {'user_id': 9999, 'preferences': []}
        self.new_item_data = {'item_id': 8888, 'features': []}

    # Test: Collaborative Filtering Recommendations
    def test_cf_recommendation(self):
        recommendations = self.cf_model.recommend(self.user_data_valid)
        self.assertIsNotNone(recommendations)
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)

    # Test: Deep Learning Recommendations
    def test_dl_recommendation(self):
        recommendations = self.dl_model.recommend(self.user_data_valid)
        self.assertIsNotNone(recommendations)
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)

    # Test: Matrix Factorization Recommendations
    def test_mf_recommendation(self):
        recommendations = self.mf_model.recommend(self.user_data_valid)
        self.assertIsNotNone(recommendations)
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)

    # Test: Invalid User Data
    def test_cf_invalid_user_data(self):
        with self.assertRaises(ValueError):
            self.cf_model.recommend(self.user_data_invalid)

    def test_dl_invalid_user_data(self):
        with self.assertRaises(ValueError):
            self.dl_model.recommend(self.user_data_invalid)

    def test_mf_invalid_user_data(self):
        with self.assertRaises(ValueError):
            self.mf_model.recommend(self.user_data_invalid)

    # Test: Minimal Input for Recommendations
    def test_cf_minimal_input(self):
        recommendations = self.cf_model.recommend(self.minimal_user_data)
        self.assertIsNotNone(recommendations)
        self.assertIsInstance(recommendations, list)

    def test_dl_minimal_input(self):
        recommendations = self.dl_model.recommend(self.minimal_user_data)
        self.assertIsNotNone(recommendations)
        self.assertIsInstance(recommendations, list)

    def test_mf_minimal_input(self):
        recommendations = self.mf_model.recommend(self.minimal_user_data)
        self.assertIsNotNone(recommendations)
        self.assertIsInstance(recommendations, list)

    # Test: Cold-Start Problem Handling
    def test_cf_cold_start(self):
        recommendations = self.cf_model.recommend(self.new_user_data)
        self.assertIsInstance(recommendations, list)

    def test_dl_cold_start(self):
        recommendations = self.dl_model.recommend(self.new_user_data)
        self.assertIsInstance(recommendations, list)

    def test_mf_cold_start(self):
        recommendations = self.mf_model.recommend(self.new_user_data)
        self.assertIsInstance(recommendations, list)

    # Test: Large Dataset for User Preferences
    def test_cf_large_user_data(self):
        recommendations = self.cf_model.recommend(self.large_user_data)
        self.assertIsNotNone(recommendations)
        self.assertGreater(len(recommendations), 0)

    def test_dl_large_user_data(self):
        recommendations = self.dl_model.recommend(self.large_user_data)
        self.assertIsNotNone(recommendations)
        self.assertGreater(len(recommendations), 0)

    def test_mf_large_user_data(self):
        recommendations = self.mf_model.recommend(self.large_user_data)
        self.assertIsNotNone(recommendations)
        self.assertGreater(len(recommendations), 0)

    # Test: Invalid Item Data
    def test_cf_invalid_item_data(self):
        with self.assertRaises(ValueError):
            self.cf_model.recommend(self.item_data_invalid)

    def test_dl_invalid_item_data(self):
        with self.assertRaises(ValueError):
            self.dl_model.recommend(self.item_data_invalid)

    def test_mf_invalid_item_data(self):
        with self.assertRaises(ValueError):
            self.mf_model.recommend(self.item_data_invalid)

    # Test: Large Dataset for Item Features
    def test_cf_large_item_data(self):
        recommendations = self.cf_model.recommend(self.large_item_data)
        self.assertIsNotNone(recommendations)
        self.assertGreater(len(recommendations), 0)

    def test_dl_large_item_data(self):
        recommendations = self.dl_model.recommend(self.large_item_data)
        self.assertIsNotNone(recommendations)
        self.assertGreater(len(recommendations), 0)

    def test_mf_large_item_data(self):
        recommendations = self.mf_model.recommend(self.large_item_data)
        self.assertIsNotNone(recommendations)
        self.assertGreater(len(recommendations), 0)

    # Test: Model handles empty preferences
    def test_cf_empty_preferences(self):
        empty_user_data = {'user_id': 123, 'preferences': []}
        recommendations = self.cf_model.recommend(empty_user_data)
        self.assertIsInstance(recommendations, list)

    def test_dl_empty_preferences(self):
        empty_user_data = {'user_id': 123, 'preferences': []}
        recommendations = self.dl_model.recommend(empty_user_data)
        self.assertIsInstance(recommendations, list)

    def test_mf_empty_preferences(self):
        empty_user_data = {'user_id': 123, 'preferences': []}
        recommendations = self.mf_model.recommend(empty_user_data)
        self.assertIsInstance(recommendations, list)

    # Test: Invalid input types
    def test_cf_invalid_type(self):
        with self.assertRaises(TypeError):
            self.cf_model.recommend("invalid input")

    def test_dl_invalid_type(self):
        with self.assertRaises(TypeError):
            self.dl_model.recommend("invalid input")

    def test_mf_invalid_type(self):
        with self.assertRaises(TypeError):
            self.mf_model.recommend("invalid input")

    # Test: Performance with large user-item matrix (mock large data scenario)
    def test_performance_large_matrix_cf(self):
        large_data = {'user_id': 1, 'preferences': list(range(1000000))}
        recommendations = self.cf_model.recommend(large_data)
        self.assertGreater(len(recommendations), 0)

    def test_performance_large_matrix_dl(self):
        large_data = {'user_id': 1, 'preferences': list(range(1000000))}
        recommendations = self.dl_model.recommend(large_data)
        self.assertGreater(len(recommendations), 0)

    def test_performance_large_matrix_mf(self):
        large_data = {'user_id': 1, 'preferences': list(range(1000000))}
        recommendations = self.mf_model.recommend(large_data)
        self.assertGreater(len(recommendations), 0)

if __name__ == '__main__':
    unittest.main()