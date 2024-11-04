import unittest
import json
from inference.predict import get_prediction

class TestInference(unittest.TestCase):

    def setUp(self):
        # Basic valid input data
        self.valid_input = {
            "user_id": 12345,
            "item_id": 67890
        }
        
        # Expected valid response
        self.expected_valid_response = {
            "prediction": 0.85
        }

        # Another set of valid input data
        self.another_valid_input = {
            "user_id": 54321,
            "item_id": 98765
        }
        
        # Expected response for second valid input
        self.expected_another_response = {
            "prediction": 0.92
        }

        # Input with boundary values for IDs
        self.boundary_input = {
            "user_id": 0,
            "item_id": 1
        }
        
        # Expected response for boundary input
        self.expected_boundary_response = {
            "prediction": 0.50
        }

        # Invalid input for error cases
        self.invalid_input = {
            "user_id": None,
            "item_id": 67890
        }

        # Incomplete input with missing fields
        self.incomplete_input = {
            "user_id": 12345
        }

        # Negative input values
        self.negative_input = {
            "user_id": -12345,
            "item_id": -67890
        }

        # Extremely large numbers for user_id and item_id
        self.large_input = {
            "user_id": 9999999999,
            "item_id": 8888888888
        }

    def test_get_prediction_valid(self):
        # Test for valid input
        response = get_prediction(self.valid_input)
        response_json = json.loads(response)
        self.assertEqual(response_json['prediction'], self.expected_valid_response['prediction'])

    def test_get_another_prediction_valid(self):
        # Test for another valid input case
        response = get_prediction(self.another_valid_input)
        response_json = json.loads(response)
        self.assertEqual(response_json['prediction'], self.expected_another_response['prediction'])

    def test_get_boundary_prediction(self):
        # Test with boundary case input
        response = get_prediction(self.boundary_input)
        response_json = json.loads(response)
        self.assertEqual(response_json['prediction'], self.expected_boundary_response['prediction'])

    def test_invalid_input(self):
        # Test with invalid input values
        with self.assertRaises(ValueError):
            get_prediction(self.invalid_input)

    def test_missing_fields(self):
        # Test with missing fields in the input data
        with self.assertRaises(KeyError):
            get_prediction(self.incomplete_input)

    def test_negative_input(self):
        # Test with negative user_id and item_id
        with self.assertRaises(ValueError):
            get_prediction(self.negative_input)

    def test_large_input(self):
        # Test with extremely large user_id and item_id
        response = get_prediction(self.large_input)
        response_json = json.loads(response)
        # Check if the response is within a valid range, where 0.0 to 1.0 is valid
        self.assertTrue(0.0 <= response_json['prediction'] <= 1.0)

    def test_empty_input(self):
        # Test with empty input dictionary
        empty_input = {}
        with self.assertRaises(KeyError):
            get_prediction(empty_input)

    def test_string_input(self):
        # Test with string values instead of integers
        string_input = {
            "user_id": "12345",
            "item_id": "67890"
        }
        with self.assertRaises(TypeError):
            get_prediction(string_input)

    def test_float_input(self):
        # Test with float values instead of integers
        float_input = {
            "user_id": 12345.67,
            "item_id": 67890.12
        }
        with self.assertRaises(TypeError):
            get_prediction(float_input)

    def test_zero_input(self):
        # Test with zero values for both user_id and item_id
        zero_input = {
            "user_id": 0,
            "item_id": 0
        }
        response = get_prediction(zero_input)
        response_json = json.loads(response)
        self.assertTrue(0.0 <= response_json['prediction'] <= 1.0)

    def test_prediction_range(self):
        # Test if predictions always fall within expected range
        input_data = {
            "user_id": 123,
            "item_id": 456
        }
        response = get_prediction(input_data)
        response_json = json.loads(response)
        self.assertTrue(0.0 <= response_json['prediction'] <= 1.0)

    def test_multiple_requests(self):
        # Test multiple requests for consistency
        responses = []
        for _ in range(10):
            response = get_prediction(self.valid_input)
            response_json = json.loads(response)
            responses.append(response_json['prediction'])
        
        # Check if all predictions are the same for consistent input
        self.assertTrue(all(p == responses[0] for p in responses))

    def test_unexpected_fields(self):
        # Test input with unexpected extra fields
        input_with_extra_fields = {
            "user_id": 12345,
            "item_id": 67890,
            "extra_field": "unexpected"
        }
        response = get_prediction(input_with_extra_fields)
        response_json = json.loads(response)
        self.assertEqual(response_json['prediction'], self.expected_valid_response['prediction'])

    def test_null_user_id(self):
        # Test input with null user_id
        input_with_null_user_id = {
            "user_id": None,
            "item_id": 67890
        }
        with self.assertRaises(ValueError):
            get_prediction(input_with_null_user_id)

    def test_null_item_id(self):
        # Test input with null item_id
        input_with_null_item_id = {
            "user_id": 12345,
            "item_id": None
        }
        with self.assertRaises(ValueError):
            get_prediction(input_with_null_item_id)

    def test_boolean_input(self):
        # Test input with boolean values
        input_with_boolean = {
            "user_id": True,
            "item_id": False
        }
        with self.assertRaises(TypeError):
            get_prediction(input_with_boolean)

    def test_array_input(self):
        # Test input with arrays instead of scalars
        input_with_array = {
            "user_id": [12345],
            "item_id": [67890]
        }
        with self.assertRaises(TypeError):
            get_prediction(input_with_array)

    def test_object_input(self):
        # Test input with object values instead of scalars
        input_with_object = {
            "user_id": {"key": "value"},
            "item_id": {"another_key": "another_value"}
        }
        with self.assertRaises(TypeError):
            get_prediction(input_with_object)

if __name__ == "__main__":
    unittest.main()