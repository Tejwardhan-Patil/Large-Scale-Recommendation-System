import requests
import json
import logging
from typing import Dict, Any, Optional, Union


class APIError(Exception):
    """Exception for API errors."""
    def __init__(self, message: str, status_code: Optional[int] = None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class FileError(Exception):
    """Exception for file-related errors."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class APIExtractor:
    def __init__(self, base_url: str, headers: Optional[Dict[str, str]] = None, params: Optional[Dict[str, str]] = None):
        """
        Initializes the APIExtractor with base URL, headers, and optional parameters.

        :param base_url: The base URL of the API
        :param headers: Optional headers for API requests
        :param params: Optional default parameters for API requests
        """
        self.base_url = base_url
        self.headers = headers if headers else {}
        self.params = params if params else {}

    def fetch_data(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Union[Dict, None]:
        """
        Fetches data from the API.

        :param endpoint: The API endpoint to query
        :param params: Additional parameters for the API request
        :return: Parsed JSON data from the API response
        :raises: APIError if an HTTP or connection error occurs.
        """
        url = f"{self.base_url}/{endpoint}"
        request_params = self.params.copy()
        if params:
            request_params.update(params)

        try:
            logging.info(f"Requesting data from {url} with params {request_params}")
            response = requests.get(url, headers=self.headers, params=request_params)
            response.raise_for_status()
            logging.info(f"Data successfully fetched from {url}")
            return response.json()

        except requests.exceptions.HTTPError as http_err:
            logging.error(f"HTTP error occurred: {http_err}")
            raise APIError("HTTP error occurred while fetching data", response.status_code)
        except requests.exceptions.ConnectionError as conn_err:
            logging.error(f"Connection error occurred: {conn_err}")
            raise APIError("Connection error occurred while fetching data")
        except requests.exceptions.Timeout as timeout_err:
            logging.error(f"Timeout error occurred: {timeout_err}")
            raise APIError("Timeout error occurred while fetching data")
        except Exception as err:
            logging.error(f"An error occurred: {err}")
            raise APIError("An unknown error occurred while fetching data")

        return None

    def save_data(self, data: Any, file_path: str):
        """
        Saves the extracted data to a JSON file.

        :param data: The data to save
        :param file_path: Path to the output file
        :raises: FileError if there is an issue saving the file.
        """
        try:
            with open(file_path, 'w') as json_file:
                json.dump(data, json_file, indent=4)
            logging.info(f"Data successfully saved to {file_path}")

        except IOError as e:
            logging.error(f"File I/O error: {e}")
            raise FileError(f"Failed to save data to {file_path}")

    def load_data(self, file_path: str) -> Union[Dict, None]:
        """
        Loads data from a JSON file.

        :param file_path: Path to the input file
        :return: Parsed JSON data from the file
        :raises: FileError if there is an issue loading the file.
        """
        try:
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
            logging.info(f"Data successfully loaded from {file_path}")
            return data

        except IOError as e:
            logging.error(f"File I/O error: {e}")
            raise FileError(f"Failed to load data from {file_path}")

    def extract_to_file(self, endpoint: str, output_file: str, params: Optional[Dict[str, Any]] = None):
        """
        Extracts data from the API and saves it to a file.

        :param endpoint: API endpoint
        :param output_file: Output file path
        :param params: Additional parameters for the API request
        """
        data = self.fetch_data(endpoint, params)
        if data:
            self.save_data(data, output_file)
        else:
            logging.error("No data extracted from the API.")

    def validate_response(self, data: Dict, validation_key: str) -> bool:
        """
        Validates the response data from the API.

        :param data: The API response data
        :param validation_key: The key to validate in the response data
        :return: True if the key is present, False otherwise
        """
        if validation_key in data:
            logging.info(f"Validation passed: Key '{validation_key}' found in the response.")
            return True
        logging.error(f"Validation failed: Key '{validation_key}' not found in the response.")
        return False

    def retry_fetch(self, endpoint: str, params: Optional[Dict[str, Any]] = None, retries: int = 3) -> Union[Dict, None]:
        """
        Attempts to fetch data from the API with a specified number of retries.

        :param endpoint: The API endpoint to query
        :param params: Additional parameters for the API request
        :param retries: Number of retry attempts in case of failure
        :return: Parsed JSON data from the API response or None after retries
        """
        attempt = 0
        while attempt < retries:
            try:
                data = self.fetch_data(endpoint, params)
                if data:
                    return data
            except APIError as e:
                logging.error(f"Attempt {attempt + 1} failed: {e.message}")
            attempt += 1
            logging.info(f"Retrying... ({attempt}/{retries})")
        logging.error(f"Failed to fetch data after {retries} attempts.")
        return None

    def extract_with_retry(self, endpoint: str, output_file: str, params: Optional[Dict[str, Any]] = None, retries: int = 3):
        """
        Extracts data from the API with retries and saves it to a file.

        :param endpoint: API endpoint
        :param output_file: Output file path
        :param params: Additional parameters for the API request
        :param retries: Number of retry attempts in case of failure
        """
        data = self.retry_fetch(endpoint, params, retries)
        if data:
            self.save_data(data, output_file)
        else:
            logging.error("Failed to extract data from the API after multiple retries.")

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Extractor initialization
api_extractor = APIExtractor(
    base_url="https://api.website.com",
    headers={"Authorization": "Bearer TOKEN"}
)

# Data extraction from an API endpoint to a file
api_extractor.extract_with_retry(
    endpoint="data-endpoint",
    output_file="output.json",
    params={"limit": 100},
    retries=3
)