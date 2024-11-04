import pytest
from data_ingestion.extractor import DataExtractor, ExtractionError
import requests
from unittest import mock

@pytest.fixture
def sample_data():
    return [
        {"id": 1, "name": "Item 1", "url": "https://website.com/item1"},
        {"id": 2, "name": "Item 2", "url": "https://website.com/item2"},
        {"id": 3, "name": "Item 3", "url": "https://website.com/item3"}
    ]

@pytest.fixture
def empty_data():
    return []

@pytest.fixture
def malformed_data():
    return [
        {"id": "a", "name": "Invalid ID", "url": "https://website.com/item-invalid"},
        {"id": 2, "url": "https://website.com/item-missing-name"},
    ]

@pytest.fixture
def data_extractor():
    return DataExtractor()

def test_extractor_init(data_extractor):
    assert isinstance(data_extractor, DataExtractor)

def test_extract_data_valid(data_extractor, sample_data, mocker):
    mocker.patch('requests.get', return_value=mock.Mock(status_code=200, json=lambda: sample_data))
    
    extracted_data = data_extractor.extract_data("https://website.com/api/data")
    assert len(extracted_data) == 3
    assert extracted_data[0]["id"] == 1
    assert extracted_data[1]["name"] == "Item 2"
    assert extracted_data[2]["url"] == "https://website.com/item3"

def test_extract_data_empty(data_extractor, empty_data, mocker):
    mocker.patch('requests.get', return_value=mock.Mock(status_code=200, json=lambda: empty_data))
    
    extracted_data = data_extractor.extract_data("https://website.com/api/empty")
    assert extracted_data == []

def test_extract_data_malformed(data_extractor, malformed_data, mocker):
    mocker.patch('requests.get', return_value=mock.Mock(status_code=200, json=lambda: malformed_data))
    
    with pytest.raises(ExtractionError):
        data_extractor.extract_data("https://website.com/api/malformed")

def test_data_format(data_extractor, sample_data):
    formatted_data = data_extractor.format_data(sample_data)
    
    assert formatted_data[0]["id"] == 1
    assert formatted_data[0]["name"] == "Item 1"
    assert formatted_data[0]["url"] == "https://website.com/item1"
    assert formatted_data[1]["id"] == 2
    assert formatted_data[1]["name"] == "Item 2"
    assert formatted_data[1]["url"] == "https://website.com/item2"

def test_data_format_empty(data_extractor, empty_data):
    formatted_data = data_extractor.format_data(empty_data)
    assert formatted_data == []

def test_data_format_invalid(data_extractor, malformed_data):
    formatted_data = data_extractor.format_data(malformed_data)
    assert len(formatted_data) == 2
    assert formatted_data[0]["id"] == "a"
    assert formatted_data[1]["url"] == "https://website.com/item-missing-name"

def test_extractor_timeout(data_extractor, mocker):
    mocker.patch('requests.get', side_effect=requests.exceptions.Timeout)
    
    with pytest.raises(ExtractionError):
        data_extractor.extract_data("https://website.com/api/timeout")

def test_extractor_connection_error(data_extractor, mocker):
    mocker.patch('requests.get', side_effect=requests.exceptions.ConnectionError)
    
    with pytest.raises(ExtractionError):
        data_extractor.extract_data("https://website.com/api/connection-error")

def test_extractor_http_error(data_extractor, mocker):
    mocker.patch('requests.get', return_value=mock.Mock(status_code=404))
    
    with pytest.raises(ExtractionError):
        data_extractor.extract_data("https://website.com/api/404")

def test_extractor_server_error(data_extractor, mocker):
    mocker.patch('requests.get', return_value=mock.Mock(status_code=500))
    
    with pytest.raises(ExtractionError):
        data_extractor.extract_data("https://website.com/api/500")

def test_extractor_invalid_json(data_extractor, mocker):
    mocker.patch('requests.get', return_value=mock.Mock(status_code=200, json=mock.Mock(side_effect=ValueError)))
    
    with pytest.raises(ExtractionError):
        data_extractor.extract_data("https://website.com/api/invalid-json")

def test_retry_on_failure(data_extractor, mocker):
    mocker.patch('requests.get', side_effect=[requests.exceptions.ConnectionError, mock.Mock(status_code=200, json=lambda: [{"id": 1, "name": "Retry Item", "url": "https://website.com/retry"}])])
    
    extracted_data = data_extractor.extract_data("https://website.com/api/retry")
    assert len(extracted_data) == 1
    assert extracted_data[0]["name"] == "Retry Item"

def test_format_data_case_insensitive(data_extractor):
    raw_data = [{"ID": 1, "NAME": "Item 1", "URL": "https://website.com/item1"}]
    formatted_data = data_extractor.format_data(raw_data)

    assert formatted_data[0]["id"] == 1
    assert formatted_data[0]["name"] == "Item 1"
    assert formatted_data[0]["url"] == "https://website.com/item1"

def test_data_validator_valid(data_extractor, sample_data):
    assert data_extractor.validate_data(sample_data) is True

def test_data_validator_invalid(data_extractor, malformed_data):
    assert data_extractor.validate_data(malformed_data) is False

def test_data_validator_empty(data_extractor, empty_data):
    assert data_extractor.validate_data(empty_data) is True

def test_transform_data(data_extractor, sample_data):
    transformed_data = data_extractor.transform_data(sample_data)
    
    assert transformed_data[0]["name"] == "ITEM 1"
    assert transformed_data[1]["name"] == "ITEM 2"
    assert transformed_data[2]["name"] == "ITEM 3"

def test_transform_data_empty(data_extractor, empty_data):
    transformed_data = data_extractor.transform_data(empty_data)
    assert transformed_data == []

def test_transform_data_invalid(data_extractor, malformed_data):
    transformed_data = data_extractor.transform_data(malformed_data)
    assert transformed_data[0]["name"] == "INVALID ID"
    assert transformed_data[1]["name"] is None

def test_pagination_handling(data_extractor, mocker):
    page1 = [{"id": 1, "name": "Item 1", "url": "https://website.com/item1"}]
    page2 = [{"id": 2, "name": "Item 2", "url": "https://website.com/item2"}]
    
    mocker.patch('requests.get', side_effect=[mock.Mock(status_code=200, json=lambda: page1), mock.Mock(status_code=200, json=lambda: page2)])
    
    extracted_data = data_extractor.extract_data("https://website.com/api/paginated", paginate=True)
    
    assert len(extracted_data) == 2
    assert extracted_data[0]["name"] == "Item 1"
    assert extracted_data[1]["name"] == "Item 2"

def test_pagination_empty(data_extractor, mocker):
    mocker.patch('requests.get', return_value=mock.Mock(status_code=200, json=lambda: []))
    
    extracted_data = data_extractor.extract_data("https://website.com/api/paginated", paginate=True)
    assert extracted_data == []

def test_handle_rate_limiting(data_extractor, mocker):
    mocker.patch('requests.get', side_effect=[mock.Mock(status_code=429), mock.Mock(status_code=200, json=lambda: [{"id": 1, "name": "Item after limit", "url": "https://website.com/item-after-limit"}])])
    
    extracted_data = data_extractor.extract_data("https://website.com/api/rate-limited")
    
    assert len(extracted_data) == 1
    assert extracted_data[0]["name"] == "Item after limit"

def test_authentication_required(data_extractor, mocker):
    mocker.patch('requests.get', return_value=mock.Mock(status_code=401))
    
    with pytest.raises(ExtractionError):
        data_extractor.extract_data("https://website.com/api/protected")