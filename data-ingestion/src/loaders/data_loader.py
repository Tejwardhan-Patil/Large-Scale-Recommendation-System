import os
import logging
import boto3
from hdfs import InsecureClient
import yaml
import json
import csv
from datetime import datetime


class DataLoader:
    def __init__(self, storage_type, config):
        self.storage_type = storage_type.lower()
        self.config = config
        self._setup_logging()

        if self.storage_type == "s3":
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=config['aws_access_key_id'],
                aws_secret_access_key=config['aws_secret_access_key'],
                region_name=config['region']
            )
            logging.info(f"Initialized S3 client for bucket: {config['s3_bucket_name']}")
        elif self.storage_type == "hadoop":
            self.hdfs_client = InsecureClient(config['hadoop_url'], user=config['hadoop_user'])
            logging.info(f"Initialized HDFS client with URL: {config['hadoop_url']}")
        else:
            logging.error(f"Unsupported storage type: {self.storage_type}")
            raise ValueError(f"Unsupported storage type: {self.storage_type}")

    def _setup_logging(self):
        logging.basicConfig(
            filename='data_loader.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logging.info("Logger initialized for DataLoader.")

    def load_to_storage(self, file_path, destination):
        file_type = self._detect_file_type(file_path)
        logging.info(f"Detected file type {file_type} for file {file_path}.")

        if self.storage_type == "s3":
            self._load_to_s3(file_path, destination)
        elif self.storage_type == "hadoop":
            self._load_to_hadoop(file_path, destination)
        else:
            logging.error(f"Unknown storage type: {self.storage_type}")

    def _load_to_s3(self, file_path, bucket_name):
        file_name = os.path.basename(file_path)
        try:
            self.s3_client.upload_file(file_path, bucket_name, file_name)
            logging.info(f"File {file_name} uploaded to S3 bucket {bucket_name}.")
        except Exception as e:
            logging.error(f"Failed to upload {file_name} to S3: {e}")
            raise

    def _load_to_hadoop(self, file_path, hdfs_path):
        try:
            with open(file_path, 'rb') as file_data:
                self.hdfs_client.write(hdfs_path, file_data, overwrite=True)
            logging.info(f"File {file_path} uploaded to HDFS at {hdfs_path}.")
        except Exception as e:
            logging.error(f"Failed to upload {file_path} to HDFS: {e}")
            raise

    def load_directory(self, dir_path, destination):
        logging.info(f"Starting to load directory {dir_path} to {destination}.")
        for root, _, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                self.load_to_storage(file_path, destination)

    def _detect_file_type(self, file_path):
        _, ext = os.path.splitext(file_path)
        return ext.lower()

    def parse_config(self, config_file):
        ext = self._detect_file_type(config_file)
        if ext == '.yaml':
            with open(config_file, 'r') as file:
                return yaml.safe_load(file)
        elif ext == '.json':
            with open(config_file, 'r') as file:
                return json.load(file)
        else:
            logging.error(f"Unsupported config file format: {ext}")
            raise ValueError(f"Unsupported config file format: {ext}")

    def _convert_csv_to_json(self, csv_file, json_file):
        try:
            data = []
            with open(csv_file, mode='r') as file:
                csv_reader = csv.DictReader(file)
                for row in csv_reader:
                    data.append(row)
            with open(json_file, mode='w') as file:
                json.dump(data, file)
            logging.info(f"Converted {csv_file} to {json_file} in JSON format.")
        except Exception as e:
            logging.error(f"Failed to convert CSV to JSON: {e}")
            raise

    def _convert_json_to_csv(self, json_file, csv_file):
        try:
            with open(json_file, mode='r') as file:
                data = json.load(file)
            with open(csv_file, mode='w', newline='') as file:
                csv_writer = csv.DictWriter(file, fieldnames=data[0].keys())
                csv_writer.writeheader()
                for row in data:
                    csv_writer.writerow(row)
            logging.info(f"Converted {json_file} to {csv_file} in CSV format.")
        except Exception as e:
            logging.error(f"Failed to convert JSON to CSV: {e}")
            raise

    def file_exists_in_storage(self, destination, file_name):
        if self.storage_type == "s3":
            try:
                self.s3_client.head_object(Bucket=destination, Key=file_name)
                logging.info(f"File {file_name} exists in S3 bucket {destination}.")
                return True
            except self.s3_client.exceptions.ClientError:
                return False
        elif self.storage_type == "hadoop":
            try:
                exists = self.hdfs_client.status(destination, strict=False) is not None
                logging.info(f"File {file_name} exists in HDFS path {destination}.")
                return exists
            except Exception as e:
                logging.error(f"Error checking if file exists in HDFS: {e}")
                return False

    def load_file_if_not_exists(self, file_path, destination):
        file_name = os.path.basename(file_path)
        if not self.file_exists_in_storage(destination, file_name):
            self.load_to_storage(file_path, destination)
        else:
            logging.info(f"File {file_name} already exists in {self.storage_type} storage.")

    def log_storage_usage(self):
        if self.storage_type == "s3":
            response = self.s3_client.list_objects_v2(Bucket=self.config['s3_bucket_name'])
            total_size = sum([obj['Size'] for obj in response.get('Contents', [])])
            logging.info(f"Total size of objects in S3 bucket: {total_size} bytes.")
        elif self.storage_type == "hadoop":
            try:
                status = self.hdfs_client.status(self.config['hadoop_base_path'])
                logging.info(f"Hadoop usage at {self.config['hadoop_base_path']}: {status['length']} bytes.")
            except Exception as e:
                logging.error(f"Failed to retrieve HDFS storage usage: {e}")
        else:
            logging.error("Unsupported storage type for logging storage usage.")

    def delete_file_from_storage(self, destination, file_name):
        if self.storage_type == "s3":
            try:
                self.s3_client.delete_object(Bucket=destination, Key=file_name)
                logging.info(f"File {file_name} deleted from S3 bucket {destination}.")
            except Exception as e:
                logging.error(f"Failed to delete file {file_name} from S3: {e}")
        elif self.storage_type == "hadoop":
            try:
                self.hdfs_client.delete(destination, recursive=False)
                logging.info(f"File {file_name} deleted from HDFS path {destination}.")
            except Exception as e:
                logging.error(f"Failed to delete file {file_name} from HDFS: {e}")

    def archive_old_files(self, dir_path, archive_path, days_threshold=30):
        current_time = datetime.now()
        logging.info(f"Archiving files older than {days_threshold} days from {dir_path}.")
        for root, _, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                if (current_time - file_mtime).days > days_threshold:
                    archive_file_path = os.path.join(archive_path, file)
                    os.rename(file_path, archive_file_path)
                    logging.info(f"Archived {file} to {archive_file_path}.")