import boto3
import os
import logging
from botocore.exceptions import NoCredentialsError, ClientError
from botocore.config import Config
from tqdm import tqdm
import time
import sys

# Logging configuration
LOG_FILE = '/logs/s3_manager.log'
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Constants
MAX_RETRIES = 5
CHUNK_SIZE = 1024 * 1024 * 50  # 50 MB

# S3 Manager class
class S3Manager:
    def __init__(self, aws_access_key, aws_secret_key, bucket_name, region_name='us-east-1'):
        self.s3 = boto3.client('s3', aws_access_key_id=aws_access_key,
                               aws_secret_access_key=aws_secret_key,
                               region_name=region_name,
                               config=Config(retries={'max_attempts': 10, 'mode': 'adaptive'}))
        self.bucket_name = bucket_name

    def upload_file(self, file_path, s3_key):
        """Upload a file to S3 with progress bar and multipart upload for large files."""
        try:
            file_size = os.path.getsize(file_path)
            if file_size > CHUNK_SIZE:
                logging.info(f"Starting multipart upload for large file: {file_path}")
                self._multipart_upload(file_path, s3_key)
            else:
                with tqdm(total=file_size, unit='B', unit_scale=True, desc=f'Uploading {file_path}') as pbar:
                    self.s3.upload_file(file_path, self.bucket_name, s3_key,
                                        Callback=lambda bytes_transferred: pbar.update(bytes_transferred))
                logging.info(f"File uploaded successfully: {file_path} to {s3_key}")
        except FileNotFoundError:
            logging.error(f"File not found: {file_path}")
            raise
        except NoCredentialsError:
            logging.error("AWS credentials not available")
            raise
        except ClientError as e:
            logging.error(f"Client error: {e}")
            raise

    def _multipart_upload(self, file_path, s3_key):
        """Handle multipart upload for large files."""
        try:
            file_size = os.path.getsize(file_path)
            part_size = CHUNK_SIZE
            part_count = (file_size + part_size - 1) // part_size

            # Initiate multipart upload
            multipart_upload = self.s3.create_multipart_upload(Bucket=self.bucket_name, Key=s3_key)
            upload_id = multipart_upload['UploadId']
            parts = []

            # Upload parts
            with open(file_path, 'rb') as f, tqdm(total=file_size, unit='B', unit_scale=True, desc=f'Uploading {file_path}') as pbar:
                for part_num in range(1, part_count + 1):
                    part_data = f.read(part_size)
                    response = self.s3.upload_part(Bucket=self.bucket_name, Key=s3_key,
                                                   PartNumber=part_num, UploadId=upload_id, Body=part_data)
                    parts.append({'ETag': response['ETag'], 'PartNumber': part_num})
                    pbar.update(len(part_data))

            # Complete multipart upload
            self.s3.complete_multipart_upload(Bucket=self.bucket_name, Key=s3_key, UploadId=upload_id,
                                              MultipartUpload={'Parts': parts})
            logging.info(f"Multipart upload completed: {file_path} to {s3_key}")

        except Exception as e:
            logging.error(f"Multipart upload failed: {e}")
            raise

    def download_file(self, s3_key, download_path):
        """Download a file from S3 with progress bar and retry logic."""
        attempt = 0
        while attempt < MAX_RETRIES:
            try:
                metadata = self.s3.head_object(Bucket=self.bucket_name, Key=s3_key)
                file_size = metadata['ContentLength']
                with tqdm(total=file_size, unit='B', unit_scale=True, desc=f'Downloading {s3_key}') as pbar:
                    self.s3.download_file(self.bucket_name, s3_key, download_path,
                                          Callback=lambda bytes_transferred: pbar.update(bytes_transferred))
                logging.info(f"File downloaded successfully: {s3_key} to {download_path}")
                return
            except FileNotFoundError:
                logging.error(f"File not found for download path: {download_path}")
                raise
            except NoCredentialsError:
                logging.error("AWS credentials not available")
                raise
            except ClientError as e:
                attempt += 1
                logging.error(f"Client error on attempt {attempt}: {e}")
                if attempt >= MAX_RETRIES:
                    logging.error("Maximum retries reached. Download failed.")
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff

    def list_files(self, prefix=''):
        """List files in an S3 bucket with an optional prefix."""
        try:
            response = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
            files = [content['Key'] for content in response.get('Contents', [])]
            logging.info(f"Files listed successfully for prefix: {prefix}")
            return files
        except NoCredentialsError:
            logging.error("AWS credentials not available")
            raise
        except ClientError as e:
            logging.error(f"Client error: {e}")
            raise

    def delete_file(self, s3_key):
        """Delete a file from S3 with retry logic."""
        attempt = 0
        while attempt < MAX_RETRIES:
            try:
                self.s3.delete_object(Bucket=self.bucket_name, Key=s3_key)
                logging.info(f"File deleted successfully: {s3_key}")
                return
            except NoCredentialsError:
                logging.error("AWS credentials not available")
                raise
            except ClientError as e:
                attempt += 1
                logging.error(f"Client error on attempt {attempt}: {e}")
                if attempt >= MAX_RETRIES:
                    logging.error("Maximum retries reached. Deletion failed.")
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff

    def generate_presigned_url(self, s3_key, expiration=3600):
        """Generate a presigned URL for a file in S3."""
        try:
            url = self.s3.generate_presigned_url('get_object',
                                                 Params={'Bucket': self.bucket_name, 'Key': s3_key},
                                                 ExpiresIn=expiration)
            logging.info(f"Presigned URL generated for {s3_key}")
            return url
        except NoCredentialsError:
            logging.error("AWS credentials not available")
            raise
        except ClientError as e:
            logging.error(f"Client error: {e}")
            raise

# Usage
if __name__ == "__main__":
    # AWS credentials and bucket configuration
    AWS_ACCESS_KEY = 'access-key'
    AWS_SECRET_KEY = 'secret-key'
    BUCKET_NAME = 'bucket-name'

    # Initialize the S3 manager
    s3_manager = S3Manager(AWS_ACCESS_KEY, AWS_SECRET_KEY, BUCKET_NAME)

    # Operations
    try:
        # Upload a file
        s3_manager.upload_file('/local/file.txt', 'folder/file.txt')

        # Download a file
        s3_manager.download_file('folder/file.txt', '/download/file.txt')

        # List files in a bucket (optional prefix)
        files = s3_manager.list_files('folder/')
        print("Files in S3:", files)

        # Delete a file
        s3_manager.delete_file('folder/file.txt')

        # Generate presigned URL
        url = s3_manager.generate_presigned_url('folder/file.txt')
        print("Presigned URL:", url)

    except Exception as e:
        logging.error(f"Operation failed: {e}")
        sys.exit(1)