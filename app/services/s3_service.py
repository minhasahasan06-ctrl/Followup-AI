"""
S3 Service for HIPAA-compliant symptom image storage
Images are encrypted at rest and in transit
Supports local filesystem fallback for development/testing
"""

import os
import boto3
from botocore.exceptions import ClientError
from datetime import datetime, timedelta
from typing import Dict, Optional
import uuid
import shutil
import logging

from app.models.symptom_journal import BodyArea

logger = logging.getLogger(__name__)


# Parse AWS_REGION (handle both "ap-southeast-2" and "Asia Pacific (Sydney) ap-southeast-2" formats)
def parse_aws_region(region_str: str) -> str:
    """Extract actual region code from potentially formatted string"""
    if not region_str:
        return 'us-east-1'
    # Extract region code pattern (e.g., "ap-southeast-2" from "Asia Pacific (Sydney) ap-southeast-2")
    parts = region_str.split()
    for part in parts:
        if '-' in part and len(part) > 5:  # Region codes have hyphens
            return part
    return region_str.strip()


class S3Service:
    """
    Environment-aware S3 service with local filesystem fallback
    
    Automatically detects if AWS credentials are available:
    - If available: Uses S3 with encryption
    - If not available: Falls back to local filesystem storage
    """
    
    def __init__(self):
        self.use_s3 = self._should_use_s3()
        
        if self.use_s3:
            # Initialize S3 client with credentials
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                region_name=parse_aws_region(os.getenv('AWS_REGION', 'us-east-1'))
            )
            self.bucket_name = os.getenv('AWS_S3_BUCKET_NAME')
            logger.info(f"S3Service initialized with S3 storage (bucket: {self.bucket_name})")
        else:
            # Use local filesystem
            self.local_storage_dir = 'tmp/guided_exam_frames'
            os.makedirs(self.local_storage_dir, exist_ok=True)
            self.s3_client = None
            self.bucket_name = None
            logger.warning("S3Service initialized with LOCAL STORAGE fallback - AWS credentials not available")
    
    def _should_use_s3(self) -> bool:
        """Check if S3 credentials are available"""
        aws_key = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')
        bucket = os.getenv('AWS_S3_BUCKET_NAME')
        return all([aws_key, aws_secret, bucket])
    
    def upload_file(self, file_data: bytes, s3_key: str, content_type: str = 'application/octet-stream', metadata: Optional[Dict] = None) -> str:
        """
        Upload file to S3 or local storage
        
        Args:
            file_data: File bytes
            s3_key: S3 key path (e.g., "guided-exam/patient123/eyes/frame.jpg")
            content_type: MIME type
            metadata: Optional metadata dict
        
        Returns:
            URI string (s3://... or file://...)
        """
        if self.use_s3:
            try:
                # Upload to S3 with server-side encryption
                upload_params = {
                    'Bucket': self.bucket_name,
                    'Key': s3_key,
                    'Body': file_data,
                    'ContentType': content_type,
                    'ServerSideEncryption': 'AES256'
                }
                
                if metadata:
                    upload_params['Metadata'] = metadata
                
                kms_key_id = os.getenv('AWS_KMS_KEY_ID')
                if kms_key_id:
                    upload_params['ServerSideEncryption'] = 'aws:kms'
                    upload_params['SSEKMSKeyId'] = kms_key_id
                
                self.s3_client.put_object(**upload_params)
                s3_uri = f"s3://{self.bucket_name}/{s3_key}"
                logger.info(f"Uploaded to S3: {s3_uri}")
                return s3_uri
            except ClientError as e:
                logger.error(f"S3 upload failed: {e}")
                raise
        else:
            # Local filesystem fallback
            local_filename = s3_key.replace('/', '_')
            local_path = os.path.join(self.local_storage_dir, local_filename)
            
            with open(local_path, 'wb') as f:
                f.write(file_data)
            
            file_uri = f"file://{os.path.abspath(local_path)}"
            logger.info(f"Saved to local storage: {file_uri}")
            return file_uri
    
    def download_file(self, uri: str) -> bytes:
        """
        Download file from S3 or local storage
        
        Args:
            uri: S3 URI (s3://...) or file URI (file://...)
        
        Returns:
            File bytes
        """
        if uri.startswith('s3://'):
            # Download from S3
            parts = uri.replace('s3://', '').split('/', 1)
            bucket = parts[0]
            key = parts[1]
            
            try:
                response = self.s3_client.get_object(Bucket=bucket, Key=key)
                return response['Body'].read()
            except ClientError as e:
                logger.error(f"S3 download failed: {e}")
                raise
        elif uri.startswith('file://'):
            # Read from local filesystem
            local_path = uri.replace('file://', '')
            with open(local_path, 'rb') as f:
                return f.read()
        else:
            raise ValueError(f"Invalid URI format: {uri}")
    
    def generate_presigned_url(self, s3_key: str, expiration: int = 3600) -> Optional[str]:
        """
        Generate presigned URL for S3 object
        Returns None for local storage (not applicable)
        """
        if not self.use_s3:
            logger.warning("Presigned URLs not available in local storage mode")
            return None
        
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': s3_key
                },
                ExpiresIn=expiration
            )
            return url
        except ClientError as e:
            logger.error(f"Error generating presigned URL: {e}")
            return None


# Initialize global S3 service instance
s3_service = S3Service()


# Legacy compatibility: Keep old module-level client for backward compatibility
try:
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=parse_aws_region(os.getenv('AWS_REGION', 'us-east-1'))
    )
    BUCKET_NAME = os.getenv('AWS_S3_BUCKET_NAME')
except Exception as e:
    logger.warning(f"Failed to initialize legacy S3 client: {e}")
    s3_client = None
    BUCKET_NAME = None


async def upload_symptom_image(
    file_data: bytes,
    patient_id: str,
    body_area: BodyArea,
    content_type: str
) -> Dict:
    """
    Upload symptom image to S3 with server-side encryption
    Now uses S3Service with local fallback
    
    Returns:
        Dict with bucket, key, and presigned URL (or local path)
    """
    try:
        # Generate unique filename
        file_extension = content_type.split('/')[-1]
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        
        # Organize by patient and body area
        s3_key = f"symptom-journal/{patient_id}/{body_area.value}/{timestamp}_{unique_id}.{file_extension}"
        
        # Upload using S3Service (with local fallback)
        metadata = {
            'patient-id': patient_id,
            'body-area': body_area.value,
            'upload-timestamp': timestamp
        }
        
        uri = s3_service.upload_file(file_data, s3_key, content_type, metadata)
        
        # Generate presigned URL if using S3
        presigned_url = None
        if uri.startswith('s3://'):
            presigned_url = s3_service.generate_presigned_url(s3_key, expiration=3600)
        
        return {
            "bucket": s3_service.bucket_name if s3_service.use_s3 else "local",
            "key": s3_key,
            "url": presigned_url or uri
        }
        
    except Exception as e:
        logger.error(f"Error uploading symptom image: {e}")
        raise Exception("Failed to upload image to storage")


def generate_presigned_url(bucket: str, key: str, expiration: int = 3600) -> Optional[str]:
    """
    Generate a presigned URL for secure image access
    URL expires after specified time (default 1 hour)
    
    Now uses S3Service - returns None if in local storage mode
    """
    return s3_service.generate_presigned_url(key, expiration)


async def upload_file_to_s3(
    file_data: bytes,
    folder: str,
    filename: str,
    content_type: str
) -> Dict:
    """
    Generic S3 file upload for exam coach and other features
    Now uses S3Service with local fallback
    
    Args:
        file_data: File bytes
        folder: S3 folder path (e.g., "exam-coach")
        filename: Desired filename
        content_type: MIME type
    
    Returns:
        Dict with bucket, key, and presigned URL (or local path)
    """
    try:
        s3_key = f"{folder}/{filename}"
        
        # Upload using S3Service (with local fallback)
        uri = s3_service.upload_file(file_data, s3_key, content_type)
        
        # Generate presigned URL if using S3
        presigned_url = None
        if uri.startswith('s3://'):
            presigned_url = s3_service.generate_presigned_url(s3_key)
        
        return {
            "bucket": s3_service.bucket_name if s3_service.use_s3 else "local",
            "key": s3_key,
            "url": presigned_url or uri
        }
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise
