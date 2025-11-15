"""
S3 Service for HIPAA-compliant symptom image storage
Images are encrypted at rest and in transit
"""

import os
import boto3
from botocore.exceptions import ClientError
from datetime import datetime, timedelta
from typing import Dict
import uuid

from app.models.symptom_journal import BodyArea


# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION', 'us-east-1')
)

BUCKET_NAME = os.getenv('AWS_S3_BUCKET_NAME')


async def upload_symptom_image(
    file_data: bytes,
    patient_id: str,
    body_area: BodyArea,
    content_type: str
) -> Dict:
    """
    Upload symptom image to S3 with server-side encryption
    
    Returns:
        Dict with bucket, key, and presigned URL
    """
    try:
        # Generate unique filename
        file_extension = content_type.split('/')[-1]
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        
        # Organize by patient and body area
        s3_key = f"symptom-journal/{patient_id}/{body_area.value}/{timestamp}_{unique_id}.{file_extension}"
        
        # Upload with server-side encryption (HIPAA compliance)
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=s3_key,
            Body=file_data,
            ContentType=content_type,
            ServerSideEncryption='AES256',  # Server-side encryption
            Metadata={
                'patient-id': patient_id,
                'body-area': body_area.value,
                'upload-timestamp': timestamp
            }
        )
        
        # Generate presigned URL (valid for 1 hour)
        presigned_url = generate_presigned_url(BUCKET_NAME, s3_key, expiration=3600)
        
        return {
            "bucket": BUCKET_NAME,
            "key": s3_key,
            "url": presigned_url
        }
        
    except ClientError as e:
        print(f"Error uploading to S3: {e}")
        raise Exception("Failed to upload image to secure storage")


def generate_presigned_url(bucket: str, key: str, expiration: int = 3600) -> str:
    """
    Generate a presigned URL for secure image access
    URL expires after specified time (default 1 hour)
    """
    try:
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': bucket,
                'Key': key
            },
            ExpiresIn=expiration
        )
        return url
    except ClientError as e:
        print(f"Error generating presigned URL: {e}")
        return None
