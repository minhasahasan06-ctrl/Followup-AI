"""
S3 Service for HIPAA-compliant symptom image storage
Images are encrypted at rest and in transit
"""

import os
import boto3
from botocore.exceptions import ClientError
from datetime import datetime, timedelta
from typing import Dict, Optional
import uuid

from app.models.symptom_journal import BodyArea


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

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=parse_aws_region(os.getenv('AWS_REGION', 'us-east-1'))
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
    if not BUCKET_NAME:
        raise Exception("AWS_S3_BUCKET_NAME environment variable not configured")
    
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


def generate_presigned_url(bucket: str, key: str, expiration: int = 3600) -> Optional[str]:
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


async def upload_file_to_s3(
    file_data: bytes,
    folder: str,
    filename: str,
    content_type: str
) -> Dict:
    """
    Generic S3 file upload for exam coach and other features
    HIPAA Compliance: Server-side encryption enabled
    
    Args:
        file_data: File bytes
        folder: S3 folder path (e.g., "exam-coach")
        filename: Desired filename
        content_type: MIME type
    
    Returns:
        Dict with bucket, key, and presigned URL
    """
    if not BUCKET_NAME:
        raise Exception("AWS_S3_BUCKET_NAME environment variable not configured")
    
    try:
        s3_key = f"{folder}/{filename}"
        
        # Upload with server-side encryption (HIPAA compliance)
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=s3_key,
            Body=file_data,
            ContentType=content_type,
            ServerSideEncryption='AES256'
        )
        
        # Generate presigned URL
        url = generate_presigned_url(BUCKET_NAME, s3_key)
        
        return {
            "bucket": BUCKET_NAME,
            "key": s3_key,
            "url": url
        }
    except ClientError as e:
        print(f"Error uploading file to S3: {e}")
        raise
