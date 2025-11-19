"""
Pytest configuration for guided exam API tests
"""

import pytest
import sys
import os

# Set mock AWS credentials BEFORE importing any app modules
# This prevents InvalidRegionError during import
# Override any existing malformed AWS_REGION
os.environ["AWS_REGION"] = "us-east-1"
os.environ["AWS_ACCESS_KEY_ID"] = "test_access_key"
os.environ["AWS_SECRET_ACCESS_KEY"] = "test_secret_key"
os.environ["AWS_S3_BUCKET_NAME"] = "test-bucket"

# Add parent directory to path to import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure pytest-asyncio
pytest_plugins = ('pytest_asyncio',)


@pytest.fixture(scope="session")
def anyio_backend():
    """Use asyncio backend for async tests"""
    return 'asyncio'
