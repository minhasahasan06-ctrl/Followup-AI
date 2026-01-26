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

# Set test DATABASE_URL to prevent validation errors during import
# Individual tests will override this with their own test databases
# Use SQLite for testing - simpler and doesn't require PostgreSQL
os.environ["DATABASE_URL"] = "sqlite:///./test_import.db"

# Set other required environment variables for test imports
os.environ.setdefault("OPENAI_API_KEY", "test_openai_key")
os.environ.setdefault("DEV_MODE_SECRET", "test_dev_mode_secret_min_32_characters_long")
os.environ.setdefault("SESSION_SECRET", "test_session_secret")

# Add parent directory to path to import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure pytest-asyncio
pytest_plugins = ('pytest_asyncio',)


@pytest.fixture(scope="session")
def anyio_backend():
    """Use asyncio backend for async tests"""
    return 'asyncio'
