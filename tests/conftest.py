"""
Pytest configuration for guided exam API tests
"""

import pytest
import sys
import os

# Set mock GCP credentials BEFORE importing any app modules
os.environ["GCP_PROJECT_ID"] = "test-project"
os.environ["GCP_REGION"] = "us-central1"
os.environ["GCP_STORAGE_BUCKET"] = "test-bucket"

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
