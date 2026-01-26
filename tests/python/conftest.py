"""
Pytest configuration for Python unit tests.

This conftest.py handles environment-dependent test skipping:
- Tests requiring DATABASE_URL are completely ignored during collection
- Uses pytest_ignore_collect to prevent import errors before collection
- Provides reusable fixtures for mocking database connections
"""

import os
import pytest
from unittest.mock import MagicMock

# Check if DATABASE_URL is available BEFORE any app imports
DATABASE_URL_AVAILABLE = bool(os.environ.get("DATABASE_URL"))

# Files that require DATABASE_URL for their imports - these will be
# completely ignored during collection when DATABASE_URL is not set
DB_DEPENDENT_TEST_FILES = {
    "test_access_control.py",
    "test_auth.py", 
    "test_governance_service.py",
    "test_payments.py",
}


def pytest_ignore_collect(collection_path, config):
    """
    Ignore database-dependent test files BEFORE import/collection.
    This prevents ValueError from app.database import validation.
    
    Returns True to ignore the file, False/None to collect it.
    """
    if DATABASE_URL_AVAILABLE:
        return False
    
    filename = os.path.basename(str(collection_path))
    if filename in DB_DEPENDENT_TEST_FILES:
        return True
    
    return False


def pytest_configure(config):
    """
    Configure pytest with custom markers.
    """
    config.addinivalue_line(
        "markers", 
        "requires_database: mark test as requiring DATABASE_URL"
    )
    config.addinivalue_line(
        "markers",
        "integration: mark test as an integration test"
    )


@pytest.fixture
def mock_db_session():
    """Provide a mock database session for unit tests."""
    session = MagicMock()
    session.query.return_value.filter.return_value.first.return_value = None
    session.query.return_value.filter.return_value.all.return_value = []
    session.add = MagicMock()
    session.commit = MagicMock()
    session.rollback = MagicMock()
    session.close = MagicMock()
    return session


@pytest.fixture
def mock_settings():
    """Provide mock settings for unit tests."""
    settings = MagicMock()
    settings.DATABASE_URL = "postgresql://test:test@localhost:5432/test"
    settings.OPENAI_API_KEY = "test-openai-key"
    settings.DEV_MODE_SECRET = "test-dev-secret-32-characters-long"
    settings.ENV = "test"
    settings.validate_database_url = MagicMock()
    settings.is_dev_mode_enabled = MagicMock(return_value=True)
    return settings


@pytest.fixture
def skip_if_no_database():
    """
    Fixture to skip individual tests when DATABASE_URL is not available.
    Usage: 
        def test_something(skip_if_no_database):
            # test code that needs database
    """
    if not DATABASE_URL_AVAILABLE:
        pytest.skip("DATABASE_URL not configured")
