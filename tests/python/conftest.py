"""
Pytest configuration for Python unit tests.

This conftest.py handles environment-dependent test skipping:
- Tests requiring DATABASE_URL are skipped when database is unavailable
- Provides reusable fixtures for mocking database connections
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# Check if DATABASE_URL is available before imports that require it
DATABASE_URL_AVAILABLE = bool(os.environ.get("DATABASE_URL"))

# Files that require DATABASE_URL for their imports
DB_DEPENDENT_TEST_FILES = [
    "test_access_control.py",
    "test_auth.py", 
    "test_governance_service.py",
    "test_payments.py",
]


def pytest_collection_modifyitems(config, items):
    """
    Skip database-dependent tests when DATABASE_URL is not configured.
    This prevents collection errors from imports that validate DATABASE_URL.
    """
    if DATABASE_URL_AVAILABLE:
        return
    
    skip_no_db = pytest.mark.skip(
        reason="DATABASE_URL not configured - skipping database-dependent tests"
    )
    
    for item in items:
        # Check if the test file requires database
        test_file = os.path.basename(item.fspath)
        if test_file in DB_DEPENDENT_TEST_FILES:
            item.add_marker(skip_no_db)


def pytest_configure(config):
    """
    Configure pytest with custom markers and environment setup.
    """
    config.addinivalue_line(
        "markers", 
        "requires_database: mark test as requiring DATABASE_URL"
    )
    config.addinivalue_line(
        "markers",
        "integration: mark test as an integration test"
    )


# Skip imports that require DATABASE_URL if not available
if not DATABASE_URL_AVAILABLE:
    # Provide mock modules to prevent import errors during collection
    class MockSettings:
        DATABASE_URL = None
        OPENAI_API_KEY = "test-key"
        DEV_MODE_SECRET = None
        ENV = "test"
        
        def validate_database_url(self):
            pass
        
        def is_dev_mode_enabled(self):
            return False
    
    # Create mock module for app.config
    mock_config_module = MagicMock()
    mock_config_module.settings = MockSettings()
    sys.modules['app.config'] = mock_config_module
    
    # Create mock module for app.database
    mock_db_module = MagicMock()
    mock_db_module.get_db = MagicMock(return_value=MagicMock())
    mock_db_module.Session = MagicMock()
    sys.modules['app.database'] = mock_db_module


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
