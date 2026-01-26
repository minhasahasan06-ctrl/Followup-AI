"""
Pytest configuration for Python unit tests.

This conftest.py handles environment-dependent test skipping:
- Tests requiring DATABASE_URL are completely ignored during collection
- Tests requiring asyncpg/llama_index are skipped when modules unavailable
- Uses pytest_ignore_collect to prevent import errors before collection
- Provides reusable fixtures for mocking database connections
"""

import os
import sys
import pytest
from unittest.mock import MagicMock


# =============================================================================
# Environment and Module Availability Checks
# =============================================================================
# These checks run BEFORE any app imports to determine which tests to skip

DATABASE_URL_AVAILABLE = bool(os.environ.get("DATABASE_URL"))

def _check_module_available(module_name: str) -> bool:
    """Check if a module is importable without actually importing it."""
    try:
        import importlib.util
        spec = importlib.util.find_spec(module_name)
        return spec is not None
    except (ImportError, ModuleNotFoundError):
        return False

ASYNCPG_AVAILABLE = _check_module_available("asyncpg")
LLAMA_INDEX_AVAILABLE = _check_module_available("llama_index")


# =============================================================================
# Test File Skip Configuration
# =============================================================================
# Maps test files to their required dependencies for intelligent skipping

# Tests that require DATABASE_URL (import app.database at module level)
DB_DEPENDENT_TEST_FILES = {
    "test_access_control.py",
    "test_auth.py", 
    "test_governance_service.py",
    "test_payments.py",
}

# Tests that require asyncpg (for async database operations)
ASYNCPG_DEPENDENT_TEST_FILES = {
    "test_memory_service.py",  # Uses memory_db which imports asyncpg
    "test_integration_rag.py",  # Integration tests with memory services
    "test_performance.py",  # Performance tests with async DB
}

# Tests that require llama_index (for RAG/vector operations)
LLAMA_INDEX_DEPENDENT_TEST_FILES = {
    "test_memory_service.py",  # Uses llama_memory_service
    "test_integration_rag.py",  # RAG integration tests
}


def pytest_ignore_collect(collection_path, config):
    """
    Ignore test files BEFORE import/collection based on available dependencies.
    This prevents ModuleNotFoundError and ValueError during collection.
    
    Returns True to ignore the file, False/None to collect it.
    """
    filename = os.path.basename(str(collection_path))
    
    # Skip database-dependent tests when DATABASE_URL is missing
    if not DATABASE_URL_AVAILABLE and filename in DB_DEPENDENT_TEST_FILES:
        return True
    
    # Skip asyncpg-dependent tests when asyncpg is not installed
    if not ASYNCPG_AVAILABLE and filename in ASYNCPG_DEPENDENT_TEST_FILES:
        return True
    
    # Skip llama_index-dependent tests when llama_index is not installed
    if not LLAMA_INDEX_AVAILABLE and filename in LLAMA_INDEX_DEPENDENT_TEST_FILES:
        return True
    
    return False


def pytest_configure(config):
    """
    Configure pytest with custom markers and report skipped dependencies.
    """
    config.addinivalue_line(
        "markers", 
        "requires_database: mark test as requiring DATABASE_URL"
    )
    config.addinivalue_line(
        "markers",
        "requires_asyncpg: mark test as requiring asyncpg module"
    )
    config.addinivalue_line(
        "markers",
        "requires_llama_index: mark test as requiring llama_index module"
    )
    config.addinivalue_line(
        "markers",
        "integration: mark test as an integration test"
    )


def pytest_report_header(config):
    """Report which optional dependencies are available."""
    lines = []
    if not DATABASE_URL_AVAILABLE:
        lines.append("⚠️  DATABASE_URL not set - skipping database-dependent tests")
    if not ASYNCPG_AVAILABLE:
        lines.append("⚠️  asyncpg not installed - skipping async database tests")
    if not LLAMA_INDEX_AVAILABLE:
        lines.append("⚠️  llama_index not installed - skipping RAG tests")
    return lines if lines else None


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
