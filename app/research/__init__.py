"""
Research-Only Module
======================
This package contains research-only code that is NOT approved for clinical production use.

IMPORTANT: All code in this package:
1. Requires USE_RESEARCH_EMBEDDINGS=true to be enabled
2. Must not be used for clinical decision support
3. Has not completed validation protocol for production deployment
4. Is subject to IRB approval requirements for any human data use

See individual modules for specific research-only notices.
"""

import os
import logging

logger = logging.getLogger(__name__)

USE_RESEARCH_EMBEDDINGS = os.getenv("USE_RESEARCH_EMBEDDINGS", "false").lower() == "true"


def require_research_flag(func):
    """Decorator that enforces USE_RESEARCH_EMBEDDINGS=true"""
    def wrapper(*args, **kwargs):
        if not USE_RESEARCH_EMBEDDINGS:
            raise RuntimeError(
                f"Research function '{func.__name__}' requires USE_RESEARCH_EMBEDDINGS=true. "
                "This functionality is not approved for clinical production use."
            )
        logger.info(f"Research function called: {func.__name__}")
        return func(*args, **kwargs)
    return wrapper


def is_research_enabled() -> bool:
    """Check if research features are enabled"""
    return USE_RESEARCH_EMBEDDINGS
