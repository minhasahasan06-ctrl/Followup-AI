"""
Authentication Module for Followup AI

Auth flow:
1. Express backend authenticates users via Stytch (magic links, sessions)
2. Express calls Python FastAPI using Stytch M2M tokens or DEV_MODE_SECRET JWTs
3. Python validates tokens via app/dependencies.py

Note: Auth0 support has been removed. Use Stytch for all authentication.
"""

from app.dependencies import get_current_user

__all__ = [
    "get_current_user",
]
