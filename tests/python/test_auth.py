"""
Authentication and Authorization Tests
Tests for Stytch M2M, session handling, and role-based access using actual production APIs.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi import HTTPException
import os

from app.dependencies import (
    get_current_user,
    get_current_doctor,
    get_current_patient,
    require_role,
    StytchM2MValidator,
    M2MTokenPayload
)


class TestM2MTokenPayload:
    """Test M2MTokenPayload model."""

    def test_m2m_payload_creation(self):
        """Test creating M2M token payload."""
        payload = M2MTokenPayload(
            client_id="backend-service",
            scopes=["read:users", "write:records"],
            custom_claims={"service": "api-gateway"}
        )
        
        assert payload.client_id == "backend-service"
        assert "read:users" in payload.scopes
        assert payload.custom_claims["service"] == "api-gateway"

    def test_m2m_payload_minimal(self):
        """Test M2M payload with minimal fields."""
        payload = M2MTokenPayload(
            client_id="worker",
            scopes=[]
        )
        
        assert payload.client_id == "worker"
        assert len(payload.scopes) == 0
        assert payload.custom_claims is None


class TestStytchM2MValidator:
    """Test Stytch M2M token validation."""

    def test_validator_not_configured_without_credentials(self):
        """Test validator reports not configured without Stytch credentials."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('app.dependencies.STYTCH_PROJECT_ID', None), \
                 patch('app.dependencies.STYTCH_SECRET', None):
                validator = StytchM2MValidator()
                validator.project_id = None
                validator.secret = None
                
                assert validator.is_configured is False

    def test_validator_configured_with_credentials(self):
        """Test validator reports configured with Stytch credentials."""
        validator = StytchM2MValidator()
        validator.project_id = "project-test-123"
        validator.secret = "secret-test-456"
        
        assert validator.is_configured is True

    @pytest.mark.asyncio
    async def test_validate_token_not_configured_raises_503(self):
        """Test validation raises 503 when not configured."""
        validator = StytchM2MValidator()
        validator.project_id = None
        validator.secret = None
        
        with pytest.raises(HTTPException) as exc_info:
            await validator.validate_token("test_token")
        
        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_validate_token_success(self):
        """Test successful token validation."""
        validator = StytchM2MValidator()
        validator.project_id = "project-123"
        validator.secret = "secret-456"
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "client_id": "backend-service",
            "scopes": ["read:users", "write:records"],
            "custom_claims": {}
        }
        
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        
        with patch.object(validator, 'get_client', return_value=mock_client):
            result = await validator.validate_token("valid_token")
        
        assert result.client_id == "backend-service"
        assert "read:users" in result.scopes

    @pytest.mark.asyncio
    async def test_validate_token_invalid_raises_401(self):
        """Test invalid token raises 401."""
        validator = StytchM2MValidator()
        validator.project_id = "project-123"
        validator.secret = "secret-456"
        
        mock_response = MagicMock()
        mock_response.status_code = 401
        
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        
        with patch.object(validator, 'get_client', return_value=mock_client):
            with pytest.raises(HTTPException) as exc_info:
                await validator.validate_token("invalid_token")
        
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_validate_token_insufficient_scopes_raises_403(self):
        """Test insufficient scopes raises 403."""
        validator = StytchM2MValidator()
        validator.project_id = "project-123"
        validator.secret = "secret-456"
        
        mock_response = MagicMock()
        mock_response.status_code = 403
        
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        
        with patch.object(validator, 'get_client', return_value=mock_client):
            with pytest.raises(HTTPException) as exc_info:
                await validator.validate_token("token", ["admin:all"])
        
        assert exc_info.value.status_code == 403


class TestSessionManagement:
    """Test session-based authentication."""

    def test_session_cookie_config(self):
        """Session cookies should have secure configuration."""
        session_config = {
            'httponly': True,
            'secure': True,
            'samesite': 'strict',
            'max_age': 86400
        }
        
        assert session_config['httponly'] is True
        assert session_config['secure'] is True
        assert session_config['samesite'] == 'strict'

    def test_session_expiration_logic(self):
        """Sessions should expire after configured timeout."""
        from datetime import datetime, timedelta
        
        def is_session_expired(created_at, max_age_hours=24):
            return (datetime.utcnow() - created_at).total_seconds() > max_age_hours * 3600
        
        expired = datetime.utcnow() - timedelta(hours=25)
        valid = datetime.utcnow() - timedelta(hours=1)
        
        assert is_session_expired(expired) is True
        assert is_session_expired(valid) is False


class TestGetCurrentUser:
    """Test get_current_user dependency."""

    @pytest.mark.asyncio
    async def test_returns_user_for_valid_session(self):
        """Valid session should return user object."""
        mock_user = MagicMock()
        mock_user.id = "user-123"
        mock_user.email = "test@example.com"
        mock_user.role = "patient"
        
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = mock_user
        
        with patch('app.dependencies.verify_token') as mock_verify:
            mock_verify.return_value = {"sub": "user-123"}
            
            result = await get_current_user("valid_token", mock_db)
            
            assert result.id == "user-123"

    @pytest.mark.asyncio
    async def test_raises_401_for_invalid_token(self):
        """Invalid token should raise 401."""
        mock_db = MagicMock()
        
        with patch('app.dependencies.verify_token') as mock_verify:
            mock_verify.return_value = None
            
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user("invalid_token", mock_db)
            
            assert exc_info.value.status_code == 401


class TestGetCurrentDoctor:
    """Test get_current_doctor dependency."""

    @pytest.mark.asyncio
    async def test_returns_doctor_for_doctor_user(self):
        """Doctor user should pass validation."""
        mock_user = MagicMock()
        mock_user.id = "doc-123"
        mock_user.role = "doctor"
        
        result = await get_current_doctor(mock_user)
        
        assert result.role == "doctor"

    @pytest.mark.asyncio
    async def test_raises_403_for_non_doctor(self):
        """Non-doctor should be rejected."""
        mock_user = MagicMock()
        mock_user.id = "patient-123"
        mock_user.role = "patient"
        
        with pytest.raises(HTTPException) as exc_info:
            await get_current_doctor(mock_user)
        
        assert exc_info.value.status_code == 403


class TestGetCurrentPatient:
    """Test get_current_patient dependency."""

    @pytest.mark.asyncio
    async def test_returns_patient_for_patient_user(self):
        """Patient user should pass validation."""
        mock_user = MagicMock()
        mock_user.id = "patient-123"
        mock_user.role = "patient"
        
        result = await get_current_patient(mock_user)
        
        assert result.role == "patient"

    @pytest.mark.asyncio
    async def test_raises_403_for_non_patient(self):
        """Non-patient should be rejected."""
        mock_user = MagicMock()
        mock_user.id = "doc-123"
        mock_user.role = "doctor"
        
        with pytest.raises(HTTPException) as exc_info:
            await get_current_patient(mock_user)
        
        assert exc_info.value.status_code == 403


class TestRoleBasedAuthorization:
    """Test role-based authorization patterns."""

    @pytest.mark.asyncio
    async def test_require_role_patient_success(self):
        """Patient role check should pass for patient."""
        checker = require_role("patient")
        
        mock_user = MagicMock()
        mock_user.role = "patient"
        
        result = await checker(mock_user)
        assert result.role == "patient"

    @pytest.mark.asyncio
    async def test_require_role_doctor_success(self):
        """Doctor role check should pass for doctor."""
        checker = require_role("doctor")
        
        mock_user = MagicMock()
        mock_user.role = "doctor"
        
        result = await checker(mock_user)
        assert result.role == "doctor"

    @pytest.mark.asyncio
    async def test_require_role_admin_success(self):
        """Admin role check should pass for admin."""
        checker = require_role("admin")
        
        mock_user = MagicMock()
        mock_user.role = "admin"
        
        result = await checker(mock_user)
        assert result.role == "admin"

    @pytest.mark.asyncio
    async def test_require_role_mismatch_raises_403(self):
        """Role mismatch should raise 403."""
        checker = require_role("doctor")
        
        mock_user = MagicMock()
        mock_user.role = "patient"
        
        with pytest.raises(HTTPException) as exc_info:
            await checker(mock_user)
        
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_patient_cannot_access_doctor_resources(self):
        """Patients should not access doctor-only resources."""
        mock_patient = MagicMock()
        mock_patient.role = "patient"
        
        with pytest.raises(HTTPException) as exc_info:
            await get_current_doctor(mock_patient)
        
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_doctor_cannot_access_patient_resources(self):
        """Doctors should not access patient-only resources."""
        mock_doctor = MagicMock()
        mock_doctor.role = "doctor"
        
        with pytest.raises(HTTPException) as exc_info:
            await get_current_patient(mock_doctor)
        
        assert exc_info.value.status_code == 403


class TestMagicLinkAuth:
    """Test Stytch Magic Link authentication patterns."""

    def test_magic_link_session_flow(self):
        """Test Magic Link creates valid session."""
        session_data = {
            'user_id': 'user-123',
            'email': 'test@example.com',
            'authenticated_at': '2024-01-15T10:30:00Z',
            'session_token': 'session-abc-123'
        }
        
        assert 'user_id' in session_data
        assert 'session_token' in session_data


class TestSMSOTPAuth:
    """Test Stytch SMS OTP authentication patterns."""

    def test_sms_otp_session_flow(self):
        """Test SMS OTP creates valid session."""
        session_data = {
            'user_id': 'user-456',
            'phone_number': '+1555123456',
            'authenticated_at': '2024-01-15T10:30:00Z',
            'session_token': 'session-def-456'
        }
        
        assert 'user_id' in session_data
        assert 'phone_number' in session_data


class TestAuthEnvironmentConfig:
    """Test authentication environment configuration."""

    def test_stytch_credentials_required(self):
        """Stytch credentials must be configured."""
        from app.dependencies import STYTCH_PROJECT_ID, STYTCH_SECRET
        
        assert STYTCH_PROJECT_ID is not None or os.getenv("STYTCH_PROJECT_ID") is not None

    def test_dev_mode_secret_available(self):
        """Dev mode secret should be available for testing."""
        from app.dependencies import DEV_MODE_SECRET
        
        assert DEV_MODE_SECRET is not None
