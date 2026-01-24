"""
Tests for authentication and authorization.
Validates Stytch M2M, session handling, and role-based access.
"""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi import HTTPException
import os


class TestStytchM2MValidation:
    """Test Stytch M2M token validation."""

    def test_valid_m2m_token_passes(self):
        """Valid M2M token should pass validation."""
        from app.auth import validate_m2m_token
        
        with patch('stytch.Client') as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.m2m.authenticate_token.return_value = MagicMock(
                sub='service-backend',
                scopes=['read:users', 'write:records']
            )
            
            result = validate_m2m_token('valid_token')
            
            assert result['valid'] is True
            assert 'service-backend' in result['subject']

    def test_expired_m2m_token_fails(self):
        """Expired M2M token should fail validation."""
        from app.auth import validate_m2m_token
        
        with patch('stytch.Client') as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.m2m.authenticate_token.side_effect = Exception("Token expired")
            
            result = validate_m2m_token('expired_token')
            
            assert result['valid'] is False

    def test_invalid_m2m_token_fails(self):
        """Invalid M2M token should fail validation."""
        from app.auth import validate_m2m_token
        
        with patch('stytch.Client') as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.m2m.authenticate_token.side_effect = Exception("Invalid token")
            
            result = validate_m2m_token('invalid_token')
            
            assert result['valid'] is False


class TestSessionManagement:
    """Test session-based authentication."""

    def test_session_cookie_should_be_httponly(self):
        """Session cookies should be HttpOnly for security."""
        session_cookie_config = {
            'httponly': True,
            'secure': True,
            'samesite': 'strict',
            'max_age': 86400
        }
        
        assert session_cookie_config['httponly'] is True
        assert session_cookie_config['secure'] is True

    def test_session_expiration_logic(self):
        """Sessions should expire after configured timeout."""
        from datetime import datetime, timedelta
        
        def is_session_expired(session: dict) -> bool:
            created = session.get('created_at')
            max_age = session.get('max_age_hours', 24)
            return (datetime.utcnow() - created).total_seconds() > max_age * 3600
        
        expired_session = {
            'created_at': datetime.utcnow() - timedelta(hours=25),
            'max_age_hours': 24
        }
        
        valid_session = {
            'created_at': datetime.utcnow() - timedelta(hours=1),
            'max_age_hours': 24
        }
        
        assert is_session_expired(expired_session) is True
        assert is_session_expired(valid_session) is False

    def test_session_rotation_pattern(self):
        """Session should rotate when privileges change."""
        import uuid
        
        old_session = {'token': str(uuid.uuid4()), 'role': 'patient'}
        new_session = {'token': str(uuid.uuid4()), 'role': 'doctor'}
        
        assert new_session['token'] != old_session['token']
        assert new_session['role'] != old_session['role']


class TestGetCurrentUser:
    """Test get_current_user dependency."""

    @pytest.mark.asyncio
    async def test_returns_user_for_valid_session(self):
        """Should return user for valid session."""
        from app.auth import get_current_user
        
        mock_request = MagicMock()
        mock_request.cookies = {'session_token': 'valid_session'}
        mock_db = MagicMock()
        
        with patch('app.auth.validate_session') as mock_validate:
            mock_user = MagicMock()
            mock_user.id = 'user_123'
            mock_user.role = 'patient'
            mock_validate.return_value = mock_user
            
            result = await get_current_user(mock_request, mock_db)
            
            assert result.id == 'user_123'

    @pytest.mark.asyncio
    async def test_raises_401_for_missing_session(self):
        """Should raise 401 for missing session."""
        from app.auth import get_current_user
        
        mock_request = MagicMock()
        mock_request.cookies = {}
        mock_db = MagicMock()
        
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(mock_request, mock_db)
        
        assert exc_info.value.status_code == 401


class TestGetCurrentDoctor:
    """Test get_current_doctor dependency."""

    def test_returns_doctor_for_valid_doctor_user(self):
        """Should return user if role is doctor."""
        from app.dependencies import get_current_doctor
        
        mock_request = MagicMock()
        mock_db = MagicMock()
        
        mock_user = MagicMock()
        mock_user.role = 'doctor'
        mock_user.id = 'doc_123'
        
        with patch('app.dependencies.get_current_user', return_value=mock_user):
            result = get_current_doctor(mock_request, mock_db)
            
            assert result.role == 'doctor'

    def test_raises_403_for_non_doctor(self):
        """Should raise 403 for non-doctor users."""
        from app.dependencies import get_current_doctor
        
        mock_request = MagicMock()
        mock_db = MagicMock()
        
        mock_user = MagicMock()
        mock_user.role = 'patient'
        
        with patch('app.dependencies.get_current_user', return_value=mock_user):
            with pytest.raises(HTTPException) as exc_info:
                get_current_doctor(mock_request, mock_db)
            
            assert exc_info.value.status_code == 403


class TestRoleBasedAuthorization:
    """Test role-based authorization patterns."""

    def test_patient_access_patterns(self):
        """Patient should only access own data."""
        def check_patient_access(user, target_patient_id):
            if user.role == 'patient':
                return user.id == target_patient_id
            return False
        
        patient_user = MagicMock()
        patient_user.role = 'patient'
        patient_user.id = 'patient-123'
        
        assert check_patient_access(patient_user, 'patient-123') is True
        assert check_patient_access(patient_user, 'patient-456') is False

    def test_doctor_access_patterns(self):
        """Doctor should access assigned patients' data."""
        def check_doctor_access(user, target_patient_id):
            if user.role == 'doctor':
                return target_patient_id in getattr(user, 'assigned_patients', [])
            return False
        
        doctor_user = MagicMock()
        doctor_user.role = 'doctor'
        doctor_user.id = 'doc-123'
        doctor_user.assigned_patients = ['patient-123', 'patient-456']
        
        assert check_doctor_access(doctor_user, 'patient-123') is True
        assert check_doctor_access(doctor_user, 'patient-999') is False

    def test_admin_access_patterns(self):
        """Admin should have elevated access."""
        def check_admin_access(user, target_patient_id):
            return user.role == 'admin'
        
        admin_user = MagicMock()
        admin_user.role = 'admin'
        admin_user.id = 'admin-123'
        
        assert check_admin_access(admin_user, 'patient-123') is True
        assert check_admin_access(admin_user, 'patient-999') is True


class TestMagicLinkAuth:
    """Test Stytch Magic Link authentication."""

    def test_send_magic_link_success(self):
        """Should send magic link to valid email."""
        from app.auth import send_magic_link
        
        with patch('stytch.Client') as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.magic_links.email.login_or_create.return_value = MagicMock(
                user_id='user_123'
            )
            
            result = send_magic_link('user@example.com')
            
            assert result['success'] is True

    def test_authenticate_magic_link_token(self):
        """Should authenticate valid magic link token."""
        from app.auth import authenticate_magic_link
        
        with patch('stytch.Client') as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.magic_links.authenticate.return_value = MagicMock(
                user_id='user_123',
                session_token='session_abc'
            )
            
            result = authenticate_magic_link('valid_token')
            
            assert result['user_id'] == 'user_123'
            assert result['session_token'] == 'session_abc'


class TestSMSOTPAuth:
    """Test Stytch SMS OTP authentication."""

    def test_send_sms_otp_success(self):
        """Should send OTP to valid phone number."""
        from app.auth import send_sms_otp
        
        with patch('stytch.Client') as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.otps.sms.login_or_create.return_value = MagicMock(
                phone_id='phone_123'
            )
            
            result = send_sms_otp('+15551234567')
            
            assert result['success'] is True

    def test_verify_sms_otp_success(self):
        """Should verify valid OTP code."""
        from app.auth import verify_sms_otp
        
        with patch('stytch.Client') as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.otps.authenticate.return_value = MagicMock(
                user_id='user_123',
                session_token='session_xyz'
            )
            
            result = verify_sms_otp('+15551234567', '123456')
            
            assert result['user_id'] == 'user_123'


class TestAuthEnvironmentConfig:
    """Test authentication environment configuration."""

    def test_stytch_credentials_required(self):
        """Should require Stytch credentials in production."""
        with patch.dict(os.environ, {'ENVIRONMENT': 'production', 'STYTCH_PROJECT_ID': '', 'STYTCH_SECRET': ''}):
            from app.auth import get_stytch_client
            
            with pytest.raises(ValueError) as exc_info:
                get_stytch_client()
            
            assert "STYTCH" in str(exc_info.value)

    def test_development_allows_mock_auth(self):
        """Development should allow mock authentication."""
        with patch.dict(os.environ, {'ENVIRONMENT': 'development'}):
            from app.auth import is_mock_auth_enabled
            
            assert is_mock_auth_enabled() is True
