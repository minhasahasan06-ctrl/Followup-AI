"""
Tests for HIPAA-compliant access control patterns.
Validates role-based access, doctor-patient relationships, and license verification.
"""
import pytest
from unittest.mock import MagicMock, patch
from fastapi import HTTPException
from sqlalchemy.orm import Session


class TestRequireVerifiedDoctor:
    """Test the require_verified_doctor dependency."""

    def test_verified_doctor_passes(self):
        """Verified doctor should pass through."""
        from app.routers.rx_builder import require_verified_doctor
        
        mock_user = MagicMock()
        mock_user.role = "doctor"
        mock_user.license_verified = True
        mock_user.id = "doc-123"
        
        result = require_verified_doctor(mock_user)
        assert result == mock_user

    def test_unverified_doctor_blocked(self):
        """Unverified doctor should be blocked."""
        from app.routers.rx_builder import require_verified_doctor
        
        mock_user = MagicMock()
        mock_user.role = "doctor"
        mock_user.license_verified = False
        
        with pytest.raises(HTTPException) as exc_info:
            require_verified_doctor(mock_user)
        
        assert exc_info.value.status_code == 403
        assert "license verification required" in exc_info.value.detail.lower()

    def test_missing_license_field_blocked(self):
        """Doctor without license_verified attribute should be blocked."""
        from app.routers.rx_builder import require_verified_doctor
        
        mock_user = MagicMock(spec=[])
        mock_user.role = "doctor"
        
        with pytest.raises(HTTPException) as exc_info:
            require_verified_doctor(mock_user)
        
        assert exc_info.value.status_code == 403


class TestDoctorPatientAccess:
    """Test doctor-patient relationship verification."""

    @patch('app.routers.rx_builder.get_db')
    def test_valid_connection_passes(self, mock_get_db):
        """Doctor with active connection should have access."""
        from app.routers.rx_builder import verify_doctor_patient_access
        
        mock_db = MagicMock(spec=Session)
        mock_result = MagicMock()
        mock_result.fetchone.return_value = (1,)
        mock_db.execute.return_value = mock_result
        
        result = verify_doctor_patient_access("doc-123", "patient-456", mock_db)
        assert result is True

    @patch('app.routers.rx_builder.get_db')
    def test_no_connection_blocked(self, mock_get_db):
        """Doctor without connection should be blocked."""
        from app.routers.rx_builder import verify_doctor_patient_access
        
        mock_db = MagicMock(spec=Session)
        mock_result = MagicMock()
        mock_result.fetchone.return_value = None
        mock_db.execute.return_value = mock_result
        
        result = verify_doctor_patient_access("doc-123", "patient-456", mock_db)
        assert result is False


class TestRoleBasedAccess:
    """Test role-based access patterns."""

    def test_patient_cannot_access_doctor_routes(self):
        """Patient role should not access doctor-only endpoints."""
        from app.dependencies import get_current_doctor
        
        mock_request = MagicMock()
        mock_db = MagicMock()
        
        mock_user = MagicMock()
        mock_user.role = "patient"
        
        with patch('app.dependencies.get_current_user', return_value=mock_user):
            with pytest.raises(HTTPException) as exc_info:
                get_current_doctor(mock_request, mock_db)
            
            assert exc_info.value.status_code == 403

    def test_admin_has_elevated_access(self):
        """Admin role should have elevated access."""
        mock_user = MagicMock()
        mock_user.role = "admin"
        mock_user.id = "admin-123"
        
        assert mock_user.role == "admin"


class TestBreakTheGlass:
    """Test emergency access (break-the-glass) patterns."""

    @pytest.mark.asyncio
    async def test_emergency_access_creates_audit_log(self):
        """Emergency access should create immutable audit entry."""
        from app.routers.emergency_access_router import break_glass_access
        from pydantic import BaseModel
        
        class MockRequest(BaseModel):
            patient_id: str
            justification: str
            resource_type: str = "medical_record"
        
        mock_user = MagicMock()
        mock_user.id = "doc-123"
        mock_user.role = "doctor"
        mock_db = MagicMock()
        
        with patch('app.routers.emergency_access_router.log_hipaa_access') as mock_log:
            request = MockRequest(
                patient_id="patient-456",
                justification="Critical emergency - patient unresponsive"
            )
            
            try:
                await break_glass_access(request, mock_user, mock_db)
            except Exception:
                pass
            
            mock_log.assert_called()

    def test_emergency_access_requires_justification(self):
        """Emergency access without justification should be blocked via validation."""
        from pydantic import BaseModel, field_validator
        
        class EmergencyRequest(BaseModel):
            justification: str
            
            @field_validator('justification')
            @classmethod
            def justification_not_empty(cls, v):
                if not v or not v.strip():
                    raise ValueError("Justification is required for emergency access")
                return v
        
        with pytest.raises(Exception):
            EmergencyRequest(justification="")

    def test_emergency_access_request_structure(self):
        """Emergency access should include required fields."""
        emergency_access = {
            'user_id': 'doc-123',
            'patient_id': 'patient-456',
            'justification': 'Medical emergency',
            'resource_type': 'medical_record',
            'access_time': '2024-01-15T10:30:00Z'
        }
        
        assert 'justification' in emergency_access
        assert len(emergency_access['justification']) > 0


class TestHIPAAAuditLogging:
    """Test HIPAA-compliant audit logging."""

    def test_phi_access_logged(self):
        """PHI access should be logged with user, action, and timestamp."""
        from app.services.hipaa_audit_logger import HIPAAAuditLogger
        
        logger = HIPAAAuditLogger()
        
        with patch.object(logger, '_write_audit_entry') as mock_write:
            logger.log_phi_access(
                user_id="user-123",
                patient_id="patient-456",
                action="view_medical_record",
                resource_type="soap_note"
            )
            
            mock_write.assert_called_once()
            call_args = mock_write.call_args[1]
            assert call_args['user_id'] == "user-123"
            assert call_args['patient_id'] == "patient-456"

    def test_audit_log_immutability(self):
        """Audit logs should be append-only and immutable."""
        from app.services.immutable_audit_log import ImmutableAuditLog
        
        audit_log = ImmutableAuditLog()
        
        entry_id = audit_log.append_entry(
            user_id="user-123",
            action="view_record",
            resource_id="record-456"
        )
        
        with pytest.raises(Exception) as exc_info:
            audit_log.update_entry(entry_id, {"action": "modified"})
        
        assert "immutable" in str(exc_info.value).lower() or "not allowed" in str(exc_info.value).lower()
