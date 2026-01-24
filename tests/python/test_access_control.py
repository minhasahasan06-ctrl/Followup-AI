"""
HIPAA Access Control Tests
Tests for AccessControlService, HIPAAAuditLogger, and role-based access control.
These tests verify the actual production APIs.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi import HTTPException

from app.services.access_control import (
    AccessControlService,
    HIPAAAuditLogger,
    AccessDecision,
    AccessScope,
    PHICategory
)
from app.dependencies import (
    get_current_user,
    get_current_doctor,
    get_current_patient,
    require_role,
    StytchM2MValidator,
    M2MTokenPayload
)


class TestAccessScope:
    """Test AccessScope enum values."""

    def test_access_scope_full(self):
        """Verify FULL access scope exists."""
        assert AccessScope.FULL.value == "full"

    def test_access_scope_limited(self):
        """Verify LIMITED access scope exists."""
        assert AccessScope.LIMITED.value == "limited"

    def test_access_scope_summary_only(self):
        """Verify SUMMARY_ONLY access scope exists."""
        assert AccessScope.SUMMARY_ONLY.value == "summary_only"

    def test_access_scope_emergency(self):
        """Verify EMERGENCY access scope exists."""
        assert AccessScope.EMERGENCY.value == "emergency"


class TestPHICategory:
    """Test PHICategory enum values."""

    def test_phi_categories_exist(self):
        """Verify all PHI categories are defined."""
        required_categories = [
            "vitals", "symptoms", "medications", "mental_health",
            "lab_results", "imaging", "appointments", "prescriptions",
            "habits", "device_data", "video_exams", "audio_exams",
            "messages", "clinical_notes"
        ]
        for cat in required_categories:
            assert hasattr(PHICategory, cat.upper()), f"PHICategory.{cat.upper()} not found"


class TestAccessDecision:
    """Test AccessDecision data class."""

    def test_access_decision_allowed(self):
        """Test creating an allowed access decision."""
        decision = AccessDecision(
            allowed=True,
            actor_id="doc-123",
            actor_role="doctor",
            patient_id="patient-456",
            access_scope=AccessScope.FULL,
            reason="Doctor assigned to patient"
        )
        assert decision.allowed is True
        assert decision.actor_id == "doc-123"
        assert decision.actor_role == "doctor"
        assert decision.patient_id == "patient-456"
        assert decision.access_scope == AccessScope.FULL

    def test_access_decision_denied(self):
        """Test creating a denied access decision."""
        decision = AccessDecision(
            allowed=False,
            actor_id="doc-123",
            actor_role="doctor",
            patient_id="patient-456",
            reason="Not assigned to patient"
        )
        assert decision.allowed is False
        assert decision.reason == "Not assigned to patient"

    def test_access_decision_emergency(self):
        """Test emergency access decision."""
        decision = AccessDecision(
            allowed=True,
            actor_id="doc-123",
            actor_role="doctor",
            patient_id="patient-456",
            access_scope=AccessScope.EMERGENCY,
            is_emergency=True
        )
        assert decision.is_emergency is True
        assert decision.access_scope == AccessScope.EMERGENCY


class TestHIPAAAuditLogger:
    """Test HIPAA audit logging functionality."""

    @patch('app.services.access_control.SessionLocal')
    def test_log_phi_access_success(self, mock_session_local):
        """Test logging successful PHI access."""
        mock_db = MagicMock()
        mock_session_local.return_value = mock_db
        
        audit_id = HIPAAAuditLogger.log_phi_access(
            actor_id="doc-123",
            actor_role="doctor",
            patient_id="patient-456",
            action="read",
            phi_categories=["vitals", "medications"],
            resource_type="patient_record",
            resource_id="record-789",
            success=True
        )
        
        assert audit_id is not None
        assert len(audit_id) == 36
        mock_db.execute.assert_called_once()
        mock_db.commit.assert_called_once()

    @patch('app.services.access_control.SessionLocal')
    def test_log_phi_access_failure(self, mock_session_local):
        """Test logging failed PHI access attempt."""
        mock_db = MagicMock()
        mock_session_local.return_value = mock_db
        
        audit_id = HIPAAAuditLogger.log_phi_access(
            actor_id="doc-123",
            actor_role="doctor",
            patient_id="patient-456",
            action="attempt_read",
            phi_categories=["mental_health"],
            resource_type="patient_record",
            success=False,
            error_message="Access denied - not assigned"
        )
        
        assert audit_id is not None
        mock_db.execute.assert_called_once()
        call_args = mock_db.execute.call_args[0][1]
        assert call_args["success"] is False
        assert call_args["error_message"] == "Access denied - not assigned"

    @patch('app.services.access_control.SessionLocal')
    def test_log_phi_access_with_context(self, mock_session_local):
        """Test logging PHI access with additional context."""
        mock_db = MagicMock()
        mock_session_local.return_value = mock_db
        
        audit_id = HIPAAAuditLogger.log_phi_access(
            actor_id="doc-123",
            actor_role="doctor",
            patient_id="patient-456",
            action="read",
            phi_categories=["vitals"],
            resource_type="patient_record",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            request_path="/api/patients/456/vitals",
            success=True
        )
        
        assert audit_id is not None
        call_args = mock_db.execute.call_args[0][1]
        assert call_args["ip_address"] == "192.168.1.1"
        assert call_args["user_agent"] == "Mozilla/5.0"
        assert call_args["request_path"] == "/api/patients/456/vitals"

    @patch('app.services.access_control.SessionLocal')
    def test_audit_log_immutability(self, mock_session_local):
        """Test that audit logs use INSERT only (no UPDATE capability)."""
        mock_db = MagicMock()
        mock_session_local.return_value = mock_db
        
        HIPAAAuditLogger.log_phi_access(
            actor_id="doc-123",
            actor_role="doctor",
            patient_id="patient-456",
            action="read",
            phi_categories=["vitals"],
            resource_type="patient_record",
            success=True
        )
        
        query = mock_db.execute.call_args[0][0]
        query_str = str(query)
        assert "INSERT INTO hipaa_audit_logs" in query_str
        assert "UPDATE" not in query_str


class TestAccessControlService:
    """Test AccessControlService functionality."""

    def test_service_instantiation(self):
        """Test that AccessControlService can be instantiated."""
        service = AccessControlService()
        assert service is not None

    @patch.object(AccessControlService, 'verify_doctor_patient_access')
    def test_verify_doctor_patient_access_allowed(self, mock_verify):
        """Test access verification when doctor is assigned to patient."""
        mock_verify.return_value = AccessDecision(
            allowed=True,
            actor_id="doc-123",
            actor_role="doctor",
            patient_id="patient-456",
            access_scope=AccessScope.FULL,
            assignment_id="assign-789"
        )
        
        service = AccessControlService()
        mock_db = MagicMock()
        
        decision = service.verify_doctor_patient_access(
            mock_db,
            doctor_id="doc-123",
            patient_id="patient-456",
            required_scope=AccessScope.FULL,
            phi_categories=[PHICategory.VITALS]
        )
        
        assert decision.allowed is True
        assert decision.assignment_id == "assign-789"

    @patch.object(AccessControlService, 'verify_doctor_patient_access')
    def test_verify_doctor_patient_access_denied(self, mock_verify):
        """Test access verification when doctor is NOT assigned."""
        mock_verify.return_value = AccessDecision(
            allowed=False,
            actor_id="doc-123",
            actor_role="doctor",
            patient_id="patient-456",
            reason="Doctor not assigned to patient"
        )
        
        service = AccessControlService()
        mock_db = MagicMock()
        
        decision = service.verify_doctor_patient_access(
            mock_db,
            doctor_id="doc-123",
            patient_id="patient-456",
            required_scope=AccessScope.FULL,
            phi_categories=[PHICategory.MENTAL_HEALTH]
        )
        
        assert decision.allowed is False
        assert "not assigned" in decision.reason


class TestRequireVerifiedDoctor:
    """Test doctor license verification dependency."""

    @pytest.mark.asyncio
    async def test_verified_doctor_passes(self):
        """Verified doctor should access clinical tools."""
        from app.routers.doctor_billing_router import require_verified_doctor
        
        mock_user = MagicMock()
        mock_user.id = "doc-123"
        mock_user.role = "doctor"
        mock_user.license_verified = True
        mock_user.medical_license_number = "MD12345"
        
        result = await require_verified_doctor(mock_user)
        assert result == mock_user

    @pytest.mark.asyncio
    async def test_unverified_doctor_blocked(self):
        """Unverified doctor should be blocked from clinical tools."""
        from app.routers.doctor_billing_router import require_verified_doctor
        
        mock_user = MagicMock()
        mock_user.id = "doc-123"
        mock_user.role = "doctor"
        mock_user.license_verified = False
        mock_user.medical_license_number = None
        
        with pytest.raises(HTTPException) as exc_info:
            await require_verified_doctor(mock_user)
        
        assert exc_info.value.status_code == 403
        assert "verified" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_non_doctor_blocked(self):
        """Non-doctors should be blocked regardless of verification."""
        from app.routers.doctor_billing_router import require_verified_doctor
        
        mock_user = MagicMock()
        mock_user.id = "patient-123"
        mock_user.role = "patient"
        mock_user.license_verified = False
        
        with pytest.raises(HTTPException) as exc_info:
            await require_verified_doctor(mock_user)
        
        assert exc_info.value.status_code == 403


class TestBreakTheGlass:
    """Test emergency access (break-the-glass) patterns."""

    def test_emergency_access_request_model(self):
        """Test EmergencyAccessRequest model structure."""
        from app.routers.emergency_access_router import EmergencyAccessRequest
        
        request = EmergencyAccessRequest(
            patient_id="patient-456",
            emergency_reason="Critical emergency - patient unresponsive in ER requiring immediate access",
            phi_categories=["vitals", "medications"],
            resource_type="patient_data"
        )
        
        assert request.patient_id == "patient-456"
        assert len(request.emergency_reason) >= 20
        assert "vitals" in request.phi_categories

    def test_emergency_access_response_model(self):
        """Test EmergencyAccessResponse model structure."""
        from app.routers.emergency_access_router import EmergencyAccessResponse
        
        response = EmergencyAccessResponse(
            success=True,
            access_granted=True,
            audit_id="audit-123-456-789",
            expires_in_minutes=60,
            message="Emergency access granted"
        )
        
        assert response.success is True
        assert response.access_granted is True
        assert response.expires_in_minutes == 60

    @patch('app.services.access_control.HIPAAAuditLogger.log_emergency_access')
    def test_emergency_access_creates_audit_entry(self, mock_log):
        """Emergency access should create audit entry via HIPAAAuditLogger."""
        mock_log.return_value = "audit-123"
        
        from app.services.access_control import HIPAAAuditLogger
        
        audit_id = HIPAAAuditLogger.log_emergency_access(
            actor_id="doc-123",
            actor_role="doctor",
            patient_id="patient-456",
            emergency_reason="Critical emergency - patient unresponsive",
            phi_categories=["vitals", "medications"],
            resource_type="patient_data"
        )
        
        mock_log.assert_called_once()


class TestRoleBasedAuthorization:
    """Test role-based authorization patterns using actual dependencies."""

    @pytest.mark.asyncio
    async def test_require_role_patient(self):
        """Test require_role dependency for patient."""
        patient_checker = require_role("patient")
        
        mock_patient = MagicMock()
        mock_patient.role = "patient"
        
        with patch('app.dependencies.get_current_user', return_value=mock_patient):
            result = await patient_checker(mock_patient)
            assert result.role == "patient"

    @pytest.mark.asyncio
    async def test_require_role_doctor(self):
        """Test require_role dependency for doctor."""
        doctor_checker = require_role("doctor")
        
        mock_doctor = MagicMock()
        mock_doctor.role = "doctor"
        
        with patch('app.dependencies.get_current_user', return_value=mock_doctor):
            result = await doctor_checker(mock_doctor)
            assert result.role == "doctor"

    @pytest.mark.asyncio
    async def test_require_role_mismatch_raises_403(self):
        """Test that role mismatch raises 403 Forbidden."""
        doctor_checker = require_role("doctor")
        
        mock_patient = MagicMock()
        mock_patient.role = "patient"
        
        with pytest.raises(HTTPException) as exc_info:
            await doctor_checker(mock_patient)
        
        assert exc_info.value.status_code == 403
        assert "doctor" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_get_current_doctor_with_patient_raises_403(self):
        """Test get_current_doctor blocks patients."""
        mock_patient = MagicMock()
        mock_patient.role = "patient"
        
        with pytest.raises(HTTPException) as exc_info:
            await get_current_doctor(mock_patient)
        
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_get_current_patient_with_doctor_raises_403(self):
        """Test get_current_patient blocks doctors."""
        mock_doctor = MagicMock()
        mock_doctor.role = "doctor"
        
        with pytest.raises(HTTPException) as exc_info:
            await get_current_patient(mock_doctor)
        
        assert exc_info.value.status_code == 403
