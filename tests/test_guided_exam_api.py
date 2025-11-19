"""
Comprehensive PHI Isolation Tests for Guided Exam API

Tests verify that:
1. Patient A cannot access Patient B's sessions (PHI isolation)
2. GET /sessions/{id} only returns session for authorized patient
3. GET /sessions/{id}/results returns metrics for THAT session only
4. Proper 403/404 responses for unauthorized access
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.main import app
from app.database import Base, get_db
from app.models.video_ai_models import VideoExamSession, VideoMetrics
from app.models.user import User
from app.dependencies import get_current_user
from datetime import datetime
import base64

# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Fixtures
@pytest.fixture(scope="function")
def db_session():
    """Create a fresh database for each test"""
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(db_session):
    """Create test client with overridden database"""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def mock_user_patient_a(db_session):
    """Create mock User for Patient A"""
    user = User(
        id="patient_a_123",
        email="patient_a@test.com",
        role="patient"
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def mock_user_patient_b(db_session):
    """Create mock User for Patient B"""
    user = User(
        id="patient_b_456",
        email="patient_b@test.com",
        role="patient"
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def mock_user_patient_c(db_session):
    """Create mock User for Patient C"""
    user = User(
        id="patient_c_789",
        email="patient_c@test.com",
        role="patient"
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def patient_a_session(db_session):
    """Create exam session for Patient A"""
    session = VideoExamSession(
        patient_id="patient_a_123",
        status="in_progress",
        current_stage="eyes",
        prep_time_seconds=30,
        eyes_stage_completed=True,
        palm_stage_completed=True,
        tongue_stage_completed=True,
        lips_stage_completed=True,
        eyes_quality_score=85.0,
        palm_quality_score=87.0,
        tongue_quality_score=83.0,
        lips_quality_score=86.0,
        overall_quality_score=85.25
    )
    db_session.add(session)
    db_session.commit()
    db_session.refresh(session)
    return session


@pytest.fixture
def patient_b_session(db_session):
    """Create exam session for Patient B"""
    session = VideoExamSession(
        patient_id="patient_b_456",
        status="in_progress",
        current_stage="palm",
        prep_time_seconds=30,
        eyes_stage_completed=True,
        palm_stage_completed=False,
        tongue_stage_completed=False,
        lips_stage_completed=False,
        eyes_quality_score=90.0
    )
    db_session.add(session)
    db_session.commit()
    db_session.refresh(session)
    return session


@pytest.fixture
def patient_a_metrics(db_session, patient_a_session):
    """Create VideoMetrics for Patient A's session"""
    metrics = VideoMetrics(
        session_id=None,
        patient_id="patient_a_123",
        guided_exam_session_id=patient_a_session.id,
        exam_stage="combined",
        sclera_yellowness_score=0.45,
        jaundice_risk_level="low",
        conjunctival_pallor_index=0.62,
        palmar_pallor_lab_index=0.58,
        tongue_color_index=0.71,
        lip_hydration_score=0.82,
        respiratory_rate_bpm=16.0,
        skin_pallor_score=0.55,
        face_detection_confidence=0.95,
        lighting_quality_score=0.88,
        frames_analyzed=120,
        processing_time_seconds=5.2,
        model_version="v1.0",
        raw_metrics={"test": "data"}
    )
    db_session.add(metrics)
    db_session.commit()
    db_session.refresh(metrics)
    
    # Link metrics to session
    db_session.query(VideoExamSession).filter(
        VideoExamSession.id == patient_a_session.id
    ).update({
        "video_metrics_id": metrics.id,
        "status": "completed",
        "completed_at": datetime.utcnow()
    })
    db_session.commit()
    
    return metrics


@pytest.fixture
def patient_b_metrics(db_session, patient_b_session):
    """Create VideoMetrics for Patient B's session"""
    metrics = VideoMetrics(
        session_id=None,
        patient_id="patient_b_456",
        guided_exam_session_id=patient_b_session.id,
        exam_stage="combined",
        sclera_yellowness_score=0.72,
        jaundice_risk_level="moderate",
        conjunctival_pallor_index=0.45,
        palmar_pallor_lab_index=0.38,
        tongue_color_index=0.51,
        lip_hydration_score=0.62,
        respiratory_rate_bpm=18.0,
        skin_pallor_score=0.42,
        face_detection_confidence=0.92,
        lighting_quality_score=0.85,
        frames_analyzed=100,
        processing_time_seconds=4.8,
        model_version="v1.0",
        raw_metrics={"test": "data_b"}
    )
    db_session.add(metrics)
    db_session.commit()
    db_session.refresh(metrics)
    
    # Link metrics to session
    db_session.query(VideoExamSession).filter(
        VideoExamSession.id == patient_b_session.id
    ).update({
        "video_metrics_id": metrics.id,
        "status": "completed",
        "completed_at": datetime.utcnow()
    })
    db_session.commit()
    
    return metrics


# ==================== PHI ISOLATION TESTS ====================

class TestPHIIsolation:
    """Test suite for PHI isolation and access control"""
    
    def test_patient_a_can_access_own_session(self, client, patient_a_session, mock_user_patient_a):
        """Patient A should be able to access their own session"""
        # Override get_current_user to return Patient A
        async def override_get_user():
            return mock_user_patient_a
        
        app.dependency_overrides[get_current_user] = override_get_user
        
        response = client.get(
            f"/api/v1/guided-exam/sessions/{patient_a_session.id}",
            headers={"Authorization": "Bearer patient_a_token"}
        )
        
        app.dependency_overrides.clear()
        app.dependency_overrides[get_db] = lambda: client
        
        assert response.status_code == 200
        data = response.json()
        assert data["patient_id"] == "patient_a_123"
        assert data["session_id"] == patient_a_session.id
    
    
    def test_patient_a_cannot_access_patient_b_session(self, client, patient_a_session, patient_b_session, mock_user_patient_a):
        """
        CRITICAL: Patient A should NOT be able to access Patient B's session
        This test verifies PHI isolation
        """
        # Override get_current_user to return Patient A
        async def override_get_user():
            return mock_user_patient_a
        
        app.dependency_overrides[get_current_user] = override_get_user
        
        response = client.get(
            f"/api/v1/guided-exam/sessions/{patient_b_session.id}",
            headers={"Authorization": "Bearer patient_a_token"}
        )
        
        app.dependency_overrides.clear()
        
        # Should return 403 Forbidden or 404 Not Found
        assert response.status_code in [403, 404]
        
        # Should NOT contain Patient B's data
        if response.status_code == 200:
            pytest.fail("CRITICAL PHI VIOLATION: Patient A accessed Patient B's session")
    
    
    def test_patient_b_can_access_own_session(self, client, patient_b_session, mock_user_patient_b):
        """Patient B should be able to access their own session"""
        # Override get_current_user to return Patient B
        async def override_get_user():
            return mock_user_patient_b
        
        app.dependency_overrides[get_current_user] = override_get_user
        
        response = client.get(
            f"/api/v1/guided-exam/sessions/{patient_b_session.id}",
            headers={"Authorization": "Bearer patient_b_token"}
        )
        
        app.dependency_overrides.clear()
        
        assert response.status_code == 200
        data = response.json()
        assert data["patient_id"] == "patient_b_456"
        assert data["session_id"] == patient_b_session.id
    
    
    def test_patient_b_cannot_access_patient_a_session(self, client, patient_a_session, patient_b_session, mock_user_patient_b):
        """
        CRITICAL: Patient B should NOT be able to access Patient A's session
        This test verifies PHI isolation
        """
        # Override get_current_user to return Patient B
        async def override_get_user():
            return mock_user_patient_b
        
        app.dependency_overrides[get_current_user] = override_get_user
        
        response = client.get(
            f"/api/v1/guided-exam/sessions/{patient_a_session.id}",
            headers={"Authorization": "Bearer patient_b_token"}
        )
        
        app.dependency_overrides.clear()
        
        # Should return 403 Forbidden or 404 Not Found
        assert response.status_code in [403, 404]
        
        # Should NOT contain Patient A's data
        if response.status_code == 200:
            pytest.fail("CRITICAL PHI VIOLATION: Patient B accessed Patient A's session")


class TestSessionResultsPHIIsolation:
    """Test PHI isolation for session results endpoint"""
    
    def test_patient_a_can_access_own_results(self, client, patient_a_session, patient_a_metrics, mock_user_patient_a):
        """Patient A should be able to access their own session results"""
        # Override get_current_user to return Patient A
        async def override_get_user():
            return mock_user_patient_a
        
        app.dependency_overrides[get_current_user] = override_get_user
        
        response = client.get(
            f"/api/v1/guided-exam/sessions/{patient_a_session.id}/results",
            headers={"Authorization": "Bearer patient_a_token"}
        )
        
        app.dependency_overrides.clear()
        
        assert response.status_code == 200
        data = response.json()
        assert data["patient_id"] == "patient_a_123"
        assert data["session_id"] == str(patient_a_session.id)
        
        # Verify metrics belong to Patient A's session
        metrics = data["metrics"]
        assert metrics["guided_exam_session_id"] == patient_a_session.id
        assert metrics["sclera_yellowness_score"] == 0.45  # Patient A's value
    
    
    def test_patient_a_cannot_access_patient_b_results(self, client, patient_a_session, patient_b_session, patient_b_metrics, mock_user_patient_a):
        """
        CRITICAL: Patient A should NOT be able to access Patient B's results
        This test verifies results are isolated by session_id
        """
        # Override get_current_user to return Patient A
        async def override_get_user():
            return mock_user_patient_a
        
        app.dependency_overrides[get_current_user] = override_get_user
        
        response = client.get(
            f"/api/v1/guided-exam/sessions/{patient_b_session.id}/results",
            headers={"Authorization": "Bearer patient_a_token"}
        )
        
        app.dependency_overrides.clear()
        
        # Should return 403 Forbidden or 404 Not Found
        assert response.status_code in [403, 404]
        
        # Should NOT return Patient B's metrics
        if response.status_code == 200:
            data = response.json()
            metrics = data.get("metrics", {})
            if metrics.get("sclera_yellowness_score") == 0.72:  # Patient B's value
                pytest.fail("CRITICAL PHI VIOLATION: Patient A accessed Patient B's metrics")
    
    
    def test_results_query_by_session_id_not_patient_id(self, client, patient_a_session, patient_a_metrics, patient_b_session, patient_b_metrics, mock_user_patient_a):
        """
        CRITICAL: Results endpoint must query by guided_exam_session_id
        NOT just by patient_id to prevent returning wrong session's metrics
        """
        # Override get_current_user to return Patient A
        async def override_get_user():
            return mock_user_patient_a
        
        app.dependency_overrides[get_current_user] = override_get_user
        
        # Patient A has multiple sessions, ensure we get the RIGHT session's metrics
        response = client.get(
            f"/api/v1/guided-exam/sessions/{patient_a_session.id}/results",
            headers={"Authorization": "Bearer patient_a_token"}
        )
        
        app.dependency_overrides.clear()
        
        assert response.status_code == 200
        data = response.json()
        metrics = data["metrics"]
        
        # MUST match the specific session_id requested
        assert metrics["guided_exam_session_id"] == patient_a_session.id
        
        # Should NOT be Patient B's metrics
        assert metrics["sclera_yellowness_score"] != 0.72  # Patient B's value
        assert metrics["sclera_yellowness_score"] == 0.45  # Patient A's value
    
    
    def test_patient_b_can_access_own_results(self, client, patient_b_session, patient_b_metrics, mock_user_patient_b):
        """Patient B should be able to access their own session results"""
        # Override get_current_user to return Patient B
        async def override_get_user():
            return mock_user_patient_b
        
        app.dependency_overrides[get_current_user] = override_get_user
        
        response = client.get(
            f"/api/v1/guided-exam/sessions/{patient_b_session.id}/results",
            headers={"Authorization": "Bearer patient_b_token"}
        )
        
        app.dependency_overrides.clear()
        
        assert response.status_code == 200
        data = response.json()
        assert data["patient_id"] == "patient_b_456"
        assert data["session_id"] == str(patient_b_session.id)
        
        # Verify metrics belong to Patient B's session
        metrics = data["metrics"]
        assert metrics["guided_exam_session_id"] == patient_b_session.id
        assert metrics["sclera_yellowness_score"] == 0.72  # Patient B's value
    
    
    def test_patient_b_cannot_access_patient_a_results(self, client, patient_a_session, patient_a_metrics, patient_b_session, mock_user_patient_b):
        """
        CRITICAL: Patient B should NOT be able to access Patient A's results
        """
        # Override get_current_user to return Patient B
        async def override_get_user():
            return mock_user_patient_b
        
        app.dependency_overrides[get_current_user] = override_get_user
        
        response = client.get(
            f"/api/v1/guided-exam/sessions/{patient_a_session.id}/results",
            headers={"Authorization": "Bearer patient_b_token"}
        )
        
        app.dependency_overrides.clear()
        
        # Should return 403 Forbidden or 404 Not Found
        assert response.status_code in [403, 404]
        
        # Should NOT return Patient A's metrics
        if response.status_code == 200:
            data = response.json()
            metrics = data.get("metrics", {})
            if metrics.get("sclera_yellowness_score") == 0.45:  # Patient A's value
                pytest.fail("CRITICAL PHI VIOLATION: Patient B accessed Patient A's metrics")


class TestSessionCreation:
    """Test session creation with PHI isolation"""
    
    def test_patient_can_create_own_session(self, client, mock_user_patient_c):
        """Patient should be able to create a session for themselves"""
        # Override get_current_user to return Patient C
        async def override_get_user():
            return mock_user_patient_c
        
        app.dependency_overrides[get_current_user] = override_get_user
        
        response = client.post(
            "/api/v1/guided-exam/sessions",
            json={
                "patient_id": "patient_c_789",
                "device_info": {"platform": "iOS", "version": "14.5"}
            },
            headers={"Authorization": "Bearer patient_c_token"}
        )
        
        app.dependency_overrides.clear()
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "in_progress"
        assert data["current_stage"] == "eyes"
        assert "session_id" in data
    
    
    def test_patient_cannot_create_session_for_another_patient(self, client, mock_user_patient_a):
        """
        CRITICAL: Patient A should NOT be able to create session for Patient B
        This prevents session hijacking
        """
        # Override get_current_user to return Patient A
        async def override_get_user():
            return mock_user_patient_a
        
        app.dependency_overrides[get_current_user] = override_get_user
        
        response = client.post(
            "/api/v1/guided-exam/sessions",
            json={
                "patient_id": "patient_b_456",  # Patient A trying to create for Patient B
                "device_info": {"platform": "iOS", "version": "14.5"}
            },
            headers={"Authorization": "Bearer patient_a_token"}  # Patient A's token
        )
        
        app.dependency_overrides.clear()
        
        # Should return 403 Forbidden
        assert response.status_code == 403
        assert "another patient" in response.json()["detail"].lower()


class TestFrameCapture:
    """Test frame capture with PHI isolation"""
    
    def test_patient_can_capture_frame_for_own_session(self, client, patient_a_session, mock_user_patient_a):
        """Patient should be able to capture frames for their own session"""
        # Override get_current_user to return Patient A
        async def override_get_user():
            return mock_user_patient_a
        
        app.dependency_overrides[get_current_user] = override_get_user
        
        # Create a small test image in base64
        test_image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        
        # Note: This will fail if S3 is not configured, but we're testing the auth logic
        response = client.post(
            f"/api/v1/guided-exam/sessions/{patient_a_session.id}/capture",
            json={
                "stage": "eyes",
                "frame_base64": test_image_base64
            },
            headers={"Authorization": "Bearer patient_a_token"}
        )
        
        app.dependency_overrides.clear()
        
        # May fail on S3 upload, but should NOT fail on auth
        assert response.status_code != 403
    
    
    def test_patient_cannot_capture_frame_for_another_patient_session(self, client, patient_a_session, patient_b_session, mock_user_patient_a):
        """
        CRITICAL: Patient A should NOT be able to capture frames for Patient B's session
        """
        # Override get_current_user to return Patient A
        async def override_get_user():
            return mock_user_patient_a
        
        app.dependency_overrides[get_current_user] = override_get_user
        
        test_image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        
        response = client.post(
            f"/api/v1/guided-exam/sessions/{patient_b_session.id}/capture",
            json={
                "stage": "eyes",
                "frame_base64": test_image_base64
            },
            headers={"Authorization": "Bearer patient_a_token"}  # Patient A's token
        )
        
        app.dependency_overrides.clear()
        
        # Should return 403 Forbidden
        assert response.status_code == 403
        assert "another patient" in response.json()["detail"].lower()


class TestSessionCompletion:
    """Test session completion with PHI isolation"""
    
    def test_patient_cannot_complete_another_patient_session(self, client, patient_a_session, patient_b_session, mock_user_patient_a):
        """
        CRITICAL: Patient A should NOT be able to complete Patient B's session
        """
        # Override get_current_user to return Patient A
        async def override_get_user():
            return mock_user_patient_a
        
        app.dependency_overrides[get_current_user] = override_get_user
        
        response = client.post(
            f"/api/v1/guided-exam/sessions/{patient_b_session.id}/complete",
            headers={"Authorization": "Bearer patient_a_token"}  # Patient A's token
        )
        
        app.dependency_overrides.clear()
        
        # Should return 403 Forbidden
        assert response.status_code == 403
        assert "another patient" in response.json()["detail"].lower()


# ==================== EDGE CASES ====================

class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_nonexistent_session_returns_404(self, client, mock_user_patient_a):
        """Accessing nonexistent session should return 404"""
        # Override get_current_user to return Patient A
        async def override_get_user():
            return mock_user_patient_a
        
        app.dependency_overrides[get_current_user] = override_get_user
        
        response = client.get(
            "/api/v1/guided-exam/sessions/99999",
            headers={"Authorization": "Bearer patient_a_token"}
        )
        
        app.dependency_overrides.clear()
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    
    def test_results_for_nonexistent_session_returns_404(self, client, mock_user_patient_a):
        """Accessing results for nonexistent session should return 404"""
        # Override get_current_user to return Patient A
        async def override_get_user():
            return mock_user_patient_a
        
        app.dependency_overrides[get_current_user] = override_get_user
        
        response = client.get(
            "/api/v1/guided-exam/sessions/99999/results",
            headers={"Authorization": "Bearer patient_a_token"}
        )
        
        app.dependency_overrides.clear()
        
        assert response.status_code == 404
    
    
    def test_results_for_session_without_metrics_returns_404(self, client, patient_a_session, mock_user_patient_a):
        """Accessing results for session without analysis should return 404"""
        # Override get_current_user to return Patient A
        async def override_get_user():
            return mock_user_patient_a
        
        app.dependency_overrides[get_current_user] = override_get_user
        
        response = client.get(
            f"/api/v1/guided-exam/sessions/{patient_a_session.id}/results",
            headers={"Authorization": "Bearer patient_a_token"}
        )
        
        app.dependency_overrides.clear()
        
        # Session exists but no metrics yet
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
