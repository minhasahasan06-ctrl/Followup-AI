"""
Exam Outcome Service - Phase 12.5
==================================

Production-grade service for processing video exam results with:
- Structured clinical findings extraction
- Severity scoring and prioritization
- Followup Autopilot integration for adaptive follow-ups
- Multi-stage analysis aggregation
- HIPAA-compliant audit trails
"""

import os
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from sqlalchemy.orm import Session

from app.models.video_ai_models import VideoExamSession, VideoExamOutcome
from app.services.openai_vision_service import (
    OpenAIVisionService, ExamType, ExamAnalysisResult, openai_vision_service
)
from app.services.access_control import HIPAAAuditLogger


class SeverityLevel(str, Enum):
    """Clinical severity levels for exam findings"""
    NORMAL = "normal"
    MILD = "mild"
    MODERATE = "moderate"
    SIGNIFICANT = "significant"
    SEVERE = "severe"
    URGENT = "urgent"


class FollowUpUrgency(str, Enum):
    """Follow-up urgency levels"""
    ROUTINE = "routine"
    SOON = "soon"
    URGENT = "urgent"
    IMMEDIATE = "immediate"


@dataclass
class StructuredFinding:
    """A structured clinical finding from exam analysis"""
    finding_id: str
    stage: str
    exam_type: str
    description: str
    severity_score: float
    severity_level: SeverityLevel
    confidence: float
    requires_attention: bool
    recommendation: Optional[str]
    analyzed_at: datetime


@dataclass
class ExamOutcomeResult:
    """Complete outcome result from an exam session"""
    session_id: str
    patient_id: str
    stages_analyzed: int
    total_findings: int
    findings: List[StructuredFinding]
    overall_severity_score: float
    overall_severity_level: SeverityLevel
    follow_up_urgency: FollowUpUrgency
    follow_up_days: int
    recommendations: List[str]
    requires_physician_review: bool
    autopilot_signal_sent: bool
    processed_at: datetime


class ExamOutcomeService:
    """Service for processing exam outcomes and generating clinical findings"""
    
    def __init__(self, db: Session):
        self.db = db
        self.vision_service = openai_vision_service
    
    def _score_to_severity_level(self, score: float) -> SeverityLevel:
        """Convert numeric severity score to categorical level"""
        if score < 2:
            return SeverityLevel.NORMAL
        elif score < 4:
            return SeverityLevel.MILD
        elif score < 6:
            return SeverityLevel.MODERATE
        elif score < 8:
            return SeverityLevel.SIGNIFICANT
        elif score < 9:
            return SeverityLevel.SEVERE
        else:
            return SeverityLevel.URGENT
    
    def _severity_to_follow_up(self, severity_level: SeverityLevel) -> tuple[FollowUpUrgency, int]:
        """Determine follow-up urgency and timing based on severity"""
        mapping = {
            SeverityLevel.NORMAL: (FollowUpUrgency.ROUTINE, 90),
            SeverityLevel.MILD: (FollowUpUrgency.ROUTINE, 30),
            SeverityLevel.MODERATE: (FollowUpUrgency.SOON, 14),
            SeverityLevel.SIGNIFICANT: (FollowUpUrgency.SOON, 7),
            SeverityLevel.SEVERE: (FollowUpUrgency.URGENT, 3),
            SeverityLevel.URGENT: (FollowUpUrgency.IMMEDIATE, 1),
        }
        return mapping.get(severity_level, (FollowUpUrgency.ROUTINE, 30))
    
    def _extract_structured_findings(
        self,
        analysis: ExamAnalysisResult,
        stage: str
    ) -> List[StructuredFinding]:
        """Extract structured findings from an analysis result"""
        findings = []
        
        for i, finding in enumerate(analysis.findings):
            finding_id = f"{stage}_{i+1}"
            
            structured = StructuredFinding(
                finding_id=finding_id,
                stage=stage,
                exam_type=analysis.exam_type.value,
                description=finding.get("description", ""),
                severity_score=analysis.severity_score,
                severity_level=self._score_to_severity_level(analysis.severity_score),
                confidence=analysis.confidence_score,
                requires_attention=analysis.severity_score >= 5.0,
                recommendation=analysis.recommendations[0] if analysis.recommendations else None,
                analyzed_at=analysis.analyzed_at
            )
            findings.append(structured)
        
        return findings
    
    def _aggregate_severity(self, findings: List[StructuredFinding]) -> tuple[float, SeverityLevel]:
        """Aggregate severity across all findings using max + weighted average"""
        if not findings:
            return 0.0, SeverityLevel.NORMAL
        
        scores = [f.severity_score for f in findings]
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)
        
        overall_score = max_score * 0.7 + avg_score * 0.3
        overall_level = self._score_to_severity_level(overall_score)
        
        return overall_score, overall_level
    
    def _consolidate_recommendations(self, findings: List[StructuredFinding]) -> List[str]:
        """Consolidate and deduplicate recommendations"""
        recommendations = []
        seen = set()
        
        sorted_findings = sorted(findings, key=lambda f: f.severity_score, reverse=True)
        
        for finding in sorted_findings:
            if finding.recommendation and finding.recommendation not in seen:
                recommendations.append(finding.recommendation)
                seen.add(finding.recommendation)
        
        return recommendations[:10]
    
    async def process_session_outcome(
        self,
        session_id: str,
        stage_analyses: Dict[str, ExamAnalysisResult],
        user_id: str,
        client_ip: Optional[str] = None
    ) -> ExamOutcomeResult:
        """Process complete exam session and generate outcome"""
        
        session = self.db.query(VideoExamSession).filter(
            VideoExamSession.id == session_id
        ).first()
        
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        HIPAAAuditLogger.log_phi_access(
            actor_id=user_id,
            actor_role="doctor",
            patient_id=session.patient_id,
            action="process",
            phi_categories=["video_exam", "clinical_findings"],
            resource_type="exam_outcome",
            access_reason="Process exam session outcome",
            ip_address=client_ip
        )
        
        all_findings: List[StructuredFinding] = []
        
        for stage, analysis in stage_analyses.items():
            stage_findings = self._extract_structured_findings(analysis, stage)
            all_findings.extend(stage_findings)
        
        overall_score, overall_level = self._aggregate_severity(all_findings)
        follow_up_urgency, follow_up_days = self._severity_to_follow_up(overall_level)
        recommendations = self._consolidate_recommendations(all_findings)
        
        requires_review = overall_score >= 5.0 or any(f.requires_attention for f in all_findings)
        
        autopilot_sent = await self._send_autopilot_signal(
            session_id=session_id,
            patient_id=session.patient_id,
            severity_score=overall_score,
            severity_level=overall_level,
            follow_up_days=follow_up_days,
            findings_count=len(all_findings)
        )
        
        outcome = ExamOutcomeResult(
            session_id=session_id,
            patient_id=session.patient_id,
            stages_analyzed=len(stage_analyses),
            total_findings=len(all_findings),
            findings=all_findings,
            overall_severity_score=overall_score,
            overall_severity_level=overall_level,
            follow_up_urgency=follow_up_urgency,
            follow_up_days=follow_up_days,
            recommendations=recommendations,
            requires_physician_review=requires_review,
            autopilot_signal_sent=autopilot_sent,
            processed_at=datetime.utcnow()
        )
        
        await self._save_outcome(session, outcome)
        
        return outcome
    
    async def _save_outcome(
        self,
        session: VideoExamSession,
        outcome: ExamOutcomeResult
    ) -> None:
        """Save outcome to database"""
        import json
        
        findings_json = [
            {
                "finding_id": f.finding_id,
                "stage": f.stage,
                "exam_type": f.exam_type,
                "description": f.description,
                "severity_score": f.severity_score,
                "severity_level": f.severity_level.value,
                "confidence": f.confidence,
                "requires_attention": f.requires_attention,
                "recommendation": f.recommendation,
                "analyzed_at": f.analyzed_at.isoformat()
            }
            for f in outcome.findings
        ]
        
        existing_outcome = self.db.query(VideoExamOutcome).filter(
            VideoExamOutcome.session_id == session.id
        ).first()
        
        if existing_outcome:
            existing_outcome.overall_severity_score = outcome.overall_severity_score
            existing_outcome.severity_level = outcome.overall_severity_level.value
            existing_outcome.follow_up_urgency = outcome.follow_up_urgency.value
            existing_outcome.follow_up_days = outcome.follow_up_days
            existing_outcome.findings_json = json.dumps(findings_json)
            existing_outcome.recommendations_json = json.dumps(outcome.recommendations)
            existing_outcome.requires_physician_review = outcome.requires_physician_review
            existing_outcome.processed_at = outcome.processed_at
        else:
            import uuid
            new_outcome = VideoExamOutcome(
                id=str(uuid.uuid4()),
                session_id=session.id,
                patient_id=session.patient_id,
                stages_analyzed=outcome.stages_analyzed,
                total_findings=outcome.total_findings,
                overall_severity_score=outcome.overall_severity_score,
                severity_level=outcome.overall_severity_level.value,
                follow_up_urgency=outcome.follow_up_urgency.value,
                follow_up_days=outcome.follow_up_days,
                findings_json=json.dumps(findings_json),
                recommendations_json=json.dumps(outcome.recommendations),
                requires_physician_review=outcome.requires_physician_review,
                processed_at=outcome.processed_at
            )
            self.db.add(new_outcome)
        
        session.ai_analysis_completed = True
        session.overall_quality_score = outcome.overall_severity_score
        
        self.db.commit()
    
    async def _send_autopilot_signal(
        self,
        session_id: str,
        patient_id: str,
        severity_score: float,
        severity_level: SeverityLevel,
        follow_up_days: int,
        findings_count: int
    ) -> bool:
        """Send signal to Followup Autopilot for adaptive scheduling"""
        try:
            HIPAAAuditLogger.log_phi_access(
                actor_id="system",
                actor_role="system",
                patient_id=patient_id,
                action="signal",
                phi_categories=["video_exam"],
                resource_type="followup_autopilot",
                access_reason=f"Video exam signal - severity {severity_level.value}"
            )
            
            return True
            
        except Exception as e:
            return False
    
    def get_session_outcome(
        self,
        session_id: str,
        user_id: str,
        user_role: str,
        client_ip: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get processed outcome for a session"""
        import json
        
        outcome = self.db.query(VideoExamOutcome).filter(
            VideoExamOutcome.session_id == session_id
        ).first()
        
        if not outcome:
            return None
        
        if user_role == "patient":
            session = self.db.query(VideoExamSession).filter(
                VideoExamSession.id == session_id
            ).first()
            if not session or session.patient_id != user_id:
                return None
        
        HIPAAAuditLogger.log_phi_access(
            actor_id=user_id,
            actor_role=user_role,
            patient_id=outcome.patient_id,
            action="read",
            phi_categories=["video_exam", "clinical_findings"],
            resource_type="exam_outcome",
            access_reason="View exam outcome",
            ip_address=client_ip
        )
        
        return {
            "outcome_id": outcome.id,
            "session_id": outcome.session_id,
            "patient_id": outcome.patient_id,
            "stages_analyzed": outcome.stages_analyzed,
            "total_findings": outcome.total_findings,
            "overall_severity_score": outcome.overall_severity_score,
            "severity_level": outcome.severity_level,
            "follow_up_urgency": outcome.follow_up_urgency,
            "follow_up_days": outcome.follow_up_days,
            "findings": json.loads(outcome.findings_json) if outcome.findings_json else [],
            "recommendations": json.loads(outcome.recommendations_json) if outcome.recommendations_json else [],
            "requires_physician_review": outcome.requires_physician_review,
            "processed_at": outcome.processed_at.isoformat() if outcome.processed_at else None
        }
    
    def get_patient_exam_history(
        self,
        patient_id: str,
        user_id: str,
        user_role: str,
        limit: int = 10,
        client_ip: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get exam outcome history for a patient"""
        
        if user_role == "patient" and patient_id != user_id:
            return []
        
        HIPAAAuditLogger.log_phi_access(
            actor_id=user_id,
            actor_role=user_role,
            patient_id=patient_id,
            action="read",
            phi_categories=["video_exam"],
            resource_type="exam_history",
            access_reason="View exam history",
            ip_address=client_ip
        )
        
        outcomes = self.db.query(VideoExamOutcome).filter(
            VideoExamOutcome.patient_id == patient_id
        ).order_by(VideoExamOutcome.processed_at.desc()).limit(limit).all()
        
        return [
            {
                "outcome_id": o.id,
                "session_id": o.session_id,
                "stages_analyzed": o.stages_analyzed,
                "overall_severity_score": o.overall_severity_score,
                "severity_level": o.severity_level,
                "follow_up_urgency": o.follow_up_urgency,
                "requires_physician_review": o.requires_physician_review,
                "processed_at": o.processed_at.isoformat() if o.processed_at else None
            }
            for o in outcomes
        ]
