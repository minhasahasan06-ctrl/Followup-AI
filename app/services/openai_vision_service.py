"""
OpenAI Vision Service for HCEC Exam Analysis - Phase 12.3
==========================================================

Production-grade vision analysis service for Health Care Exam Capture with:
- Exam-type specific clinical prompts (skin/oral/joint/wound/eye/palm/tongue/lips)
- Batch analysis for multiple frames
- Rate limiting with exponential backoff
- Fallback handling for API failures
- PHI-free prompts (no patient identifying information)
- HIPAA audit logging
- Quality scoring for captured images
"""

import os
import time
import base64
import asyncio
import httpx
from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime
from dataclasses import dataclass
from openai import AsyncOpenAI

from app.services.access_control import HIPAAAuditLogger


class ExamType(str, Enum):
    """Supported exam types for vision analysis"""
    SKIN = "skin"
    ORAL = "oral"
    JOINT = "joint"
    WOUND = "wound"
    EYE = "eye"
    PALM = "palm"
    TONGUE = "tongue"
    LIPS = "lips"
    RESPIRATORY = "respiratory"
    CUSTOM = "custom"


@dataclass
class ImageQualityResult:
    """Result of image quality assessment"""
    quality_score: float
    is_acceptable: bool
    issues: List[str]
    recommendations: List[str]


@dataclass
class ExamAnalysisResult:
    """Result of clinical exam analysis"""
    exam_type: ExamType
    findings: List[Dict[str, Any]]
    severity_score: float
    confidence_score: float
    recommendations: List[str]
    follow_up_suggested: bool
    raw_analysis: str
    analyzed_at: datetime
    model_used: str
    processing_time_ms: int


EXAM_PROMPTS = {
    ExamType.SKIN: """Analyze this dermatological image for clinical assessment. Identify:
1. Any visible lesions, rashes, discoloration, or abnormalities
2. Characteristics (size estimation, color, texture, pattern, distribution)
3. Potential concerns that may warrant further examination
4. Signs of infection, inflammation, or changes from normal skin appearance

Provide findings in a structured clinical format. Do not provide diagnosis - only observations.
Rate severity from 0-10 (0=normal, 10=requires immediate attention).
Rate your confidence in observations from 0-100%.""",

    ExamType.ORAL: """Analyze this oral/mouth examination image for clinical assessment. Identify:
1. Condition of gums, teeth, tongue, and oral mucosa
2. Any visible lesions, discoloration, swelling, or abnormalities
3. Signs of infection, inflammation, or periodontal issues
4. Any areas requiring further clinical examination

Provide findings in a structured clinical format. Do not provide diagnosis - only observations.
Rate severity from 0-10 (0=normal, 10=requires immediate attention).
Rate your confidence in observations from 0-100%.""",

    ExamType.JOINT: """Analyze this joint/extremity examination image for clinical assessment. Identify:
1. Visible swelling, redness, or deformity
2. Skin changes overlying the joint area
3. Asymmetry compared to expected normal appearance
4. Signs of inflammation or injury

Provide findings in a structured clinical format. Do not provide diagnosis - only observations.
Rate severity from 0-10 (0=normal, 10=requires immediate attention).
Rate your confidence in observations from 0-100%.""",

    ExamType.WOUND: """Analyze this wound examination image for clinical assessment. Identify:
1. Wound characteristics (estimated size, depth appearance, edges)
2. Signs of healing or concerning features
3. Tissue appearance (granulation, necrosis, epithelialization)
4. Surrounding skin condition and any signs of infection

Provide findings in a structured clinical format. Do not provide diagnosis - only observations.
Rate severity from 0-10 (0=normal healing, 10=requires immediate attention).
Rate your confidence in observations from 0-100%.""",

    ExamType.EYE: """Analyze this eye examination image for clinical assessment. Identify:
1. Appearance of sclera, iris, pupil, and surrounding structures
2. Any visible redness, discharge, swelling, or abnormalities
3. Pupil symmetry and response appearance if visible
4. Eyelid condition and any concerning features

Provide findings in a structured clinical format. Do not provide diagnosis - only observations.
Rate severity from 0-10 (0=normal, 10=requires immediate attention).
Rate your confidence in observations from 0-100%.""",

    ExamType.PALM: """Analyze this palm examination image for clinical assessment. Identify:
1. Skin condition (texture, color, hydration)
2. Any visible lesions, discoloration, or abnormalities
3. Nail bed appearance if visible
4. Signs of systemic conditions visible in the palm

Provide findings in a structured clinical format. Do not provide diagnosis - only observations.
Rate severity from 0-10 (0=normal, 10=requires immediate attention).
Rate your confidence in observations from 0-100%.""",

    ExamType.TONGUE: """Analyze this tongue examination image for clinical assessment. Identify:
1. Color, coating, and texture of the tongue
2. Any visible lesions, discoloration, or swelling
3. Papillae appearance and distribution
4. Signs of dehydration, nutritional deficiencies, or oral conditions

Provide findings in a structured clinical format. Do not provide diagnosis - only observations.
Rate severity from 0-10 (0=normal, 10=requires immediate attention).
Rate your confidence in observations from 0-100%.""",

    ExamType.LIPS: """Analyze this lip examination image for clinical assessment. Identify:
1. Color, texture, and moisture of the lips
2. Any visible lesions, cracks, swelling, or abnormalities
3. Surrounding skin condition
4. Signs of dehydration, infection, or systemic conditions

Provide findings in a structured clinical format. Do not provide diagnosis - only observations.
Rate severity from 0-10 (0=normal, 10=requires immediate attention).
Rate your confidence in observations from 0-100%.""",

    ExamType.RESPIRATORY: """Analyze this respiratory examination image for clinical assessment. Identify:
1. Chest wall appearance and symmetry
2. Any visible abnormalities in breathing pattern if video
3. Skin changes or visible concerns
4. General posture related to respiratory function

Provide findings in a structured clinical format. Do not provide diagnosis - only observations.
Rate severity from 0-10 (0=normal, 10=requires immediate attention).
Rate your confidence in observations from 0-100%.""",

    ExamType.CUSTOM: """Analyze this clinical examination image. Identify:
1. All visible anatomical features
2. Any abnormalities, lesions, or concerns
3. Areas requiring further examination
4. General observations about tissue condition

Provide findings in a structured clinical format. Do not provide diagnosis - only observations.
Rate severity from 0-10 (0=normal, 10=requires immediate attention).
Rate your confidence in observations from 0-100%."""
}

IMAGE_QUALITY_PROMPT = """Assess the quality of this clinical examination image. Evaluate:
1. Focus/sharpness (is the image in focus?)
2. Lighting (is there adequate, even lighting?)
3. Framing (is the area of interest properly centered and visible?)
4. Resolution (is there sufficient detail for clinical assessment?)
5. Obstructions (are there any objects blocking the view?)

Provide a quality score from 0-100 and list any issues that affect clinical utility.
If quality score is below 60, suggest how to improve the capture."""


class OpenAIVisionService:
    """Service for analyzing exam images using OpenAI Vision API"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY") or os.getenv("AI_INTEGRATIONS_OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=self.api_key) if self.api_key else None
        self.model = "gpt-4o"
        self.max_retries = 3
        self.base_delay = 1.0
        self.max_tokens = 1500
        self._rate_limit_remaining = 100
        self._rate_limit_reset = None
    
    async def _make_vision_request(
        self,
        image_data: str,
        prompt: str,
        detail: str = "high"
    ) -> Optional[str]:
        """Make a vision API request with retry logic"""
        if not self.client:
            return None
        
        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_data}",
                                        "detail": detail
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=self.max_tokens
                )
                
                if response.choices and response.choices[0].message:
                    return response.choices[0].message.content
                return None
                
            except Exception as e:
                error_str = str(e)
                
                if "rate_limit" in error_str.lower():
                    delay = self.base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                    continue
                    
                if "invalid_api_key" in error_str.lower():
                    return None
                    
                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                else:
                    raise
        
        return None
    
    async def assess_image_quality(
        self,
        image_data: str,
        user_id: str,
        client_ip: Optional[str] = None
    ) -> ImageQualityResult:
        """Assess the quality of an exam image"""
        start_time = time.time()
        
        HIPAAAuditLogger.log_phi_access(
            actor_id=user_id,
            actor_role="system",
            patient_id=None,
            action="analyze",
            phi_categories=["video_exam"],
            resource_type="image_quality_assessment",
            access_reason="AI image quality assessment",
            ip_address=client_ip
        )
        
        try:
            response = await self._make_vision_request(
                image_data=image_data,
                prompt=IMAGE_QUALITY_PROMPT,
                detail="low"
            )
            
            if not response:
                return ImageQualityResult(
                    quality_score=50.0,
                    is_acceptable=True,
                    issues=["Unable to assess quality - using default"],
                    recommendations=["Ensure good lighting and focus"]
                )
            
            quality_score = self._extract_quality_score(response)
            issues = self._extract_issues(response)
            recommendations = self._extract_recommendations(response)
            
            return ImageQualityResult(
                quality_score=quality_score,
                is_acceptable=quality_score >= 60,
                issues=issues,
                recommendations=recommendations
            )
            
        except Exception as e:
            return ImageQualityResult(
                quality_score=50.0,
                is_acceptable=True,
                issues=[f"Quality assessment unavailable: {str(e)}"],
                recommendations=["Ensure good lighting and focus"]
            )
    
    async def analyze_exam_image(
        self,
        image_data: str,
        exam_type: ExamType,
        session_id: str,
        patient_id: str,
        user_id: str,
        client_ip: Optional[str] = None,
        additional_context: Optional[str] = None
    ) -> ExamAnalysisResult:
        """Analyze an exam image for clinical observations"""
        start_time = time.time()
        
        HIPAAAuditLogger.log_phi_access(
            actor_id=user_id,
            actor_role="doctor",
            patient_id=patient_id,
            action="analyze",
            phi_categories=["video_exam", "clinical_findings"],
            resource_type="exam_image_analysis",
            access_reason=f"AI analysis of {exam_type.value} exam image",
            ip_address=client_ip
        )
        
        prompt = EXAM_PROMPTS.get(exam_type, EXAM_PROMPTS[ExamType.CUSTOM])
        if additional_context:
            prompt += f"\n\nAdditional context: {additional_context}"
        
        try:
            response = await self._make_vision_request(
                image_data=image_data,
                prompt=prompt,
                detail="high"
            )
            
            processing_time = int((time.time() - start_time) * 1000)
            
            if not response:
                return ExamAnalysisResult(
                    exam_type=exam_type,
                    findings=[{"type": "error", "description": "Analysis unavailable"}],
                    severity_score=0.0,
                    confidence_score=0.0,
                    recommendations=["Manual review required"],
                    follow_up_suggested=True,
                    raw_analysis="Analysis service unavailable",
                    analyzed_at=datetime.utcnow(),
                    model_used=self.model,
                    processing_time_ms=processing_time
                )
            
            findings = self._parse_findings(response)
            severity_score = self._extract_severity_score(response)
            confidence_score = self._extract_confidence_score(response)
            recommendations = self._extract_clinical_recommendations(response)
            
            return ExamAnalysisResult(
                exam_type=exam_type,
                findings=findings,
                severity_score=severity_score,
                confidence_score=confidence_score,
                recommendations=recommendations,
                follow_up_suggested=severity_score >= 5.0,
                raw_analysis=response,
                analyzed_at=datetime.utcnow(),
                model_used=self.model,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            return ExamAnalysisResult(
                exam_type=exam_type,
                findings=[{"type": "error", "description": str(e)}],
                severity_score=0.0,
                confidence_score=0.0,
                recommendations=["Manual review required due to analysis error"],
                follow_up_suggested=True,
                raw_analysis=f"Error: {str(e)}",
                analyzed_at=datetime.utcnow(),
                model_used=self.model,
                processing_time_ms=processing_time
            )
    
    async def batch_analyze_session(
        self,
        images: List[Dict[str, Any]],
        session_id: str,
        patient_id: str,
        user_id: str,
        client_ip: Optional[str] = None
    ) -> Dict[str, ExamAnalysisResult]:
        """Analyze multiple images from an exam session"""
        HIPAAAuditLogger.log_phi_access(
            actor_id=user_id,
            actor_role="doctor",
            patient_id=patient_id,
            action="batch_analyze",
            phi_categories=["video_exam", "clinical_findings"],
            resource_type="exam_session_batch_analysis",
            access_reason=f"Batch AI analysis of {len(images)} exam images",
            ip_address=client_ip
        )
        
        results = {}
        
        tasks = []
        for img in images:
            task = self.analyze_exam_image(
                image_data=img["data"],
                exam_type=ExamType(img.get("exam_type", "custom")),
                session_id=session_id,
                patient_id=patient_id,
                user_id=user_id,
                client_ip=client_ip,
                additional_context=img.get("context")
            )
            tasks.append((img.get("stage", "unknown"), task))
        
        for stage, task in tasks:
            result = await task
            results[stage] = result
            await asyncio.sleep(0.1)
        
        return results
    
    def _extract_quality_score(self, response: str) -> float:
        """Extract quality score from response"""
        import re
        
        patterns = [
            r'quality[:\s]+score[:\s]+(\d+(?:\.\d+)?)',
            r'score[:\s]+(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)[:\s]*(?:out of|/)?\s*100'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response.lower())
            if match:
                score = float(match.group(1))
                return min(100.0, max(0.0, score))
        
        return 70.0
    
    def _extract_issues(self, response: str) -> List[str]:
        """Extract quality issues from response"""
        issues = []
        
        keywords = ["blur", "dark", "overexposed", "out of focus", "obscured", "poor lighting", "motion"]
        for keyword in keywords:
            if keyword in response.lower():
                issues.append(f"Detected issue: {keyword}")
        
        return issues if issues else ["No significant issues detected"]
    
    def _extract_recommendations(self, response: str) -> List[str]:
        """Extract recommendations from response"""
        recommendations = []
        
        if "lighting" in response.lower():
            recommendations.append("Improve lighting conditions")
        if "focus" in response.lower() or "blur" in response.lower():
            recommendations.append("Ensure camera is in focus before capture")
        if "distance" in response.lower():
            recommendations.append("Adjust distance from subject")
        if "angle" in response.lower():
            recommendations.append("Capture from a better angle")
        
        return recommendations if recommendations else ["Image quality is acceptable"]
    
    def _parse_findings(self, response: str) -> List[Dict[str, Any]]:
        """Parse clinical findings from response"""
        findings = []
        
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                clean_line = line.lstrip('0123456789.-•) ').strip()
                if clean_line and len(clean_line) > 10:
                    findings.append({
                        "type": "observation",
                        "description": clean_line
                    })
        
        if not findings:
            findings.append({
                "type": "observation",
                "description": "See raw analysis for detailed findings"
            })
        
        return findings
    
    def _extract_severity_score(self, response: str) -> float:
        """Extract severity score from response"""
        import re
        
        patterns = [
            r'severity[:\s]+(\d+(?:\.\d+)?)',
            r'severity[:\s]+score[:\s]+(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)[:\s]*(?:out of|/)?\s*10'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response.lower())
            if match:
                score = float(match.group(1))
                return min(10.0, max(0.0, score))
        
        if any(word in response.lower() for word in ["urgent", "immediate", "concerning", "severe"]):
            return 7.0
        if any(word in response.lower() for word in ["moderate", "notable", "attention"]):
            return 5.0
        if any(word in response.lower() for word in ["mild", "minor", "slight"]):
            return 3.0
        if any(word in response.lower() for word in ["normal", "healthy", "unremarkable"]):
            return 1.0
        
        return 3.0
    
    def _extract_confidence_score(self, response: str) -> float:
        """Extract confidence score from response"""
        import re
        
        patterns = [
            r'confidence[:\s]+(\d+(?:\.\d+)?)',
            r'confidence[:\s]+score[:\s]+(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*%?\s*confidence'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response.lower())
            if match:
                score = float(match.group(1))
                if score > 1:
                    return min(100.0, max(0.0, score)) / 100.0
                return min(1.0, max(0.0, score))
        
        return 0.75
    
    def _extract_clinical_recommendations(self, response: str) -> List[str]:
        """Extract clinical recommendations from response"""
        recommendations = []
        
        keywords = [
            "recommend", "suggest", "advise", "should", "consider",
            "follow-up", "follow up", "further", "additional"
        ]
        
        lines = response.split('\n')
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in keywords):
                clean_line = line.strip().lstrip('0123456789.-•) ').strip()
                if clean_line and len(clean_line) > 10:
                    recommendations.append(clean_line)
        
        if not recommendations:
            recommendations.append("No specific recommendations from analysis")
        
        return recommendations[:5]


openai_vision_service = OpenAIVisionService()
