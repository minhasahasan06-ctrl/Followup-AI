"""
LLM Safe Client - HIPAA-compliant OpenAI wrapper
Enforces PHI detection and BAA/ZDR checks before all OpenAI API calls.
"""

import os
import logging
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

from openai import OpenAI, AsyncOpenAI

from app.services.phi_detection_service import get_phi_detection_service, PHIDetectionResult
from app.services.access_control import HIPAAAuditLogger

logger = logging.getLogger(__name__)


@dataclass
class SafeCompletionResult:
    """Result of a safe LLM completion."""
    success: bool
    content: Optional[str] = None
    phi_detected: bool = False
    phi_blocked: bool = False
    redacted_prompt: Optional[str] = None
    audit_id: Optional[str] = None
    error: Optional[str] = None
    model: str = "gpt-4o"
    usage: Optional[Dict[str, int]] = None


class LLMSafeClient:
    """
    HIPAA-compliant OpenAI wrapper that:
    1. Detects PHI in prompts before sending
    2. Blocks requests if BAA/ZDR not configured
    3. Logs all API calls for audit
    4. Optionally redacts PHI before sending
    """
    
    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)
        
        self.baa_signed = os.environ.get("OPENAI_BAA_SIGNED", "").lower() == "true"
        self.zdr_enabled = os.environ.get("OPENAI_ZDR_ENABLED", "").lower() == "true"
        
        self._phi_service = None
    
    @property
    def phi_service(self):
        if self._phi_service is None:
            self._phi_service = get_phi_detection_service()
        return self._phi_service
    
    @property
    def is_hipaa_compliant(self) -> bool:
        """Check if OpenAI configuration is HIPAA compliant."""
        return self.baa_signed and self.zdr_enabled
    
    def _validate_compliance(self, allow_phi: bool = False) -> Optional[str]:
        """Validate HIPAA compliance before API call."""
        if allow_phi and not self.baa_signed:
            return "BLOCKED: BAA not signed with OpenAI. Cannot process PHI."
        if allow_phi and not self.zdr_enabled:
            return "BLOCKED: Zero Data Retention not enabled. Cannot process PHI."
        return None
    
    def safe_completion(
        self,
        messages: List[Dict[str, str]],
        actor_id: str,
        actor_role: str,
        patient_id: Optional[str] = None,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        allow_phi: bool = False,
        redact_phi: bool = True,
        access_reason: str = "clinical_ai",
        **kwargs
    ) -> SafeCompletionResult:
        """
        Execute a safe OpenAI completion with PHI protection.
        
        Args:
            messages: OpenAI chat messages
            actor_id: User making the request
            actor_role: Role of the user (doctor, patient, admin)
            patient_id: Patient ID if accessing patient data
            model: OpenAI model to use
            temperature: Model temperature
            max_tokens: Maximum tokens in response
            allow_phi: Whether to allow PHI in prompts (requires BAA)
            redact_phi: Whether to redact PHI before sending
            access_reason: Reason for access (for audit logging)
            **kwargs: Additional OpenAI parameters
        
        Returns:
            SafeCompletionResult with response or error
        """
        compliance_error = self._validate_compliance(allow_phi)
        if compliance_error and allow_phi:
            return SafeCompletionResult(
                success=False,
                phi_blocked=True,
                error=compliance_error
            )
        
        combined_content = " ".join([m.get("content", "") for m in messages])
        phi_result: PHIDetectionResult = self.phi_service.detect_phi(combined_content)
        
        if phi_result.phi_detected:
            if not allow_phi:
                audit_id = HIPAAAuditLogger.log_phi_access(
                    actor_id=actor_id,
                    actor_role=actor_role,
                    patient_id=patient_id or "unknown",
                    action="llm_phi_blocked",
                    phi_categories=[e.category.value for e in phi_result.phi_entities],
                    resource_type="llm_completion",
                    access_reason=access_reason,
                    success=False,
                    error_message="PHI detected and blocked"
                )
                return SafeCompletionResult(
                    success=False,
                    phi_detected=True,
                    phi_blocked=True,
                    audit_id=audit_id,
                    error="PHI detected in prompt. Set allow_phi=True with proper BAA/ZDR."
                )
            
            if redact_phi:
                redacted_messages = []
                for msg in messages:
                    content = msg.get("content", "")
                    redacted = self.phi_service.detect_phi(content)
                    redacted_messages.append({
                        **msg,
                        "content": redacted.redacted_text
                    })
                messages = redacted_messages
        
        audit_id = HIPAAAuditLogger.log_phi_access(
            actor_id=actor_id,
            actor_role=actor_role,
            patient_id=patient_id or "system",
            action="llm_completion",
            phi_categories=[e.category.value for e in phi_result.phi_entities] if phi_result.phi_detected else [],
            resource_type="llm_completion",
            access_reason=access_reason,
            success=True,
            additional_context={
                "model": model,
                "phi_detected": phi_result.phi_detected,
                "phi_redacted": redact_phi and phi_result.phi_detected
            }
        )
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            return SafeCompletionResult(
                success=True,
                content=response.choices[0].message.content,
                phi_detected=phi_result.phi_detected,
                phi_blocked=False,
                redacted_prompt=phi_result.redacted_text if phi_result.phi_detected and redact_phi else None,
                audit_id=audit_id,
                model=model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                } if response.usage else None
            )
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return SafeCompletionResult(
                success=False,
                phi_detected=phi_result.phi_detected,
                audit_id=audit_id,
                error=str(e)
            )
    
    async def safe_completion_async(
        self,
        messages: List[Dict[str, str]],
        actor_id: str,
        actor_role: str,
        patient_id: Optional[str] = None,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        allow_phi: bool = False,
        redact_phi: bool = True,
        access_reason: str = "clinical_ai",
        **kwargs
    ) -> SafeCompletionResult:
        """Async version of safe_completion."""
        compliance_error = self._validate_compliance(allow_phi)
        if compliance_error and allow_phi:
            return SafeCompletionResult(
                success=False,
                phi_blocked=True,
                error=compliance_error
            )
        
        combined_content = " ".join([m.get("content", "") for m in messages])
        phi_result: PHIDetectionResult = await self.phi_service.detect_phi_async(combined_content)
        
        if phi_result.phi_detected:
            if not allow_phi:
                audit_id = HIPAAAuditLogger.log_phi_access(
                    actor_id=actor_id,
                    actor_role=actor_role,
                    patient_id=patient_id or "unknown",
                    action="llm_phi_blocked",
                    phi_categories=[e.category.value for e in phi_result.phi_entities],
                    resource_type="llm_completion",
                    access_reason=access_reason,
                    success=False,
                    error_message="PHI detected and blocked"
                )
                return SafeCompletionResult(
                    success=False,
                    phi_detected=True,
                    phi_blocked=True,
                    audit_id=audit_id,
                    error="PHI detected in prompt. Set allow_phi=True with proper BAA/ZDR."
                )
            
            if redact_phi:
                redacted_messages = []
                for msg in messages:
                    content = msg.get("content", "")
                    redacted = await self.phi_service.detect_phi_async(content)
                    redacted_messages.append({
                        **msg,
                        "content": redacted.redacted_text
                    })
                messages = redacted_messages
        
        audit_id = HIPAAAuditLogger.log_phi_access(
            actor_id=actor_id,
            actor_role=actor_role,
            patient_id=patient_id or "system",
            action="llm_completion",
            phi_categories=[e.category.value for e in phi_result.phi_entities] if phi_result.phi_detected else [],
            resource_type="llm_completion",
            access_reason=access_reason,
            success=True,
            additional_context={
                "model": model,
                "phi_detected": phi_result.phi_detected,
                "phi_redacted": redact_phi and phi_result.phi_detected
            }
        )
        
        try:
            response = await self.async_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            return SafeCompletionResult(
                success=True,
                content=response.choices[0].message.content,
                phi_detected=phi_result.phi_detected,
                phi_blocked=False,
                redacted_prompt=phi_result.redacted_text if phi_result.phi_detected and redact_phi else None,
                audit_id=audit_id,
                model=model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                } if response.usage else None
            )
            
        except Exception as e:
            logger.error(f"Async OpenAI API error: {e}")
            return SafeCompletionResult(
                success=False,
                phi_detected=phi_result.phi_detected,
                audit_id=audit_id,
                error=str(e)
            )


_llm_client_instance: Optional[LLMSafeClient] = None


def get_llm_safe_client() -> LLMSafeClient:
    """Get singleton LLM safe client instance."""
    global _llm_client_instance
    if _llm_client_instance is None:
        _llm_client_instance = LLMSafeClient()
    return _llm_client_instance
