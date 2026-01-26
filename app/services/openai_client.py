"""
Centralized OpenAI Client Wrapper
HIPAA-compliant with PHI detection, BAA/ZDR enforcement, and audit logging.

This module provides:
1. Runtime enforcement of OPENAI_BAA and OPENAI_ZDR in production
2. PHI detection and optional redaction before sending to OpenAI
3. Comprehensive audit logging of all OpenAI API calls
4. Centralized configuration and error handling
"""

import os
import re
import hashlib
import logging
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from functools import wraps

from openai import OpenAI, AsyncOpenAI

logger = logging.getLogger(__name__)

ENV = os.getenv("ENV", "dev")
OPENAI_BAA = os.getenv("OPENAI_BAA", "false").lower() == "true"
OPENAI_ZDR = os.getenv("OPENAI_ZDR", "false").lower() == "true"

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_VERSION = "v1.0.0"
EMBEDDING_DIMENSION = 1536


class PHIDetectionError(Exception):
    """Raised when PHI is detected and not allowed"""
    pass


class OpenAIConfigError(Exception):
    """Raised when OpenAI configuration is invalid"""
    pass


class DirectIdentifierPatterns:
    """Regex patterns for direct PHI identifiers"""
    SSN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
    MRN = re.compile(r'\b(MRN|mrn|Medical Record Number)[:\s#]*\d{6,}\b', re.IGNORECASE)
    EMAIL = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    PHONE = re.compile(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b')
    CREDIT_CARD = re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b')


class OpenAIClientWrapper:
    """
    Centralized OpenAI client with HIPAA compliance enforcement.
    
    Features:
    - Runtime BAA/ZDR enforcement in production
    - PHI detection and redaction
    - Comprehensive audit logging
    - Standardized embedding generation
    """
    
    def __init__(self, audit_logger=None):
        self._validate_production_config()
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise OpenAIConfigError("OPENAI_API_KEY is required")
        
        self._sync_client = OpenAI(api_key=api_key)
        self._async_client = AsyncOpenAI(api_key=api_key)
        self._audit_logger = audit_logger
        self._phi_detection_enabled = os.getenv("PHI_DETECTION_ENABLED", "true").lower() == "true"
        self._phi_block_on_detect = os.getenv("PHI_BLOCK_ON_DETECT", "true").lower() == "true"
        
        logger.info(f"OpenAI client initialized - ENV={ENV}, BAA={OPENAI_BAA}, ZDR={OPENAI_ZDR}")
    
    def _validate_production_config(self):
        """Validate that production has BAA and ZDR enabled"""
        if ENV == "prod" and not (OPENAI_BAA and OPENAI_ZDR):
            raise OpenAIConfigError(
                "Production environment requires OPENAI_BAA=true and OPENAI_ZDR=true. "
                "OpenAI BAA (Business Associate Agreement) and ZDR (Zero Data Retention) "
                "are required for HIPAA compliance."
            )
    
    def _detect_phi(self, text: str) -> List[Dict[str, Any]]:
        """Detect PHI in text using regex patterns for direct identifiers"""
        if not self._phi_detection_enabled:
            return []
        
        findings = []
        
        patterns = [
            (DirectIdentifierPatterns.SSN, "SSN"),
            (DirectIdentifierPatterns.MRN, "MRN"),
            (DirectIdentifierPatterns.EMAIL, "EMAIL"),
            (DirectIdentifierPatterns.PHONE, "PHONE"),
            (DirectIdentifierPatterns.CREDIT_CARD, "CREDIT_CARD"),
        ]
        
        for pattern, category in patterns:
            for match in pattern.finditer(text):
                findings.append({
                    "text": match.group(),
                    "category": category,
                    "start": match.start(),
                    "end": match.end(),
                })
        
        return findings
    
    def _redact_phi(self, text: str, findings: List[Dict[str, Any]]) -> str:
        """Redact PHI from text based on findings"""
        if not findings:
            return text
        
        sorted_findings = sorted(findings, key=lambda x: x["start"], reverse=True)
        redacted = text
        
        for finding in sorted_findings:
            placeholder = f"[{finding['category']}_REDACTED]"
            redacted = redacted[:finding["start"]] + placeholder + redacted[finding["end"]:]
        
        return redacted
    
    def _hash_input(self, text: str) -> str:
        """Create a hash of input for audit logging (no PHI in logs)"""
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    
    def _log_event(self, event_type: str, details: Dict[str, Any]):
        """Log audit event"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "env": ENV,
            "baa_enabled": OPENAI_BAA,
            "zdr_enabled": OPENAI_ZDR,
            **details
        }
        
        logger.info(f"OpenAI Audit: {event_type} - {details.get('model', 'unknown')}")
        
        if self._audit_logger:
            try:
                self._audit_logger.log_event(event_type, log_entry)
            except Exception as e:
                logger.error(f"Failed to write audit log: {e}")
    
    def _check_and_handle_phi(self, text: str, operation: str) -> str:
        """Check for PHI and handle according to configuration"""
        findings = self._detect_phi(text)
        
        if findings:
            self._log_event("phi_detected", {
                "operation": operation,
                "phi_categories": list(set(f["category"] for f in findings)),
                "phi_count": len(findings),
            })
            
            if self._phi_block_on_detect:
                raise PHIDetectionError(
                    f"PHI detected in {operation} request. Categories: "
                    f"{list(set(f['category'] for f in findings))}. "
                    "Either redact PHI before calling or disable PHI blocking."
                )
            else:
                return self._redact_phi(text, findings)
        
        return text
    
    async def embeddings_create(
        self,
        input_text: Union[str, List[str]],
        model: str = EMBEDDING_MODEL,
        **kwargs
    ):
        """
        Create embeddings with PHI detection and audit logging.
        
        Args:
            input_text: Text or list of texts to embed
            model: Embedding model to use
            **kwargs: Additional arguments for OpenAI API
        
        Returns:
            OpenAI embedding response
        """
        texts = [input_text] if isinstance(input_text, str) else input_text
        processed_texts = []
        
        for text in texts:
            processed_texts.append(self._check_and_handle_phi(text, "embedding"))
        
        input_for_api = processed_texts[0] if len(processed_texts) == 1 else processed_texts
        
        self._log_event("openai_embedding_request", {
            "model": model,
            "input_count": len(texts),
            "input_hash": self._hash_input(str(texts)),
        })
        
        try:
            response = await self._async_client.embeddings.create(
                model=model,
                input=input_for_api,
                **kwargs
            )
            
            self._log_event("openai_embedding_response", {
                "model": model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "total_tokens": response.usage.total_tokens,
                } if response.usage else None,
            })
            
            return response
            
        except Exception as e:
            self._log_event("openai_error", {
                "operation": "embedding",
                "model": model,
                "error": str(e),
            })
            raise
    
    async def chat_completions_create(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4o",
        **kwargs
    ):
        """
        Create chat completion with PHI detection and audit logging.
        
        Args:
            messages: List of message dicts with role and content
            model: Chat model to use
            **kwargs: Additional arguments for OpenAI API
        
        Returns:
            OpenAI chat completion response
        """
        processed_messages = []
        for msg in messages:
            processed_content = self._check_and_handle_phi(
                msg.get("content", ""), 
                "chat_completion"
            )
            processed_messages.append({**msg, "content": processed_content})
        
        self._log_event("openai_chat_request", {
            "model": model,
            "message_count": len(messages),
            "has_system_message": any(m.get("role") == "system" for m in messages),
        })
        
        try:
            response = await self._async_client.chat.completions.create(
                model=model,
                messages=processed_messages,
                **kwargs
            )
            
            self._log_event("openai_chat_response", {
                "model": model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                } if response.usage else None,
            })
            
            return response
            
        except Exception as e:
            self._log_event("openai_error", {
                "operation": "chat_completion",
                "model": model,
                "error": str(e),
            })
            raise
    
    def embeddings_create_sync(
        self,
        input_text: Union[str, List[str]],
        model: str = EMBEDDING_MODEL,
        **kwargs
    ):
        """Synchronous version of embeddings_create"""
        texts = [input_text] if isinstance(input_text, str) else input_text
        processed_texts = []
        
        for text in texts:
            processed_texts.append(self._check_and_handle_phi(text, "embedding"))
        
        input_for_api = processed_texts[0] if len(processed_texts) == 1 else processed_texts
        
        self._log_event("openai_embedding_request", {
            "model": model,
            "input_count": len(texts),
            "input_hash": self._hash_input(str(texts)),
        })
        
        try:
            response = self._sync_client.embeddings.create(
                model=model,
                input=input_for_api,
                **kwargs
            )
            
            self._log_event("openai_embedding_response", {
                "model": model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "total_tokens": response.usage.total_tokens,
                } if response.usage else None,
            })
            
            return response
            
        except Exception as e:
            self._log_event("openai_error", {
                "operation": "embedding",
                "model": model,
                "error": str(e),
            })
            raise
    
    def chat_completions_create_sync(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4o",
        **kwargs
    ):
        """Synchronous version of chat_completions_create"""
        processed_messages = []
        for msg in messages:
            processed_content = self._check_and_handle_phi(
                msg.get("content", ""), 
                "chat_completion"
            )
            processed_messages.append({**msg, "content": processed_content})
        
        self._log_event("openai_chat_request", {
            "model": model,
            "message_count": len(messages),
            "has_system_message": any(m.get("role") == "system" for m in messages),
        })
        
        try:
            response = self._sync_client.chat.completions.create(
                model=model,
                messages=processed_messages,
                **kwargs
            )
            
            self._log_event("openai_chat_response", {
                "model": model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                } if response.usage else None,
            })
            
            return response
            
        except Exception as e:
            self._log_event("openai_error", {
                "operation": "chat_completion",
                "model": model,
                "error": str(e),
            })
            raise


_openai_client: Optional[OpenAIClientWrapper] = None


def get_openai_client(audit_logger=None) -> OpenAIClientWrapper:
    """Get singleton OpenAI client instance"""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAIClientWrapper(audit_logger=audit_logger)
    return _openai_client


async def generate_embedding(text: str) -> Optional[List[float]]:
    """
    Generate embedding for text using the centralized OpenAI client.
    
    This is a convenience function that uses the singleton client.
    
    Args:
        text: Text to generate embedding for
        
    Returns:
        List of floats representing the embedding, or None on error
    """
    try:
        client = get_openai_client()
        response = await client.embeddings_create(text)
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        return None


def get_embedding_metadata() -> Dict[str, str]:
    """Get metadata about the current embedding configuration"""
    return {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_version": EMBEDDING_VERSION,
        "embedding_dimension": str(EMBEDDING_DIMENSION),
    }
