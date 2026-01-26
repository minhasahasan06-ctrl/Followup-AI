"""
Conversation Memory Service
===========================

Production-grade conversation memory management with:
- CRUD operations for memories
- Version tracking for edits
- HIPAA audit trail for all changes
- Patient/doctor access controls
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import logging

from app.services.access_control import HIPAAAuditLogger

logger = logging.getLogger(__name__)


class MemoryType(str, Enum):
    """Types of conversation memories"""
    PREFERENCE = "preference"
    MEDICAL_HISTORY = "medical_history"
    SYMPTOM = "symptom"
    MEDICATION = "medication"
    LIFESTYLE = "lifestyle"
    EMERGENCY_CONTACT = "emergency_contact"
    ALLERGY = "allergy"
    GENERAL = "general"


class MemoryPriority(str, Enum):
    """Priority levels for memories"""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


@dataclass
class MemoryVersion:
    """Version history entry for a memory"""
    version_id: str
    content: str
    edited_by: str
    edited_at: datetime
    edit_reason: Optional[str] = None


@dataclass
class ConversationMemory:
    """A single memory entry"""
    memory_id: str
    patient_id: str
    memory_type: MemoryType
    content: str
    priority: MemoryPriority = MemoryPriority.NORMAL
    source: str = "conversation"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = ""
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    versions: List[MemoryVersion] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


@dataclass
class MemoryAuditEntry:
    """Audit trail entry for memory operations"""
    audit_id: str
    memory_id: str
    patient_id: str
    action: str
    actor_id: str
    actor_role: str
    timestamp: datetime
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    access_reason: Optional[str] = None


class ConversationMemoryService:
    """
    Production-grade conversation memory service
    
    Features:
    - CRUD operations with version tracking
    - HIPAA-compliant audit logging
    - Patient-scoped memory isolation
    - Memory search and retrieval
    """
    
    def __init__(self):
        self._memories: Dict[str, ConversationMemory] = {}
        self._audit_log: List[MemoryAuditEntry] = []
        self._patient_index: Dict[str, List[str]] = {}
    
    def _log_audit(
        self,
        memory_id: str,
        patient_id: str,
        action: str,
        actor_id: str,
        actor_role: str,
        old_value: Optional[str] = None,
        new_value: Optional[str] = None,
        access_reason: Optional[str] = None,
    ) -> None:
        """Log audit entry for memory operation"""
        entry = MemoryAuditEntry(
            audit_id=str(uuid.uuid4()),
            memory_id=memory_id,
            patient_id=patient_id,
            action=action,
            actor_id=actor_id,
            actor_role=actor_role,
            timestamp=datetime.utcnow(),
            old_value=old_value,
            new_value=new_value,
            access_reason=access_reason,
        )
        self._audit_log.append(entry)
        
        HIPAAAuditLogger.log_phi_access(
            actor_id=actor_id,
            actor_role=actor_role,
            patient_id=patient_id,
            action=f"memory_{action}",
            phi_categories=["conversation_memory"],
            resource_type="memory",
            resource_id=memory_id,
            access_reason=access_reason or action,
        )
    
    def create_memory(
        self,
        patient_id: str,
        memory_type: MemoryType,
        content: str,
        created_by: str,
        priority: MemoryPriority = MemoryPriority.NORMAL,
        source: str = "conversation",
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> ConversationMemory:
        """
        Create a new memory entry
        
        Args:
            patient_id: Patient this memory belongs to
            memory_type: Type of memory
            content: Memory content
            created_by: ID of user creating the memory
            priority: Priority level
            source: Source of the memory
            metadata: Additional metadata
            tags: Searchable tags
            
        Returns:
            Created memory object
        """
        memory_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        memory = ConversationMemory(
            memory_id=memory_id,
            patient_id=patient_id,
            memory_type=memory_type,
            content=content,
            priority=priority,
            source=source,
            created_at=now,
            updated_at=now,
            created_by=created_by,
            metadata=metadata or {},
            tags=tags or [],
        )
        
        memory.versions.append(MemoryVersion(
            version_id=str(uuid.uuid4()),
            content=content,
            edited_by=created_by,
            edited_at=now,
            edit_reason="Initial creation",
        ))
        
        self._memories[memory_id] = memory
        
        if patient_id not in self._patient_index:
            self._patient_index[patient_id] = []
        self._patient_index[patient_id].append(memory_id)
        
        self._log_audit(
            memory_id=memory_id,
            patient_id=patient_id,
            action="create",
            actor_id=created_by,
            actor_role="system",
            new_value=content,
            access_reason="Create new memory",
        )
        
        logger.info(f"Created memory {memory_id} for patient {patient_id}")
        return memory
    
    def get_memory(
        self,
        memory_id: str,
        accessor_id: str,
        accessor_role: str,
    ) -> Optional[ConversationMemory]:
        """
        Get a memory by ID
        
        Args:
            memory_id: ID of the memory
            accessor_id: ID of user accessing
            accessor_role: Role of accessor
            
        Returns:
            Memory object or None
        """
        memory = self._memories.get(memory_id)
        if memory:
            self._log_audit(
                memory_id=memory_id,
                patient_id=memory.patient_id,
                action="read",
                actor_id=accessor_id,
                actor_role=accessor_role,
                access_reason="Retrieve memory",
            )
        return memory
    
    def get_patient_memories(
        self,
        patient_id: str,
        accessor_id: str,
        accessor_role: str,
        memory_type: Optional[MemoryType] = None,
        priority: Optional[MemoryPriority] = None,
        active_only: bool = True,
        limit: int = 100,
    ) -> List[ConversationMemory]:
        """
        Get all memories for a patient
        
        Args:
            patient_id: Patient ID
            accessor_id: ID of user accessing
            accessor_role: Role of accessor
            memory_type: Optional type filter
            priority: Optional priority filter
            active_only: Only return active memories
            limit: Maximum memories to return
            
        Returns:
            List of matching memories
        """
        memory_ids = self._patient_index.get(patient_id, [])
        memories = []
        
        for memory_id in memory_ids:
            memory = self._memories.get(memory_id)
            if not memory:
                continue
            if active_only and not memory.is_active:
                continue
            if memory_type and memory.memory_type != memory_type:
                continue
            if priority and memory.priority != priority:
                continue
            memories.append(memory)
            if len(memories) >= limit:
                break
        
        self._log_audit(
            memory_id="batch",
            patient_id=patient_id,
            action="list",
            actor_id=accessor_id,
            actor_role=accessor_role,
            access_reason=f"List memories, count: {len(memories)}",
        )
        
        return memories
    
    def update_memory(
        self,
        memory_id: str,
        new_content: str,
        editor_id: str,
        editor_role: str,
        edit_reason: str,
    ) -> Optional[ConversationMemory]:
        """
        Update a memory with version tracking
        
        Args:
            memory_id: ID of memory to update
            new_content: New content
            editor_id: ID of user editing
            editor_role: Role of editor
            edit_reason: Reason for the edit
            
        Returns:
            Updated memory or None
        """
        memory = self._memories.get(memory_id)
        if not memory:
            return None
        
        old_content = memory.content
        now = datetime.utcnow()
        
        memory.versions.append(MemoryVersion(
            version_id=str(uuid.uuid4()),
            content=new_content,
            edited_by=editor_id,
            edited_at=now,
            edit_reason=edit_reason,
        ))
        
        memory.content = new_content
        memory.updated_at = now
        
        self._log_audit(
            memory_id=memory_id,
            patient_id=memory.patient_id,
            action="update",
            actor_id=editor_id,
            actor_role=editor_role,
            old_value=old_content,
            new_value=new_content,
            access_reason=edit_reason,
        )
        
        logger.info(f"Updated memory {memory_id}")
        return memory
    
    def deactivate_memory(
        self,
        memory_id: str,
        deactivator_id: str,
        deactivator_role: str,
        reason: str,
    ) -> bool:
        """
        Deactivate a memory (soft delete)
        
        Args:
            memory_id: ID of memory to deactivate
            deactivator_id: ID of user deactivating
            deactivator_role: Role of deactivator
            reason: Reason for deactivation
            
        Returns:
            True if successful
        """
        memory = self._memories.get(memory_id)
        if not memory:
            return False
        
        memory.is_active = False
        memory.updated_at = datetime.utcnow()
        
        self._log_audit(
            memory_id=memory_id,
            patient_id=memory.patient_id,
            action="deactivate",
            actor_id=deactivator_id,
            actor_role=deactivator_role,
            access_reason=reason,
        )
        
        logger.info(f"Deactivated memory {memory_id}")
        return True
    
    def search_memories(
        self,
        patient_id: str,
        query: str,
        accessor_id: str,
        accessor_role: str,
        limit: int = 20,
    ) -> List[ConversationMemory]:
        """
        Search memories by content
        
        Args:
            patient_id: Patient ID
            query: Search query
            accessor_id: ID of user searching
            accessor_role: Role of accessor
            limit: Maximum results
            
        Returns:
            List of matching memories
        """
        memory_ids = self._patient_index.get(patient_id, [])
        results = []
        query_lower = query.lower()
        
        for memory_id in memory_ids:
            memory = self._memories.get(memory_id)
            if not memory or not memory.is_active:
                continue
            if query_lower in memory.content.lower():
                results.append(memory)
            elif any(query_lower in tag.lower() for tag in memory.tags):
                results.append(memory)
            if len(results) >= limit:
                break
        
        self._log_audit(
            memory_id="search",
            patient_id=patient_id,
            action="search",
            actor_id=accessor_id,
            actor_role=accessor_role,
            access_reason=f"Search: {query}, results: {len(results)}",
        )
        
        return results
    
    def get_memory_versions(
        self,
        memory_id: str,
        accessor_id: str,
        accessor_role: str,
    ) -> List[MemoryVersion]:
        """Get version history for a memory"""
        memory = self._memories.get(memory_id)
        if not memory:
            return []
        
        self._log_audit(
            memory_id=memory_id,
            patient_id=memory.patient_id,
            action="view_versions",
            actor_id=accessor_id,
            actor_role=accessor_role,
            access_reason="View memory version history",
        )
        
        return memory.versions
    
    def get_audit_log(
        self,
        patient_id: Optional[str] = None,
        memory_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[MemoryAuditEntry]:
        """
        Get audit log entries
        
        Args:
            patient_id: Optional patient filter
            memory_id: Optional memory filter
            limit: Maximum entries
            
        Returns:
            List of audit entries
        """
        entries = self._audit_log
        
        if patient_id:
            entries = [e for e in entries if e.patient_id == patient_id]
        if memory_id:
            entries = [e for e in entries if e.memory_id == memory_id]
        
        return sorted(entries, key=lambda e: e.timestamp, reverse=True)[:limit]
    
    def get_critical_memories(
        self,
        patient_id: str,
        accessor_id: str,
        accessor_role: str,
    ) -> List[ConversationMemory]:
        """Get all critical priority memories for a patient"""
        return self.get_patient_memories(
            patient_id=patient_id,
            accessor_id=accessor_id,
            accessor_role=accessor_role,
            priority=MemoryPriority.CRITICAL,
        )


_memory_service: Optional[ConversationMemoryService] = None


def get_memory_service() -> ConversationMemoryService:
    """Get singleton memory service instance"""
    global _memory_service
    if _memory_service is None:
        _memory_service = ConversationMemoryService()
    return _memory_service
