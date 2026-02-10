"""
Memory Schema Definition

Defines the structure for long-term memory storage.
Memories are persistent, reusable user information.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal
import uuid


@dataclass
class Memory:
    """Represents a single memory entry."""
    
    id: str
    user_id: str
    type: Literal["preference", "constraint", "fact"]
    key: str
    value: str
    confidence: float
    created_at: datetime
    last_updated: datetime
    
    def __post_init__(self):
        """Validate memory data."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        
        if self.type not in ["preference", "constraint", "fact"]:
            raise ValueError("Type must be 'preference', 'constraint', or 'fact'")
    
    @classmethod
    def create(
        cls,
        user_id: str,
        type: Literal["preference", "constraint", "fact"],
        key: str,
        value: str,
        confidence: float = 0.7,
    ) -> "Memory":
        """Create a new memory with auto-generated ID and timestamps."""
        now = datetime.now()
        return cls(
            id=str(uuid.uuid4()),
            user_id=user_id,
            type=type,
            key=key,
            value=value,
            confidence=confidence,
            created_at=now,
            last_updated=now
        )
    
    def to_dict(self) -> dict:
        """Convert memory to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "type": self.type,
            "key": self.key,
            "value": self.value,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Memory":
        """Create memory from dictionary."""
        return cls(
            id=data["id"],
            user_id=data.get("user_id", "guest"),
            type=data["type"],
            key=data["key"],
            value=data["value"],
            confidence=data["confidence"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_updated=datetime.fromisoformat(data["last_updated"])
        )
