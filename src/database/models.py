"""
Data models for the chatbot application.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
from typing_extensions import TypedDict


class MemoryType(Enum):
    """Memory type enumeration."""
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"


@dataclass
class User:
    """User model."""
    id: str
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class MemoryItem:
    """Memory item model."""
    id: str
    user_id: str
    content: str
    is_user: bool
    importance: float
    embedding: Optional[List[float]]
    memory_type: MemoryType
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "content": self.content,
            "is_user": self.is_user,
            "importance": self.importance,
            "embedding": self.embedding,
            "memory_type": self.memory_type.value,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_db_row(cls, row: tuple) -> 'MemoryItem':
        """Create MemoryItem from database row."""
        return cls(
            id=row[0],
            user_id=row[1],
            content=row[2],
            is_user=bool(row[3]),
            importance=row[4],
            embedding=row[5] if row[5] else None,
            memory_type=MemoryType(row[6]),
            created_at=datetime.fromisoformat(row[7])
        )


@dataclass
class UserFile:
    """User file model."""
    id: str
    user_id: str
    filename: str
    content: str
    chunks: List[str]
    embeddings: List[List[float]]
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "filename": self.filename,
            "content": self.content,
            "chunks": self.chunks,
            "embeddings": self.embeddings,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_db_row(cls, row: tuple) -> 'UserFile':
        """Create UserFile from database row."""
        import json
        return cls(
            id=row[0],
            user_id=row[1],
            filename=row[2],
            content=row[3],
            chunks=json.loads(row[4]),
            embeddings=json.loads(row[5]),
            created_at=datetime.fromisoformat(row[6])
        )


@dataclass
class ChatMessage:
    """Chat message model."""
    role: str
    content: str
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ChatState:
    """Chat state model for LangGraph."""
    messages: List[Dict[str, Any]]
    short_term_memory: List[Dict[str, Any]]
    long_term_memory: List[Dict[str, Any]]
    uploaded_files: List[Dict[str, Any]]
    current_query: str
    tool_results: List[str]
    context: str
    user_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "messages": self.messages,
            "short_term_memory": self.short_term_memory,
            "long_term_memory": self.long_term_memory,
            "uploaded_files": self.uploaded_files,
            "current_query": self.current_query,
            "tool_results": self.tool_results,
            "context": self.context,
            "user_id": self.user_id
        }