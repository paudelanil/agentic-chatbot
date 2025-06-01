"""
Configuration settings for the LangGraph Chatbot application.
"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    path: str = "chatbot.db"
    short_term_memory_limit: int = 50
    long_term_memory_limit: int = 100


@dataclass
class LLMConfig:
    """LLM configuration settings."""
    api_key: str = ""
    model_name: str = "llama3-8b-8192"
    temperature: float = 0.7
    max_tokens: Optional[int] = None


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class FileConfig:
    """File processing configuration."""
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_extensions: tuple = ('.txt', '.pdf', '.md', '.doc', '.docx')
    chunk_size: int = 1000
    chunk_overlap: int = 200


@dataclass
class WebConfig:
    """Web application configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    secret_key: str = "your-secret-key-change-in-production"
    session_timeout: int = 30 * 24 * 60 * 60  # 30 days


@dataclass
class MemoryConfig:
    """Memory management configuration."""
    importance_threshold: float = 7.0
    similarity_top_k: int = 5
    context_window: int = 5


class Settings:
    """Main settings class that loads configuration from environment variables."""
    
    def __init__(self):
        self.database = DatabaseConfig(
            path=os.getenv("DATABASE_PATH", "chatbot.db"),
            short_term_memory_limit=int(os.getenv("SHORT_TERM_MEMORY_LIMIT", "50")),
            long_term_memory_limit=int(os.getenv("LONG_TERM_MEMORY_LIMIT", "100"))
        )
        
        self.llm = LLMConfig(
            api_key=os.getenv("GROQ_API_KEY", ""),
            model_name=os.getenv("MODEL_NAME", "llama3-8b-8192"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("MAX_TOKENS")) if os.getenv("MAX_TOKENS") else None
        )
        
        self.embedding = EmbeddingConfig(
            model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        )
        
        self.file = FileConfig(
            max_file_size=int(os.getenv("MAX_FILE_SIZE", str(10 * 1024 * 1024))),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200"))
        )
        
        self.web = WebConfig(
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
            debug=os.getenv("DEBUG", "False").lower() == "true",
            secret_key=os.getenv("SECRET_KEY", "your-secret-key-change-in-production"),
            session_timeout=int(os.getenv("SESSION_TIMEOUT", str(30 * 24 * 60 * 60)))
        )
        
        self.memory = MemoryConfig(
            importance_threshold=float(os.getenv("IMPORTANCE_THRESHOLD", "7.0")),
            similarity_top_k=int(os.getenv("SIMILARITY_TOP_K", "5")),
            context_window=int(os.getenv("CONTEXT_WINDOW", "5"))
        )
    
    def validate(self) -> None:
        """Validate configuration settings."""
        if not self.llm.api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        
        if self.file.chunk_size <= self.file.chunk_overlap:
            raise ValueError("Chunk size must be greater than chunk overlap")
        
        if self.memory.importance_threshold < 0 or self.memory.importance_threshold > 10:
            raise ValueError("Importance threshold must be between 0 and 10")


# Global settings instance
settings = Settings()