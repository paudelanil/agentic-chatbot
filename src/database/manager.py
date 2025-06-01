"""
Database manager for handling all database operations.
"""
import sqlite3
import json
import logging
from typing import List, Optional
from datetime import datetime
from contextlib import contextmanager

from ..database.models import User, MemoryItem, UserFile, MemoryType
from config.settings import settings


logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages all database operations for the chatbot."""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or settings.database.path
        self.init_database()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def init_database(self) -> None:
        """Initialize the database with required tables."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Memory table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    content TEXT,
                    is_user BOOLEAN,
                    importance FLOAT,
                    embedding TEXT,
                    memory_type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # Files table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_files (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    filename TEXT,
                    content TEXT,
                    chunks TEXT,
                    embeddings TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_memories_user_type 
                ON memories (user_id, memory_type)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_memories_created_at 
                ON memories (created_at)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_files_user 
                ON user_files (user_id)
            ''')
            
            conn.commit()
            logger.info("Database initialized successfully")
    
    def create_user(self, user_id: str) -> User:
        """Create a new user or return existing one."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Check if user exists
            cursor.execute('SELECT id, created_at FROM users WHERE id = ?', (user_id,))
            row = cursor.fetchone()
            
            if row:
                return User(id=row['id'], created_at=datetime.fromisoformat(row['created_at']))
            
            # Create new user
            now = datetime.now()
            cursor.execute('''
                INSERT INTO users (id, created_at) VALUES (?, ?)
            ''', (user_id, now.isoformat()))
            
            conn.commit()
            logger.info(f"Created new user: {user_id}")
            return User(id=user_id, created_at=now)
    
    def save_memory(self, user_id: str, memory_item: MemoryItem) -> None:
        """Save a memory item to the database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Ensure user exists
            self.create_user(user_id)
            
            # Serialize embedding if it's a list
            embedding = json.dumps(memory_item.embedding) if memory_item.embedding is not None else None
            cursor.execute('''
                INSERT OR REPLACE INTO memories 
                (id, user_id, content, is_user, importance, embedding, memory_type, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                memory_item.id,
                user_id,
                memory_item.content,
                memory_item.is_user,
                memory_item.importance,
                embedding,
                memory_item.memory_type.value,
                memory_item.created_at.isoformat()
            ))
            
            conn.commit()
            
            # Clean up old memories
            self._cleanup_old_memories(user_id, memory_item.memory_type)
            logger.debug(f"Saved memory for user {user_id}: {memory_item.memory_type.value}")
    
    def _cleanup_old_memories(self, user_id: str, memory_type: MemoryType) -> None:
        """Clean up old memories to maintain limits."""
        limit = (settings.database.long_term_memory_limit 
                if memory_type == MemoryType.LONG_TERM 
                else settings.database.short_term_memory_limit)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                DELETE FROM memories 
                WHERE user_id = ? AND memory_type = ? 
                AND id NOT IN (
                    SELECT id FROM memories 
                    WHERE user_id = ? AND memory_type = ?
                    ORDER BY created_at DESC LIMIT ?
                )
            ''', (user_id, memory_type.value, user_id, memory_type.value, limit))
            
            deleted_count = cursor.rowcount
            if deleted_count > 0:
                logger.debug(f"Cleaned up {deleted_count} old {memory_type.value} memories for user {user_id}")
            
            conn.commit()
    
    def load_memories(self, user_id: str, memory_type, limit: int = 50) -> List[MemoryItem]:
        """Load memories for a user."""
        # Ensure memory_type is a string (if it's an Enum, get its value)
        if hasattr(memory_type, "value"):
            memory_type = memory_type.value
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, user_id, content, is_user, importance, embedding, memory_type, created_at 
                FROM memories 
                WHERE user_id = ? AND memory_type = ?
                ORDER BY created_at DESC LIMIT ?
            ''', (user_id, memory_type, limit))
            memories = []
            for row in cursor.fetchall():
                memory = MemoryItem(
                    id=row['id'],
                    user_id=row['user_id'],
                    content=row['content'],
                    is_user=bool(row['is_user']),
                    importance=row['importance'],
                    embedding=json.loads(row['embedding']) if row['embedding'] else None,
                    memory_type=MemoryType(row['memory_type']),
                    created_at=datetime.fromisoformat(row['created_at'])
                )
                memories.append(memory)
            return memories
    
    def save_file(self, user_id: str, file_data: UserFile) -> None:
        """Save a user file to the database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Ensure user exists
            self.create_user(user_id)
            
            # Serialize chunks and embeddings
            chunks = json.dumps(file_data.chunks)
            embeddings = json.dumps(file_data.embeddings)
            cursor.execute('''
                INSERT OR REPLACE INTO user_files 
                (id, user_id, filename, content, chunks, embeddings, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                file_data.id,
                user_id,
                file_data.filename,
                file_data.content,
                chunks,
                embeddings,
                file_data.created_at.isoformat()
            ))
            
            conn.commit()
            logger.info(f"Saved file for user {user_id}: {file_data.filename}")
    
    def load_files(self, user_id: str) -> List[UserFile]:
        """Load all files for a user."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, user_id, filename, content, chunks, embeddings, created_at 
                FROM user_files 
                WHERE user_id = ?
                ORDER BY created_at DESC
            ''', (user_id,))
            
            files = []
            for row in cursor.fetchall():
                try:
                    file_data = UserFile(
                        id=row['id'],
                        user_id=row['user_id'],
                        filename=row['filename'],
                        content=row['content'],
                        chunks=json.loads(row['chunks']),
                        embeddings=json.loads(row['embeddings']),
                        created_at=datetime.fromisoformat(row['created_at'])
                    )
                    files.append(file_data)
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Error loading file {row['id']}: {e}")
                    continue
            
            return files
    
    def delete_user_memories(self, user_id: str) -> int:
        """Delete all memories for a user. Returns count of deleted memories."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM memories WHERE user_id = ?', (user_id,))
            deleted_count = cursor.rowcount
            conn.commit()
            logger.info(f"Deleted {deleted_count} memories for user {user_id}")
            return deleted_count
    
    def delete_user_files(self, user_id: str) -> int:
        """Delete all files for a user. Returns count of deleted files."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM user_files WHERE user_id = ?', (user_id,))
            deleted_count = cursor.rowcount
            conn.commit()
            logger.info(f"Deleted {deleted_count} files for user {user_id}")
            return deleted_count
    
    def get_user_stats(self, user_id: str) -> dict:
        """Get statistics for a user."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Count memories by type
            cursor.execute('''
                SELECT memory_type, COUNT(*) as count 
                FROM memories 
                WHERE user_id = ? 
                GROUP BY memory_type
            ''', (user_id,))
            
            memory_counts = {row['memory_type']: row['count'] for row in cursor.fetchall()}
            
            # Count files
            cursor.execute('SELECT COUNT(*) as count FROM user_files WHERE user_id = ?', (user_id,))
            file_count = cursor.fetchone()['count']
            
            return {
                'short_term_memories': memory_counts.get('short_term', 0),
                'long_term_memories': memory_counts.get('long_term', 0),
                'files': file_count
            }