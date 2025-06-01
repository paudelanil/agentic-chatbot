"""
Memory manager for handling conversation memory and retrieval.
"""
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from ..database.manager import DatabaseManager
from ..database.models import MemoryItem, MemoryType
from config.settings import settings


logger = logging.getLogger(__name__)


class MemoryManager:
    """Manages conversation memory and semantic retrieval."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.embedding.model_name)
        logger.info("Memory manager initialized")
    
    def calculate_importance(self, message: str) -> float:
        """Calculate importance score for memory storage."""
        important_keywords = [
            'remember', 'important', 'note', 'save', 'crucial', 
            'key', 'always', 'never', 'must', 'should', 'critical'
        ]
        
        score = 5.0  # Base score
        message_lower = message.lower()
        
        # Keyword importance
        for keyword in important_keywords:
            if keyword in message_lower:
                score += 2.0
        
        # Length factor - longer messages might be more important
        if len(message) > 100:
            score += 1.0
        elif len(message) > 200:
            score += 2.0
        
        # Question factor - questions might be important
        if '?' in message:
            score += 1.0
        
        # Exclamation factor - emphasis might indicate importance
        if '!' in message:
            score += 0.5
        
        # Personal information indicators
        personal_indicators = ['my name is', 'i am', 'i work', 'i live', 'my job']
        for indicator in personal_indicators:
            if indicator in message_lower:
                score += 3.0
                break
        
        return min(score, 10.0)  # Cap at 10.0
    
    def add_to_memory(self, content: str, is_user: bool, user_id: str) -> MemoryItem:
        """Add item to user's memory."""
        try:
            # Generate embedding
            embedding = self.embeddings.embed_query(content)
            
            # Create memory item
            memory_item = MemoryItem(
                id=str(uuid.uuid4()),
                user_id=user_id,
                content=content,
                is_user=is_user,
                importance=self.calculate_importance(content),
                embedding=embedding,
                memory_type=MemoryType.SHORT_TERM,
                created_at=datetime.now()
            )
            
            # Save to short-term memory
            self.db_manager.save_memory(user_id, memory_item)
            
            # Also save to long-term if important enough
            if memory_item.importance >= settings.memory.importance_threshold:
                long_term_memory = MemoryItem(
                    id=str(uuid.uuid4()),
                    user_id=user_id,
                    content=content,
                    is_user=is_user,
                    importance=memory_item.importance,
                    embedding=embedding,
                    memory_type=MemoryType.LONG_TERM,
                    created_at=datetime.now()
                )
                self.db_manager.save_memory(user_id, long_term_memory)
                logger.debug(f"Added important memory to long-term storage: {content[:50]}...")
            
            logger.debug(f"Added memory for user {user_id}: {content[:50]}...")
            return memory_item
            
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            raise
    
    def retrieve_relevant_memories(
        self, 
        query: str, 
        user_id: str, 
        top_k: int = None
    ) -> List[MemoryItem]:
        """Retrieve relevant memories using semantic similarity."""
        if top_k is None:
            top_k = settings.memory.similarity_top_k
        
        try:
            # Load memories from database
            short_term = self.db_manager.load_memories(user_id, MemoryType.SHORT_TERM, 20)
            long_term = self.db_manager.load_memories(user_id, MemoryType.LONG_TERM, 50)
            
            all_memories = short_term + long_term
            if not all_memories:
                return []
            
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            similarities = []
            
            # Calculate similarities
            for memory in all_memories:
                if memory.embedding:
                    try:
                        similarity = cosine_similarity([query_embedding], [memory.embedding])[0][0]
                        similarities.append((memory, similarity))
                    except Exception as e:
                        logger.warning(f"Error calculating similarity for memory {memory.id}: {e}")
                        continue
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            relevant_memories = [mem for mem, _ in similarities[:top_k]]
            
            logger.debug(f"Retrieved {len(relevant_memories)} relevant memories for query: {query[:50]}...")
            return relevant_memories
            
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return []
    
    def get_memory_context(self, query: str, user_id: str) -> str:
        """Get formatted memory context for the query."""
        relevant_memories = self.retrieve_relevant_memories(query, user_id)
        
        if not relevant_memories:
            return ""
        
        context_parts = ["Relevant conversation history:"]
        for memory in relevant_memories:
            role = "User" if memory.is_user else "Assistant"
            context_parts.append(f"- {role}: {memory.content}")
        
        return "\n".join(context_parts)
    
    def clear_user_memories(self, user_id: str) -> Dict[str, int]:
        """Clear all memories for a user."""
        try:
            deleted_count = self.db_manager.delete_user_memories(user_id)
            logger.info(f"Cleared all memories for user {user_id}")
            return {"deleted_memories": deleted_count}
        except Exception as e:
            logger.error(f"Error clearing memories for user {user_id}: {e}")
            raise
    
    def get_memory_stats(self, user_id: str) -> Dict[str, Any]:
        """Get memory statistics for a user."""
        try:
            stats = self.db_manager.get_user_stats(user_id)
            return {
                "short_term_count": stats.get("short_term_memories", 0),
                "long_term_count": stats.get("long_term_memories", 0),
                "total_memories": stats.get("short_term_memories", 0) + stats.get("long_term_memories", 0)
            }
        except Exception as e:
            logger.error(f"Error getting memory stats for user {user_id}: {e}")
            return {"short_term_count": 0, "long_term_count": 0, "total_memories": 0}