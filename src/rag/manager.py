"""
RAG (Retrieval-Augmented Generation) manager for handling file processing and retrieval.
"""
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

from ..database.manager import DatabaseManager
from ..database.models import UserFile
from ..utils.file_processor import FileProcessor
from config.settings import settings


logger = logging.getLogger(__name__)


class RAGManager:
    """Manages file processing and retrieval for RAG functionality."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.embedding.model_name)
        self.file_processor = FileProcessor()
        logger.info("RAG manager initialized")
    
    def process_file(self, file_content: str, filename: str, user_id: str) -> UserFile:
        """Process uploaded file for RAG."""
        try:
            # Validate file
            self._validate_file(filename, file_content)
            
            # Process content into chunks
            chunks = self._chunk_text(file_content)
            
            # Generate embeddings for all chunks
            embeddings = []
            for chunk in chunks:
                try:
                    embedding = self.embeddings.embed_query(chunk)
                    embeddings.append(embedding)
                except Exception as e:
                    logger.warning(f"Error generating embedding for chunk: {e}")
                    embeddings.append([])  # Empty embedding as fallback
            
            # Create file data object
            file_data = UserFile(
                id=str(uuid.uuid4()),
                user_id=user_id,
                filename=filename,
                content=file_content,
                chunks=chunks,
                embeddings=embeddings,
                created_at=datetime.now()
            )
            
            # Save to database
            self.db_manager.save_file(user_id, file_data)
            
            logger.info(f"Processed file {filename} into {len(chunks)} chunks for user {user_id}")
            return file_data
            
        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}")
            raise
    
    def _validate_file(self, filename: str, content: str) -> None:
        """Validate file constraints."""
        # Check file extension
        if not any(filename.lower().endswith(ext) for ext in settings.file.allowed_extensions):
            raise ValueError(f"File type not supported. Allowed types: {settings.file.allowed_extensions}")
        
        # Check file size (approximate, based on content length)
        if len(content.encode('utf-8')) > settings.file.max_file_size:
            raise ValueError(f"File too large. Maximum size: {settings.file.max_file_size} bytes")
        
        # Check if content is not empty
        if not content.strip():
            raise ValueError("File content is empty")
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap."""
        chunk_size = settings.file.chunk_size
        overlap = settings.file.chunk_overlap
        
        # Simple chunking strategy
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundaries if possible
            if end < len(text) and '.' in chunk:
                last_period = chunk.rfind('.')
                if last_period > chunk_size // 2:  # Only if period is in latter half
                    chunk = chunk[:last_period + 1]
                    end = start + len(chunk)
            
            chunks.append(chunk.strip())
            
            # Move start position with overlap
            start = end - overlap
            if start >= len(text):
                break
        
        return [chunk for chunk in chunks if chunk]  # Remove empty chunks
    
    def retrieve_relevant_chunks(
        self, 
        query: str, 
        user_id: str, 
        top_k: int = 3
    ) -> List[Tuple[str, str, float]]:
        """Retrieve relevant file chunks for the query."""
        try:
            # Load user files
            files = self.db_manager.load_files(user_id)
            if not files:
                return []
            
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            all_chunks = []
            
            # Calculate similarities for all chunks
            for file in files:
                for i, chunk in enumerate(file.chunks):
                    if i < len(file.embeddings) and file.embeddings[i]:
                        try:
                            similarity = cosine_similarity([query_embedding], [file.embeddings[i]])[0][0]
                            all_chunks.append((chunk, file.filename, similarity))
                        except Exception as e:
                            logger.warning(f"Error calculating similarity for chunk in {file.filename}: {e}")
                            continue
            
            # Sort by similarity and return top chunks
            all_chunks.sort(key=lambda x: x[2], reverse=True)
            
            logger.debug(f"Retrieved {min(top_k, len(all_chunks))} relevant chunks for query: {query[:50]}...")
            return all_chunks[:top_k]
            
        except Exception as e:
            logger.error(f"Error retrieving relevant chunks: {e}")
            return []
    
    def get_file_context(self, query: str, user_id: str, top_k: int = 3) -> str:
        """Get formatted file context for the query."""
        relevant_chunks = self.retrieve_relevant_chunks(query, user_id, top_k)
        
        if not relevant_chunks:
            return ""
        
        context_parts = ["Relevant information from uploaded files:"]
        for chunk, filename, similarity in relevant_chunks:
            context_parts.append(f"From {filename} (relevance: {similarity:.2f}):")
            context_parts.append(f"  {chunk[:500]}{'...' if len(chunk) > 500 else ''}")
            context_parts.append("")  # Empty line for separation
        
        return "\n".join(context_parts)
    
    def get_user_files_info(self, user_id: str) -> List[Dict[str, Any]]:
        """Get information about user's uploaded files."""
        try:
            files = self.db_manager.load_files(user_id)
            files_info = []
            
            for file in files:
                info = {
                    'id': file.id,
                    'filename': file.filename,
                    'chunks_count': len(file.chunks),
                    'content_length': len(file.content),
                    'created_at': file.created_at.isoformat()
                }
                files_info.append(info)
            
            return files_info
            
        except Exception as e:
            logger.error(f"Error getting files info for user {user_id}: {e}")
            return []
    
    def delete_user_files(self, user_id: str) -> int:
        """Delete all files for a user."""
        try:
            deleted_count = self.db_manager.delete_user_files(user_id)
            logger.info(f"Deleted {deleted_count} files for user {user_id}")
            return deleted_count
        except Exception as e:
            logger.error(f"Error deleting files for user {user_id}: {e}")
            raise
    
    def search_in_files(self, query: str, user_id: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for text within user's files."""
        try:
            files = self.db_manager.load_files(user_id)
            results = []
            
            query_lower = query.lower()
            
            for file in files:
                # Simple text search within content
                if query_lower in file.content.lower():
                    # Find relevant chunks
                    for i, chunk in enumerate(file.chunks):
                        if query_lower in chunk.lower():
                            results.append({
                                'filename': file.filename,
                                'chunk_index': i,
                                'chunk': chunk,
                                'file_id': file.id
                            })
                            
                            if len(results) >= max_results:
                                break
                
                if len(results) >= max_results:
                    break
            
            logger.debug(f"Found {len(results)} search results for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error searching in files: {e}")
            return []