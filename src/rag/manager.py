"""
RAG (Retrieval-Augmented Generation) manager for handling file processing and retrieval.
"""
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
import os

from ..database.manager import DatabaseManager
from ..database.models import UserFile
from ..utils.file_processor import FileProcessor
from config.settings import settings


logger = logging.getLogger(__name__)


class RAGManager:
    """Manages file processing and retrieval for RAG functionality using Chroma vector DB with per-user collections."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.embedding.model_name)
        self.file_processor = FileProcessor()
        self.persist_dir = os.path.join(os.getcwd(), "chroma_db")
        logger.info("RAG manager initialized with Chroma persistence at %s", self.persist_dir)
    
    def _get_user_collection(self, user_id: str):
        # Chroma expects embedding_function to be a callable that takes a list of strings and returns a list of vectors
        # HuggingFaceEmbeddings expects you to pass the instance, not the method
        return Chroma(
            collection_name=f"user_{user_id}",
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings  # Pass the HuggingFaceEmbeddings instance, not a method
        )

    def process_file(self, file_content: str, filename: str, user_id: str) -> UserFile:
        """Process uploaded file for RAG and store in vector DB."""
        try:
            # Validate file
            self._validate_file(filename, file_content)
            
            # Process content into chunks
            chunks = self._chunk_text(file_content)
            user_collection = self._get_user_collection(user_id)
            metadatas = [{"filename": filename, "user_id": user_id, "chunk_index": i} for i in range(len(chunks))]
            user_collection.add_texts(chunks, metadatas=metadatas)
            user_collection.persist()
            
            # Create file data object
            file_data = UserFile(
                id=str(uuid.uuid4()),
                user_id=user_id,
                filename=filename,
                content=file_content,
                chunks=chunks,
                embeddings=[],  # Not needed, stored in vector DB
                created_at=datetime.now()
            )
            
            # Save to database
            self.db_manager.save_file(user_id, file_data)
            logger.info(f"Processed file {filename} into {len(chunks)} chunks for user {user_id}")
            return file_data
        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}")
            # Only log the error, do not expose internal details to the frontend
            raise RuntimeError("File upload failed due to a server error. Please contact support if the problem persists.") from None
    
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
        """Retrieve relevant file chunks for the query from vector DB."""
        try:
            user_collection = self._get_user_collection(user_id)
            docs_and_scores = user_collection.similarity_search_with_score(query, k=top_k)
            results = []
            for doc, score in docs_and_scores:
                filename = doc.metadata.get("filename", "unknown")
                results.append((doc.page_content, filename, score))
            logger.debug(f"Retrieved {len(results)} relevant chunks for query: {query[:50]}...")
            return results
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
        """Delete all files for a user from both DB and vector DB."""
        try:
            user_collection = self._get_user_collection(user_id)
            user_collection.delete_collection()
            user_collection.persist()
            deleted_count = self.db_manager.delete_user_files(user_id)
            logger.info(f"Deleted {deleted_count} files and vector collection for user {user_id}")
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