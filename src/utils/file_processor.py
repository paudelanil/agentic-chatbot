"""
File processing utilities for different file types.
"""
import logging
from io import BytesIO
from typing import Optional

# Optional imports for different file types
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False


logger = logging.getLogger(__name__)


class FileProcessor:
    """Handles processing of different file types."""
    
    def __init__(self):
        self.supported_types = {
            '.txt': self._process_text,
            '.md': self._process_text,
        }
        
        # Add PDF support if available
        if PDF_SUPPORT:
            self.supported_types['.pdf'] = self._process_pdf
        else:
            logger.warning("PyPDF2 not installed. PDF support disabled.")
        
        # Add DOCX support if available
     
    def get_supported_extensions(self) -> list:
        """Get list of supported file extensions."""
        return list(self.supported_types.keys())
    
    def is_supported(self, filename: str) -> bool:
        """Check if file type is supported."""
        extension = self._get_file_extension(filename)
        return extension in self.supported_types
    
    def process_file(self, file_content: bytes, filename: str) -> str:
        """Process file content based on file type."""
        extension = self._get_file_extension(filename)
        
        if extension not in self.supported_types:
            raise ValueError(f"Unsupported file type: {extension}")
        
        try:
            processor = self.supported_types[extension]
            return processor(file_content, filename)
        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}")
            raise ValueError(f"Failed to process file {filename}: {str(e)}")
    
    def _get_file_extension(self, filename: str) -> str:
        """Extract file extension from filename."""
        return '.' + filename.lower().split('.')[-1] if '.' in filename else ''
    
    def _process_text(self, file_content: bytes, filename: str) -> str:
        """Process plain text files."""
        try:
            # Try different encodings
            encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    content = file_content.decode(encoding)
                    logger.debug(f"Successfully decoded {filename} with {encoding}")
                    return content
                except UnicodeDecodeError:
                    continue
            
            raise ValueError("Could not decode file with any supported encoding")
            
        except Exception as e:
            logger.error(f"Error processing text file {filename}: {e}")
            raise
    
    def _process_pdf(self, file_content: bytes, filename: str) -> str:
        """Process PDF files."""
        if not PDF_SUPPORT:
            raise ValueError("PDF support not available. Install PyPDF2.")
        
        try:
            pdf_file = BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text_content = []
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():  # Only add non-empty pages
                        text_content.append(f"--- Page {page_num + 1} ---\n{page_text}")
                except Exception as e:
                    logger.warning(f"Error extracting text from page {page_num + 1} of {filename}: {e}")
                    continue
            
            if not text_content:
                raise ValueError("No text content could be extracted from PDF")
            
            result = "\n\n".join(text_content)
            logger.debug(f"Extracted text from {len(pdf_reader.pages)} pages in {filename}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing PDF file {filename}: {e}")
            raise
    
    # def _process_docx(self, file_content: bytes, filename: str) -> str:
    #     """Process DOCX files."""
    #     if not DOCX_SUPPORT:
    #         raise ValueError("DOCX support not available. Install python-docx.")
        
    #     try:
    #         doc_file = BytesIO(file_content)
    #         doc = docx.Document(doc_file)
            
    #         text_content = []
            
    #         # Extract paragraphs
    #         for paragraph in doc.paragraphs:
    #             if paragraph.text.strip():
    #                 text_content.append(paragraph.text)
            
    #         # Extract text from tables
    #         for table in doc.tables:
    #             for row in table.rows:
    #                 row_text = []
    #                 for cell in row.cells:
    #                     if cell.text.strip():
    #                         row_text.append(cell.text.strip())
    #                 if row_text:
    #                     text_content.append(" | ".join(row_text))
            
    #         if not text_content:
    #             raise ValueError("No text content could be extracted from DOCX")
            
    #         result = "\n".join(text_content)
    #         logger.debug(f"Extracted text from DOCX file {filename}")
    #         return result
            
    #     except Exception as e:
    #         logger.error(f"Error processing DOCX file {filename}: {e}")
    #         raise
    
    # def _process_doc(self, file_content: bytes, filename: str) -> str:
    #     """Process DOC files."""
    #     if not DOC_SUPPORT:
    #         raise ValueError("DOC support not available. Install python-docx2txt.")
        
    #     try:
    #         # docx2txt expects a file path, so we need to save temporarily
    #         import tempfile
    #         import os
            
    #         with tempfile.NamedTemporaryFile(delete=False, suffix='.doc') as temp_file:
    #             temp_file.write(file_content)
    #             temp_file_path = temp_file.name
            
    #         try:
    #             text_content = docx2txt.process(temp_file_path)
                
    #             if not text_content or not text_content.strip():
    #                 raise ValueError("No text content could be extracted from DOC")
                
    #             logger.debug(f"Extracted text from DOC file {filename}")
    #             return text_content
                
    #         finally:
    #             # Clean up temporary file
    #             if os.path.exists(temp_file_path):
    #                 os.unlink(temp_file_path)
            
    #     except Exception as e:
    #         logger.error(f"Error processing DOC file {filename}: {e}")
    #         raise
    
    def get_file_info(self, filename: str) -> dict:
        """Get information about file type and processing capabilities."""
        extension = self._get_file_extension(filename)
        
        info = {
            'filename': filename,
            'extension': extension,
            'supported': extension in self.supported_types,
            'processor': self.supported_types.get(extension, None).__name__ if extension in self.supported_types else None
        }

        return info