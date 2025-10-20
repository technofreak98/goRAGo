"""Generic document preprocessing service for cleaning text and extracting metadata."""

import re
import logging
from typing import List, Dict, Any, Optional
from llama_index.core.schema import Document as LlamaDocument
from app.models import DocumentMetadata

logger = logging.getLogger(__name__)


class DocumentPreprocessor:
    """Service for preprocessing generic documents."""
    
    def __init__(self):
        """Initialize preprocessor."""
        pass
    
    def preprocess_document(self, text: str, filename: str) -> Dict[str, Any]:
        """Preprocess document and extract basic metadata."""
        try:
            # Clean the text
            cleaned_text = self._clean_text(text)
            
            # Extract metadata
            metadata = self._extract_metadata(filename, cleaned_text)
            
            logger.info(f"Processed document: {filename}")
            logger.info(f"Text length: {len(cleaned_text)} characters")
            
            return {
                "cleaned_text": cleaned_text,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error preprocessing document: {e}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Fix encoding issues
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
        
        # Normalize whitespace
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\r', '\n', text)
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Remove excessive whitespace at start/end of lines
        lines = text.split('\n')
        lines = [line.strip() for line in lines]
        text = '\n'.join(lines)
        
        # Remove empty lines at the beginning and end
        text = text.strip()
        
        return text
    
    def _extract_metadata(self, filename: str, text: str) -> DocumentMetadata:
        """Extract document metadata."""
        from datetime import datetime
        
        # Extract title from filename or first line
        title = filename.replace('.txt', '').replace('_', ' ').title()
        
        # Try to extract title from first few lines
        lines = text.split('\n')
        for line in lines[:10]:
            line = line.strip()
            if line and len(line) > 5 and len(line) < 100:
                # Check if it looks like a title (no special patterns, reasonable length)
                if not re.match(r'^(PART|Chapter|Section)', line, re.IGNORECASE):
                    title = line
                    break
        
        return DocumentMetadata(
            title=title,
            author=None,  # Could be extracted from text if available
            language="en",
            upload_date=datetime.now()  # Set current datetime
        )
    
    def create_llama_document(self, text: str, document_id: str, metadata: Optional[Dict[str, Any]] = None) -> LlamaDocument:
        """Create a single LlamaIndex document from text."""
        try:
            # Create metadata for LlamaIndex document
            llama_metadata = {
                "document_id": document_id,
                "document_type": "generic",
                "text_length": len(text),
                **(metadata or {})
            }
            
            # Create LlamaIndex document
            llama_doc = LlamaDocument(
                text=text,
                metadata=llama_metadata,
                id_=document_id
            )
            
            logger.info(f"Created LlamaIndex document: {document_id}")
            return llama_doc
            
        except Exception as e:
            logger.error(f"Error creating LlamaIndex document: {e}")
            raise