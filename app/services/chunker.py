"""Generic hierarchical chunking service using LangChain's RecursiveCharacterTextSplitter."""

import logging
import uuid
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import settings
from app.models import Chunk

logger = logging.getLogger(__name__)


class HierarchicalChunker:
    """Service for generic hierarchical chunking using LangChain's RecursiveCharacterTextSplitter."""
    
    def __init__(self):
        """Initialize hierarchical chunker with LangChain components."""
        # Create text splitter for different hierarchy levels
        self.text_splitters = {
            0: RecursiveCharacterTextSplitter(  # Leaf level
                chunk_size=settings.child_chunk_size,
                chunk_overlap=settings.chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            ),
            1: RecursiveCharacterTextSplitter(  # Parent level
                chunk_size=settings.child_chunk_size * 2,
                chunk_overlap=settings.chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            ),
            2: RecursiveCharacterTextSplitter(  # Grandparent level
                chunk_size=settings.child_chunk_size * 4,
                chunk_overlap=settings.chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )
        }
        
    def chunk_document(self, text: str, document_id: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Chunk document text with hierarchical structure."""
        chunks = []
        
        try:
            # Create hierarchical chunks using different text splitters
            all_chunks = []
            
            # Level 0: Smallest chunks (leaf level)
            level_0_chunks = self.text_splitters[0].split_text(text)
            for i, chunk_text in enumerate(level_0_chunks):
                chunk = self._create_chunk(
                    text=chunk_text,
                    document_id=document_id,
                    level=0,
                    chunk_index=i,
                    metadata=metadata or {}
                )
                all_chunks.append(chunk)
            
            # Level 1: Medium chunks (parent level)
            level_1_chunks = self.text_splitters[1].split_text(text)
            for i, chunk_text in enumerate(level_1_chunks):
                chunk = self._create_chunk(
                    text=chunk_text,
                    document_id=document_id,
                    level=1,
                    chunk_index=i,
                    metadata=metadata or {}
                )
                all_chunks.append(chunk)
            
            # Level 2: Largest chunks (grandparent level)
            level_2_chunks = self.text_splitters[2].split_text(text)
            for i, chunk_text in enumerate(level_2_chunks):
                chunk = self._create_chunk(
                    text=chunk_text,
                    document_id=document_id,
                    level=2,
                    chunk_index=i,
                    metadata=metadata or {}
                )
                all_chunks.append(chunk)
            
            # Build hierarchical relationships
            chunks = self._build_hierarchical_relationships(all_chunks)
            
            logger.info(f"Created {len(chunks)} hierarchical chunks from document {document_id}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking document {document_id}: {e}")
            raise
    
    def _create_chunk(self, text: str, document_id: str, level: int, chunk_index: int, metadata: Dict[str, Any]) -> Chunk:
        """Create a Chunk object from text."""
        # Calculate token count using tiktoken
        import tiktoken
        tokenizer = tiktoken.get_encoding("cl100k_base")
        token_count = len(tokenizer.encode(text))
        
        # Generate chunk ID
        chunk_id = self._generate_chunk_id(document_id, level, chunk_index)
        
        # Extract structure information from metadata
        section_info = None
        chapter_info = None
        part_info = None
        
        if "section_info" in metadata:
            section_info = metadata["section_info"]
        if "chapter_info" in metadata:
            chapter_info = metadata["chapter_info"]
        if "part_info" in metadata:
            part_info = metadata["part_info"]
        
        # Create chunk
        chunk = Chunk(
            chunk_id=chunk_id,
            text=text,
            token_count=token_count,
            level=level,
            document_id=document_id,
            parent_window=None,  # Will be set later in _build_hierarchical_relationships
            section_info=section_info,
            chapter_info=chapter_info,
            part_info=part_info,
            metadata={
                "chunk_index": chunk_index,
                "level": level,
                **metadata
            }
        )
        
        return chunk
    
    def _build_hierarchical_relationships(self, chunks: List[Chunk]) -> List[Chunk]:
        """Build parent-child relationships between chunks of different levels."""
        # Group chunks by level
        chunks_by_level = {}
        for chunk in chunks:
            if chunk.level not in chunks_by_level:
                chunks_by_level[chunk.level] = []
            chunks_by_level[chunk.level].append(chunk)
        
        # Build relationships: smaller chunks are children of larger chunks that contain them
        # We need to check from highest level down to lowest level
        for level in sorted(chunks_by_level.keys(), reverse=True):
            if level == 0:  # Skip leaf level
                continue
                
            current_level_chunks = chunks_by_level[level]
            child_level_chunks = chunks_by_level.get(level - 1, [])
            
            for parent_chunk in current_level_chunks:
                for child_chunk in child_level_chunks:
                    # Check if parent chunk contains the child chunk text
                    # Use a more lenient check to account for whitespace differences
                    parent_text_clean = parent_chunk.text.replace('\n', ' ').replace('  ', ' ').strip()
                    child_text_clean = child_chunk.text.replace('\n', ' ').replace('  ', ' ').strip()
                    
                    if child_text_clean in parent_text_clean and len(child_text_clean) > 50:  # Avoid very short matches
                        child_chunk.parent_id = parent_chunk.chunk_id
                        parent_chunk.child_ids.append(child_chunk.chunk_id)
                        # Set parent window for context
                        child_chunk.parent_window = parent_chunk.text
        
        return chunks
    
    def _determine_hierarchy_level(self, text_length: int) -> int:
        """Determine hierarchy level based on text length."""
        if text_length <= settings.child_chunk_size:
            return 0  # Leaf level
        elif text_length <= settings.child_chunk_size * 2:
            return 1  # Parent level
        else:
            return 2  # Grandparent level
    
    
    def _generate_chunk_id(self, document_id: str, level: int, chunk_index: int) -> str:
        """Generate unique chunk ID."""
        import hashlib
        import time
        
        # Create a hash based on document ID, level, and chunk index
        content = f"{document_id}_{level}_{chunk_index}_{time.time()}"
        hash_id = hashlib.md5(content.encode()).hexdigest()[:8]
        
        return f"chunk_{document_id}_{level}_{chunk_index}_{hash_id}"
    
    def get_chunk_statistics(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """Get statistics about hierarchical chunks."""
        if not chunks:
            return {}
        
        token_counts = [chunk.token_count for chunk in chunks]
        level_counts = {}
        
        for chunk in chunks:
            level_counts[chunk.level] = level_counts.get(chunk.level, 0) + 1
        
        return {
            "total_chunks": len(chunks),
            "avg_tokens_per_chunk": sum(token_counts) / len(token_counts),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "total_tokens": sum(token_counts),
            "chunks_by_level": level_counts,
            "hierarchical_depth": max(level_counts.keys()) if level_counts else 0,
            "chunks_by_document": self._group_chunks_by_document(chunks)
        }
    
    def _group_chunks_by_document(self, chunks: List[Chunk]) -> Dict[str, int]:
        """Group chunks by document ID."""
        document_counts = {}
        for chunk in chunks:
            doc_id = chunk.document_id
            document_counts[doc_id] = document_counts.get(doc_id, 0) + 1
        return document_counts