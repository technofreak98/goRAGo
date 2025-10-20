"""Ingestion router for document upload and processing."""

import uuid
import asyncio
import logging
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from app.models import UploadResponse, ProcessingStatus, DocumentListResponse
from app.services.preprocessor import DocumentPreprocessor
from app.services.chunker import HierarchicalChunker
from app.services.embedding_service import EmbeddingService
from app.services.elasticsearch_service import ElasticsearchService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ingest", tags=["ingestion"])

# Global processing status tracking
processing_jobs = {}


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload and process a document file."""
    try:
        # Validate file type
        if not file.filename.endswith('.txt'):
            raise HTTPException(status_code=400, detail="Only .txt files are supported")
        
        # Read file content
        content = await file.read()
        text_content = content.decode('utf-8')
        
        # Generate document ID
        document_id = str(uuid.uuid4())
        
        # Initialize processing status
        processing_jobs[document_id] = ProcessingStatus(
            document_id=document_id,
            status="pending",
            message="Document uploaded, starting processing..."
        )
        
        # Start background processing
        background_tasks.add_task(
            process_document,
            document_id,
            text_content,
            file.filename
        )
        
        return UploadResponse(
            document_id=document_id,
            status="processing",
            message="Document uploaded successfully, processing started",
            total_chunks=0  # Will be updated during processing
        )
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/status/{document_id}", response_model=ProcessingStatus)
async def get_processing_status(document_id: str):
    """Get processing status for a document."""
    if document_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return processing_jobs[document_id]


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """List all indexed documents."""
    try:
        es_service = ElasticsearchService()
        documents = await es_service.list_documents()
        
        return DocumentListResponse(
            documents=documents,
            total_documents=len(documents)
        )
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


async def process_document(document_id: str, text_content: str, filename: str):
    """Process document in background."""
    try:
        # Update status to processing
        processing_jobs[document_id].status = "processing"
        processing_jobs[document_id].message = "Parsing document structure..."
        
        # Initialize services
        preprocessor = DocumentPreprocessor()
        chunker = HierarchicalChunker()
        embedding_service = EmbeddingService()
        es_service = ElasticsearchService()
        
        # Create indices if they don't exist
        await es_service.create_indices()
        
        # Step 1: Preprocess document
        processing_jobs[document_id].message = "Preprocessing document..."
        preprocessed_data = preprocessor.preprocess_document(text_content, filename)
        
        # Step 2: Chunk document with hierarchical structure
        processing_jobs[document_id].message = "Creating hierarchical text chunks..."
        chunks = chunker.chunk_document(
            preprocessed_data["cleaned_text"],
            document_id,
            preprocessed_data["metadata"].model_dump()
        )
        
        processing_jobs[document_id].total_chunks = len(chunks)
        processing_jobs[document_id].message = f"Generated {len(chunks)} hierarchical chunks, generating embeddings..."
        
        # Step 3: Generate embeddings
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = await embedding_service.generate_embeddings(chunk_texts)
        
        # Update chunks with embeddings
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        processing_jobs[document_id].processed_chunks = len(chunks)
        processing_jobs[document_id].message = "Indexing document..."
        
        # Step 4: Create document object
        from app.models import Document, DocumentStructure
        document_structure = DocumentStructure(
            metadata=preprocessed_data["metadata"],
            parts=[],
            chapters=[],
            sections=[]
        )
        
        document = Document(
            document_id=document_id,
            structure=document_structure,
            chunks=chunks,
            total_chunks=len(chunks),
            processing_status="processing"
        )
        
        # Step 5: Index in Elasticsearch
        # Index child chunks
        chunks_success = await es_service.index_child_chunks(document)
        if not chunks_success:
            raise Exception("Failed to index child chunks")
        
        # Update status to completed
        processing_jobs[document_id].status = "completed"
        processing_jobs[document_id].progress = 100
        processing_jobs[document_id].message = f"Document processed successfully. {len(chunks)} chunks indexed."
        processing_jobs[document_id].processed_chunks = len(chunks)
        
        logger.info(f"Document {document_id} processed successfully with {len(chunks)} chunks")
        
    except Exception as e:
        # Update status to failed
        processing_jobs[document_id].status = "failed"
        processing_jobs[document_id].message = f"Processing failed: {str(e)}"
        logger.error(f"Error processing document {document_id}: {e}")


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and all its chunks."""
    try:
        es_service = ElasticsearchService()
        success = await es_service.delete_document(document_id)
        
        if success:
            # Remove from processing jobs if exists
            if document_id in processing_jobs:
                del processing_jobs[document_id]
            
            return JSONResponse(
                content={"message": f"Document {document_id} deleted successfully"}
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to delete document")
            
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")
