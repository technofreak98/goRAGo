"""Pydantic models for the RAG document ingestion system."""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata for a document."""
    title: str
    author: Optional[str] = None
    isbn: Optional[str] = None
    language: str = "en"
    upload_date: datetime = Field(default_factory=datetime.now)


class PartInfo(BaseModel):
    """Information about a book part."""
    part_number: int
    title: str
    start_page: Optional[int] = None


class ChapterInfo(BaseModel):
    """Information about a book chapter."""
    chapter_number: int
    title: str
    part_number: Optional[int] = None
    start_page: Optional[int] = None


class SectionInfo(BaseModel):
    """Information about a book section."""
    section_number: int
    title: str
    chapter_number: int
    part_number: Optional[int] = None
    start_page: Optional[int] = None


class DocumentStructure(BaseModel):
    """Complete document structure with hierarchy."""
    metadata: DocumentMetadata
    parts: List[PartInfo] = []
    chapters: List[ChapterInfo] = []
    sections: List[SectionInfo] = []


class Chunk(BaseModel):
    """A text chunk with metadata."""
    chunk_id: str
    text: str
    token_count: int
    parent_id: Optional[str] = None  # Parent chunk ID for hierarchical structure
    child_ids: List[str] = []  # Child chunk IDs for hierarchical structure
    level: int = 0  # Hierarchy level (0=leaf, 1=parent, 2=grandparent, etc.)
    document_id: str  # ID of the source document
    embedding: Optional[List[float]] = None
    parent_window: Optional[str] = None  # Parent chunk text for context
    section_info: Optional[SectionInfo] = None
    chapter_info: Optional[ChapterInfo] = None
    part_info: Optional[PartInfo] = None
    metadata: Dict[str, Any] = {}  # Additional metadata for LlamaIndex compatibility


class Document(BaseModel):
    """A complete document."""
    document_id: str
    structure: DocumentStructure
    chunks: List[Chunk]
    total_chunks: int
    processing_status: str = "pending"  # pending, processing, completed, failed
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class UploadResponse(BaseModel):
    """Response for document upload."""
    document_id: str
    status: str
    message: str
    total_chunks: int


class SearchQuery(BaseModel):
    """Search query parameters."""
    query: str
    top_k: int = 20
    rerank: bool = True
    compression: bool = True
    filter_by_document: Optional[str] = None
    filter_by_level: Optional[int] = None
    filter_by_part: Optional[int] = None
    filter_by_chapter: Optional[int] = None


class SearchResult(BaseModel):
    """Individual search result."""
    chunk_id: str
    text: str
    score: float
    document_id: str
    level: int
    parent_id: Optional[str] = None
    child_ids: List[str] = []
    rank: int
    parent_window: Optional[str] = None
    section_info: Optional[SectionInfo] = None
    chapter_info: Optional[ChapterInfo] = None
    part_info: Optional[PartInfo] = None
    # RAG-specific fields
    context: Optional[str] = None  # Combined context (text + parent_window)
    relevance_score: Optional[float] = None  # Normalized relevance score
    token_count: Optional[int] = None  # Token count for the chunk


class SearchResponse(BaseModel):
    """Response for search queries."""
    query: str
    results: List[SearchResult]
    total_results: int
    processing_time_ms: float
    used_reranking: bool
    used_compression: bool
    # RAG-specific fields
    combined_context: Optional[str] = None  # All results combined for RAG
    total_tokens: Optional[int] = None  # Total token count
    max_relevance_score: Optional[float] = None  # Highest relevance score
    min_relevance_score: Optional[float] = None  # Lowest relevance score


class ChunkResponse(BaseModel):
    """Response for chunk retrieval."""
    chunk_id: str
    text: str
    document_id: str
    level: int
    parent_id: Optional[str] = None
    child_ids: List[str] = []


class ProcessingStatus(BaseModel):
    """Processing status for ingestion jobs."""
    document_id: str
    status: str  # pending, processing, completed, failed
    progress: int = 0  # 0-100
    message: str = ""
    total_chunks: int = 0
    processed_chunks: int = 0
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class DocumentListResponse(BaseModel):
    """Response for listing documents."""
    documents: List[Dict[str, Any]]
    total_documents: int


# Agent-related models
class WeatherData(BaseModel):
    """Weather data model."""
    city: str
    country: str
    temperature: Dict[str, float]
    conditions: Dict[str, str]
    humidity: int
    pressure: int
    wind: Dict[str, float]
    visibility: float
    cloudiness: int
    timestamp: int


class LocationInfo(BaseModel):
    """Location information model."""
    name: str
    context: List[Dict[str, Any]] = []
    weather_data: Optional[WeatherData] = None


class AgentQuery(BaseModel):
    """Query model for the agent."""
    query: str


class AgentResponse(BaseModel):
    """Response model for the agent."""
    answer: str
    route_taken: str  # "document", "weather", "combined", "guardrails"
    sources: List[Dict[str, Any]] = []
    weather_data: List[WeatherData] = []
    locations: List[LocationInfo] = []
    processing_time_ms: float
    error: Optional[str] = None
