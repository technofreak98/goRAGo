"""Unit tests for Pydantic models."""

import pytest
from datetime import datetime
from pydantic import ValidationError

from app.models import (
    DocumentMetadata, PartInfo, ChapterInfo, SectionInfo, DocumentStructure,
    Chunk, Document, UploadResponse, SearchQuery, SearchResult, SearchResponse,
    ChunkResponse, ProcessingStatus, DocumentListResponse, WeatherData,
    LocationInfo, AgentQuery, AgentResponse
)


@pytest.mark.unit
class TestDocumentMetadata:
    """Test cases for DocumentMetadata model."""

    def test_document_metadata_creation(self):
        """Test DocumentMetadata creation with all fields."""
        metadata = DocumentMetadata(
            title="Test Book",
            author="Test Author",
            isbn="1234567890",
            language="en"
        )
        
        assert metadata.title == "Test Book"
        assert metadata.author == "Test Author"
        assert metadata.isbn == "1234567890"
        assert metadata.language == "en"
        assert isinstance(metadata.upload_date, datetime)

    def test_document_metadata_defaults(self):
        """Test DocumentMetadata creation with defaults."""
        metadata = DocumentMetadata(title="Test Book")
        
        assert metadata.title == "Test Book"
        assert metadata.author is None
        assert metadata.isbn is None
        assert metadata.language == "en"
        assert isinstance(metadata.upload_date, datetime)

    def test_document_metadata_validation(self):
        """Test DocumentMetadata validation."""
        # Valid metadata
        metadata = DocumentMetadata(title="Test Book", language="fr")
        assert metadata.language == "fr"
        
        # Invalid language (should still work as it's just a string field)
        metadata = DocumentMetadata(title="Test Book", language="invalid")
        assert metadata.language == "invalid"


@pytest.mark.unit
class TestPartInfo:
    """Test cases for PartInfo model."""

    def test_part_info_creation(self):
        """Test PartInfo creation."""
        part = PartInfo(
            part_number=1,
            title="Part I: Introduction",
            start_page=1
        )
        
        assert part.part_number == 1
        assert part.title == "Part I: Introduction"
        assert part.start_page == 1

    def test_part_info_optional_fields(self):
        """Test PartInfo with optional fields."""
        part = PartInfo(part_number=1, title="Part I")
        
        assert part.part_number == 1
        assert part.title == "Part I"
        assert part.start_page is None


@pytest.mark.unit
class TestChapterInfo:
    """Test cases for ChapterInfo model."""

    def test_chapter_info_creation(self):
        """Test ChapterInfo creation."""
        chapter = ChapterInfo(
            chapter_number=1,
            title="Chapter 1: Getting Started",
            part_number=1,
            start_page=5
        )
        
        assert chapter.chapter_number == 1
        assert chapter.title == "Chapter 1: Getting Started"
        assert chapter.part_number == 1
        assert chapter.start_page == 5

    def test_chapter_info_optional_fields(self):
        """Test ChapterInfo with optional fields."""
        chapter = ChapterInfo(chapter_number=1, title="Chapter 1")
        
        assert chapter.chapter_number == 1
        assert chapter.title == "Chapter 1"
        assert chapter.part_number is None
        assert chapter.start_page is None


@pytest.mark.unit
class TestSectionInfo:
    """Test cases for SectionInfo model."""

    def test_section_info_creation(self):
        """Test SectionInfo creation."""
        section = SectionInfo(
            section_number=1,
            title="Section 1.1: Overview",
            chapter_number=1,
            part_number=1,
            start_page=10
        )
        
        assert section.section_number == 1
        assert section.title == "Section 1.1: Overview"
        assert section.chapter_number == 1
        assert section.part_number == 1
        assert section.start_page == 10

    def test_section_info_optional_fields(self):
        """Test SectionInfo with optional fields."""
        section = SectionInfo(
            section_number=1,
            title="Section 1.1",
            chapter_number=1
        )
        
        assert section.section_number == 1
        assert section.title == "Section 1.1"
        assert section.chapter_number == 1
        assert section.part_number is None
        assert section.start_page is None


@pytest.mark.unit
class TestDocumentStructure:
    """Test cases for DocumentStructure model."""

    def test_document_structure_creation(self):
        """Test DocumentStructure creation."""
        metadata = DocumentMetadata(title="Test Book")
        structure = DocumentStructure(
            metadata=metadata,
            parts=[PartInfo(part_number=1, title="Part I")],
            chapters=[ChapterInfo(chapter_number=1, title="Chapter 1")],
            sections=[SectionInfo(section_number=1, title="Section 1", chapter_number=1)]
        )
        
        assert structure.metadata.title == "Test Book"
        assert len(structure.parts) == 1
        assert len(structure.chapters) == 1
        assert len(structure.sections) == 1

    def test_document_structure_empty(self):
        """Test DocumentStructure with empty lists."""
        metadata = DocumentMetadata(title="Test Book")
        structure = DocumentStructure(metadata=metadata)
        
        assert structure.metadata.title == "Test Book"
        assert structure.parts == []
        assert structure.chapters == []
        assert structure.sections == []


@pytest.mark.unit
class TestChunk:
    """Test cases for Chunk model."""

    def test_chunk_creation(self):
        """Test Chunk creation with all fields."""
        chunk = Chunk(
            chunk_id="chunk_1",
            text="This is a test chunk about Rome and Italy.",
            token_count=10,
            parent_id="parent_1",
            child_ids=["child_1", "child_2"],
            level=1,
            document_id="doc_1",
            embedding=[0.1] * 1536,
            parent_window="Context about Italy",
            section_info=SectionInfo(section_number=1, title="Section 1", chapter_number=1),
            chapter_info=ChapterInfo(chapter_number=1, title="Chapter 1"),
            part_info=PartInfo(part_number=1, title="Part I"),
            metadata={"custom_field": "value"}
        )
        
        assert chunk.chunk_id == "chunk_1"
        assert chunk.text == "This is a test chunk about Rome and Italy."
        assert chunk.token_count == 10
        assert chunk.parent_id == "parent_1"
        assert chunk.child_ids == ["child_1", "child_2"]
        assert chunk.level == 1
        assert chunk.document_id == "doc_1"
        assert len(chunk.embedding) == 1536
        assert chunk.parent_window == "Context about Italy"
        assert chunk.section_info.section_number == 1
        assert chunk.chapter_info.chapter_number == 1
        assert chunk.part_info.part_number == 1
        assert chunk.metadata["custom_field"] == "value"

    def test_chunk_minimal(self):
        """Test Chunk creation with minimal fields."""
        chunk = Chunk(
            chunk_id="chunk_1",
            text="Test text",
            token_count=5,
            document_id="doc_1"
        )
        
        assert chunk.chunk_id == "chunk_1"
        assert chunk.text == "Test text"
        assert chunk.token_count == 5
        assert chunk.document_id == "doc_1"
        assert chunk.parent_id is None
        assert chunk.child_ids == []
        assert chunk.level == 0
        assert chunk.embedding is None
        assert chunk.parent_window is None
        assert chunk.section_info is None
        assert chunk.chapter_info is None
        assert chunk.part_info is None
        assert chunk.metadata == {}


@pytest.mark.unit
class TestDocument:
    """Test cases for Document model."""

    def test_document_creation(self):
        """Test Document creation."""
        metadata = DocumentMetadata(title="Test Book")
        structure = DocumentStructure(metadata=metadata)
        chunks = [
            Chunk(chunk_id="chunk_1", text="Text 1", token_count=5, document_id="doc_1"),
            Chunk(chunk_id="chunk_2", text="Text 2", token_count=5, document_id="doc_1")
        ]
        
        document = Document(
            document_id="doc_1",
            structure=structure,
            chunks=chunks,
            total_chunks=2,
            processing_status="completed"
        )
        
        assert document.document_id == "doc_1"
        assert document.structure.metadata.title == "Test Book"
        assert len(document.chunks) == 2
        assert document.total_chunks == 2
        assert document.processing_status == "completed"
        assert isinstance(document.created_at, datetime)
        assert isinstance(document.updated_at, datetime)

    def test_document_defaults(self):
        """Test Document creation with defaults."""
        metadata = DocumentMetadata(title="Test Book")
        structure = DocumentStructure(metadata=metadata)
        
        document = Document(
            document_id="doc_1",
            structure=structure,
            chunks=[],
            total_chunks=0
        )
        
        assert document.processing_status == "pending"
        assert isinstance(document.created_at, datetime)
        assert isinstance(document.updated_at, datetime)


@pytest.mark.unit
class TestSearchQuery:
    """Test cases for SearchQuery model."""

    def test_search_query_creation(self):
        """Test SearchQuery creation with all fields."""
        query = SearchQuery(
            query="Rome Italy travel",
            top_k=20,
            rerank=True,
            compression=True,
            filter_by_document="doc_1",
            filter_by_level=1,
            filter_by_part=1,
            filter_by_chapter=2
        )
        
        assert query.query == "Rome Italy travel"
        assert query.top_k == 20
        assert query.rerank is True
        assert query.compression is True
        assert query.filter_by_document == "doc_1"
        assert query.filter_by_level == 1
        assert query.filter_by_part == 1
        assert query.filter_by_chapter == 2

    def test_search_query_defaults(self):
        """Test SearchQuery creation with defaults."""
        query = SearchQuery(query="Rome Italy travel")
        
        assert query.query == "Rome Italy travel"
        assert query.top_k == 20
        assert query.rerank is True
        assert query.compression is True
        assert query.filter_by_document is None
        assert query.filter_by_level is None
        assert query.filter_by_part is None
        assert query.filter_by_chapter is None


@pytest.mark.unit
class TestSearchResult:
    """Test cases for SearchResult model."""

    def test_search_result_creation(self):
        """Test SearchResult creation with all fields."""
        section_info = SectionInfo(section_number=1, title="Section 1", chapter_number=1)
        chapter_info = ChapterInfo(chapter_number=1, title="Chapter 1")
        part_info = PartInfo(part_number=1, title="Part I")
        
        result = SearchResult(
            chunk_id="chunk_1",
            text="This is about Rome and Italy travel.",
            score=0.95,
            document_id="doc_1",
            level=0,
            parent_id="parent_1",
            child_ids=["child_1"],
            rank=1,
            parent_window="Context about Italy",
            section_info=section_info,
            chapter_info=chapter_info,
            part_info=part_info,
            context="Combined context",
            relevance_score=0.95,
            token_count=10
        )
        
        assert result.chunk_id == "chunk_1"
        assert result.text == "This is about Rome and Italy travel."
        assert result.score == 0.95
        assert result.document_id == "doc_1"
        assert result.level == 0
        assert result.parent_id == "parent_1"
        assert result.child_ids == ["child_1"]
        assert result.rank == 1
        assert result.parent_window == "Context about Italy"
        assert result.section_info.section_number == 1
        assert result.chapter_info.chapter_number == 1
        assert result.part_info.part_number == 1
        assert result.context == "Combined context"
        assert result.relevance_score == 0.95
        assert result.token_count == 10

    def test_search_result_minimal(self):
        """Test SearchResult creation with minimal fields."""
        result = SearchResult(
            chunk_id="chunk_1",
            text="Test text",
            score=0.8,
            document_id="doc_1",
            level=0,
            rank=1
        )
        
        assert result.chunk_id == "chunk_1"
        assert result.text == "Test text"
        assert result.score == 0.8
        assert result.document_id == "doc_1"
        assert result.level == 0
        assert result.rank == 1
        assert result.parent_id is None
        assert result.child_ids == []
        assert result.parent_window is None
        assert result.section_info is None
        assert result.chapter_info is None
        assert result.part_info is None
        assert result.context is None
        assert result.relevance_score is None
        assert result.token_count is None


@pytest.mark.unit
class TestSearchResponse:
    """Test cases for SearchResponse model."""

    def test_search_response_creation(self):
        """Test SearchResponse creation."""
        results = [
            SearchResult(
                chunk_id="chunk_1",
                text="Test text 1",
                score=0.95,
                document_id="doc_1",
                level=0,
                rank=1,
                relevance_score=0.95,
                token_count=10
            ),
            SearchResult(
                chunk_id="chunk_2",
                text="Test text 2",
                score=0.87,
                document_id="doc_1",
                level=0,
                rank=2,
                relevance_score=0.87,
                token_count=8
            )
        ]
        
        response = SearchResponse(
            query="Rome Italy travel",
            results=results,
            total_results=2,
            processing_time_ms=150.0,
            used_reranking=True,
            used_compression=False,
            combined_context="Combined context",
            total_tokens=18,
            max_relevance_score=0.95,
            min_relevance_score=0.87
        )
        
        assert response.query == "Rome Italy travel"
        assert len(response.results) == 2
        assert response.total_results == 2
        assert response.processing_time_ms == 150.0
        assert response.used_reranking is True
        assert response.used_compression is False
        assert response.combined_context == "Combined context"
        assert response.total_tokens == 18
        assert response.max_relevance_score == 0.95
        assert response.min_relevance_score == 0.87


@pytest.mark.unit
class TestWeatherData:
    """Test cases for WeatherData model."""

    def test_weather_data_creation(self):
        """Test WeatherData creation."""
        weather = WeatherData(
            city="Rome",
            country="IT",
            temperature={"current": 22.5, "min": 18.0, "max": 26.0},
            conditions={"description": "Partly cloudy", "main": "Clouds"},
            humidity=65,
            pressure=1013,
            wind={"speed": 3.2, "direction": 180},
            visibility=10.0,
            cloudiness=40,
            timestamp=1640995200
        )
        
        assert weather.city == "Rome"
        assert weather.country == "IT"
        assert weather.temperature["current"] == 22.5
        assert weather.conditions["description"] == "Partly cloudy"
        assert weather.humidity == 65
        assert weather.pressure == 1013
        assert weather.wind["speed"] == 3.2
        assert weather.visibility == 10.0
        assert weather.cloudiness == 40
        assert weather.timestamp == 1640995200


@pytest.mark.unit
class TestLocationInfo:
    """Test cases for LocationInfo model."""

    def test_location_info_creation(self):
        """Test LocationInfo creation."""
        weather = WeatherData(
            city="Rome",
            country="IT",
            temperature={"current": 22.5},
            conditions={"description": "Partly cloudy"},
            humidity=65,
            pressure=1013,
            wind={"speed": 3.2},
            visibility=10.0,
            cloudiness=40,
            timestamp=1640995200
        )
        
        location = LocationInfo(
            name="Rome",
            context=[{"type": "document", "text": "Rome is the capital of Italy"}],
            weather_data=weather
        )
        
        assert location.name == "Rome"
        assert len(location.context) == 1
        assert location.context[0]["type"] == "document"
        assert location.weather_data.city == "Rome"

    def test_location_info_minimal(self):
        """Test LocationInfo creation with minimal fields."""
        location = LocationInfo(name="Rome")
        
        assert location.name == "Rome"
        assert location.context == []
        assert location.weather_data is None


@pytest.mark.unit
class TestAgentQuery:
    """Test cases for AgentQuery model."""

    def test_agent_query_creation(self):
        """Test AgentQuery creation."""
        query = AgentQuery(query="What's the weather in Rome?")
        
        assert query.query == "What's the weather in Rome?"

    def test_agent_query_validation(self):
        """Test AgentQuery validation."""
        # Valid query
        query = AgentQuery(query="What's the weather in Rome?")
        assert query.query == "What's the weather in Rome?"
        
        # Empty query should still be valid (validation happens at API level)
        query = AgentQuery(query="")
        assert query.query == ""


@pytest.mark.unit
class TestAgentResponse:
    """Test cases for AgentResponse model."""

    def test_agent_response_creation(self):
        """Test AgentResponse creation with all fields."""
        weather = WeatherData(
            city="Rome",
            country="IT",
            temperature={"current": 22.5},
            conditions={"description": "Partly cloudy"},
            humidity=65,
            pressure=1013,
            wind={"speed": 3.2},
            visibility=10.0,
            cloudiness=40,
            timestamp=1640995200
        )
        
        location = LocationInfo(name="Rome")
        
        response = AgentResponse(
            answer="Rome is beautiful and currently has partly cloudy weather.",
            route_taken="combined",
            sources=[{"type": "weather_api", "city": "Rome"}],
            weather_data=[weather],
            locations=[location],
            processing_time_ms=150.0,
            error=None
        )
        
        assert response.answer == "Rome is beautiful and currently has partly cloudy weather."
        assert response.route_taken == "combined"
        assert len(response.sources) == 1
        assert len(response.weather_data) == 1
        assert len(response.locations) == 1
        assert response.processing_time_ms == 150.0
        assert response.error is None

    def test_agent_response_minimal(self):
        """Test AgentResponse creation with minimal fields."""
        response = AgentResponse(
            answer="This is a test answer.",
            route_taken="document",
            processing_time_ms=100.0
        )
        
        assert response.answer == "This is a test answer."
        assert response.route_taken == "document"
        assert response.processing_time_ms == 100.0
        assert response.sources == []
        assert response.weather_data == []
        assert response.locations == []
        assert response.error is None

    def test_agent_response_with_error(self):
        """Test AgentResponse creation with error."""
        response = AgentResponse(
            answer="I encountered an error.",
            route_taken="error",
            processing_time_ms=50.0,
            error="Processing failed"
        )
        
        assert response.answer == "I encountered an error."
        assert response.route_taken == "error"
        assert response.error == "Processing failed"
