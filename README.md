# GoRAGo: Advanced RAG-ES System with LangGraph

A production-ready Retrieval-Augmented Generation (RAG) system built with FastAPI, Elasticsearch, and LangGraph, featuring intelligent query routing, hierarchical document processing, and multi-modal response generation.

## 🚀 System Overview

GoRAGo is an advanced RAG system that combines document retrieval with weather information to provide comprehensive, context-aware responses. The system uses LangGraph for intelligent query routing and processing, ensuring optimal response quality based on query intent.

### Key Features

- **🧠 Intelligent Query Routing**: LangGraph-based routing system that classifies queries and routes them to appropriate processing nodes
- **📚 Hierarchical Document Processing**: Multi-level chunking with parent-child relationships for better context preservation
- **🔍 Hybrid Search**: Combines dense vector search (kNN) with BM25 keyword matching using Elasticsearch's RRF
- **🌤️ Multi-Modal Responses**: Integrates document content with real-time weather data
- **⚡ Advanced Reranking**: Cross-encoder reranking and contextual compression for improved relevance
- **🛡️ Production-Ready**: Comprehensive monitoring, logging, and error handling
- **🎯 Smart Guardrails**: Out-of-scope query detection with polite responses

## 🏗️ Architecture

### System Flow

```
User Query → Router Node → [Document Retrieval | Weather Retrieval | Guardrail] → Generation Node → Response
```

### Complete Pipeline Architecture

![GoRAGo Pipeline Architecture](Architecture%20Diagram.png)

*Complete flow from ingestion to response showing all phases: Ingestion, Query Processing, Retrieval, and Generation phases with their respective components and data flows.*

### Proposed Infrastructure

![Proposed Infrastructure](Proposed%20Infrastructure.png)

*Production-ready infrastructure design showing deployment architecture, security layers, and scalability considerations.*

### Core Components

1. **Router Node**: Classifies queries into weather-only, document-only, combined, or out-of-scope
2. **Document Retrieval Node**: Performs hybrid search with reranking and compression
3. **Weather Retrieval Node**: Fetches real-time weather data using location extraction
4. **Generation Node**: Combines all context and generates final responses
5. **Guardrail Node**: Handles out-of-scope queries with appropriate responses

## 🛠️ Implementation Details

### Algorithms Used

#### 1. Hierarchical Chunking Algorithm
- **Multi-level Chunking**: Creates 3 levels of chunks (small, medium, large) using LangChain's RecursiveCharacterTextSplitter
- **Parent-Child Relationships**: Builds hierarchical relationships between chunks for better context
- **Token-based Sizing**: Configurable chunk sizes with overlap for context preservation

```python
# Chunking Configuration
child_chunk_size: 400 tokens
chunk_overlap: 60 tokens (15% overlap)
parent_window_size: 1500 tokens
```

#### 2. Hybrid Search Algorithm
- **Dense Vector Search**: Uses OpenAI's text-embedding-3-small (1536 dimensions)
- **BM25 Keyword Search**: Elasticsearch's built-in BM25 implementation
- **Weighted Scoring**: Configurable weights (dense: 0.6, BM25: 0.4)

#### 3. Reranking Algorithm
- **Query-Document Overlap**: Calculates term overlap between query and document text
- **Combined Scoring**: `(original_score * 0.7) + (rerank_score * 0.3)`
- **Relevance Normalization**: Normalizes scores to 0-1 range

#### 4. Query Classification
- **LLM-based Classification**: Uses GPT-4 for intent detection
- **Multi-class Categories**: weather_only, document_only, combined, out_of_scope
- **Confidence Scoring**: Provides confidence levels for routing decisions

### Data Models

#### Chunk Structure
```python
class Chunk(BaseModel):
    chunk_id: str
    text: str
    token_count: int
    level: int  # 0=leaf, 1=parent, 2=grandparent
    parent_id: Optional[str]
    child_ids: List[str]
    document_id: str
    embedding: Optional[List[float]]
    parent_window: Optional[str]
    section_info: Optional[SectionInfo]
    chapter_info: Optional[ChapterInfo]
    part_info: Optional[PartInfo]
```

#### Search Query
```python
class SearchQuery(BaseModel):
    query: str
    top_k: int = 20
    rerank: bool = True
    compression: bool = True
    filter_by_document: Optional[str] = None
    filter_by_level: Optional[int] = None
    filter_by_part: Optional[int] = None
    filter_by_chapter: Optional[int] = None
```

## 📋 Prerequisites

- **Python**: 3.8+ (recommended: 3.11)
- **Docker**: 20.10+ with Docker Compose
- **Memory**: Minimum 8GB RAM (16GB recommended)
- **Storage**: 10GB free space
- **API Keys**: OpenAI API key, OpenWeatherMap API key (optional)

## 🚀 Quick Start

### 1. Clone and Setup

   ```bash
# Clone the repository
git clone <repository-url>
   cd goRAGo

# Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
   pip install -r requirements.txt
   ```

### 2. Environment Configuration

   ```bash
# Copy environment template
   cp env.example .env

# Edit .env file with your API keys
nano .env
```

**Required Environment Variables:**
```env
# OpenAI Configuration (Required)
OPENAI_API_KEY=your_openai_api_key_here
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.7

# Elasticsearch Configuration
ELASTICSEARCH_URL=http://localhost:9200

# Weather API (Optional)
OPENWEATHER_API_KEY=your_openweather_api_key_here

# Chunking Configuration
CHILD_CHUNK_SIZE=400
CHUNK_OVERLAP=60
PARENT_WINDOW_SIZE=1500

# Retrieval Configuration
INITIAL_TOP_K=20
FINAL_TOP_K=10
DENSE_WEIGHT=0.6
BM25_WEIGHT=0.4
```

### 3. Start Services

   ```bash
# Start Elasticsearch and Kibana
docker-compose up -d

# Verify Elasticsearch is running
curl http://localhost:9200/_cluster/health

# Open Chat UI
open http://localhost:8501

# (or) Start the API server
   python -m app.main
   ```

### 4. Verify Installation

```bash
# Check API health
curl http://localhost:8000/health

# Access API documentation
open http://localhost:8000/docs
```

## 🎯 Usage Examples

### 1. Document Upload and Processing

```bash
# Upload a document
curl -X POST "http://localhost:8000/api/ingest/upload" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@sample_book.txt"

# Check processing status
curl -X GET "http://localhost:8000/api/ingest/status/{document_id}"

# List all documents
curl -X GET "http://localhost:8000/api/ingest/documents"
```

### 2. Query Processing

```bash
# Document-only query
curl -X POST "http://localhost:8000/api/agent/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is machine learning?"}'

# Weather-only query
curl -X POST "http://localhost:8000/api/agent/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the weather in Paris?"}'

# Combined query
curl -X POST "http://localhost:8000/api/agent/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the weather like in the places mentioned in the book?"}'
```

### 3. Direct Search (Advanced)

```bash
# Hybrid search with reranking
curl -X POST "http://localhost:8000/api/search/search" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "artificial intelligence applications",
       "top_k": 10,
       "rerank": true,
       "compression": true,
       "filter_by_level": 0
     }'

# Retrieve specific chunk
curl -X GET "http://localhost:8000/api/search/chunk/{chunk_id}"
```

## 🖥️ Chat Interface

### Start Chat UI

```bash
# Using the provided script
python start_chat.py

# Or directly with Streamlit
streamlit run chat_ui.py --server.port 8501
```

**Access the chat interface at:** `http://localhost:8501`

### Chat Features

- **Session Management**: Persistent conversation history
- **Multi-modal Queries**: Handles both document and weather queries
- **Real-time Responses**: Streaming responses with typing indicators
- **Source Attribution**: Shows sources for each response
- **Query History**: Browse and replay previous conversations

## 🔧 Configuration

### Chunking Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CHILD_CHUNK_SIZE` | 400 | Tokens per smallest chunk |
| `CHUNK_OVERLAP` | 60 | Overlap tokens between chunks |
| `PARENT_WINDOW_SIZE` | 1500 | Parent context window size |
| `MAX_TOKENS_COMPRESSION` | 1500 | Maximum tokens for compression |

### Retrieval Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `INITIAL_TOP_K` | 20 | Initial search results |
| `FINAL_TOP_K` | 10 | Final results after reranking |
| `DENSE_WEIGHT` | 0.6 | Weight for dense vector search |
| `BM25_WEIGHT` | 0.4 | Weight for BM25 keyword search |

### LLM Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LLM_MODEL` | gpt-4o-mini | OpenAI model for generation |
| `LLM_TEMPERATURE` | 0.7 | Response creativity (0-1) |
| `EMBEDDING_MODEL` | text-embedding-3-small | Embedding model |

## 📊 Monitoring and Observability

### Built-in Metrics

- **Cost Tracking**: API call costs and token usage
- **Latency Tracking**: Response times for each component
- **Success Rates**: Component-level success metrics
- **Resource Usage**: Memory and CPU utilization

### Logging

```bash
# View application logs
tail -f app.log

# View Elasticsearch logs
docker-compose logs elasticsearch

# View Kibana (optional)
docker-compose logs kibana
```

### Health Checks

```bash
# API health
curl http://localhost:8000/health

# Elasticsearch health
curl http://localhost:9200/_cluster/health

# Agent health
curl http://localhost:8000/api/agent/health
```

## 🧪 Testing

### Run Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=app --cov-report=html

# Run specific test categories
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests
```

### Test Categories

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing
- **Performance Tests**: Load and stress testing
- **API Tests**: REST endpoint testing

## 🚀 Production Deployment

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose -f docker-compose.prod.yml up -d

# Scale services
docker-compose up -d --scale api=3
```

### Environment Setup

1. **Production Environment Variables**:
   ```env
   DEBUG=False
   ELASTICSEARCH_URL=https://your-elasticsearch-cluster.com
   ELASTICSEARCH_USERNAME=your_username
   ELASTICSEARCH_PASSWORD=your_password
   ```

2. **Security Configuration**:
   - Enable HTTPS
   - Configure CORS origins
   - Set up API rate limiting
   - Use secrets management

3. **Monitoring Setup**:
   - Configure log aggregation
   - Set up metrics collection
   - Enable alerting
   - Monitor resource usage

## 🔍 Troubleshooting

### Common Issues

#### 1. Elasticsearch Connection Failed
```bash
# Check if Elasticsearch is running
docker-compose ps

# Check Elasticsearch health
curl http://localhost:9200/_cluster/health

# Restart Elasticsearch
docker-compose restart elasticsearch
```

#### 2. OpenAI API Errors
```bash
# Verify API key
echo $OPENAI_API_KEY

# Check API quota
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/usage
```

#### 3. Document Processing Failures
- Ensure document is plain text (.txt)
- Check file size limits (max 50MB)
. Verify document has proper hierarchy markers
- Check processing logs for specific errors

#### 4. Memory Issues
- Increase Docker memory allocation
- Reduce `CHILD_CHUNK_SIZE` and `EMBEDDING_BATCH_SIZE`
- Use smaller embedding models

### Performance Optimization

#### 1. Elasticsearch Optimization
```bash
# Increase heap size
ES_JAVA_OPTS="-Xms2g -Xmx2g"

# Optimize index settings
curl -X PUT "localhost:9200/documents_child/_settings" \
     -H "Content-Type: application/json" \
     -d '{"index": {"number_of_replicas": 0}}'
```

#### 2. API Performance
- Enable response caching
- Use connection pooling
- Implement request batching
- Optimize database queries

## 📚 API Reference

### Core Endpoints

#### Ingestion API (`/api/ingest`)
- `POST /upload` - Upload and process document
- `GET /status/{document_id}` - Check processing status
- `GET /documents` - List all documents
- `DELETE /documents/{document_id}` - Delete document

#### Search API (`/api/search`)
- `POST /search` - Hybrid search with reranking
- `GET /chunk/{chunk_id}` - Retrieve specific chunk
- `GET /health` - Search service health check

#### Agent API (`/api/agent`)
- `POST /query` - Process query through LangGraph
- `GET /health` - Agent health check

#### Metrics API (`/api/metrics`)
- `GET /costs` - Cost tracking metrics
- `GET /latency` - Latency metrics
- `GET /health` - System health overview

### Response Format

```json
{
  "answer": "Generated response text",
  "route": "document_only|weather_only|combined|out_of_scope",
  "confidence": 0.85,
  "reasoning": "Query classified as document search",
  "sources": [
    {
      "type": "document",
      "chunk_id": "chunk_123",
      "document_id": "doc_456",
      "relevance": 0.92,
      "text": "Relevant text excerpt..."
    }
  ],
  "session_id": "session_789",
  "workflow_metrics": {
    "total_duration_ms": 1250,
    "total_cost": 0.0025,
    "api_calls": 3,
    "steps_completed": 4,
    "success_rate": 1.0
  }
}
```

## 🤝 Contributing

### Development Setup

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Install development dependencies**:
   ```bash
   pip install -r requirements-dev.txt
   ```
4. **Run tests and linting**:
   ```bash
   pytest tests/
   black app/
   isort app/
   flake8 app/
   ```
5. **Submit a pull request**

### Code Style

- **Python**: Follow PEP 8 guidelines
- **Type Hints**: Use type annotations for all functions
- **Documentation**: Add docstrings for all public functions
- **Testing**: Maintain test coverage above 80%

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for GPT models and embeddings
- **Elasticsearch** for vector search capabilities
- **LangChain** for text processing utilities
- **LangGraph** for workflow orchestration
- **FastAPI** for the web framework
- **Streamlit** for the chat interface

## 📞 Support

For questions, issues, or contributions:

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Documentation**: [Wiki](https://github.com/your-repo/wiki)

---

**GoRAGo** - Intelligent RAG System with Multi-Modal Response Generation
