"""Main FastAPI application for RAG document ingestion system."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.routers import ingestion, search, agent, metrics
from app.services.elasticsearch_service import ElasticsearchService
from app.utils.logging_config import setup_logging

# Setup simplified logging
setup_logging(log_level="INFO", include_metrics=False)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    logger.info("Starting RAG Document Ingestion System...")
    
    try:
        # Initialize Elasticsearch service
        es_service = ElasticsearchService()
        
        # Create indices if they don't exist
        success = await es_service.create_indices()
        if success:
            logger.info("Elasticsearch indices created successfully")
        else:
            logger.warning("Failed to create Elasticsearch indices")
        
        logger.info("Application startup completed")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG Document Ingestion System...")


# Create FastAPI application
app = FastAPI(
    title="RAG Document Ingestion System",
    description="A FastAPI-based document ingestion system with Elasticsearch for vector storage, implementing hierarchical document chunking and retrieval.",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(ingestion.router)
app.include_router(search.router)
app.include_router(agent.router)
app.include_router(metrics.router)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG Document Ingestion System",
        "version": "1.0.0",
        "description": "A FastAPI-based document ingestion system with Elasticsearch for vector storage",
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "ingestion": "/api/ingest",
            "search": "/api/search",
            "agent": "/api/agent"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check Elasticsearch connection
        es_service = ElasticsearchService()
        
        # Try to get cluster health
        health = es_service.client.cluster.health()
        
        return {
            "status": "healthy",
            "elasticsearch": {
                "status": health.get("status", "unknown"),
                "cluster_name": health.get("cluster_name", "unknown")
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Custom 404 handler."""
    return JSONResponse(
        status_code=404,
        content={"message": "Endpoint not found", "path": str(request.url)}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Custom 500 handler."""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    from app.config import settings
    
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
        log_level="info"
    )
