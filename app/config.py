"""Configuration management for the RAG document ingestion system."""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # OpenAI Configuration
    openai_api_key: str
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.7
    
    # Elasticsearch Configuration
    elasticsearch_url: str = "http://localhost:9200"
    elasticsearch_username: Optional[str] = None
    elasticsearch_password: Optional[str] = None
    
    # Application Configuration
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = True
    
    # Chunking Hyperparameters
    child_chunk_size: int = 400
    chunk_overlap: int = 60
    parent_window_size: int = 1500
    max_tokens_compression: int = 1500
    
    # Retrieval Hyperparameters
    initial_top_k: int = 20
    final_top_k: int = 10
    dense_weight: float = 0.6
    bm25_weight: float = 0.4
    
    # Embedding Configuration
    embedding_model: str = "text-embedding-3-small"
    embedding_batch_size: int = 100
    
    # Index Names
    child_index_name: str = "documents_child"
    
    # Weather API Configuration
    openweather_api_key: str
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance with error handling
try:
    settings = Settings()
    if not settings.openai_api_key or settings.openai_api_key == "your_openai_api_key_here":
        raise ValueError("OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file.")
except Exception as e:
    print(f"Configuration error: {e}")
    print("Please check your .env file and ensure OPENAI_API_KEY is set correctly.")
    raise
