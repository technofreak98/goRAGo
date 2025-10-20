"""Embedding service for generating OpenAI embeddings."""

import asyncio
import logging
from typing import List, Dict, Any
import openai
from app.config import settings
from app.utils.cost_tracker import cost_tracker
from app.utils.latency_tracker import track_latency
from app.utils.logging_config import get_metrics_logger

logger = logging.getLogger(__name__)
metrics_logger = get_metrics_logger(__name__)


class EmbeddingService:
    """Service for generating embeddings using OpenAI API."""
    
    def __init__(self):
        """Initialize OpenAI client."""
        openai.api_key = settings.openai_api_key
        self.client = openai.OpenAI(api_key=settings.openai_api_key)
        self.model = settings.embedding_model
        self.batch_size = settings.embedding_batch_size
    
    @track_latency("embedding_generate_batch")
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts with batching."""
        embeddings = []
        total_tokens = 0
        
        try:
            # Process in batches to avoid rate limits
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                
                # Generate embeddings for the batch
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                
                # Track cost for this batch
                batch_tokens = response.usage.total_tokens
                total_tokens += batch_tokens
                cost_metrics = cost_tracker.calculate_cost(self.model, batch_tokens, 0)
                
                # Extract embeddings from response
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
                
                # Log progress only for large batches
                if len(texts) > self.batch_size:
                    logger.debug(f"Generated embeddings for batch {i//self.batch_size + 1}/{(len(texts) + self.batch_size - 1)//self.batch_size}")
                
                # Small delay to respect rate limits
                if i + self.batch_size < len(texts):
                    await asyncio.sleep(0.1)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    @track_latency("embedding_generate_single")
    async def generate_single_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            
            # Track cost
            tokens = response.usage.total_tokens
            cost_metrics = cost_tracker.calculate_cost(self.model, tokens, 0)
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Error generating single embedding: {e}")
            raise
    
    @track_latency("embedding_generate_query")
    async def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for a search query."""
        try:
            # Use query preprocessor for enhanced keyword extraction
            from app.services.query_preprocessor import QueryPreprocessor
            preprocessor = QueryPreprocessor()
            processed_query = preprocessor.get_embedding_query(query)
            
            response = self.client.embeddings.create(
                model=self.model,
                input=processed_query
            )
            
            # Track cost
            tokens = response.usage.total_tokens
            cost_metrics = cost_tracker.calculate_cost(self.model, tokens, 0)
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """Clean text for embedding generation."""
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Remove special characters that might interfere with embedding
        # Keep alphanumeric, spaces, and common punctuation
        import re
        text = re.sub(r'[^\w\s.,!?;:-]', '', text)
        
        return text.strip()
    
    async def batch_embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for a batch of chunks and update them."""
        try:
            # Extract texts for embedding
            texts = [chunk["text"] for chunk in chunks]
            
            # Generate embeddings
            embeddings = await self.generate_embeddings(texts)
            
            # Update chunks with embeddings
            for chunk, embedding in zip(chunks, embeddings):
                chunk["embedding"] = embedding
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error batch embedding chunks: {e}")
            raise
    
    def get_embedding_dimensions(self) -> int:
        """Get the dimensions of embeddings from the current model."""
        # text-embedding-3-small has 1536 dimensions
        if "3-small" in self.model:
            return 1536
        elif "3-large" in self.model:
            return 3072
        elif "ada-002" in self.model:
            return 1536
        else:
            # Default for text-embedding-3-small
            return 1536
    
    async def health_check(self) -> bool:
        """Check if the embedding service is working."""
        try:
            test_embedding = await self.generate_single_embedding("test")
            return len(test_embedding) == self.get_embedding_dimensions()
        except Exception as e:
            logger.error(f"Embedding service health check failed: {e}")
            return False
